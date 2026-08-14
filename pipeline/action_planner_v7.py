"""
action_planner_v7.py -- PLANNING layer over the learned models.

PARADIGM SHIFT (per request): instead of the ball behavior being imitated
from data, the ball handler now EXPLICITLY ENUMERATES ACTIONS at every step
and takes the highest-value one:

    SHOOT            -> EV = P(score | Step 7 model) x shot value (2 or 3)
    DRIBBLE (8 dirs) -> EV = gamma x P(score at the new spot) x value
    PASS (4 mates)   -> EV = P(pass completes | passing-lane risk)
                             x gamma x P(score at receiver) x value

No retraining required. Division of labor:
    Step 7 XGBoost        = value function (shot quality)
    per-strategy MDN v2   = OFF-BALL movement (the "scheme" the other 4 run)
    defensive MDN         = defense reaction (unchanged)
    THIS PLANNER          = all ball decisions (shoot/dribble/pass)

TWO MODES:
    execute : planner shoots when SHOOT is the argmax action (or clock
              forces it). One animation-ready play, one resolved outcome.
    survey  : shooting is DISABLED; the possession runs its full length
              while every step with P(score) >= threshold is RECORDED as a
              shot opportunity. animate_opportunities() then renders ONE
              ANIMATION PER OPPORTUNITY (e.g. qualifying steps 4, 16, 20
              -> three GIFs, each ending with its own resolved shot).

Honest caveat: the Step 7 value function is MYOPIC (immediate shot quality
only). Greedy planning over it plays "find the best look soon", not
long-horizon offense. gamma (patience) and the shot-clock pressure rule
manage that tradeoff. Making the value function far-sighted would be an RL
project (fitted value iteration on possession outcomes) -- a natural next
step, not needed for this to work.

Requirements: mdn_full_system_v2.py + trained movement_generator_mdn_v2_*.pt
+ defensive_response_mdn.pt + shot pickles (+ step12_visualization for GIFs).
"""

import math
import numpy as np
import pandas as pd
import torch

from mdn_full_system_v2 import (
    OffenseMDNv2, DefenseMDN, sample_mdn,
    OFF_SCALE, DEF_SCALE, MAX_DELTA, CATCH_RADIUS,
    STRATEGY_NAMES, BASKET_X, BASKET_Y,
    _nearest_idx, _nearest_dist, _per_player_openness, _shot_zone,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DRIBBLE_DIRS   = [(math.cos(math.radians(a)), math.sin(math.radians(a)))
                  for a in range(0, 360, 45)]           # 8 directions
DRIBBLE_STEP   = 2.0        # ft per dribble action (one 0.2s step)
GAMMA          = 0.98       # patience: discount per step of delay
PASS_FLIGHT_FT = 8.0        # ball flight speed ft/step while passing
LANE_SAFE_FT   = 4.0        # defender this far off the lane = safe pass

# --- HOLD BUDGETS (realism: nobody dribbles for 20 steps) -----------------
# Max steps a player should hold the ball before pressure to move it on.
# Player index 0 is treated as the PG (budget ~7); everyone else ~4.
# Over budget: DRIBBLE EVs decay by HOLD_DECAY per extra step.
# Over budget + HOLD_HARD_CAP: dribbling removed entirely (pass or shoot).
DEFAULT_HOLD_BUDGETS = {0: 7, "default": 4}
HOLD_DECAY    = 0.80
HOLD_HARD_CAP = 4

# --- SPACING RULE (at least one off-ball player beyond the arc) ------------
# If NO off-ball attacker is comfortably beyond the 3pt line, the farthest
# one from the basket is pulled outward by SPACE_PULL ft/step. The
# ARC_MARGIN hysteresis pulls them ~1.5ft PAST the line so the movement
# model's inward drift doesn't immediately drag them back under it.
# Explicit inference-time rule (like pass_to_open), not learned behavior.
ARC_RADIUS  = 23.75
ARC_MARGIN  = 1.5
SPACE_PULL  = 2.5   # ft/step = 12.5 ft/s -- fast relocation, still human

# --- SHOT ACCEPTANCE (zone-dependent, EV-anchored) --------------------------
# Admission is anchored on EXPECTED VALUE: a two admits at EV 1.20
# (P > 0.60 x 2). The three admits at THREE_EV_TARGET = 1.15 -- deliberately
# CLOSE to the two's bar but a touch below it, so at the margin a three is
# slightly more rewarding than an equivalent deep two without turning the
# offense into a 3pt-only machine (the old 0.35 threshold = EV 1.05 did).
THREE_EV_TARGET = 1.15
THREE_THRESHOLD = round(THREE_EV_TARGET / 3.0, 4)     # P >= 0.3833
TWO_THRESHOLD   = 0.60

# --- DEEP-TWO PENALTY (shot-diet shaping) ------------------------------------
# 2pt spots in the long-midrange band (DEEP_TWO_MIN ft out to the arc) get
# their PLANNING EV scaled by DEEP_TWO_PENALTY: the planner steps out for
# the three instead of settling for a 20-footer. Twos INSIDE DEEP_TWO_MIN
# (paint / short range -- "the middle") are untouched, so deep in the zone
# the planner focuses on working the two. This shapes PREFERENCE only; the
# admission thresholds above still decide whether any shot qualifies.
DEEP_TWO_MIN     = 16.0
DEEP_TWO_PENALTY = 0.80

def deep_two_factor(value, dist):
    """EV multiplier: penalize long-midrange twos, leave everything else."""
    if value < 3.0 and dist >= DEEP_TWO_MIN:
        return DEEP_TWO_PENALTY
    return 1.0

def should_shoot(p, value, three_t=THREE_THRESHOLD, two_t=TWO_THRESHOLD):
    return p >= (three_t if value >= 3.0 else two_t)

# --- OPEN-SHOT RULE ---------------------------------------------------------
# A FREE player does not take wind-up dribbles: if the handler's nearest
# defender is beyond OPEN_SHOOT_DIST and the shot clears its zone
# threshold, he shoots IMMEDIATELY (overrides the argmax).
OPEN_SHOOT_DIST = 6.0

# --- MAN-TO-MAN DEFENSE BLEND -----------------------------------------------
# The learned defensive MDN is blended with explicit man-to-man tracking:
# each defender is matched to an attacker (Hungarian, fixed at possession
# start) and pulled toward the rim-blocking spot 2.5 ft off his man.
# man_to_man_weight = 0 -> pure learned model; 1 -> pure man tracking.
MTM_GAP        = 2.5    # ft off the man, toward the rim
MTM_TRACK_STEP = 2.2    # max tracking speed ft/step
DEF_STEP_CAP   = 2.5    # absolute defender speed cap ft/step

# --- SHOT CLOCK DEFAULTS (rule: half-court starts at 19s) -------------------
# Crossing half court eats ~5s, so half-court possessions default to 19s.
# Fast breaks (initial_positions["is_fast_break"]=True) default to 22s.
HALFCOURT_CLOCK = 19.0
FASTBREAK_CLOCK = 22.0

# --- SHOT CLOCK REALISM ------------------------------------------------------
# A half-court possession effectively starts around 19s (about 5s is spent
# crossing half court). Fast Break is the exception: it IS the crossing.
HALF_COURT_CLOCK   = 19.0
FASTBREAK_STRATEGY = 5

# --- TYPE-AWARE SHOT THRESHOLDS ----------------------------------------------
# (duplicate of the EV-anchored block above -- kept in sync)
# THREE at EV >= 1.15 (P >= 0.3833); TWO only at P > 0.60 (EV 1.2). These
# both trigger the shot in execute mode and define what counts as a
# recorded "opportunity" in survey mode.
THREE_THRESHOLD = round(THREE_EV_TARGET / 3.0, 4)     # P >= 0.3833
TWO_THRESHOLD   = 0.60

# --- FREE = SHOOT NOW ---------------------------------------------------------
# If the handler is open (nearest defender beyond this) and his shot clears
# its type threshold, he shoots IMMEDIATELY -- no extra wind-up steps.
OPEN_SHOOT_DIST = 6.0

# --- FINISH / WIDE-OPEN OVERRIDES ---------------------------------------------
# Two more explicit "just shoot it" rules, applied BEFORE the EV argmax.
# They exist because the Step-7 value model can be pessimistic in states it
# rarely saw (an unguarded handler parked under the rim, an empty fast-break
# floor), and the P>0.60 two-point bar then filters SHOOT out entirely --
# producing the pathological "passes out of an open layup" behavior.
#
#   POINT-BLANK: handler within RIM_FINISH_DIST of the rim and no defender
#     inside RIM_FINISH_OPEN -> finish NOW (layup/dunk), regardless of the
#     P threshold. Under the basket unguarded, anything but a shot is wrong.
#   WIDE OPEN: nobody within WIDE_OPEN_DIST of the handler -> the P bar
#     drops to WIDE_OPEN_MIN_P for that shot type. An open net does not
#     need a 0.60 probability to justify the attempt.
#
# Both respect offense_mode (a forced-3 possession never auto-finishes a 2);
# clock-forced desperation heaves remain mode-agnostic as before.
RIM_FINISH_DIST = 5.0     # ft from basket = point-blank range
RIM_FINISH_OPEN = 3.0     # nearest defender beyond this -> finish
WIDE_OPEN_DIST  = 8.0     # nobody within this -> relaxed shot admission
WIDE_OPEN_MIN_P = {2.0: 0.40, 3.0: 0.30}   # relaxed floors by shot value

# --- MAN-TO-MAN DEFENSE BLEND -------------------------------------------------
# Defensive deltas = (1-w) x learned MDN + w x move-toward-ideal-man-spot,
# where the ideal spot is DEF_GAP ft from the assigned man toward the rim.
# Matchups assigned once at possession start (Hungarian, nearest pairing).
MAN_MARK_WEIGHT = 0.5
DEF_GAP         = 3.0
DEF_MAX_STEP    = 2.5   # ft/step cap on the man-marking pull

# --- OPPORTUNITY RANKING ------------------------------------------------------
# Later opportunities carry accumulated risk (turnovers, resets) of getting
# there: score = P x value x OPP_DECAY^step. Used by
# select_best_opportunity() to answer "which step was the best look".
OPP_DECAY = 0.985

# ===========================================================================
# STRATEGY SCHEME LAYER -- explicit per-strategy behavior (inference-time)
# ===========================================================================
# The six MDNs differ statistically, but the schemes below make each
# strategy VISIBLY itself: cutting patterns, spacing shape, pass priorities
# and the primary's role are explicit rules layered on the MDN's off-ball
# deltas, in the same spirit as the spacing / open-shot rules.

# RANGE DISCIPLINE: nobody shoots from beyond MAX_SHOT_DIST -- except in
# Isolation (the iso man earns his deep look) and clock-forced heaves.
MAX_SHOT_DIST = 30.0
ISO_STRATEGY  = 3

# COURT DISCIPLINE: offense never crosses back over half court. A player
# already past x=47 (staged transition) may only move DOWN-court (x can
# only decrease) until he's inside the half.
HALF_X = 47.0

# CUTTING: designated cutter(s) oscillate between the rim and the 20ft
# radius -- constant in-and-out motion under the basket.
CUT_IN_R  = 6.0     # turn around this close to the rim...
CUT_OUT_R = 20.0    # ...and back in from this far out
CUT_PULL  = 1.8     # ft/step of cut movement blended over the MDN delta

# ISOLATION: everyone except the cutter keeps clear of the iso man.
ISO_RADIUS = 13.0   # off-ball players stay at least this far from him
ISO_PUSH   = 1.6

# FREE THROW HEAVY: pack the floor inside PACK_R; ONE hoverer holds the arc.
PACK_R    = 20.0
PACK_PULL = 1.6
HOVER_TOL = 2.0     # hoverer stays within this of the arc band

# PICK AND ROLL: the screener closes to SCREEN_NEAR of the handler, then
# rolls to the rim; passes to him are prioritized (roll_pass_bonus).
SCREEN_NEAR = 4.0
SCREEN_PULL = 2.0
ROLL_PULL   = 1.8

# PASS SHAPING: short passes prioritized (except Isolation).
SHORT_PASS_NEAR  = 12.0
SHORT_PASS_FAR   = 20.0
SHORT_PASS_BONUS = 1.10
LONG_PASS_MALUS  = 0.70

OFF_STEP_CAP = 3.0  # ft/step cap on any off-ball player AFTER shaping

SCHEMES = {
    # more movement than every other strategy: amplified deltas + 2 cutters
    0: {"move_scale": 1.35, "cutters": 2, "short_pass": True},
    # Post Up: the primary hunts his own shot and keeps the ball
    1: {"move_scale": 1.0,  "cutters": 1, "short_pass": True,
        "primary_shoot_bonus": 1.30, "primary_pass_malus": 0.70,
        "primary_hold_budget": 9},
    # Free Throw Heavy: everyone inside the arc, one hoverer at the line
    2: {"move_scale": 1.0,  "cutters": 1, "short_pass": True,
        "pack_inside": True},
    # Isolation: one cutter; everyone else clears out of the iso man's space;
    # long passes allowed (kick-outs across the clear-out are the point)
    3: {"move_scale": 1.0,  "cutters": 1, "short_pass": False,
        "isolate": True},
    # Pick and Roll: screener joins the handler then rolls; feed the roll man
    4: {"move_scale": 1.0,  "cutters": 1, "short_pass": True,
        "pick_roll": True, "roll_pass_bonus": 1.20},
    # Fast Break: the cutter is the outlet -- passes to him are prioritized
    5: {"move_scale": 1.15, "cutters": 1, "short_pass": True,
        "cutter_pass_bonus": 1.30},
}
SCHEME_DEFAULT = {"move_scale": 1.0, "cutters": 1, "short_pass": True}


def shot_ready(p, is_three):
    """Type-aware shooting rule: 3PT at P>=0.3833 (EV 1.15), 2PT at P>0.60."""
    return (p >= THREE_THRESHOLD) if is_three else (p > TWO_THRESHOLD)


# --- OFFENSE MODE ------------------------------------------------------------
# "normal" : pure EV planning. NOTE: rim-and-three offense EMERGES here, it
#            is not a bug -- 3s admit at EV>=1.15 (P>=0.3833) while contested 2s
#            need P>0.60, which the Step 7 model rarely grants with a rim
#            protector 2.5 ft away. So the planner correctly kicks out.
# "2pt"    : FORCE a two -- any action whose evaluated spot is beyond the arc
#            (3pt value) has its EV zeroed, so the planner steers the ball
#            inside and only 2pt shots qualify/record. (A clock-forced heave
#            still fires from anywhere -- desperation is mode-agnostic.)
# "3pt"    : FORCE a three -- inverse masking; only 3pt shots qualify/record.
OFFENSE_MODES = ("normal", "2pt", "3pt")

def mode_value_factor(value, offense_mode):
    """EV multiplier by shot value under the offense mode (1.0 or 0.0)."""
    if offense_mode == "2pt" and value >= 3.0:
        return 0.0
    if offense_mode == "3pt" and value < 3.0:
        return 0.0
    return 1.0

def should_shoot_mode(p, value, offense_mode,
                      three_t=THREE_THRESHOLD, two_t=TWO_THRESHOLD):
    """Zone threshold AND mode admission (a 2 never qualifies in 3pt mode)."""
    if mode_value_factor(value, offense_mode) == 0.0:
        return False
    return p >= (three_t if value >= 3.0 else two_t)


# --- DEFENSE PRESETS ----------------------------------------------------------
# "normal"        : learned MDN blended with man tracking at the weight the
#                   caller passes (default 0.6) -- current behavior.
# "man_to_man"    : pure man tracking (weight 1.0): every defender glued to
#                   his Hungarian-matched attacker, rim-side.
# "high_turnovers": ball-hawking defense. Two levers:
#                   lane_pressure > 1 shrinks effective passing-lane safety
#                   (completion = clip(closest / (LANE_SAFE_FT * pressure))),
#                   so the SAME pass completes less often -> more live-ball
#                   turnovers AND the planner is squeezed into riskier or
#                   more conservative choices;
#                   ball_pressure adds ft/step of extra pull on the handler's
#                   matched defender toward the ball, crushing handler
#                   openness (fewer open pull-ups, more forced decisions).
DEFENSE_PRESETS = {
    # normal: TIGHT only on the ball (MTM_GAP off the handler); everyone
    # else sags off_ball_gap ft from his man -- loose help defense.
    "normal":         {"man_weight": None, "lane_pressure": 1.0,
                       "ball_pressure": 0.0, "off_ball_gap": 4.0},
    "man_to_man":     {"man_weight": 1.0,  "lane_pressure": 1.0,
                       "ball_pressure": 0.0, "off_ball_gap": 2.5},
    "high_turnovers": {"man_weight": 0.5,  "lane_pressure": 1.75,
                       "ball_pressure": 1.2, "off_ball_gap": 3.0},
}

# --- HOLD-TIME PRESSURE (role realism) -------------------------------------
# The INITIAL handler (the PG bringing the ball into the play) may probe for
# ~PG_HOLD_BUDGET steps; every subsequent receiver must decide within
# ~ROLE_HOLD_BUDGET steps. Beyond budget, DRIBBLE EVs decay by HOLD_PENALTY
# per step of overage, and past budget+HARD_HOLD_EXTRA dribbling is removed
# from the action set entirely (must pass or shoot). Consecutive dribbles
# also compound a DRIBBLE_FATIGUE decay, so long probe chains self-terminate.
# Catch-and-shoot: within CATCH_SHOOT_STEPS of receiving a pass, SHOOT gets
# a CATCH_SHOOT_BONUS -- this is what lets a spot-up three fire at step 2-3
# after the catch instead of step 18.
PG_HOLD_BUDGET    = 7
ROLE_HOLD_BUDGET  = 4
HOLD_PENALTY      = 0.15    # dribble EV multiplier loses this per step over budget
HARD_HOLD_EXTRA   = 3       # over budget by this much -> dribbling forbidden
DRIBBLE_FATIGUE   = 0.95    # per consecutive dribble
CATCH_SHOOT_STEPS = 2
CATCH_SHOOT_BONUS = 1.05

# --- TEMPO PRESETS ------------------------------------------------------------
# Possession PACING, built from gates the planner already has (hold budgets,
# shot masking, clock forcing) rather than EV shaping -- gates are exactly
# controllable ("no shot before 5s left" is a rule, not a preference).
# Each preset can set:
#   hold_budgets       : max held steps per touch before dribble pressure
#                        (overridable by an explicit hold_budgets kwarg)
#   hold_hard_cap      : steps OVER budget before dribbling is removed
#                        entirely (None = global HOLD_HARD_CAP)
#   min_elapsed        : seconds of possession before ANY shot is allowed
#                        (the play must develop first)
#   shoot_below_clock  : SHOOT masked until the shot clock is at/below this
#                        ("milk it -- only shoot in the last N seconds")
#   force_after_elapsed: possession forced to its best shot once this many
#                        seconds have elapsed (quick tempo's early exit)
# Desperation still wins: once clock < shot_clock_force, every tempo shoots.
TEMPO_PRESETS = {
    # current behavior: pure EV planning
    "default":    {"hold_budgets": None,
                   "hold_hard_cap": None,
                   "min_elapsed": 0.0,
                   "shoot_below_clock": None,
                   "force_after_elapsed": None},
    # get a shot up fast: <=3 dribbles per touch, shot inside ~6 seconds
    "quick":      {"hold_budgets": {0: 3, "default": 2},
                   "hold_hard_cap": 1,
                   "min_elapsed": 0.0,
                   "shoot_below_clock": None,
                   "force_after_elapsed": 6.0},
    # let it develop: 3-5 dribbles per touch, no shot in the first 4 seconds
    "patient":    {"hold_budgets": {0: 5, "default": 4},
                   "hold_hard_cap": 1,
                   "min_elapsed": 4.0,
                   "shoot_below_clock": None,
                   "force_after_elapsed": None},
    # milk the clock: swing the ball, only shoot inside the last 5 seconds
    "milk_clock": {"hold_budgets": {0: 4, "default": 3},
                   "hold_hard_cap": None,
                   "min_elapsed": 0.0,
                   "shoot_below_clock": 5.0,
                   "force_after_elapsed": None},
}


# ===========================================================================
# VALUE FUNCTION (Step 7 wrapper)
# ===========================================================================

def p_score_at(x, y, def_coords, clock, spacing, overload,
               shot_model, le_zone, le_type):
    dist = math.dist((x, y), (BASKET_X, BASKET_Y))
    try:    ze = le_zone.transform([_shot_zone(x, y)])[0]
    except ValueError: ze = 0
    try:    te = le_type.transform(["jump_shot"])[0]
    except ValueError: te = 0
    feats = pd.DataFrame([{
        "shot_x": x, "shot_y": y, "shot_distance": dist,
        "shot_zone_enc": ze, "shot_type_enc": te,
        "shot_clock": clock, "quarter": 1,
        "is_three_pointer": 1 if dist >= 23.75 else 0,
        "nearest_def_to_ball_handler": _nearest_dist(x, y, def_coords),
        "spacing_score": spacing, "defensive_overload": int(overload),
    }])
    p = float(shot_model.predict_proba(feats)[:, 1][0])
    value = 3.0 if dist >= 23.75 else 2.0
    return p, value


def lane_completion_prob(hx, hy, rx, ry, def_coords, pressure=1.0):
    """
    P(pass completes) from the nearest defender's distance to the lane.
    pressure > 1 (high_turnovers preset) shrinks the effective safe distance:
    a defender 3 ft off the lane completes at 0.75 normally but only ~0.43 at
    pressure 1.75 -- same geometry, hawkier defense.
    """
    closest = min(_point_seg_dist(def_coords[j*2], def_coords[j*2+1],
                                  hx, hy, rx, ry) for j in range(5))
    floor = 0.10 if pressure > 1.0 else 0.15
    return float(np.clip(closest / (LANE_SAFE_FT * pressure), floor, 0.98))


def _point_seg_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    L2 = dx*dx + dy*dy
    if L2 < 1e-9:
        return math.dist((px, py), (ax, ay))
    t = max(0.0, min(1.0, ((px-ax)*dx + (py-ay)*dy) / L2))
    return math.dist((px, py), (ax + t*dx, ay + t*dy))


def _point_seg_proj(px, py, ax, ay, bx, by):
    """Closest point ON segment (a->b) to point p -- where a lane defender
    actually meets a pass in flight (the interception spot)."""
    dx, dy = bx - ax, by - ay
    L2 = dx*dx + dy*dy
    if L2 < 1e-9:
        return ax, ay
    t = max(0.0, min(1.0, ((px-ax)*dx + (py-ay)*dy) / L2))
    return ax + t*dx, ay + t*dy


# ===========================================================================
# ACTION ENUMERATION
# ===========================================================================

def enumerate_actions(handler, off, deff, ball_xy, clock, spacing, overload,
                      shot_model, le_zone, le_type, last_passer=None,
                      allow_shoot=True, offense_mode="normal",
                      lane_pressure=1.0):
    """
    Returns list of dicts: {"action", "ev", "detail", ...}, best first.
    offense_mode masks EVs by shot value ("2pt" zeroes 3pt-valued options,
    "3pt" zeroes 2pt-valued options) so the planner steers toward the forced
    shot type. lane_pressure scales pass-completion risk (defense preset).
    """
    hx, hy = off[handler*2], off[handler*2+1]
    acts = []

    # SHOOT (now)
    p, v = p_score_at(hx, hy, deff, clock, spacing, overload,
                      shot_model, le_zone, le_type)
    if allow_shoot:
        acts.append({"action": "SHOOT",
                     "ev": p * v * mode_value_factor(v, offense_mode),
                     "detail": f"P={p:.3f} x {v:.0f}pts", "p": p, "value": v})
    shoot_p = p   # recorded either way (survey mode uses it)

    # DRIBBLE in 8 directions (evaluate handler's spot after 2 ft move)
    for k, (ux, uy) in enumerate(DRIBBLE_DIRS):
        nx = max(0.5, min(46.5, hx + ux * DRIBBLE_STEP))
        ny = max(0.5, min(49.5, hy + uy * DRIBBLE_STEP))
        p2, v2 = p_score_at(nx, ny, deff, clock - 0.2, spacing, overload,
                            shot_model, le_zone, le_type)
        acts.append({"action": f"DRIBBLE_{k*45:03d}",
                     "ev": GAMMA * p2 * v2 * mode_value_factor(v2, offense_mode),
                     "detail": f"to ({nx:.0f},{ny:.0f}) P={p2:.3f}",
                     "target_xy": (nx, ny)})

    # PASS to each teammate (lane risk x their current shot quality)
    for j in range(5):
        if j == handler:
            continue
        rx, ry = off[j*2], off[j*2+1]
        comp = lane_completion_prob(hx, hy, rx, ry, deff,
                                    pressure=lane_pressure)
        p3, v3 = p_score_at(rx, ry, deff, clock - 0.4, spacing, overload,
                            shot_model, le_zone, le_type)
        ev = comp * GAMMA * p3 * v3 * mode_value_factor(v3, offense_mode)
        if last_passer is not None and j == last_passer:
            ev *= 0.85                       # discourage instant pass-back loops
        acts.append({"action": f"PASS_A{j+1}", "ev": ev,
                     "detail": f"comp={comp:.2f} recvP={p3:.3f}",
                     "receiver": j, "completion": comp})

    acts.sort(key=lambda a: -a["ev"])
    return acts, shoot_p, v


def _scheme_shape(j, ox, oy, primary, handler, cutters, cut_phase,
                  iso_on, off, pack_on, hoverer, screener, roll_phase):
    """Per-strategy off-ball INTENT vector for player j (added on top of the
    MDN delta, before the speed cap). Encodes the visible scheme behavior:
      - CUTTERS oscillate between the rim (CUT_IN_R) and 20 ft (CUT_OUT_R):
        toward the basket while phase=="in", back out while phase=="out".
      - ISOLATION clear-out: every off-ball non-cutter is pushed to at least
        ISO_RADIUS from the iso man (the current handler) so he's isolated.
      - FREE-THROW-HEAVY pack: off-ball players (except the lone hoverer) are
        pulled inside PACK_R; the hoverer is nudged to sit on the arc band.
      - PICK-AND-ROLL: the screener first closes to the handler (SCREEN_NEAR),
        then, once set (roll_phase=="roll"), rolls hard to the rim.
    Returns (sx, sy). Players with no scheme role return (0, 0).
    """
    if j == primary and not (screener == j):
        return 0.0, 0.0
    bx, by = BASKET_X, BASKET_Y

    # PICK AND ROLL screener overrides other roles
    if screener is not None and j == screener:
        if roll_phase == "roll":
            vx, vy = bx - ox, by - oy
            n = math.hypot(vx, vy) + 1e-6
            return vx/n * ROLL_PULL, vy/n * ROLL_PULL
        hx, hy = off[handler*2], off[handler*2+1]
        vx, vy = hx - ox, hy - oy
        n = math.hypot(vx, vy) + 1e-6
        if n > SCREEN_NEAR:
            return vx/n * SCREEN_PULL, vy/n * SCREEN_PULL
        return 0.0, 0.0

    # CUTTER oscillation
    if j in cut_phase:
        vx, vy = (bx - ox, by - oy) if cut_phase[j] == "in" \
                 else (ox - bx, oy - by)
        n = math.hypot(vx, vy) + 1e-6
        return vx/n * CUT_PULL, vy/n * CUT_PULL

    # FREE-THROW-HEAVY pack / hoverer
    if pack_on:
        r = math.dist((ox, oy), (bx, by))
        if j == hoverer:
            # sit on the arc band: pull toward ARC_RADIUS along own ray
            vx, vy = ox - bx, oy - by
            n = math.hypot(vx, vy) + 1e-6
            target = ARC_RADIUS
            err = target - r
            if abs(err) > HOVER_TOL:
                return -vx/n * PACK_PULL * (1 if err < 0 else -1), \
                       -vy/n * PACK_PULL * (1 if err < 0 else -1)
            return 0.0, 0.0
        if r > PACK_R:                       # pull inside the arc
            vx, vy = bx - ox, by - oy
            n = math.hypot(vx, vy) + 1e-6
            return vx/n * PACK_PULL, vy/n * PACK_PULL
        return 0.0, 0.0

    # ISOLATION clear-out: stay away from the iso man (the handler)
    if iso_on:
        hx, hy = off[handler*2], off[handler*2+1]
        d = math.dist((ox, oy), (hx, hy))
        if d < ISO_RADIUS:
            vx, vy = ox - hx, oy - hy
            n = math.hypot(vx, vy) + 1e-6
            return vx/n * ISO_PUSH, vy/n * ISO_PUSH
        return 0.0, 0.0

    return 0.0, 0.0


# ===========================================================================
# PLANNED POSSESSION
# ===========================================================================

def plan_possession_v7(
    strategy_idx, initial_positions, shot_model, le_zone, le_type,
    models_dir=".", mode="execute", max_steps=60, seed=None,
    off_temperature=0.7, def_temperature=0.5,
    opportunity_threshold=0.45, shot_clock_force=4.0,
    action_temperature=0.0,
    hold_budgets=None, enforce_spacing=True,
    three_threshold=THREE_THRESHOLD, two_threshold=TWO_THRESHOLD,
    open_shoot_dist=OPEN_SHOOT_DIST, man_to_man_weight=0.6,
    defense_preset="normal", offense_mode="normal",
    force_pass_completion=False, tempo="default",
):
    """
    (See earlier docstring.) New in this revision:
      - tempo: possession pacing preset ("default", "quick", "patient",
        "milk_clock" -- see TEMPO_PRESETS). Built from hard gates: hold
        budgets, a shooting mask (no shot before N elapsed seconds / until
        the clock is under N seconds), and a forced early exit for "quick".
        Clock desperation (clock < shot_clock_force) overrides every tempo.
        NOTE: milk_clock needs max_steps large enough to reach its window
        (e.g. 19s start -> ~(19-5)/0.2 = 70+ steps).
      - offense_mode: "normal" (EV planning), "2pt" (force a two: 3pt-valued
        actions get EV 0, only 2s qualify/record), "3pt" (force a three).
        A clock-forced heave still fires regardless of mode.
      - defense_preset: "normal", "man_to_man" (pure man tracking),
        "high_turnovers" (passing lanes shrink -> lower completion -> real
        live-ball turnovers; the handler's defender also pressures the ball).
      - TURNOVERS ARE NOW VISIBLE: an intercepted pass no longer ends the
        rollout instantly at the handler. The ball flies down the lane and
        the intercepting defender (nearest to the lane -- the one whose
        position caused the low completion) converges on the interception
        point and takes it, over a few appended steps, so the GIF shows the
        pick. The final step carries {"turnover": True}.
      - force_pass_completion=True: the interception dice are still ROLLED
        (so the RNG stream -- and therefore the whole trajectory -- stays
        bit-identical to a normal run with the same seed) but the result is
        ignored: every pass completes. Rerunning a turnover seed with this
        flag produces the exact same play up to the pass, which then
        completes -- the counterfactual "what if it wasn't picked" GIF.
      - shot clock defaults: 19s half-court, 22s if
        initial_positions["is_fast_break"] is True (5s to cross half court)
      - man_to_man_weight: blends the learned defensive MDN with explicit
        man-to-man tracking of a fixed matchup (Hungarian at start)
      - zone shot acceptance: 3s taken at P>=three_threshold (0.45),
        2s only at P>=two_threshold (0.60)
      - open-shot rule: a handler with no defender within open_shoot_dist
        shoots IMMEDIATELY if the shot clears its zone threshold --
        no wind-up dribbles when free
      - opportunities (survey) are recorded by the zone thresholds and
        carry "value"/"ev" so select_best_opportunity() can rank them
    """
    if seed is not None:
        torch.manual_seed(seed); np.random.seed(seed)

    if offense_mode not in OFFENSE_MODES:
        raise ValueError(f"offense_mode must be one of {OFFENSE_MODES}")
    preset = DEFENSE_PRESETS.get(defense_preset)
    if preset is None:
        raise ValueError(f"defense_preset must be one of "
                         f"{tuple(DEFENSE_PRESETS)}")
    if preset["man_weight"] is not None:
        man_to_man_weight = preset["man_weight"]
    lane_pressure = preset["lane_pressure"]
    ball_pressure = preset["ball_pressure"]
    off_ball_gap  = preset.get("off_ball_gap", MTM_GAP)

    # TEMPO: pacing gates (see TEMPO_PRESETS). Caller's explicit hold_budgets
    # still win over the tempo's.
    tempo_cfg = TEMPO_PRESETS.get(tempo)
    if tempo_cfg is None:
        raise ValueError(f"tempo must be one of {tuple(TEMPO_PRESETS)}")
    tempo_shoot_below = tempo_cfg["shoot_below_clock"]
    tempo_min_elapsed = tempo_cfg["min_elapsed"]
    tempo_force_after = tempo_cfg["force_after_elapsed"]
    hold_hard_cap = tempo_cfg["hold_hard_cap"] or HOLD_HARD_CAP

    budgets = dict(DEFAULT_HOLD_BUDGETS)
    if tempo_cfg["hold_budgets"]:
        budgets.update(tempo_cfg["hold_budgets"])
    if hold_budgets:
        budgets.update(hold_budgets)
    def budget_of(j):
        return budgets.get(j, budgets.get("default", 4))

    # shot clock default per rule 1
    default_clock = FASTBREAK_CLOCK if initial_positions.get("is_fast_break") \
                    else HALFCOURT_CLOCK
    start_clock = initial_positions.get("shot_clock", default_clock)

    off_model = OffenseMDNv2().to(DEVICE)
    off_model.load_state_dict(torch.load(
        f"{models_dir}/movement_generator_mdn_v2_{strategy_idx}.pt",
        map_location=DEVICE)); off_model.eval()
    def_model = DefenseMDN().to(DEVICE)
    def_model.load_state_dict(torch.load(
        f"{models_dir}/defensive_response_mdn.pt",
        map_location=DEVICE)); def_model.eval()
    off_sc = torch.tensor(OFF_SCALE, dtype=torch.float32).to(DEVICE)
    def_sc = torch.tensor(DEF_SCALE, dtype=torch.float32).to(DEVICE)

    # man-to-man matchups: fixed at possession start (Hungarian nearest)
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist
    _o = np.array(initial_positions["offense"], dtype=float)
    _d = np.array(initial_positions["defense"], dtype=float)
    _rows, _cols = linear_sum_assignment(cdist(_d, _o))
    matchup = {int(r): int(c) for r, c in zip(_rows, _cols)}

    off  = [c for p in initial_positions["offense"] for c in p]
    deff = [c for p in initial_positions["defense"] for c in p]
    ball_x, ball_y = initial_positions["ball"]
    handler = _nearest_idx(ball_x, ball_y, off)
    in_flight, flight_target, last_passer = False, None, None

    # ---- SCHEME setup (see SCHEMES): explicit per-strategy behavior ----
    scheme  = SCHEMES.get(strategy_idx, SCHEME_DEFAULT)
    move_scale = scheme.get("move_scale", 1.0)
    primary = handler                          # initial handler = the primary
    _rim0 = lambda j: math.dist((off[j*2], off[j*2+1]), (BASKET_X, BASKET_Y))
    offball0 = [j for j in range(5) if j != primary]
    iso_on  = scheme.get("isolate", False)
    pack_on = scheme.get("pack_inside", False)
    hoverer = max(offball0, key=_rim0) if pack_on else None
    screener = None
    roll_phase = "screen"
    if scheme.get("pick_roll"):
        screener = min(offball0, key=lambda j: math.dist(
            (off[j*2], off[j*2+1]), (off[primary*2], off[primary*2+1])))
    # cutters: nearest-the-rim off-ball player(s) cycle in and out; the
    # screener / arc hoverer keep their own jobs and never double as cutters
    cut_pool  = [j for j in offball0 if j != screener and j != hoverer]
    cutters   = sorted(cut_pool, key=_rim0)[:scheme.get("cutters", 1)]
    cut_phase = {j: "in" for j in cutters}
    # Post Up: the primary probes longer (tempo / caller budgets still win)
    if "primary_hold_budget" in scheme and not (
            tempo_cfg["hold_budgets"] or hold_budgets):
        budgets[primary] = scheme["primary_hold_budget"]

    steps = [{"offense": list(initial_positions["offense"]),
              "defense": list(initial_positions["defense"]),
              "ball": initial_positions["ball"],
              "shot_clock": start_clock}]
    decision_log = []
    info = {"opportunities": [], "shot_step": None, "reason": None,
            "made": None, "points": 0, "expected_points": None,
            "n_passes": 0, "turnover": False,
            "touches": [], "spacing_interventions": 0,
            "first_three_option": None,
            "offense_mode": offense_mode, "defense_preset": defense_preset,
            "tempo": tempo}
    held = 0                      # consecutive steps current handler has held
    oh, dh = None, None

    with torch.no_grad():
        for step in range(max_steps):
            clock = max(0.0, start_clock - step * 0.2)
            spacing = initial_positions.get("spacing_score", 20.0)
            overload = initial_positions.get("defensive_overload", False)
            screen = float(initial_positions.get("screen_detected", False))
            spatial = [spacing, screen, float(overload),
                       _nearest_dist(ball_x, ball_y, deff), clock]
            openness = _per_player_openness(off, deff)

            # ---- OFF-BALL movement from the strategy MDN (5 players+ball;
            #      the handler's delta and the ball delta get OVERRIDDEN
            #      by the planner's chosen action) ----
            xin = off + deff + [ball_x, ball_y] + spatial + openness
            x = torch.tensor([[xin]], dtype=torch.float32).to(DEVICE) * off_sc
            lo, mo, so, oh = off_model.step(x, oh)
            d = sample_mdn(lo, mo, so,
                           temperature=off_temperature).cpu().numpy() * MAX_DELTA

            chosen = None
            if in_flight:
                # ball travels; nobody decides this step
                tx, ty = off[flight_target*2], off[flight_target*2+1]
                vec = np.array([tx - ball_x, ty - ball_y])
                nrm = np.linalg.norm(vec) + 1e-6
                ball_x += vec[0]/nrm * min(PASS_FLIGHT_FT, nrm)
                ball_y += vec[1]/nrm * min(PASS_FLIGHT_FT, nrm)
            else:
                # ---- PLANNER: enumerate + choose ----
                held += 1
                # TEMPO GATES:
                #   force_now   -> a shot is FORCED this step (clock
                #                  desperation, or quick-tempo elapsed limit)
                #   tempo_holds -> shooting is FORBIDDEN this step (milk-
                #                  the-clock window / play still developing).
                #                  Desperation always beats the hold.
                elapsed = step * 0.2
                force_now = (clock < shot_clock_force
                             or (tempo_force_after is not None
                                 and elapsed >= tempo_force_after))
                tempo_holds = (not force_now) and (
                    (tempo_shoot_below is not None
                     and clock > tempo_shoot_below)
                    or elapsed < tempo_min_elapsed)

                acts, shoot_p, shoot_v = enumerate_actions(
                    handler, off, deff, (ball_x, ball_y), clock, spacing,
                    overload, shot_model, le_zone, le_type,
                    last_passer=last_passer,
                    allow_shoot=(mode == "execute"),
                    offense_mode=offense_mode,
                    lane_pressure=lane_pressure)

                hx_, hy_ = off[handler*2], off[handler*2+1]
                near_d = _nearest_dist(hx_, hy_, deff)
                rim_d  = math.dist((hx_, hy_), (BASKET_X, BASKET_Y))
                # RANGE DISCIPLINE: no shot beyond MAX_SHOT_DIST -- unless
                # this is Isolation, or the shot is being forced (heave).
                dist_ok = (rim_d <= MAX_SHOT_DIST
                           or strategy_idx == ISO_STRATEGY or force_now)

                # ---- SCHEME EV SHAPING ----
                # short passes prioritized (except Isolation); the roll man /
                # fast-break cutter are prioritized receivers; the Post Up
                # primary hunts his own shot and is reluctant to give it up.
                for a in acts:
                    if a["action"].startswith("PASS"):
                        rj = a["receiver"]
                        pd_ = math.dist((hx_, hy_),
                                        (off[rj*2], off[rj*2+1]))
                        if scheme.get("short_pass", True):
                            a["ev"] *= (SHORT_PASS_BONUS
                                        if pd_ <= SHORT_PASS_NEAR else
                                        1.0 if pd_ <= SHORT_PASS_FAR else
                                        LONG_PASS_MALUS)
                        if screener is not None and rj == screener:
                            a["ev"] *= scheme.get("roll_pass_bonus", 1.0)
                        if rj in cut_phase and "cutter_pass_bonus" in scheme:
                            a["ev"] *= scheme["cutter_pass_bonus"]
                        if handler == primary and "primary_pass_malus" in scheme:
                            a["ev"] *= scheme["primary_pass_malus"]
                    elif a["action"] == "SHOOT":
                        if handler == primary and "primary_shoot_bonus" in scheme:
                            a["ev"] *= scheme["primary_shoot_bonus"]
                acts.sort(key=lambda a: -a["ev"])

                # TEMPO HOLD: shooting masked -- keep moving the ball.
                if mode == "execute" and tempo_holds:
                    acts = [a for a in acts if a["action"] != "SHOOT"]

                # RANGE MASK: out-of-range shot removed from the menu.
                if mode == "execute" and not dist_ok:
                    acts = [a for a in acts if a["action"] != "SHOOT"]

                # ZONE + MODE SHOT ACCEPTANCE (execute mode): a SHOOT that
                # fails its zone threshold (3s: EV>=1.15, 2s: P>=0.60) OR its
                # offense_mode (a 2 in "3pt" mode, a 3 in "2pt" mode) is
                # removed -- the planner keeps working instead of settling.
                # Kept when a shot is being forced (desperation heave or
                # quick-tempo elapsed limit).
                if mode == "execute" and not force_now:
                    if not should_shoot_mode(shoot_p, shoot_v, offense_mode,
                                             three_threshold, two_threshold):
                        acts = [a for a in acts if a["action"] != "SHOOT"]

                # SHOOT-NOW OVERRIDES (execute mode), strongest first:
                #   1. POINT-BLANK: at the rim with no defender in reach ->
                #      finish regardless of the value model's P (fixes
                #      "passes out of an open layup").
                #   2. OPEN SHOT: free handler + shot clears its normal zone
                #      threshold -> shoot, no wind-up steps (as before).
                #   3. WIDE OPEN: nobody within WIDE_OPEN_DIST -> shoot at a
                #      RELAXED P floor (open looks don't need P>0.60).
                # All respect offense_mode masking, the tempo hold (a
                # milk-the-clock possession does not auto-finish early) AND
                # range discipline (a wide-open 32-footer stays a no).
                mode_ok = mode_value_factor(shoot_v, offense_mode) > 0.0
                shoot_now = None
                if mode == "execute" and mode_ok and not tempo_holds \
                        and dist_ok:
                    if rim_d <= RIM_FINISH_DIST and near_d > RIM_FINISH_OPEN:
                        shoot_now = (f"POINT-BLANK finish ({rim_d:.1f}ft out, "
                                     f"nearest D {near_d:.1f}ft)")
                    elif (near_d > open_shoot_dist
                          and should_shoot_mode(shoot_p, shoot_v, offense_mode,
                                                three_threshold, two_threshold)):
                        shoot_now = f"OPEN shot P={shoot_p:.3f}"
                    elif (near_d > WIDE_OPEN_DIST
                          and shoot_p >= WIDE_OPEN_MIN_P.get(shoot_v, 0.40)):
                        shoot_now = (f"WIDE OPEN ({near_d:.1f}ft) "
                                     f"P={shoot_p:.3f}")
                if shoot_now:
                    acts = [{"action": "SHOOT",
                             "ev": shoot_p * shoot_v,
                             "detail": shoot_now,
                             "p": shoot_p, "value": shoot_v}]

                # HOLD BUDGET: decay dribble EVs past budget; hard-cap
                # removes dribbling so the ball MUST move on. The cap is
                # tempo-aware (quick/patient tempos cut probing short).
                over = held - budget_of(handler)
                if over > 0:
                    if over >= hold_hard_cap:
                        acts = [a for a in acts
                                if not a["action"].startswith("DRIBBLE")]
                    else:
                        for a in acts:
                            if a["action"].startswith("DRIBBLE"):
                                a["ev"] *= HOLD_DECAY ** over
                    acts.sort(key=lambda a: -a["ev"])

                # first step a 3pt-valued action reaches the top-3
                if info["first_three_option"] is None:
                    for a in acts[:3]:
                        is3 = (a["action"] == "SHOOT"
                               and a.get("value") == 3.0)
                        if a["action"].startswith("PASS"):
                            rj = a["receiver"]
                            is3 = math.dist(
                                (off[rj*2], off[rj*2+1]),
                                (BASKET_X, BASKET_Y)) >= ARC_RADIUS
                        if is3:
                            info["first_three_option"] = step + 1
                            break

                # record opportunity regardless of mode="survey"/"execute"
                # (zone thresholds: 3s at EV>=1.15, 2s at P>=0.60) -- but a
                # shot excluded by offense_mode OR beyond range never qualifies.
                if dist_ok and should_shoot_mode(shoot_p, shoot_v, offense_mode,
                                                 three_threshold, two_threshold):
                    info["opportunities"].append(
                        {"step": step + 1, "p": shoot_p,
                         "value": shoot_v, "ev": round(shoot_p * shoot_v, 3),
                         "clock": round(clock, 1),
                         "xy": (off[handler*2], off[handler*2+1]),
                         "handler": handler})

                if action_temperature > 0 and len(acts) > 1:
                    evs = np.array([a["ev"] for a in acts])
                    w = np.exp((evs - evs.max()) / action_temperature)
                    chosen = acts[int(np.random.choice(len(acts), p=w/w.sum()))]
                else:
                    chosen = acts[0]

                # forced shot: clock desperation OR the quick tempo's
                # elapsed-time limit (execute mode)
                if mode == "execute" and force_now:
                    chosen = next((a for a in acts if a["action"] == "SHOOT"),
                                  chosen)

                decision_log.append(
                    {"step": step + 1, "handler": handler, "held": held,
                     "chosen": chosen["action"], "ev": round(chosen["ev"], 3),
                     "alternatives": [(a["action"], round(a["ev"], 3))
                                      for a in acts[:4]]})

            # ---- apply movement: off-ball via MDN, handler via action ----
            # Off-ball deltas are (a) SCALED per scheme (Motion amplified,
            # Fast Break slightly), then (b) SHAPED by explicit scheme intent
            # -- cutters oscillate rim<->20ft, the iso clear-out pushes
            # players off the iso man, Free-Throw-Heavy packs the arc with
            # one hoverer, the screener sets then rolls -- and finally
            # (c) SPEED-CAPPED so shaping never teleports anyone.
            new_off = []
            for j in range(5):
                if (not in_flight) and j == handler and chosen and \
                        chosen["action"].startswith("DRIBBLE"):
                    nx, ny = chosen["target_xy"]
                elif (not in_flight) and j == handler:
                    nx = off[j*2] + d[j, 0] * 0.3   # damped while deciding
                    ny = off[j*2+1] + d[j, 1] * 0.3
                else:
                    ox, oy = off[j*2], off[j*2+1]
                    dx, dy = d[j, 0] * move_scale, d[j, 1] * move_scale
                    # (b) scheme shaping -- adds an intent vector for this role
                    sx, sy = _scheme_shape(
                        j, ox, oy, primary, handler, cutters, cut_phase,
                        iso_on, off, pack_on, hoverer, screener, roll_phase)
                    dx += sx; dy += sy
                    # (c) speed cap on the combined off-ball move
                    mag = math.hypot(dx, dy)
                    if mag > OFF_STEP_CAP:
                        dx *= OFF_STEP_CAP / mag; dy *= OFF_STEP_CAP / mag
                    nx, ny = ox + dx, oy + dy
                # (2) BACKCOURT DISCIPLINE: never cross half court. A player
                # already past HALF_X (staged transition) may ONLY move
                # down-court -- x is not allowed to increase.
                if off[j*2] > HALF_X:
                    nx = min(nx, off[j*2])
                nx = max(0.0, min(HALF_X, nx))
                new_off.extend([nx, max(0.0, min(50.0, ny))])
            off = new_off

            # advance cutter phases (rim<->out oscillation)
            for c in cutters:
                r = math.dist((off[c*2], off[c*2+1]), (BASKET_X, BASKET_Y))
                if cut_phase[c] == "in" and r <= CUT_IN_R:
                    cut_phase[c] = "out"
                elif cut_phase[c] == "out" and r >= CUT_OUT_R:
                    cut_phase[c] = "in"
            # pick-and-roll: once the screener is set next to the handler,
            # he rolls to the rim
            if screener is not None and roll_phase == "screen":
                if math.dist((off[screener*2], off[screener*2+1]),
                             (off[handler*2], off[handler*2+1])) <= SCREEN_NEAR + 0.5:
                    roll_phase = "roll"

            # ---- SPACING RULE: keep at least one OFF-BALL player beyond
            # the 3pt arc so a three is always a live option ----
            if enforce_spacing:
                offball = [j for j in range(5) if j != handler]
                best_out = max(math.dist((off[j*2], off[j*2+1]),
                                         (BASKET_X, BASKET_Y))
                               for j in offball)
                if best_out < ARC_RADIUS + ARC_MARGIN:
                    far = max(offball, key=lambda j: math.dist(
                        (off[j*2], off[j*2+1]), (BASKET_X, BASKET_Y)))
                    fx, fy = off[far*2], off[far*2+1]
                    vx, vy = fx - BASKET_X, fy - BASKET_Y
                    n = math.hypot(vx, vy) + 1e-6
                    off[far*2]   = max(0.0, min(HALF_X, fx + vx/n * SPACE_PULL))
                    off[far*2+1] = max(0.0, min(50.0, fy + vy/n * SPACE_PULL))
                    if best_out < ARC_RADIUS:
                        info["spacing_interventions"] += 1

            # ---- apply ball action ----
            if in_flight:
                j = _nearest_idx(ball_x, ball_y, off)
                if math.dist((ball_x, ball_y),
                             (off[j*2], off[j*2+1])) < CATCH_RADIUS:
                    in_flight, handler = False, j
                    held = 0                       # new touch begins
                    last_passer_tmp = flight_target
                    flight_target = None
                    ball_x, ball_y = off[j*2], off[j*2+1]
            elif chosen["action"] == "SHOOT":
                info["touches"].append((handler, held))
                p, v = chosen["p"], chosen["value"]
                info["shot_step"] = step + 1
                info["reason"] = f"planner chose SHOOT (EV={chosen['ev']:.3f})"
                info["made"] = bool(np.random.random() < p)
                info["points"] = int(v) if info["made"] else 0
                info["expected_points"] = round(p * v, 3)
                steps.append({
                    "offense": [(off[j*2], off[j*2+1]) for j in range(5)],
                    "defense": [(deff[j*2], deff[j*2+1]) for j in range(5)],
                    "ball": (ball_x, ball_y), "p_score": p,
                    "shot_clock": clock})
                break
            elif chosen["action"].startswith("PASS"):
                info["touches"].append((handler, held))
                recv = chosen["receiver"]
                # Dice are ALWAYS rolled (keeps the RNG stream identical
                # between a normal run and a force_pass_completion rerun of
                # the same seed, so the counterfactual matches to the pass).
                roll = np.random.random()
                if (not force_pass_completion) and roll > chosen["completion"]:
                    # ---- TURNOVER: show the pick instead of vanishing ----
                    # Intercepting defender = the one nearest the passing
                    # lane (he is WHY completion was low). The ball flies
                    # from the handler toward the receiver; the defender
                    # converges on the interception point (his projection
                    # onto the lane); everyone else holds for the beat.
                    hx, hy = off[handler*2], off[handler*2+1]
                    rx, ry = off[recv*2], off[recv*2+1]
                    ij = min(range(5), key=lambda j: _point_seg_dist(
                        deff[j*2], deff[j*2+1], hx, hy, rx, ry))
                    ix, iy = _point_seg_proj(deff[ij*2], deff[ij*2+1],
                                             hx, hy, rx, ry)
                    dx0, dy0 = deff[ij*2], deff[ij*2+1]
                    n_fly = max(1, int(math.ceil(
                        math.dist((hx, hy), (ix, iy)) / PASS_FLIGHT_FT)))
                    for f in range(1, n_fly + 1):
                        t = f / n_fly
                        bx_ = hx + (ix - hx) * t
                        by_ = hy + (iy - hy) * t
                        deff[ij*2]   = dx0 + (ix - dx0) * t
                        deff[ij*2+1] = dy0 + (iy - dy0) * t
                        steps.append({
                            "offense": [(off[j*2], off[j*2+1])
                                        for j in range(5)],
                            "defense": [(deff[j*2], deff[j*2+1])
                                        for j in range(5)],
                            "ball": (bx_, by_), "p_score": None,
                            "in_flight": True,
                            "shot_clock": max(0.0, clock - 0.2 * f),
                            "turnover": f == n_fly})
                    info["turnover"] = True
                    info["reason"] = (f"TURNOVER: pass to A{recv+1} picked "
                                      f"off by D{ij+1} "
                                      f"(comp={chosen['completion']:.2f})")
                    info["expected_points"] = 0.0
                    break
                in_flight, flight_target = True, recv
                last_passer, info["n_passes"] = handler, info["n_passes"] + 1
                ball_x, ball_y = off[handler*2], off[handler*2+1]
            else:  # DRIBBLE (handler already moved) -- ball with handler
                ball_x, ball_y = off[handler*2], off[handler*2+1]

            if not in_flight and chosen and not chosen["action"].startswith("PASS"):
                ball_x, ball_y = off[handler*2], off[handler*2+1]

            # ---- defense responds to the new world ----
            din = off + deff + [ball_x, ball_y] + spatial
            xd = torch.tensor([[din]], dtype=torch.float32).to(DEVICE) * def_sc
            ld, md, sd, dh = def_model.step(xd, dh)
            dd = sample_mdn(ld, md, sd,
                            temperature=def_temperature).cpu().numpy() * MAX_DELTA

            # MAN-TO-MAN BLEND: pull each defender toward the rim-blocking
            # spot off HIS matched man. GAP is role-aware: TIGHT (MTM_GAP)
            # on the current ball handler, LOOSE (preset off_ball_gap, e.g.
            # 4 ft in "normal") on everyone else -- off-ball defenders sag.
            if man_to_man_weight > 0:
                for j in range(5):
                    mj = matchup[j]
                    gap = MTM_GAP if mj == handler else off_ball_gap
                    mx, my = off[mj*2], off[mj*2+1]
                    vx, vy = BASKET_X - mx, BASKET_Y - my
                    n = math.hypot(vx, vy) + 1e-6
                    tx, ty = mx + vx/n * gap, my + vy/n * gap
                    ux, uy = tx - deff[j*2], ty - deff[j*2+1]
                    un = math.hypot(ux, uy)
                    if un > MTM_TRACK_STEP:
                        ux, uy = ux/un * MTM_TRACK_STEP, uy/un * MTM_TRACK_STEP
                    dd[j, 0] = (1 - man_to_man_weight) * dd[j, 0] \
                               + man_to_man_weight * ux
                    dd[j, 1] = (1 - man_to_man_weight) * dd[j, 1] \
                               + man_to_man_weight * uy

            # BALL PRESSURE (high_turnovers preset): the defender matched to
            # the CURRENT handler gets an extra pull straight at the ball,
            # on top of the man blend -- crushes handler openness so open
            # pull-ups vanish and decisions get forced. Speed cap below
            # still applies, so this never teleports anyone.
            if ball_pressure > 0 and not in_flight:
                pj = next((jj for jj in range(5) if matchup[jj] == handler),
                          None)
                if pj is not None:
                    ux, uy = ball_x - deff[pj*2], ball_y - deff[pj*2+1]
                    un = math.hypot(ux, uy) + 1e-6
                    dd[pj, 0] += ux/un * ball_pressure
                    dd[pj, 1] += uy/un * ball_pressure

            new_def = []
            for j in range(5):
                # absolute defender speed cap
                sp = math.hypot(dd[j, 0], dd[j, 1])
                if sp > DEF_STEP_CAP:
                    dd[j, 0] *= DEF_STEP_CAP / sp
                    dd[j, 1] *= DEF_STEP_CAP / sp
                new_def.extend([max(0.0, min(94.0, deff[j*2] + dd[j, 0])),
                                max(0.0, min(50.0, deff[j*2+1] + dd[j, 1]))])
            deff = new_def

            # record P for display (possessed steps only)
            p_disp = None
            if not in_flight:
                p_disp, _ = p_score_at(ball_x, ball_y, deff, clock, spacing,
                                       overload, shot_model, le_zone, le_type)
            steps.append({
                "offense": [(off[j*2], off[j*2+1]) for j in range(5)],
                "defense": [(deff[j*2], deff[j*2+1]) for j in range(5)],
                "ball": (ball_x, ball_y), "p_score": p_disp,
                "in_flight": in_flight, "shot_clock": clock})

    return steps, decision_log, info


# ===========================================================================
# MULTI-OPPORTUNITY ANIMATION (survey mode)
# ===========================================================================

def select_best_opportunity(opportunities, epsilon=0.05):
    """
    Answers "which step was THE best scoring option?" with a principle:

      1. ADMISSION is already handled upstream by the zone thresholds
         (3s need EV>=1.15, 2s need P>=0.60) -- the "pushing to shoot"
         early junk never becomes an opportunity in the first place.
      2. RANK by EXPECTED POINTS (P x shot value), not raw P -- a 0.38
         three (EV 1.14) beats a 0.55 two (EV 1.10).
      3. TIE-BREAK EARLIEST: among opportunities within epsilon EV of the
         peak, take the earliest step. Waiting has cost (turnover risk,
         clock) -- if the play stopped improving, the later "stretched"
         opportunities add risk without adding value. This is what cuts
         off the trailing GIFs that were 'stretching it for no reason'.

    Returns (best_opportunity, sorted_opportunities_by_step).
    """
    if not opportunities:
        return None, []
    by_step = sorted(opportunities, key=lambda o: o["step"])
    peak = max(o["ev"] for o in by_step)
    contenders = [o for o in by_step if o["ev"] >= peak - epsilon]
    return contenders[0], by_step


def animate_best_opportunity(steps, opportunities, strategy_name,
                             prefix="best_opportunity", epsilon=0.05):
    """One GIF: only the EV-peak (earliest-tie) opportunity."""
    best, _ = select_best_opportunity(opportunities, epsilon=epsilon)
    if best is None:
        print("  no admissible opportunities to animate")
        return None
    return animate_opportunities(steps, [best], strategy_name,
                                 prefix=prefix, min_gap=1)


def dedupe_opportunities(opps, min_gap=3):
    """Keep the best-P opportunity within any window of min_gap steps."""
    kept = []
    for o in sorted(opps, key=lambda o: o["step"]):
        if kept and o["step"] - kept[-1]["step"] < min_gap:
            if o["p"] > kept[-1]["p"]:
                kept[-1] = o
        else:
            kept.append(o)
    return kept


def animate_opportunities(steps, opportunities, strategy_name,
                          prefix="opportunity", min_gap=3):
    """
    One animation per qualifying shot opportunity: the play UP TO that step
    plus a shot at that moment, with its own resolved (sampled) outcome.
    Steps 4, 16, 20 above threshold -> 3 separate GIFs.
    """
    from step12_visualization import create_play_animation

    opps = dedupe_opportunities(opportunities, min_gap=min_gap)
    print(f"  {len(opportunities)} qualifying steps -> "
          f"{len(opps)} distinct opportunities after min_gap={min_gap}")

    paths = []
    for o in opps:
        cut = steps[:o["step"] + 1]
        # ensure the final frame carries this opportunity's P for the shot arc
        cut[-1] = dict(cut[-1]); cut[-1]["p_score"] = o["p"]
        made = bool(np.random.random() < o["p"])
        dist = math.dist(o["xy"], (BASKET_X, BASKET_Y))
        pts  = (3 if dist >= 23.75 else 2) if made else 0
        gif = f"{prefix}_step{o['step']:02d}_P{o['p']:.2f}" \
              f"_{'MADE' if made else 'MISS'}.gif"
        fr = {"play": cut, "recommended_strategy": strategy_name,
              "all_strategies": [{"name": strategy_name,
                                  "score_prob": o["p"],
                                  "shot_at_step": o["step"]}],
              "outcome": {"made": made, "points": pts, "p": o["p"]}}
        create_play_animation(fr, f"{strategy_name} (opportunity @ step "
                                  f"{o['step']}, {'MADE' if made else 'MISS'} "
                                  f"{pts}pts)", output_path=gif)
        paths.append(gif)
    return paths


# ===========================================================================
# QUICK CLI
# ===========================================================================

if __name__ == "__main__":
    import joblib
    shot = joblib.load("shot_scoring_model.pkl")
    lz   = joblib.load("le_zone.pkl")
    lt   = joblib.load("le_type.pkl")

    init = {"offense": [(22.,25.),(15.,10.),(15.,40.),(8.,8.),(8.,42.)],
            "defense": [(19.6,25.),(13.2,12.8),(13.2,37.2),(7.5,11.1),(7.5,38.9)],
            "ball": (22.,25.), "shot_clock": 24.0, "spacing_score": 28.0,
            "screen_detected": False, "defensive_overload": False,
            "nearest_def_to_ball_handler": 2.5}

    # ---- execute mode: planner plays the possession ----
    print("=== EXECUTE MODE (strategy 0) ===")
    steps, log, info = plan_possession_v7(0, init, shot, lz, lt,
                                          mode="execute", seed=1)
    for e in log:
        print(f"  step {e['step']:>2} A{e['handler']+1}: {e['chosen']:<12} "
              f"EV={e['ev']:.3f}   alts: {e['alternatives']}")
    print(f"  -> {info['reason']}  made={info['made']} pts={info['points']}")

    # ---- survey mode: full possession, one GIF per opportunity ----
    print("\n=== SURVEY MODE ===")
    steps, log, info = plan_possession_v7(0, init, shot, lz, lt,
                                          mode="survey", seed=2, max_steps=60)
    print(f"  opportunities >= 0.45: "
          f"{[(o['step'], round(o['p'],2)) for o in info['opportunities']]}")
    animate_opportunities(steps, info["opportunities"],
                          STRATEGY_NAMES[0], prefix="opportunity_s0")