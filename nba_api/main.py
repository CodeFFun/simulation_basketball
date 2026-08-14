"""
nba_api/main.py  --  FastAPI service around the Step-1..12 pipeline.

Wraps action_planner_v7.plan_possession_v7 (single strategy or best-of-6),
resolves the shot outcome, and renders a slow-mo GIF per request that the
frontend can display. All heavy artifacts (6 offense MDNs, 1 defense MDN,
shot-scoring XGBoost + 2 label encoders) are loaded ONCE at startup.

Run:
    uvicorn nba_api.main:app --host 0.0.0.0 --port 8000
Then open http://localhost:8000/  (static frontend) or the interactive API
docs at http://localhost:8000/docs .
"""

import os
import time
import uuid
import math
import pathlib
import threading
from typing import List, Optional, Tuple, Dict, Any

# Lock matplotlib to the headless, thread-safe Agg backend at the process
# entry point -- BEFORE any import below can pull in matplotlib.pyplot (the
# planner / visualization modules do). GIFs are rendered from FastAPI worker
# threads, where an interactive (Tk) backend crashes. Honors MPLBACKEND.
import matplotlib
if not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg", force=True)

import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# --- pipeline imports (these modules must be importable: see PYTHONPATH) -----
from action_planner_v7 import (
    plan_possession_v7, select_best_opportunity,
    OFFENSE_MODES, DEFENSE_PRESETS, TEMPO_PRESETS, BASKET_X, BASKET_Y,
    THREE_THRESHOLD, TWO_THRESHOLD,
)
from mdn_full_system_v2 import STRATEGY_NAMES
from step12_visualization import create_play_animation


# ===========================================================================
# CONFIG
# ===========================================================================

MODELS_DIR = os.environ.get("NBA_MODELS_DIR", ".")
HERE       = pathlib.Path(__file__).parent
GIF_DIR    = HERE / "static" / "gifs"
GIF_DIR.mkdir(parents=True, exist_ok=True)

# The movement/defense MDNs were trained ONLY on offensive-half-court
# possessions, so the model's valid territory is x in [0, HALF_COURT_X].
# Positions beyond it (a staged fast break) are out of distribution; we snap
# them to the half-court entry line before the model ever sees them.
HALF_COURT_X   = 47.0
ENTRY_INSET    = 2.0     # how far inside half court a snapped player is placed

# Torch model inference is not thread-safe across requests here; serialize the
# planner with a lock so concurrent requests don't corrupt the RNG/model state.
_PLANNER_LOCK = threading.Lock()
# turnover -> how many fresh seeds to try for a non-turnover alternate play
ALT_MAX_TRIES = 6

# GIF retention: prune files older than this many seconds on each new render.
GIF_TTL_SECONDS = int(os.environ.get("NBA_GIF_TTL", "3600"))


# ===========================================================================
# MODEL REGISTRY (loaded once)
# ===========================================================================

class Models:
    shot = None
    le_zone = None
    le_type = None
    ready = False
    error = None

def load_models():
    try:
        Models.shot    = joblib.load(os.path.join(MODELS_DIR, "shot_scoring_model.pkl"))
        Models.le_zone = joblib.load(os.path.join(MODELS_DIR, "le_zone.pkl"))
        Models.le_type = joblib.load(os.path.join(MODELS_DIR, "le_type.pkl"))
        # The 6 offense MDNs + defense MDN are loaded lazily inside
        # plan_possession_v7 from MODELS_DIR; we just verify they exist so the
        # health check fails loudly at startup instead of mid-request.
        missing = []
        for k in range(6):
            f = os.path.join(MODELS_DIR, f"movement_generator_mdn_v2_{k}.pt")
            if not os.path.exists(f):
                missing.append(os.path.basename(f))
        if not os.path.exists(os.path.join(MODELS_DIR, "defensive_response_mdn.pt")):
            missing.append("defensive_response_mdn.pt")
        if missing:
            raise FileNotFoundError(f"missing model files in {MODELS_DIR}: {missing}")
        Models.ready = True
    except Exception as e:                       # noqa: BLE001
        Models.error = str(e)
        Models.ready = False


# ===========================================================================
# SCHEMAS
# ===========================================================================

Point = Tuple[float, float]

class PlayRequest(BaseModel):
    offense: List[Point] = Field(..., description="5 [x,y] offensive positions")
    defense: List[Point] = Field(..., description="5 [x,y] defensive positions")
    ball:    Point       = Field(..., description="[x,y] ball position")

    strategy: Optional[int] = Field(
        None, ge=0, le=5,
        description="0-5 to force a strategy; null = auto-pick the best of 6")
    offense_mode:   str = Field("normal", description="normal | 2pt | 3pt")
    defense_preset: str = Field("normal",
        description="normal | man_to_man | high_turnovers")
    tempo: str = Field("default",
        description="possession pacing: default | quick (<=3 dribbles/touch, "
                    "shot inside ~6s) | patient (3-5 dribbles/touch, play "
                    "develops >=4s first) | milk_clock (swing it, only shoot "
                    "in the last 5s)")

    shot_clock:    float = Field(19.0, ge=1.0, le=24.0)
    is_fast_break: bool  = False
    spacing_score: Optional[float] = Field(
        None, description="offensive spread; null = auto-computed from positions")
    screen_detected:    bool = False
    defensive_overload: bool = False
    nearest_def_to_ball_handler: Optional[float] = None

    make_gif: bool = Field(True, description="render + return a GIF")
    seed:     Optional[int] = Field(None, description="fix RNG for reproducibility")
    n_rollouts: int = Field(5, ge=1, le=25,
        description="rollouts to median over when auto-picking a strategy")
    full_court: bool = Field(False,
        description="positions may span the full court (staged fast break); "
                    "back-court players are snapped to the half-court entry "
                    "line before the model runs and shown arriving in transition")

    @field_validator("offense", "defense")
    @classmethod
    def _five_points(cls, v):
        if len(v) != 5:
            raise ValueError("must be exactly 5 [x,y] points")
        for p in v:
            if len(p) != 2:
                raise ValueError("each point must be [x, y]")
        return v

    @field_validator("offense_mode")
    @classmethod
    def _valid_off(cls, v):
        if v not in OFFENSE_MODES:
            raise ValueError(f"offense_mode must be one of {list(OFFENSE_MODES)}")
        return v

    @field_validator("defense_preset")
    @classmethod
    def _valid_def(cls, v):
        if v not in DEFENSE_PRESETS:
            raise ValueError(f"defense_preset must be one of {list(DEFENSE_PRESETS)}")
        return v

    @field_validator("tempo")
    @classmethod
    def _valid_tempo(cls, v):
        if v not in TEMPO_PRESETS:
            raise ValueError(f"tempo must be one of {list(TEMPO_PRESETS)}")
        return v


class StrategyScore(BaseModel):
    strategy: int
    name: str
    median_ev: float

class PlayResponse(BaseModel):
    request_id: str
    recommended_strategy: int
    recommended_name: str
    offense_mode: str
    defense_preset: str
    tempo: str
    expected_points: float
    made: Optional[bool]
    points: int
    turnover: bool
    shot_p: Optional[float]
    shot_step: Optional[int]
    n_passes: int
    reason: Optional[str]
    ball_chain: List[int]
    ranking: List[StrategyScore]
    n_steps: int
    spacing_score: float
    full_court: bool
    transition: List[Dict[str, Any]]
    gif_url: Optional[str]
    # set when the play ended in a TURNOVER: a SEPARATE possession of the
    # same strategy/setup (fresh seeds, retried until one doesn't turn it
    # over) that runs to a shot -- shown alongside the turnover film.
    gif_url_counterfactual: Optional[str] = None
    counterfactual: Optional[Dict[str, Any]] = None
    seed: int = 0            # the seed this play actually ran with


# ===========================================================================
# HELPERS
# ===========================================================================

def compute_spacing_score(offense) -> float:
    """Proxy for the Step-4 spacing feature: mean pairwise distance among the
    five offensive players (in feet). More spread -> higher score. This mirrors
    what the shot model saw at training time closely enough to stand in when the
    caller doesn't supply one -- the user places players, we measure the spread
    so they never have to guess a number."""
    n = len(offense)
    if n < 2:
        return 20.0
    dists = [math.dist(tuple(offense[i]), tuple(offense[j]))
             for i in range(n) for j in range(i + 1, n)]
    return round(float(sum(dists) / len(dists)), 2)


def stage_full_court(offense, defense, ball, full_court):
    """Option A: keep the model in-distribution.

    If full_court is on, any player past the half-court line (x > HALF_COURT_X)
    is 'in transition'. For the MODEL we snap them just inside the line at
    x = HALF_COURT_X - ENTRY_INSET (keeping their y lane), so every coordinate
    the network sees is within the offensive-half distribution it was trained
    on. We return:
      - model_off / model_def / model_ball : snapped, safe to simulate
      - transition : list of {team, idx, from:[x,y], to:[x,y]} so the frontend
        can animate those players arriving before the simulated play begins.
    When full_court is off this is a no-op (just clamps stray values).
    """
    transition = []

    def snap(pts, team):
        out = []
        for i, (x, y) in enumerate(pts):
            if full_court and x > HALF_COURT_X:
                nx = HALF_COURT_X - ENTRY_INSET
                ny = min(49.5, max(0.5, y))
                transition.append({"team": team, "idx": i,
                                   "from": [round(x, 2), round(y, 2)],
                                   "to":   [round(nx, 2), round(ny, 2)]})
                out.append((nx, ny))
            else:
                out.append((min(HALF_COURT_X, max(0.0, x)),
                            min(50.0, max(0.0, y))))
        return out

    m_off = snap([tuple(p) for p in offense], "offense")
    m_def = snap([tuple(p) for p in defense], "defense")
    bx, by = ball
    m_ball = (min(HALF_COURT_X, max(0.0, bx)), min(50.0, max(0.0, by)))
    return m_off, m_def, m_ball, transition


def _scenario(req: PlayRequest) -> Dict[str, Any]:
    m_off, m_def, m_ball, transition = stage_full_court(
        req.offense, req.defense, req.ball, req.full_court)

    ndh = req.nearest_def_to_ball_handler
    if ndh is None:
        bx, by = m_ball
        ndh = min(math.dist((bx, by), tuple(d)) for d in m_def)
    spacing = (req.spacing_score if req.spacing_score is not None
               else compute_spacing_score(m_off))
    scenario = {
        "offense": m_off,
        "defense": m_def,
        "ball": m_ball,
        "shot_clock": req.shot_clock,
        "is_fast_break": req.is_fast_break,
        "spacing_score": spacing,
        "screen_detected": req.screen_detected,
        "defensive_overload": req.defensive_overload,
        "nearest_def_to_ball_handler": ndh,
    }
    return scenario, transition

def _holder_seq(steps) -> List[int]:
    seq = []
    for s in steps:
        if s.get("in_flight", False):
            continue
        bx, by = s["ball"]
        j = min(range(5), key=lambda k: (s["offense"][k][0]-bx)**2
                                       + (s["offense"][k][1]-by)**2)
        if not seq or seq[-1] != j:
            seq.append(j)
    return seq

def _shot_p_of(steps) -> Optional[float]:
    for s in reversed(steps):
        if s.get("p_score") is not None:
            return round(float(s["p_score"]), 4)
    return None

def _outcome_dict(steps, info) -> Dict[str, Any]:
    return {"made": bool(info.get("made")), "points": info.get("points", 0),
            "p": _shot_p_of(steps), "turnover": bool(info.get("turnover"))}

def _prepend_transition(steps, transition, n_frames=6):
    """Build 'arrival' steps that move the transitioning players from their
    back-court origin to where the simulation actually starts, then hand off
    to the real (model-simulated) steps. Purely visual: these frames are NOT
    model output, they interpolate the approach so the GIF reads as a fast
    break resolving into the half court. Non-transitioning players hold at
    their simulation-start spots."""
    if not transition or not steps:
        return steps
    start = steps[0]
    off0 = [list(p) for p in start["offense"]]
    def0 = [list(p) for p in start["defense"]]
    origin_off = [list(p) for p in off0]
    origin_def = [list(p) for p in def0]
    # The GIF diagram spans x in [-1, 50]; a player staged at x=70+ would be
    # off-frame and appear to teleport in. Cap the VISUAL origin to the frame
    # edge (x=49) so transitioning players slide in from the right sideline
    # toward the half-court line. This is a rendering choice only -- the model
    # already received the snapped (<=47) positions; nothing here touches it.
    GIF_EDGE_X = 49.0
    for t in transition:
        fx, fy = t["from"]
        vx = min(GIF_EDGE_X, fx)
        (origin_off if t["team"] == "offense" else origin_def)[t["idx"]] = [vx, fy]

    intro = []
    for f in range(n_frames):
        a = f / n_frames
        blend = lambda o, d: [(o[i][0]*(1-a) + d[i][0]*a,
                               o[i][1]*(1-a) + d[i][1]*a) for i in range(5)]
        intro.append({
            "offense": blend(origin_off, off0),
            "defense": blend(origin_def, def0),
            "ball": tuple(start["ball"]),
            "p_score": None, "in_flight": False,
            "shot_clock": start.get("shot_clock", 24.0),
            "transition": True,
        })
    return intro + steps


def _prune_old_gifs():
    now = time.time()
    for f in GIF_DIR.glob("*.gif"):
        try:
            if now - f.stat().st_mtime > GIF_TTL_SECONDS:
                f.unlink()
        except OSError:
            pass

def _max_steps_for(req: PlayRequest) -> int:
    """Steps needed to cover the whole clock (0.2s each) with headroom --
    milk_clock tempo must be able to reach its late-shot window."""
    return max(60, int(math.ceil(req.shot_clock / 0.2)) + 10)

def _rank_all(scenario, req) -> Tuple[int, List[StrategyScore]]:
    """Median EV per strategy over n_rollouts; returns (best_k, ranking)."""
    ranking = []
    steps_cap = _max_steps_for(req)
    for k in range(6):
        evs = []
        for s in range(req.n_rollouts):
            base = (req.seed if req.seed is not None else 0)
            _st, _lg, info = plan_possession_v7(
                k, scenario, Models.shot, Models.le_zone, Models.le_type,
                models_dir=MODELS_DIR, mode="execute",
                seed=base + 1000*k + s, max_steps=steps_cap,
                offense_mode=req.offense_mode, defense_preset=req.defense_preset,
                tempo=req.tempo)
            evs.append(info["expected_points"] or 0.0)
        ranking.append(StrategyScore(strategy=k, name=STRATEGY_NAMES[k],
                                     median_ev=round(float(np.median(evs)), 4)))
    best_k = max(range(6), key=lambda k: ranking[k].median_ev)
    ranking.sort(key=lambda r: -r.median_ev)
    return best_k, ranking


# ===========================================================================
# APP
# ===========================================================================

app = FastAPI(title="NBA Offensive Strategy Recommender",
              version="1.0",
              description="Positions in -> strategy + simulated play + GIF out.")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
    allow_headers=["*"])

@app.on_event("startup")
def _startup():
    load_models()

@app.get("/api/health")
def health():
    return {"status": "ok" if Models.ready else "not_ready",
            "models_dir": os.path.abspath(MODELS_DIR),
            "error": Models.error}

@app.get("/api/options")
def options():
    """Everything the frontend needs to build its controls."""
    return {
        "strategies": [{"id": k, "name": STRATEGY_NAMES[k]} for k in range(6)],
        "offense_modes": [
            {"id": "normal", "label": "Normal (best EV)"},
            {"id": "2pt",    "label": "Force 2-pointer"},
            {"id": "3pt",    "label": "Force 3-pointer"},
        ],
        "defense_presets": [
            {"id": "normal",         "label": "Normal"},
            {"id": "man_to_man",     "label": "Man-to-man"},
            {"id": "high_turnovers", "label": "High turnovers (ball-hawk)"},
        ],
        "tempos": [
            {"id": "default",    "label": "Default"},
            {"id": "quick",      "label": "Quick (shot in ~6s)"},
            {"id": "patient",    "label": "Patient (develop ≥4s)"},
            {"id": "milk_clock", "label": "Milk clock (shoot last 5s)"},
        ],
        "court": {"basket": [BASKET_X, BASKET_Y],
                  "x_range": [0, 47], "y_range": [0, 50],
                  "attack_direction": "left"},
        "thresholds": {"three": THREE_THRESHOLD, "two": TWO_THRESHOLD},
        "roster": {"offense": 5, "defense": 5,
                   "note": "The model requires exactly 5 per side. Place the "
                           "players you care about; the rest are auto-filled."},
    }

@app.post("/api/simulate", response_model=PlayResponse)
def simulate(req: PlayRequest):
    if not Models.ready:
        raise HTTPException(503, f"models not loaded: {Models.error}")

    scenario, transition = _scenario(req)
    request_id = uuid.uuid4().hex[:12]

    # A CONCRETE seed for the returned play (random if the caller didn't fix
    # one), echoed back so a specific play can be reproduced. Alternate
    # possessions on a turnover derive their seeds from it. Ranking keeps its
    # own deterministic base (req.seed or 0), unchanged.
    play_seed = req.seed if req.seed is not None \
                else int(np.random.randint(0, 2**31 - 1))

    with _PLANNER_LOCK:
        # 1) pick strategy
        if req.strategy is None:
            best_k, ranking = _rank_all(scenario, req)
        else:
            best_k = req.strategy
            ranking = [StrategyScore(strategy=best_k, name=STRATEGY_NAMES[best_k],
                                     median_ev=0.0)]

        # 2) one execute rollout of the chosen strategy = the play we return
        steps_cap = _max_steps_for(req)
        steps, _log, info = plan_possession_v7(
            best_k, scenario, Models.shot, Models.le_zone, Models.le_type,
            models_dir=MODELS_DIR, mode="execute",
            seed=play_seed, max_steps=steps_cap,
            offense_mode=req.offense_mode,
            defense_preset=req.defense_preset, tempo=req.tempo)

        # 2b) TURNOVER -> ALTERNATE POSSESSION. Instead of "the same play if
        # the pass had connected", run the SAME setup again with fresh seeds
        # until one resolves in a shot (made or missed) rather than a
        # turnover -- a genuinely different possession of the same strategy.
        # Give up after a few tries (some setups turn it over most of the
        # time; then we just show the best non-turnover we found, or None).
        cf_steps, cf_info = None, None
        if info.get("turnover"):
            for k in range(ALT_MAX_TRIES):
                alt_seed = (play_seed + 7919 * (k + 1)) % (2**31 - 1)
                a_steps, _a_log, a_info = plan_possession_v7(
                    best_k, scenario, Models.shot, Models.le_zone, Models.le_type,
                    models_dir=MODELS_DIR, mode="execute",
                    seed=alt_seed, max_steps=steps_cap,
                    offense_mode=req.offense_mode,
                    defense_preset=req.defense_preset, tempo=req.tempo)
                if not a_info.get("turnover"):
                    cf_steps, cf_info = a_steps, a_info
                    break

        # 3) render GIF(s) (optional)
        gif_url, cf_gif_url = None, None
        if req.make_gif:
            _prune_old_gifs()
            safe = STRATEGY_NAMES[best_k].replace(" ", "_")
            fname = f"play_{request_id}_{safe}.gif"
            play_steps = _prepend_transition(steps, transition)
            fr = {"play": play_steps,
                  "recommended_strategy": STRATEGY_NAMES[best_k],
                  "all_strategies": [{"name": STRATEGY_NAMES[best_k],
                                      "score_prob": info["expected_points"] or 0.0,
                                      "shot_at_step": info.get("shot_step") or 0}],
                  "outcome": _outcome_dict(steps, info)}
            create_play_animation(fr, STRATEGY_NAMES[best_k],
                                  output_path=str(GIF_DIR / fname))
            gif_url = f"/static/gifs/{fname}"

            if cf_steps is not None:
                cf_name = f"play_{request_id}_{safe}_alt.gif"
                cf_play = _prepend_transition(cf_steps, transition)
                cf_fr = {"play": cf_play,
                         "recommended_strategy": STRATEGY_NAMES[best_k],
                         "all_strategies": [{
                             "name": STRATEGY_NAMES[best_k],
                             "score_prob": cf_info["expected_points"] or 0.0,
                             "shot_at_step": cf_info.get("shot_step") or 0}],
                         "outcome": _outcome_dict(cf_steps, cf_info)}
                create_play_animation(
                    cf_fr, f"{STRATEGY_NAMES[best_k]} (alternate possession)",
                    output_path=str(GIF_DIR / cf_name))
                cf_gif_url = f"/static/gifs/{cf_name}"

    return PlayResponse(
        request_id=request_id,
        recommended_strategy=best_k,
        recommended_name=STRATEGY_NAMES[best_k],
        offense_mode=req.offense_mode,
        defense_preset=req.defense_preset,
        tempo=req.tempo,
        expected_points=round(info["expected_points"] or 0.0, 4),
        made=info.get("made"),
        points=info.get("points", 0),
        turnover=bool(info.get("turnover")),
        shot_p=_shot_p_of(steps),
        shot_step=info.get("shot_step"),
        n_passes=info.get("n_passes", 0),
        reason=info.get("reason"),
        ball_chain=_holder_seq(steps),
        ranking=ranking,
        n_steps=len(steps),
        spacing_score=scenario["spacing_score"],
        full_court=req.full_court,
        transition=transition,
        gif_url=gif_url,
        gif_url_counterfactual=cf_gif_url,
        counterfactual=(None if cf_info is None else {
            "made": cf_info.get("made"),
            "points": cf_info.get("points", 0),
            "shot_p": _shot_p_of(cf_steps),
            "expected_points": round(cf_info["expected_points"] or 0.0, 4),
            "n_passes": cf_info.get("n_passes", 0),
            "reason": cf_info.get("reason"),
            "kind": "alternate_possession",
        }),
        seed=play_seed,
    )


# --- static frontend + served GIFs (mounted last so /api/* wins) ------------
app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")

@app.get("/")
def index():
    return FileResponse(str(HERE / "static" / "index.html"))