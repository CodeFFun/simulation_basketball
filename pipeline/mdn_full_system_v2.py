"""
mdn_full_system_v2.py -- consolidated offense+defense+simulation, fixing:

  FIX 7 (BALL PASSING): the ball is now a 6th predicted entity. Ball deltas
      are DERIVED FROM YOUR EXISTING X files (cols 20:22 hold ball_x/ball_y
      per timestep -- consecutive differences include real passes, which
      appear as large fast jumps toward a teammate). No dataset rebuild.
      Inference adds flight/catch logic: a large sampled ball delta launches
      a pass; the ball travels until caught (within CATCH_RADIUS of an
      offensive player), and possession transfers.

  FIX 8 (SHOT OUTCOME): the shot is RESOLVED, not just scored -- outcome
      sampled ~ Bernoulli(P(score)), points 2/3 by distance. Rollouts
      return made/missed/points/expected_points, so at P=0.45 the shot
      misses ~55% of the time, as it should.

  FIX 9 (OFFENSE IGNORES DEFENSE): raw defender coordinates are nearly
      redundant given offense coordinates (defender ~= his man + offset),
      so the model learned ~zero weight on them -- which is why offense
      trajectories looked identical with and without defense. Now each
      offensive player's NEAREST-DEFENDER DISTANCE (openness) is added as
      an explicit input feature (5 dims, computed on the fly from X -- no
      rebuild). Openness is NOT derivable from offense positions alone, so
      the model has a learnable, non-redundant channel for defense -- and
      the ball model gets a reason to pass to open players.

  DEFENSE: unchanged from step10_mdn_defense.py (27 inputs) -- your
      already-trained defensive_response_mdn.pt LOADS AS-IS, no retrain.

  OFFENSE: input 27 -> 32 (adds 5 openness dims), output 5 players+ball
      mixtures -- REQUIRES RETRAINING the 6 offensive models (but not
      rebuilding movement_sequences_per_strategy/).

Workflow:
  1. train_all_offense_v2()        # retrains 6 offense models w/ ball+openness
  2. (defense: reuse defensive_response_mdn.pt, or retrain w/ step10 file)
  3. simulate_play_v6(...) / rank_strategies_v6(...)
"""

import os
import glob
import math
import random
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_SIZE   = 128
NUM_LAYERS    = 2
BATCH_SIZE    = 64
LEARNING_RATE = 0.001
DROPOUT_RATE  = 0.3
MAX_DELTA     = 10.0
K_MIX         = 4
BASKET_X, BASKET_Y = 5.25, 25.0

OFF_INPUT_SIZE  = 32   # 10 off + 10 def + 2 ball + 5 spatial + 5 openness
OFF_N_ENTITIES  = 6    # 5 players + ball
DEF_INPUT_SIZE  = 27   # unchanged -- existing defensive checkpoint loads
DEF_N_ENTITIES  = 5

CATCH_RADIUS  = 3.0    # ft -- ball within this of a player = possessed
PASS_TRIGGER  = 3.5    # ft/step -- sampled ball delta above this = pass launch

STRATEGY_NAMES = {
    0: "Motion Offense", 1: "Post Up", 2: "Free Throw Heavy",
    3: "Isolation", 4: "Pick and Roll", 5: "Fast Break",
}


def _coord_scale(n, spatial_start):
    s = np.ones(n, dtype=np.float32)
    for j in range(0, 22, 2): s[j] = 1.0 / 47.0
    for j in range(1, 22, 2): s[j] = 1.0 / 50.0
    s[spatial_start + 0] = 1.0 / 50.0   # spacing
    s[spatial_start + 3] = 1.0 / 47.0   # nearest_def_to_ball_handler
    s[spatial_start + 4] = 1.0 / 24.0   # shot_clock
    return s

OFF_SCALE = _coord_scale(OFF_INPUT_SIZE, 22)
OFF_SCALE[27:32] = 1.0 / 47.0            # per-player openness dims
DEF_SCALE = _coord_scale(DEF_INPUT_SIZE, 22)


# ===========================================================================
# ON-THE-FLY FEATURE/TARGET DERIVATION (from existing 27-col X files)
# ===========================================================================

def derive_openness(X27):
    """
    X27: (N, S, 27). Returns (N, S, 5): each offensive player's distance to
    his nearest defender. Not derivable from offense alone -> gives the
    model a non-redundant defense channel.
    """
    off = X27[:, :, 0:10].reshape(*X27.shape[:2], 5, 2)
    de  = X27[:, :, 10:20].reshape(*X27.shape[:2], 5, 2)
    diff = off[:, :, :, None, :] - de[:, :, None, :, :]     # (N,S,5,5,2)
    dist = np.sqrt((diff ** 2).sum(-1))                     # (N,S,5,5)
    return dist.min(-1)                                     # (N,S,5)


def derive_ball_targets(X27):
    """
    Ball deltas from consecutive stored ball positions (cols 20:22),
    normalized by MAX_DELTA and clipped. Passes are fast (can exceed
    MAX_DELTA) -- clipping preserves direction; the pass just takes an
    extra step to arrive. Returns (N, S, 2); last step / padded steps = 0.
    """
    ball = X27[:, :, 20:22]
    d = np.zeros_like(ball)
    d[:, :-1] = (ball[:, 1:] - ball[:, :-1]) / MAX_DELTA
    return np.clip(d, -1.0, 1.0)


def build_offense_batch(X27, y12):
    """
    From stored per-strategy chunks, build:
      X32 (N,S,32)  = X27 + openness(5)
      T   (N,S,6,2) = 5 player deltas (from y) + ball delta (derived)
      mask (N,S)    = real steps, player-glitch excluded, and requiring the
                      NEXT step to also be real (ball delta needs t+1)
    """
    openness = derive_openness(X27).astype(np.float32)
    X32 = np.concatenate([X27, openness], axis=-1)

    players = y12[:, :, :10].reshape(*y12.shape[:2], 5, 2)
    ball    = derive_ball_targets(X27)[:, :, None, :]
    T = np.concatenate([players, ball], axis=2).astype(np.float32)  # (N,S,6,2)

    real = (np.abs(X27).sum(-1) > 0)
    next_real = np.zeros_like(real)
    next_real[:, :-1] = real[:, 1:]
    glitch = (np.abs(y12[:, :, :10]) >= 0.999).any(-1)
    mask = (real & next_real & ~glitch).astype(np.float32)
    return X32, T, mask


# ===========================================================================
# MODELS
# ===========================================================================

class _MDNBase(nn.Module):
    def __init__(self, input_size, n_entities):
        super().__init__()
        self.n_entities = n_entities
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=HIDDEN_SIZE,
                            num_layers=NUM_LAYERS, batch_first=True,
                            dropout=DROPOUT_RATE)
        self.trunk = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(128, 128),         nn.LayerNorm(128), nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
        )
        self.mdn_out = nn.Linear(128, n_entities * K_MIX * 5)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        b, s, h = lstm_out.shape
        p = self.mdn_out(self.trunk(lstm_out.reshape(b*s, h)))
        p = p.reshape(b, s, self.n_entities, K_MIX, 5)
        return p[..., 0], p[..., 1:3], p[..., 3:5].clamp(-6.0, 1.0)

    def step(self, x_t, hidden):
        lstm_out, hidden = self.lstm(x_t, hidden)
        p = self.mdn_out(self.trunk(lstm_out.reshape(1, HIDDEN_SIZE)))
        p = p.reshape(self.n_entities, K_MIX, 5)
        return p[..., 0], p[..., 1:3], p[..., 3:5].clamp(-6.0, 1.0), hidden


class OffenseMDNv2(_MDNBase):
    def __init__(self): super().__init__(OFF_INPUT_SIZE, OFF_N_ENTITIES)

class DefenseMDN(_MDNBase):
    """Structurally identical to step10_mdn_defense.MDNDefensiveModel, so
    the existing defensive_response_mdn.pt state_dict loads directly."""
    def __init__(self): super().__init__(DEF_INPUT_SIZE, DEF_N_ENTITIES)


def mdn_nll(logits, mu, logsig, target, mask):
    t = target.unsqueeze(-2)
    inv_var = torch.exp(-2.0 * logsig)
    log_comp = (-0.5 * ((t - mu) ** 2 * inv_var).sum(-1)
                - logsig.sum(-1) - math.log(2 * math.pi))
    log_mix = torch.logsumexp(torch.log_softmax(logits, -1) + log_comp, -1)
    return -(log_mix.mean(-1) * mask).sum() / mask.sum().clamp(min=1.0)


def sample_mdn(logits, mu, logsig, temperature=1.0):
    n = logits.shape[0]
    w = torch.softmax(logits, dim=-1)
    if temperature <= 0.0:
        k = w.argmax(dim=-1)
        return mu[torch.arange(n), k]
    k = torch.multinomial(w, 1).squeeze(-1)
    cm  = mu[torch.arange(n), k]
    cs  = torch.exp(logsig[torch.arange(n), k])
    return cm + temperature * cs * torch.randn_like(cm)


# ===========================================================================
# TRAINING -- offense v2 (per strategy, existing split dirs)
# ===========================================================================

def train_offense_v2(strategy_idx, seq_dir="movement_sequences_per_strategy",
                     epochs=80, save_path=None):
    save_path = save_path or f"movement_generator_mdn_v2_{strategy_idx}.pt"
    strat_dir = os.path.join(seq_dir, f"strategy_{strategy_idx}")
    x_files = sorted(glob.glob(os.path.join(strat_dir, "X_*.npy")))
    if not x_files:
        print(f"  No data for {STRATEGY_NAMES[strategy_idx]} -- skipping.")
        return None

    model = OffenseMDNv2().to(DEVICE)
    opt   = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min",
                                                 patience=3, factor=0.5)
    scale_t = torch.tensor(OFF_SCALE, dtype=torch.float32).to(DEVICE)

    best, best_state = float("inf"), None
    print(f"\nTraining OffenseV2 {STRATEGY_NAMES[strategy_idx]} "
          f"({len(x_files)} chunks, ball+openness)...")

    for epoch in range(epochs):
        el, nb = 0.0, 0
        random.shuffle(x_files)
        for x_path in x_files:
            idx = x_path.split("_")[-1].replace(".npy", "")
            X27 = np.load(x_path)
            y12 = np.load(os.path.join(strat_dir, f"y_{idx}.npy"))
            X32, T, Mk = build_offense_batch(X27, y12)

            ds = TensorDataset(torch.tensor(X32), torch.tensor(T),
                               torch.tensor(Mk))
            dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
            model.train()
            for xb, tb, mb in dl:
                xb = xb.to(DEVICE) * scale_t
                tb, mb = tb.to(DEVICE), mb.to(DEVICE)
                opt.zero_grad()
                logits, mu, logsig = model(xb)
                loss = mdn_nll(logits, mu, logsig, tb, mb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                el += loss.item(); nb += 1
            del X27, y12, X32, T, Mk, ds, dl
            gc.collect()

        avg = el / max(nb, 1)
        sched.step(avg)
        if avg < best:
            best = avg
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save(best_state, save_path)
        if (epoch + 1) % 10 == 0:
            print(f"  [{STRATEGY_NAMES[strategy_idx]}] epoch {epoch+1}/{epochs} "
                  f"NLL {avg:.4f} best {best:.4f}")

    model.load_state_dict(best_state)
    print(f"  Saved {save_path} (best NLL {best:.4f})")
    return model


def train_all_offense_v2(seq_dir="movement_sequences_per_strategy", epochs=80):
    return {k: train_offense_v2(k, seq_dir, epochs=epochs) for k in range(6)}


# ===========================================================================
# SIMULATION v6 -- passing + resolved shot + both teams
# ===========================================================================

def _nearest_idx(bx, by, coords):
    return min(range(5), key=lambda j: (coords[j*2]-bx)**2 + (coords[j*2+1]-by)**2)

def _nearest_dist(px, py, coords):
    return min(((px-coords[j*2])**2 + (py-coords[j*2+1])**2) ** 0.5
               for j in range(5))

def _per_player_openness(off, deff):
    out = []
    for j in range(5):
        out.append(_nearest_dist(off[j*2], off[j*2+1], deff))
    return out

def _shot_zone(x, y):
    dist = ((x-BASKET_X)**2 + (y-BASKET_Y)**2) ** 0.5
    dy = abs(y-BASKET_Y)
    if dist <= 8.0 and dy <= 8.0: return "paint"
    if dist <= 15.0 and dy <= 6.0 and x > 13.0: return "free_throw"
    if x <= 14.0 and (y <= 8.0 or y >= 42.0): return "corner_3"
    if dist >= 23.75: return "above_break_3"
    return "mid_range"


def load_models_v6(strategy_idx, models_dir="."):
    om = OffenseMDNv2().to(DEVICE)
    om.load_state_dict(torch.load(
        os.path.join(models_dir, f"movement_generator_mdn_v2_{strategy_idx}.pt"),
        map_location=DEVICE))
    om.eval()
    dm = DefenseMDN().to(DEVICE)
    dm.load_state_dict(torch.load(
        os.path.join(models_dir, "defensive_response_mdn.pt"),
        map_location=DEVICE))
    dm.eval()
    return om, dm


def simulate_play_v6(
    strategy_idx, initial_positions, shot_scoring_model, le_zone, le_type,
    models_dir=".", max_steps=40, seed=None,
    off_temperature=0.7, def_temperature=0.5,
    score_prob_threshold=0.45, open_defender_threshold=6.0,
    shot_clock_threshold=4.0, pass_to_open=True,
):
    """
    Per step:
      1. offense MDN samples 5 player deltas + 1 ball delta
      2. players move; ball logic:
           possessed  + small ball delta -> ball stays with handler
           possessed  + LARGE ball delta -> PASS LAUNCHED (in flight)
           in flight  -> ball travels by its delta; caught when within
                         CATCH_RADIUS of an offensive player
         pass_to_open=True (default): when a pass launches, the RECEIVER is
         chosen with probability weighted by openness (softmax over each
         teammate's nearest-defender distance) and the flight is aimed at
         them. This is an explicit inference-time rule, not learned
         behavior -- it guarantees the most visible form of defensive
         awareness (ball finds the open man). pass_to_open=False uses the
         raw sampled flight direction instead.
      3. defense MDN responds to new offense
      4. shot decision only while POSSESSED; shot RESOLVED by sampling
         Bernoulli(P(score)); points 2/3 by distance
    Returns steps, score_probs, info dict (shot_step, reason, made, points,
    expected_points, n_passes).
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    off_model, def_model = load_models_v6(strategy_idx, models_dir)
    off_sc = torch.tensor(OFF_SCALE, dtype=torch.float32).to(DEVICE)
    def_sc = torch.tensor(DEF_SCALE, dtype=torch.float32).to(DEVICE)

    off  = [c for p in initial_positions["offense"] for c in p]
    deff = [c for p in initial_positions["defense"] for c in p]
    ball_x, ball_y = initial_positions["ball"]

    in_flight  = False
    flight_target = None
    handler    = _nearest_idx(ball_x, ball_y, off)
    n_passes   = 0

    steps = [{"offense": list(initial_positions["offense"]),
              "defense": list(initial_positions["defense"]),
              "ball": initial_positions["ball"],
              "shot_clock": initial_positions.get("shot_clock", 24.0)}]
    score_probs = []
    oh, dh = None, None
    info = {"shot_step": None, "reason": None, "made": None,
            "points": 0, "expected_points": None, "n_passes": 0}
    best_step, best_p, best_dist = None, -1.0, None

    with torch.no_grad():
        for step in range(max_steps):
            clock   = max(0.0, initial_positions.get("shot_clock", 24.0) - step*0.2)
            spacing = initial_positions.get("spacing_score", 20.0)
            overload= initial_positions.get("defensive_overload", False)
            screen  = float(initial_positions.get("screen_detected", False))
            pre_d   = _nearest_dist(ball_x, ball_y, deff)
            spatial = [spacing, screen, float(overload), pre_d, clock]
            openness= _per_player_openness(off, deff)

            # --- 1-2. offense + ball ---
            xin = off + deff + [ball_x, ball_y] + spatial + openness
            x = torch.tensor([[xin]], dtype=torch.float32).to(DEVICE) * off_sc
            lo, mo, so, oh = off_model.step(x, oh)
            d = sample_mdn(lo, mo, so, temperature=off_temperature).cpu().numpy()
            d_players, d_ball = d[:5] * MAX_DELTA, d[5] * MAX_DELTA

            new_off = []
            for j in range(5):
                nx = max(0.0, min(94.0, off[j*2]   + d_players[j, 0]))
                ny = max(0.0, min(50.0, off[j*2+1] + d_players[j, 1]))
                new_off.extend([nx, ny])
            off = new_off

            ball_speed = float(np.hypot(*d_ball))
            if in_flight:
                if pass_to_open and flight_target is not None:
                    # fly toward the receiver's CURRENT position (lead the man)
                    tx, ty = off[flight_target*2], off[flight_target*2+1]
                    vec = np.array([tx - ball_x, ty - ball_y])
                    nrm = np.linalg.norm(vec) + 1e-6
                    spd = max(ball_speed, 6.0)          # passes don't stall
                    ball_x = max(0.0, min(94.0, ball_x + vec[0]/nrm*min(spd, nrm)))
                    ball_y = max(0.0, min(50.0, ball_y + vec[1]/nrm*min(spd, nrm)))
                else:
                    ball_x = max(0.0, min(94.0, ball_x + d_ball[0]))
                    ball_y = max(0.0, min(50.0, ball_y + d_ball[1]))
                j = _nearest_idx(ball_x, ball_y, off)
                if math.dist((ball_x, ball_y),
                             (off[j*2], off[j*2+1])) < CATCH_RADIUS:
                    in_flight, handler = False, j
                    flight_target = None
                    ball_x, ball_y = off[j*2], off[j*2+1]
            else:
                if ball_speed > PASS_TRIGGER:
                    in_flight = True
                    n_passes += 1
                    hx, hy = off[handler*2], off[handler*2+1]
                    if pass_to_open:
                        # choose receiver ~ softmax(openness); aim flight
                        cand = [j for j in range(5) if j != handler]
                        opn  = np.array([openness[j] for j in cand])
                        w    = np.exp(opn / 3.0)
                        recv = cand[int(np.random.choice(len(cand),
                                                         p=w / w.sum()))]
                        flight_target = recv
                        tx, ty = off[recv*2], off[recv*2+1]
                        vec = np.array([tx - hx, ty - hy])
                        nrm = np.linalg.norm(vec) + 1e-6
                        step_vec = vec / nrm * min(ball_speed, nrm)
                        ball_x = max(0.0, min(94.0, hx + step_vec[0]))
                        ball_y = max(0.0, min(50.0, hy + step_vec[1]))
                    else:
                        ball_x = max(0.0, min(94.0, hx + d_ball[0]))
                        ball_y = max(0.0, min(50.0, hy + d_ball[1]))
                else:
                    ball_x, ball_y = off[handler*2], off[handler*2+1]

            # --- 3. defense responds ---
            din = off + deff + [ball_x, ball_y] + spatial
            xd = torch.tensor([[din]], dtype=torch.float32).to(DEVICE) * def_sc
            ld, md, sd, dh = def_model.step(xd, dh)
            dd = sample_mdn(ld, md, sd,
                            temperature=def_temperature).cpu().numpy() * MAX_DELTA
            new_def = []
            for j in range(5):
                nx = max(0.0, min(94.0, deff[j*2]   + dd[j, 0]))
                ny = max(0.0, min(50.0, deff[j*2+1] + dd[j, 1]))
                new_def.extend([nx, ny])
            deff = new_def

            # --- 4. shot decision (only while possessed) ---
            p_score = None
            if not in_flight:
                def_dist = _nearest_dist(ball_x, ball_y, deff)
                dist = ((ball_x-BASKET_X)**2 + (ball_y-BASKET_Y)**2) ** 0.5
                try:    ze = le_zone.transform([_shot_zone(ball_x, ball_y)])[0]
                except ValueError: ze = 0
                try:    te = le_type.transform(["jump_shot"])[0]
                except ValueError: te = 0
                feats = pd.DataFrame([{
                    "shot_x": ball_x, "shot_y": ball_y, "shot_distance": dist,
                    "shot_zone_enc": ze, "shot_type_enc": te,
                    "shot_clock": clock, "quarter": 1,
                    "is_three_pointer": 1 if dist >= 23.75 else 0,
                    "nearest_def_to_ball_handler": def_dist,
                    "spacing_score": spacing,
                    "defensive_overload": int(overload),
                }])
                p_score = float(shot_scoring_model.predict_proba(feats)[:, 1][0])
                score_probs.append(p_score)
                if p_score > best_p:
                    best_p, best_step, best_dist = p_score, step + 1, dist

            steps.append({
                "offense": [(off[j*2],  off[j*2+1])  for j in range(5)],
                "defense": [(deff[j*2], deff[j*2+1]) for j in range(5)],
                "ball": (ball_x, ball_y), "p_score": p_score,
                "in_flight": in_flight,
                "shot_clock": clock,
            })

            if p_score is not None:
                fire = None
                if p_score > score_prob_threshold:
                    fire = f"high quality shot (P={p_score:.3f})"
                elif def_dist > open_defender_threshold:
                    fire = f"wide open ({def_dist:.1f}ft)"
                elif clock < shot_clock_threshold:
                    fire = f"forced ({clock:.1f}s)"
                if fire:
                    info["shot_step"], info["reason"] = step + 1, fire
                    _resolve_shot(info, p_score, dist)
                    break

    if info["shot_step"] is None:
        info["shot_step"] = best_step
        info["reason"]    = f"fallback best (P={best_p:.3f})"
        if best_p >= 0:
            _resolve_shot(info, best_p, best_dist or 10.0)

    info["n_passes"] = n_passes
    return steps, score_probs, info


def _resolve_shot(info, p_score, dist):
    """FIX 8: shot resolved probabilistically -- misses happen."""
    made  = bool(np.random.random() < p_score)
    worth = 3 if dist >= 23.75 else 2
    info["made"]            = made
    info["points"]          = worth if made else 0
    info["expected_points"] = round(p_score * worth, 3)


def rank_strategies_v6(initial_positions, shot_scoring_model, le_zone, le_type,
                       models_dir=".", n_rollouts=7,
                       off_temperature=0.7, def_temperature=0.5, max_steps=40):
    """
    Rank by median EXPECTED POINTS (P(score) x shot value) -- more honest
    than raw P, since it credits open threes properly. Reports pass counts
    and simulated make rate too.
    """
    results = {}
    for k in range(6):
        evs, passes, mades, plays = [], [], [], []
        for r in range(n_rollouts):
            steps, probs, info = simulate_play_v6(
                k, initial_positions, shot_scoring_model, le_zone, le_type,
                models_dir=models_dir, max_steps=max_steps, seed=1000*k + r,
                off_temperature=off_temperature, def_temperature=def_temperature)
            evs.append(info["expected_points"] or 0.0)
            passes.append(info["n_passes"])
            mades.append(1 if info["made"] else 0)
            plays.append((info["expected_points"] or 0.0, steps, info))
        med = float(np.median(evs))
        bp = max(plays, key=lambda t: t[0])
        results[k] = {"median_ev": med, "evs": evs,
                      "avg_passes": float(np.mean(passes)),
                      "sim_make_rate": float(np.mean(mades)),
                      "best_play": bp[1], "best_info": bp[2]}
        print(f"{STRATEGY_NAMES[k]:<18} median EV={med:.3f} pts  "
              f"avg passes={np.mean(passes):.1f}  "
              f"sim makes={np.mean(mades):.0%}")
    best_k = max(results, key=lambda k: results[k]["median_ev"])
    print(f"\n-> Recommended: {STRATEGY_NAMES[best_k]} "
          f"(median EV={results[best_k]['median_ev']:.3f} pts)")
    return results, best_k


if __name__ == "__main__":
    train_all_offense_v2(epochs=80)
    print("\nDone. Reuse your existing defensive_response_mdn.pt; then use")
    print("simulate_play_v6 / rank_strategies_v6.")