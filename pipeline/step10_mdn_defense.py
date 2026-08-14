"""
step10_mdn_defense.py

FIX 6 applied to Step 10 -- MDN defensive response model -- plus the
INTEGRATED rollout that finally moves BOTH teams (fixing the frozen-defense
gap visible in every diagnostic so far: "def mean d = 0.00").

Same principles as step9_mdn_models.py:
  - input normalization (~[0,1]) so gates don't saturate
  - masked NLL over real timesteps only, glitch steps excluded
  - Mixture Density head: each defender's next (dx,dy) is a K-component
    Gaussian mixture; inference SAMPLES so defenders COMMIT to discrete
    choices (help vs stay home, go over vs under a screen) instead of
    outputting the average of all choices (= the collapse-to-paint drift)

Differences from Step 9, by design:
  - ONE model, not six: defense reacts to what it sees, not to the
    offense's play call, so there's no strategy split
  - trains on your EXISTING defensive_sequences/def_X_i.npy etc.
    (built after the identity-tracking fix) -- NO REBUILD NEEDED
  - default temperature lower (0.5): defensive reaction is genuinely more
    deterministic than offensive creativity

Also provides simulate_play_v5(): the full two-sided rollout --
  offense MDN moves -> ball tracks handler -> defense MDN responds to the
  NEW offensive positions -> shot quality evaluated with POST-movement
  defender distances (same ordering your step11 described).
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

from step9_mdn_models import (
    MDNMovementModel, sample_mdn, INPUT_SCALE as OFF_INPUT_SCALE,
    STRATEGY_NAMES,
)

DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEF_INPUT_SIZE = 27     # 10 off + 10 def + 2 ball + 5 spatial
HIDDEN_SIZE    = 128
NUM_LAYERS     = 2
BATCH_SIZE     = 64
LEARNING_RATE  = 0.001
DROPOUT_RATE   = 0.3
MAX_DELTA      = 10.0
N_PLAYERS      = 5
K_MIX          = 4
BASKET_X, BASKET_Y = 5.25, 25.0


def build_def_input_scale():
    s = np.ones(DEF_INPUT_SIZE, dtype=np.float32)
    for j in range(0, 22, 2):
        s[j] = 1.0 / 47.0
    for j in range(1, 22, 2):
        s[j] = 1.0 / 50.0
    s[22] = 1.0 / 50.0
    s[25] = 1.0 / 47.0
    s[26] = 1.0 / 24.0
    return s

DEF_INPUT_SCALE = build_def_input_scale()


# ===========================================================================
# MODEL
# ===========================================================================

class MDNDefensiveModel(nn.Module):
    """LSTM -> per-defender 2D Gaussian mixture over the next delta."""

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=DEF_INPUT_SIZE, hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS, batch_first=True, dropout=DROPOUT_RATE,
        )
        self.trunk = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(128, 128),         nn.LayerNorm(128), nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
        )
        self.mdn_out = nn.Linear(128, N_PLAYERS * K_MIX * 5)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        b, s, h = lstm_out.shape
        z = self.trunk(lstm_out.reshape(b * s, h))
        p = self.mdn_out(z).reshape(b, s, N_PLAYERS, K_MIX, 5)
        return p[..., 0], p[..., 1:3], p[..., 3:5].clamp(-6.0, 1.0)

    def step(self, x_t, hidden):
        lstm_out, hidden = self.lstm(x_t, hidden)
        z = self.trunk(lstm_out.reshape(1, HIDDEN_SIZE))
        p = self.mdn_out(z).reshape(N_PLAYERS, K_MIX, 5)
        return p[..., 0], p[..., 1:3], p[..., 3:5].clamp(-6.0, 1.0), hidden


def mdn_nll(logits, mu, logsig, target, mask):
    t = target.unsqueeze(-2)
    inv_var = torch.exp(-2.0 * logsig)
    log_comp = (-0.5 * ((t - mu) ** 2 * inv_var).sum(-1)
                - logsig.sum(-1) - math.log(2 * math.pi))
    log_w   = torch.log_softmax(logits, dim=-1)
    log_mix = torch.logsumexp(log_w + log_comp, dim=-1)
    return -(log_mix.mean(-1) * mask).sum() / mask.sum().clamp(min=1.0)


# ===========================================================================
# TRAINING -- on existing defensive_sequences/ (no rebuild)
# ===========================================================================

def train_mdn_defensive_model(seq_dir="defensive_sequences", epochs=80,
                              save_path="defensive_response_mdn.pt"):
    x_files = sorted(glob.glob(os.path.join(seq_dir, "def_X_*.npy")))
    if not x_files:
        raise FileNotFoundError(f"No def_X_*.npy in {seq_dir}")

    model     = MDNDefensiveModel().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                     patience=3, factor=0.5)
    scale_t   = torch.tensor(DEF_INPUT_SCALE, dtype=torch.float32).to(DEVICE)

    best_loss, best_state = float("inf"), None
    print(f"Training MDN defensive model ({len(x_files)} chunks, "
          f"{epochs} epochs, device={DEVICE})...")
    print("NOTE: NLL values are not comparable to the old MSE numbers.\n")

    for epoch in range(epochs):
        epoch_loss, epoch_batches = 0.0, 0
        random.shuffle(x_files)

        for x_path in x_files:
            idx = x_path.split("_")[-1].replace(".npy", "")
            X = np.load(x_path)                                    # (N,S,27)
            y = np.load(os.path.join(seq_dir, f"def_y_{idx}.npy")) # (N,S,10)

            ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                               torch.tensor(y, dtype=torch.float32))
            dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

            model.train()
            for xb, yb in dl:
                xb = xb.to(DEVICE) * scale_t
                yb = yb.to(DEVICE)

                mask   = (xb.abs().sum(dim=-1) > 0).float()
                glitch = (yb.abs() >= 0.999).any(dim=-1).float()
                mask   = mask * (1.0 - glitch)

                target = yb.reshape(yb.shape[0], yb.shape[1], N_PLAYERS, 2)

                optimizer.zero_grad()
                logits, mu, logsig = model(xb)
                loss = mdn_nll(logits, mu, logsig, target, mask)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                epoch_batches += 1

            del X, y, ds, dl
            gc.collect()

        avg = epoch_loss / max(epoch_batches, 1)
        scheduler.step(avg)
        if avg < best_loss:
            best_loss = avg
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save(best_state, save_path)
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1}/{epochs}  NLL {avg:.4f}  best {best_loss:.4f}")

    model.load_state_dict(best_state)
    print(f"\nSaved {save_path}  (best NLL {best_loss:.4f})")
    return model


# ===========================================================================
# INTEGRATED ROLLOUT -- both teams move
# ===========================================================================

def _nearest_player_idx(bx, by, coords):
    return min(range(5), key=lambda j: (coords[j*2]-bx)**2 + (coords[j*2+1]-by)**2)

def _nearest_dist_to_point(px, py, coords):
    return min(((px-coords[j*2])**2 + (py-coords[j*2+1])**2) ** 0.5
               for j in range(5))

def _shot_zone(x, y):
    dist = ((x-BASKET_X)**2 + (y-BASKET_Y)**2) ** 0.5
    dy = abs(y-BASKET_Y)
    if dist <= 8.0 and dy <= 8.0: return "paint"
    if dist <= 15.0 and dy <= 6.0 and x > 13.0: return "free_throw"
    if x <= 14.0 and (y <= 8.0 or y >= 42.0): return "corner_3"
    if dist >= 23.75: return "above_break_3"
    return "mid_range"


def load_models(strategy_idx, models_dir="."):
    off_model = MDNMovementModel().to(DEVICE)
    off_model.load_state_dict(torch.load(
        os.path.join(models_dir, f"movement_generator_mdn_{strategy_idx}.pt"),
        map_location=DEVICE))
    off_model.eval()

    def_model = MDNDefensiveModel().to(DEVICE)
    def_model.load_state_dict(torch.load(
        os.path.join(models_dir, "defensive_response_mdn.pt"),
        map_location=DEVICE))
    def_model.eval()
    return off_model, def_model


def simulate_play_v5(
    strategy_idx, initial_positions, shot_scoring_model, le_zone, le_type,
    models_dir=".", max_steps=40, seed=None,
    off_temperature=0.7, def_temperature=0.5,
    score_prob_threshold=0.45, open_defender_threshold=6.0,
    shot_clock_threshold=4.0,
):
    """
    Full two-sided rollout, per step:
      1. offense MDN moves (sampled)
      2. ball tracks to the nearest offensive player (dynamic handler)
      3. defense MDN responds, SEEING THE NEW offensive positions (sampled)
      4. shot quality evaluated with POST-movement defender distances
    Both teams' positions are recorded live each step (no frozen values).
    """
    if seed is not None:
        torch.manual_seed(seed)

    off_model, def_model = load_models(strategy_idx, models_dir)
    off_scale = torch.tensor(OFF_INPUT_SCALE, dtype=torch.float32).to(DEVICE)
    def_scale = torch.tensor(DEF_INPUT_SCALE, dtype=torch.float32).to(DEVICE)

    off  = [c for pos in initial_positions["offense"] for c in pos]
    deff = [c for pos in initial_positions["defense"] for c in pos]
    ball_x, ball_y = initial_positions["ball"]

    steps = [{"offense": list(initial_positions["offense"]),
              "defense": list(initial_positions["defense"]),
              "ball":    initial_positions["ball"]}]
    score_probs = []
    off_hidden, def_hidden = None, None
    shot_step, shot_reason = None, None
    best_step, best_score  = None, -1.0

    with torch.no_grad():
        for step in range(max_steps):
            clock   = max(0.0, initial_positions.get("shot_clock", 24.0) - step*0.2)
            spacing = initial_positions.get("spacing_score", 20.0)
            overload= initial_positions.get("defensive_overload", False)
            screen  = float(initial_positions.get("screen_detected", False))
            pre_d   = _nearest_dist_to_point(ball_x, ball_y, deff)
            spatial = [spacing, screen, float(overload), pre_d, clock]

            # --- 1. offense moves ---
            off_in = off + deff + [ball_x, ball_y] + spatial
            x = torch.tensor([[off_in]], dtype=torch.float32).to(DEVICE) * off_scale
            logits, mu, logsig, off_hidden = off_model.step(x, off_hidden)
            d_off = sample_mdn(logits, mu, logsig,
                               temperature=off_temperature).cpu().numpy()
            new_off = []
            for j in range(5):
                nx = max(0.0, min(94.0, off[j*2]   + d_off[j,0] * MAX_DELTA))
                ny = max(0.0, min(50.0, off[j*2+1] + d_off[j,1] * MAX_DELTA))
                new_off.extend([nx, ny])
            off = new_off

            # --- 2. ball tracks handler ---
            h = _nearest_player_idx(ball_x, ball_y, off)
            ball_x, ball_y = off[h*2], off[h*2+1]

            # --- 3. defense responds to NEW offensive positions ---
            def_in = off + deff + [ball_x, ball_y] + spatial
            xd = torch.tensor([[def_in]], dtype=torch.float32).to(DEVICE) * def_scale
            dl_, dm_, ds_, def_hidden = def_model.step(xd, def_hidden)
            d_def = sample_mdn(dl_, dm_, ds_,
                               temperature=def_temperature).cpu().numpy()
            new_def = []
            for j in range(5):
                nx = max(0.0, min(94.0, deff[j*2]   + d_def[j,0] * MAX_DELTA))
                ny = max(0.0, min(50.0, deff[j*2+1] + d_def[j,1] * MAX_DELTA))
                new_def.extend([nx, ny])
            deff = new_def

            # --- 4. shot quality with POST-movement defender distances ---
            def_dist = _nearest_dist_to_point(ball_x, ball_y, deff)
            dist = ((ball_x-BASKET_X)**2 + (ball_y-BASKET_Y)**2) ** 0.5
            zone = _shot_zone(ball_x, ball_y)
            try:    zone_enc = le_zone.transform([zone])[0]
            except ValueError: zone_enc = 0
            try:    type_enc = le_type.transform(["jump_shot"])[0]
            except ValueError: type_enc = 0

            feats = pd.DataFrame([{
                "shot_x": ball_x, "shot_y": ball_y, "shot_distance": dist,
                "shot_zone_enc": zone_enc, "shot_type_enc": type_enc,
                "shot_clock": clock, "quarter": 1,
                "is_three_pointer": 1 if dist >= 23.75 else 0,
                "nearest_def_to_ball_handler": def_dist,
                "spacing_score": spacing, "defensive_overload": int(overload),
            }])
            p_score = float(shot_scoring_model.predict_proba(feats)[:, 1][0])
            score_probs.append(p_score)
            if p_score > best_score:
                best_score, best_step = p_score, step + 1

            steps.append({
                "offense": [(off[j*2],  off[j*2+1])  for j in range(5)],
                "defense": [(deff[j*2], deff[j*2+1]) for j in range(5)],
                "ball": (ball_x, ball_y), "p_score": p_score,
            })

            if p_score > score_prob_threshold:
                shot_step, shot_reason = step+1, f"high quality shot (P={p_score:.3f})"
                break
            if def_dist > open_defender_threshold:
                shot_step, shot_reason = step+1, f"wide open ({def_dist:.1f}ft)"
                break
            if clock < shot_clock_threshold:
                shot_step, shot_reason = step+1, f"forced ({clock:.1f}s)"
                break

    if shot_step is None:
        shot_step, shot_reason = best_step, f"fallback best (P={best_score:.3f})"
    return steps, score_probs, shot_step, shot_reason


def rank_strategies_v5(initial_positions, shot_scoring_model, le_zone, le_type,
                       models_dir=".", n_rollouts=5,
                       off_temperature=0.7, def_temperature=0.5,
                       max_steps=40):
    """Median-of-N ranking using the full two-sided simulation."""
    results = {}
    for k in range(6):
        ps, plays = [], []
        for r in range(n_rollouts):
            steps, probs, s_step, s_reason = simulate_play_v5(
                k, initial_positions, shot_scoring_model, le_zone, le_type,
                models_dir=models_dir, max_steps=max_steps,
                seed=1000*k + r,
                off_temperature=off_temperature,
                def_temperature=def_temperature)
            best = max(probs) if probs else 0.0
            ps.append(best)
            plays.append((best, steps, s_step, s_reason))
        median_p = float(np.median(ps))
        bp = max(plays, key=lambda t: t[0])
        results[k] = {"median_p": median_p, "rollout_ps": ps,
                      "best_play": bp[1], "shot_step": bp[2], "reason": bp[3]}
        print(f"{STRATEGY_NAMES[k]:<18} median P={median_p:.3f}  "
              f"(spread {min(ps):.3f}-{max(ps):.3f})")
    best_k = max(results, key=lambda k: results[k]["median_p"])
    print(f"\n-> Recommended: {STRATEGY_NAMES[best_k]} "
          f"(median P={results[best_k]['median_p']:.3f})")
    return results, best_k


if __name__ == "__main__":
    # Phase 1: train the defensive MDN on your existing sequences
    train_mdn_defensive_model(seq_dir="defensive_sequences", epochs=80)

    print("\nDone. Use simulate_play_v5 / rank_strategies_v5 for the full")
    print("two-sided rollout -- offense AND defense both move now.")