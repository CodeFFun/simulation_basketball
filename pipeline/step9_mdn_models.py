"""
step9_mdn_models.py

FIX 6 -- Mixture Density Network (MDN) movement head, replacing the
deterministic MSE head in the per-strategy models.

WHY: a deterministic model trained with MSE must output ONE vector per
player per step. Real player movement is MULTIMODAL -- from the same
configuration a corner shooter sometimes lifts to the wing, sometimes cuts
baseline, sometimes holds. The average of those options is a small vector
pointing at the basket (the only direction that never cancels), which is
exactly the parallel straight-line drift you observed. Splitting into
per-strategy models fixed WHICH distribution each model learns; the MDN
fixes the model outputting that distribution's MEAN instead of a sample
from it.

WHAT: each player's next (dx, dy) is modeled as a K-component 2D Gaussian
mixture (diagonal covariance). Training minimizes masked negative
log-likelihood. Inference SAMPLES: pick a component (weighted), then sample
the Gaussian, scaled by a temperature knob:
    temperature = 0.0  -> always take the most likely component's mean
                          (deterministic but mode-seeking, NOT the average:
                          this alone already commits to one option)
    temperature = 1.0  -> full sampling (most lifelike, stochastic)

USAGE: run AFTER step9_per_strategy_models.py's split step (reuses
movement_sequences_per_strategy/). Trains movement_generator_mdn_{k}.pt.
Rollouts are stochastic -- for strategy RANKING, run N rollouts per
strategy and use the median best-P(score) (see rank_strategies()).
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

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE   = 27      # no one-hot: one model per strategy
HIDDEN_SIZE  = 128
NUM_LAYERS   = 2
BATCH_SIZE   = 64
LEARNING_RATE= 0.001
DROPOUT_RATE = 0.3
MAX_DELTA    = 10.0
N_PLAYERS    = 5
K_MIX        = 4       # mixture components per player
BASKET_X, BASKET_Y = 5.25, 25.0

STRATEGY_NAMES = {
    0: "Motion Offense", 1: "Post Up", 2: "Free Throw Heavy",
    3: "Isolation", 4: "Pick and Roll", 5: "Fast Break",
}

def build_input_scale():
    s = np.ones(INPUT_SIZE, dtype=np.float32)
    for j in range(0, 22, 2):
        s[j] = 1.0 / 47.0
    for j in range(1, 22, 2):
        s[j] = 1.0 / 50.0
    s[22] = 1.0 / 50.0
    s[25] = 1.0 / 47.0
    s[26] = 1.0 / 24.0
    return s

INPUT_SCALE = build_input_scale()


# ===========================================================================
# MODEL
# ===========================================================================

class MDNMovementModel(nn.Module):
    """
    LSTM backbone -> per-player 2D Gaussian mixture over the next delta.
    Output params per player: K logits, K*(mu_x, mu_y), K*(logsig_x, logsig_y)
    """

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS, batch_first=True, dropout=DROPOUT_RATE,
        )
        self.trunk = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(128, 128),         nn.LayerNorm(128), nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
        )
        # per player: K logits + K*2 means + K*2 logsigmas = K*5
        self.mdn_out = nn.Linear(128, N_PLAYERS * K_MIX * 5)

    def forward(self, x):
        """
        x: (B, S, INPUT_SIZE) already scaled
        returns logits (B,S,P,K), mu (B,S,P,K,2), logsig (B,S,P,K,2)
        """
        lstm_out, _ = self.lstm(x)
        b, s, h = lstm_out.shape
        z = self.trunk(lstm_out.reshape(b * s, h))
        p = self.mdn_out(z).reshape(b, s, N_PLAYERS, K_MIX, 5)
        logits = p[..., 0]
        mu     = p[..., 1:3]
        logsig = p[..., 3:5].clamp(-6.0, 1.0)   # sigma in [~0.0025, ~2.7] normalized
        return logits, mu, logsig

    def step(self, x_t, hidden):
        """Single-step inference. x_t: (1,1,INPUT_SIZE) scaled."""
        lstm_out, hidden = self.lstm(x_t, hidden)
        z = self.trunk(lstm_out.reshape(1, HIDDEN_SIZE))
        p = self.mdn_out(z).reshape(N_PLAYERS, K_MIX, 5)
        return p[..., 0], p[..., 1:3], p[..., 3:5].clamp(-6.0, 1.0), hidden


def mdn_nll(logits, mu, logsig, target, mask):
    """
    Masked NLL of target deltas under the mixture.
    target: (B,S,P,2) normalized deltas; mask: (B,S)
    """
    t = target.unsqueeze(-2)                          # (B,S,P,1,2)
    inv_var = torch.exp(-2.0 * logsig)
    log_comp = (
        -0.5 * ((t - mu) ** 2 * inv_var).sum(-1)      # (B,S,P,K)
        - logsig.sum(-1)
        - math.log(2 * math.pi)
    )
    log_w   = torch.log_softmax(logits, dim=-1)
    log_mix = torch.logsumexp(log_w + log_comp, dim=-1)   # (B,S,P)
    nll = -(log_mix.mean(-1) * mask).sum() / mask.sum().clamp(min=1.0)
    return nll


def sample_mdn(logits, mu, logsig, temperature=1.0):
    """
    Sample one delta per player. logits (P,K), mu (P,K,2), logsig (P,K,2).
    temperature=0 -> most likely component's mean (mode-seeking commit).
    Returns (P,2) normalized deltas.
    """
    w = torch.softmax(logits, dim=-1)                 # (P,K)
    if temperature <= 0.0:
        k = w.argmax(dim=-1)                          # (P,)
        chosen_mu = mu[torch.arange(N_PLAYERS), k]    # (P,2)
        return chosen_mu
    k = torch.multinomial(w, 1).squeeze(-1)           # (P,)
    chosen_mu  = mu[torch.arange(N_PLAYERS), k]
    chosen_sig = torch.exp(logsig[torch.arange(N_PLAYERS), k])
    eps = torch.randn_like(chosen_mu)
    return chosen_mu + temperature * chosen_sig * eps


# ===========================================================================
# TRAINING (reuses the per-strategy split directories)
# ===========================================================================

def train_mdn_strategy_model(strategy_idx,
                             seq_dir="movement_sequences_per_strategy",
                             epochs=80, save_path=None):
    save_path = save_path or f"movement_generator_mdn_{strategy_idx}.pt"
    strat_dir = os.path.join(seq_dir, f"strategy_{strategy_idx}")
    x_files   = sorted(glob.glob(os.path.join(strat_dir, "X_*.npy")))
    if not x_files:
        print(f"  No data for {STRATEGY_NAMES[strategy_idx]} -- skipping.")
        return None

    model     = MDNMovementModel().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                     patience=3, factor=0.5)
    scale_t   = torch.tensor(INPUT_SCALE, dtype=torch.float32).to(DEVICE)

    best_loss, best_state = float("inf"), None
    print(f"\nTraining MDN {STRATEGY_NAMES[strategy_idx]} ({len(x_files)} chunks)...")

    for epoch in range(epochs):
        epoch_loss, epoch_batches = 0.0, 0
        random.shuffle(x_files)

        for x_path in x_files:
            idx = x_path.split("_")[-1].replace(".npy", "")
            X = np.load(x_path)
            y = np.load(os.path.join(strat_dir, f"y_{idx}.npy"))
            # weights kept for API parity; MDN NLL already averages per step
            ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                               torch.tensor(y, dtype=torch.float32))
            dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

            model.train()
            for xb, yb in dl:
                xb = xb.to(DEVICE) * scale_t
                yb = yb.to(DEVICE)

                mask   = (xb.abs().sum(dim=-1) > 0).float()
                glitch = (yb[:, :, :10].abs() >= 0.999).any(dim=-1).float()
                mask   = mask * (1.0 - glitch)

                target = yb[:, :, :10].reshape(yb.shape[0], yb.shape[1],
                                               N_PLAYERS, 2)

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
            print(f"  [{STRATEGY_NAMES[strategy_idx]}] epoch {epoch+1}/{epochs}  "
                  f"NLL {avg:.4f}  best {best_loss:.4f}")

    model.load_state_dict(best_state)
    print(f"  Saved {save_path}  (best NLL {best_loss:.4f})")
    return model


def train_all_mdn_models(seq_dir="movement_sequences_per_strategy", epochs=80):
    return {k: train_mdn_strategy_model(k, seq_dir, epochs=epochs)
            for k in range(6)}


# ===========================================================================
# INFERENCE
# ===========================================================================

def find_nearest_player_idx(bx, by, off):
    return min(range(5), key=lambda j: (off[j*2]-bx)**2 + (off[j*2+1]-by)**2)

def nearest_defender_to_point(px, py, deff):
    return min(((px-deff[j*2])**2 + (py-deff[j*2+1])**2) ** 0.5 for j in range(5))

def get_shot_zone_label(x, y):
    dist = ((x-BASKET_X)**2 + (y-BASKET_Y)**2) ** 0.5
    dy = abs(y-BASKET_Y)
    if dist <= 8.0 and dy <= 8.0: return "paint"
    if dist <= 15.0 and dy <= 6.0 and x > 13.0: return "free_throw"
    if x <= 14.0 and (y <= 8.0 or y >= 42.0): return "corner_3"
    if dist >= 23.75: return "above_break_3"
    return "mid_range"


def play_out_strategy_v4(
    strategy_idx, initial_positions, shot_scoring_model, le_zone, le_type,
    models_dir=".", max_steps=40, temperature=0.7, seed=None,
    score_prob_threshold=0.45, open_defender_threshold=6.0,
    shot_clock_threshold=4.0,
):
    """
    MDN rollout: at each step, SAMPLE each player's next move from the
    mixture (temperature-scaled) instead of taking the collapsed mean.
    Stochastic -- set seed for reproducibility, or run several rollouts
    and aggregate (see rank_strategies).
    """
    if seed is not None:
        torch.manual_seed(seed)

    model = MDNMovementModel().to(DEVICE)
    model.load_state_dict(torch.load(
        os.path.join(models_dir, f"movement_generator_mdn_{strategy_idx}.pt"),
        map_location=DEVICE))
    model.eval()

    scale_t = torch.tensor(INPUT_SCALE, dtype=torch.float32).to(DEVICE)

    off = [c for pos in initial_positions["offense"] for c in pos]
    deff = [c for pos in initial_positions["defense"] for c in pos]
    ball_x, ball_y = initial_positions["ball"]

    steps = [{"offense": list(initial_positions["offense"]),
              "defense": list(initial_positions["defense"]),
              "ball":    initial_positions["ball"]}]
    score_probs = []
    hidden = None
    shot_step, shot_reason = None, None
    best_step, best_score = None, -1.0

    with torch.no_grad():
        for step in range(max_steps):
            clock   = max(0.0, initial_positions.get("shot_clock", 24.0) - step*0.2)
            spacing = initial_positions.get("spacing_score", 20.0)
            overload= initial_positions.get("defensive_overload", False)
            pre_d   = nearest_defender_to_point(ball_x, ball_y, deff)

            spatial   = [spacing, float(initial_positions.get("screen_detected", False)),
                         float(overload), pre_d, clock]
            input_vec = off + deff + [ball_x, ball_y] + spatial
            x = (torch.tensor([[input_vec]], dtype=torch.float32).to(DEVICE)
                 * scale_t)

            logits, mu, logsig, hidden = model.step(x, hidden)
            deltas = sample_mdn(logits, mu, logsig,
                                temperature=temperature).cpu().numpy()  # (5,2)

            new_off = []
            for j in range(5):
                nx = max(0.0, min(94.0, off[j*2]   + deltas[j, 0] * MAX_DELTA))
                ny = max(0.0, min(50.0, off[j*2+1] + deltas[j, 1] * MAX_DELTA))
                new_off.extend([nx, ny])
            off = new_off

            handler = find_nearest_player_idx(ball_x, ball_y, off)
            ball_x, ball_y = off[handler*2], off[handler*2+1]
            def_dist = nearest_defender_to_point(ball_x, ball_y, deff)

            dist = ((ball_x-BASKET_X)**2 + (ball_y-BASKET_Y)**2) ** 0.5
            zone = get_shot_zone_label(ball_x, ball_y)
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
                "offense": [(off[j*2], off[j*2+1]) for j in range(5)],
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


def rank_strategies(initial_positions, shot_scoring_model, le_zone, le_type,
                    models_dir=".", n_rollouts=5, temperature=0.7,
                    max_steps=40):
    """
    Because MDN rollouts are stochastic, rank strategies by the MEDIAN of
    n_rollouts best-P(score) values. Returns dict {k: median_p, best_play}.
    """
    results = {}
    for k in range(6):
        ps, plays = [], []
        for r in range(n_rollouts):
            steps, probs, s_step, s_reason = play_out_strategy_v4(
                k, initial_positions, shot_scoring_model, le_zone, le_type,
                models_dir=models_dir, max_steps=max_steps,
                temperature=temperature, seed=1000*k + r)
            best = max(probs) if probs else 0.0
            ps.append(best)
            plays.append((best, steps, s_step, s_reason))
        median_p = float(np.median(ps))
        best_play = max(plays, key=lambda t: t[0])
        results[k] = {"median_p": median_p, "rollout_ps": ps,
                      "best_play": best_play[1],
                      "shot_step": best_play[2], "reason": best_play[3]}
        print(f"{STRATEGY_NAMES[k]:<18} median P(score) over {n_rollouts} "
              f"rollouts: {median_p:.3f}  (spread {min(ps):.3f}-{max(ps):.3f})")
    best_k = max(results, key=lambda k: results[k]["median_p"])
    print(f"\n-> Recommended: {STRATEGY_NAMES[best_k]} "
          f"(median P={results[best_k]['median_p']:.3f})")
    return results, best_k


if __name__ == "__main__":
    # 1. Requires the split from step9_per_strategy_models.py already done.
    train_all_mdn_models(epochs=80)
    print("\nDone. Use play_out_strategy_v4 / rank_strategies for inference.")