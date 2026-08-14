"""
NBA Pipeline - Step 12: Visualization  (v2)

Changes from v1:
    - COLORS SWAPPED: offense = BLUE circles, defense = RED circles (user request)
    - Both teams drawn as BIG circles (defense slightly bigger and drawn UNDER
      offense, so when a defender guards tightly you see a red ring behind the
      blue circle instead of one marker hiding the other)
    - Ball = mid-size ORANGE circle, always on top, with its own orange trail
      so even small movement is visible
    - NEW: print_play_diagnostics(result)  -> per-step table of raw model output
      (ball delta, player deltas) so you can tell MODEL problems from ANIMATION
      problems
    - NEW: create_movement_diagnostic(result) -> static PNG with start->end
      arrows + per-player total path length bar chart

Requires: matplotlib, pillow
"""

import numpy as np
# Force the non-interactive Agg backend BEFORE pyplot is imported. This file
# renders GIFs to disk and is called from FastAPI worker threads; the default
# interactive backend (TkAgg) can only run on the main thread and crashes with
# "main thread is not in main loop" / "Tcl_AsyncDelete: async handler deleted
# by the wrong thread" when driven off-thread. Agg has no GUI and is
# thread-safe. An MPLBACKEND env override is honored if someone wants a GUI.
import os
import matplotlib
if not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
import math


# ===========================================================================
# COLORS (single source of truth)
# ===========================================================================
OFF_COLOR   = "#3B7BFF"   # BLUE  = offense / attackers   (user request)
DEF_COLOR   = "#E53935"   # RED   = defense               (user request)
BALL_COLOR  = "#FF8C00"   # orange ball
TRAIL_OFF   = "#7FA8FF"
TRAIL_DEF   = "#FF9E9B"
TRAIL_BALL  = "#FFB84D"
SHOT_COLOR  = "#FFD700"   # gold shot arc

OFF_SIZE    = 480         # big circles
DEF_SIZE    = 620         # defense slightly bigger, drawn under offense
BALL_SIZE   = 200         # mid-size ball


# ===========================================================================
# INTERPOLATION
# ===========================================================================

def interpolate_play_for_animation(play_steps, fps=25, downsample_rate=5):
    """
    Interpolate pipeline output steps into smooth animation frames.
    Each step = downsample_rate raw frames (0.2s at 25fps, rate=5).
    """
    frames = []

    for i in range(len(play_steps) - 1):
        curr      = play_steps[i]
        next_step = play_steps[i + 1]

        for f in range(downsample_rate):
            t = f / downsample_rate

            off = [
                (curr["offense"][j][0]*(1-t) + next_step["offense"][j][0]*t,
                 curr["offense"][j][1]*(1-t) + next_step["offense"][j][1]*t)
                for j in range(5)
            ]
            dff = [
                (curr["defense"][j][0]*(1-t) + next_step["defense"][j][0]*t,
                 curr["defense"][j][1]*(1-t) + next_step["defense"][j][1]*t)
                for j in range(5)
            ]
            bx = curr["ball"][0]*(1-t) + next_step["ball"][0]*t
            by = curr["ball"][1]*(1-t) + next_step["ball"][1]*t

            p_score = curr.get("p_score") or 0.0
            # Interpolate the REAL per-step shot clock (now attached to each
            # step by plan_possession_v7 / simulate_play_v6) the same way
            # position is interpolated, rather than recomputing it from a
            # hardcoded 24.0 baseline minus elapsed frames -- that old formula
            # ignored the scenario's actual starting clock (19s half-court,
            # 22s fast break, or any custom value) and was wrong for every
            # animation except ones that happened to start at 24.0s.
            sc_curr    = curr.get("shot_clock", 24.0)
            sc_next    = next_step.get("shot_clock", sc_curr)
            shot_clock = max(0.0, sc_curr*(1-t) + sc_next*t)

            frames.append({
                "frame":      i*downsample_rate + f,
                "offense":    off,
                "defense":    dff,
                "ball":       (bx, by),
                "p_score":    p_score,
                "shot_clock": shot_clock,
                "is_shot":    False,
                "is_turnover": False,
            })

    last = play_steps[-1]
    # A play ending in a TURNOVER gets no shot arc -- the final frame is
    # flagged is_turnover instead and rendered as a steal (red X + label).
    ends_in_turnover = bool(last.get("turnover", False))
    frames.append({
        "frame":      len(frames),
        "offense":    last["offense"],
        "defense":    last["defense"],
        "ball":       last["ball"],
        "p_score":    last.get("p_score") or 0.0,
        "shot_clock": last.get("shot_clock", 0.0),
        "is_shot":    not ends_in_turnover,
        "is_turnover": ends_in_turnover,
    })

    print(f"Total frames:   {len(frames)}")
    print(f"Duration:       {len(frames)/fps:.1f} seconds at {fps}fps")
    return frames


# ===========================================================================
# COURT DRAWING
# ===========================================================================

def draw_half_court(ax):
    """Draw NBA half court (offense attacks LEFT basket at x=5.25)."""
    ax.set_facecolor("#F5DEB3")

    court = patches.Rectangle((0, 0), 47, 50, linewidth=2,
                              edgecolor="white", facecolor="#C8A96E", zorder=0)
    ax.add_patch(court)

    paint = patches.Rectangle((0, 17), 19, 16, linewidth=2,
                              edgecolor="white", facecolor="#C8A96E", zorder=1)
    ax.add_patch(paint)

    ax.add_patch(patches.Arc((19, 25), 12, 12, theta1=0, theta2=360,
                             color="white", linewidth=1.5, zorder=2))
    ax.add_patch(patches.Arc((5.25, 25), 47.5, 47.5, theta1=-68, theta2=68,
                             color="white", linewidth=2, zorder=2))

    ax.plot([0, 14], [3, 3],   color="white", linewidth=2, zorder=2)
    ax.plot([0, 14], [47, 47], color="white", linewidth=2, zorder=2)

    ax.add_patch(patches.Circle((5.25, 25), 0.75, fill=False,
                                edgecolor="white", linewidth=2, zorder=3))
    ax.plot([0, 0], [22.5, 27.5], color="white", linewidth=3, zorder=3)
    ax.plot([47, 47], [0, 50],   color="white", linewidth=2, zorder=2)
    ax.add_patch(patches.Arc((5.25, 25), 8, 8, theta1=-90, theta2=90,
                             color="white", linewidth=1, zorder=2))

    ax.set_xlim(-1, 50)
    ax.set_ylim(-1, 51)
    ax.set_aspect("equal")
    ax.axis("off")


# ===========================================================================
# SHOT ARC
# ===========================================================================

def compute_shot_arc(ball_x, ball_y, basket_x=5.25, basket_y=25.0, n_points=40):
    dist   = math.sqrt((ball_x-basket_x)**2 + (ball_y-basket_y)**2)
    height = dist * 0.35
    t_vals = np.linspace(0, 1, n_points)
    arc_x  = ball_x + (basket_x-ball_x)*t_vals
    arc_y  = ball_y + (basket_y-ball_y)*t_vals + height*4*t_vals*(1-t_vals)
    return arc_x, arc_y


# ===========================================================================
# MAIN ANIMATION
# ===========================================================================

def create_play_animation(
    result,
    strategy_name,
    output_path     = "play_animation.gif",
    fps             = 25,
    trail_length    = 20,
    downsample_rate = 5,
    playback_speed  = 0.5,
):
    """
    Animated GIF of the recommended play.

    playback_speed scales real-time playback (0.5 = half speed / slow-mo,
    1.0 = real time, 2.0 = double). It ONLY changes the GIF's written frame
    rate; the same frames are rendered, so motion is identical but plays back
    slower/faster. The physics interpolation and on-screen shot-clock math
    still use the true `fps`, so the clock countdown stays physically correct
    while the video itself plays in slow motion.
    Offense = big BLUE circles, Defense = big RED circles (drawn under offense),
    Ball = mid ORANGE circle with its own trail, always on top.

    Layout: the strategy TITLE sits at the TOP MIDDLE of the figure. If
    result["outcome"] is provided ({"made": bool, "points": int, "p": float,
    optional "turnover": bool}), the final frame shows an OUTCOME BANNER at
    the BOTTOM MIDDLE:
        made      -> "SHOT MADE (P=0.463): 3 points"   in green
        missed    -> "MISSED (P=0.463): 0 points"      in red
        turnover  -> "TURNOVER: 0 points"              in red
    Without result["outcome"] the banner is simply absent (backward
    compatible with older harnesses).
    """
    print(f"Creating animation for: {strategy_name}")

    frames   = interpolate_play_for_animation(result["play"], fps=fps,
                                              downsample_rate=downsample_rate)
    n_frames = len(frames)
    outcome  = result.get("outcome")     # optional {"made","points","p",...}

    fig = plt.figure(figsize=(10, 7), facecolor="#1a1a2e")
    ax_court = fig.add_axes([0.02, 0.12, 0.70, 0.82])
    ax_info  = fig.add_axes([0.73, 0.10, 0.25, 0.84])
    ax_info.set_facecolor("#1a1a2e")
    ax_info.axis("off")

    # persistent figure-level artists (animate() only clears the axes, so
    # these are created ONCE and mutated per frame; adding them inside
    # animate() would stack a new text every frame)
    title_txt = fig.text(
        0.5, 0.965, f"NBA Offensive Strategy Recommender  |  {strategy_name}",
        color="white", fontsize=11, fontweight="bold",
        ha="center", va="top")
    banner_txt = fig.text(
        0.5, 0.03, "", fontsize=14, fontweight="bold",
        ha="center", va="bottom")   # bottom middle of the figure

    p_scores = [f["p_score"] for f in frames if f["p_score"] is not None]
    p_max = max(p_scores) if p_scores else 0.6
    p_min = min(p_scores) if p_scores else 0.2

    def p_score_to_color(p):
        if p is None:
            return "white"
        t = max(0, min(1, (p - p_min) / max(p_max - p_min, 0.01)))
        if t < 0.5:
            return (1.0, t*2, 0.0)
        return (1.0-(t-0.5)*2, 1.0, 0.0)

    def animate(frame_idx):
        ax_court.clear()
        ax_info.clear()
        ax_info.set_facecolor("#1a1a2e")
        ax_info.axis("off")

        draw_half_court(ax_court)

        frame      = frames[frame_idx]
        offense    = frame["offense"]
        defense    = frame["defense"]
        ball       = frame["ball"]
        p_score    = frame["p_score"]
        shot_clock = frame["shot_clock"]
        is_shot    = frame["is_shot"]

        # -- Trails (players + BALL) --------------------------------------
        trail_start = max(0, frame_idx - trail_length)
        for trail_idx in range(trail_start, frame_idx):
            alpha = 0.15 + 0.55*(trail_idx-trail_start)/max(trail_length, 1)
            tf = frames[trail_idx]
            tn = frames[trail_idx+1]
            for j in range(5):
                ax_court.plot([tf["offense"][j][0], tn["offense"][j][0]],
                              [tf["offense"][j][1], tn["offense"][j][1]],
                              color=TRAIL_OFF, alpha=alpha,
                              linewidth=1.6, zorder=3)
                ax_court.plot([tf["defense"][j][0], tn["defense"][j][0]],
                              [tf["defense"][j][1], tn["defense"][j][1]],
                              color=TRAIL_DEF, alpha=alpha,
                              linewidth=1.6, zorder=3)
            # ball trail — thicker so small ball movement is visible
            ax_court.plot([tf["ball"][0], tn["ball"][0]],
                          [tf["ball"][1], tn["ball"][1]],
                          color=TRAIL_BALL, alpha=min(1.0, alpha+0.15),
                          linewidth=2.6, zorder=6)

        # -- Shot arc (final frame) ... or TURNOVER marker ------------------
        if is_shot:
            arc_x, arc_y = compute_shot_arc(ball[0], ball[1])
            ax_court.plot(arc_x, arc_y, color=SHOT_COLOR, linewidth=2.5,
                          linestyle="--", zorder=9, alpha=0.9)
            ax_court.scatter([arc_x[-1]], [arc_y[-1]], color=SHOT_COLOR,
                             s=90, zorder=9, marker="x")
        elif frame.get("is_turnover", False):
            # possession ends in a steal: big red X on the ball + label,
            # no shot arc (the ball never went up)
            ax_court.scatter([ball[0]], [ball[1]], color="#FF3333", s=700,
                             zorder=10, marker="x", linewidths=4)
            ax_court.text(ball[0], min(48.0, ball[1] + 3.2), "TURNOVER",
                          color="#FF3333", fontsize=13, fontweight="bold",
                          ha="center", va="bottom", zorder=10,
                          path_effects=None)

        # -- Players: DEFENSE first (bigger, RED), OFFENSE on top (BLUE) --
        # Defense bigger + lower zorder means a tightly-guarding defender
        # shows as a red ring behind the blue attacker instead of hiding it.
        for j, (dx, dy) in enumerate(defense):
            ax_court.scatter(dx, dy, color=DEF_COLOR, s=DEF_SIZE,
                             zorder=4, edgecolors="white", linewidths=1.8,
                             marker="o")
            ax_court.text(dx, dy - 2.4, str(j+1), color=DEF_COLOR,
                          fontsize=9, fontweight="bold",
                          ha="center", va="center", zorder=5)

        for j, (ox, oy) in enumerate(offense):
            ax_court.scatter(ox, oy, color=OFF_COLOR, s=OFF_SIZE,
                             zorder=5, edgecolors="white", linewidths=1.8,
                             marker="o")
            ax_court.text(ox, oy, str(j+1), color="white",
                          fontsize=9, fontweight="bold",
                          ha="center", va="center", zorder=6)

        # -- Ball: mid orange circle, ALWAYS on top -----------------------
        ax_court.scatter(ball[0], ball[1], color=BALL_COLOR, s=BALL_SIZE,
                         zorder=8, edgecolors="#663300", linewidths=1.8)

        # -- Info panel ----------------------------------------------------
        ax_info.set_xlim(0, 1)
        ax_info.set_ylim(0, 1)

        ax_info.text(0.5, 0.96, "STRATEGY", color="#AAAAAA", fontsize=9,
                     ha="center", va="top", transform=ax_info.transAxes)
        ax_info.text(0.5, 0.90, strategy_name, color="white", fontsize=11,
                     fontweight="bold", ha="center", va="top",
                     transform=ax_info.transAxes)
        ax_info.axhline(y=0.87, color="#444444", linewidth=1)

        sc_color = "#FF4444" if shot_clock < 5 else \
                   "#FFD700" if shot_clock < 10 else "white"
        ax_info.text(0.5, 0.82, "SHOT CLOCK", color="#AAAAAA", fontsize=9,
                     ha="center", va="top", transform=ax_info.transAxes)
        ax_info.text(0.5, 0.73, f"{shot_clock:.1f}s", color=sc_color,
                     fontsize=28, fontweight="bold", ha="center", va="top",
                     transform=ax_info.transAxes)
        ax_info.axhline(y=0.65, color="#444444", linewidth=1)

        p_color = p_score_to_color(p_score)
        ax_info.text(0.5, 0.61, "P(SCORE)", color="#AAAAAA", fontsize=9,
                     ha="center", va="top", transform=ax_info.transAxes)
        p_text = f"{p_score:.3f}" if p_score else "--"
        ax_info.text(0.5, 0.52, p_text, color=p_color, fontsize=28,
                     fontweight="bold", ha="center", va="top",
                     transform=ax_info.transAxes)

        bar_bg = patches.Rectangle((0.05, 0.43), 0.90, 0.04, color="#333333",
                                   transform=ax_info.transAxes, zorder=2)
        ax_info.add_patch(bar_bg)
        if p_score:
            bar_fill = patches.Rectangle(
                (0.05, 0.43),
                0.90*max(0, min(1, (p_score-0.3)/0.5)), 0.04,
                color=p_color, transform=ax_info.transAxes, zorder=3)
            ax_info.add_patch(bar_fill)
        ax_info.axhline(y=0.40, color="#444444", linewidth=1)

        step_num = frame_idx // downsample_rate
        ax_info.text(0.5, 0.36, "STEP", color="#AAAAAA", fontsize=9,
                     ha="center", va="top", transform=ax_info.transAxes)
        ax_info.text(0.5, 0.29, f"{step_num} / {len(result['play'])-1}",
                     color="white", fontsize=16, fontweight="bold",
                     ha="center", va="top", transform=ax_info.transAxes)
        ax_info.axhline(y=0.23, color="#444444", linewidth=1)

        ax_info.text(0.5, 0.19, "LEGEND", color="#AAAAAA", fontsize=9,
                     ha="center", va="top", transform=ax_info.transAxes)
        ax_info.plot([0.15], [0.12], "o", color=OFF_COLOR, ms=11,
                     transform=ax_info.transAxes, zorder=5,
                     markeredgecolor="white", markeredgewidth=1)
        ax_info.text(0.25, 0.12, "Offense (attack)", color="white",
                     fontsize=8, va="center", transform=ax_info.transAxes)
        ax_info.plot([0.15], [0.07], "o", color=DEF_COLOR, ms=11,
                     transform=ax_info.transAxes, zorder=5,
                     markeredgecolor="white", markeredgewidth=1)
        ax_info.text(0.25, 0.07, "Defense", color="white", fontsize=8,
                     va="center", transform=ax_info.transAxes)
        ax_info.plot([0.15], [0.02], "o", color=BALL_COLOR, ms=9,
                     transform=ax_info.transAxes, zorder=5,
                     markeredgecolor="#663300", markeredgewidth=1)
        ax_info.text(0.25, 0.02, "Ball", color="white", fontsize=8,
                     va="center", transform=ax_info.transAxes)

        if is_shot:
            ax_court.text(ball[0]+1.5, ball[1]+1.5,
                          f"SHOT!\nP={p_score:.3f}",
                          color=SHOT_COLOR, fontsize=10, fontweight="bold",
                          bbox=dict(boxstyle="round,pad=0.3",
                                    facecolor="#1a1a2e", alpha=0.8),
                          zorder=10)

        # ---- OUTCOME BANNER (final frame, bottom middle) ------------------
        # Title is a persistent fig.text at the TOP middle (set once above).
        if (is_shot or frame.get("is_turnover", False)) and outcome:
            if outcome.get("turnover", False):
                banner_txt.set_text("TURNOVER: 0 points")
                banner_txt.set_color("#FF4444")
            elif outcome.get("made"):
                p_txt = (f"P={outcome['p']:.3f}"
                         if outcome.get("p") is not None else "P=?")
                banner_txt.set_text(
                    f"SHOT MADE ({p_txt}): {outcome.get('points', 0)} points")
                banner_txt.set_color("#39FF6A")
            else:
                p_txt = (f"P={outcome['p']:.3f}"
                         if outcome.get("p") is not None else "P=?")
                banner_txt.set_text(f"MISSED ({p_txt}): 0 points")
                banner_txt.set_color("#FF4444")
        else:
            banner_txt.set_text("")

    # Slow-motion: write the GIF at a reduced frame rate. Same frames, but
    # each is shown longer, so the play runs back at `playback_speed` x real
    # time (0.5 -> half speed). Physics/clock math above still use true fps.
    out_fps = max(1.0, fps * playback_speed)

    print(f"Rendering {n_frames} frames... "
          f"(playback {playback_speed:.2f}x -> {out_fps:.0f} gif fps, "
          f"{n_frames/out_fps:.1f}s runtime)")
    anim = FuncAnimation(fig, animate, frames=n_frames,
                         interval=1000/out_fps, repeat=False)
    writer = PillowWriter(fps=out_fps)
    try:
        anim.save(output_path, writer=writer)
    finally:
        # always release the figure, even if saving raised -- a leaked
        # figure keeps backend state alive and can wedge later renders.
        plt.close(fig)

    print(f"Saved: {output_path}  ({os.path.getsize(output_path)/1024:.0f} KB)")
    return output_path


# ===========================================================================
# DIAGNOSTICS -- "is it the model or the animation?"
# ===========================================================================

def print_play_diagnostics(result, strategy_name=""):
    """
    Print raw per-step model output so you can see EXACTLY how much the
    model is moving players/ball, independent of any animation rendering.

    If numbers here are tiny (< ~0.5 ft/step), the MODEL is producing
    near-zero deltas and no animation change will make it look dynamic.
    """
    play = result["play"]
    n    = len(play)
    print("=" * 78)
    print(f"RAW PIPELINE OUTPUT DIAGNOSTICS  {strategy_name}  ({n} steps)")
    print("=" * 78)
    print(f"{'step':>4} | {'ball (x,y)':>14} | {'ball d':>7} | "
          f"{'off mean d':>10} | {'off max d':>9} | {'def mean d':>10} | "
          f"{'P(score)':>8}")
    print("-" * 78)

    for i in range(n):
        s  = play[i]
        bx, by = s["ball"]
        if i == 0:
            bd = om = ox = dm = 0.0
        else:
            p = play[i-1]
            bd = math.dist(s["ball"], p["ball"])
            od = [math.dist(s["offense"][j], p["offense"][j]) for j in range(5)]
            dd = [math.dist(s["defense"][j], p["defense"][j]) for j in range(5)]
            om, ox, dm = float(np.mean(od)), float(np.max(od)), float(np.mean(dd))
        ps = s.get("p_score")
        ps_txt = f"{ps:.3f}" if ps is not None else "--"
        print(f"{i:>4} | ({bx:5.1f},{by:5.1f}) | {bd:7.2f} | "
              f"{om:10.2f} | {ox:9.2f} | {dm:10.2f} | {ps_txt:>8}")

    # Totals: net displacement + path length per player
    print("-" * 78)
    print("TOTALS over play (feet):")
    first, last = play[0], play[-1]
    ball_net  = math.dist(first["ball"], last["ball"])
    ball_path = sum(math.dist(play[i]["ball"], play[i-1]["ball"])
                    for i in range(1, n))
    print(f"  Ball:  net {ball_net:6.1f}   path {ball_path:6.1f}")
    for team, key, tag in [("OFF", "offense", "O"), ("DEF", "defense", "D")]:
        for j in range(5):
            net  = math.dist(first[key][j], last[key][j])
            path = sum(math.dist(play[i][key][j], play[i-1][key][j])
                       for i in range(1, n))
            print(f"  {tag}{j+1}:   net {net:6.1f}   path {path:6.1f}")

    print("-" * 78)
    print("INTERPRETATION GUIDE:")
    print("  Real NBA possession: players cover ~15-25+ ft, ball much more.")
    print("  If path lengths above are < ~3 ft, the movement model is")
    print("  producing near-zero deltas -> MODEL problem, not animation.")
    print("=" * 78)


def create_movement_diagnostic(result, strategy_name="",
                               output_path="movement_diagnostic.png"):
    """
    Static PNG: (left) court with start->end arrows per player + ball,
    (right) bar chart of total path length per player.
    Makes model movement magnitude obvious at a glance.
    """
    play  = result["play"]
    n     = len(play)
    first, last = play[0], play[-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                   facecolor="#1a1a2e",
                                   gridspec_kw={"width_ratios": [1.3, 1]})
    fig.suptitle(f"Movement Diagnostic  |  {strategy_name}",
                 color="white", fontsize=13, fontweight="bold")

    # ---- Left: arrows on court ----
    draw_half_court(ax1)
    for j in range(5):
        x0, y0 = first["offense"][j]; x1, y1 = last["offense"][j]
        ax1.scatter(x0, y0, s=220, facecolors="none", edgecolors=OFF_COLOR,
                    linewidths=2, zorder=5)
        ax1.scatter(x1, y1, s=220, color=OFF_COLOR, edgecolors="white",
                    linewidths=1.2, zorder=5)
        ax1.annotate("", xy=(x1, y1), xytext=(x0, y0),
                     arrowprops=dict(arrowstyle="->", color=OFF_COLOR,
                                     lw=1.8), zorder=4)
        ax1.text(x1, y1, str(j+1), color="white", fontsize=7,
                 fontweight="bold", ha="center", va="center", zorder=6)

        x0, y0 = first["defense"][j]; x1, y1 = last["defense"][j]
        ax1.scatter(x0, y0, s=220, facecolors="none", edgecolors=DEF_COLOR,
                    linewidths=2, zorder=5)
        ax1.scatter(x1, y1, s=220, color=DEF_COLOR, edgecolors="white",
                    linewidths=1.2, zorder=5)
        ax1.annotate("", xy=(x1, y1), xytext=(x0, y0),
                     arrowprops=dict(arrowstyle="->", color=DEF_COLOR,
                                     lw=1.8), zorder=4)
        ax1.text(x1, y1, str(j+1), color="white", fontsize=7,
                 fontweight="bold", ha="center", va="center", zorder=6)

    bx0, by0 = first["ball"]; bx1, by1 = last["ball"]
    ax1.scatter(bx0, by0, s=140, facecolors="none", edgecolors=BALL_COLOR,
                linewidths=2, zorder=7)
    ax1.scatter(bx1, by1, s=140, color=BALL_COLOR, edgecolors="#663300",
                linewidths=1.5, zorder=7)
    ax1.annotate("", xy=(bx1, by1), xytext=(bx0, by0),
                 arrowprops=dict(arrowstyle="->", color=BALL_COLOR, lw=2.2),
                 zorder=6)
    ax1.set_title("Start (hollow) -> End (filled)", color="white", fontsize=10)

    # ---- Right: path length bars ----
    labels, paths, colors = [], [], []
    ball_path = sum(math.dist(play[i]["ball"], play[i-1]["ball"])
                    for i in range(1, n))
    labels.append("Ball"); paths.append(ball_path); colors.append(BALL_COLOR)
    for j in range(5):
        labels.append(f"O{j+1}")
        paths.append(sum(math.dist(play[i]["offense"][j],
                                   play[i-1]["offense"][j])
                         for i in range(1, n)))
        colors.append(OFF_COLOR)
    for j in range(5):
        labels.append(f"D{j+1}")
        paths.append(sum(math.dist(play[i]["defense"][j],
                                   play[i-1]["defense"][j])
                         for i in range(1, n)))
        colors.append(DEF_COLOR)

    ax2.set_facecolor("#1a1a2e")
    bars = ax2.barh(labels[::-1], paths[::-1], color=colors[::-1],
                    edgecolor="white", linewidth=0.5, height=0.65)
    ax2.axvline(x=15, color="#FFD700", linestyle="--", linewidth=1.2)
    ax2.text(15.3, len(labels)-0.5, "realistic NBA\nmovement (~15ft+)",
             color="#FFD700", fontsize=8, va="top")
    ax2.set_xlabel("Total path length over play (feet)", color="white",
                   fontsize=10)
    ax2.set_title("How far did the model actually move each entity?",
                  color="white", fontsize=10)
    ax2.tick_params(colors="white")
    for spine in ["top", "right"]:
        ax2.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax2.spines[spine].set_color("#444444")
    for bar, p in zip(bars, paths[::-1]):
        ax2.text(bar.get_width()+0.2, bar.get_y()+bar.get_height()/2,
                 f"{p:.1f}", color="white", va="center", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_path, dpi=150, facecolor="#1a1a2e",
                bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")
    return output_path


# ===========================================================================
# COMPARISON CHART (unchanged logic, colors follow new scheme)
# ===========================================================================

def create_strategy_comparison_chart(result, output_path="strategy_comparison.png"):
    strategies = result["all_strategies"]
    names      = [s["name"] for s in strategies]
    scores     = [s["score_prob"] for s in strategies]
    steps      = [s["shot_at_step"] for s in strategies]
    best_name  = result["recommended_strategy"]

    colors = ["#FFD700" if n == best_name else OFF_COLOR for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="#1a1a2e")
    fig.suptitle("Strategy Comparison", color="white",
                 fontsize=14, fontweight="bold")

    ax1 = axes[0]
    ax1.set_facecolor("#1a1a2e")
    bars = ax1.barh(names, scores, color=colors, edgecolor="white",
                    linewidth=0.5, height=0.6)
    ax1.set_xlabel("P(score)", color="white", fontsize=10)
    ax1.set_title("Scoring Probability", color="white", fontsize=11)
    ax1.tick_params(colors="white")
    ax1.spines["bottom"].set_color("#444444")
    ax1.spines["left"].set_color("#444444")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set_xlim(0, max(scores) * 1.2)
    for bar, score in zip(bars, scores):
        ax1.text(score+0.005, bar.get_y()+bar.get_height()/2,
                 f"{score:.3f}", color="white", va="center",
                 fontsize=9, fontweight="bold")

    ax2 = axes[1]
    ax2.set_facecolor("#1a1a2e")
    bars2 = ax2.barh(names, steps, color=colors, edgecolor="white",
                     linewidth=0.5, height=0.6)
    ax2.set_xlabel("Step", color="white", fontsize=10)
    ax2.set_title("Shot Taken at Step", color="white", fontsize=11)
    ax2.tick_params(colors="white")
    ax2.spines["bottom"].set_color("#444444")
    ax2.spines["left"].set_color("#444444")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    for bar, step in zip(bars2, steps):
        ax2.text(step+0.3, bar.get_y()+bar.get_height()/2,
                 str(step), color="white", va="center",
                 fontsize=9, fontweight="bold")

    gold_patch = plt.Rectangle((0, 0), 1, 1, color="#FFD700")
    blue_patch = plt.Rectangle((0, 0), 1, 1, color=OFF_COLOR)
    fig.legend([gold_patch, blue_patch], ["Best strategy", "Other strategies"],
               loc="lower center", ncol=2, facecolor="#1a1a2e",
               labelcolor="white", framealpha=0, fontsize=9)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(output_path, dpi=150, facecolor="#1a1a2e",
                bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")
    return output_path


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    # Usage after running step11_pipeline.py:
    #
    #   from step11_pipeline import load_all_models, run_full_pipeline
    #   from step12_visualization import (create_play_animation,
    #                                     create_strategy_comparison_chart,
    #                                     print_play_diagnostics,
    #                                     create_movement_diagnostic)
    #
    #   models = load_all_models()
    #   result = run_full_pipeline(input_data, models)
    #
    #   # 1) FIRST check raw model output (model vs animation question):
    #   print_play_diagnostics(result, result["recommended_strategy"])
    #   create_movement_diagnostic(result, result["recommended_strategy"])
    #
    #   # 2) Then render the GIF:
    #   create_play_animation(result, result["recommended_strategy"],
    #                         "play_animation.gif")
    #   create_strategy_comparison_chart(result)
    print("Step 12 Visualization v2 loaded.")
    print("Offense=BLUE circles, Defense=RED circles, Ball=orange w/ trail.")
    print("Run print_play_diagnostics(result) to check raw model movement.")