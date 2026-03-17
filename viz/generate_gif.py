"""Generate an animated GIF of the circular track environment."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as mtransforms
import numpy as np
from PIL import Image
import io
import torch

from src.env import CircularTrack

TRAIN_COLORS = ["#2E7D32", "#E65100", "#1565C0", "#6A1B9A", "#C62828"]
TRAIN_BODY_LIGHT = ["#43A047", "#FF8F00", "#1E88E5", "#8E24AA", "#E53935"]


def _cell_angle(i: int, num_cells: int) -> float:
    """Angle for cell i (0 at top, clockwise)."""
    return -np.pi / 2 + 2 * np.pi * i / num_cells


def _draw_train(ax, cx, cy, tangent_deg, body_color, roof_color):
    """Draw a small train car oriented along the track."""
    w, h = 0.22, 0.10

    # Body
    body = patches.FancyBboxPatch(
        (-w / 2, -h / 2), w, h,
        boxstyle=patches.BoxStyle.Round(pad=0.02),
        facecolor=body_color, edgecolor="#263238", linewidth=0.8, zorder=5,
    )
    t = (mtransforms.Affine2D()
         .rotate_deg(tangent_deg)
         .translate(cx, cy)
         + ax.transData)
    body.set_transform(t)
    ax.add_patch(body)

    # Roof stripe
    roof = patches.FancyBboxPatch(
        (-w / 2 + 0.02, -h / 2 + 0.01), w - 0.04, h * 0.35,
        boxstyle=patches.BoxStyle.Round(pad=0.01),
        facecolor=roof_color, edgecolor="none", zorder=6,
    )
    roof.set_transform(t)
    ax.add_patch(roof)

    # Windows
    for wx_off in [-0.055, 0.0, 0.055]:
        win = patches.FancyBboxPatch(
            (wx_off - 0.018, 0.0), 0.036, h * 0.32,
            boxstyle=patches.BoxStyle.Round(pad=0.005),
            facecolor="#E3F2FD", edgecolor="#90CAF9", linewidth=0.4, zorder=7,
        )
        win.set_transform(t)
        ax.add_patch(win)

    # Headlight (front)
    for side, color in [(w / 2 - 0.01, "#FFF176"), (-w / 2 + 0.01, "#EF5350")]:
        light = plt.Circle((0, 0), 0.012, color=color, zorder=7)
        lt = (mtransforms.Affine2D()
              .translate(side, 0)
              .rotate_deg(tangent_deg)
              .translate(cx, cy)
              + ax.transData)
        light.set_transform(lt)
        ax.add_patch(light)



# Muted versions of train colors for the space-time diagram
SPACETIME_COLORS = ["#6A9E6C", "#C49560", "#7BAAC8", "#A882B5", "#C48A8A"]


def _hex_to_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def _track_train_ids(history, num_cells):
    """Assign persistent train IDs across timesteps by tracking movement.

    Two-pass approach:
      1. First assign IDs to trains that stayed in the same cell (blocked/stalled).
      2. Then assign IDs to trains that moved forward by one cell.
    This prevents a moving train from stealing the ID of a stationary one.
    """
    id_maps = []
    first = {}
    tid = 0
    for c in range(num_cells):
        if history[0][c] == 1:
            first[c] = tid
            tid += 1
    id_maps.append(first)

    for t in range(1, len(history)):
        prev = id_maps[t - 1]
        curr = {}
        used_ids = set()

        # Pass 1: trains that stayed in place (blocked or stalled)
        for old_cell, train_id in sorted(prev.items()):
            if history[t][old_cell] == 1 and old_cell not in curr:
                next_cell = (old_cell + 1) % num_cells
                # A train stayed if: old_cell still occupied AND either
                # (a) next_cell is empty in t (train didn't advance), or
                # (b) next_cell was already occupied at t-1 (someone else was there)
                stayed = (
                    history[t][next_cell] == 0 or
                    history[t - 1][next_cell] == 1
                )
                if stayed:
                    curr[old_cell] = train_id
                    used_ids.add(train_id)

        # Pass 2: trains that moved forward by 1
        for old_cell, train_id in sorted(prev.items()):
            if train_id in used_ids:
                continue
            next_cell = (old_cell + 1) % num_cells
            if history[t][next_cell] == 1 and next_cell not in curr:
                curr[next_cell] = train_id
                used_ids.add(train_id)

        # Fallback for any unmatched
        for c in range(num_cells):
            if history[t][c] == 1 and c not in curr:
                for fid in range(tid):
                    if fid not in used_ids:
                        curr[c] = fid
                        used_ids.add(fid)
                        break
        id_maps.append(curr)
    return id_maps


def _render_track(ax, state, step, num_cells, id_map=None, stall_cell=None):
    """Draw the circular track on the given axes."""
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.0, 2.0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("#F5F5F0")

    radius = 1.35
    rail_w = 0.07

    # Ballast
    ax.add_patch(plt.Circle((0, 0), radius + 0.14, color="#E0DDD4", zorder=0))
    ax.add_patch(plt.Circle((0, 0), radius - 0.14, color="#F5F5F0", zorder=0))

    # Sleepers
    for i in range(60):
        angle = 2 * np.pi * i / 60
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        ax.plot([(radius - 0.11) * cos_a, (radius + 0.11) * cos_a],
                [(radius - 0.11) * sin_a, (radius + 0.11) * sin_a],
                color="#8D6E63", linewidth=2.0, solid_capstyle="round", zorder=1)

    # Rails
    for offset in [-rail_w, rail_w]:
        ax.add_patch(plt.Circle((0, 0), radius + offset, fill=False,
                                color="#616161", linewidth=1.8, zorder=2))

    # Cell markers
    for i in range(num_cells):
        angle = _cell_angle(i, num_cells)
        x, y = radius * np.cos(angle), radius * np.sin(angle)
        ax.add_patch(plt.Circle((x, y), 0.015, color="#9E9E9E", alpha=0.5, zorder=3))

    # Trains
    for i in range(num_cells):
        if state[i] != 1:
            continue
        angle = _cell_angle(i, num_cells)
        cx, cy = radius * np.cos(angle), radius * np.sin(angle)
        tangent_deg = np.degrees(angle + np.pi / 2)
        if id_map is not None and i in id_map:
            color_idx = id_map[i] % len(TRAIN_COLORS)
        else:
            color_idx = 0
        _draw_train(ax, cx, cy, tangent_deg,
                    TRAIN_COLORS[color_idx], TRAIN_BODY_LIGHT[color_idx])

    ax.text(0, 0.12, f"t = {step}", ha="center", va="center",
            fontsize=15, color="#37474F", fontweight="bold", fontfamily="sans-serif")

    # Stall indicator: warning sign at the stalled cell
    if stall_cell is not None:
        angle = _cell_angle(stall_cell, num_cells)
        # Place it slightly outside the track
        r_icon = radius + 0.32
        ix, iy = r_icon * np.cos(angle), r_icon * np.sin(angle)
        ax.text(ix, iy, "⚠", ha="center", va="center",
                fontsize=14, color="#E65100", zorder=10)

    ax.text(0, -0.12, "circular track", ha="center", va="center",
            fontsize=7, color="#90A4AE", fontfamily="sans-serif")


def _render_spacetime(ax, history, id_maps, current_step, num_steps, num_cells):
    """Draw the space-time diagram with per-train muted colors."""
    bg = np.array([0.96, 0.96, 0.96])
    color_lut = [_hex_to_rgb(c) for c in SPACETIME_COLORS]

    img = np.full((num_cells, num_steps, 3), bg)
    for t in range(current_step + 1):
        for c in range(num_cells):
            if history[t][c] == 1 and c in id_maps[t]:
                tid = id_maps[t][c]
                img[c, t] = color_lut[tid % len(color_lut)]

    ax.imshow(img, aspect="auto", interpolation="nearest", origin="lower",
              extent=[-0.5, num_steps - 0.5, -0.5, num_cells - 0.5])
    ax.set_facecolor("#FAFAFA")

    ax.axvline(x=current_step, color="#E53935", linewidth=1.2, linestyle="-",
               alpha=0.6, zorder=3)

    ax.set_xlabel("Timestep", fontsize=10, color="#424242")
    ax.set_ylabel("Cell", fontsize=10, color="#424242")
    ax.set_title("Space-Time Diagram", fontsize=11, color="#333333", pad=6)
    ax.set_xlim(-0.5, num_steps - 0.5)
    ax.set_ylim(-0.5, num_cells - 0.5)
    ax.tick_params(colors="#616161", labelsize=8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def _render_anomaly(ax, distances, current_step, num_steps, stall_step, stall_end):
    """Draw the animated latent-distance line chart."""
    # Draw line in segments: blue for normal, red during stall
    for t in range(current_step):
        t0, t1 = t, t + 1
        in_stall = (t0 >= stall_step - 1 and t0 < stall_end)
        color = "#C62828" if in_stall else "#1565C0"
        ax.plot([t0, t1], [distances[t0], distances[t1]],
                color=color, linewidth=1.8, zorder=3)

    # Current-step dot
    dot_in_stall = (current_step >= stall_step - 1 and current_step < stall_end)
    dot_color = "#C62828" if dot_in_stall else "#E53935"
    ax.scatter([current_step], [distances[current_step]], color=dot_color,
               s=30, zorder=4, edgecolors="white", linewidths=0.5)

    ax.set_xlim(-0.5, num_steps - 0.5)
    y_max = max(distances) * 1.15 if max(distances) > 0 else 1.0
    ax.set_ylim(-y_max * 0.05, y_max)
    ax.set_xlabel("Timestep", fontsize=10, color="#424242")
    ax.set_ylabel("Latent Distance", fontsize=10, color="#424242")
    ax.set_title("Anomaly Detection", fontsize=11, color="#333333", pad=6)
    ax.tick_params(colors="#616161", labelsize=8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def render_combined_frame(state, step, num_cells, history, id_maps, num_steps,
                          distances=None, stall_step=None, stall_end=None,
                          active_stall_cell=None):
    """Render track + space-time on top, anomaly chart below."""
    fig = plt.figure(figsize=(10, 7))
    fig.patch.set_facecolor("#F5F5F0")

    # Top row: track (full height of top section) + space-time (shorter)
    # Bottom row: anomaly line chart spanning full width
    gs = fig.add_gridspec(nrows=10, ncols=2, width_ratios=[1.4, 1],
                          hspace=0.6, wspace=0.05)
    ax_track = fig.add_subplot(gs[0:6, 0])         # top-left, full height
    ax_st = fig.add_subplot(gs[1:5, 1])             # top-right, centered
    ax_anom = fig.add_subplot(gs[7:10, :])           # bottom, full width

    _render_track(ax_track, state, step, num_cells, id_map=id_maps[step],
                  stall_cell=active_stall_cell)
    _render_spacetime(ax_st, history, id_maps, step, num_steps, num_cells)
    if distances is not None:
        _render_anomaly(ax_anom, distances, step, num_steps, stall_step, stall_end)

    # Separator line between top plots and anomaly chart
    fig.add_artist(plt.Line2D([0.05, 0.95], [0.365, 0.365],
                              transform=fig.transFigure, color="#BDBDBD",
                              linewidth=0.8, zorder=10))

    fig.subplots_adjust(left=0.06, right=0.82, top=0.96, bottom=0.06)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="#F5F5F0")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def main():
    torch.manual_seed(42)
    env = CircularTrack(num_trains=5)
    env.reset()

    num_steps = 40
    stall_step = 15
    stall_duration = 8
    stall_end = stall_step + stall_duration
    history = []
    states = []
    state_pairs = []  # (x_t, x_next) for JEPA distance computation

    stall_cell = None
    stall_remaining = 0
    stall_cell_per_step = []  # which cell is stalled at each step (or None)

    for step in range(num_steps):
        x_t = env.state.clone()
        x_next, _ = env.step()
        x_next = x_next.clone()
        perturbed = x_next.clone()

        # Start stall
        if step == stall_step:
            for i in range(env.STATE_SIZE):
                if x_t[i] == 1 and x_next[i] == 0:
                    nc = (i + 1) % env.STATE_SIZE
                    if x_next[nc] == 1:
                        perturbed[i] = 1
                        perturbed[nc] = 0
                        stall_cell = i
                        stall_remaining = stall_duration - 1
                        break
            env.state = perturbed.clone()
        # Continue stall
        elif stall_cell is not None and stall_remaining > 0:
            if x_t[stall_cell] == 1 and x_next[stall_cell] == 0:
                nc = (stall_cell + 1) % env.STATE_SIZE
                if x_next[nc] == 1:
                    perturbed[stall_cell] = 1
                    perturbed[nc] = 0
            env.state = perturbed.clone()
            stall_remaining -= 1
            if stall_remaining == 0:
                stall_cell = None

        stall_cell_per_step.append(stall_cell)

        state_pairs.append((x_t, perturbed.clone()))
        state_list = [int(perturbed[i].item()) for i in range(env.STATE_SIZE)]
        history.append(state_list)
        states.append(perturbed.clone())

    id_maps = _track_train_ids(history, env.STATE_SIZE)

    # Train a small JEPA for anomaly detection
    from src.train import train as train_jepa
    print("Training JEPA (500 epochs)...")
    model = train_jepa(num_epochs=500, lr=1e-3)
    model.eval()

    # Compute latent distances
    cond = torch.zeros(env.CONDITION_SIZE)
    distances = []
    with torch.no_grad():
        for x_t, x_next in state_pairs:
            predicted, target = model(x_t, cond, x_next)
            d = torch.sum((predicted - target) ** 2).item()
            distances.append(d)

    frames = []
    for step in range(num_steps):
        frames.append(render_combined_frame(
            states[step], step, env.STATE_SIZE, history, id_maps, num_steps,
            distances=distances, stall_step=stall_step, stall_end=stall_end,
            active_stall_cell=stall_cell_per_step[step]))

    frames[0].save(
        "circular_track.gif",
        save_all=True,
        append_images=frames[1:],
        duration=400,
        loop=0,
    )
    print("Saved circular_track.gif")


if __name__ == "__main__":
    main()
