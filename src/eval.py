"""Evaluation with normal/perturbed scenarios and matplotlib visualization."""

import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.env import CircularTrack
from src.jepa import JEPA


def run_normal_scenario(
    model: JEPA, env: CircularTrack, num_steps: int = 20
) -> list[float]:
    """Run normal scenario. Returns list of latent distances per timestep."""
    result = run_scenario_with_states(model, env, num_steps=num_steps)
    return result["distances"]


def run_perturbed_scenario(
    model: JEPA,
    env: CircularTrack,
    num_steps: int = 20,
    perturb_step: int = 10,
) -> list[float]:
    """Run perturbed scenario. Returns latent distances."""
    result = run_scenario_with_states(
        model, env, num_steps=num_steps, stall_step=perturb_step,
        stall_duration=1, stall_count=1,
    )
    return result["distances"]



def run_scenario_with_states(
    model: JEPA,
    env: CircularTrack,
    num_steps: int = 20,
    stall_step: int | None = None,
    stall_duration: int = 3,
    stall_count: int = 1,
    stall_cell_target: int | None = None,
) -> dict:
    """Run a scenario, return distances and track states.

    Perturbation: at stall_step, stall_count trains are frozen for
    stall_duration steps, then all resume.
    stall_cell_target: if set, prefer freezing the train closest to this cell.
    """
    model.eval()
    distances: list[float] = []
    states: list[list[int]] = []
    perturbed_states: list[list[int]] = []
    env.reset()
    cond = torch.zeros(env.CONDITION_SIZE)

    stall_cells: list[int] = []  # cells where stalled trains sit
    stall_remaining = 0

    with torch.no_grad():
        for step in range(num_steps):
            x_t = env.state.clone()

            x_next, _ = env.step(cond)
            x_next = x_next.clone()
            x_next_for_jepa = x_next.clone()

            # --- Start stall: freeze stall_count trains ---
            if stall_step is not None and step == stall_step:
                # Find all movable trains
                candidates = []
                for i in range(env.STATE_SIZE):
                    if x_t[i] == 1 and x_next[i] == 0:
                        next_cell = (i + 1) % env.STATE_SIZE
                        if x_next[next_cell] == 1:
                            candidates.append(i)
                # Sort by proximity to target cell if specified
                if stall_cell_target is not None and candidates:
                    sz = env.STATE_SIZE
                    candidates.sort(
                        key=lambda c: min(abs(c - stall_cell_target),
                                          sz - abs(c - stall_cell_target))
                    )
                for i in candidates[:stall_count]:
                    x_next_for_jepa[i] = 1
                    next_cell = (i + 1) % env.STATE_SIZE
                    x_next_for_jepa[next_cell] = 0
                    stall_cells.append(i)
                stall_remaining = stall_duration - 1
                env.state = x_next_for_jepa.clone()

            # --- Continue stall: keep frozen trains in place ---
            elif stall_cells and stall_remaining > 0:
                for cell in stall_cells:
                    if x_t[cell] == 1 and x_next[cell] == 0:
                        next_cell = (cell + 1) % env.STATE_SIZE
                        if x_next[next_cell] == 1:
                            x_next_for_jepa[cell] = 1
                            x_next_for_jepa[next_cell] = 0
                env.state = x_next_for_jepa.clone()
                stall_remaining -= 1
                if stall_remaining == 0:
                    stall_cells.clear()

            predicted, target = model(x_t, cond, x_next_for_jepa)
            d = torch.sum((predicted - target) ** 2).item()

            distances.append(d)
            states.append([int(x_next[i].item()) for i in range(env.STATE_SIZE)])
            perturbed_states.append(
                [int(x_next_for_jepa[i].item()) for i in range(env.STATE_SIZE)]
            )

    return {
        "distances": distances,
        "states": states,
        "perturbed_states": perturbed_states,
    }



def is_anomaly(
    normal_distances: list[float],
    perturbed_distances: list[float],
    step: int,
    sigma_factor: float = 3.0,
) -> bool:
    """Return True if perturbed distance at step exceeds μ + sigma_factor × σ."""
    mu = statistics.mean(normal_distances)
    sigma = statistics.stdev(normal_distances)
    threshold = mu + sigma_factor * sigma
    return perturbed_distances[step] > threshold


def plot_results(
    normal_distances: list[float],
    perturbed_distances: list[float],
    save_path: str,
) -> None:
    """Plot both scenarios with anomaly threshold and save to file."""
    mu = statistics.mean(normal_distances)
    sigma = statistics.stdev(normal_distances)
    threshold = mu + 3.0 * sigma
    timesteps = list(range(len(normal_distances)))

    fig, ax = plt.subplots()
    ax.plot(timesteps, normal_distances, label="Normal", marker="o", markersize=3)
    ax.plot(timesteps, perturbed_distances, label="Perturbed", marker="s", markersize=3)
    ax.axhline(y=threshold, color="r", linestyle="--",
               label=f"Threshold (μ+3σ = {threshold:.4f})")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Latent Distance")
    ax.set_title("JEPA Anomaly Detection: Normal vs Perturbed")
    ax.legend()
    fig.savefig(save_path)
    plt.close(fig)


def plot_full_dashboard(
    normal_result: dict,
    stall_results: list[dict],
    save_path: str,
    stall_step: int = 10,
    durations: list[int] | None = None,
) -> None:
    """Plot latent distances + track heatmaps in a compact 2-row layout.

    Row 1: Latent prediction error (line chart, full width)
    Row 2: Heatmaps side by side (normal + stall scenarios)
    """
    if durations is None:
        durations = [2, 5, 10]

    n_dists = normal_result["distances"]
    n_states = np.array(normal_result["perturbed_states"])
    num_steps = len(n_dists)

    colors = ["#FF9800", "#9C27B0", "#2196F3"]
    markers = ["s", "^", "D"]
    heatmap_colors = [
        (1.0, 0.60, 0.0),
        (0.61, 0.15, 0.69),
        (0.13, 0.59, 0.95),
    ]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    num_heatmaps = 1 + len(stall_results)  # normal + stalls
    fig = plt.figure(figsize=(14, 8), facecolor="white")
    gs = fig.add_gridspec(2, num_heatmaps, height_ratios=[1.3, 1],
                          hspace=0.35, wspace=0.25)

    # --- Row 1: Latent distances (spans full width) ---
    ax_dist = fig.add_subplot(gs[0, :])
    ax_dist.set_facecolor("#FAFAFA")
    ts = list(range(num_steps))
    for idx in range(len(stall_results) - 1, -1, -1):
        s_dists = stall_results[idx]["distances"]
        dur = durations[idx]
        ax_dist.plot(ts, s_dists, label=f"Stall {dur} steps", marker=markers[idx],
                     markersize=5, color=colors[idx], alpha=0.85, linewidth=1.8)
    ax_dist.plot(ts, n_dists, label="Normal", marker="o", markersize=4,
                 color="#4CAF50", linewidth=2.0, zorder=5)
    ax_dist.axvspan(stall_step - 0.5, stall_step + max(durations) - 0.5,
                    alpha=0.06, color="#F44336", zorder=0)
    ax_dist.axvline(x=stall_step, color="#9E9E9E", linestyle="--", alpha=0.5,
                    linewidth=0.8)
    ax_dist.set_ylabel("Latent Distance", fontsize=11, color="#424242")
    ax_dist.set_xlabel("Timestep", fontsize=11, color="#424242")
    ax_dist.legend(loc="upper right", fontsize=9, framealpha=0.9,
                   edgecolor="#E0E0E0", fancybox=True)
    ax_dist.set_title("Latent Prediction Error", fontsize=12, pad=8, color="#333333")
    ax_dist.set_xlim(-0.5, num_steps - 0.5)
    ax_dist.grid(axis="y", alpha=0.3, linewidth=0.5, color="#BDBDBD")
    ax_dist.tick_params(colors="#616161", labelsize=9)

    # --- Row 2: Heatmaps side by side ---
    # Normal
    ax_n = fig.add_subplot(gs[1, 0])
    _plot_circular_heatmap(ax_n, n_states, "Normal",
                           train_color=(0.30, 0.69, 0.31), compact=True)

    # Stall scenarios
    for idx, (result, dur) in enumerate(zip(stall_results, durations)):
        ax_s = fig.add_subplot(gs[1, 1 + idx])
        s_states = np.array(result["perturbed_states"])
        end = stall_step + dur
        _plot_circular_heatmap(
            ax_s, s_states,
            f"Stall {dur} steps",
            highlight_range=(stall_step, end),
            train_color=heatmap_colors[idx],
            compact=True,
        )
        ax_s.set_ylabel("")  # only leftmost gets y-label

    fig.suptitle("JEPA Anomaly Detection — Circular Track",
                 fontsize=15, fontweight="bold", y=1.0, color="#212121")
    fig.savefig(save_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)



def _plot_circular_heatmap(
    ax, states: np.ndarray, title: str,
    highlight_step: int | None = None,
    highlight_range: tuple[int, int] | None = None,
    train_color: tuple[float, float, float] = (0.13, 0.59, 0.95),
    compact: bool = False,
) -> None:
    """Heatmap: x=timestep, y=cell (space-time diagram convention)."""
    num_steps, num_cells = states.shape

    bg = np.array([0.96, 0.96, 0.96])
    cmap_data = np.full((num_steps, num_cells, 3), bg)
    for t in range(num_steps):
        for i in range(num_cells):
            if states[t, i] == 1:
                cmap_data[t, i] = train_color

    # Transpose so x=timestep, y=cell
    cmap_transposed = np.transpose(cmap_data, (1, 0, 2))

    ax.imshow(cmap_transposed, aspect="auto", interpolation="nearest",
              origin="lower")
    ax.set_facecolor("#FAFAFA")

    if highlight_step is not None and highlight_step < num_steps:
        ax.axvline(x=highlight_step, color="#E53935", linewidth=1.5,
                   linestyle="--", alpha=0.7)

    if highlight_range is not None:
        start, end = highlight_range
        ax.axvspan(start - 0.5, end - 0.5, alpha=0.08, color="#F44336", zorder=0)
        ax.axvline(x=start - 0.5, color="#E53935", linewidth=0.8,
                   linestyle="--", alpha=0.5)
        ax.axvline(x=end - 0.5, color="#E53935", linewidth=0.8,
                   linestyle="--", alpha=0.5)

    step_size = 10 if compact else 5
    ax.set_xticks(range(0, num_steps, step_size))
    ax.set_xticklabels([str(i) for i in range(0, num_steps, step_size)],
                       fontsize=7, color="#616161")
    cell_step = 4 if compact else 2
    ax.set_yticks(range(0, num_cells, cell_step))
    ax.set_yticklabels([str(i) for i in range(0, num_cells, cell_step)],
                       fontsize=7, color="#616161")
    ax.set_xlabel("Timestep", fontsize=9, color="#424242")
    ax.set_ylabel("Cell", fontsize=9, color="#424242")
    ax.set_title(title, fontsize=10, pad=4, color="#333333")
    ax.tick_params(colors="#616161", labelsize=7)



if __name__ == "__main__":
    from src.train import train

    NUM_STEPS = 70
    STALL_STEP = 15
    NUM_TRAINS = 5

    # Different stall durations → different queue lengths
    DURATIONS = [3, 5, 15]

    print("=== Training JEPA model ===")
    model = train()

    print("\n=== Running evaluation scenarios ===")

    # All scenarios share the same initial state
    ref_env = CircularTrack(num_trains=NUM_TRAINS)
    ref_env.reset()
    init_state = ref_env.state.clone()

    # Normal (no perturbation)
    env = CircularTrack(num_trains=NUM_TRAINS)
    env.reset(init_state)
    normal_result = run_scenario_with_states(model, env, num_steps=NUM_STEPS)

    # Stall scenarios with increasing duration
    stall_results = []
    for dur in DURATIONS:
        env = CircularTrack(num_trains=NUM_TRAINS)
        env.reset(init_state)
        result = run_scenario_with_states(
            model, env, num_steps=NUM_STEPS,
            stall_step=STALL_STEP, stall_duration=dur, stall_count=1,
            stall_cell_target=8,
        )
        stall_results.append(result)

    for label, result, dur in [("Normal", normal_result, 0)] + [
        (f"Stall {d}s", r, d) for d, r in zip(DURATIONS, stall_results)
    ]:
        print(f"\n{label} distances:")
        for i, d in enumerate(result["distances"]):
            marker = ""
            if dur > 0 and STALL_STEP <= i < STALL_STEP + dur:
                marker = " <-- stall"
            print(f"  Step {i:2d}: {d:.6f}{marker}")

    plot_full_dashboard(
        normal_result, stall_results,
        "anomaly_dashboard.png",
        stall_step=STALL_STEP,
        durations=DURATIONS,
    )
    print("\nDashboard saved to anomaly_dashboard.png")
