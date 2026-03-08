"""
visualize.py — Side-by-side comparison: GBM vs Merton Jump Diffusion

Final output of the project. Four-panel plot showing:
  1. Sample price paths (GBM)
  2. Sample price paths (Jump Diffusion)
  3. Terminal price distributions overlaid
  4. Tail risk comparison (log scale)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import lognorm

from gbm import simulate_gbm
from merton import simulate_merton, get_path_stats


# ── Shared simulation params ─────────────────────────────────────────────────
PARAMS = dict(
    S0=100,
    mu=0.05,
    sigma=0.20,
    T=1.0,
    dt=1 / 252,
    n_paths=5000,
    seed=42,
)
JUMP_PARAMS = dict(lam=3.0, mu_j=-0.05, sigma_j=0.10)

N_DISPLAY = 40   # paths shown in path plots


def make_comparison_plot() -> None:
    """
    Four-panel figure comparing GBM and Merton Jump Diffusion.
    """
    print("Simulating paths…")
    t_gbm, paths_gbm = simulate_gbm(**PARAMS)
    t_jd,  paths_jd  = simulate_merton(**PARAMS, **JUMP_PARAMS)

    terminal_gbm = paths_gbm[:, -1]
    terminal_jd  = paths_jd[:, -1]

    stats_gbm = get_path_stats(paths_gbm)
    stats_jd  = get_path_stats(paths_jd)

    # ── Layout ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 11))
    fig.patch.set_facecolor("#0f0f14")

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]

    style = {"facecolor": "#0f0f14", "spines": "#2a2a3a"}

    for ax in axes:
        ax.set_facecolor("#13131c")
        for spine in ax.spines.values():
            spine.set_color(style["spines"])
        ax.tick_params(colors="#aaaacc", labelsize=9)
        ax.xaxis.label.set_color("#aaaacc")
        ax.yaxis.label.set_color("#aaaacc")
        ax.title.set_color("#e0e0f0")

    # ── Panel 1: GBM paths ────────────────────────────────────────────────────
    ax = axes[0]
    for i in range(N_DISPLAY):
        ax.plot(t_gbm, paths_gbm[i], alpha=0.25, linewidth=0.6, color="#4da6ff")
    ax.plot(t_gbm, np.mean(paths_gbm, axis=0), color="#ffffff", linewidth=2,
            label=f"Mean: ${stats_gbm['mean']:.1f}")
    ax.set_title("Geometric Brownian Motion — Price Paths", fontweight="bold")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Price ($)")
    ax.legend(fontsize=9, facecolor="#1e1e2e", labelcolor="#e0e0f0")

    # ── Panel 2: Jump Diffusion paths ─────────────────────────────────────────
    ax = axes[1]
    for i in range(N_DISPLAY):
        ax.plot(t_jd, paths_jd[i], alpha=0.25, linewidth=0.6, color="#ff6b6b")
    ax.plot(t_jd, np.mean(paths_jd, axis=0), color="#ffffff", linewidth=2,
            label=f"Mean: ${stats_jd['mean']:.1f}")
    ax.set_title("Merton Jump Diffusion — Price Paths", fontweight="bold")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Price ($)")
    ax.legend(fontsize=9, facecolor="#1e1e2e", labelcolor="#e0e0f0")

    # Annotate a visible jump on a selected path
    # Find a path with the most jumps visible
    sample_path = paths_jd[2]
    diffs = np.diff(np.log(sample_path))
    jump_idx = np.argmax(np.abs(diffs))
    ax.annotate("← jump!", xy=(t_jd[jump_idx], sample_path[jump_idx]),
                xytext=(t_jd[jump_idx] + 0.08, sample_path[jump_idx] * 1.08),
                color="#ffdd57", fontsize=8, arrowprops=dict(arrowstyle="->", color="#ffdd57"))

    # ── Panel 3: Terminal distribution overlay ────────────────────────────────
    ax = axes[2]
    bins = np.linspace(
        min(terminal_gbm.min(), terminal_jd.min()),
        max(terminal_gbm.max(), terminal_jd.max()),
        80
    )
    ax.hist(terminal_gbm, bins=bins, alpha=0.55, color="#4da6ff", label="GBM", density=True)
    ax.hist(terminal_jd,  bins=bins, alpha=0.55, color="#ff6b6b", label="Jump Diffusion", density=True)
    ax.axvline(stats_gbm["mean"],  color="#4da6ff", linewidth=1.5, linestyle="--")
    ax.axvline(stats_jd["mean"],   color="#ff6b6b", linewidth=1.5, linestyle="--")
    ax.set_title("Terminal Price Distribution (T=1yr)", fontweight="bold")
    ax.set_xlabel("Terminal Price ($)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9, facecolor="#1e1e2e", labelcolor="#e0e0f0")

    # ── Panel 4: Tail risk — log scale ────────────────────────────────────────
    ax = axes[3]
    ax.hist(terminal_gbm, bins=80, alpha=0.6, color="#4da6ff", label="GBM", density=True, log=True)
    ax.hist(terminal_jd,  bins=80, alpha=0.6, color="#ff6b6b", label="Jump Diffusion", density=True, log=True)

    # Mark 5th percentile (tail risk)
    ax.axvline(stats_gbm["p5"], color="#4da6ff", linewidth=1.5, linestyle=":",
               label=f"GBM p5: ${stats_gbm['p5']:.1f}")
    ax.axvline(stats_jd["p5"],  color="#ff6b6b", linewidth=1.5, linestyle=":",
               label=f"JD  p5: ${stats_jd['p5']:.1f}")
    ax.set_title("Tail Risk Comparison (log scale)", fontweight="bold")
    ax.set_xlabel("Terminal Price ($)")
    ax.set_ylabel("Density (log)")
    ax.legend(fontsize=8, facecolor="#1e1e2e", labelcolor="#e0e0f0")

    # ── Super-title ────────────────────────────────────────────────────────────
    fig.suptitle(
        "GBM vs Merton Jump Diffusion  |  Monte Carlo Simulation",
        fontsize=14, fontweight="bold", color="#e0e0f0", y=0.98
    )

    plt.savefig("comparison.png", dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    print("Saved: comparison.png")


def print_stats_table() -> None:
    """Print a clean stats comparison table."""
    t_gbm, paths_gbm = simulate_gbm(**PARAMS)
    t_jd,  paths_jd  = simulate_merton(**PARAMS, **JUMP_PARAMS)

    s_gbm = get_path_stats(paths_gbm)
    s_jd  = get_path_stats(paths_jd)

    print("\n" + "=" * 52)
    print(f"{'Metric':<18} {'GBM':>10} {'Jump Diffusion':>16}")
    print("=" * 52)
    metrics = [
        ("Mean ($)",  "mean"),
        ("Median ($)", "median"),
        ("Std ($)",   "std"),
        ("Skewness",  "skew"),
        ("5th pct ($)", "p5"),
        ("95th pct ($)", "p95"),
    ]
    for label, key in metrics:
        print(f"{label:<18} {s_gbm[key]:>10.2f} {s_jd[key]:>16.2f}")
    print("=" * 52)

    delta_p5 = s_jd["p5"] - s_gbm["p5"]
    print(f"\nJump Diffusion 5th percentile is ${abs(delta_p5):.2f} {'lower' if delta_p5 < 0 else 'higher'}")
    print("→ That's the fat tail. That's what GBM misses.\n")


if __name__ == "__main__":
    print_stats_table()
    make_comparison_plot()
