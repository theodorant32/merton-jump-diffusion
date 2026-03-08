"""
gbm.py — Geometric Brownian Motion baseline

This is the model I built before. Keeping it here as the comparison baseline
for jump diffusion. Clean reimplementation with consistent params so the
comparison plots are apples-to-apples.

GBM: dS = μS dt + σS dW
"""

import numpy as np
import matplotlib.pyplot as plt


def simulate_gbm(
    S0: float = 100.0,
    mu: float = 0.05,       # annual drift
    sigma: float = 0.2,     # annual volatility
    T: float = 1.0,         # time horizon in years
    dt: float = 1/252,      # daily steps (252 trading days)
    n_paths: int = 1000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate GBM price paths using Euler-Maruyama discretization.

    Returns:
        t: time grid (shape: n_steps+1)
        paths: simulated price paths (shape: n_paths x n_steps+1)
    """
    rng = np.random.default_rng(seed)

    n_steps = int(T / dt)
    t = np.linspace(0, T, n_steps + 1)

    # Brownian increments: shape (n_paths, n_steps)
    dW = rng.standard_normal((n_paths, n_steps)) * np.sqrt(dt)

    # Log-return increments
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * dW

    # Cumulative sum to get log price path, then exponentiate
    log_paths = np.cumsum(log_returns, axis=1)
    log_paths = np.hstack([np.zeros((n_paths, 1)), log_paths])

    paths = S0 * np.exp(log_paths)

    return t, paths


def get_terminal_distribution(paths: np.ndarray) -> np.ndarray:
    """Return the terminal (final) prices across all simulated paths."""
    return paths[:, -1]


def plot_gbm(t: np.ndarray, paths: np.ndarray, n_display: int = 50) -> None:
    """Quick sanity-check plot of GBM paths."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: sample paths
    ax = axes[0]
    for i in range(min(n_display, paths.shape[0])):
        ax.plot(t, paths[i], alpha=0.3, linewidth=0.7, color="steelblue")
    ax.plot(t, np.mean(paths, axis=0), color="navy", linewidth=2, label="Mean path")
    ax.set_title("GBM — Simulated Price Paths")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Price")
    ax.legend()

    # Right: terminal distribution
    ax = axes[1]
    terminal = get_terminal_distribution(paths)
    ax.hist(terminal, bins=60, color="steelblue", alpha=0.7, edgecolor="white")
    ax.axvline(np.mean(terminal), color="navy", linewidth=2, label=f"Mean: {np.mean(terminal):.2f}")
    ax.axvline(np.median(terminal), color="orange", linewidth=2, linestyle="--", label=f"Median: {np.median(terminal):.2f}")
    ax.set_title("GBM — Terminal Price Distribution")
    ax.set_xlabel("Terminal Price")
    ax.set_ylabel("Frequency")
    ax.legend()

    plt.tight_layout()
    plt.savefig("gbm_baseline.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: gbm_baseline.png")


if __name__ == "__main__":
    print("Running GBM baseline simulation...")
    t, paths = simulate_gbm(
        S0=100,
        mu=0.05,
        sigma=0.2,
        T=1.0,
        n_paths=1000,
    )

    terminal = get_terminal_distribution(paths)
    print(f"  Paths simulated : {paths.shape[0]}")
    print(f"  Time steps      : {paths.shape[1]}")
    print(f"  Terminal mean   : ${np.mean(terminal):.2f}")
    print(f"  Terminal std    : ${np.std(terminal):.2f}")
    print(f"  Min / Max       : ${terminal.min():.2f} / ${terminal.max():.2f}")

    plot_gbm(t, paths)
