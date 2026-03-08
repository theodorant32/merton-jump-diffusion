"""
merton.py — Merton Jump Diffusion Model

Combines GBM continuous diffusion with a Poisson jump process.

Full model:
    dS = (μ - λk̄) S dt + σ S dW + J S dN

Where:
    μ      = drift
    σ      = diffusion volatility
    λ      = jump arrival rate (Poisson)
    k̄      = E[J - 1] = mean jump size factor (compensator, keeps model risk-neutral)
    dW     = Brownian motion increment
    dN     = Poisson increment (0 or 1 per step)
    J      = jump size multiplier ~ LogNormal(μ_j, σ_j²)

The compensated drift (μ - λk̄) ensures that adding jumps doesn't
accidentally inflate the expected return.
"""

import numpy as np
import matplotlib.pyplot as plt

# Reuse components from previous days
from gbm import simulate_gbm
from poisson_process import simulate_jump_counts, simulate_jump_sizes


def simulate_merton(
    S0: float = 100.0,
    mu: float = 0.05,       # annual drift
    sigma: float = 0.2,     # diffusion volatility
    lam: float = 3.0,       # jump frequency (jumps/year)
    mu_j: float = -0.05,    # mean log jump size (negative = crash bias)
    sigma_j: float = 0.10,  # std of log jump size
    T: float = 1.0,
    dt: float = 1 / 252,
    n_paths: int = 1000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate Merton Jump Diffusion price paths.

    Returns:
        t     : time grid, shape (n_steps+1,)
        paths : simulated paths, shape (n_paths, n_steps+1)
    """
    rng = np.random.default_rng(seed)
    n_steps = int(T / dt)
    t = np.linspace(0, T, n_steps + 1)

    # --- Compensator: adjust drift so expected return remains μ ---
    k_bar = np.exp(mu_j + 0.5 * sigma_j**2) - 1   # E[J - 1]
    mu_comp = mu - lam * k_bar                       # compensated drift

    # --- Diffusion component (same as GBM) ---
    dW = rng.standard_normal((n_paths, n_steps)) * np.sqrt(dt)
    diffusion = (mu_comp - 0.5 * sigma**2) * dt + sigma * dW

    # --- Jump component ---
    jump_counts = simulate_jump_counts(lam, dt, n_steps, n_paths, seed=seed + 1)
    jump_sizes = simulate_jump_sizes(mu_j, sigma_j, shape=(n_paths, n_steps), seed=seed + 2)

    # Log jump contribution per step: count * log(J)  (0 if no jump)
    log_jumps = jump_counts * np.log(jump_sizes)

    # --- Combine: log price path ---
    log_returns = diffusion + log_jumps
    log_paths = np.hstack([np.zeros((n_paths, 1)), np.cumsum(log_returns, axis=1)])

    paths = S0 * np.exp(log_paths)
    return t, paths


def get_path_stats(paths: np.ndarray) -> dict:
    """Compute summary statistics for the terminal distribution."""
    terminal = paths[:, -1]
    return {
        "mean"    : np.mean(terminal),
        "median"  : np.median(terminal),
        "std"     : np.std(terminal),
        "skew"    : float(((terminal - terminal.mean()) / terminal.std()) ** 3).real
                    if False else _skewness(terminal),
        "min"     : terminal.min(),
        "max"     : terminal.max(),
        "p5"      : np.percentile(terminal, 5),
        "p95"     : np.percentile(terminal, 95),
    }


def _skewness(x: np.ndarray) -> float:
    mu, sigma = x.mean(), x.std()
    return float(np.mean(((x - mu) / sigma) ** 3))


if __name__ == "__main__":
    print("=== Merton Jump Diffusion Simulation ===\n")

    params = dict(S0=100, mu=0.05, sigma=0.2, T=1.0, n_paths=5000)

    t_gbm, paths_gbm = simulate_gbm(**params)
    t_jd,  paths_jd  = simulate_merton(**params, lam=3.0, mu_j=-0.05, sigma_j=0.10)

    stats_gbm = get_path_stats(paths_gbm)
    stats_jd  = get_path_stats(paths_jd)

    print(f"{'Metric':<16} {'GBM':>12} {'Jump Diffusion':>16}")
    print("-" * 46)
    for k in ["mean", "median", "std", "skew", "p5", "p95"]:
        print(f"{k:<16} {stats_gbm[k]:>12.2f} {stats_jd[k]:>16.2f}")

    print("\nNote how Jump Diffusion shows:")
    print("  • Lower p5  (fatter left tail — crash risk)")
    print("  • Higher std (more spread)")
    print("  • More negative skewness")
