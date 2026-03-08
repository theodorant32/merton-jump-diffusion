"""
poisson_process.py — Jump process components for Merton model

Two pieces needed to add jumps:
  1. *When* do jumps happen?  → Poisson process (parameter: λ)
  2. *How big* are the jumps? → Log-normal jump sizes (parameters: μ_j, σ_j)

A Poisson process with rate λ fires, on average, λ times per year.
Each firing = a market jump event.
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Jump timing: Poisson process
# ---------------------------------------------------------------------------

def simulate_jump_counts(
    lam: float,         # λ: expected jumps per year
    dt: float,          # time step size (fraction of a year)
    n_steps: int,
    n_paths: int,
    seed: int = 42,
) -> np.ndarray:
    """
    Simulate the number of Poisson jumps in each time step.

    For small dt, the number of jumps in [t, t+dt] is Poisson(λ·dt).
    In practice we use a Bernoulli approximation: at most 1 jump per step.

    Returns:
        jump_counts: shape (n_paths, n_steps), values in {0, 1}
    """
    rng = np.random.default_rng(seed)
    jump_prob = lam * dt   # probability of a jump in this tiny interval
    jump_counts = rng.poisson(jump_prob, size=(n_paths, n_steps))
    return jump_counts


# ---------------------------------------------------------------------------
# Jump size: log-normal distribution
# ---------------------------------------------------------------------------

def simulate_jump_sizes(
    mu_j: float,        # mean of log jump size
    sigma_j: float,     # std of log jump size
    shape: tuple,
    seed: int = 43,
) -> np.ndarray:
    """
    Simulate jump magnitudes drawn from a log-normal distribution.

    In Merton's model, jump size J = exp(μ_j + σ_j * Z) where Z ~ N(0,1).
    So log(J) ~ N(μ_j, σ_j²).

    A negative μ_j biases toward downward jumps (crashes).
    A positive μ_j biases toward upward jumps (gap-ups).

    Returns:
        jump_sizes: shape matching `shape`, values are *multiplicative* factors
    """
    rng = np.random.default_rng(seed)
    log_jumps = rng.normal(mu_j, sigma_j, size=shape)
    return np.exp(log_jumps)   # multiplicative factor applied to price


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def plot_poisson_arrivals(lam: float = 3.0, T: float = 1.0, n_paths: int = 5) -> None:
    """
    Visualise a few sample jump arrival timelines.
    Useful for building intuition about the Poisson process.
    """
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(12, 4))

    for i in range(n_paths):
        # Simulate inter-arrival times (exponential with rate λ)
        arrivals = []
        t = 0.0
        while t < T:
            t += rng.exponential(1 / lam)
            if t < T:
                arrivals.append(t)

        ax.scatter(arrivals, [i] * len(arrivals), marker="|", s=200,
                   color=f"C{i}", linewidth=2)
        ax.text(-0.02, i, f"Path {i+1}", ha="right", va="center", fontsize=9)

    ax.set_xlim(0, T)
    ax.set_xlabel("Time (years)")
    ax.set_title(f"Poisson Jump Arrivals (λ={lam} jumps/year) — {n_paths} sample paths")
    ax.set_yticks([])
    ax.axvline(0, color="black", linewidth=0.5)
    plt.tight_layout()
    plt.savefig("poisson_arrivals.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: poisson_arrivals.png")


def plot_jump_size_distribution(
    mu_j: float = -0.05,
    sigma_j: float = 0.1,
    n_samples: int = 10_000,
) -> None:
    """
    Show the distribution of jump sizes and their effect (% price change).
    """
    rng = np.random.default_rng(1)
    log_jumps = rng.normal(mu_j, sigma_j, n_samples)
    jump_sizes = np.exp(log_jumps)
    pct_changes = (jump_sizes - 1) * 100  # convert to % change

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].hist(log_jumps, bins=80, color="tomato", alpha=0.75, edgecolor="white")
    axes[0].axvline(mu_j, color="black", linewidth=2, label=f"μ_j = {mu_j}")
    axes[0].set_title("Log Jump Size  log(J) ~ N(μ_j, σ_j²)")
    axes[0].set_xlabel("log(J)")
    axes[0].legend()

    axes[1].hist(pct_changes, bins=80, color="tomato", alpha=0.75, edgecolor="white")
    axes[1].axvline(np.mean(pct_changes), color="black", linewidth=2,
                    label=f"Mean jump: {np.mean(pct_changes):.1f}%")
    axes[1].set_title("Jump Size as % Price Change")
    axes[1].set_xlabel("% change")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("jump_size_distribution.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: jump_size_distribution.png")


if __name__ == "__main__":
    print("=== Poisson Process Demo ===\n")

    lam = 3.0        # 3 jumps per year on average
    dt = 1 / 252
    n_steps = 252
    n_paths = 10_000

    jump_counts = simulate_jump_counts(lam, dt, n_steps, n_paths)
    total_jumps_per_path = jump_counts.sum(axis=1)

    print(f"λ (jumps/year)          : {lam}")
    print(f"Avg jumps per path      : {total_jumps_per_path.mean():.2f}  (should be ≈ {lam})")
    print(f"Std jumps per path      : {total_jumps_per_path.std():.2f}")

    print("\n=== Jump Size Demo ===\n")
    mu_j, sigma_j = -0.05, 0.10
    sizes = simulate_jump_sizes(mu_j, sigma_j, shape=(10_000,))
    print(f"μ_j={mu_j}, σ_j={sigma_j}")
    print(f"Mean jump factor        : {sizes.mean():.4f}  ({(sizes.mean()-1)*100:.2f}%)")
    print(f"Median jump factor      : {np.median(sizes):.4f}  ({(np.median(sizes)-1)*100:.2f}%)")

    plot_poisson_arrivals(lam=lam, T=1.0, n_paths=8)
    plot_jump_size_distribution(mu_j=mu_j, sigma_j=sigma_j)
