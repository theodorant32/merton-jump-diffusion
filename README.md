# Jump Diffusion Monte Carlo 📈

So I was reading the news a few weeks back — markets had one of those days where everything just *dropped* instantly, no warning, no gradual slide. Not a slow bleed. Just: gap down, chaos, recovery. And I kept thinking — my GBM simulator would have never predicted that. It *can't*. By design.

That's what got me down this rabbit hole.

GBM assumes prices move smoothly. Continuously. It's elegant, it's mathematically clean, and it's *wrong* in the ways that actually matter. Real markets gap. Earnings drop a stock 20% overnight. A central bank surprises everyone. A tweet tanks a coin. None of that shows up in a Brownian motion path.

So I built a **Merton Jump Diffusion Model** — an upgrade to GBM that layers a Poisson jump process on top of continuous diffusion. This repo is the full build, documented as I went.

---

## What is Jump Diffusion?

GBM models price movement as:

```
dS = μS dt + σS dW
```

Merton Jump Diffusion extends this to:

```
dS = (μ - λk̄) S dt + σS dW + JS dN
```

Where:
- `dW` = standard Brownian motion (the smooth part, same as GBM)
- `dN` = Poisson process — fires when a jump happens
- `J` = jump size, drawn from `LogNormal(μ_j, σ_j²)`
- `λ` = jump frequency (jumps per year)
- `k̄ = E[J - 1]` = compensator that keeps expected return honest

The compensated drift `(μ - λk̄)` is the part that took me longest to internalize. Without it, adding jumps silently inflates expected returns — the model lies to you about what you're getting.

---

## Results

Simulated **5,000 paths** over 1 year (252 trading days), `S0=$100`, `μ=5%`, `σ=20%`, with jump parameters `λ=3`, `μ_j=-0.05`, `σ_j=0.10`.

| Metric         | GBM      | Jump Diffusion |
|----------------|----------|----------------|
| Mean ($)       | 105.35   | 105.53         |
| Median ($)     | 103.13   | 103.25         |
| Std ($)        | 21.03    | **29.04**      |
| 5th pct ($)    | 75.01    | **63.24**      |
| 95th pct ($)   | 143.27   | **155.14**     |

The means are nearly identical — both models agree on *expected* outcome. But look at the tails:

- **5th percentile drops from $75 to $63** — that's an extra $11.77 of downside risk that GBM simply doesn't see. That's the fat tail. That's a crash the smooth model would call a near-impossibility.
- **Std jumps from $21 to $29** — 38% more spread, driven entirely by the jump component.
- **95th percentile rises too** — jumps go both ways. Gap-ups happen.

This is exactly what I wanted to demonstrate. The models have the same drift, same volatility, same starting point. The only difference is whether you believe prices can teleport.

---

## How It Works

### 1. GBM Baseline (`gbm.py`)
Standard log-normal diffusion. Euler-Maruyama discretization. This is the "null model" — what everyone uses, what options pricing was built on, and what misses every crash.

### 2. Poisson Jump Timing (`poisson_process.py`)
For each time step, draw jump counts from `Poisson(λ·dt)`. With daily steps and `λ=3`, that's roughly a 1.2% chance of a jump on any given day — which gives you ~3 jumps per year on average.

### 3. Jump Sizes (`poisson_process.py`)
Each jump is a multiplicative factor `J = exp(μ_j + σ_j·Z)` where `Z ~ N(0,1)`. With `μ_j = -0.05`, the average jump is about **-4.9%**. Not catastrophic on its own, but compounded with timing uncertainty, it reshapes the tail entirely.

### 4. Merton Model (`merton.py`)
Combines both components. The log price path becomes:

```
log(S_t) = log(S_0) + Σ [ (μ_comp - σ²/2)dt + σ·dW + N_t·log(J) ]
```

The jump contribution is zero on quiet days, and a sudden log-shift on jump days. Clean, composable, fast.

### 5. Visualization (`visualize.py`)
Four-panel comparison plot:
- Sample paths side-by-side (GBM vs Jump Diffusion)  
- Terminal price distributions overlaid
- Log-scale tail comparison with p5 markers

---

## Project Roadmap

- [x] Project setup
- [x] GBM baseline
- [x] Poisson process implementation
- [x] Merton jump diffusion model
- [x] Comparison visualizations + analysis

---

## Why This Matters

I'm not building this to trade. I'm building it because understanding *how* a model breaks is more valuable than understanding *why* it works. GBM breaks at the tails. Jump diffusion is one fix. Fat tails, black swans, leptokurtosis — all that stuff starts making sense once you model the jumps explicitly.

The $11.77 gap in 5th percentile outcomes isn't just a number. It's the difference between a risk model that says "this scenario has 0.1% probability" and one that says "this scenario happens every few years." That gap is where real money gets lost.

---

## Stack

- Python 3.11+
- NumPy, Matplotlib, SciPy
- No ML libraries — just stochastic math

---

## Usage

```bash
pip install -r requirements.txt

python gbm.py              # GBM baseline
python poisson_process.py  # jump component demo
python merton.py           # model comparison stats
python visualize.py        # four-panel comparison plot
```

---

*Built incrementally. Each module adds one layer. The commit history tells the story.*
