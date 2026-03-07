# Jump Diffusion Monte Carlo 📈

So I was reading the news a few weeks back — markets had one of those days where everything just *dropped* instantly, no warning, no gradual slide. Not a slow bleed. Just: gap down, chaos, recovery. And I kept thinking — my GBM simulator would have never predicted that. It *can't*. By design.

That's what got me down this rabbit hole.

GBM assumes prices move smoothly. Continuously. It's elegant, it's mathematically clean, and it's *wrong* in the ways that actually matter. Real markets gap. Earnings drop a stock 20% overnight. A central bank surprises everyone. A tweet tanks a coin. None of that shows up in a Brownian motion path.

So I'm building a **Merton Jump Diffusion Model** — an upgrade to GBM that adds a Poisson jump process on top of the continuous diffusion. It's the next logical step, and I wanted to document the build as I go.

---

## What is Jump Diffusion?

GBM models price movement as:

```
dS = μS dt + σS dW
```

Jump Diffusion extends this to:

```
dS = μS dt + σS dW + JS dN
```

Where:
- `dW` = standard Brownian motion (the smooth part)
- `dN` = Poisson process that fires when a "jump" happens
- `J` = the jump size (drawn from a normal distribution)
- `λ` = jump frequency (how often jumps happen per year)

---

## Project Roadmap

- [x] Project setup
- [ ] GBM baseline (to compare against)
- [ ] Poisson process implementation
- [ ] Merton jump diffusion model
- [ ] Comparison visualizations + analysis

---

## Why This Matters

I'm not building this to trade. I'm building it because understanding *how* a model breaks is more valuable than understanding *why* it works. GBM breaks at the tails. Jump diffusion is one way to fix that. Fat tails, black swans, leptokurtosis — all that stuff starts making sense once you model the jumps explicitly.

---

## Stack

- Python 3.11+
- NumPy, Matplotlib, SciPy
- No ML libraries — just stochastic math

---

*Building this incrementally. Each commit adds one layer. Commit history tells the story.*
