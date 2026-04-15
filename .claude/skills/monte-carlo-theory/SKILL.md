---
name: monte-carlo-theory
description: Mathematical reference for Monte Carlo option pricing — Black-Scholes formula, Geometric Brownian Motion path simulation, payoff functions, variance reduction techniques (antithetic variates, control variates), and convergence analysis. Use whenever the user asks about the math behind the project, needs to validate a result, derives a new option type (e.g., Asian, barrier), or needs a reference for the analytical ground truth.
---

# Monte Carlo Option Pricing — Theory Reference

This skill is the math handbook for our project. Use it to ground all numerical work in correct theory.

## Black-Scholes Closed-Form Formula (Ground Truth)

For a European call option with parameters:
- `S0` = current asset price
- `K`  = strike price
- `r`  = risk-free interest rate (annualized)
- `sigma` = volatility (annualized standard deviation of log-returns)
- `T`  = time to maturity (years)

The price is:

```
C = S0 * N(d1) - K * exp(-r*T) * N(d2)

where:
  d1 = [ ln(S0/K) + (r + sigma^2/2)*T ] / (sigma * sqrt(T))
  d2 = d1 - sigma * sqrt(T)
  N(x) = standard normal CDF
```

For a put:
```
P = K * exp(-r*T) * N(-d2) - S0 * N(-d1)
```

Put-call parity check: `C - P = S0 - K*exp(-r*T)`

**Reference values for the project's default parameters** (S0=100, K=100, r=0.05, sigma=0.2, T=1.0):
- Call price ≈ **10.4506**
- Put price  ≈ **5.5735**

Any GPU implementation must converge to these (within Monte Carlo error) when given enough paths.

## Geometric Brownian Motion (GBM) — The Simulation Model

Under the Black-Scholes model, asset price follows:

```
dS = r*S*dt + sigma*S*dW
```

where `dW` is a Wiener process increment. For numerical simulation, the **exact discrete solution** (no time-discretization error) is:

```
S(t + dt) = S(t) * exp( (r - sigma^2/2)*dt + sigma * sqrt(dt) * Z )
```

where `Z ~ N(0, 1)` is a standard normal random variable.

**Important:** Use this exact form, NOT the Euler discretization `S(t+dt) = S(t) * (1 + r*dt + sigma*sqrt(dt)*Z)`. Exact GBM is unconditionally positive and has no time-step error for European options.

For European options, you can take **one big step from t=0 to T**:
```
S_T = S0 * exp( (r - sigma^2/2)*T + sigma * sqrt(T) * Z )
```
This is what we'll use in the naive kernel — one random number per path, no time loop.

For path-dependent options (Asian, barrier), you need many small steps.

## Payoff Functions

| Option Type | Payoff at Maturity |
|---|---|
| European call | `max(S_T - K, 0)` |
| European put  | `max(K - S_T, 0)` |
| Asian call (arithmetic) | `max(mean(S_t) - K, 0)` over the path |
| Up-and-out barrier call | `max(S_T - K, 0) * I(max(S_t) < B)` |
| Lookback call | `S_T - min(S_t)` |

The **price** is the discounted expected payoff:
```
Price = exp(-r*T) * E[ payoff ]
```

For Monte Carlo, we estimate `E[ payoff ]` as the sample mean over N simulated paths:
```
Price ≈ exp(-r*T) * (1/N) * sum_i payoff_i
```

## Standard Error & Convergence Rate

Monte Carlo's central limit theorem result:

```
standard_error = sigma_payoff / sqrt(N)
```

where `sigma_payoff` is the standard deviation of the payoff samples. This means **error decreases as O(1/sqrt(N))** — to halve the error, you need 4× more paths. To get one more digit of precision, 100× more paths.

The **95% confidence interval** for the price estimate:
```
CI = price ± 1.96 * standard_error
```

When validating a kernel, check that the analytical price falls within the 95% CI.

## Variance Reduction: Antithetic Variates

The simplest and cheapest variance reduction. For each random sample `Z`, also use `-Z`:

```
For each pair:
  S_T_plus  = S0 * exp( (r - sigma^2/2)*T + sigma * sqrt(T) * (+Z) )
  S_T_minus = S0 * exp( (r - sigma^2/2)*T + sigma * sqrt(T) * (-Z) )
  payoff_plus  = max(S_T_plus  - K, 0)
  payoff_minus = max(S_T_minus - K, 0)
  paired_payoff = (payoff_plus + payoff_minus) / 2

Estimate = mean of paired_payoffs over N/2 pairs
```

**Why it works:** the call payoff is monotonic in `Z`, so `payoff_plus` and `payoff_minus` are negatively correlated. Their average has lower variance than two independent samples. Typical variance reduction for vanilla European: ~2-4×, meaning equivalent to 2-4× more paths at no extra random number cost.

**GPU implementation note:** pair threads with `tid` and `tid + N/2`, or have each thread compute both samples. The latter doubles work per thread but eliminates inter-thread coordination.

## Variance Reduction: Control Variates

If you have a related quantity with known expectation, use it as a "control":

```
Estimate = mean(payoff) - beta * (mean(control) - E[control])

where beta is chosen to minimize variance.
```

For European options, a common control is the asset price itself (`E[S_T] = S0 * exp(r*T)`). For Asian options, the geometric Asian (which has a closed form) is a great control for the arithmetic Asian (which doesn't).

We're not implementing this in Plan-to-Achieve, but it's a Hope-to-Achieve candidate if antithetic variates work and we have time.

## Random Number Generators in cuRAND

| Generator | Period | Memory per state | Speed | Notes |
|---|---|---|---|---|
| XORWOW | ~2^192 | ~48 bytes | Medium | Default; well-tested |
| MRG32k3a | ~2^191 | ~80 bytes | Medium | Higher-quality statistics |
| Philox4_32_10 | ~2^128 per stream | ~16 bytes | **Fast** | Counter-based, parallel-friendly |
| Sobol32 | quasi-random | varies | Slow | Better convergence (1/N instead of 1/sqrt(N)) for low-dim problems |

For our project: start with **XORWOW**, then benchmark **Philox** as the optimization. Sobol is interesting for the report ("Hope to Achieve: quasi-Monte Carlo") because it changes the convergence rate fundamentally.

## Validation Procedure

When a new kernel version is written, validate as follows:

1. **Pick standard parameters:** S0=100, K=100, r=0.05, sigma=0.2, T=1.0.
2. **Compute analytical price** via `black_scholes.h` → expect ~10.4506 for call.
3. **Run kernel** with N = 1,000,000 paths.
4. **Compute price estimate** and **standard error** from kernel output.
5. **Check:** analytical price should be within `estimate ± 2 * std_error` ~95% of the time.
6. If **not**, the kernel has a bug (likely RNG correlation across threads, or reduction error, or units mistake on r/sigma/T).

## Common Bugs to Watch For

- **Time units mismatch.** r and sigma must be in the same time unit as T. If T is in years and sigma is daily volatility, you'll get garbage.
- **Forgetting the discount factor.** Final price needs `exp(-r*T)` applied.
- **Signed payoff.** A call payoff is `max(S - K, 0)`, NOT `S - K`. Forgetting the `max` produces a wrong answer that looks plausible.
- **Seeding all threads identically.** Every thread getting the same Z value means N=1 effectively.
- **Using the wrong discretization for path-dependent options.** Euler scheme breaks for barrier options near the barrier.

## References

- F. Black, M. Scholes (1973). *The Pricing of Options and Corporate Liabilities*. JPE.
- P. Glasserman (2003). *Monte Carlo Methods in Financial Engineering*. Springer. **The reference book.**
- M. Joshi (2008). *More Mathematical Finance*. For Asian, barrier, and exotic option pricing.
- NVIDIA cuRAND Library Programming Guide.
