# Poster Content Sheet (plain-text mirror)

Every string that appears on `poster.tex` is reproduced here in section
order so you can red-pen the prose without touching LaTeX. After
editing, copy each block back into the matching `\begin{posterblock}{...}`
in `poster.tex`.

---

## Title banner

**Title:** GPU-Accelerated Monte Carlo Simulation for Financial Option Pricing

**Authors:** Talha Sarlık (704241052) · Mehmet Bedirhan Önder (704251026)

**Course:** BLG 562E — Parallel Computing for GPUs using CUDA · Asst. Prof. Ayşe Yılmazer · Spring 2026

**Institution:** Faculty of Computer and Informatics Engineering · Istanbul Technical University

---

## Column 1

### Introduction

Monte Carlo (MC) is the standard tool for pricing path-dependent financial options when no closed-form solution exists. For European options the analytical Black–Scholes formula provides exact ground truth, which we use as a *validation oracle*.

The algorithm is **embarrassingly parallel**: each simulated price path is an independent random walk under Geometric Brownian Motion. This makes MC a textbook fit for CUDA — but extracting the parallelism requires careful attention to memory hierarchy and reduction patterns.

### Motivation

A single-threaded CPU prices one European call at 5×10⁷ paths in about **1.6 s**. The same workload on a consumer GTX 1650 takes **~1 ms**. Translating this textbook parallelism into >1000× wall-clock speedup is not automatic — the difference between a naive and an optimised kernel is itself a **3.7× kernel-level speedup**, and an unsafe FP32 atomic reduction produces a *numerically wrong* answer at large N. This poster quantifies that gap and identifies the bottleneck per implementation with Nsight Compute.

### Contributions

1. Three progressively-tuned CUDA kernels for European MC option pricing: `mc_naive` (global atomics), `mc_shared` (block-level reduction), `mc_antithetic` (variance reduction).
2. Validation against the analytical Black–Scholes price to within sub-σ tolerance at N=10⁶.
3. Profiler-driven bottleneck analysis: each optimisation step is justified by a specific Nsight Compute signal (atomic contention → compute-bound).
4. Reproducible benchmark sweep over 10⁴ → 5×10⁷ paths, median of 5 runs.
5. Headline: **1133×** (shared) and **1652×** (antithetic) speedup over the CPU baseline at N=5×10⁷.

### Algorithm Flow

(diagram — see `diagrams/algorithm_flow.tex`)
Host setup → init_rng_kernel (cuRAND XORWOW state per thread) → mc_*_kernel grid-stride loop (Z ∼ N(0,1); S_T = S₀·exp((r-σ²/2)T + σ√T·Z); g = max(S_T-K, 0)) → block-level reduction (warp shuffle → shared memory) → one atomicAdd per block → host: discount, average, 95% CI.

---

## Column 2

### Problem Definition

**European call option.** Right to buy underlying at strike K at expiry T. Under Black-Scholes assumptions the underlying follows Geometric Brownian Motion:

    dS_t = r·S_t·dt + σ·S_t·dW_t

Closed-form for the terminal price (one Euler step is exact under GBM):

    S_T = S₀·exp((r - σ²/2)·T + σ·√T·Z),    Z ∼ N(0,1)

**Monte Carlo estimator** — discounted average of N i.i.d. payoffs:

    Ĉ = e^(-rT) · (1/N) · Σ max(S_T^(i) - K, 0)

**Convergence.** Standard error shrinks as O(N⁻¹ᐟ²): quadrupling N halves the error, motivating GPU throughput.

**Validation oracle.** Analytical Black–Scholes call C = S₀·Φ(d₁) - K·e^(-rT)·Φ(d₂).

**Parameters used throughout:** S₀ = K = 100, r = 0.05, σ = 0.2, T = 1.0. Analytical price = **10.4506**.

### Kernel Variants

**mc_naive.** One thread per path slice (grid-stride loop). `atomicAdd` per path on a global accumulator. Memory pattern is the bottleneck — every payoff serialises on `*d_sum`, *and* FP32 accumulation loses precision at large N (mc_naive @ 5×10⁷ paths reports price = 9.95 vs analytical 10.45).

**mc_shared.** Each thread accumulates its slice's (sum, sum²) in **registers**, then a single `block_reduce_sum()` (warp shuffle → shared-memory tree → thread 0) writes **one** `atomicAdd` per block. Atomic count drops from O(N) to O(N/blockDim).

**mc_antithetic.** For each Z drawn, evaluate the payoff at +Z and -Z and average as one effective sample. Identical block-level reduction as mc_shared. Halves the variance for monotonic payoffs (proven for European calls/puts) without doubling RNG cost.

(architectural figure — see `figures/fig_kernel_architecture.pdf`)

### Reduction Internals

(diagram — see `diagrams/reduction_tree.tex`)
Per-thread register accumulation → warp shuffle (5 levels collapses each warp to lane 0) → shared memory (lane-0 of each warp writes) → final tree → one atomicAdd to global per block.

---

## Column 3

### Experimental Setup

- **GPU:** NVIDIA GTX 1650 (sm_75, 16 SMs, 4 GB), CUDA 12.4, driver 566.36, Windows 11.
- **CPU baseline:** single-threaded, `-O3 -march=native`.
- **Sweep:** N ∈ {10⁴, 10⁵, 10⁶, 10⁷, 2.5×10⁷, 5×10⁷}.
- **Reporting:** median of 5 runs, warmup discarded; warm-cache kernel time only (does not include the one-shot `curand_init`).

### A. Validation

At N = 10⁶ paths, all four implementations agree with the analytical Black-Scholes price within 4-σ:

| Implementation  | Estimate | Abs. Err. |  σ  | Status |
|-----------------|---------:|----------:|----:|:------:|
| mc_cpu          | 10.4501  |  0.0005   | 0.0 |  PASS  |
| mc_naive        | 10.4718  |  0.0212   | 1.4 |  PASS  |
| mc_shared       | 10.4719  |  0.0213   | 1.4 |  PASS  |
| mc_antithetic   | 10.4593  |  0.0088   | 0.8 |  PASS  |

### B. Throughput & Speedup

(3 figures: `fig_throughput.pdf`, `fig_speedup.pdf`, `fig_speedup_bars.pdf`)

Headline @ N = 5×10⁷:

| Kernel          | Kernel time (median, 5 runs) | Speedup vs CPU |
|-----------------|-----------------------------:|---------------:|
| mc_cpu          |                    1586.3 ms |          1.0×  |
| mc_naive        |                     120.5 ms |         13.2×  |
| mc_shared       |                       1.40 ms |       **1133×** |
| mc_antithetic   |                       0.96 ms |       **1652×** |

### C. Profiler-Driven Bottleneck Analysis

Nsight Compute at N = 10⁷ paths, GTX 1650:

| Kernel          | Occupancy | Compute (SM) Tput | DRAM Tput | Bottleneck                |
|-----------------|----------:|------------------:|----------:|---------------------------|
| mc_naive        |     93.8% |          **2.8%** |    0.03%  | atomic contention         |
| mc_shared       |     74.6% |         **79.4%** |    3.5%   | compute (arith.-bound)    |
| mc_antithetic   |     72.8% |         **76.4%** |    5.1%   | same; statistical win     |

Occupancy "looks healthy" for mc_naive but compute throughput exposes the truth: SMs are full of warps, all stalled on `atomicAdd`. (bar chart — `fig_nsight_metrics.pdf`)

### D. Convergence

(figure — `fig_convergence.pdf`)

All implementations track the 1/√N reference; antithetic variates sit **below** the line — variance reduction at no additional architectural cost. The naive kernel *leaves* the line at N ≳ 2.5×10⁷ because FP32 atomic accumulation loses precision.

### Conclusion & Future Work

- Block-level reduction with register accumulation **eliminates atomic contention** and lifts achieved compute throughput from 2.8% to 79%, a **3.7× kernel-level speedup** over the naive version, on top of the GPU-vs-CPU gain.
- The optimised kernel is then **arithmetic-bound**; further wins must attack the RNG (Philox vs XORWOW: smaller state, ~3× faster init).
- **Antithetic variates** cut variance by ~50% on monotonic European payoffs at the *same* architectural profile — a statistical, not architectural, speedup.
- **FP32 atomic accumulation is unsafe** at scale; this is the strongest reason on its own to reduce in shared memory.
- End-to-end: **1652×** faster than single-threaded CPU at 5×10⁷ paths, validated to within 0.8 σ of the Black-Scholes oracle.

**Future work.** XORWOW → Philox RNG; CUDA streams to overlap `curand_init` with the pricing kernel; extension to path-dependent options (Asian, barrier) where MC has no closed-form competitor.

### References

1. F. Black and M. Scholes, "The Pricing of Options and Corporate Liabilities," *J. Political Economy*, 1973.
2. P. Glasserman, *Monte Carlo Methods in Financial Engineering*, Springer, 2003.
3. J. Salmon et al., "Parallel Random Numbers: As Easy as 1, 2, 3," SC'11.
4. D. Kirk and W.-m. Hwu, *Programming Massively Parallel Processors*, 4th ed., 2022.
5. NVIDIA cuRAND Library Programming Guide (CUDA 12.4).
6. NVIDIA CUDA C Best Practices Guide.
7. NVIDIA Nsight Compute Documentation.

---

## Footer

Faculty of Computer and Informatics Engineering · Department of Computer Engineering · Istanbul Technical University · İTÜ Ayazağa Campus, 34469, Maslak/Istanbul · talhasarlik10@gmail.com
