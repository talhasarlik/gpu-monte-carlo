# Poster Presenter Guide — GPU Monte Carlo Option Pricing (poster_v4)

## The 60-second pitch

> "Monte Carlo is the standard way to price options, but its error shrinks only as 1/√N — so you need tens of millions of simulated paths, and a CPU takes seconds per price. Each path is independent, which makes this a textbook GPU problem. We built three CUDA kernels, each isolating one performance lesson. The **naive** kernel gives every path its own thread and atomicAdd — the profiler shows the SMs doing useful work only 2.8% of the time, because millions of atomics serialize in L2. The **shared** kernel fixes that with a warp-shuffle + shared-memory reduction: one atomic per block instead of two per path, and SM throughput jumps to 79%. The **antithetic** kernel then halves the variance for free by mirroring every random draw. The change that mattered most, though, was the **grid-stride refactor**: a fixed 16K-thread grid striding over paths makes RNG state O(threads) instead of O(N) — init gets 65× cheaper and the kernels flip from memory-bound to compute-bound. End result: **382× end-to-end over a single-core CPU at 10 million paths, 980× at 50 million**, with every run validated against the closed-form Black–Scholes price."

## Walking order (matches the numbered panels)

Overview chips (the strip is your map) → **1 naive** → **2 shared** → **3 antithetic** → **4 grid-stride** → **5 roofline** → results charts → hero band numbers.

One sentence per panel if the visitor is in a hurry:

1. **Naive** — "94% occupancy but 2.8% compute: occupancy isn't utilization."
2. **Shared** — "Atomics go from ~2·N to ~128; the diagram below is the whole trick."
3. **Antithetic** — "Pair +Z with −Z, noise cancels: half the variance, one extra exp()."
4. **Grid-stride** — "The table is before/after: RNG init 44.6 ms → 0.68 ms. Biggest single win."
5. **Roofline** — "Proof, not vibes: optimized kernels sit on the FP32 compute ceiling; naive is stuck far below."

## Anticipated questions (and honest answers)

- **"Why is the naive curve rising in the validation plot?"** At 5×10⁷ paths the naive kernel's FP32 atomicAdd accumulation degrades — adding tiny payoffs into one huge float sum loses precision. The hierarchical reduction avoids this. It's a correctness argument for the optimized kernels, not just a speed one.
- **"1.4σ for naive/shared — is that a failure?"** No. σ here is the Monte Carlo standard error; ≤1.4σ is statistically consistent. Antithetic lands at 0.8σ. Our automated gate accepts runs within 4σ.
- **"Why a GTX 1650 and not the Colab T4/A100?"** Development happened on Colab, but benchmarks were run locally for stable clocks and exclusive access. Same sm_75 architecture as the T4.
- **"Why FP32?"** Consumer GPUs have ~1/32 FP64 throughput; FP32 with a validated error budget is the right engineering trade. The precision analysis (naive blow-up) is exactly why we measure it.
- **"Is 382× a fair comparison?"** It's end-to-end (RNG init + kernel) vs a single CPU thread, median of 5 runs. Kernel-only is ≈1652×; a multi-threaded CPU would close some gap — we say so if asked.
- **"Why one thread = one path?"** European options need only the terminal price — one exact GBM step. For path-dependent (Asian) options we'd revisit the mapping.

## Key numbers (memorize these five)

| Number | Meaning |
|---|---|
| 382× / 980× | end-to-end speedup vs 1-core CPU at 10⁷ / 5×10⁷ paths |
| ≈1652× | kernel-only speedup at 5×10⁷ |
| 2.8% → 79% | SM throughput, naive → shared (the atomics lesson) |
| 65× | RNG init cost reduction from grid-stride |
| 0.8σ | antithetic error vs analytical Black–Scholes at 10⁶ |

These five are the whole story: two say *how fast* (382×/980×, 1652×), two say *why* (2.8→79%, 65×), one says *it's correct* (0.8σ).

### How to explain each number

**382× / 980× (end-to-end).** Same job — simulate N paths and produce a price — timed wall-to-wall on both sides. One CPU core does ~30 M paths/s → ≈360 ms at 10⁷; the GPU (mc_antithetic, RNG init + kernel + readback included) ≈0.9 ms → 382×. At 5×10⁷: ≈1.6 s vs ≈1.6 ms → 980×. Two numbers because speedup grows with N as GPU fixed costs amortize — quoting both is the honest version.

**≈1652× (kernel-only).** Same comparison but simulation kernel time only (init excluded). Read it together with 980×: "one price from cold start = 980×; many prices back-to-back, init paid once → approaches 1652×."

**2.8% → 79% (SM throughput).** Nsight's measure of how much of peak compute the SMs actually sustained. Naive: every path atomicAdds the same two global addresses → atomics serialize in L2 → cores idle ~97% → 2.8%, despite 94% occupancy (resident ≠ working). Shared: register → warp shuffle → shared-memory tree → one atomic per block (~128 total) → 79%. Same math, only the summation changed — 28× better utilization. This is the poster's core lesson.

**65× (RNG init).** One-thread-per-path needs N cuRAND states (48 B × 10⁶ ≈ 46 MB); init_rng alone took 44.6 ms at N=10⁶ — longer than the kernel itself. The fixed 16,384-thread grid-stride needs only 16,384 states → 0.68 ms = 65×. The side effect matters most: state traffic gone, kernels flip memory-bound → compute-bound (exactly what the roofline panel shows).

**0.8σ (correctness).** σ = Monte Carlo standard-error units. Antithetic at N=10⁶: estimate 10.4593 vs analytical 10.4506 → error 0.0088; its SE ≈ 0.011 → 0.8σ — the deviation is pure sampling noise, no bug. Values near 1σ are *healthy* for an unbiased estimator; don't confuse this with the 4σ acceptance gate.

**Use them in this order:** "We're 382× faster end-to-end — 1652× kernel-only — because we fixed the atomic bottleneck (2.8% → 79%) and made RNG init 65× cheaper; and the answer is right: 0.8σ from the analytical price."

## Demo line (if the live demo is running)

"Pick any N — price and runtime update instantly; the analytical value is printed next to it, so you can watch the error shrink as 1/√N."
