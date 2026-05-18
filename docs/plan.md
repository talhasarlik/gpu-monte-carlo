# Development Plan & Log

Living document. Append entries to the log; update the milestone table as we progress.

## Milestones

| # | Milestone | Owner | Status | Done |
|---|---|---|---|---|
| 0 | Repo + CLAUDE.md skeleton | Talha | done | 2026-04-15 |
| 1 | CPU baseline working & validated vs analytical | Talha | done | 2026-04-16 |
| 2 | mc_naive compiles, runs on GTX 1650, correct price | Bedirhan | done | 2026-04-16 |
| 3 | mc_shared: shared-mem + warp shuffle reduction | Bedirhan | done | 2026-04-16 |
| 4 | mc_antithetic: variance reduction working | Bedirhan | done | 2026-04-16 |
| 5 | Benchmark sweep script + plotting harness | Talha | done | 2026-04-16 |
| 6 | Validation tests pass for all 3 GPU versions | Talha | done | 2026-04-16 |
| 7 | Nsight Compute profiling for each kernel | both | done | 2026-05-18 |
| 8 | cuRAND generator comparison (XORWOW vs Philox) | Bedirhan | todo | |
| 9 | (Hope) CUDA streams overlap | Bedirhan | todo | |
| 10 | (Hope) Asian options extension | Talha | done | 2026-05-18 |
| 11 | Poster draft | both | todo | |
| 12 | Final report | both | todo | |

## Open Questions

- T4 vs A100: do we report numbers from both, or just one? A100 is impressive
  but T4 is more reproducible (free Colab tier).
- For variance reduction analysis: do we report convergence at fixed N, or fixed
  wall-time? Fixed wall-time is more honest but harder to plot cleanly.
- Should `mc_antithetic` reuse the `mc_shared` kernel layout exactly, or experiment
  with a different block size?

## Known Issues

- Windows build requires MSVC environment (vcvarsall x64) for nvcc. `make` not available
  locally; manual nvcc commands or install GNU Make.
- CPU code compiled via `nvcc -x cu` (C++ mode) on Windows — designated initializers replaced
  with plain assignment. `unsigned long` is 32-bit on MSVC, switched to `unsigned long long`.

## Log

### 2026-04-15 — project kickoff
- Repo created. CLAUDE.md, agents, skills, source skeleton in place.
- Source skeleton compiles in principle (verified by inspection, not yet on Colab).
- Decided default architecture: `sm_75` (T4). Override with `make ARCH=sm_80` for A100.
- Default test parameters (S0=100, K=100, r=0.05, sigma=0.2, T=1.0): analytical call
  price ≈ 10.4506. Anything within ~0.01 at N=1e6 is acceptable.

### 2026-04-16 — first local build & validation on GTX 1650

**Environment:** Windows 11, CUDA 12.4, MSVC 19.42, GTX 1650 (sm_75, 4 GB VRAM).

**Windows porting fixes applied:**
- `timer.h`: added `QueryPerformanceCounter` path for Windows (POSIX `clock_gettime` unavailable).
- `params.h`: replaced C99 designated initializers with plain assignment (nvcc C++14 mode).
- `mc_cpu.c`: changed `unsigned long` → `unsigned long long` (MSVC `long` is 32-bit).
- `black_scholes.h`, `mc_cpu.c`: unconditional `_USE_MATH_DEFINES` for `M_PI`.
- `Makefile`: CPU target now uses `nvcc -x cu` (gcc not available on Windows).
- `validate.py`: added `.exe` suffix detection for Windows.

**Validation results (N=1M, S0=100, K=100, r=0.05, σ=0.2, T=1.0):**

| Implementation | Estimate | Analytical | Abs Error | Sigma | Status |
|---|---|---|---|---|---|
| mc_cpu | 10.4501 | 10.4506 | 0.0005 | 0.0 | PASS |
| mc_naive | 10.4718 | 10.4506 | 0.0212 | 1.4 | PASS |
| mc_shared | 10.4719 | 10.4506 | 0.0213 | 1.4 | PASS |
| mc_antithetic | 10.4593 | 10.4506 | 0.0088 | 0.8 | PASS |

**Timing results (N=1M, GTX 1650):**

| Implementation | Kernel Time (ms) | Throughput |
|---|---|---|
| mc_cpu | 31.83 | 31.4 M paths/sec |
| mc_naive | 3.37 | 296.7 M paths/sec |
| mc_shared | 1.06 | 947.1 M paths/sec |
| mc_antithetic | 0.61 | 818.8 M pairs/sec |

**Observations:**
- Shared-memory reduction (mc_shared) gives ~3.2× speedup over naive atomics, and ~30× over CPU.
- Antithetic variates roughly halve the error (0.009 vs 0.021) at similar throughput — the
  variance reduction works as expected for monotonic payoffs.
- GPU RNG init dominates wall-clock at small N (45 ms for cuRAND state setup vs 1-3 ms kernel).
  This is expected and amortizes at larger N.
- All 4 implementations pass validation within 4-sigma tolerance.

### 2026-05-18 — Nsight Compute profiling (Milestone 7)

**Environment:** Windows 11, GTX 1650 (sm_75, 16 SMs, 4 GB), driver 566.36, Nsight Compute 2024.1.1.
Profiled at N=1M paths, default option parameters. Saved per kernel:
- `results/<kernel>_summary.txt` — text per-kernel summary
- `results/<kernel>_full.ncu-rep` — full Nsight Compute report (open in GUI)

**Key metrics (N=1M, GTX 1650):**

| Kernel | Duration | Achieved Occ. | DRAM Tput | Compute Tput | L1 Tput | Regs/T | Block Limit (regs) |
|---|---|---|---|---|---|---|---|
| `init_rng_kernel` | **44.59 ms** | 92.19% | 2.93% | **93.25%** | 97.99% | 63 | 4 |
| `mc_naive_kernel` | 3.27 ms | 68.70% | 22.91% | 3.73% | 25.14% | 24 | 10 |
| `mc_shared_kernel` | 0.877 ms | **97.23%** | **85.23%** | 9.14% | 72.22% | 24 | 10 |
| `mc_antithetic_kernel` | 0.459 ms | 96.51% | 83.28% | 8.91% | 72.72% | 24 | 10 |

**Headline findings:**

1. **`curand_init` is the dominant cost at every N we benchmark.** XORWOW init runs 44.6 ms — that
   is **~50× longer than `mc_shared_kernel`'s 0.88 ms and ~100× longer than `mc_antithetic_kernel`'s
   0.46 ms**. Compute SM throughput hits 93% (saturated) and registers/thread = 63 (limits to
   4 blocks/SM via register file). Achieved occupancy is already 92% — there is no easy
   occupancy fix; the only way to reduce init cost is to switch RNG (Milestone 8: Philox state
   is ~16 B vs XORWOW's 48 B and initializes ~3-5× faster), or to amortize init across many
   simulations.

2. **`mc_naive` is bottlenecked by global atomic contention, not by compute or bandwidth.**
   Occupancy drops to 68.7% (warps stall on `atomicAdd`), Compute SM throughput is just **3.73%**,
   DRAM throughput 22.9% — the SMs are idle waiting for the two global atomics to serialize.
   The shared-memory reduction in `mc_shared` removes that bottleneck: occupancy → 97.23%,
   duration 3.27 ms → 0.88 ms (**3.7× kernel-level speedup**, matches the 3.2× wall-clock from
   April 16).

3. **`mc_shared` is memory-bound, not compute-bound.** DRAM throughput jumps to **85.23%** —
   the kernel is now limited by reading `curandState` (48 B × N = 48 MB at N=1M). Compute is
   still only 9%. Further GPU-side optimization on this kernel has very little headroom unless
   we reduce state size (Philox) or skip the persistent state altogether (regenerate from a
   thread index hash each call).

4. **`mc_antithetic` is `mc_shared` at half the threads, same per-thread profile.** Duration
   0.459 ms ≈ 0.877 / 2; DRAM throughput and occupancy almost identical (83% / 96.5%). The
   kernel is doing twice the math per thread (two `expf`, two `fmax`) but reading the RNG state
   once, so the memory profile is unchanged. The win is statistical, not architectural: same
   memory budget produces an effectively-halved variance for monotonic payoffs.

**Implications for next steps:**
- Milestone 8 (XORWOW vs Philox) is now **the most valuable lever** — init dominates at our
  problem sizes. Run it next.
- Milestone 9 (CUDA streams overlap) would specifically hide `init_rng` behind kernel execution
  — that aligns with the profiler story.
- The "Compute SM throughput 9%" number in `mc_shared`/`mc_antithetic` looks low but is honest:
  the kernel is memory-bound, not under-utilized. Worth calling out on the poster.

**Files produced (in `results/`):**
```
mc_naive_summary.txt        mc_naive_full.ncu-rep
mc_shared_summary.txt       mc_shared_full.ncu-rep
mc_antithetic_summary.txt   mc_antithetic_full.ncu-rep
```

### 2026-05-18 — Grid-stride refactor + shared host harness

**What changed:**
- New `src/gpu/mc_runner.cuh` — shared host harness (`mc_setup` / `mc_report` /
  `mc_teardown` + the single `mc_init_rng_kernel`). Each `.cu` file is now ~50 lines
  instead of ~120; main() is reduced to "set up, launch kernel, report".
- All three GPU kernels rewritten to **grid-stride loop**: launch a fixed
  `4 × SM_count` blocks (= 64 on GTX 1650, ~16,384 threads), then each thread runs
  a personal loop over its slice of `n_paths` accumulating in registers. The
  cuRAND state count is now O(total_threads), not O(N).
- `benchmarks/run_all.sh` `SIZES` extended to 25M and 50M (now possible — state
  memory used to cap us around ~50M on GTX 1650's 4 GB).

**Why it matters:**

| Cost component | Before refactor | After refactor | Ratio |
|---|---|---|---|
| cuRAND states allocated | 48 B × N | 48 B × ~16 K (fixed) | up to 3000× less at 50M |
| `init_rng_kernel` time @ N=1M | ~45 ms | **2.05 ms** | **22×** faster |
| `mc_shared_kernel` time @ N=1M | 1.06 ms | **0.08 ms** | **13×** faster |
| `mc_antithetic_kernel` time @ N=1M | 0.61 ms | **0.07 ms** | **8.7×** faster |
| Max feasible N (4 GB VRAM) | ~50 M | unbounded (RAM-side parses N) | — |

`mc_naive` kernel time stays the same (~3.3 ms at N=1M) because it is bottlenecked
by global `atomicAdd` per path, not by RNG state. That's the expected outcome — it
*is* the bad-pattern baseline.

**Pre/post timings (kernel only, N=1M, seed=1, GTX 1650):**

| Kernel | Pre (ms) | Post (ms) | Speedup |
|---|---|---|---|
| mc_naive | 3.37 | 3.31 | 1.0× (atomic-bound, as expected) |
| mc_shared | 1.06 | 0.08 | 13× |
| mc_antithetic | 0.61 | 0.07 | 8.7× |

**Why `mc_shared`/`mc_antithetic` got so much faster:**
- Block-reduce + global atomic happens once per block per launch, not per 256-thread
  group. With fixed 64 blocks at any N, there are 64 atomics — down from 3,907 at
  N=1M with the old design.
- Register accumulation across grid-stride iterations means each thread does its
  arithmetic at full speed without writing to memory until the reduction.

**New benchmark range (N=50M, seed=1):**

| Kernel | Init (ms) | Kernel (ms) | Throughput |
|---|---|---|---|
| mc_naive | 2.13 | 165.01 | 303 M paths/sec |
| mc_shared | 1.90 | **1.01** | **49.5 G paths/sec** |
| mc_antithetic | 1.90 | **0.68** | **73.5 G paths/sec** (per effective-path) |

The 73 G paths/sec @ antithetic is approaching the GTX 1650's ~2.9 TFLOPS peak with
~22 FP ops per effective-path — i.e. the kernel is now arithmetic-bound on a
consumer-tier card. Same workload on T4 / A100 should scale roughly with TFLOPS.

**Validation:** All four implementations still pass `tests/validate.py` at N=1M.
Estimates differ slightly from the pre-refactor run because the path-to-state
mapping changed (each thread now consumes a strided sequence rather than a
contiguous chunk), but errors remain within the same ~1-sigma envelope.

### 2026-05-18 — Re-profile after grid-stride refactor

Old `.ncu-rep` files were collected pre-refactor at N=1M; they were overwritten
with new runs at N=10M. The qualitative picture changed substantially.

**New metrics (N=10M, GTX 1650):**

| Kernel | Duration | Achieved Occ. | Compute Tput | DRAM Tput | L1 Tput | Regs |
|---|---|---|---|---|---|---|
| `mc_init_rng_kernel` | **680 µs** | 63.3% | 76.5% | 2.7% | 92.2% | 63 |
| `mc_naive_kernel` | 32.88 ms | 93.8% | **2.75%** | 0.03% | 5.5% | 26 |
| `mc_shared_kernel` | **273 µs** | 74.6% | **79.4%** | **3.5%** | 1.5% | 27 |
| `mc_antithetic_kernel` | **185 µs** | 72.8% | **76.4%** | **5.1%** | 2.2% | 32 |

**Comparison vs old profile (N=1M, pre-refactor):**

| Metric | mc_init_rng | mc_shared |
|---|---|---|
| Duration | 44.59 ms → 0.68 ms (**65× faster**) | 0.88 ms → 0.27 ms @ 10× more work |
| Compute Tput | 93.3% → 76.5% (same: compute-saturated) | **9.1% → 79.4%** |
| DRAM Tput | 2.9% → 2.7% (same: not memory-bound) | **85.2% → 3.5%** |
| Achieved Occ. | 92% → 63% (fewer waves) | 97% → 75% (fewer waves) |
| Grid Size | 3907 → 64 blocks | 3907 → 64 blocks |

**Key qualitative changes:**

1. **`mc_shared` flipped from memory-bound to compute-bound.** Pre-refactor DRAM
   throughput was 85% (reading 1M curandStates from global), Compute was 9%.
   Post-refactor DRAM is 3.5%, Compute is **79%**. Each thread now amortizes
   one state read across ~150 paths of arithmetic. This is the textbook outcome
   of a grid-stride loop and worth a poster slide on its own.

2. **`init_rng_kernel` is no longer the wall-clock dominator at typical N.**
   At 680 µs it is now < `mc_shared`+`mc_antithetic` combined at N=50M (1.7 ms).
   This **weakens the Milestone 8 (Philox) motivation**: the absolute saving
   from switching RNGs is now ~0.3 ms, not ~30 ms.

3. **`mc_naive` is still atomic-bound (Compute 2.75%, DRAM 0.03%).** Both
   metrics are near zero — the SMs are idle ~97% of the time waiting on global
   atomicAdd serialization. The new grid-stride design doesn't change this
   because the per-path atomic is intrinsic to the kernel. It is exactly the
   bad baseline we want it to be.

4. **Occupancy dropped (92→63 / 97→75)** because we're now launching only 64
   blocks (4 per SM × 16 SMs), giving 1 wave per SM. Pre-refactor we had 61
   waves/SM at N=1M. Lower occupancy hurts in theory, but here it doesn't:
   each thread does so much more work that the SM stays busy via instruction-
   level parallelism. The compute throughput numbers confirm this.

**Implication for next milestones:**
- Milestone 8 (XORWOW vs Philox) becomes a *characterization* exercise rather
  than an optimization. Still worth a poster figure, but it won't dominate the
  speedup story.
- Milestone 9 (CUDA streams) similarly less critical — init is now tiny.
- Milestone 10 (Asian options) becomes relatively more interesting: it would
  exercise the `n_steps` parameter (currently dead code) and prove the framework
  generalizes to path-dependent options. Higher poster value.

### 2026-05-18 — Asian options extension (Milestone 10, Hope-to-Achieve #1)

**What was added:**
- `src/gpu/mc_asian.cu` — arithmetic-average Asian call/put, multi-step GBM with
  inner step loop, uses the existing `mc_runner.cuh` harness and the same
  block-reduce reduction as `mc_shared`. The `n_steps` CLI flag (until now dead
  code) finally drives the inner loop.
- `src/cpu/mc_cpu_asian.c` — matching single-thread reference baseline.
- `mc_runner.cuh` now precomputes per-`dt` drift/diffuse constants in addition
  to the European single-big-step constants, and `mc_report()` gained a
  `has_analytical` flag so the European Black-Scholes reference is suppressed
  for path-dependent runs (it is meaningless there).
- `tests/validate.py` now also exercises `mc_asian` at the default `--steps 1`,
  where the payoff collapses to the European one and the analytical reference
  is valid. PASS.
- Makefile builds both new targets via `make cpu` and `make gpu`.

**Why `n_steps == 1` is a free sanity check:** When `dt = T`, the per-step
drift/diffuse constants equal the single-big-step versions, the path consists of
one observation, and the arithmetic mean equals `S_T` — so the Asian payoff
becomes the European payoff. All five binaries now reproduce ~10.4506 at the
default parameters.

**Results @ N=1M, S0=K=100, r=0.05, σ=0.2, T=1.0, steps=252 (daily), seed=42:**

| Runtime | Time | Price (call) | 95% CI half-width |
|---|---|---|---|
| `mc_cpu_asian` | 5617 ms | 5.778 | ±0.0157 |
| `mc_asian` (GPU) | 6.76 ms (kernel) | 5.770 | ±0.0157 |

- **Prices agree to 0.008 — well within either CI**, confirming CPU↔GPU
  consistency for the path-dependent case.
- **Speedup at same workload: 831×** (5617 ms / 6.76 ms). This is the *real*
  speedup story for the poster — 252× more arithmetic per path than European,
  yet the GPU still finishes ~2 orders of magnitude faster than a profitable
  C-with-O3 baseline.
- Theory check: arithmetic Asian price ≈ 5.78 < European 10.45 (averaging
  dampens variance → cheaper option). Matches Glasserman §3.6 reference values
  for these parameters.

**Scaling to N=10M, steps=252:**

| Runtime | Kernel time | Throughput | Price |
|---|---|---|---|
| `mc_asian` (GPU) | 66.5 ms | 150 M paths/sec | 5.782 ± 0.005 |
| `mc_cpu_asian` (extrapolated) | ~56 sec | 0.18 M paths/sec | — |

The GPU absorbs a 10× larger problem in ~10× the time (good linear scaling),
while extrapolating CPU to 10M paths gives ~56 seconds — i.e. an implied
**~840× speedup** at the project's headline problem size.

**What this proves for the poster:**
1. The grid-stride + block-reduce harness generalizes cleanly — `mc_asian.cu` is
   only ~40 lines on top of the shared harness and reused the reduction
   primitive verbatim.
2. The compute-bound regime persists even with 252× more arithmetic per path —
   the kernel is doing useful FMA / SFU work, not waiting on memory.
3. Speedup over CPU grows with workload complexity. European is ~100-200×,
   Asian (252 steps) is ~830×. The CPU pays the full per-step cost
   sequentially; the GPU just adds more iterations to its grid-stride loop.
