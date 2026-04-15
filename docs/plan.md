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
| 5 | Benchmark sweep script + plotting harness | Talha | todo | |
| 6 | Validation tests pass for all 3 GPU versions | Talha | done | 2026-04-16 |
| 7 | Nsight Compute profiling for each kernel | both | todo | |
| 8 | cuRAND generator comparison (XORWOW vs Philox) | Bedirhan | todo | |
| 9 | (Hope) CUDA streams overlap | Bedirhan | todo | |
| 10 | (Hope) Asian options extension | Talha | todo | |
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
