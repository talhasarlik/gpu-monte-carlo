# CLAUDE.md

This file gives Claude Code the context it needs to help us develop this project efficiently.

## Project Summary

**Title:** GPU-Accelerated Monte Carlo Simulation for Financial Option Pricing
**Course:** BLG 562E — Parallel Computing for GPUs using CUDA (Spring 2026, ITU)
**Instructor:** Asst. Prof. Ayse Yilmazer

We are building a CUDA-based Monte Carlo simulation engine for pricing European call/put options
using the Black-Scholes model. The project emphasizes **performance analysis**: we implement
multiple versions (naive → shared-memory optimized → antithetic variates) and benchmark each
against a CPU baseline, profiling with Nsight Compute/Systems.

## Team

| Member | Role | Strengths |
|---|---|---|
| Talha Sarlık (704241052) | Data scientist at invent.ai | Python, data analysis, ML, benchmarking, visualization, statistical validation |
| Mehmet Bedirhan Önder (704251026) | SW engineer at Analog Devices | C, low-level optimization, embedded, kernel-level CUDA, memory hierarchy |

**Work split guideline:** Talha owns the CPU baseline, validation, benchmarking harness, plotting,
and the analysis/report. Bedirhan owns the CUDA kernels, cuRAND state management, reduction
optimization, and Nsight profiling. Both collaborate on algorithm design and the poster.

## Development Environment

- **Primary dev platform:** Google Colab (free T4, Pro A100). See `notebooks/colab_dev.ipynb`.
- **Local editing:** VS Code + Claude Code on our own machines.
- **GPU:** NVIDIA T4 (compute capability 7.5) or A100 (8.0). Build with `-arch=sm_75` by default.
- **CUDA Toolkit:** 12.x (whatever Colab provides — check with `nvcc --version`).
- **CPU baseline compiler:** gcc / clang with `-O3 -march=native`.

## Repository Layout

```
gpu-monte-carlo/
├── CLAUDE.md                  # This file
├── README.md                  # Human-facing overview
├── Makefile                   # Build system (all/cpu/gpu/clean/bench)
├── .claude/
│   ├── agents/                # Custom Claude Code agents
│   │   ├── cuda-kernel-reviewer.md
│   │   └── performance-analyzer.md
│   └── skills/                # Custom skills
│       ├── cuda-profiling/
│       ├── monte-carlo-theory/
│       └── colab-workflow/
├── docs/
│   ├── proposal.pdf           # Submitted proposal (April 15)
│   ├── plan.md                # Development roadmap
│   └── references.md          # Papers, docs, links
├── src/
│   ├── cpu/
│   │   └── mc_cpu.c           # Single-threaded CPU baseline
│   ├── gpu/
│   │   ├── mc_naive.cu        # v1: one thread per path, global memory
│   │   ├── mc_shared.cu       # v2: shared-memory reduction
│   │   ├── mc_antithetic.cu   # v3: variance reduction
│   │   └── reduction.cuh      # Reusable reduction primitives
│   └── common/
│       ├── black_scholes.h    # Analytical Black-Scholes formula (ground truth)
│       ├── params.h           # Shared struct for option parameters
│       └── timer.h            # Wall-clock + CUDA event timers
├── benchmarks/
│   ├── run_all.sh             # Sweep path counts, save CSV
│   └── plot.py                # Matplotlib plots from CSV
├── tests/
│   └── validate.py            # Diff GPU output vs analytical formula
├── notebooks/
│   └── colab_dev.ipynb        # Colab workflow notebook
└── results/                   # Output CSVs, Nsight reports, plots
```

## Build & Run

```bash
# Build everything
make

# Build specific targets
make cpu                        # Just the CPU baseline
make gpu                        # All GPU versions
make mc_naive                   # Single kernel version

# Run sanity check (small N, compare vs analytical)
./build/mc_cpu    --paths 100000 --S0 100 --K 100 --r 0.05 --sigma 0.2 --T 1.0
./build/mc_naive  --paths 1000000 --S0 100 --K 100 --r 0.05 --sigma 0.2 --T 1.0

# Full benchmark sweep
bash benchmarks/run_all.sh

# Profile with Nsight Compute (per kernel)
ncu --set full -o results/mc_naive ./build/mc_naive --paths 10000000

# Profile with Nsight Systems (timeline view)
nsys profile -o results/mc_naive_timeline ./build/mc_naive --paths 10000000

# Validate correctness
python tests/validate.py
```

## Key Design Decisions

1. **Float32 vs Float64.** Start with `float` (faster on GPU, aligns with Colab T4 strengths).
   Add an optional `double` build flag later if precision analysis is needed.
2. **One thread = one full path.** Baseline design. Each thread loops over time steps. Good for
   European options (only terminal price matters). For Asian options we'll re-evaluate.
3. **cuRAND device API, not host API.** Each thread owns a local `curandState`. States are
   initialized once in a separate kernel and stored in global memory.
4. **Reduction strategy.** Partial reduction in shared memory per block → single atomicAdd
   on a global accumulator. Compare against warp-shuffle (`__shfl_down_sync`) tree reduction.
5. **Input parameters.** No real market data needed for the core project. We use synthetic
   parameters (S0=100, K=100, r=0.05, sigma=0.2, T=1.0) where the analytical Black-Scholes
   formula gives an exact answer (≈10.4506 for these values). This is our ground truth.

## Data

We **do not need real market data** to complete this project — Monte Carlo validation uses
the analytical Black-Scholes formula as ground truth. The algorithm's correctness is verified
by convergence to this known value.

**If we want a "real-world demo"** for the poster (optional, Hope to Achieve):
- **Yahoo Finance** via `yfinance` Python package — free, historical stock prices.
- **Alpha Vantage** — free tier, requires API key.
- **CBOE DataShop** — paid, actual option contract history.

For the poster, we'd use `yfinance` to pull a real stock's price and historical volatility,
feed those parameters to our GPU engine, and compare to what the market is actually pricing
the option at. This is aesthetics, not core to the grade.

## Coding Conventions

- **CUDA files:** `.cu` for kernels that are compiled by nvcc, `.cuh` for device-side headers.
- **Kernel naming:** `mc_<version>_kernel` (e.g., `mc_naive_kernel`). Launcher functions
  drop the `_kernel` suffix.
- **Error checking:** Every CUDA API call wrapped in `CUDA_CHECK(...)` macro. Defined in
  `src/common/timer.h`. Never skip this.
- **No magic numbers in kernels.** Use `#define` or `constexpr` with descriptive names.
- **Timing:** Use CUDA events (`cudaEventRecord`) for kernel timing, `clock_gettime` for
  wall-clock. Wrappers in `src/common/timer.h`.
- **Comments in Turkish or English — team's choice per file**, but **public API and CLAUDE-facing
  comments in English**.

## Development Workflow

1. Design an experiment → write down the question ("does shared-memory reduction beat atomics?").
2. Implement in smallest possible kernel.
3. Run validation test: does it converge to the analytical price?
4. Run benchmark sweep, save CSV.
5. Profile with Nsight Compute → take screenshot/export.
6. Write a 1-paragraph observation in `docs/plan.md` log section.
7. Commit.

## Important Don'ts

- **Don't optimize prematurely.** Get a correct naive version first. Every optimization must
  be justified by a profiler number.
- **Don't reuse Python RNG across threads.** Each thread needs its own `curandState`.
- **Don't sum with `atomicAdd(float*)` at the leaf level for millions of values** — it serializes.
- **Don't trust a single benchmark run.** Report median of 5+ runs; discard the first (warmup).
- **Don't copy code from papers/Stack Overflow without attribution.** The proposal explicitly
  states what we reused vs. wrote ourselves. Keep a `docs/references.md` up to date.

## Slash Commands Worth Knowing

When working with Claude Code in this repo, useful custom capabilities live in `.claude/`:

- **Agent: `cuda-kernel-reviewer`** — point it at a `.cu` file and it reviews for memory
  access patterns, warp divergence, occupancy, race conditions.
- **Agent: `performance-analyzer`** — feed it Nsight Compute output (text/CSV) and it
  explains bottlenecks in plain language.
- **Skill: `cuda-profiling`** — reference guide for Nsight Compute/Systems commands and
  metric interpretation.
- **Skill: `monte-carlo-theory`** — Black-Scholes math reference, variance reduction theory.
- **Skill: `colab-workflow`** — how to develop efficiently on Colab (uploading sources,
  running nvcc, downloading results).

## References

See `docs/references.md` for the full list. Top three:
1. NVIDIA cuRAND Library User Guide.
2. NVIDIA CUDA C Best Practices Guide.
3. P. Glasserman, *Monte Carlo Methods in Financial Engineering*, Springer 2003.

## Poster Session Target

End of semester poster session. We aim to demonstrate:
- ~200× speedup over single-threaded CPU at 10M paths.
- Clean convergence plots (error vs N, naive vs antithetic).
- Live interactive demo: user changes N, price + runtime update instantly.
- Nsight timeline screenshots showing stream overlap (if Hope-to-Achieve #2 works out).
