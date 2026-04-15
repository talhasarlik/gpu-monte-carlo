# GPU-Accelerated Monte Carlo Option Pricing

Final project for **BLG 562E — Parallel Computing for GPUs using CUDA**, ITU Spring 2026.

**Team:** Talha Sarlık (704241052) · Mehmet Bedirhan Önder (704251026)

## What this is

A CUDA implementation of Monte Carlo simulation for pricing European options under the
Black-Scholes model. We progressively optimize three GPU versions and benchmark them
against a CPU baseline:

| Version | Strategy | Expected Win |
|---|---|---|
| `mc_cpu` | Single-threaded C reference | baseline |
| `mc_naive` | One thread per path, global atomic reduction | first speedup |
| `mc_shared` | Block-local reduction with warp shuffle + one atomic per block | reduce atomic contention |
| `mc_antithetic` | mc_shared + antithetic variates (variance reduction) | better convergence per FLOP |

Goal: ~200× speedup vs CPU at 10M+ paths, with detailed Nsight Compute analysis.

## Quick Start

```bash
# Build
make all

# Sanity check (small N)
./build/mc_cpu    --paths 100000
./build/mc_naive  --paths 1000000

# Full benchmark sweep
make bench

# Profile a kernel
ncu --set full -o results/mc_naive ./build/mc_naive --paths 10000000

# Validate against analytical Black-Scholes
make validate
```

For Colab-specific instructions see `notebooks/colab_dev.ipynb` and
`.claude/skills/colab-workflow/SKILL.md`.

## Repository Map

- `CLAUDE.md` — full project context for AI-assisted development.
- `src/cpu/` — single-threaded CPU baseline.
- `src/gpu/` — three CUDA kernel versions + reduction primitives.
- `src/common/` — shared headers (params, timer, analytical formula).
- `benchmarks/` — sweep script + plotting.
- `tests/` — correctness validation against closed-form Black-Scholes.
- `docs/` — proposal, plan, references.
- `.claude/` — agents and skills for Claude Code.
- `notebooks/` — Colab development notebook.
- `results/` — benchmark CSVs, Nsight reports, plots.

## Documentation

- **Proposal:** [`docs/proposal.pdf`](docs/proposal.pdf)
- **Development plan & log:** [`docs/plan.md`](docs/plan.md)
- **References:** [`docs/references.md`](docs/references.md)

## Status

Started: April 15, 2026. Poster session: end of semester.

See `docs/plan.md` for the running development log.
