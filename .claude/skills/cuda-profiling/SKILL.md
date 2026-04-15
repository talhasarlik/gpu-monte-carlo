---
name: cuda-profiling
description: Reference for profiling CUDA kernels with Nsight Compute and Nsight Systems. Use when the user wants to profile a kernel, set up a benchmark, interpret profiler metrics, capture a roofline analysis, or troubleshoot why a kernel is slow. Triggers include "profile this", "run nsight", "what metrics matter", "ncu command", or any mention of profiling/benchmarking CUDA code.
---

# CUDA Profiling Reference

This skill is the working manual for profiling our Monte Carlo kernels. It is opinionated for *this* project — not a general Nsight tutorial.

## Two Tools, Different Jobs

| Tool | Purpose | Typical Output |
|---|---|---|
| **Nsight Compute** (`ncu`) | Per-kernel deep dive: occupancy, memory throughput, warp efficiency | `.ncu-rep` or CSV |
| **Nsight Systems** (`nsys`) | System timeline: kernel ordering, stream overlap, host-device transfer | `.nsys-rep` |

Use `ncu` when asking *"why is this kernel slow?"*. Use `nsys` when asking *"are we using the GPU efficiently overall?"* (e.g., is there idle time between kernels, is `cudaMemcpy` blocking).

## Setup Check (Run First)

```bash
nvcc --version              # CUDA toolkit version
ncu --version               # Nsight Compute version
nsys --version              # Nsight Systems version
nvidia-smi                  # Which GPU, driver version
```

On Colab: ncu and nsys are usually preinstalled. If not:
```bash
apt-get install -y nsight-compute nsight-systems
```

## Nsight Compute — Common Recipes

### Recipe 1: Quick health check on a kernel
```bash
ncu --print-summary per-kernel ./build/mc_naive --paths 1000000
```
Gives one-line summary per kernel: time, throughput, occupancy. First thing to run.

### Recipe 2: Full report (use this most often)
```bash
ncu --set full -o results/mc_naive_full ./build/mc_naive --paths 10000000
```
Open `results/mc_naive_full.ncu-rep` in Nsight Compute GUI. This collects the standard metric set.

### Recipe 3: Roofline analysis
```bash
ncu --set roofline -o results/mc_naive_roofline ./build/mc_naive --paths 10000000
```
Plots arithmetic intensity vs. throughput against device peaks. Tells you immediately if you're memory- or compute-bound.

### Recipe 4: Specific metrics, CSV output (for plotting)
```bash
ncu --csv \
    --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
smsp__thread_inst_executed_per_inst_executed.ratio \
    ./build/mc_naive --paths 10000000 > results/mc_naive_metrics.csv
```

### Recipe 5: Filter by kernel name (for runs with multiple kernels)
```bash
ncu --kernel-name regex:mc_.*_kernel --set full ./build/mc_naive --paths 10000000
```

## Nsight Systems — Common Recipes

### Recipe 1: Timeline trace
```bash
nsys profile --stats=true -o results/mc_naive_timeline ./build/mc_naive --paths 10000000
```
Open the `.nsys-rep` in Nsight Systems GUI to see CPU/GPU timeline.

### Recipe 2: Capture only CUDA APIs (smaller file)
```bash
nsys profile --trace=cuda,nvtx -o results/mc_naive ./build/mc_naive --paths 10000000
```

### Recipe 3: Stream overlap analysis
```bash
nsys profile --trace=cuda --gpu-metrics-device=all -o results/streams_test ./build/mc_streams
```
Critical for the "Hope to Achieve" CUDA streams part of the project.

## Metrics That Matter for Monte Carlo

These are the metrics we report in the final analysis. Memorize what each one means.

| Metric | What it tells you | Target |
|---|---|---|
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | Compute pipeline utilization | >70% (compute-bound is fine here) |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | Memory bandwidth utilization | <50% (we're not memory-bound) |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | Achieved occupancy | >50% on T4, >40% on A100 |
| `smsp__thread_inst_executed_per_inst_executed.ratio` | Active threads per warp | Close to 32 (no divergence) |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` | L1 cache sector loads | Lower is better (fewer wasted bytes) |
| `smsp__inst_executed_op_global_atom.sum` | Global atomic operations | Should be near zero in optimized version |

## Build with Profiling-Friendly Flags

When profiling, build with debug line info so metrics map back to source:
```bash
nvcc -O3 -arch=sm_75 -lineinfo -o build/mc_naive src/gpu/mc_naive.cu
```
**Do not use `-g` for profiling** — it disables many optimizations and skews numbers.

## Interpreting "Why Is My Kernel Slow?"

Decision tree:

1. Run `ncu --print-summary`. Note: kernel time, achieved occupancy.
2. If **occupancy < 30%**: the kernel doesn't have enough parallelism to hide latency.
   - Check register usage: `nvcc --ptxas-options=-v` → if registers > 64, simplify the kernel or split it.
   - Check shared memory per block: too much shared memory limits blocks per SM.
   - Try smaller block size (128 instead of 256/512).
3. If **occupancy is fine but throughput is low**: bandwidth issue.
   - Check coalescing: `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct` should be ~100%.
   - If access is strided, restructure data layout.
4. If **everything looks fine but kernel is still slow**: probably cuRAND.
   - cuRAND XORWOW state init is expensive. Cache states across kernel launches.
   - Try Philox4_32_10 — usually faster.

## Variance & Reproducibility

GPU benchmarks are noisy. Always:
- **Warm up** the GPU with a throwaway kernel run before measuring.
- Run 5–10 times, report **median**, not mean.
- Pin the GPU clock if possible (Colab usually doesn't allow this).
- Run in isolation (no other notebook cells consuming GPU).

In our `benchmarks/run_all.sh`, we do this with:
```bash
for i in $(seq 1 10); do
  ./build/mc_naive --paths $N --seed $i
done | sort -n | awk 'NR==5 || NR==6 {sum+=$1; n++} END {print sum/n}'
```

## Debugging vs Profiling

| Goal | Tool |
|---|---|
| Find a memory bug (out-of-bounds) | `compute-sanitizer` (replaces cuda-memcheck) |
| Find race conditions | `compute-sanitizer --tool racecheck` |
| Find uninitialized memory | `compute-sanitizer --tool initcheck` |
| Step through kernel | `cuda-gdb` |
| Profile performance | `ncu` / `nsys` (this skill) |

## What to Save in `results/`

For every kernel version, store:
- `<version>_summary.txt` — `ncu --print-summary` output
- `<version>_full.ncu-rep` — full Nsight Compute report
- `<version>_timeline.nsys-rep` — Nsight Systems timeline
- `<version>_metrics.csv` — raw metric CSV for plotting
- `<version>_bench.csv` — runtime/throughput across N

This makes the report-writing painless at the end.
