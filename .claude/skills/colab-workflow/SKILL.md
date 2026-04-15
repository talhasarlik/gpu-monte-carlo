---
name: colab-workflow
description: Practical guide for developing this CUDA project on Google Colab — connecting Drive, syncing source code with the local repo, compiling with nvcc, running benchmarks, downloading results, and avoiding common Colab pitfalls (session timeouts, GPU type changes, ephemeral filesystem). Use when the user asks about Colab setup, "how do I run this on Colab", "my Colab session died", or anything about syncing code between local and Colab.
---

# Google Colab Workflow

This skill covers how we develop and run the CUDA Monte Carlo project on Google Colab.

## Why Colab?

- Free GPU (T4) — sufficient for development and most benchmarks.
- Pro/Pro+ gives A100 — needed for the high-end performance numbers in the poster.
- No local CUDA toolkit setup required.
- Same nvcc and Nsight versions as production NVIDIA cloud.

**Trade-off:** Colab sessions are ephemeral (~12h on free, longer on Pro). Filesystem resets when the runtime disconnects. Always sync source and results to Google Drive or GitHub.

## One-Time Setup

### 1. Pick the right runtime

`Runtime → Change runtime type → Hardware accelerator: GPU → GPU type: T4 (or A100 if Pro)`

### 2. Verify the GPU
```python
!nvidia-smi
!nvcc --version
```

### 3. Mount Google Drive (for persistence)
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 4. Clone the repo (recommended over Drive)
```python
!git clone https://github.com/<your-org>/gpu-monte-carlo.git /content/project
%cd /content/project
```

If the repo is private, use a personal access token:
```python
!git clone https://<TOKEN>@github.com/<your-org>/gpu-monte-carlo.git /content/project
```

## Daily Workflow

### Pull latest changes
```python
%cd /content/project
!git pull
```

### Build
```python
!make clean && make all
```

### Run a quick sanity check
```python
!./build/mc_cpu --paths 100000 --S0 100 --K 100 --r 0.05 --sigma 0.2 --T 1.0
!./build/mc_naive --paths 1000000 --S0 100 --K 100 --r 0.05 --sigma 0.2 --T 1.0
```

### Run full benchmark sweep
```python
!bash benchmarks/run_all.sh
```

### Profile with Nsight Compute
```python
!ncu --set full -o results/mc_naive_full ./build/mc_naive --paths 10000000
```

### Download results
```python
from google.colab import files
files.download('results/mc_naive_full.ncu-rep')
```
Or copy to mounted Drive:
```python
!cp results/*.ncu-rep /content/drive/MyDrive/cuda-project/results/
```

### Commit and push back
```python
!git config user.email "talha@invent.ai"
!git config user.name "Talha Sarlık"
!git add results/ docs/plan.md
!git commit -m "Add benchmark results from T4 run"
!git push
```

## Useful Colab Cells (Boilerplate)

The notebook `notebooks/colab_dev.ipynb` has these as ready cells. Key ones:

### Cell: GPU info
```python
!nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv
```

### Cell: Build & test in one go
```python
!cd /content/project && make clean && make all && \
  ./build/mc_cpu --paths 100000 && \
  ./build/mc_naive --paths 1000000
```

### Cell: Sweep N and capture CSV
```python
import subprocess, csv
results = []
for N in [10000, 100000, 1000000, 10000000, 100000000]:
    out = subprocess.run(['./build/mc_naive', '--paths', str(N)],
                         capture_output=True, text=True)
    # parse output, append to results
    results.append({'N': N, 'output': out.stdout})

with open('results/sweep.csv', 'w') as f:
    w = csv.DictWriter(f, fieldnames=['N', 'output'])
    w.writeheader()
    w.writerows(results)
```

## Pitfalls to Avoid

### Pitfall 1: GPU type changes between sessions
You might compile for `sm_75` (T4) and then the next session gives you a P100 (`sm_60`). Always check `nvidia-smi` first, and rebuild if the architecture differs.

In the Makefile we set `-arch=sm_75` by default. Override:
```python
!make ARCH=sm_80 all   # for A100
```

### Pitfall 2: Filesystem resets
`/content/*` is **ephemeral**. When the runtime disconnects, everything except `/content/drive` is gone. **Always push results to Drive or git before closing**.

### Pitfall 3: Session timeout while benchmarking
Long benchmarks can hit the 12h limit. Strategies:
- Break sweeps into chunks; checkpoint to Drive between chunks.
- Use Colab Pro for longer sessions (~24h).
- Use `&` to background long runs and `nohup`-style logging.

### Pitfall 4: nvprof is deprecated
Old tutorials use `nvprof`. **Don't use it.** It doesn't support modern GPUs (Volta+). Use `ncu` (Nsight Compute) instead.

### Pitfall 5: Profiling permissions
`ncu` needs profiler counters enabled. On Colab they usually are, but if you see "ERR_NVGPUCTRPERM", try:
```bash
ncu --target-processes all ./build/mc_naive
```
If that fails, the GPU type doesn't allow profiling on this Colab tier — switch GPU types or use `nsys` (which needs fewer permissions).

### Pitfall 6: cuda-memcheck is deprecated
Use `compute-sanitizer` instead. Same UX:
```python
!compute-sanitizer ./build/mc_naive --paths 10000
```

## Recommended Project Layout on Colab

```
/content/
├── project/                    # Cloned repo (ephemeral)
│   ├── src/
│   ├── build/                  # Compiled binaries (ephemeral)
│   └── results/                # Local results before push (ephemeral)
└── drive/MyDrive/cuda-project/ # Persistent storage
    ├── results/                # Final benchmark outputs
    ├── ncu-reports/            # Saved Nsight Compute reports
    └── poster-assets/          # Plots, screenshots for poster
```

## Switching Between Local and Colab

- **Local (Windows/Linux with NVIDIA GPU):** Same Makefile works. Just need CUDA toolkit installed.
- **Local (no GPU):** Can still write/edit code and run CPU baseline. Use Colab for any GPU work.
- **Colab to local:** `git pull` brings code; results download via the `files.download()` snippet above.

## Quick Reference Commands

```bash
# GPU info
nvidia-smi
nvcc --version

# Build everything
make all

# Single CPU run
./build/mc_cpu --paths 100000

# Single GPU run
./build/mc_naive --paths 1000000

# Profile
ncu --set full -o results/profile.ncu-rep ./build/mc_naive --paths 10000000

# Validate correctness
python tests/validate.py

# Plot results
python benchmarks/plot.py
```
