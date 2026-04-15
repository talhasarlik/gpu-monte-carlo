"""
Plot benchmark results from results/benchmark.csv.

Produces:
  results/throughput.png  - throughput (M paths/sec) vs N, per kernel
  results/speedup.png     - speedup over CPU vs N, per GPU kernel
  results/convergence.png - error vs N, per kernel
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

CSV = Path("results/benchmark.csv")
if not CSV.exists():
    print(f"Missing {CSV}. Run `bash benchmarks/run_all.sh` first.", file=sys.stderr)
    sys.exit(1)

df = pd.read_csv(CSV)
df["kernel_ms"] = pd.to_numeric(df["kernel_ms"], errors="coerce")
df["abs_error"] = pd.to_numeric(df["abs_error"], errors="coerce")
df = df.dropna(subset=["kernel_ms"])

# Median over the 5 runs per (kernel, n_paths)
agg = (df.groupby(["kernel", "n_paths"])
         .agg(kernel_ms=("kernel_ms", "median"),
              abs_error=("abs_error", "median"))
         .reset_index())
agg["throughput"] = agg["n_paths"] / agg["kernel_ms"] / 1000.0  # M paths/sec

# Throughput plot
fig, ax = plt.subplots(figsize=(8, 5))
for k, sub in agg.groupby("kernel"):
    sub = sub.sort_values("n_paths")
    ax.loglog(sub["n_paths"], sub["throughput"], marker="o", label=k)
ax.set_xlabel("Number of paths (N)")
ax.set_ylabel("Throughput (M paths / sec)")
ax.set_title("Monte Carlo throughput vs problem size")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig("results/throughput.png", dpi=150)

# Speedup plot
cpu = agg[agg["kernel"] == "mc_cpu"].set_index("n_paths")["kernel_ms"]
sp = agg[agg["kernel"] != "mc_cpu"].copy()
sp["speedup"] = sp.apply(lambda r: cpu.get(r["n_paths"], np.nan) / r["kernel_ms"], axis=1)

fig, ax = plt.subplots(figsize=(8, 5))
for k, sub in sp.groupby("kernel"):
    sub = sub.sort_values("n_paths")
    ax.semilogx(sub["n_paths"], sub["speedup"], marker="o", label=k)
ax.set_xlabel("Number of paths (N)")
ax.set_ylabel("Speedup vs single-threaded CPU")
ax.set_title("GPU speedup over CPU baseline")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig("results/speedup.png", dpi=150)

# Convergence plot
fig, ax = plt.subplots(figsize=(8, 5))
for k, sub in agg.groupby("kernel"):
    sub = sub.sort_values("n_paths")
    ax.loglog(sub["n_paths"], sub["abs_error"], marker="o", label=k)
# Reference 1/sqrt(N) line
N = np.array(sorted(agg["n_paths"].unique()))
ax.loglog(N, 0.5 / np.sqrt(N), "k--", alpha=0.4, label="1/sqrt(N) reference")
ax.set_xlabel("Number of paths (N)")
ax.set_ylabel("|estimate - analytical|")
ax.set_title("Monte Carlo convergence")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig("results/convergence.png", dpi=150)

print("Wrote results/throughput.png, results/speedup.png, results/convergence.png")
