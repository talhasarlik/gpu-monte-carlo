"""
Generate fig_roofline.pdf / .png — the roofline analysis for the final report.

Plots arithmetic intensity (FLOPs/byte of DRAM traffic) vs achieved throughput
(GFLOPS) for the three GPU kernels, against the GTX 1650 compute and DRAM
ceilings. Numbers are derived from the Nsight Compute summary files in
results/ (post-refactor, N=10M). The hardware peaks are pinned at the observed
clock frequencies reported by Nsight, not vendor boost specs.

Style matches poster/figures/make_figures.py (serif fonts, colorblind palette).

Run from report/figures/ with the project root two levels up:
    python make_roofline.py
"""

from pathlib import Path
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
# Hardware ceilings — GTX 1650 (Turing TU117) at observed clocks
# --------------------------------------------------------------------------- #
# Observed in the Nsight Compute summaries:
#   SM Frequency   = 1.39 GHz   (not the marketing boost of 1.665 GHz)
#   DRAM Frequency = 3.99 GHz   (effective 7.98 Gbps per pin)
# Peak FP32 = 14 SMs * 64 cores/SM * 2 (FMA) * 1.39 GHz = 2,490 GFLOPS
# Peak DRAM = 128-bit bus * 7.98 Gbps / 8 bits/byte    =   128 GB/s
PEAK_GFLOPS = 2490.0
PEAK_BW_GBPS = 128.0
RIDGE_AI = PEAK_GFLOPS / PEAK_BW_GBPS  # ~19.5 FLOPS/byte

# --------------------------------------------------------------------------- #
# Kernel achievements — from results/*_summary.txt (N=10M, post-refactor)
# --------------------------------------------------------------------------- #
# Compute (SM) Throughput % and DRAM Throughput % are direct Nsight metrics.
# We translate them through the peaks above to get achieved GFLOPS / GB/s,
# then derive the empirical arithmetic intensity (AI = GFLOPS / GB/s).
KERNELS = [
    {
        "name":    "mc_naive",
        "label":   "GPU naive (atomicAdd per path)",
        "color":   "#D55E00",
        "marker":  "o",
        "compute_pct": 2.75,
        "dram_pct":    0.03,
    },
    {
        "name":    "mc_shared",
        "label":   "GPU shared (block-reduce)",
        "color":   "#0072B2",
        "marker":  "^",
        "compute_pct": 79.41,
        "dram_pct":    3.46,
    },
    {
        "name":    "mc_antithetic",
        "label":   "GPU antithetic",
        "color":   "#009E73",
        "marker":  "D",
        "compute_pct": 76.37,
        "dram_pct":    5.11,
    },
]

for k in KERNELS:
    k["achieved_gflops"] = (k["compute_pct"] / 100.0) * PEAK_GFLOPS
    k["achieved_gbps"]   = (k["dram_pct"]    / 100.0) * PEAK_BW_GBPS
    k["arith_intensity"] = k["achieved_gflops"] / k["achieved_gbps"]
    # Fraction of compute ceiling actually used
    k["pct_of_peak"]     = k["compute_pct"]

# --------------------------------------------------------------------------- #
# Plot style — match poster/figures/make_figures.py
# --------------------------------------------------------------------------- #
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         15,
    "axes.titlesize":    17,
    "axes.labelsize":    16,
    "xtick.labelsize":   13,
    "ytick.labelsize":   13,
    "legend.fontsize":   12,
    "figure.dpi":        150,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.18,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "lines.linewidth":   2.2,
    "lines.markersize":  11,
})

# --------------------------------------------------------------------------- #
# Build the plot
# --------------------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=(9.5, 6.5))

# X range covers all kernel AIs plus the ridge with margin
ai_grid = np.logspace(-1, 4.5, 400)

# Memory ceiling: y = bw * x (slope on log-log)
mem_line = PEAK_BW_GBPS * ai_grid
# Compute ceiling: horizontal at PEAK_GFLOPS
compute_line = np.full_like(ai_grid, PEAK_GFLOPS)
# Effective roof = min of the two
roof = np.minimum(mem_line, compute_line)

# Shade the unattainable region above the roof
ax.fill_between(ai_grid, roof, PEAK_GFLOPS * 10, color="#cccccc", alpha=0.35,
                label="_unattainable", zorder=0)

# Memory-bound and compute-bound region tints
ax.axvspan(ai_grid[0], RIDGE_AI, color="#0072B2", alpha=0.06, zorder=0)
ax.axvspan(RIDGE_AI, ai_grid[-1], color="#009E73", alpha=0.06, zorder=0)

# Ceilings
ax.plot(ai_grid, mem_line, color="#444444", linestyle="--", linewidth=1.8,
        label=f"DRAM bandwidth ceiling ({PEAK_BW_GBPS:.0f} GB/s)")
ax.plot(ai_grid, compute_line, color="#444444", linestyle="-", linewidth=1.8,
        label=f"FP32 compute ceiling ({PEAK_GFLOPS:.0f} GFLOPS)")
# Ridge marker
ax.axvline(RIDGE_AI, color="#888888", linestyle=":", linewidth=1.2, alpha=0.7)
ax.annotate(f"ridge\nAI = {RIDGE_AI:.1f}",
            xy=(RIDGE_AI, 18), xytext=(RIDGE_AI * 1.25, 13),
            fontsize=11, color="#555555", ha="left")

# Kernel points (legend carries the labels; annotate each with a leader arrow
# at a hand-chosen offset to avoid overlap).
LABEL_OFFSETS = {
    "mc_naive":      (60, 30),       # mc_naive sits low-right; label up-right
    "mc_shared":     (-110, 35),     # shared sits high; label up-left
    "mc_antithetic": (-50, -55),     # antithetic close to shared; label down-left
}
for k in KERNELS:
    ax.scatter([k["arith_intensity"]], [k["achieved_gflops"]],
               s=210, marker=k["marker"], color=k["color"],
               edgecolor="black", linewidth=1.2, zorder=5,
               label=k["label"] +
                     f"  ({k['pct_of_peak']:.1f}% of peak FP32)")
    dx, dy = LABEL_OFFSETS[k["name"]]
    ax.annotate(k["name"],
                xy=(k["arith_intensity"], k["achieved_gflops"]),
                xytext=(dx, dy), textcoords="offset points",
                fontsize=12, color=k["color"], fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=k["color"],
                                lw=1.0, alpha=0.8))

# Region labels
ax.text(0.4, 2200, "memory-bound\nregion",
        fontsize=12, color="#0072B2", ha="center", alpha=0.85)
ax.text(2500, 20, "compute-bound\nregion",
        fontsize=12, color="#009E73", ha="center", alpha=0.85)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1e-1, 1e4)
ax.set_ylim(10, 10000)
ax.set_xlabel(r"Arithmetic intensity (FLOPs / byte of DRAM)")
ax.set_ylabel(r"Achieved throughput (GFLOPS)")
ax.set_title("Roofline analysis — GTX 1650 at observed clocks")
ax.legend(loc="lower right", framealpha=0.92)

fig.tight_layout()
fig.savefig(HERE / "fig_roofline.pdf")
fig.savefig(HERE / "fig_roofline.png", dpi=200)
fig.savefig(HERE / "fig_roofline.svg")

# --------------------------------------------------------------------------- #
# Console summary for the report writer
# --------------------------------------------------------------------------- #
print(f"Peak compute : {PEAK_GFLOPS:>7.1f} GFLOPS")
print(f"Peak DRAM    : {PEAK_BW_GBPS:>7.1f} GB/s")
print(f"Ridge point  : {RIDGE_AI:>7.2f} FLOPS/byte")
print()
print(f"{'kernel':<16} {'AI':>10} {'GFLOPS':>10} {'GB/s':>10} {'% peak':>10}")
for k in KERNELS:
    print(f"{k['name']:<16} {k['arith_intensity']:>10.1f} "
          f"{k['achieved_gflops']:>10.1f} {k['achieved_gbps']:>10.2f} "
          f"{k['pct_of_peak']:>9.1f}%")
print()
print("Wrote fig_roofline.pdf and fig_roofline.png")
