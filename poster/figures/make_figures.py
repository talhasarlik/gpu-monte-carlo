"""
Regenerate poster-quality figures from results/benchmark.csv and the on-disk
Nsight Compute summaries.

Outputs (vector PDF; serif fonts; colorblind-safe palette):
    fig_throughput.pdf       — paths/sec vs N, all 4 implementations, log-log
    fig_speedup.pdf          — speedup over CPU vs N, semi-log
    fig_convergence.pdf      — absolute error vs N with 1/sqrt(N) reference
    fig_nsight_metrics.pdf   — grouped bar chart of occupancy / DRAM / compute throughput
    fig_speedup_bars.pdf     — horizontal bars at N=50M, log x-axis

Run from poster/figures/ with the project root two levels up:
    python make_figures.py
"""

from pathlib import Path
import csv
from statistics import median

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

# Each figure is saved in three formats:
#   .pdf  — vector, for print / LaTeX
#   .png  — 200-dpi raster, for inline rendering in markdown viewers
#   .svg  — vector, for web / editing in Inkscape / GitHub previews
def save_both(fig, out_no_ext: Path):
    fig.savefig(out_no_ext.with_suffix(".pdf"))
    fig.savefig(out_no_ext.with_suffix(".png"), dpi=200)
    fig.savefig(out_no_ext.with_suffix(".svg"))

# --------------------------------------------------------------------------- #
# Project paths
# --------------------------------------------------------------------------- #

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
CSV_PATH = PROJECT_ROOT / "results" / "benchmark.csv"
DIAGRAMS_DIR = HERE.parent / "diagrams"   # architecture diagrams live here

# --------------------------------------------------------------------------- #
# Global matplotlib style
# --------------------------------------------------------------------------- #

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         16,
    "axes.titlesize":    18,
    "axes.labelsize":    17,
    "xtick.labelsize":   14,
    "ytick.labelsize":   14,
    "legend.fontsize":   14,
    "figure.dpi":        150,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.15,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "lines.linewidth":   2.2,
    "lines.markersize":  9,
})

# Colorblind-safe ordering: CPU=grey, naive=red, shared=blue, antithetic=green.
KERNEL_STYLE = {
    "mc_cpu":        {"label": "CPU (1 thread)",          "color": "#555555", "marker": "s"},
    "mc_naive":      {"label": "GPU naive (atomics)",     "color": "#D55E00", "marker": "o"},
    "mc_shared":     {"label": "GPU shared (block-reduce)", "color": "#0072B2", "marker": "^"},
    "mc_antithetic": {"label": "GPU antithetic",          "color": "#009E73", "marker": "D"},
}
KERNEL_ORDER = ["mc_cpu", "mc_naive", "mc_shared", "mc_antithetic"]

# --------------------------------------------------------------------------- #
# Load and aggregate benchmark.csv
# --------------------------------------------------------------------------- #

def load_csv(path: Path):
    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            kernel = r["kernel"].replace(".exe", "")
            rows.append({
                "kernel":     kernel,
                "n_paths":    int(r["n_paths"]),
                "kernel_ms":  float(r["kernel_ms"]),
                "price":      float(r["price"]),
                "abs_error":  float(r["abs_error"]),
            })
    return rows

def aggregate(rows):
    """{kernel: {n_paths: {ms, err, throughput}}} — medians of all runs."""
    buckets = {}
    for r in rows:
        buckets.setdefault(r["kernel"], {}).setdefault(r["n_paths"], []).append(r)
    agg = {}
    for k, byN in buckets.items():
        agg[k] = {}
        for N, samples in byN.items():
            ms_med = median(s["kernel_ms"] for s in samples)
            err_med = median(s["abs_error"] for s in samples)
            agg[k][N] = {
                "ms":         ms_med,
                "err":        err_med,
                "throughput": (N / (ms_med / 1000.0)) / 1e6,  # M paths/sec
            }
    return agg

# --------------------------------------------------------------------------- #
# Figure 1: Throughput (M paths/sec) vs N
# --------------------------------------------------------------------------- #

def fig_throughput(agg, out: Path):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for k in KERNEL_ORDER:
        Ns  = sorted(agg[k].keys())
        thp = [agg[k][N]["throughput"] for N in Ns]
        s = KERNEL_STYLE[k]
        ax.plot(Ns, thp, color=s["color"], marker=s["marker"], label=s["label"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of paths $N$")
    ax.set_ylabel("Throughput (M paths / sec)")
    ax.set_title("Monte Carlo throughput vs problem size")
    ax.legend(loc="lower right", frameon=True)
    save_both(fig, out.with_suffix(""))
    plt.close(fig)

# --------------------------------------------------------------------------- #
# Figure 2: Speedup over CPU
# --------------------------------------------------------------------------- #

def fig_speedup(agg, out: Path):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    cpu = agg["mc_cpu"]
    for k in ["mc_naive", "mc_shared", "mc_antithetic"]:
        Ns = sorted(set(agg[k]) & set(cpu))
        speedup = [cpu[N]["ms"] / agg[k][N]["ms"] for N in Ns]
        s = KERNEL_STYLE[k]
        ax.plot(Ns, speedup, color=s["color"], marker=s["marker"], label=s["label"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1.2, label="CPU baseline (1$\\times$)")
    ax.set_xlabel("Number of paths $N$")
    ax.set_ylabel("Speedup over CPU (kernel time)")
    ax.set_title("GPU speedup vs single-threaded CPU baseline")
    ax.legend(loc="lower right", frameon=True)
    save_both(fig, out.with_suffix(""))
    plt.close(fig)

# --------------------------------------------------------------------------- #
# Figure 3: Convergence (|error| vs N with 1/sqrt(N) reference)
# --------------------------------------------------------------------------- #

def fig_convergence(agg, out: Path):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for k in KERNEL_ORDER:
        Ns  = sorted(agg[k].keys())
        err = [agg[k][N]["err"] for N in Ns]
        s = KERNEL_STYLE[k]
        ax.plot(Ns, err, color=s["color"], marker=s["marker"], label=s["label"])

    Ns = sorted(agg["mc_cpu"].keys())
    e0 = agg["mc_cpu"][Ns[0]]["err"]
    ref = [e0 * (Ns[0] / N) ** 0.5 for N in Ns]
    ax.plot(Ns, ref, color="black", linestyle="--", linewidth=1.4,
            label=r"$\mathcal{O}(N^{-1/2})$ reference")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of paths $N$")
    ax.set_ylabel("Absolute error  $|\\hat{C} - C_\\mathrm{BS}|$")
    ax.set_title("Convergence to analytical Black-Scholes price")
    ax.legend(loc="upper right", frameon=True, ncol=1)
    save_both(fig, out.with_suffix(""))
    plt.close(fig)

# --------------------------------------------------------------------------- #
# Figure 4: Nsight metrics grouped bar chart
# Numbers transcribed from results/mc_*_summary.txt (N=10M profiling runs).
# --------------------------------------------------------------------------- #

NSIGHT = {
    "mc_naive":      {"occupancy": 93.79, "compute": 2.75,  "memory": 8.39},
    "mc_shared":     {"occupancy": 74.64, "compute": 79.41, "memory": 3.46},
    "mc_antithetic": {"occupancy": 72.78, "compute": 76.37, "memory": 5.11},
}

def fig_nsight(out: Path):
    metrics = ["occupancy", "compute", "memory"]
    metric_labels = ["Achieved Occupancy", "Compute (SM) Throughput", "Memory Throughput"]
    kernels = ["mc_naive", "mc_shared", "mc_antithetic"]

    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    bar_w = 0.26
    x = list(range(len(metrics)))
    for i, k in enumerate(kernels):
        vals = [NSIGHT[k][m] for m in metrics]
        positions = [xi + (i - 1) * bar_w for xi in x]
        s = KERNEL_STYLE[k]
        bars = ax.bar(positions, vals, bar_w, color=s["color"], label=s["label"],
                      edgecolor="black", linewidth=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}%",
                    ha="center", va="bottom", fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Percent of peak")
    ax.set_ylim(0, 110)
    ax.set_title("Nsight Compute metrics, $N = 10^7$ paths (GTX 1650)")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="y", alpha=0.3)
    ax.grid(axis="x", visible=False)
    save_both(fig, out.with_suffix(""))
    plt.close(fig)

# --------------------------------------------------------------------------- #
# Figure 5: Speedup horizontal bars at N=50M
# --------------------------------------------------------------------------- #

def fig_speedup_bars(agg, out: Path):
    N = 50_000_000
    cpu_ms = agg["mc_cpu"][N]["ms"]
    kernels = ["mc_naive", "mc_shared", "mc_antithetic"]
    labels  = [KERNEL_STYLE[k]["label"] for k in kernels]
    speedup = [cpu_ms / agg[k][N]["ms"] for k in kernels]
    colors  = [KERNEL_STYLE[k]["color"] for k in kernels]

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    bars = ax.barh(labels, speedup, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_xscale("log")
    ax.set_xlim(1, max(speedup) * 2.5)
    ax.axvline(1.0, color="black", linestyle=":", linewidth=1.2)
    ax.set_xlabel(r"Speedup over single-threaded CPU ($N = 5\times 10^7$)")
    ax.set_title("Headline speedup at the largest tested problem size")
    for b, v in zip(bars, speedup):
        ax.text(v * 1.05, b.get_y() + b.get_height() / 2, f"{v:,.0f}$\\times$",
                va="center", fontsize=14)
    ax.grid(axis="x", which="both", alpha=0.3)
    save_both(fig, out.with_suffix(""))
    plt.close(fig)

# --------------------------------------------------------------------------- #
# Diagram 1: Algorithm flow (host -> RNG init -> path loop -> reduce -> host)
# --------------------------------------------------------------------------- #

def _rounded_box(ax, xy, w, h, label, facecolor, edgecolor="black",
                 textcolor="black", fontsize=12, fontweight="normal"):
    from matplotlib.patches import FancyBboxPatch
    x, y = xy
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.18",
        linewidth=1.4, facecolor=facecolor, edgecolor=edgecolor,
    )
    ax.add_patch(box)
    ax.text(x, y, label, ha="center", va="center",
            fontsize=fontsize, color=textcolor, fontweight=fontweight)

def _arrow(ax, xy_from, xy_to, color="black"):
    ax.annotate("", xy=xy_to, xytext=xy_from,
                arrowprops=dict(arrowstyle="-|>", lw=1.6, color=color,
                                mutation_scale=18))

HOST_FILL = "#FFE9CC"   # light orange
GPU_FILL  = "#D4EAD0"   # light green
NOTE_COL  = "#666666"

def fig_algorithm_flow(out: Path):
    fig, ax = plt.subplots(figsize=(8.5, 11.0))
    ax.set_xlim(0, 11); ax.set_ylim(0, 13.5)
    ax.set_aspect("equal"); ax.axis("off")

    # (centre_x, centre_y, width, height, label, fill)
    # gaps between consecutive boxes are ~0.7 units so arrows have room to live
    nodes = [
        (5.5, 12.4, 7.8, 1.1, "Host: parse params, allocate device buffers", HOST_FILL),
        (5.5, 10.5, 7.8, 1.4, "init_rng_kernel\n(cuRAND XORWOW state per thread)", GPU_FILL),
        (5.5,  7.9, 7.8, 2.3, "mc_*_kernel  —  grid-stride loop\n"
                              "$Z \\sim \\mathcal{N}(0,1)$\n"
                              "$S_T = S_0 \\cdot \\exp((r-\\sigma^2/2)T + \\sigma\\sqrt{T}\\,Z)$\n"
                              "$g = \\max(S_T - K, 0)$", GPU_FILL),
        (5.5,  5.0, 7.8, 1.4, "Block-level reduction\n(warp shuffle $\\rightarrow$ shared memory)", GPU_FILL),
        (5.5,  3.0, 7.8, 1.1, "One atomicAdd per block $\\rightarrow$ $d\\_sum$, $d\\_sum^2$", GPU_FILL),
        (5.5,  1.1, 7.8, 1.1, "Host: discount, $\\hat{C} = e^{-rT}\\,\\bar{g}$, 95% CI", HOST_FILL),
    ]
    for x, y, w, h, label, fc in nodes:
        _rounded_box(ax, (x, y), w, h, label, fc, fontsize=13)

    # Arrows: from the BOTTOM edge of node i to the TOP edge of node i+1.
    # Small inset (0.05) keeps the arrow head and tail just outside the box stroke.
    inset = 0.05
    for (yi, hi), (yj, hj) in zip(
        [(n[1], n[3]) for n in nodes][:-1],
        [(n[1], n[3]) for n in nodes][1:],
    ):
        y_from = yi - hi / 2 - inset
        y_to   = yj + hj / 2 + inset
        _arrow(ax, (5.5, y_from), (5.5, y_to))

    # Side annotations aligned with the matching node centre.
    ax.text(9.9, 10.5, "one-shot;\namortised across\nkernel launches",
            ha="left", va="center", fontsize=11, color=NOTE_COL, style="italic")
    ax.text(9.9,  7.9, "$N$ paths,\n64 blocks $\\times$ 256 threads",
            ha="left", va="center", fontsize=11, color=NOTE_COL, style="italic")
    ax.text(9.9,  5.0, "$\\mathcal{O}(N/\\mathrm{blockDim})$\natomics, not $\\mathcal{O}(N)$",
            ha="left", va="center", fontsize=11, color=NOTE_COL, style="italic")

    ax.set_title("Algorithm flow (all three GPU kernels)", fontsize=16, pad=10)
    save_both(fig, out.with_suffix(""))
    plt.close(fig)

# --------------------------------------------------------------------------- #
# Diagram 2: Kernel architecture comparison (where the reduction happens)
# --------------------------------------------------------------------------- #

def fig_kernel_architecture(out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 7.5))
    panels = [
        ("mc_naive",      "one atomicAdd per path",      "#FBE0DA", "global atomic\n(contended)",         "compute SM = 2.8%",  "atomic contention", KERNEL_STYLE["mc_naive"]["color"]),
        ("mc_shared",     "block-reduce + 1 atomic/block", "#D9E7F2", "block reduce\n(shared memory)",   "compute SM = 79%",   "arithmetic-bound",  KERNEL_STYLE["mc_shared"]["color"]),
        ("mc_antithetic", "+Z / −Z pair, then block-reduce", "#D5ECE0", "block reduce of\n½(g₊ + g₋)",    "compute SM = 76%",   "≈½ variance",       KERNEL_STYLE["mc_antithetic"]["color"]),
    ]
    for ax, (name, subtitle, panel_fc, reduce_label, kpi, status, accent) in zip(axes, panels):
        ax.set_xlim(0, 10); ax.set_ylim(0, 12)
        ax.set_aspect("equal"); ax.axis("off")
        # panel background
        from matplotlib.patches import FancyBboxPatch
        ax.add_patch(FancyBboxPatch(
            (0.3, 0.3), 9.4, 11.4,
            boxstyle="round,pad=0.02,rounding_size=0.25",
            linewidth=1.8, facecolor=panel_fc, edgecolor=accent))
        # heading
        ax.text(5, 11.0, name, ha="center", va="center",
                fontsize=22, fontweight="bold", color=accent, family="monospace")
        ax.text(5, 10.1, subtitle, ha="center", va="center",
                fontsize=12, color="#444", style="italic")
        # 6 threads
        for i in range(6):
            x = 1.5 + i * 1.4
            _rounded_box(ax, (x, 8.3), 1.1, 0.7, f"t{i}",
                         "#FFFFFF", edgecolor="#555", fontsize=11)
        # arrows from threads to reduction
        for i in range(6):
            x = 1.5 + i * 1.4
            color = "#C0392B" if name == "mc_naive" else "#333333"
            _arrow(ax, (x, 7.9), (5, 5.6), color=color)
        # reduction node
        if name == "mc_naive":
            _rounded_box(ax, (5, 5.0), 6.5, 1.2, reduce_label,
                         "#F4B7A8", edgecolor="#A33", fontsize=13, fontweight="bold")
            ax.text(5, 3.0, "O(N) global atomics", ha="center", va="center",
                    fontsize=12, color="#A33", fontweight="bold")
        else:
            _rounded_box(ax, (5, 5.0), 6.5, 1.2, reduce_label,
                         "#C8DDF0" if "shared" in name else "#C7E6D4",
                         edgecolor=accent, fontsize=13, fontweight="bold")
            _arrow(ax, (5, 4.4), (5, 3.4))
            _rounded_box(ax, (5, 2.8), 6.5, 1.0, "one atomicAdd per block",
                         "#FFE9CC", edgecolor="#B7791F", fontsize=12)
        # KPI at bottom
        ax.text(5, 1.4, kpi, ha="center", va="center",
                fontsize=13, fontweight="bold", color=accent)
        ax.text(5, 0.9, status, ha="center", va="center",
                fontsize=11, color="#444", style="italic")

    fig.suptitle("Where the reduction happens — and why it matters",
                 fontsize=17, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_both(fig, out.with_suffix(""))
    plt.close(fig)

# --------------------------------------------------------------------------- #
# Diagram 3: Reduction tree (warp shuffle -> shared memory -> 1 atomic)
# --------------------------------------------------------------------------- #

def fig_reduction_tree(out: Path):
    fig, ax = plt.subplots(figsize=(10.0, 7.5))
    ax.set_xlim(0, 14); ax.set_ylim(0, 11)
    ax.set_aspect("equal"); ax.axis("off")

    # Row 1: 32 lanes of a warp (drawn as small boxes)
    for i in range(32):
        x = 1 + i * 0.38
        _rounded_box(ax, (x, 9.5), 0.35, 0.5, "",
                     "#D9E7F2", edgecolor="#4A6FA5", fontsize=8)
    ax.text(0.6, 9.5, "32 lanes\n(1 warp)", ha="right", va="center",
            fontsize=11, color="#333")

    # Row 2: warp result (after 5 shuffle levels)
    _rounded_box(ax, (7, 7.5), 1.2, 0.7, "lane 0",
                 "#7FA8D6", edgecolor="#2B4D7A", fontsize=11, fontweight="bold")
    # arrows from lanes to lane 0 (fade)
    for i in range(32):
        x = 1 + i * 0.38
        ax.annotate("", xy=(7, 7.85), xytext=(x, 9.2),
                    arrowprops=dict(arrowstyle="-", lw=0.5, color="#888", alpha=0.4))
    ax.text(11.0, 8.5,
            "(1) warp shuffle:\n     __shfl_down_sync,\n     5 levels, lane 0 owns sum",
            ha="left", va="center", fontsize=11, color="#444", style="italic")

    # Row 3: 8 warps (1 block = 256 threads / 32 = 8 warps), shown as their lane-0 results
    warps_x = [1.5, 3.3, 5.1, 6.9, 8.7, 10.5]
    for i, x in enumerate(warps_x):
        _rounded_box(ax, (x, 5.5), 1.2, 0.7, f"warp {i}",
                     "#7FA8D6", edgecolor="#2B4D7A", fontsize=10)
    ax.text(12.5, 5.5, "… 8 warps", ha="left", va="center",
            fontsize=11, color="#444")
    # arrow from warp result down
    ax.annotate("", xy=(6.5, 5.95), xytext=(7, 7.15),
                arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#444"))

    # Row 4: shared-memory aggregation
    _rounded_box(ax, (6.5, 3.5), 8.0, 0.9, "shared memory tree",
                 "#FFE9CC", edgecolor="#B7791F", fontsize=13, fontweight="bold")
    for x in warps_x:
        ax.annotate("", xy=(6.5, 3.95), xytext=(x, 5.15),
                    arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#666"))
    ax.text(12.5, 3.5,
            "(2)–(3) lane-0 of\n     each warp writes",
            ha="left", va="center", fontsize=11, color="#444", style="italic")

    # Row 5: single atomicAdd
    _rounded_box(ax, (6.5, 1.5), 8.0, 1.0, "atomicAdd  →  global d_sum",
                 "#F4B7A8", edgecolor="#A33", fontsize=13, fontweight="bold")
    ax.annotate("", xy=(6.5, 2.0), xytext=(6.5, 3.05),
                arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#A33"))
    ax.text(12.5, 1.5,
            "(4) one atomic\n     per block",
            ha="left", va="center", fontsize=11, color="#444", style="italic")

    ax.set_title("Block-level reduction: warp shuffle → shared memory → one atomic",
                 fontsize=15, pad=10)
    save_both(fig, out.with_suffix(""))
    plt.close(fig)

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    rows = load_csv(CSV_PATH)
    agg  = aggregate(rows)

    out_dir = HERE
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {CSV_PATH} ({len(rows)} rows).")
    print(f"Writing PDFs+PNGs to {out_dir}.")

    # Benchmark plots -> poster/figures/
    fig_throughput(agg,        out_dir / "fig_throughput.pdf")
    fig_speedup(agg,           out_dir / "fig_speedup.pdf")
    fig_convergence(agg,       out_dir / "fig_convergence.pdf")
    fig_nsight(                out_dir / "fig_nsight_metrics.pdf")
    fig_speedup_bars(agg,      out_dir / "fig_speedup_bars.pdf")

    # Architecture diagrams -> poster/diagrams/
    DIAGRAMS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Writing architecture diagrams to {DIAGRAMS_DIR}.")
    fig_algorithm_flow(        DIAGRAMS_DIR / "algorithm_flow.pdf")
    fig_kernel_architecture(   DIAGRAMS_DIR / "kernel_architecture.pdf")
    fig_reduction_tree(        DIAGRAMS_DIR / "reduction_tree.pdf")

    print("Headline numbers (median of 5 runs):")
    for k in KERNEL_ORDER:
        Nmax = max(agg[k])
        print(f"  {k:14s} @ N={Nmax:>10}:  {agg[k][Nmax]['ms']:8.2f} ms  "
              f"price err={agg[k][Nmax]['err']:.4f}")
    cpu_ms = agg["mc_cpu"][50_000_000]["ms"]
    for k in ["mc_naive", "mc_shared", "mc_antithetic"]:
        print(f"  speedup vs CPU at 50M: {k:14s} = {cpu_ms / agg[k][50_000_000]['ms']:7.1f}x")

if __name__ == "__main__":
    main()
