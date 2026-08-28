import csv
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

STAGE_ORDER = ["naive", "coalesced", "smem", "blocktile_1d", "blocktile_2d",
               "vectorized", "cublas"]
STAGE_LABEL = {
    "naive": "1. naive",
    "coalesced": "2. coalesced",
    "smem": "3. shared-mem tiling",
    "blocktile_1d": "4. 1D thread tiling",
    "blocktile_2d": "5. 2D thread tiling",
    "vectorized": "6. float4 vectorized",
    "cublas": "cuBLAS (ceiling)",
}

def load_rows(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        if r.get("status") != "ok":
            continue
        rec = {"size": int(r["size"]), "stage": r["stage"]}
        for k in ("gflops", "arithmetic_intensity", "compulsory_gbs"):
            rec[k] = float(r[k])
        out.append(rec)
    return out

def plot_throughput(rows, specs, path):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for stage in STAGE_ORDER:
        pts = sorted([r for r in rows if r["stage"] == stage],
                     key=lambda r: r["size"])
        if not pts:
            continue
        style = dict(marker="o", linewidth=2)
        if stage == "cublas":
            style = dict(marker="s", linewidth=2.5, linestyle="--", color="black")
        ax.plot([p["size"] for p in pts], [p["gflops"] for p in pts],
                label=STAGE_LABEL[stage], **style)

    peak = specs.get("peak_fp32_gflops")
    if peak:
        ax.axhline(peak, color="crimson", linestyle=":", linewidth=1.5)
        ax.text(rows[0]["size"], peak * 1.04,
                f"theoretical FP32 peak = {peak:,.0f} GFLOP/s",
                color="crimson", fontsize=9)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Matrix size N (square N x N x N SGEMM)")
    ax.set_ylabel("Throughput (GFLOP/s)")
    ax.set_title(f"SGEMM optimization ladder on {specs.get('name', 'GPU')}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print("wrote", path)

def plot_roofline(rows, specs, path):
    """Roofline: y = min(peak_flops, bandwidth * intensity)."""
    peak = specs.get("peak_fp32_gflops")
    bw = specs.get("peak_bandwidth_gbs")
    if not (peak and bw):
        print("skipping roofline: unknown peak flops or bandwidth")
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))
    xs = [2 ** (i / 4.0) for i in range(-16, 45)]
    ax.plot(xs, [min(peak, bw * x) for x in xs], color="black", linewidth=2.5,
            label="roofline")

    ridge = peak / bw
    ax.axvline(ridge, color="grey", linestyle=":", linewidth=1.2)
    ax.text(ridge * 0.92, peak * 0.004, f"ridge point\n{ridge:.1f} FLOP/byte",
            fontsize=9, color="grey", ha="right")

    markers = {"naive": "v", "coalesced": "^", "smem": "s", "blocktile_1d": "D",
               "blocktile_2d": "o", "vectorized": "*", "cublas": "X"}
    for stage in STAGE_ORDER:
        pts = [r for r in rows if r["stage"] == stage]
        if not pts:
            continue
        ax.scatter([p["arithmetic_intensity"] for p in pts],
                   [p["gflops"] for p in pts],
                   label=STAGE_LABEL[stage], marker=markers[stage],
                   s=90 if stage != "vectorized" else 160, alpha=0.85,
                   zorder=3, edgecolors="white", linewidths=0.5)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("Compulsory arithmetic intensity (FLOP / byte)")
    ax.set_ylabel("Achieved throughput (GFLOP/s)")
    ax.set_title(f"Roofline: {specs.get('name','GPU')} "
                 f"({peak:,.0f} GFLOP/s, {bw:,.0f} GB/s)")
    ax.text(0.015, 0.03,
            "Each column = one matrix size (256\u20134096).\n"
            "Compulsory intensity depends only on shape, so all stages share an x;\n"
            "vertical spread within a column is performance left on the table.",
            transform=ax.transAxes, fontsize=8, color="dimgrey", va="bottom")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print("wrote", path)

def main():
    from bench.harness import device_specs
    specs = device_specs()
    rows = load_rows(os.path.join(RESULTS_DIR, "gemm_results.csv"))
    plot_throughput(rows, specs, os.path.join(RESULTS_DIR, "gflops_vs_size.png"))
    plot_roofline(rows, specs, os.path.join(RESULTS_DIR, "roofline.png"))

if __name__ == "__main__":
    main()
