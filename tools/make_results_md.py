"""Turn the benchmark CSVs into the markdown tables the README embeds."""
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "bench", "results")
sys.path.insert(0, ROOT)

LABEL = {
    "naive": "1. Naive", "coalesced": "2. Coalesced", "smem": "3. Shared-mem tiling",
    "blocktile_1d": "4. 1D thread tiling", "blocktile_2d": "5. 2D thread tiling",
    "vectorized": "6. float4 vectorized", "cublas": "cuBLAS",
}
ORDER = list(LABEL)

def f(v, spec="{:.1f}"):
    try:
        return spec.format(float(v))
    except (TypeError, ValueError):
        return "-"

def gemm_table():
    path = os.path.join(RESULTS, "gemm_results.csv")
    if not os.path.exists(path):
        return "_(run bench/bench_gemm.py first)_\n"
    rows = [r for r in csv.DictReader(open(path)) if r.get("status") == "ok"]
    if not rows:
        return "_(no successful rows)_\n"
    sizes = sorted({int(r["size"]) for r in rows})

    out = []
    for size in sizes:
        sub = {r["stage"]: r for r in rows if int(r["size"]) == size}
        out.append(f"\n#### {size} x {size} x {size}\n")
        out.append("| Stage | GFLOP/s | % of FP32 peak | Speedup vs naive | "
                   "% of cuBLAS | Max abs err | ms (mean ± sd) |")
        out.append("|---|---:|---:|---:|---:|---:|---:|")
        for st in ORDER:
            r = sub.get(st)
            if not r:
                continue
            out.append(
                f"| {LABEL[st]} | {f(r['gflops'])} | {f(r.get('pct_peak_flops'))}% | "
                f"{f(r.get('speedup_vs_naive'), '{:.1f}')}x | "
                f"{f(r.get('pct_of_cublas'))}% | {f(r['max_abs_err'], '{:.2e}')} | "
                f"{f(r['mean_ms'], '{:.3f}')} ± {f(r['std_ms'], '{:.3f}')} |")
    return "\n".join(out) + "\n"

def fused_table():
    path = os.path.join(RESULTS, "fused_results.csv")
    if not os.path.exists(path):
        return "_(run bench/bench_fused.py first)_\n"
    rows = list(csv.DictReader(open(path)))
    out = ["| Shape | Implementation | Launches | ms | GB/s | % of peak BW | Max abs err |",
           "|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        out.append(
            f"| {r['rows']} x {r['cols']} | {r['impl']} | {r['launches']} | "
            f"{f(r['mean_ms'], '{:.4f}')} | {f(r['achieved_gbs'])} | "
            f"{f(r.get('pct_peak_bw'))}% | {f(r['max_abs_err'], '{:.2e}')} |")
    return "\n".join(out) + "\n"

def inject(readme_path, block_name, content):
    """Replace the region between <!-- RESULTS:X --> markers in the README."""
    with open(readme_path) as f:
        text = f.read()
    start = f"<!-- RESULTS:{block_name} -->"
    end = f"<!-- /RESULTS:{block_name} -->"
    if start not in text or end not in text:
        return False
    pre, rest = text.split(start, 1)
    _, post = rest.split(end, 1)
    with open(readme_path, "w") as f:
        f.write(pre + start + "\n" + content + "\n" + end + post)
    return True

def load_specs():
    """Live device specs, or the archived ones when no GPU is present."""
    import json
    archived = os.path.join(RESULTS, "device_specs.json")
    try:
        import torch
        if torch.cuda.is_available():
            from bench.harness import device_specs
            spec = device_specs()
            with open(archived, "w") as fh:
                json.dump(spec, fh, indent=2)
            return spec
    except Exception:
        pass
    with open(archived) as fh:
        return json.load(fh)

def main():
    s = load_specs()
    print("=" * 70)
    print("COPY EVERYTHING BELOW THIS LINE")
    print("=" * 70)
    print(f"""
## Hardware

| | |
|---|---|
| GPU | {s['name']} (sm_{s['compute_capability'].replace('.','')}) |
| SMs | {s['sms']} x {s['cores_per_sm']} FP32 cores |
| Max SM clock | {s['max_sm_clock_mhz']} MHz |
| Theoretical FP32 peak | {f(s['peak_fp32_gflops'], '{:,.0f}')} GFLOP/s |
| Peak memory bandwidth | {s['peak_bandwidth_gbs']} GB/s |
| Roofline ridge point | {f((s['peak_fp32_gflops'] or 0)/(s['peak_bandwidth_gbs'] or 1))} FLOP/byte |
| PyTorch / CUDA | {s['torch']} / {s['cuda']} |

## SGEMM ladder
{gemm_table()}
## Fused kernels
{fused_table()}""")

if __name__ == "__main__":
    if "--inject" in sys.argv:
        readme = os.path.join(ROOT, "README.md")
        ok_g = inject(readme, "GEMM", gemm_table())
        ok_f = inject(readme, "FUSED", fused_table())
        print(f"injected into README.md: GEMM={ok_g} FUSED={ok_f}")
    else:
        main()
