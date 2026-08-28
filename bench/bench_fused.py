import csv
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extension.build import load_extension
from bench.harness import (
    device_specs, elementwise_metrics, print_specs, time_cuda,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

SHAPES = [
    (4096, 256), (4096, 1024), (8192, 1024), (16384, 768), (4096, 4096),
]

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    torch.manual_seed(0)
    ext = load_extension(verbose=False)
    specs = device_specs()
    print_specs(specs)

    rows_out = []
    for rows, cols in SHAPES:
        X = torch.randn(rows, cols, device="cuda", dtype=torch.float32)
        gamma = torch.randn(cols, device="cuda", dtype=torch.float32)
        beta = torch.randn(cols, device="cuda", dtype=torch.float32)
        n = rows * cols

        print(f"\n=== {rows} x {cols} ===")
        print(f"{'impl':<26} {'launches':>9} {'ms':>9} {'±ms':>8} "
              f"{'GB/s':>9} {'%peak BW':>9} {'max abs err':>12}")

        ref_sm = torch.softmax(X, dim=-1)
        ref_ln = torch.nn.functional.layer_norm(X, (cols,), gamma, beta, 1e-5)

        cands = [
            ("softmax 3-pass (ours)", 3, lambda: ext.softmax(X, fused=False), ref_sm, 5 * 4),
            ("softmax fused (ours)",  1, lambda: ext.softmax(X, fused=True),  ref_sm, 3 * 4),
            ("torch.softmax",         1, lambda: torch.softmax(X, dim=-1),    ref_sm, 3 * 4),
            ("layernorm fused (ours)", 1, lambda: ext.layernorm(X, gamma, beta, 1e-5), ref_ln, 3 * 4),
            ("torch.layer_norm",       1, lambda: torch.nn.functional.layer_norm(
                X, (cols,), gamma, beta, 1e-5), ref_ln, 3 * 4),
        ]

        for name, launches, fn, ref, bpe in cands:
            out = fn()
            err = (out - ref).abs().max().item()
            t = time_cuda(fn, warmup=25, iters=100)
            m = elementwise_metrics(n, bpe, t["mean_ms"], specs)
            print(f"{name:<26} {launches:>9} {t['mean_ms']:>9.4f} "
                  f"{t['std_ms']:>8.4f} {m['achieved_gbs']:>9.1f} "
                  f"{m.get('pct_peak_bw', 0):>8.1f}% {err:>12.3e}")
            rows_out.append(dict(
                rows=rows, cols=cols, impl=name, launches=launches,
                mean_ms=t["mean_ms"], std_ms=t["std_ms"], min_ms=t["min_ms"],
                achieved_gbs=m["achieved_gbs"], pct_peak_bw=m.get("pct_peak_bw"),
                max_abs_err=err, gpu=specs["name"],
            ))

    path = os.path.join(RESULTS_DIR, "fused_results.csv")
    fields = sorted({k for r in rows_out for k in r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_out)
    print(f"\nWrote {path}")
    return rows_out

if __name__ == "__main__":
    main()
