import argparse
import csv
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extension.build import load_extension
from bench.harness import (
    device_specs, gemm_metrics, print_specs, time_cuda, time_cuda_roundrobin,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

def _probe_ms(fn):
    """One timed call, to size the interleaved round count."""
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return max(start.elapsed_time(end), 1e-4)

def autotune_iters(fn, target_seconds=1.0, max_iters=100, min_iters=5):
    """Pick an iteration count from a probe run."""
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    probe_s = max(start.elapsed_time(end) * 1e-3, 1e-6)
    n = int(target_seconds / probe_s)
    return max(min_iters, min(max_iters, n))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[256, 512, 1024, 2048, 4096])
    ap.add_argument("--skip-naive-above", type=int, default=2048,
                    help="stage 1 is ~1000x slower than cuBLAS; timing it at "
                         "4096 costs minutes for a number nobody is surprised by")
    ap.add_argument("--round-budget-s", type=float, default=20.0,
                    help="wall-clock budget per matrix size for the interleaved "
                         "timing sweep")
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "gemm_results.csv"))
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    torch.manual_seed(0)

    print("Building CUDA extension (first run compiles, ~1-2 min)...")
    ext = load_extension(verbose=False)
    specs = device_specs()
    print_specs(specs)

    stages = ext.sgemm_stages()
    rows = []

    for size in args.sizes:
        M = N = K = size
        A = torch.randn(M, K, device="cuda", dtype=torch.float32)
        B = torch.randn(K, N, device="cuda", dtype=torch.float32)
        ref = torch.matmul(A, B)
        if size <= 1024:
            ref64 = torch.matmul(A.double(), B.double())
            torch_err = (ref.double() - ref64).abs().max().item()
        else:
            torch_err = float("nan")

        print(f"\n=== {M}x{N}x{K} "
              f"(torch-vs-fp64 max abs err: {torch_err:.3e}) ===")
        print(f"{'stage':<14} {'ms':>9} {'±ms':>8} {'GFLOP/s':>10} "
              f"{'%boost':>7} {'%actual':>7} {'max abs err':>12} {'iters':>6}")

        candidates = []
        for stage in stages:
            if stage == "naive" and size > args.skip_naive_above:
                print(f"{stage:<14} skipped (>{args.skip_naive_above}, ~1000x "
                      f"slower than cuBLAS)")
                continue

            fn = (lambda st: (lambda: ext.sgemm(A, B, st)))(stage)

            out = fn()
            err = (out - ref).abs().max().item()
            rel = err / ref.abs().max().item()
            tol = 3e-6 * K
            if not (err <= tol):
                print(f"{stage:<14}  FAILED CORRECTNESS: max abs err {err:.3e} "
                      f"> tol {tol:.3e} -- not timing this kernel")
                rows.append(dict(size=size, stage=stage, status="FAILED",
                                 max_abs_err=err))
                continue
            candidates.append((stage, fn, err, rel))

        probe = max(_probe_ms(fn) for _, fn, _, _ in candidates)
        rounds = max(5, min(100, int(args.round_budget_s * 1000 /
                                     (probe * len(candidates) + 1e-9))))
        print(f"(interleaved: {rounds} rounds x {len(candidates)} kernels)")
        results = time_cuda_roundrobin([(st, fn) for st, fn, _, _ in candidates],
                                       warmup=3, rounds=rounds)

        for stage, _fn, err, rel in candidates:
            t = results[stage]
            m = gemm_metrics(M, N, K, t["mean_ms"], specs,
                             sm_clock_mhz=t.get("sm_clock_mean_mhz"))

            print(f"{stage:<14} {t['mean_ms']:>9.3f} {t['std_ms']:>8.3f} "
                  f"{m['gflops']:>10.1f} {m.get('pct_peak_flops', 0):>6.1f}% "
                  f"{m.get('pct_effective_peak', 0):>6.1f}% "
                  f"{err:>12.3e} {t['iters']:>6}")

            rows.append(dict(
                size=size, stage=stage, status="ok",
                mean_ms=t["mean_ms"], std_ms=t["std_ms"], min_ms=t["min_ms"],
                median_ms=t["median_ms"], p95_ms=t["p95_ms"], iters=t["iters"],
                sm_clock_mean_mhz=t.get("sm_clock_mean_mhz"),
                sm_clock_min_mhz=t.get("sm_clock_min_mhz"),
                sm_clock_max_mhz=t.get("sm_clock_max_mhz"),
                sm_clock_samples=t.get("sm_clock_samples"),
                max_abs_err=err, max_rel_err=rel,
                gflops=m["gflops"], pct_peak_flops=m.get("pct_peak_flops"),
                pct_effective_peak=m.get("pct_effective_peak"),
                effective_peak_gflops=m.get("effective_peak_gflops"),
                compulsory_gbs=m["compulsory_gbs"],
                pct_peak_bw=m.get("pct_peak_bw"),
                arithmetic_intensity=m["arithmetic_intensity"],
                gpu=specs["name"],
            ))

        by_stage = {r["stage"]: r for r in rows
                    if r["size"] == size and r.get("status") == "ok"}
        base = by_stage.get("naive", {}).get("gflops")
        ceil = by_stage.get("cublas", {}).get("gflops")
        for r in rows:
            if r["size"] == size and r.get("status") == "ok":
                r["speedup_vs_naive"] = (r["gflops"] / base) if base else None
                r["pct_of_cublas"] = (100.0 * r["gflops"] / ceil) if ceil else None

    fields = sorted({k for r in rows for k in r})
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {args.out}  ({len(rows)} rows)")
    return rows

if __name__ == "__main__":
    main()
