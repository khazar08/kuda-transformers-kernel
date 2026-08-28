import argparse
import glob
import os
import re
import subprocess
import sys

ARCH_LIMITS = {
    75: dict(regs_per_sm=65536, max_threads_per_sm=1024, max_blocks_per_sm=16,
             smem_per_sm=65536, warp_size=32, name="Turing"),
    80: dict(regs_per_sm=65536, max_threads_per_sm=2048, max_blocks_per_sm=32,
             smem_per_sm=167936, warp_size=32, name="Ampere A100"),
    86: dict(regs_per_sm=65536, max_threads_per_sm=1536, max_blocks_per_sm=16,
             smem_per_sm=102400, warp_size=32, name="Ampere GA10x"),
    89: dict(regs_per_sm=65536, max_threads_per_sm=1536, max_blocks_per_sm=24,
             smem_per_sm=102400, warp_size=32, name="Ada"),
    90: dict(regs_per_sm=65536, max_threads_per_sm=2048, max_blocks_per_sm=32,
             smem_per_sm=233472, warp_size=32, name="Hopper"),
}

BLOCK_SIZES = {
    "sgemm_naive_kernel": 1024,
    "sgemm_coalesced_kernel": 1024,
    "sgemm_smem_kernel": 1024,
    "sgemm_blocktile_1d_kernel": 512,
    "sgemm_blocktile_2d_kernel": 256,
    "sgemm_vectorized_kernel": 256,
    "softmax_fused_kernel": 256,
    "layernorm_fused_kernel": 256,
    "softmax_pass1_max": 256,
    "softmax_pass2_exp_sum": 256,
    "softmax_pass3_normalize": 256,
    "softmax_fused_vec4_kernel": None,
    "layernorm_fused_vec4_kernel": None,
}

def compile_and_parse(src, arch, root):
    """Compile one .cu and pull register/smem usage out of ptxas's report."""
    cmd = ["nvcc", f"-arch=sm_{arch}", "-O3", "-std=c++17",
           "--ptxas-options=-v", "-c", src, "-o", "/dev/null",
           "-I", os.path.join(root, "kernels")]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    text = proc.stderr + proc.stdout
    out = []
    for block in re.split(r"ptxas info\s*:\s*Compiling entry function", text):
        m = re.search(r"'([^']+)'", block)
        if not m:
            continue
        mangled = m.group(1)
        name = next((k for k in BLOCK_SIZES if k in mangled), mangled)
        regs = re.search(r"Used (\d+) registers", block)
        smem = re.search(r"(\d+) bytes smem", block)
        lmem = re.search(r"(\d+) bytes (?:stack frame|lmem)", block)
        if regs:
            out.append(dict(kernel=name, regs=int(regs.group(1)),
                            smem=int(smem.group(1)) if smem else 0,
                            lmem=int(lmem.group(1)) if lmem else 0))
    if not out and proc.returncode != 0:
        print(f"  nvcc failed for {os.path.basename(src)}:\n{text[-800:]}")
    return out

def occupancy(regs, smem, threads, lim):
    """Theoretical occupancy: the tightest of four independent limits."""
    warps = -(-threads // lim["warp_size"])

    regs_per_warp = -(-(regs * lim["warp_size"]) // 256) * 256
    by_regs = lim["regs_per_sm"] // regs_per_warp // warps if regs_per_warp else 999
    by_smem = lim["smem_per_sm"] // smem if smem else 999
    by_threads = lim["max_threads_per_sm"] // threads
    by_blocks = lim["max_blocks_per_sm"]

    blocks = max(0, min(by_regs, by_smem, by_threads, by_blocks))
    active_warps = blocks * warps
    max_warps = lim["max_threads_per_sm"] // lim["warp_size"]
    limiter = min(
        [("registers", by_regs), ("shared memory", by_smem),
         ("thread slots", by_threads), ("blocks/SM", by_blocks)],
        key=lambda kv: kv[1])[0]
    return dict(blocks_per_sm=blocks, active_warps=active_warps,
                max_warps=max_warps,
                occupancy=100.0 * active_warps / max_warps, limiter=limiter)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", type=int, default=None,
                    help="compute capability as an int, e.g. 75")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    arch = args.arch
    if arch is None:
        try:
            import torch
            major, minor = torch.cuda.get_device_capability()
            arch = major * 10 + minor
        except Exception:
            arch = 75
    lim = ARCH_LIMITS.get(arch)
    if lim is None:
        print(f"no architectural limits recorded for sm_{arch}; aborting")
        return 1

    print(f"Occupancy analysis for sm_{arch} ({lim['name']})")
    print(f"  {lim['regs_per_sm']} regs/SM, {lim['max_threads_per_sm']} threads/SM, "
          f"{lim['smem_per_sm']//1024} KB smem/SM, {lim['max_blocks_per_sm']} blocks/SM\n")
    print(f"{'kernel':<28} {'regs':>5} {'smem':>7} {'lmem':>5} {'thr':>5} "
          f"{'blk/SM':>7} {'warps':>7} {'occ':>7}  limiter")
    print("-" * 92)

    for src in sorted(glob.glob(os.path.join(root, "kernels", "*.cu"))):
        if "cublas" in src:
            continue
        for k in compile_and_parse(src, arch, root):
            threads = k.get("tmpl_block") or BLOCK_SIZES.get(k["kernel"]) or 256
            o = occupancy(k["regs"], k["smem"], threads, lim)
            warn = "  <-- REGISTER SPILL" if k["lmem"] else ""
            label = k["kernel"]
            if k.get("tmpl_block"):
                label = f"{label}<{k['tmpl_block']}>"
            print(f"{label[:28]:<28} {k['regs']:>5} {k['smem']:>7} "
                  f"{k['lmem']:>5} {threads:>5} {o['blocks_per_sm']:>7} "
                  f"{o['active_warps']:>3}/{o['max_warps']:<3} "
                  f"{o['occupancy']:>6.1f}%  {o['limiter']}{warn}")

    print("\nA nonzero lmem figure means registers spilled to local memory and")
    print("performance will be far below what the tiling predicts.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
