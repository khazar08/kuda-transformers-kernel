import base64
import io
import json
import os
import tarfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXCLUDE_DIRS = {".git", "build", "__pycache__", "notebooks", ".ipynb_checkpoints"}
EXCLUDE_EXT = {".png", ".ncu-rep", ".so", ".o"}

def build_tarball():
    buf = io.BytesIO()
    n = 0
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for dirpath, dirnames, filenames in os.walk(ROOT):
            dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
            for fn in sorted(filenames):
                if os.path.splitext(fn)[1] in EXCLUDE_EXT:
                    continue
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, ROOT)
                tar.add(full, arcname=rel)
                n += 1
    return base64.b64encode(buf.getvalue()).decode(), n

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(True)}

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.splitlines(True)}

HANDBACK = r"""import csv, os, sys
ROOT = "/content/cuda-transformer-kernels"; os.chdir(ROOT); sys.path.insert(0, ROOT)
from bench.harness import device_specs

def rule(t): print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78)

rule("HARDWARE")
for k, v in device_specs().items(): print(f"  {k:<26} {v}")

for path, title in [("bench/results/gemm_results.csv", "GEMM LADDER (raw CSV)"),
                    ("bench/results/fused_results.csv", "FUSED KERNELS (raw CSV)"),
                    ("bench/results/parity.txt", "PARITY TESTS"),
                    ("bench/results/occupancy.txt", "OCCUPANCY"),
                    ("bench/results/ncu.txt", "NSIGHT COMPUTE"),
                    ("bench/results/llm.txt", "END-TO-END LLM")]:
    rule(title)
    print(open(path).read() if os.path.exists(path)
          else "MISSING: " + path + "  (that cell did not run)")

rule("END OF HAND-BACK")
"""

def main():
    payload, nfiles = build_tarball()
    kb = len(payload) / 1024

    cells = [
        md("""# Hand-optimized CUDA kernels for transformer inference

**Runtime -> Change runtime type -> T4 GPU**, then **Runtime -> Run all**.

This notebook is self-contained: the entire repo is embedded below. It will
build the CUDA extension, verify every kernel against torch, benchmark the
six-stage SGEMM ladder against cuBLAS, benchmark the fused kernels, render the
plots, analyse occupancy, and measure end-to-end tokens/sec on a real model.

Expect ~10-15 minutes total, most of it the first CUDA compile.
"""),
        code(f'''# --- bootstrap: unpack the embedded repo ({nfiles} files, {kb:.0f} KB) ---
import base64, io, os, tarfile

REPO_B64 = "{payload}"

os.makedirs("/content/cuda-transformer-kernels", exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(REPO_B64)), mode="r:gz") as t:
    t.extractall("/content/cuda-transformer-kernels")
os.chdir("/content/cuda-transformer-kernels")
print("unpacked into", os.getcwd())
print(sorted(os.listdir(".")))'''),

        md("## 1. Hardware and environment\n\nEverything downstream is reported "
           "relative to these peaks, so they are established first."),
        code('''!nvidia-smi
import sys, torch
sys.path.insert(0, "/content/cuda-transformer-kernels")
from bench.harness import device_specs, print_specs
specs = device_specs()
print()
print_specs(specs)
assert torch.cuda.is_available(), "No GPU. Runtime -> Change runtime type -> T4 GPU"'''),

        md("## 2. Build the CUDA extension\n\nFirst run compiles all kernels "
           "(~1-2 min). `--ptxas-options=-v` prints per-kernel register and "
           "shared-memory usage, which drives the occupancy analysis later."),
        code('''from extension.build import load_extension
ext = load_extension(verbose=True)
print("\\nstages available:", ext.sgemm_stages())'''),

        md("## 3. Correctness before performance\n\nA fast kernel that is wrong "
           "is worth nothing. Every stage is checked against a float64 reference, "
           "including shapes that are not multiples of any tile size."),
        code('''!cd /content/cuda-transformer-kernels && mkdir -p bench/results && python -m pytest extension/test_parity.py -q 2>&1 | tee bench/results/parity.txt | tail -30'''),

        md("## 4. The SGEMM ladder vs cuBLAS\n\nSweeps 256 -> 4096. Each stage is "
           "verified, then timed with CUDA events with an iteration count scaled "
           "to the kernel's cost."),
        code('''!cd /content/cuda-transformer-kernels && python bench/bench_gemm.py'''),

        md("## 5. Fused kernels\n\nMemory-bound ops, so the headline metric is "
           "achieved bandwidth as a fraction of peak -- not GFLOP/s."),
        code('''!cd /content/cuda-transformer-kernels && python bench/bench_fused.py'''),

        md("## 6. Plots"),
        code('''!cd /content/cuda-transformer-kernels && python bench/plots.py
from IPython.display import Image, display
display(Image("/content/cuda-transformer-kernels/bench/results/gflops_vs_size.png"))
display(Image("/content/cuda-transformer-kernels/bench/results/roofline.png"))'''),

        md("""## 7. Occupancy and the memory-vs-compute limiter

Colab disables GPU performance counters, so `ncu` will most likely fail with a
permissions error. That is host policy, not a bug. The script detects it and
falls back to deriving occupancy from ptxas register/shared-memory usage, which
is a compile-time property and needs no counters."""),
        code('''!cd /content/cuda-transformer-kernels && bash bench/profile_ncu.sh 2048 2>&1 | tee bench/results/ncu.txt | tail -40'''),
        code('''!cd /content/cuda-transformer-kernels && python bench/occupancy.py 2>&1 | tee bench/results/occupancy.txt'''),

        md("""## 8. End-to-end: a real transformer

Swaps the kernels into a model's inference path and measures tokens/sec.

Two regimes are measured separately and they behave differently. **Prefill**
(the whole prompt at once) gives large-M matmuls where a hand-written kernel can
win. **Decode** (one token at a time) makes M = batch size, which is a GEMV, not
a GEMM -- memory-bound with no reuse, where cuBLAS's dedicated path wins. That
is why `FastLinear` routes small-M calls back to torch.

To use your own LoRA checkpoint, add `--model <base> --lora <adapter>`."""),
        code('''!pip -q install transformers >/dev/null 2>&1
!cd /content/cuda-transformer-kernels && python bench/bench_llm.py --model gpt2 --prompt-len 512 2>&1 | tee bench/results/llm.txt'''),

        md("""## 9. HAND-BACK \u2014 copy this cell's entire output

This is the only output you need to send back. It collects every measurement
from the cells above into one block, at full precision."""),
        code(HANDBACK),
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }

    out = os.path.join(ROOT, "notebooks", "run_on_colab.ipynb")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"wrote {out}  ({os.path.getsize(out)/1024:.0f} KB, "
          f"{nfiles} files embedded)")

if __name__ == "__main__":
    main()
