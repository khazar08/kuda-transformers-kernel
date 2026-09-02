import glob, os, re, shutil, subprocess, sys, textwrap
import torch
from torch.utils import cpp_extension

NAME = "sgemm_kernels"

ROOT = "/content/cuda-transformer-kernels"
if not os.path.isdir(os.path.join(ROOT, "kernels")):
    hits = glob.glob("/content/**/kernels/01_sgemm_naive.cu", recursive=True)
    ROOT = os.path.dirname(os.path.dirname(hits[0])) if hits else None
assert ROOT, "repo not found -- run the bootstrap cell first"
os.chdir(ROOT)
print("repo:", ROOT)
print(f"torch {torch.__version__} | CUDA {torch.version.cuda} | "
      f"python {sys.version.split()[0]}")
print("device:", torch.cuda.get_device_name(0),
      "sm_" + "".join(map(str, torch.cuda.get_device_capability())))

b = os.path.join(ROOT, "extension", "bindings.cpp")
src = open(b).read()
before = src
src = src.replace("c10::optional<", "std::optional<").replace("c10::nullopt", "std::nullopt")
if "#include <optional>" not in src:
    src = src.replace("#include <string>", "#include <optional>\n#include <string>", 1)
if "#include <vector>" not in src:
    src = src.replace("#include <unordered_map>", "#include <unordered_map>\n#include <vector>", 1)
if src != before:
    open(b, "w").write(src)
    print("\npatched bindings.cpp: c10::optional -> std::optional")
else:
    print("\nbindings.cpp already uses std::optional")

versioner = getattr(cpp_extension, "JIT_EXTENSION_VERSIONER", None)
entries = getattr(versioner, "entries", None)
if isinstance(entries, dict):
    popped = entries.pop(NAME, None)
    print(f"in-memory version entry: {'cleared' if popped is not None else 'was absent'}")
try:
    BUILD_DIR = cpp_extension._get_build_directory(NAME, verbose=False)
except Exception:
    BUILD_DIR = os.path.expanduser(f"~/.cache/torch_extensions/{NAME}")
if os.path.isdir(BUILD_DIR):
    shutil.rmtree(BUILD_DIR, ignore_errors=True)
    print("removed", BUILD_DIR)
for m in [k for k in sys.modules if k.startswith(NAME)]:
    del sys.modules[m]

major, minor = torch.cuda.get_device_capability()
ARCH = f"sm_{major}{minor}"
print(f"\n--- standalone nvcc check ({ARCH}) ---")
all_ok = True
for cu in sorted(glob.glob(os.path.join(ROOT, "kernels", "*.cu"))):
    p = subprocess.run(
        ["nvcc", f"-arch={ARCH}", "-O3", "-std=c++17", "--ptxas-options=-v",
         "-c", cu, "-o", "/dev/null", "-I", os.path.join(ROOT, "kernels")],
        capture_output=True, text=True)
    name = os.path.basename(cu)
    if p.returncode == 0:
        regs = re.findall(r"Used (\d+) registers", p.stderr)
        smem = re.findall(r"(\d+) bytes smem", p.stderr)
        spill = re.findall(r"(\d+) bytes stack frame", p.stderr)
        extra = f"  regs={regs} smem={smem}" if regs else ""
        warn = "  <-- REGISTER SPILL" if any(int(x) > 0 for x in spill) else ""
        print(f"  OK   {name}{extra}{warn}")
    else:
        all_ok = False
        print(f"  FAIL {name}\n{textwrap.indent((p.stderr or p.stdout)[:4000], '       ')}")
print("all kernels compile standalone" if all_ok else "*** kernel compile errors above ***")

print("\n--- building extension (verbose) ---")
try:
    ext = cpp_extension.load(
        name=NAME,
        sources=([os.path.join(ROOT, "extension", "bindings.cpp")]
                 + sorted(glob.glob(os.path.join(ROOT, "kernels", "*.cu")))),
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-std=c++17", "-lineinfo",
                           "--ptxas-options=-v",
                           f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"],
        extra_ldflags=["-lcublas"],
        verbose=True,
    )
    print("\nBUILD OK. stages:", ext.sgemm_stages())
    a = torch.randn(512, 512, device="cuda"); bb = torch.randn(512, 512, device="cuda")
    err = (ext.sgemm(a, bb, "vectorized") - a @ bb).abs().max().item()
    print(f"smoke test 512x512 vs torch: max abs err {err:.3e}")
except Exception as exc:
    print(f"\nBUILD FAILED: {type(exc).__name__}: {exc}\n", file=sys.stderr)
    nf = os.path.join(BUILD_DIR, "build.ninja")
    if os.path.exists(nf):
        print("=" * 72, "\nre-running ninja -v for the actual nvcc command + error\n", "=" * 72)
        p = subprocess.run(["ninja", "-v"], cwd=BUILD_DIR, capture_output=True, text=True)
        print(p.stdout[-20000:]); print(p.stderr[-20000:])
        print("ninja exit:", p.returncode)
    else:
        print(f"no build.ninja in {BUILD_DIR} -- torch never got as far as writing one")
    if isinstance(entries, dict):
        entries.pop(NAME, None)
    raise
