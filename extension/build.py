import glob
import os
import shutil
import subprocess
import sys

import torch
from torch.utils.cpp_extension import load

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAME = "sgemm_kernels"
_EXT = None

def cuda_arch_flags():
    """Detect compute capability rather than hardcoding it."""
    major, minor = torch.cuda.get_device_capability()
    cc = f"{major}{minor}"
    return [f"-gencode=arch=compute_{cc},code=sm_{cc}"]

def sources():
    return ([os.path.join(ROOT, "extension", "bindings.cpp")]
            + sorted(glob.glob(os.path.join(ROOT, "kernels", "*.cu"))))

def build_dir():
    try:
        from torch.utils.cpp_extension import _get_build_directory
        return _get_build_directory(NAME, verbose=False)
    except Exception:
        return os.path.join(
            os.path.expanduser("~/.cache/torch_extensions"), NAME)

def clear_build_cache(verbose=True):
    """Clear BOTH caches. The in-memory one is the one that actually bites."""
    from torch.utils import cpp_extension

    versioner = getattr(cpp_extension, "JIT_EXTENSION_VERSIONER", None)
    entries = getattr(versioner, "entries", None)
    if isinstance(entries, dict) and entries.pop(NAME, None) is not None and verbose:
        print(f"cleared in-memory version entry for '{NAME}' "
              f"(this is what makes torch skip the compile)")

    d = build_dir()
    if d and os.path.isdir(d):
        shutil.rmtree(d, ignore_errors=True)
        if verbose:
            print(f"removed build directory {d}")

    global _EXT
    _EXT = None

def _dump_ninja(verbose_build_dir):
    """Re-run ninja verbosely so the actual nvcc command line and error print."""
    ninja_file = os.path.join(verbose_build_dir, "build.ninja")
    if not os.path.exists(ninja_file):
        print(f"(no build.ninja in {verbose_build_dir}; nothing to re-run)")
        return
    print("\n" + "=" * 72)
    print("Re-running ninja verbosely to surface the real compiler error")
    print("=" * 72)
    proc = subprocess.run(["ninja", "-v"], cwd=verbose_build_dir,
                          capture_output=True, text=True)
    print(proc.stdout[-20000:])
    print(proc.stderr[-20000:])
    print(f"(ninja exit code {proc.returncode})")

def precheck_nvcc(verbose=True):
    """Compile each kernel standalone with nvcc, independent of torch."""
    major, minor = torch.cuda.get_device_capability()
    arch = f"sm_{major}{minor}"
    ok = True
    for src in sorted(glob.glob(os.path.join(ROOT, "kernels", "*.cu"))):
        cmd = ["nvcc", f"-arch={arch}", "-O3", "-std=c++17", "-c", src,
               "-o", "/dev/null", "-I", os.path.join(ROOT, "kernels")]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        name = os.path.basename(src)
        if proc.returncode == 0:
            if verbose:
                print(f"  OK   {name}")
        else:
            ok = False
            print(f"  FAIL {name}")
            print((proc.stderr or proc.stdout)[:4000])
    return ok

def load_extension(verbose=False, ptxas_verbose=True, fresh=False):
    """Compile and load the extension. Cached after the first success."""
    global _EXT
    if _EXT is not None and not fresh:
        return _EXT

    if fresh:
        clear_build_cache(verbose=verbose)

    d = build_dir()
    if os.path.isdir(d) and not glob.glob(os.path.join(d, f"{NAME}*.so")):
        if verbose:
            print(f"found a build directory with no .so -- a previous build "
                  f"failed; clearing so the compile actually re-runs")
        clear_build_cache(verbose=verbose)

    cuda_flags = ["-O3", "-std=c++17", "-lineinfo"]
    if ptxas_verbose:
        cuda_flags.append("--ptxas-options=-v")
    cuda_flags += cuda_arch_flags()

    try:
        _EXT = load(
            name=NAME,
            sources=sources(),
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=cuda_flags,
            extra_ldflags=["-lcublas"],
            verbose=verbose,
        )
    except Exception as exc:
        print(f"\nExtension build/load failed: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        print("\nIsolating the failure -- compiling each kernel standalone:")
        precheck_nvcc()
        _dump_ninja(build_dir())
        clear_build_cache(verbose=False)
        raise

    return _EXT

if __name__ == "__main__":
    print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, "
          f"python {sys.version.split()[0]}")
    if not torch.cuda.is_available():
        sys.exit("no CUDA device visible")
    print(f"device: {torch.cuda.get_device_name(0)} "
          f"sm_{''.join(map(str, torch.cuda.get_device_capability()))}\n")
    print("standalone nvcc check of each kernel:")
    precheck_nvcc()
    print("\nbuilding extension...")
    ext = load_extension(verbose=True, fresh=True)
    print("\nbuild OK. stages:", ext.sgemm_stages())
