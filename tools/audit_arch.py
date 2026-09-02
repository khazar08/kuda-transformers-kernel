"""Assert the kernels use nothing newer than the target architecture supports."""
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SM80_PLUS = {
    r"cp\.async": "cp.async (sm_80+)",
    r"cuda::pipeline|__pipeline_|memcpy_async": "async pipeline API (sm_80+)",
    r"__nv_bfloat|bfloat16": "bfloat16 (sm_80+)",
    r"wgmma": "wgmma (sm_90+)",
    r"mbarrier|CUtensorMap|\btma_": "mbarrier / TMA (sm_90+)",
    r"ldmatrix|__hmma|mma\.sync": "tensor-core MMA (sm_70+, but not used here)",
    r"__reduce_add_sync|redux\.sync": "warp reduce intrinsics (sm_80+)",
}

def main():
    files = (sorted(glob.glob(os.path.join(ROOT, "kernels", "*.cu")))
             + sorted(glob.glob(os.path.join(ROOT, "kernels", "*.cuh")))
             + [os.path.join(ROOT, "extension", "bindings.cpp")])
    findings = []
    for path in files:
        text = open(path).read()
        text = re.sub(r"//[^\n]*", "", text)
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
        for pattern, label in SM80_PLUS.items():
            for m in re.finditer(pattern, text):
                line = text[:m.start()].count("\n") + 1
                findings.append((os.path.relpath(path, ROOT), line, label))

    if findings:
        print("FOUND features newer than sm_75:")
        for f, line, label in findings:
            print(f"  {f}:{line}  {label}")
        return 1
    print(f"clean: {len(files)} files use nothing newer than sm_75")
    print("intrinsics in use: __align__, __syncthreads, fmaf, fmaxf, expf, rsqrtf")
    return 0

if __name__ == "__main__":
    sys.exit(main())
