#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."

SIZE="${1:-4096}"
OUT="bench/results/ncu_sgemm_${SIZE}"

cat > /tmp/ncu_target.py <<'PY'
import sys, torch
sys.path.insert(0, ".")
from extension.build import load_extension
ext = load_extension()
n = int(sys.argv[1])
A = torch.randn(n, n, device="cuda"); B = torch.randn(n, n, device="cuda")
for _ in range(3):
    ext.sgemm(A, B, "vectorized")
torch.cuda.synchronize()
PY

echo "Profiling sgemm_vectorized_kernel at ${SIZE}x${SIZE}..."
ncu --set full \
    --kernel-name regex:"sgemm_vectorized_kernel" \
    --launch-count 1 \
    --export "${OUT}" \
    --force-overwrite \
    python /tmp/ncu_target.py "${SIZE}"
status=$?

if [ $status -ne 0 ]; then
  echo
  echo "ncu failed (exit ${status}). If this is a permissions error, the host"
  echo "has GPU performance counters disabled. Falling back to the compile-time"
  echo "occupancy analysis, which needs no counters:"
  echo
  python bench/occupancy.py
  exit 0
fi

echo "Wrote ${OUT}.ncu-rep"
ncu --import "${OUT}.ncu-rep" --page details | head -80
