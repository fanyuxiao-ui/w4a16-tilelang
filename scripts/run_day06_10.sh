#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH="/usr/local/cuda-12.8/bin:${PATH}"

python -m compileall -q kernels baseline benchmarks scripts tests
pytest -q tests

python benchmarks/sweep_dequant_v2.py
python benchmarks/autotune_dequant_v2.py
python benchmarks/bench_w4a16.py --output day06_w4a16_all.json --swizzle

python benchmarks/bench_cuda_vs_tilelang.py --output day07_cuda_vs_tilelang.json
python benchmarks/bench_int4pack_mm.py

(
  cd ../tilelang/examples/dequantize_gemm
  python - <<'PY'
from example_dequant_gemv_fp16xint4 import run_regression_perf
print(f"official fp16xint4 GEMV 8192x8192: {run_regression_perf():.9f} ms")
PY
)

python benchmarks/bench_decode.py --output day09_decode.json
bash scripts/download_moe.sh
python scripts/moe_quant_validation.py

echo "Day 06-10 completed; JSON results are under benchmarks/results/."
