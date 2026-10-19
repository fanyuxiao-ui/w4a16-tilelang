#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH="/usr/local/cuda-12.8/bin:${PATH}"

if [[ -z "${W4A16_MODEL_DIR:-}" ]]; then
  echo "W4A16_MODEL_DIR must point to the local DeepSeek-Coder-6.7B directory" >&2
  exit 2
fi

python -m compileall -q kernels baseline benchmarks scripts tests
pytest -q tests
python scripts/plot_decode.py

# The model-facing steps require W4A16_MODEL_DIR and an A800-class GPU.
python scripts/bench_e2e.py --new-tokens 8 --warmup 1 --repeats 2
python scripts/ppl_eval.py --max-tokens 128

python - <<'PY'
from pathlib import Path
import subprocess

root = Path.cwd()
patterns = ("/" + "root" + "/", "/" + "home" + "/", "token" + "=", "api" + "_key" + "=", "BEGIN " + "PRIVATE KEY")
for pattern in patterns:
    cmd = ["rg", "-n", pattern, "--glob", "*.py", "--glob", "*.md", "--glob", "*.sh", "."]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    if result.returncode == 0:
        print(f"scan match ({pattern}):\n{result.stdout[:1000]}")
print("release scans completed")
PY

echo "Day 11-14 local execution completed. External blog publication requires account-side posting."
