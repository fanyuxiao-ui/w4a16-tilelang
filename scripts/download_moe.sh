#!/usr/bin/env bash
set -euo pipefail

# Qwen1.5-MoE-A2.7B is roughly 29 GB in bf16.  Keep this guard so a repeat
# run does not fill a small workspace halfway through a benchmark.
MODEL_DIR="${1:-${HOME}/autodl-tmp/models/qwen-moe-a2.7b}"
FREE_GB=$(df -Pk "$(dirname "$MODEL_DIR")" | awk 'NR==2 {printf "%d", $4/1024/1024}')
if [ "$FREE_GB" -lt 35 ]; then
  echo "skip Qwen1.5-MoE download: ${FREE_GB} GB free, at least 35 GB required"
  echo "run scripts/moe_quant_validation.py for the deterministic ToyMoE fallback"
  exit 0
fi

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
huggingface-cli download Qwen/Qwen1.5-MoE-A2.7B --local-dir "$MODEL_DIR"
