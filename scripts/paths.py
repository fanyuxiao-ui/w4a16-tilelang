"""Configurable paths used by model-facing scripts."""

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(
    os.environ.get(
        "W4A16_MODEL_DIR",
        os.path.expanduser("~/autodl-tmp/models/deepseek-coder-6.7b-base"),
    )
).expanduser()
RESULT_DIR = Path(os.environ.get("W4A16_RESULT_DIR", str(ROOT / "benchmarks/results")))
