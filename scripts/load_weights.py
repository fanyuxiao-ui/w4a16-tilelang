"""从 DeepSeek-Coder-6.7B 的 safetensors 分片读取单个权重。"""
import json
import os
from pathlib import Path

from safetensors import safe_open


DEFAULT_MODEL_DIR = Path(os.path.expanduser("~/autodl-tmp/models/deepseek-coder-6.7b-base"))
_INDEX_CACHE = {}


def _weight_map(model_dir: Path):
    model_dir = Path(model_dir).expanduser()
    key = str(model_dir.resolve())
    if key not in _INDEX_CACHE:
        index_path = model_dir / "model.safetensors.index.json"
        if not index_path.exists():
            raise FileNotFoundError(
                f"找不到模型索引 {index_path}；请用 --model-dir 指向 DeepSeek-Coder-6.7B 权重目录"
            )
        _INDEX_CACHE[key] = json.loads(index_path.read_text())["weight_map"]
    return _INDEX_CACHE[key]


def load_tensor(name: str, model_dir=DEFAULT_MODEL_DIR, device=None):
    """按张量名读取权重；device=None 时跟随 safetensors 的默认 CPU 路径。"""
    model_dir = Path(model_dir).expanduser()
    weight_map = _weight_map(model_dir)
    if name not in weight_map:
        examples = list(weight_map)[:3]
        raise KeyError(f"{name!r} 不在权重索引中；示例：{examples}")
    path = model_dir / weight_map[name]
    kwargs = {"framework": "pt"}
    if device is not None:
        kwargs["device"] = device
    with safe_open(str(path), **kwargs) as handle:
        return handle.get_tensor(name)
