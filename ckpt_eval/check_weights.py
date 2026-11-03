"""Check that a converted Hugging Face model has exactly the weight names of the original model.

  python ckpt_eval/check_weights.py <origin_hf_dir> <converted_dir>     (exit 1 on any difference)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def weight_names(model_dir: Path) -> set[str]:
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        return set(json.loads(index.read_text(encoding="utf-8"))["weight_map"])
    from safetensors import safe_open        # single-file models have no index
    with safe_open(str(model_dir / "model.safetensors"), framework="pt") as f:
        return set(f.keys())


def main(argv: list[str] | None = None) -> int:
    origin, converted = (Path(p) for p in (argv or sys.argv[1:]))
    a, b = weight_names(origin), weight_names(converted)
    missing, extra = sorted(a - b), sorted(b - a)
    print(f"weights: original {len(a)}, converted {len(b)}, missing {missing[:5]}, extra {extra[:5]}")
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        if (origin / name).exists() and not (converted / name).exists():
            print(f"missing file: {name}")
            return 1
    return 0 if not missing and not extra else 1


if __name__ == "__main__":
    raise SystemExit(main())
