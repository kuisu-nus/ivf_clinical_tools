"""Resolve the retrieval config against the training set.

    python -m ovu.similar.build [--data outputs/ovu] [--config ovu/config/similar.yaml]
                                [--data-config ovu/config/data.yaml] [--model-config ovu/config/model.yaml]

Writes <data>/similar/similar.json (weights and scales only, no patient rows).
The oocyte-count groups are the model's (model.yaml), so both tools speak of the same groups.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from ovu.ml.train import read_rows
from ovu.pipeline.clean import bin_labels, load_config
from ovu.similar.index import SimilarIndex, resolve

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="outputs/ovu")
    parser.add_argument("--config", default=str(ROOT / "config" / "similar.yaml"))
    parser.add_argument("--data-config", default=str(ROOT / "config" / "data.yaml"))
    parser.add_argument("--model-config", default=str(ROOT / "config" / "model.yaml"))
    args = parser.parse_args()

    data_config = load_config(args.data_config)
    model_config = yaml.safe_load(Path(args.model_config).read_text(encoding="utf-8"))
    bins = model_config["label"].get("bins") or data_config["label"]["bins"]
    rows = read_rows(Path(args.data) / "train.csv")
    spec = resolve(yaml.safe_load(Path(args.config).read_text(encoding="utf-8")), rows, data_config, bins, bin_labels(bins))
    out = Path(args.data) / "similar"
    out.mkdir(parents=True, exist_ok=True)
    (out / "similar.json").write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
    SimilarIndex(spec, rows)                                         # fails here if a configured field is not in train.csv
    print(f"检索配置：{out / 'similar.json'}（{len(spec['features'])} 个比对特征，{len(rows)} 例训练病例）")


if __name__ == "__main__":
    main()
