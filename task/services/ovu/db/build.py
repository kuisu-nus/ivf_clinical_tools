"""Resolve the database config and check every query example against the training set.

    python -m ovu.db.build [--data outputs/ovu] [--config ovu/config/db.yaml]
                           [--data-config ovu/config/data.yaml] [--model-config ovu/config/model.yaml]

Writes <data>/db/db.json (columns, limits and examples; no patient rows) and
<data>/db/overview.json (the aggregate overview the agent gets from the overview endpoint).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from ovu.db.database import CaseDatabase, resolve
from ovu.ml.train import read_rows
from ovu.pipeline.clean import bin_labels, feature_stages, load_config

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="outputs/ovu")
    parser.add_argument("--config", default=str(ROOT / "config" / "db.yaml"))
    parser.add_argument("--data-config", default=str(ROOT / "config" / "data.yaml"))
    parser.add_argument("--model-config", default=str(ROOT / "config" / "model.yaml"))
    args = parser.parse_args()

    data_config = load_config(args.data_config)
    model_config = yaml.safe_load(Path(args.model_config).read_text(encoding="utf-8"))
    bins = model_config["label"].get("bins") or data_config["label"]["bins"]
    spec = resolve(yaml.safe_load(Path(args.config).read_text(encoding="utf-8")), data_config, feature_stages(data_config),
                   bin_labels(bins), bins)
    database = CaseDatabase(spec, read_rows(Path(args.data) / "train.csv"))
    usable = []
    for example in spec["examples"]:
        try:
            database.sql(example["sql"])
            usable.append(example)
        except ValueError as exc:
            if "no such column" not in str(exc):
                raise SystemExit(f"查询示例“{example['title']}”执行失败：{exc}") from None
            print(f"提示：查询示例“{example['title']}”用到了已不存在的列，已跳过（{exc}）", file=sys.stderr)
    spec["examples"] = database.spec["examples"] = usable
    out = Path(args.data) / "db"
    out.mkdir(parents=True, exist_ok=True)
    (out / "db.json").write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "overview.json").write_text(json.dumps(database.overview(), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"数据库配置：{out / 'db.json'}（{len(spec['columns'])} 列，{database.n} 例，{len(usable)} 条示例均可执行）")


if __name__ == "__main__":
    main()
