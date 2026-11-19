"""Build the case database from the configs and the training set.

    python -m ovu.db.build [--data outputs/ovu] [--config ovu/config/db.yaml] [--schema-only]
                           [--data-config ovu/config/data.yaml] [--model-config ovu/config/model.yaml]

1. Resolves db.yaml against data.yaml: which columns, their types (int -> INT, float -> DOUBLE,
   category -> VARCHAR, diagnosis categories -> TINYINT) and Chinese names; a field removed
   from data.yaml is left out.
2. Runs every query example on an in-memory copy of the training set; an example that uses a
   removed column is dropped, any other failure stops the build.
3. With ``backend: mysql`` (db.yaml, or OVU_DB_BACKEND=mysql): creates the databases, the table,
   the per-fold views and the read-only accounts on the MySQL server and loads train.csv
   (connection settings: see ovu/db/mysql.py). --schema-only skips this step.

Writes <data>/db/db.json (columns, limits and examples; no patient rows), <data>/db/overview.json
(the aggregate overview) and <data>/db/schema.sql (the CREATE statements, for a DBA to read or run).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

from ovu.db.database import CaseDatabase, records, resolve
from ovu.db.mysql import accounts, schema
from ovu.ml.train import read_rows
from ovu.pipeline.clean import available_fields, bin_labels, load_config

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="outputs/ovu")
    parser.add_argument("--config", default=str(ROOT / "config" / "db.yaml"))
    parser.add_argument("--data-config", default=str(ROOT / "config" / "data.yaml"))
    parser.add_argument("--model-config", default=str(ROOT / "config" / "model.yaml"))
    parser.add_argument("--schema-only", action="store_true", help="只生成 db.json 和 schema.sql，不连接 MySQL")
    args = parser.parse_args()

    data_config = load_config(args.data_config)
    model_config = yaml.safe_load(Path(args.model_config).read_text(encoding="utf-8"))
    bins = model_config["label"].get("bins") or data_config["label"]["bins"]
    spec = resolve(yaml.safe_load(Path(args.config).read_text(encoding="utf-8")), data_config, available_fields(data_config),
                   bin_labels(bins), bins)
    spec["backend"] = backend(spec)
    rows = read_rows(Path(args.data) / "train.csv")
    database = CaseDatabase(spec, rows)
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
    folds = sorted({int(r["cv_fold"]) for r in rows if r.get("cv_fold", "") != ""})
    statements = schema(spec, folds)
    accounts_sql = accounts(spec, folds, "<OVU_MYSQL_PASSWORD>")
    (out / "schema.sql").write_text(
        "-- 由 python -m ovu.db.build 按 db.yaml 和 data.yaml 生成；数据由该命令导入\n"
        + ";\n\n".join(statements) + ";\n\n-- 只读账号（每个账号只能读自己的视图库）：\n"
        + "\n".join(f"-- {a};" for a in accounts_sql) + "\n", encoding="utf-8")
    print(f"数据库配置：{out / 'db.json'}（{len(spec['columns'])} 列，{database.n} 例，{len(usable)} 条示例均可执行）；"
          f"建表语句：{out / 'schema.sql'}")
    if spec["backend"] != "mysql" or args.schema_only:
        return
    from ovu.db import mysql
    config = mysql.settings()
    values = records(spec, rows)
    notes = mysql.load(spec, values, [None if r.get("cv_fold", "") == "" else int(r["cv_fold"]) for r in rows], folds, config)
    view_db, reader = mysql.names(spec)
    print(f"已导入 MySQL：{config['host']}:{config['port']}，库 {view_db}（另有 {len(folds)} 个分折视图库），"
          f"{len(values)} 例；只读账号 {reader}")
    if config["reader_password"] and not notes:
        live = CaseDatabase.connect(spec)
        for example in usable:
            live.sql(example["sql"])
        print(f"用只读账号连上了 MySQL，{len(usable)} 条查询示例都能执行")
    for note in notes:
        print(f"提示：{note}", file=sys.stderr)


def backend(spec: dict) -> str:
    chosen = os.environ.get("OVU_DB_BACKEND") or spec.get("backend", "mysql")
    if chosen not in ("mysql", "sqlite"):
        raise SystemExit(f"backend 必须是 mysql 或 sqlite，不是 {chosen}")
    return chosen


if __name__ == "__main__":
    main()
