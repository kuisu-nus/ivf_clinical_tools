"""Clean, split and summarise the oocyte-yield data.

    python -m ovu.pipeline <raw.csv> [--config ovu/config/data.yaml] [--out outputs/ovu]

Writes to --out (patient-level files; the directory is git-ignored):
    train.csv, val.csv          cleaned cycles, no IDs
    sealed/test.csv             the fixed test set; sealed/test_ids.txt pins it for later runs
    report.md, report.json      aggregate statistics only (safe to share)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ovu.pipeline import clean as cleaning
from ovu.pipeline import split, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("raw")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config" / "data.yaml"))
    parser.add_argument("--out", default="outputs/ovu")
    args = parser.parse_args()

    config = cleaning.load_config(args.config)
    data, log = cleaning.clean(cleaning.read_raw(args.raw), config)
    out_dir = Path(args.out)
    sealed = split.read_sealed(out_dir)
    parts = split.assign(data, config, sealed)
    sizes = split.write_split(data, parts, out_dir, int(config["split"].get("folds", 5)))
    report = stats.build_report(data, parts, config, log)
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "report.md").write_text(stats.to_markdown(report), encoding="utf-8")
    print(f"清洗后 {len(data)} 例：train {sizes['train']}，val {sizes['val']}，test {sizes['test']}"
          f"（测试集{'沿用已封存的' if sealed is not None else '本次新封存'}）")
    print(f"报告：{out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
