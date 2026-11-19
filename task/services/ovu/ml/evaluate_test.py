"""The one-time evaluation of the final model on the sealed test set.

    python -m ovu.ml.evaluate_test [--data outputs/ovu] [--model outputs/ovu/model/model.json] [--again]

Run this once, after every choice about features, groups and settings is final: the test
set is there to give an unbiased number, and looking at it again after changing something
turns it into a second validation set. Each run is recorded in sealed/test_evaluations.json
(time and the model file's hash, no patient data); when a record exists the script stops
unless --again is given, and the report then says how many times the test set has been used.

Writes next to the model: test_report.json (metrics against the baseline, confusion matrix,
calibration by group, permutation importance on the test set; aggregates only).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ovu.ml.evaluation import evaluate, permutation_importance, read_rows

LOG = "test_evaluations.json"


def evaluate_test(test_rows: list[dict[str, Any]], artifact: dict[str, Any], repeats: int = 5, seed: int = 42) -> dict[str, Any]:
    return {"test": evaluate(test_rows, artifact),
            "permutation_importance": permutation_importance(test_rows, artifact, repeats, seed)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="outputs/ovu")
    parser.add_argument("--model", default=None)
    parser.add_argument("--again", action="store_true", help="测试集已经用过时仍然再评估一次（会记录在案）")
    args = parser.parse_args()

    data_dir = Path(args.data)
    model_path = Path(args.model or data_dir / "model" / "model.json")
    log_path = data_dir / "sealed" / LOG
    history = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else []
    if history and not args.again:
        last = history[-1]
        sys.exit(f"测试集已经在 {last['evaluated_utc']} 评估过（共 {len(history)} 次），结果在 {last['report']}。"
                 "测试集只应在最终评估时用一次；确实要再评估，加 --again（会记录在案）。")

    artifact = json.loads(model_path.read_text(encoding="utf-8"))
    test_rows = read_rows(data_dir / "sealed" / "test.csv")
    report = evaluate_test(test_rows, artifact)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()[:16]
    out = model_path.with_name("test_report.json")
    report["record"] = {"evaluated_utc": now, "model_sha256": digest, "n": len(test_rows), "times_used": len(history) + 1}
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    history.append({"evaluated_utc": now, "model_sha256": digest, "n": len(test_rows), "report": str(out)})
    log_path.write_text(json.dumps(history, ensure_ascii=False, indent=1), encoding="utf-8")

    m, b = report["test"]["model"], report["test"]["baseline"]
    print(f"测试集 {len(test_rows)} 例：RPS {m['rps']}（基线 {b['rps']}），加权 kappa {m['quadratic_kappa']}，"
          f"准确率 {m['accuracy']}，相邻组以内 {m['within_one']}")
    print(f"结果：{out}；这是第 {len(history)} 次使用测试集")


if __name__ == "__main__":
    main()
