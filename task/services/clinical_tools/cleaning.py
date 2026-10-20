"""Outcome-blind cleaning of the training CSV for the clinical tools.

The golden data replaced missing values with per-outcome class means. Any value
equal to one of those fills becomes missing whatever the case's outcome, so no
rule reads the label; implausible values become missing too. Running it on v2
data is safe: values already removed stay removed.

  python -m clinical_tools.cleaning --input xgb_train_1132.csv --out-dir OUT

OUT/train_clean.csv keeps the input columns and rows (stays local, never
committed); OUT/cleaning_summary.json holds only counts per rule and column.
Only the 1132-case training set is accepted; the sealed test set is refused.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from clinical_tools.common import FEATURE_NAMES, assign_folds, parse_label, parse_value

EXPECTED_TRAIN_ROWS = 1132
# v2 exported this column as "处理前NP"; its meaning is unconfirmed either way.
ALIASES = {"男方精液处理前NP": "男方精液处理后NP"}

# Both outcome-specific fill values per column, including the positive-class
# values 29.11 (男方PR) and 23.90 (男方NP) that purity-only scans miss.
OUTCOME_BLIND_FILLS = {
    "女方基础FSH": (6.64, 6.54), "女方基础PRL": (17.64, 17.98), "女方基础P": (1.39, 1.48),
    "女方AMH": (3.11, 3.39), "女方D二聚体": (0.21, 0.20), "女方甲功TPO": (24.34, 24.64),
    "女方空腹血糖": (5.05, 5.04), "女方肾功尿酸": (255.68,), "男方浓度": (65.35, 65.26),
    "男方正常形态率": (3.54, 3.51), "男方PR": (28.63, 29.11), "男方NP": (23.67, 23.90),
    "男方存活率": (53.58, 54.35), "男方空腹血糖": (5.12, 5.10),
    "男方精液处理后NP": (11.80, 12.09), "男方精液处理后密度": (37.55, 38.36),
    "男方精液处理后PR": (87.81, 87.84), "男方精液处理后IM": (9.44, 9.48),
}
SEMEN_ROUTINE = ("男方浓度", "男方PR", "男方NP", "男方存活率")
PERCENT_COLUMNS = ("男方正常形态率", "男方PR", "男方NP", "男方存活率",
                   "男方精液处理后PR", "男方精液处理后IM", "男方精液处理后NP")
RULES = ("结局回填值", "精液常规全为0", "肌酐疑似尿素", "PRL>200", "压积>60", "百分比超出0-100")


@dataclass(frozen=True)
class TrainingSet:
    case_ids: list[str]
    matrix: np.ndarray          # rows x FEATURE_NAMES, NaN for missing
    labels: np.ndarray

    def visible(self, exclude_case: str | None = None, k_folds: int = 5) -> tuple["TrainingSet", str]:
        """Cases a tool may use, and a description of that scope. With a case under
        evaluation (agentic RL on training cases), its whole fold is hidden."""
        if exclude_case is None:
            return self, "全部训练病例"
        if exclude_case not in self.case_ids:
            raise ValueError(f"exclude_case {exclude_case} is not a training case")
        folds = np.array(assign_folds(self.case_ids, self.labels.tolist(), k_folds))
        keep = folds != folds[self.case_ids.index(exclude_case)]
        subset = TrainingSet([c for c, k in zip(self.case_ids, keep) if k], self.matrix[keep], self.labels[keep])
        return subset, f"训练病例中排除当前病例所在的 1/{k_folds}"


def clean_matrix(x: np.ndarray) -> tuple[np.ndarray, Counter]:
    """Cleaned copy and counts keyed by (rule, column); never reads labels."""
    out = x.copy()
    counts: Counter = Counter()
    col = {name: i for i, name in enumerate(FEATURE_NAMES)}

    def null(mask: np.ndarray, name: str, rule: str) -> None:
        mask = mask & ~np.isnan(out[:, col[name]])
        counts[(rule, name)] += int(mask.sum())
        out[mask, col[name]] = np.nan

    with np.errstate(invalid="ignore"):
        for name, values in OUTCOME_BLIND_FILLS.items():
            null(np.isin(np.round(out[:, col[name]], 6), np.round(values, 6)), name, "结局回填值")
        zero = out[:, col["男方浓度"]] == 0
        for name in SEMEN_ROUTINE:
            null(zero, name, "精液常规全为0")
        null(out[:, col["男方血肌酐"]] < 20, "男方血肌酐", "肌酐疑似尿素")
        null(out[:, col["女方基础PRL"]] > 200, "女方基础PRL", "PRL>200")
        null(out[:, col["女方血细胞压积"]] > 60, "女方血细胞压积", "压积>60")
        for name in PERCENT_COLUMNS:
            values = out[:, col[name]]
            null((values < 0) | (values > 100), name, "百分比超出0-100")
    return out, counts


def _features(row: Mapping[str, object]) -> list[float]:
    row = {ALIASES.get(k, k): v for k, v in row.items()}
    missing = [n for n in FEATURE_NAMES if n not in row]
    if missing:
        raise ValueError(f"Rows lack features: {missing}")
    return [np.nan if (v := parse_value(row[n])[0]) is None else v for n in FEATURE_NAMES]


def rows_to_training(rows: Sequence[Mapping[str, object]], *, id_column: str = "id",
                     label_column: str = "label") -> TrainingSet:
    """Clean rows in memory (the service accepts raw golden, v2 or already-cleaned CSVs)."""
    ids = [str(r[id_column]) for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate case ids in training data")
    matrix, _ = clean_matrix(np.array([_features(r) for r in rows], dtype=float))
    return TrainingSet(ids, matrix, np.array([parse_label(r[label_column]) for r in rows]))


def load_training(path: str | Path) -> TrainingSet:
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return rows_to_training(list(csv.DictReader(handle)))


def _format(value: float) -> str:
    return "" if math.isnan(value) else f"{value:.10g}"


def clean_file(source: Path, out_dir: Path, *, expected_rows: int = EXPECTED_TRAIN_ROWS) -> dict:
    with open(source, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        header, rows = list(reader.fieldnames or []), list(reader)
    if len(rows) != expected_rows:
        raise SystemExit(f"Expected the {expected_rows}-case training set, got {len(rows)} rows: "
                         "the sealed test set must not be used here")
    raw = np.array([_features(r) for r in rows], dtype=float)
    clean, counts = clean_matrix(raw)
    out_dir.mkdir(parents=True, exist_ok=False)
    target = out_dir / "train_clean.csv"
    index = {ALIASES.get(h, h): h for h in header}
    with open(target, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for row, values in zip(rows, clean):
            out = dict(row)
            for name, value in zip(FEATURE_NAMES, values):
                out[index[name]] = _format(value)
            writer.writerow(out)
    summary = {
        "input": str(source), "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "rows": len(rows),
        "rules_read_labels": False,
        "changes_by_rule": {rule: sum(n for (r, _), n in counts.items() if r == rule) for rule in RULES},
        "changes_by_rule_and_column": {f"{r}|{c}": n for (r, c), n in sorted(counts.items()) if n},
        "missing_before": {n: int(np.isnan(raw[:, i]).sum()) for i, n in enumerate(FEATURE_NAMES)},
        "missing_after": {n: int(np.isnan(clean[:, i]).sum()) for i, n in enumerate(FEATURE_NAMES)},
    }
    (out_dir / "cleaning_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                                   encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True, help="training CSV (golden or v2, 1132 rows)")
    parser.add_argument("--out-dir", type=Path, required=True, help="new directory; existing ones are refused")
    args = parser.parse_args(argv)
    summary = clean_file(args.input, args.out_dir)
    print(json.dumps({"output": str(args.out_dir / "train_clean.csv"), "rows": summary["rows"],
                      "changes_by_rule": summary["changes_by_rule"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
