"""Write the shareable training file: cleaned public data, keeping only the three
prediction inputs (female age, AMH, Gn days) per case; every other indicator is
replaced by its training mean.

  python -m clinical_tools.public_data --input outputs/clean_v1/train_clean.csv \
      --out task/services/data/train_age_amh_gn.csv

Source: the public IVF-ET dataset (Mendeley x5s3ky9cpv), 1132-case training set
after the outcome-blind cleaning (clinical_tools.cleaning). The ids and labels are
kept: the similar-patient tool needs the outcomes, and the ids only seed hashing.
With this file the similar-patient statistics are the same as with the full
cleaned data (they use age, AMH and the outcome only); indicators requested through
extra_fields come back as the training mean for every case.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from clinical_tools.cleaning import EXPECTED_TRAIN_ROWS, clean_matrix, load_training
from clinical_tools.common import FEATURE_NAMES

KEPT = ("女方年龄", "女方AMH", "女方用药Gn总天数")


def _fmt(value: float, kept: bool) -> str:
    """Kept inputs exactly as in the source; means to 4 decimals."""
    if np.isnan(value):
        return ""
    return f"{value:.10g}" if kept else f"{value:.4f}".rstrip("0").rstrip(".")


def build(source: Path) -> list[dict[str, str]]:
    training = load_training(source)            # applies the cleaning again (idempotent)
    if len(training.labels) != EXPECTED_TRAIN_ROWS:
        raise SystemExit(f"expected the {EXPECTED_TRAIN_ROWS}-case training set, got {len(training.labels)} rows")
    matrix = training.matrix.copy()
    means = np.nanmean(matrix, axis=0)
    for i, name in enumerate(FEATURE_NAMES):
        if name not in KEPT:
            matrix[:, i] = means[i]
    cleaned, counts = clean_matrix(matrix)         # a mean must not look like a fill value or outlier
    if sum(counts.values()) or not np.array_equal(np.isnan(cleaned), np.isnan(matrix)):
        raise SystemExit(f"a filled mean is removed by the cleaning rules: {dict(counts)}")
    return [{"id": case_id, **{name: _fmt(matrix[r, i], name in KEPT)
                               for i, name in enumerate(FEATURE_NAMES)}, "label": str(int(training.labels[r]))}
            for r, case_id in enumerate(training.case_ids)]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True, help="cleaned 1132-case training CSV")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = build(args.input)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", *FEATURE_NAMES, "label"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"{args.out}: {len(rows)} cases; kept {', '.join(KEPT)}; other indicators set to the training mean")


if __name__ == "__main__":
    main()
