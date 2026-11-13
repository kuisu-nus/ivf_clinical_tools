"""Fixed train / val / test split.

A cycle's place depends only on ``sha256(seed:cycle_id)``, never on row order.
Within each label bin the cycles are sorted by that hash; the first ``test``
share goes to the test set and the next ``val`` share to validation.

The test set is sealed: the first run writes the hashes of its cycles to
``sealed/test_ids.txt``; every later run reads that file, so the test set stays
the same when the config, the cleaning rules or the bins change. A cycle that
was not in the data when the test set was sealed can only go to train or val.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

SEALED_DIR = "sealed"
TEST_IDS = "test_ids.txt"


def assign(data: pd.DataFrame, config: dict[str, Any], sealed_keys: set[str] | None = None) -> pd.Series:
    """'train' / 'val' / 'test' for every row of the cleaned table."""
    spec = config["split"]
    strata = data[config["label"]["name"] + "_bin"] if spec.get("stratify_by_label_bin", True) else pd.Series(0, index=data.index)
    result = pd.Series("train", index=data.index)
    if sealed_keys is not None:
        result[data["_key"].isin(sealed_keys)] = "test"
    for _, group in data.groupby(strata):
        ordered = group.sort_values("_key")
        if sealed_keys is None:
            n_test = round(len(ordered) * spec["test"])
            result[ordered.index[:n_test]] = "test"
        rest = ordered[result[ordered.index] != "test"]
        result[rest.index[:round(len(ordered) * spec["val"])]] = "val"
    return result


def read_sealed(out_dir: Path) -> set[str] | None:
    path = out_dir / SEALED_DIR / TEST_IDS
    return set(path.read_text().split()) if path.exists() else None


def write_split(data: pd.DataFrame, parts: pd.Series, out_dir: Path, folds: int = 5) -> dict[str, int]:
    """train.csv, val.csv and sealed/test.csv, without IDs and ordered by hash.

    Every row gets ``cv_fold`` (0..folds-1), fixed by the same hash: when an agent is
    trained on a training case, the tools hide that case's fold so its own outcome
    never comes back in a tool result. ``cv_fold`` is bookkeeping, never a feature."""
    data = data.assign(cv_fold=[int(key[-8:], 16) % folds for key in data["_key"]])
    (out_dir / SEALED_DIR).mkdir(parents=True, exist_ok=True)
    ids_path = out_dir / SEALED_DIR / TEST_IDS
    if not ids_path.exists():
        ids_path.write_text("\n".join(sorted(data.loc[parts == "test", "_key"])) + "\n")
    sizes = {}
    for part, path in (("train", out_dir / "train.csv"), ("val", out_dir / "val.csv"),
                       ("test", out_dir / SEALED_DIR / "test.csv")):
        rows = data[parts == part].sort_values("_key")
        rows = rows[[c for c in rows.columns if not c.startswith("_")]]
        rows.to_csv(path, index=False, encoding="utf-8")
        sizes[part] = len(rows)
    return sizes
