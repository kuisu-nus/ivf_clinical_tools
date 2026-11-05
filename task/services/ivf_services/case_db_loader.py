"""Loading the case database for the two case-database services (same rules for both)."""

from __future__ import annotations

import os
from pathlib import Path

from clinical_tools.case_db import CaseDatabase, fold_of

DEFAULT_TABLE = Path(__file__).resolve().parent.parent / "data" / "train_cases_35.csv"
DEFAULT_TRAIN_CSV = Path(__file__).resolve().parent.parent / "data" / "train_age_amh_gn.csv"


def load_database() -> tuple[CaseDatabase, str]:
    """IVF_CASE_TABLE (default: the table in the repository). The fold of the case under evaluation is
    hidden: IVF_EXCLUDE_FOLD, or the fold of IVF_EXCLUDE_CASE looked up in IVF_TRAIN_CSV (ID-bearing)."""
    table = os.environ.get("IVF_CASE_TABLE") or str(DEFAULT_TABLE)
    fold = os.environ.get("IVF_EXCLUDE_FOLD")
    case = os.environ.get("IVF_EXCLUDE_CASE")
    if fold:
        exclude = int(fold)
    elif case:
        exclude = fold_of(case, os.environ.get("IVF_TRAIN_CSV") or DEFAULT_TRAIN_CSV)
    else:
        exclude = None
    return CaseDatabase(table, exclude_fold=exclude), table
