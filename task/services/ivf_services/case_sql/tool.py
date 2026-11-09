"""Tool ivf_case_sql: one read-only SELECT on the training-case database (clinical_tools.case_db).
Served by ivf_services/server.py at POST /tools/ivf_case_sql.

The database is in memory and holds only the visible cases (no IDs, the fold of the case under
evaluation removed); writes, ATTACH and PRAGMA are refused, queries stop after 2 s, at most 200 rows.
A refused or failing statement is a client error (400, invalid_sql).

Environment:
  IVF_CASE_TABLE    case table (default: services/data/train_cases_35.csv, in the repository, no IDs)
  IVF_EXCLUDE_CASE  training case under evaluation: its fold, looked up in IVF_TRAIN_CSV (ID-bearing,
                    default services/data/train_age_amh_gn.csv), is hidden, as in the other tools
  IVF_EXCLUDE_FOLD  hide this fold instead
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from clinical_tools.case_db import COLUMNS, OUTCOME, SQL_MAX_ROWS, SQL_TIMEOUT_S, CaseDatabase, fold_of

TOOL = "ivf_case_sql"
TOOL_DESCRIPTION = (
    "在去掉病例编号的 IVF-ET 训练病例库上执行一条只读 SQL（SQLite 语法，只允许 SELECT）。"
    f"表 cases：每行一个训练病例，列为 35 项指标和 {OUTCOME}（1 妊娠，0 未妊娠），缺失为 NULL；"
    "表 fields：列名与中文指标名对照。列名："
    + "、".join(f"{c}（{n}）" for n, c in COLUMNS.items())
    + f"。最多返回 {SQL_MAX_ROWS} 行，单次查询超过 {SQL_TIMEOUT_S:g} 秒中止；尽量用 COUNT、AVG、GROUP BY 等聚合。"
    "例：SELECT CASE WHEN age < 35 THEN '<35' ELSE '≥35' END AS g, COUNT(*), AVG(pregnant) FROM cases GROUP BY g"
)


class CaseSqlRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sql: str = Field(..., min_length=1, max_length=4000, description="一条只读 SELECT 语句（SQLite）")


REQUEST_MODEL = CaseSqlRequest


STATE = "case_db"
CLIENT_ERROR = "invalid_sql"   # a ValueError from call() is answered 400 with this error code
DATA = Path(__file__).resolve().parents[2] / "data"


def load() -> CaseDatabase:
    table = os.environ.get("IVF_CASE_TABLE") or DATA / "train_cases_35.csv"
    fold, case = os.environ.get("IVF_EXCLUDE_FOLD"), os.environ.get("IVF_EXCLUDE_CASE")
    if fold:
        exclude = int(fold)
    elif case:
        exclude = fold_of(case, os.environ.get("IVF_TRAIN_CSV") or DATA / "train_age_amh_gn.csv")
    else:
        exclude = None
    return CaseDatabase(table, exclude_fold=exclude)


def health(db: CaseDatabase) -> dict[str, Any]:
    return {"scope": db.scope, "cases": len(db.labels)}


def call(db: CaseDatabase, req: CaseSqlRequest) -> dict[str, Any]:
    return db.sql(req.sql)


def summary(resp: dict[str, Any]) -> str:
    return f"rows={resp['返回行数']}"
