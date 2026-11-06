"""Tool ivf_case_sql: one read-only SELECT on the training-case database (clinical_tools.case_db).
Served by ivf_services/server.py at POST /tools/ivf_case_sql; shares the database with ivf_query_cases.

The database is in memory and holds only the visible cases (no IDs, the fold of the case under
evaluation removed); writes, ATTACH and PRAGMA are refused, queries stop after 2 s, at most 200 rows.
A refused or failing statement is a client error (400, invalid_sql). Environment: as cases/tool.py.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from clinical_tools.case_db import COLUMNS, OUTCOME, SQL_MAX_ROWS, SQL_TIMEOUT_S
from clinical_tools.case_db import CaseDatabase
from ivf_services.case_db_loader import load_database

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


def load() -> CaseDatabase:
    return load_database()[0]


def health(db: CaseDatabase) -> dict[str, Any]:
    return {"scope": db.scope, "cases": int(len(db.labels))}


def call(db: CaseDatabase, req: CaseSqlRequest) -> dict[str, Any]:
    return db.sql(req.sql)


def summary(resp: dict[str, Any]) -> str:
    return f"rows={resp['返回行数']}"
