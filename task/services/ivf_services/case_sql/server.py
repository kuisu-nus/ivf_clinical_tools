"""IVF training-case database, read-only SQL (FastAPI on port 9204).

  POST /ivf/case_sql           one read-only SELECT on the case database (tool ivf_case_sql)
  GET  /health                 status, request counts and load
  GET  /ivf/audit              tool calls since start;  POST /ivf/reset clears them

The database is in memory and holds only the visible cases (no IDs, the fold of the case under
evaluation removed); writes, ATTACH and PRAGMA are refused, queries stop after 2 s, at most 200 rows.

Environment: as ivf_services/cases/server.py, with IVF_CASE_SQL_PORT (9204).
"""

from __future__ import annotations

import asyncio
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from clinical_tools.case_db import COLUMNS, OUTCOME, SQL_MAX_ROWS, SQL_TIMEOUT_S
from ivf_services._base import ServiceKit, run
from ivf_services.case_db_loader import load_database

SERVICE, TOOL, ENDPOINT = "ivf_case_sql", "ivf_case_sql", "/ivf/case_sql"
PORT_ENV, DEFAULT_PORT = "IVF_CASE_SQL_PORT", 9204
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.db, table = await asyncio.to_thread(load_database)
    except Exception:
        kit.log.exception("failed to start: could not load the case table")
        raise
    kit.health_info.update(scope=app.state.db.scope, cases=int(len(app.state.db.labels)))
    kit.log.info("started: %s cases (%s) from %s", len(app.state.db.labels), app.state.db.scope, table)
    yield
    kit.shutdown()


app = FastAPI(title="IVF Case Database (SQL)", lifespan=lifespan)
kit = ServiceKit(app, SERVICE, TOOL)


@app.post(ENDPOINT)
async def case_sql(req: CaseSqlRequest):
    try:
        resp: dict[str, Any] = await kit.compute(app.state.db.sql, req.sql)
    except ValueError as exc:
        body = {"error": "invalid_sql", "message": str(exc)}
        kit.record(ENDPOINT, req.model_dump(), body, status=400)
        return JSONResponse(status_code=400, content=body)
    kit.record(ENDPOINT, req.model_dump(), resp, summary=f"rows={resp['返回行数']}")
    return resp


if __name__ == "__main__":
    run(app, PORT_ENV, DEFAULT_PORT)
