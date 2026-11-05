"""IVF training-case database, structured queries (FastAPI on port 9203).

  POST /ivf/query_cases        filter training cases on any indicator, group, describe (tool ivf_query_cases)
  GET  /health                 status, request counts and load
  GET  /ivf/audit              tool calls since start;  POST /ivf/reset clears them

Environment:
  IVF_CASE_TABLE     case table without IDs (default: services/data/train_cases_35.csv)
  IVF_EXCLUDE_CASE   training case under evaluation (agentic RL); its fold is hidden (IVF_TRAIN_CSV gives its fold)
  IVF_EXCLUDE_FOLD   alternatively the fold to hide
  IVF_CASES_PORT     9203
  Logging and concurrency: see ivf_services/_base.py.
"""

from __future__ import annotations

import asyncio
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, model_validator

from clinical_tools.case_db import MIN_COHORT
from clinical_tools.common import FEATURE_NAMES
from ivf_services._base import ServiceKit, run
from ivf_services.case_db_loader import load_database

SERVICE, TOOL, ENDPOINT = "ivf_cases", "ivf_query_cases", "/ivf/query_cases"
PORT_ENV, DEFAULT_PORT = "IVF_CASES_PORT", 9203
TOOL_DESCRIPTION = (
    "查询去掉病例编号的 IVF-ET 训练病例库（35 项指标和临床妊娠结局）：按任意指标的条件筛选人群，"
    "可再按某个指标分段统计各段的妊娠率，并查看指标在人群中（以及妊娠者、未妊娠者中）的分布。"
    "适合自己探索哪些指标、在什么范围内与结局相关，以及当前患者所在人群的情况。"
    f"人群或分段少于 {MIN_COHORT} 例时不报告妊娠率。病例未提供的指标不要拿来筛选，不要编造数值。"
)
Field_ = Literal[tuple(FEATURE_NAMES)]  # type: ignore[valid-type]


class CaseFilter(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: Field_ = Field(..., description="指标名（训练数据中的中文指标名，如 女方AMH）")
    op: Literal[">", ">=", "<", "<=", "=", "between", "missing", "present"] = Field(
        ..., description="比较方式；between 需要 value 和 value2（含两端）；missing / present 表示该指标缺失 / 有记录")
    value: float | None = Field(None, description="比较值")
    value2: float | None = Field(None, description="between 的上界")

    @model_validator(mode="after")
    def values_match_op(self) -> CaseFilter:
        if self.op in ("missing", "present"):
            return self
        if self.value is None or isinstance(self.value, bool):
            raise ValueError(f"op {self.op} 需要数值 value")
        if self.op == "between" and (self.value2 is None or self.value2 < self.value):
            raise ValueError("between 需要 value2，且 value2 ≥ value")
        return self


class GroupBy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: Field_ = Field(..., description="分段的指标名")
    cuts: list[float] = Field(..., min_length=1, max_length=9,
                              description="分段点，例如 [30, 35, 40] 分为 <30、30–35、35–40、≥40 四段")


class QueryCasesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    filters: list[CaseFilter] = Field(default_factory=list, max_length=10,
                                      description="筛选条件（同时满足），不填表示全部可见病例")
    group_by: GroupBy | None = Field(None, description="按某个指标分段，统计每段的例数和妊娠率")
    describe: list[Field_] = Field(default_factory=list, max_length=10,
                                   description="要查看分布的指标（中位数、四分位数、缺失数，以及妊娠者与未妊娠者的中位数）")


REQUEST_MODEL = QueryCasesRequest


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


app = FastAPI(title="IVF Case Database (queries)", lifespan=lifespan)
kit = ServiceKit(app, SERVICE, TOOL)


@app.post(ENDPOINT)
async def query_cases(req: QueryCasesRequest) -> dict[str, Any]:
    body = req.model_dump()
    resp = await kit.compute(app.state.db.query, body["filters"], body["group_by"], body["describe"])
    kit.record(ENDPOINT, body, resp, summary=f"matched={resp['符合条件例数']}")
    return resp


if __name__ == "__main__":
    run(app, PORT_ENV, DEFAULT_PORT)
