"""IVF similar-patient retrieval API (FastAPI on port 9201).

  POST /ivf/similar_patients   similar training cases by female age and AMH (tool ivf_similar_patients)
  GET  /health                 status, request counts and load
  GET  /ivf/audit              tool calls since start;  POST /ivf/reset clears them

Environment:
  IVF_TRAIN_CSV      required; the cleaned 1132-case training CSV (never the sealed test set)
  IVF_SIMILAR_PORT   9201
  IVF_EXCLUDE_CASE   training case under evaluation (agentic RL); its fold is hidden
  Logging and concurrency: see ivf_services/_base.py.

Extending: a new retrieval strategy is a class in clinical_tools/similar.py
registered in STRATEGIES (it then appears in ``strategy`` below); more
indicators for the listed patients are requested through ``extra_fields``.
"""

from __future__ import annotations

import asyncio
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, field_validator

from clinical_tools.cleaning import load_training
from clinical_tools.common import FEATURE_NAMES
from clinical_tools.similar import (AGE_WINDOW, DEFAULT_STRATEGY, LISTED_ALWAYS, MIN_COHORT, STRATEGIES, Query,
                                    SimilarCohort)
from ivf_services._base import ServiceKit, run

SERVICE, TOOL, ENDPOINT = "ivf_similar", "ivf_similar_patients", "/ivf/similar_patients"
PORT_ENV, DEFAULT_PORT = "IVF_SIMILAR_PORT", 9201
EXTRA_FIELDS = [f for f in FEATURE_NAMES if f not in LISTED_ALWAYS]
TOOL_DESCRIPTION = (
    "按女方年龄和 AMH 在去泄露的 IVF-ET 训练病例中检索相似人群（默认策略：年龄 ±age_window 岁、AMH 同一档："
    "<1.2 / 1.2–3.5 / >3.5 ng/mL），返回相似患者的年龄、AMH 和临床妊娠结局分布（妊娠率及 95% 置信区间）。"
    f"符合条件少于 {MIN_COHORT} 例时拒绝返回统计结果，只告知例数。"
)


class SimilarPatientsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: float = Field(..., ge=18, le=55, description="当前患者女方年龄（岁）")
    amh: float = Field(..., ge=0, le=50, description="当前患者女方 AMH（ng/mL）")
    age_window: int = Field(AGE_WINDOW[0], ge=AGE_WINDOW[0], le=AGE_WINDOW[1],
                            description="年龄上下浮动的岁数，默认 1")
    strategy: str = Field(DEFAULT_STRATEGY, json_schema_extra={"enum": list(STRATEGIES)},
                          description="检索策略。" + "；".join(f"{s.name}：{s.description}" for s in STRATEGIES.values()))
    extra_fields: list[str] = Field(default_factory=list, max_length=10,
                                    description="相似患者列表中除年龄、AMH、妊娠结局外额外返回的指标，"
                                                "用训练数据中的中文指标名（如 女方基础FSH），默认不返回")

    @field_validator("age", "amh", "age_window", mode="before")
    @classmethod
    def not_boolean(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("必须是数值，不能是 true/false")
        return value

    @field_validator("strategy")
    @classmethod
    def known_strategy(cls, value: str) -> str:
        if value not in STRATEGIES:
            raise ValueError(f"未知检索策略 {value!r}，可选：{', '.join(STRATEGIES)}")
        return value

    @field_validator("extra_fields")
    @classmethod
    def known_fields(cls, value: list[str]) -> list[str]:
        unknown = [f for f in value if f not in EXTRA_FIELDS]
        if unknown:
            raise ValueError(f"不支持的指标：{'、'.join(unknown)}（年龄、AMH、妊娠结局已默认返回；可选指标见工具说明）")
        return list(dict.fromkeys(value))


REQUEST_MODEL = SimilarPatientsRequest

@asynccontextmanager
async def lifespan(app: FastAPI):
    csv_path = os.environ.get("IVF_TRAIN_CSV")
    try:
        if not csv_path:
            raise RuntimeError("IVF_TRAIN_CSV is required (the cleaned training CSV)")
        training = await asyncio.to_thread(load_training, csv_path)
        app.state.cohort = SimilarCohort(training, exclude_case=os.environ.get("IVF_EXCLUDE_CASE") or None)
    except Exception:
        kit.log.exception("failed to start: could not load training data from %s", csv_path)
        raise
    kit.health_info.update(scope=app.state.cohort.scope, cases=int(len(app.state.cohort.labels)),
                           strategies=list(STRATEGIES))
    kit.log.info("started: %s cases (%s) from %s", len(app.state.cohort.labels), app.state.cohort.scope, csv_path)
    yield
    kit.shutdown()


app = FastAPI(title="IVF Similar Patients API", lifespan=lifespan)
kit = ServiceKit(app, SERVICE, TOOL)


@app.post(ENDPOINT)
async def similar_patients(req: SimilarPatientsRequest) -> dict[str, Any]:
    resp = await kit.compute(app.state.cohort.query, Query(req.age, req.amh, req.age_window),
                             strategy=req.strategy, extra_fields=req.extra_fields)
    kit.record(ENDPOINT, req.model_dump(), resp,
               summary=f"matched={resp['符合条件例数']} refused={resp['拒绝统计']}")
    return resp


if __name__ == "__main__":
    run(app, PORT_ENV, DEFAULT_PORT)
