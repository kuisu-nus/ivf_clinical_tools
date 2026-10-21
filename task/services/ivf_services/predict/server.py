"""IVF clinical pregnancy prediction API (FastAPI on port 9202).

  POST /ivf/predict_outcome    pregnancy probability and 95% interval (tool ivf_predict_outcome)
  GET  /health                 status, model metrics, request counts and load
  GET  /ivf/audit              tool calls since start;  POST /ivf/reset clears them

The service only loads a model artifact written by clinical_tools.train_predictor;
it never reads training data. The model is trained on all 35 indicators; here
only age, AMH and Gn days are accepted and every other indicator, and any of
the three left out, takes its training mean.

Environment:
  IVF_MODEL          model artifact (default: services/models/lr_35_indicators.json)
  IVF_PREDICT_PORT   9202
  Logging and concurrency: see ivf_services/_base.py.
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
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from clinical_tools.predict import Predictor
from ivf_services._base import ServiceKit, run

SERVICE, TOOL, ENDPOINT = "ivf_predict", "ivf_predict_outcome", "/ivf/predict_outcome"
PORT_ENV, DEFAULT_PORT = "IVF_PREDICT_PORT", 9202
DEFAULT_MODEL = Path(__file__).resolve().parent.parent.parent / "models" / "lr_35_indicators.json"
TOOL_DESCRIPTION = (
    "预测当前患者本次移植的临床妊娠概率及 95% 置信区间。模型为在 35 项指标上训练的逻辑回归，"
    "预测时只接收女方年龄、AMH、Gn 用药总天数（至少提供一项），其余指标及未提供的指标按训练集均值代替。"
    "模型区分能力有限（交叉验证 AUC 约 0.6），置信区间只反映模型系数的不确定性，不是个体结局的范围。"
)


class PredictOutcomeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: float | None = Field(None, ge=18, le=55, description="女方年龄（岁）")
    amh: float | None = Field(None, ge=0, le=50, description="女方 AMH（ng/mL）")
    gn_days: float | None = Field(None, ge=1, le=30, description="女方 Gn 用药总天数（天）")

    @field_validator("age", "amh", "gn_days", mode="before")
    @classmethod
    def not_boolean(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("必须是数值，不能是 true/false")
        return value

    @model_validator(mode="after")
    def at_least_one(self) -> PredictOutcomeRequest:
        if self.age is None and self.amh is None and self.gn_days is None:
            raise ValueError("至少提供 age、amh、gn_days 中的一个")
        return self


REQUEST_MODEL = PredictOutcomeRequest

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = os.environ.get("IVF_MODEL") or str(DEFAULT_MODEL)
    try:
        app.state.predictor = await asyncio.to_thread(Predictor.from_file, model_path)
    except Exception:
        kit.log.exception("failed to start: could not load model artifact %s", model_path)
        raise
    training = app.state.predictor.training
    kit.health_info.update(model=Path(model_path).name, model_training_n=training["n"],
                           model_cv_auc_three_inputs=training["cv_auc_three_inputs"])
    kit.log.info("started: model %s (trained on %s cases, CV AUC with three inputs %s)", model_path,
                 training["n"], training["cv_auc_three_inputs"])
    yield
    kit.shutdown()


app = FastAPI(title="IVF Pregnancy Prediction API", lifespan=lifespan)
kit = ServiceKit(app, SERVICE, TOOL)


@app.post(ENDPOINT)
async def predict_outcome(req: PredictOutcomeRequest) -> dict[str, Any]:
    resp = await kit.compute(app.state.predictor.predict, **req.model_dump())
    kit.record(ENDPOINT, req.model_dump(), resp, summary=f"p={resp['妊娠概率']}")
    return resp


if __name__ == "__main__":
    run(app, PORT_ENV, DEFAULT_PORT)
