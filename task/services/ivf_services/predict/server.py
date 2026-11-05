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
    "预测时接收女方年龄、AMH、Gn 用药总天数，以及模型中系数较大的女方凝血 PT、TT、白细胞计数、"
    "男方精子存活率、精液处理后 IM（至少提供一项；病例未提供的指标不要传，也不要填 0），其余指标及未提供的指标按训练集均值代替。"
    "同时返回每个输入对概率的影响（只把该项从训练集均值换成当前取值时，概率升高或降低多少）。"
    "模型区分能力有限（交叉验证 AUC 约 0.6），置信区间只反映模型系数的不确定性，不是个体结局的范围。"
)


ZERO_IS_MISSING = "0 不是有效值：病例未提供该指标时不要传这个参数（缺失按训练集均值处理），不要用 0 代替"


class PredictOutcomeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: float | None = Field(None, ge=18, le=55, description="女方年龄（岁）")
    amh: float | None = Field(None, ge=0, le=50, description="女方 AMH（ng/mL）")
    gn_days: float | None = Field(None, ge=1, le=30, description="女方 Gn 用药总天数（天）")
    pt: float | None = Field(None, ge=5, le=60, description="女方凝血常规 PT（秒）")
    tt: float | None = Field(None, ge=5, le=60, description="女方凝血常规 TT（秒）")
    wbc: float | None = Field(None, ge=0.5, le=50, description="女方白细胞计数（×10^9/L）")
    sperm_survival: float | None = Field(None, ge=0, le=100, description="男方精子存活率（%）")
    post_wash_im: float | None = Field(None, ge=0, le=100, description="男方精液处理后 IM（按病例中给出的数值填写）")

    @field_validator("age", "amh", "gn_days", "pt", "tt", "wbc", "sperm_survival", "post_wash_im", mode="before")
    @classmethod
    def not_boolean(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("必须是数值，不能是 true/false")
        # A missing indicator is left out, never 0: in the cleaned data 0 marks a missing value
        # (e.g. an all-zero semen analysis), and the model treats a missing input as the training mean.
        if isinstance(value, (int, float)) and value == 0:
            raise ValueError(ZERO_IS_MISSING)
        return value

    @model_validator(mode="after")
    def at_least_one(self) -> PredictOutcomeRequest:
        if all(value is None for value in self.model_dump().values()):
            raise ValueError("至少提供一个输入（age、amh、gn_days、pt、tt、wbc、sperm_survival、post_wash_im）")
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
