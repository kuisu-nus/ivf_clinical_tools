"""Tool ivf_predict_outcome: pregnancy probability, 95% interval and each input's effect.
Served by ivf_services/server.py at POST /tools/ivf_predict_outcome.

Only loads a model artifact written by clinical_tools.train_predictor; never reads training data.
The model is trained on all 35 indicators; eight of them are accepted as inputs and every other
indicator, and any input left out, takes its training mean.

Environment:
  IVF_MODEL          model artifact (default: services/models/$IVF_MODEL_NAME, lr_35_indicators.json)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from clinical_tools.predict import Predictor

TOOL = "ivf_predict_outcome"
MODELS = Path(__file__).resolve().parent.parent.parent / "models"
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

STATE = "predictor"


def load() -> Predictor:
    return Predictor.from_file(os.environ.get("IVF_MODEL")
                               or MODELS / (os.environ.get("IVF_MODEL_NAME") or "lr_35_indicators.json"))


def health(predictor: Predictor) -> dict[str, Any]:
    name = Path(os.environ.get("IVF_MODEL") or (os.environ.get("IVF_MODEL_NAME") or "lr_35_indicators.json")).name
    return {"model": name, "model_training_n": predictor.training["n"],
            "model_cv_auc_three_inputs": predictor.training["cv_auc_three_inputs"]}


def call(predictor: Predictor, req: PredictOutcomeRequest) -> dict[str, Any]:
    return predictor.predict(**req.model_dump())


def summary(resp: dict[str, Any]) -> str:
    return f"p={resp['妊娠概率']}"
