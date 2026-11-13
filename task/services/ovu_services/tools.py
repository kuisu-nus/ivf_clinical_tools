"""The oocyte-yield tools: request schemas, loading and calls.

  ovu_predict_oocytes   probability of each oocyte-count group, expected count, each input's effect
  ovu_similar_cases     the most similar training cycles and how their oocyte counts are distributed
  ovu_case_sql          one read-only SELECT on the training-case database
  ovu_data_overview     what the database holds (columns, meaning, stage, missing rate, distributions)
  ovu_sql_examples      ready-to-run example queries

Nothing about the fields is written here: the request schema of the first two tools is built from
the model file and the retrieval spec (their ``inputs``), which come from the yaml configs. Change a
config, rebuild, restart: the tools follow.

Environment:
  OVU_DATA_DIR    directory written by the offline steps (default: services/ovu_data), holding
                  train.csv, model/model.json (+ model_fold<k>.json), similar/similar.json, db/db.json
  OVU_HIDE_FOLD   fold of the training case an agent is being trained on: the predictor uses the
                  model trained without it, retrieval and the database leave its cases out
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, create_model

from ovu.db.database import CaseDatabase
from ovu.ml.predict import Predictor
from ovu.similar.index import SimilarIndex

DATA_DIR = Path(os.environ.get("OVU_DATA_DIR") or Path(__file__).resolve().parent.parent / "ovu_data")
CLIENT_ERROR = "invalid_arguments"      # a ValueError from call() is answered 400 with this error code


def hide_fold() -> int | None:
    value = os.environ.get("OVU_HIDE_FOLD", "")
    return int(value) if value != "" else None


def _json(*parts: str) -> dict[str, Any]:
    path = DATA_DIR.joinpath(*parts)
    if not path.exists():
        raise FileNotFoundError(f"{path} 不存在：先运行离线步骤（ovu.pipeline、ovu.ml.train、ovu.similar.build、ovu.db.build），"
                                f"或把 OVU_DATA_DIR 指向它们的输出目录")
    return json.loads(path.read_text(encoding="utf-8"))


def _rows() -> list[dict[str, str]]:
    with (DATA_DIR / "train.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def request_model(name: str, inputs: list[dict[str, Any]], **extra: Any) -> type[BaseModel]:
    """A Pydantic model with one optional field per input (unknown fields are refused)."""
    fields: dict[str, Any] = {}
    for item in inputs:
        if item["kind"] in ("number", "integer"):
            unit = "，整数" if item["kind"] == "integer" else ""
            fields[item["name"]] = (Optional[float], Field(None, ge=item.get("minimum"), le=item.get("maximum"),
                                                           description=f"{item['description']}{unit}"))
        elif item["kind"] == "binary":
            fields[item["name"]] = (Optional[Literal[0, 1]], Field(None, description=item["description"]))
        else:
            levels = "、".join(item.get("levels", []))
            fields[item["name"]] = (Optional[str], Field(None, max_length=100, description=f"{item['description']}；取值：{levels}"))
    return create_model(name, __config__=ConfigDict(extra="forbid"), **fields, **extra)


class NoArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SqlRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sql: str = Field(..., min_length=1, max_length=4000, description="一条只读 SELECT 语句（SQLite 语法）")


# --- prediction ---------------------------------------------------------------------------------
def _model_parts() -> tuple[str, ...]:
    fold = hide_fold()
    return ("model", "model.json" if fold is None else f"model_fold{fold}.json")


def _predict_tool() -> SimpleNamespace:
    artifact = _json(*_model_parts())
    groups = "、".join(artifact["groups"])
    return SimpleNamespace(
        TOOL="ovu_predict_oocytes", STATE="predictor", CLIENT_ERROR=CLIENT_ERROR,
        TOOL_DESCRIPTION=(
            f"预测当前患者本次促排的获卵数落在各分组（{groups}）的概率，并给出最可能的分组和预期获卵数。"
            "模型是在历史促排周期上训练的多分类逻辑回归，只使用进周期前和启动方案时已知的信息。"
            "至少提供一项输入；病例里没有的指标不要传，也不要用 0 代替（未提供的数值按训练集中位数处理）。"
            "同时返回每项输入让预期获卵数升高或降低多少。结果是统计关联，不是对个体的保证。"),
        REQUEST_MODEL=request_model("PredictOocytesRequest", artifact["inputs"]),
        load=lambda: Predictor(_json(*_model_parts())),
        health=lambda p: {"model": _model_parts()[-1], "training_n": p.info["n"], "inputs": len(p.inputs)},
        call=lambda p, req: p.predict(**req.model_dump()),
        summary=lambda resp: f"group={resp['最可能的分组']} expected={resp['预期获卵数']}")


# --- similar cases ------------------------------------------------------------------------------
def _similar_tool() -> SimpleNamespace:
    spec = _json("similar", "similar.json")
    names = "、".join(f"{i['name']}（权重 {spec['features'][i['name']]['weight']:g}）" for i in spec["inputs"])

    def call(index: SimilarIndex, req: BaseModel) -> dict[str, Any]:
        values = req.model_dump()
        k = values.pop("k")
        return index.query(values, k=k, hide_fold=hide_fold())

    return SimpleNamespace(
        TOOL="ovu_similar_cases", STATE="similar", CLIENT_ERROR=CLIENT_ERROR,
        TOOL_DESCRIPTION=(
            "在历史促排周期中检索与当前患者最相似的 k 例，返回这些病例的获卵数分组占比、中位数和四分位数，"
            "以及最相似的几位患者（不含任何编号）。相似度是各比对特征差异的加权平均，只计算你提供的特征："
            f"{names}。至少提供一项；病例里没有的指标不要传，也不要用 0 代替。"),
        REQUEST_MODEL=request_model("SimilarCasesRequest", spec["inputs"], k=(
            int, Field(spec["k_default"], ge=spec["min_cohort"], le=spec["k_max"], description="取最相似的多少例"))),
        load=lambda: SimilarIndex(_json("similar", "similar.json"), _rows()),
        health=lambda index: {"cases": len(index.rows), "hidden_fold": hide_fold(), "features": len(index.features)},
        call=call,
        summary=lambda resp: f"k={resp['检索条件']['取最相似的例数']}")


# --- case database ------------------------------------------------------------------------------
def _load_db() -> CaseDatabase:
    return CaseDatabase(_json("db", "db.json"), _rows(), hide_fold=hide_fold())


def _db_tools() -> list[SimpleNamespace]:
    spec = _json("db", "db.json")
    table = spec["table"]
    common = dict(STATE="case_db", CLIENT_ERROR="invalid_sql", load=_load_db,
                  health=lambda db: {"cases": db.n, "scope": db.scope, "columns": len(db.spec["columns"])})
    return [
        SimpleNamespace(
            TOOL="ovu_case_sql", REQUEST_MODEL=SqlRequest,
            TOOL_DESCRIPTION=(
                f"在历史促排周期的病例库上执行一条只读 SQL（SQLite 语法，只允许 SELECT）。表 {table}：每行一个周期，"
                f"没有任何编号，缺失为 NULL；结局列是 {spec['label']}（获卵数）和 {spec['label']}_group（分组）。"
                f"最多返回 {spec['max_rows']} 行，超过 {spec['timeout_seconds']:g} 秒中止，尽量用 COUNT、AVG、GROUP BY 等聚合。"
                "不知道有哪些列时先调用 ovu_data_overview；需要写法参考时调用 ovu_sql_examples。"),
            call=lambda db, req: db.sql(req.sql), summary=lambda resp: f"rows={resp['返回行数']}", **common),
        SimpleNamespace(
            TOOL="ovu_data_overview", REQUEST_MODEL=NoArguments,
            TOOL_DESCRIPTION="病例库的数据概览：病例数、获卵数分组占比，以及每一列的列名、含义、类型、预测时点、缺失率和分布。不需要参数。",
            call=lambda db, req: db.overview(), summary=lambda resp: f"columns={len(resp['列'])}", **common),
        SimpleNamespace(
            TOOL="ovu_sql_examples", REQUEST_MODEL=NoArguments,
            TOOL_DESCRIPTION="病例库的常用查询示例（标题和可直接执行的 SQL），把其中的数值换成当前患者的即可。不需要参数。",
            call=lambda db, req: db.examples(), summary=lambda resp: f"examples={len(resp['示例'])}", **common),
    ]


def build_tools() -> tuple[SimpleNamespace, ...]:
    return (_predict_tool(), _similar_tool(), *_db_tools())
