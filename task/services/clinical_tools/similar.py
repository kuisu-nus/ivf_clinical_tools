"""Similar-patient cohort by female age and AMH.

A cohort is every visible training case within the age window and in the same
AMH band as the current patient. Below MIN_COHORT cases the tool refuses to
report outcome statistics or individual outcomes, and says how many matched.

Visible cases: the training set, minus the fold of the case under evaluation
when one is set (agentic RL on training cases), so a case's own label never
reaches its tool results. Case IDs are never returned: in this dataset the ID
order alone separates the outcomes, so IDs are not used for ordering either.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np

from clinical_tools.cleaning import TrainingSet
from clinical_tools.common import FEATURE_NAMES, InputError, number_argument, wilson_interval

MIN_COHORT = 15
AGE_WINDOW = (1, 3)              # allowed half-widths in years; default the lower bound
AMH_BANDS = ((0.0, 1.2, "低（<1.2 ng/mL）"), (1.2, 3.5, "中（1.2–3.5 ng/mL）"),
             (3.5, math.inf, "高（>3.5 ng/mL）"))
AMH_DETECTION_FLOOR = 0.02       # most common lowest value; likely the assay's lower limit
MAX_LISTED = 50
OUTCOME = {1: "临床妊娠", 0: "未妊娠"}
AGE, AMH = FEATURE_NAMES.index("女方年龄"), FEATURE_NAMES.index("女方AMH")


def amh_band(value: float) -> tuple[float, float, str]:
    """Bands are [low, high): 1.2 falls in the middle band, 3.5 in the high band."""
    return next(band for band in AMH_BANDS if band[0] <= value < band[1])


class SimilarCohort:
    def __init__(self, training: TrainingSet, *, exclude_case: str | None = None, k_folds: int = 5):
        training, self.scope = training.visible(exclude_case, k_folds)
        ids = training.case_ids
        self.age, self.amh = training.matrix[:, AGE], training.matrix[:, AMH]
        self.labels = training.labels
        self.searchable = ~np.isnan(self.age) & ~np.isnan(self.amh)
        self._tiebreak = np.array([int(hashlib.sha256(f"tie:{c}".encode()).hexdigest()[:12], 16) for c in ids])

    def query(self, arguments: dict) -> dict:
        age = number_argument(arguments, "age", 18, 55)
        amh = number_argument(arguments, "amh", 0, 50)
        window = arguments.get("age_window", AGE_WINDOW[0])
        if not isinstance(window, int) or isinstance(window, bool) or not AGE_WINDOW[0] <= window <= AGE_WINDOW[1]:
            raise InputError(f"age_window 必须是 {AGE_WINDOW[0]}–{AGE_WINDOW[1]} 之间的整数（岁）")
        low_amh, high_amh, band = amh_band(amh)
        with np.errstate(invalid="ignore"):
            match = (self.searchable & (np.abs(self.age - age) <= window)
                     & (self.amh >= low_amh) & (self.amh < high_amh))
        n = int(match.sum())
        result = {
            "筛选条件": f"女方年龄 {age - window:g}–{age + window:g} 岁；AMH {band}",
            "检索范围": self.scope,
            "可检索病例数": int(self.searchable.sum()),
            "符合条件例数": n,
            "最低例数门槛": MIN_COHORT,
        }
        if n < MIN_COHORT:
            return {**result, "拒绝统计": True,
                    "原因": f"符合条件的病例只有 {n} 例，少于 {MIN_COHORT} 例，不报告妊娠结局分布和个体结局",
                    "建议": "可以放宽年龄窗口（最大 3 岁）后重新检索；仍不足时说明相似人群证据不足"
                            if window < AGE_WINDOW[1] else "已是最大年龄窗口，应说明相似人群证据不足"}
        positives = int(self.labels[match].sum())
        low, high = wilson_interval(positives, n)
        index = np.flatnonzero(match)
        closeness = np.abs(self.age[index] - age) + np.abs(np.log(np.maximum(self.amh[index], AMH_DETECTION_FLOOR))
                                                         - math.log(max(amh, AMH_DETECTION_FLOOR)))
        listed = index[np.lexsort((self._tiebreak[index], closeness))][:MAX_LISTED]
        floor = int(np.isclose(self.amh[index], AMH_DETECTION_FLOOR).sum())
        return {
            **result, "拒绝统计": False,
            "妊娠结局分布": {"临床妊娠": positives, "未妊娠": n - positives, "妊娠率": round(positives / n, 3),
                       "95%置信区间": [round(low, 3), round(high, 3)],
                       "训练集总体妊娠率": round(float(self.labels.mean()), 3)},
            "相似人群临床信息": {
                "女方年龄中位数（范围）": f"{np.median(self.age[index]):g}（{self.age[index].min():g}–{self.age[index].max():g}）",
                "AMH中位数（范围）": f"{np.median(self.amh[index]):.2f}（{self.amh[index].min():g}–{self.amh[index].max():g}）",
                **({"AMH=0.02 的病例数": floor, "AMH=0.02 说明": "可能是检测下限（<0.02）"} if floor else {})},
            "相似患者": [{"女方年龄": float(self.age[i]), "女方AMH": float(self.amh[i]),
                      "妊娠结局": OUTCOME[int(self.labels[i])]} for i in listed],
            "说明": f"相似患者按年龄和 AMH 的接近程度排列，最多列出 {MAX_LISTED} 例，不含病例编号。"
                  "妊娠率是相似人群的观察结果，是相关性，不是该患者的预测概率。",
        }


def tool_spec(base_url: str = "http://127.0.0.1:9201") -> dict:
    """Tool spec in the Claw task format."""
    return {
        "name": "ivf_similar_patients",
        "description": "按女方年龄和 AMH 在去泄露的 IVF-ET 训练病例中检索相似人群（年龄 ±age_window 岁、AMH 同一档："
                       "<1.2 / 1.2–3.5 / >3.5 ng/mL），返回相似患者的年龄、AMH 和临床妊娠结局分布（妊娠率及 95% 置信区间）。"
                       f"符合条件少于 {MIN_COHORT} 例时拒绝返回统计结果，只告知例数。",
        "input_schema": {
            "type": "object",
            "properties": {
                "age": {"type": "number", "description": "当前患者女方年龄（岁）"},
                "amh": {"type": "number", "description": "当前患者女方 AMH（ng/mL）"},
                "age_window": {"type": "integer", "minimum": AGE_WINDOW[0], "maximum": AGE_WINDOW[1],
                               "default": AGE_WINDOW[0], "description": "年龄上下浮动的岁数，默认 1"},
            },
            "required": ["age", "amh"],
            "additionalProperties": False,
        },
        "backend": {"type": "http", "service": "ivf_clinical", "url": f"{base_url}/ivf/similar_patients",
                    "method": "POST"},
    }
