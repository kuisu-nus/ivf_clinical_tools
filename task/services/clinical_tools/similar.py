"""Similar-patient retrieval over the cleaned training cases.

Retrieval has two parts, so a new behaviour is a small addition:

* a strategy decides which visible cases are similar and how close each one is
  (subclass ``SimilarityStrategy`` and register it in ``STRATEGIES``);
* ``SimilarCohort.query`` applies the shared rules to any strategy's selection:
  the minimum cohort size, outcome counts with a Wilson interval, and the list
  of similar patients with the indicators the caller asks for.

Visible cases: the training set, minus the fold of the case under evaluation
when one is set (agentic RL on training cases), so a case's own label never
reaches its tool results. Case IDs are never returned and never used for
ordering: in this dataset the ID order alone separates the outcomes.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from clinical_tools.cleaning import TrainingSet
from clinical_tools.common import FEATURE_NAMES, wilson_interval

MIN_COHORT = 15
MAX_LISTED = 50
AGE_WINDOW = (1, 3)              # allowed half-widths in years; default the lower bound
AMH_BANDS = ((0.0, 1.2, "低（<1.2 ng/mL）"), (1.2, 3.5, "中（1.2–3.5 ng/mL）"),
             (3.5, math.inf, "高（>3.5 ng/mL）"))
AMH_DETECTION_FLOOR = 0.02       # most common lowest value; likely the assay's lower limit
OUTCOME = {1: "临床妊娠", 0: "未妊娠"}
LISTED_ALWAYS = ("女方年龄", "女方AMH")   # every listed patient shows these, then any extra fields


def amh_band(value: float) -> tuple[float, float, str]:
    """Bands are [low, high): 1.2 falls in the middle band, 3.5 in the high band."""
    return next(band for band in AMH_BANDS if band[0] <= value < band[1])


@dataclass(frozen=True)
class Query:
    """The current patient and retrieval settings (validated by the service)."""
    age: float
    amh: float
    age_window: int = AGE_WINDOW[0]


@dataclass(frozen=True)
class Selection:
    """What a strategy returns: which cases match and how close each case is."""
    criteria: str                # human-readable filter, shown to the agent
    searchable: np.ndarray       # cases the strategy can judge (needed values present)
    matched: np.ndarray          # cases in the cohort
    distance: np.ndarray         # smaller is more similar; only orders the listed patients
    widen_hint: str              # what to try when the cohort is below MIN_COHORT


class SimilarityStrategy:
    """Base class: a strategy has a name, a one-line description and ``select``."""
    name = ""
    description = ""

    def select(self, cohort: SimilarCohort, query: Query) -> Selection:
        raise NotImplementedError

    def profile(self, cohort: SimilarCohort, index: np.ndarray) -> dict:
        """Summary of the matched cases' clinical values; strategies may override."""
        return {}


class AgeAmhBand(SimilarityStrategy):
    """Female age within ±age_window years and AMH in the same band."""
    name = "age_amh_band"
    description = "女方年龄 ±age_window 岁，且 AMH 同一档（<1.2 / 1.2–3.5 / >3.5 ng/mL）"

    def select(self, cohort: SimilarCohort, query: Query) -> Selection:
        age, amh = cohort.column("女方年龄"), cohort.column("女方AMH")
        low, high, band = amh_band(query.amh)
        floor = AMH_DETECTION_FLOOR
        searchable = ~np.isnan(age) & ~np.isnan(amh)
        with np.errstate(invalid="ignore"):
            matched = searchable & (np.abs(age - query.age) <= query.age_window) & (amh >= low) & (amh < high)
            distance = np.abs(age - query.age) + np.abs(np.log(np.maximum(amh, floor)) - math.log(max(query.amh, floor)))
        return Selection(
            criteria=f"女方年龄 {query.age - query.age_window:g}–{query.age + query.age_window:g} 岁；AMH {band}",
            searchable=searchable, matched=matched, distance=distance,
            widen_hint="可以放宽年龄窗口（最大 3 岁）后重新检索；仍不足时说明相似人群证据不足"
                       if query.age_window < AGE_WINDOW[1] else "已是最大年龄窗口，应说明相似人群证据不足")

    def profile(self, cohort: SimilarCohort, index: np.ndarray) -> dict:
        age, amh = cohort.column("女方年龄")[index], cohort.column("女方AMH")[index]
        floor = int(np.isclose(amh, AMH_DETECTION_FLOOR).sum())
        return {
            "女方年龄中位数（范围）": f"{np.median(age):g}（{age.min():g}–{age.max():g}）",
            "AMH中位数（范围）": f"{np.median(amh):.2f}（{amh.min():g}–{amh.max():g}）",
            **({"AMH=0.02 的病例数": floor, "AMH=0.02 说明": "可能是检测下限（<0.02）"} if floor else {}),
        }


# Registered strategies; the first one is the default. Add new strategies here.
STRATEGIES: dict[str, SimilarityStrategy] = {s.name: s for s in (AgeAmhBand(),)}
DEFAULT_STRATEGY = next(iter(STRATEGIES))


def _value(x: float) -> float | None:
    return None if math.isnan(x) else float(x)


class SimilarCohort:
    def __init__(self, training: TrainingSet, *, exclude_case: str | None = None, k_folds: int = 5):
        training, self.scope = training.visible(exclude_case, k_folds)
        self.matrix, self.labels = training.matrix, training.labels
        self._tiebreak = np.array([int(hashlib.sha256(f"tie:{c}".encode()).hexdigest()[:12], 16)
                                   for c in training.case_ids])

    def column(self, name: str) -> np.ndarray:
        return self.matrix[:, FEATURE_NAMES.index(name)]

    def query(self, query: Query, *, strategy: str = DEFAULT_STRATEGY, extra_fields: Sequence[str] = ()) -> dict:
        chosen = STRATEGIES[strategy]
        selection = chosen.select(self, query)
        n = int(selection.matched.sum())
        result = {
            "检索策略": chosen.name,
            "筛选条件": selection.criteria,
            "检索范围": self.scope,
            "可检索病例数": int(selection.searchable.sum()),
            "符合条件例数": n,
            "最低例数门槛": MIN_COHORT,
        }
        if n < MIN_COHORT:
            return {**result, "拒绝统计": True,
                    "原因": f"符合条件的病例只有 {n} 例，少于 {MIN_COHORT} 例，不报告妊娠结局分布和个体结局",
                    "建议": selection.widen_hint}
        index = np.flatnonzero(selection.matched)
        positives = int(self.labels[index].sum())
        low, high = wilson_interval(positives, n)
        listed = index[np.lexsort((self._tiebreak[index], selection.distance[index]))][:MAX_LISTED]
        fields = list(dict.fromkeys((*LISTED_ALWAYS, *extra_fields)))
        columns = [FEATURE_NAMES.index(f) for f in fields]
        return {
            **result, "拒绝统计": False,
            "妊娠结局分布": {"临床妊娠": positives, "未妊娠": n - positives, "妊娠率": round(positives / n, 3),
                       "95%置信区间": [round(low, 3), round(high, 3)],
                       "训练集总体妊娠率": round(float(self.labels.mean()), 3)},
            "相似人群临床信息": chosen.profile(self, index),
            "相似患者": [{**{f: _value(self.matrix[i, c]) for f, c in zip(fields, columns)},
                      "妊娠结局": OUTCOME[int(self.labels[i])]} for i in listed],
            "说明": f"相似患者按与当前患者的接近程度排列，最多列出 {MAX_LISTED} 例，不含病例编号；"
                  "指标为 null 表示该病例未记录或已在清洗中置空。"
                  "妊娠率是相似人群的观察结果，是相关性，不是该患者的预测概率。",
        }
