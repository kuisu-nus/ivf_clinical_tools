"""Small shared definitions so the package runs on its own (numpy only).

Kept identical to the IVF agent repository's features.definitions.FEATURE_NAMES,
data.leakage_scan.parse_value/parse_label and the agentic cohort's assign_folds
and wilson_interval; a test in that repository checks the parity.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

FEATURE_NAMES = (
    "女方年龄", "女方基础FSH", "女方基础PRL", "女方基础T", "女方基础P", "女方AMH",
    "女方白细胞计数", "女方血小板", "女方血细胞压积", "女方凝血常规PT", "女方凝血常规APPT",
    "女方凝血常规TT", "女方D二聚体", "女方甲功TPO", "女方空腹血糖", "女方肾功尿酸",
    "男方浓度", "男方正常形态率", "男方PR", "男方NP", "男方存活率", "男方DFI", "男方ALT",
    "男方AST", "男方血肌酐", "男方白细胞", "男方血小板", "男方空腹血糖", "女方用药Gn总天数",
    "女方用药HCG日P", "男方精液处理后NP", "男方精液处理方法", "男方精液处理后密度",
    "男方精液处理后PR", "男方精液处理后IM",
)
MISSING_TOKENS = ("", "na", "nan", "none", "null")


def parse_value(raw) -> tuple[float | None, int | None]:
    """(numeric value, written decimal places); non-numeric text counts as missing."""
    if raw is None or isinstance(raw, bool):
        return None, None
    if isinstance(raw, (int, float)):
        if not math.isfinite(raw):
            return None, None
        text = repr(raw)
    else:
        text = str(raw).strip()
        if text.lower() in MISSING_TOKENS:
            return None, None
    try:
        value = float(text)
    except ValueError:
        return None, None
    if not math.isfinite(value):
        return None, None
    mantissa = text.lower().split("e")[0]
    return value, len(mantissa.split(".")[1]) if "." in mantissa else 0


def parse_label(raw) -> int:
    value, _ = parse_value(raw)
    if value not in (0.0, 1.0):
        raise ValueError(f"Label must be 0 or 1, got {raw!r}")
    return int(value)


def wilson_interval(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def assign_folds(case_ids: Sequence[str], labels: Sequence[int], k: int = 5, seed: str = "agentic-v1") -> list[int]:
    """Deterministic stratified folds: hash order within each class, then round robin."""
    folds = [0] * len(case_ids)
    for cls in (0, 1):
        members = [i for i, y in enumerate(labels) if y == cls]
        members.sort(key=lambda i: hashlib.sha256(f"{seed}:{case_ids[i]}".encode()).hexdigest())
        for rank, i in enumerate(members):
            folds[i] = rank % k
    return folds
