"""Cleaning driven by ovu/config/data.yaml: unified field names, types, valid ranges.

``clean(raw, config)`` returns the cleaned table and a log of what was changed.
Nothing here looks at the label to decide a feature value (outcome-blind): a value
outside its range becomes missing, and rows are dropped only for a missing or
out-of-range label or a duplicated cycle ID.

Columns of the cleaned table: ``_key`` (a salted hash of the cycle ID, used for
the split), ``_row`` and ``_id_number`` (file order and the numeric ID, used only by
the report's order check; no column starting with ``_`` is ever written), the configured fields, the
diagnosis flags, ``diagnosis_text``, the label and ``<label>_bin``.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

OTHER = "其他"
GROUPS = ("clinical", "medication")      # which table of the population report a field goes to
ENCODINGS = ("utf-8-sig", "gb18030", "utf-16")


def load_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    fields = {**config.get("columns", {}), **config.get("split_columns", {})}
    for name, spec in fields.items():
        if spec.get("group", "clinical") not in GROUPS:
            raise ValueError(f"{name}: group 必须是 {GROUPS} 之一")
    for name, spec in config.get("columns", {}).items():
        if spec.get("type") not in ("int", "float", "category"):
            raise ValueError(f"{name}: type 必须是 int、float 或 category")
    bins = config["label"]["bins"]
    if bins != sorted(set(bins)):
        raise ValueError("label.bins 必须从小到大且不重复")
    return config


def read_raw(path: str | Path) -> pd.DataFrame:
    for encoding in ENCODINGS:
        try:
            raw = pd.read_csv(path, encoding=encoding, dtype=str, keep_default_na=True)
        except (UnicodeDecodeError, UnicodeError):
            continue
        raw.columns = [str(c).strip() for c in raw.columns]
        return raw
    raise ValueError(f"无法识别文件编码（试过 {', '.join(ENCODINGS)}）")


def bin_labels(bins: list[int]) -> list[str]:
    """[3, 9] -> ['0–3', '4–9', '≥10']."""
    lows = [0] + [b + 1 for b in bins]
    return [f"{lo}–{hi}" if lo != hi else str(lo) for lo, hi in zip(lows, bins)] + [f"≥{bins[-1] + 1}"]


def bin_index(values: pd.Series, bins: list[int]) -> pd.Series:
    return pd.Series(np.searchsorted(np.asarray(bins), values.to_numpy(), side="left"), index=values.index)


def available_fields(config: dict[str, Any]) -> list[str]:
    """Every field the data config loads (diagnosis flags included). Tools may use only these;
    which of them a tool actually uses is decided by that tool's own config."""
    fields = list(config.get("columns", {})) + list(config.get("split_columns", {}))
    if config.get("diagnosis"):
        fields += diagnosis_flags(config)
    return fields


def diagnosis_flags(config: dict[str, Any]) -> list[str]:
    """The diagnosis categories: the detailed ones (the catch-all "other" last), then the groups."""
    diagnosis = config.get("diagnosis") or {}
    return (list(diagnosis.get("flags", {})) + ([diagnosis["other"]] if diagnosis.get("other") else [])
            + list(diagnosis.get("groups") or {}))


def diagnosis_groups(config: dict[str, Any]) -> dict[str, list[str]]:
    return dict((config.get("diagnosis") or {}).get("groups") or {})


def grouped_details(config: dict[str, Any]) -> set[str]:
    """The detailed categories that belong to some group."""
    return {m for members in diagnosis_groups(config).values() for m in members}


def _numeric(text: pd.Series, spec: dict[str, Any], log: dict[str, int]) -> pd.Series:
    text = text.str.strip().replace("", np.nan)
    values = pd.to_numeric(text, errors="coerce")
    log["non_numeric"] = int((values.isna() & text.notna()).sum())
    if spec.get("zero_is_missing"):
        zero = values == 0
        log["zero_as_missing"] = int(zero.sum())
        values = values.mask(zero)
    if "range" in spec:
        low, high = spec["range"]
        outside = (values < low) | (values > high)
        log["out_of_range"] = int(outside.sum())
        values = values.mask(outside)
    return values


def _category(text: pd.Series, spec: dict[str, Any], log: dict[str, int]) -> pd.Series:
    values = text.str.strip().replace("", np.nan).replace(spec.get("replace", {}))
    counts = values.value_counts()
    rare = counts[counts < spec.get("min_count", 1)].index
    log["merged_into_other"] = int(values.isin(rare).sum())
    return values.where(~values.isin(rare) | values.isna(), OTHER)


def _source(raw: pd.DataFrame, name: str, spec: dict[str, Any]) -> pd.Series:
    if spec["source"] not in raw.columns:
        raise KeyError(f"{name}: 原始数据里没有列“{spec['source']}”")
    return raw[spec["source"]]


def key_of(cycle_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{cycle_id}".encode()).hexdigest()


def clean(raw: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    log: dict[str, Any] = {"rows_raw": len(raw), "fields": {}}
    id_column = config["id_column"]
    ids = raw[id_column].str.strip()
    duplicated = ids.duplicated(keep=False if config.get("duplicates", "drop_all") == "drop_all" else "first")
    log["rows_duplicate_id"] = int(duplicated.sum())

    out = pd.DataFrame(index=raw.index)
    out["_key"] = [key_of(i, config["split"]["seed"]) for i in ids.fillna("")]
    for name, spec in config.get("columns", {}).items():
        field_log = log["fields"].setdefault(name, {})
        source = _source(raw, name, spec)
        if spec["type"] == "category":
            out[name] = _category(source, spec, field_log)
        else:
            out[name] = _numeric(source, spec, field_log)
    for name, spec in config.get("split_columns", {}).items():
        parts = _source(raw, name, spec).str.split(spec["separator"], expand=True)
        part = parts[spec["index"]] if spec["index"] in parts.columns else pd.Series(np.nan, index=raw.index, dtype=object)
        out[name] = _numeric(part.astype("string").astype(object), spec, log["fields"].setdefault(name, {}))

    diagnosis = config.get("diagnosis")
    if diagnosis:
        text = _source(raw, "diagnosis", diagnosis).fillna("").str.strip()
        upper = text.str.upper()
        for flag, keywords in diagnosis["flags"].items():
            pattern = "|".join(re.escape(k.upper()) for k in keywords)
            out[flag] = upper.str.contains(pattern, regex=True).astype(int)
        out["diagnosis_text"] = text
        flags = list(diagnosis["flags"])
        unmatched = (out[flags].sum(axis=1) == 0) & (text != "")
        if diagnosis.get("other"):
            out[diagnosis["other"]] = unmatched.astype(int)
        for group, members in (diagnosis.get("groups") or {}).items():
            missing = [m for m in members if m not in out.columns]
            if missing:
                raise ValueError(f"diagnosis.groups.{group} 里的细类不存在：{missing}")
            out[group] = out[members].max(axis=1).astype(int)
        log["diagnosis_unmatched"] = int(unmatched.sum())
        log["diagnosis_empty"] = int((text == "").sum())

    label = config["label"]
    label_log: dict[str, int] = {}
    values = _numeric(_source(raw, label["name"], label), label, label_log)
    log["label"] = label_log
    log["rows_label_missing"] = int(values.isna().sum())
    out[label["name"]] = values

    keep = ~duplicated & values.notna() & ids.notna()
    out["_row"] = range(len(out))                                   # file order; only for the report's order check
    out["_id_number"] = pd.to_numeric(ids, errors="coerce")
    out = out[keep].copy()
    out[label["name"]] = out[label["name"]].astype(int)
    out[label["name"] + "_bin"] = bin_index(out[label["name"]], label["bins"])
    for name, spec in config.get("columns", {}).items():
        if spec["type"] == "int":
            out[name] = out[name].astype("Int64")
    log["rows_clean"] = len(out)
    return out.reset_index(drop=True), log

