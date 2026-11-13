"""What a tool needs to know about each field it accepts: kind, meaning, valid range, levels.

Built from the data config when a model or a retrieval spec is written, and stored inside that
file, so the tool service can build its request schema without reading any config.
"""

from __future__ import annotations

from typing import Any


def describe(names: list[str], data_config: dict[str, Any], levels: dict[str, list[str]] | None = None) -> list[dict[str, Any]]:
    """One entry per field: name, kind (number | integer | category | binary), description,
    and minimum / maximum (numbers) or levels (categories)."""
    specs = {**data_config.get("columns", {}), **data_config.get("split_columns", {})}
    out = []
    for name in names:
        flag = flag_description(name, data_config)
        if flag:
            out.append({"name": name, "kind": "binary", "description": flag})
            continue
        spec = specs[name]
        description = spec.get("description") or spec["source"]
        if spec.get("type") == "category":
            out.append({"name": name, "kind": "category", "description": description, "levels": (levels or {}).get(name, [])})
            continue
        entry = {"name": name, "kind": "integer" if spec.get("type") == "int" else "number", "description": description}
        if "range" in spec:
            entry["minimum"], entry["maximum"] = spec["range"]
        out.append(entry)
    return out


def flag_description(name: str, data_config: dict[str, Any]) -> str | None:
    """Description of a diagnosis category, or None when ``name`` is not one."""
    diagnosis = data_config.get("diagnosis") or {}
    title = diagnosis.get("names", {}).get(name)
    if name in diagnosis.get("flags", {}):
        keywords = "、".join(diagnosis["flags"][name][:3])
        return f"{title + '：' if title else ''}诊断中含“{keywords}”等关键词（1 是，0 否）"
    if name == diagnosis.get("other"):
        return f"{title or '其他'}：有诊断但不属于任何已定义的类别（1 是，0 否）"
    return None
