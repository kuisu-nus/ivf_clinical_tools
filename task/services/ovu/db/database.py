"""Read-only SQL over the training cases, plus an overview and query examples. Standard library only.

``resolve`` turns ovu/config/db.yaml and the data config into a plain dict
(columns with types and descriptions, limits, examples) that
``python -m ovu.db.build`` writes to db.json. ``CaseDatabase`` loads that dict
and the training rows into an in-memory SQLite database.

What the agent can reach: one table of training cases with the configured
columns. Never in the table: the cycle ID (not in train.csv at all), the
bookkeeping column ``cv_fold``, the free-text diagnosis unless the config asks
for it, and the cases of ``hide_fold`` (the fold of the training case an agent
is being trained on). The sealed test set is never loaded.

The connection only allows SELECT (an authorizer refuses writes, ATTACH and
PRAGMA), stops a query after the time limit and returns at most ``max_rows``.
"""

from __future__ import annotations

import math
import sqlite3
import sys
import statistics
import threading
import time
from typing import Any

from ovu.inputs import flag_description

NEVER = ("cv_fold",)
PRIVATE_BY_DEFAULT = ("diagnosis_text",)
Row = dict[str, Any]


def resolve(config: dict[str, Any], data_config: dict[str, Any], stages: dict[str, str], groups: list[str], bins: list[int]) -> dict[str, Any]:
    label = data_config["label"]["name"]
    specs = {**data_config.get("columns", {}), **data_config.get("split_columns", {})}
    available = list(stages) + [label, f"{label}_group"]
    wanted = available if config.get("columns", "all") == "all" else list(config["columns"])
    unknown = [c for c in wanted if c not in available + list(PRIVATE_BY_DEFAULT)]
    if unknown:                                                     # removed from data.yaml: leave them out
        print(f"提示：db.yaml 中这些列不在 data.yaml 里，已跳过：{unknown}", file=sys.stderr)
        wanted = [c for c in wanted if c not in unknown]
    columns = []
    for name in wanted:
        if name in NEVER or name in config.get("exclude", []):
            continue
        spec = specs.get(name, {})
        if name == f"{label}_group" or spec.get("type") == "category" or name in PRIVATE_BY_DEFAULT:
            kind = "TEXT"
        elif name == label or spec.get("type") == "int" or name not in specs:
            kind = "INTEGER"                                        # the label, int fields, 0/1 diagnosis flags
        else:
            kind = "REAL"
        description = (config.get("descriptions", {}).get(name) or spec.get("name") or spec.get("source")
                       or flag_description(name, data_config) or name)
        columns.append({"name": name, "type": kind, "description": description, "stage": stages.get(name, "outcome")})
    table = config.get("table", "cases")
    limits = config.get("limits", {})
    return {"table": table, "label": label, "bins": bins, "groups": groups, "columns": columns,
            "max_rows": int(limits.get("max_rows", 200)), "timeout_seconds": float(limits.get("timeout_seconds", 2)),
            "min_group_size": int(limits.get("min_group_size", 0)),
            "examples": [{"title": e["title"], "sql": e["sql"].format(table=table)} for e in config.get("examples", [])]}


def _missing(value: Any) -> bool:
    return value is None or value == "" or (isinstance(value, float) and math.isnan(value))


class CaseDatabase:
    def __init__(self, spec: dict[str, Any], rows: list[Row], *, hide_fold: int | None = None):
        self.spec = spec
        self.table = spec["table"]
        label, bins, groups = spec["label"], spec["bins"], spec["groups"]
        visible = [r for r in rows if hide_fold is None or _missing(r.get("cv_fold")) or int(r["cv_fold"]) != hide_fold]
        self.scope = "全部训练病例" if hide_fold is None else "训练病例（不含当前病例所在的一折）"
        self.n = len(visible)
        records = []
        for row in visible:
            count = int(float(row[label]))
            group = groups[sum(count > b for b in bins)]
            record = []
            for column in spec["columns"]:
                value = group if column["name"] == f"{label}_group" else row.get(column["name"])
                if _missing(value):
                    record.append(None)
                elif column["type"] == "TEXT":
                    record.append(str(value))
                else:
                    record.append(int(float(value)) if column["type"] == "INTEGER" else float(value))
            records.append(record)
        db = sqlite3.connect(":memory:", check_same_thread=False)
        db.execute(f"CREATE TABLE {self.table} ({', '.join(c['name'] + ' ' + c['type'] for c in spec['columns'])})")
        db.executemany(f"INSERT INTO {self.table} VALUES ({', '.join('?' * len(spec['columns']))})", records)
        db.execute("CREATE TABLE fields (column_name TEXT, type TEXT, description TEXT, stage TEXT)")
        db.executemany("INSERT INTO fields VALUES (?, ?, ?, ?)", [(c["name"], c["type"], c["description"], c["stage"]) for c in spec["columns"]])
        db.commit()
        db.execute("PRAGMA query_only = 1")
        db.set_authorizer(_read_only)
        self.db = db
        self._lock = threading.Lock()                               # one connection, used from worker threads
        self._overview = self._build_overview(records)

    # --- SQL -------------------------------------------------------------------------------------
    def sql(self, statement: str) -> dict[str, Any]:
        with self._lock:
            deadline = time.monotonic() + self.spec["timeout_seconds"]
            self.db.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
            try:
                cursor = self.db.execute(statement)
                rows = cursor.fetchmany(self.spec["max_rows"] + 1)
            except sqlite3.Error as exc:
                message = str(exc)
                if "interrupted" in message:
                    message = f"查询超过 {self.spec['timeout_seconds']:g} 秒被中止，请缩小范围或简化查询"
                raise ValueError(f"SQL 执行失败：{message}") from None
            except sqlite3.Warning as exc:                           # e.g. two statements in one call
                raise ValueError(f"SQL 执行失败：{exc}") from None
            finally:
                self.db.set_progress_handler(None, 0)
        limit = self.spec["max_rows"]
        note = f"只读查询（SELECT）；最多返回 {limit} 行，请尽量用 COUNT、AVG、GROUP BY 等聚合。"
        if self.spec["min_group_size"]:
            note += f"少于 {self.spec['min_group_size']} 例的分组结果不稳定，不要据此下结论。"
        return {"检索范围": self.scope, "columns": [d[0] for d in cursor.description or ()],
                "rows": [[_jsonable(v) for v in r] for r in rows[:limit]], "返回行数": min(len(rows), limit),
                "已截断": len(rows) > limit, "说明": note}

    # --- overview and examples ---------------------------------------------------------------------
    def _build_overview(self, records: list[list[Any]]) -> dict[str, Any]:
        columns = []
        for i, column in enumerate(self.spec["columns"]):
            values = [r[i] for r in records if r[i] is not None]
            entry = {"列名": column["name"], "含义": column["description"], "类型": column["type"], "时点": column["stage"],
                     "缺失率": round(1 - len(values) / max(len(records), 1), 3)}
            if column["type"] == "TEXT":
                counts: dict[str, int] = {}
                for v in values:
                    counts[v] = counts.get(v, 0) + 1
                top = sorted(counts.items(), key=lambda kv: -kv[1])[:12]
                entry["取值"] = {k: c for k, c in top} if len(counts) <= 40 else f"{len(counts)} 种取值（自由文本）"
            elif values and set(values) <= {0, 1}:
                entry["为 1 的比例"] = round(sum(values) / len(values), 3)
            elif values:
                q = statistics.quantiles(values, n=4) if len(values) > 1 else [values[0]] * 3
                entry.update({"中位数": _jsonable(q[1]), "P25–P75": [_jsonable(q[0]), _jsonable(q[2])],
                              "范围": [_jsonable(min(values)), _jsonable(max(values))]})
            columns.append(entry)
        label_index = [c["name"] for c in self.spec["columns"]].index(f"{self.spec['label']}_group") \
            if f"{self.spec['label']}_group" in [c["name"] for c in self.spec["columns"]] else None
        groups = {}
        if label_index is not None:
            for g in self.spec["groups"]:
                n = sum(r[label_index] == g for r in records)
                groups[g] = {"例数": n, "占比": round(n / max(len(records), 1), 3)}
        return {"表": self.table, "检索范围": self.scope, "病例数": len(records), "获卵数分组": groups, "列": columns,
                "说明": f"每行一个促排周期，没有任何编号；缺失为 NULL。另有表 fields（列名、类型、含义、时点）。"
                      f"时点：baseline 进周期前已知，plan 启动方案，during 促排过程中才知道，outcome 结局。"
                      f"SQL 用 SQLite 语法，只允许 SELECT。"}

    def overview(self) -> dict[str, Any]:
        return self._overview

    def examples(self) -> dict[str, Any]:
        return {"表": self.table, "示例": self.spec["examples"], "说明": "示例里的数值按当前患者的情况替换；可以直接交给 SQL 工具执行。"}


def _read_only(action: int, *args: Any) -> int:
    allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, getattr(sqlite3, "SQLITE_RECURSIVE", 33)}
    return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY


def _jsonable(value: Any) -> Any:
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return int(value) if value.is_integer() else round(value, 4)
    return value
