"""Training-case database the agent explores itself with read-only SQL (tool ivf_case_sql).

Data: ``services/data/train_cases_35.csv`` holds the 1132 cleaned training cases (35 indicators and
the outcome) without case IDs, in an order fixed by a hash of the ID. The IDs and their order are
left out because in this dataset they alone separate the outcomes. A ``fold`` column (the same
stratified folds as the other tools) lets a service hide the fold of the case under evaluation;
the agent never sees it. The sealed test set is not in the table.

``build_table`` writes that file from the training CSV (run once; the file is in the repository).
``CaseDatabase`` loads it into an in-memory SQLite database holding only the visible cases, with
ASCII column names for SQL (``COLUMNS``) and a ``fields`` table mapping them to the indicator names.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sqlite3
import threading
import time
from collections.abc import Sequence
from pathlib import Path

from clinical_tools.cleaning import load_training
from clinical_tools.common import FEATURE_NAMES, assign_folds

SQL_MAX_ROWS = 200
SQL_TIMEOUT_S = 2.0
K_FOLDS = 5
OUTCOME = "pregnant"
# indicator name -> SQL column (ASCII, so the agent can write SQL without quoting)
COLUMNS = {
    "女方年龄": "age", "女方基础FSH": "fsh", "女方基础PRL": "prl", "女方基础T": "testosterone",
    "女方基础P": "progesterone", "女方AMH": "amh", "女方白细胞计数": "wbc", "女方血小板": "platelets",
    "女方血细胞压积": "hematocrit", "女方凝血常规PT": "pt", "女方凝血常规APPT": "aptt", "女方凝血常规TT": "tt",
    "女方D二聚体": "d_dimer", "女方甲功TPO": "tpo", "女方空腹血糖": "glucose", "女方肾功尿酸": "uric_acid",
    "男方浓度": "m_concentration", "男方正常形态率": "m_normal_morphology", "男方PR": "m_pr", "男方NP": "m_np",
    "男方存活率": "m_survival", "男方DFI": "m_dfi", "男方ALT": "m_alt", "男方AST": "m_ast",
    "男方血肌酐": "m_creatinine", "男方白细胞": "m_wbc", "男方血小板": "m_platelets", "男方空腹血糖": "m_glucose",
    "女方用药Gn总天数": "gn_days", "女方用药HCG日P": "hcg_day_p", "男方精液处理后NP": "wash_np",
    "男方精液处理方法": "wash_method", "男方精液处理后密度": "wash_density", "男方精液处理后PR": "wash_pr",
    "男方精液处理后IM": "wash_im",
}
assert tuple(COLUMNS) == tuple(FEATURE_NAMES)
TABLE_HEADER = ("fold", *FEATURE_NAMES, OUTCOME)


def _fmt(value: float) -> str:
    return "" if math.isnan(value) else f"{value:.10g}"


def build_table(source: Path, target: Path) -> dict:
    """Write the ID-free case table from the training CSV (cleaned with the tools' rules)."""
    training = load_training(source)
    if len(training.case_ids) != 1132:
        raise SystemExit("expected the 1132-case training set; the sealed test set must not be used")
    folds = assign_folds(training.case_ids, training.labels.tolist(), K_FOLDS)
    order = sorted(range(len(folds)), key=lambda i: hashlib.sha256(f"case-table:{training.case_ids[i]}".encode()).hexdigest())
    with open(target, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(TABLE_HEADER)
        for i in order:
            writer.writerow([folds[i], *(_fmt(v) for v in training.matrix[i]), int(training.labels[i])])
    return {"rows": len(order), "pregnant": int(training.labels.sum()), "target": str(target)}


def fold_of(case_id: str, training_csv: str | Path) -> int:
    """Fold of a training case, from the ID-bearing training CSV the other tools read."""
    training = load_training(training_csv)
    if case_id not in training.case_ids:
        raise ValueError(f"exclude_case {case_id} is not a training case")
    return assign_folds(training.case_ids, training.labels.tolist(), K_FOLDS)[training.case_ids.index(case_id)]


class CaseDatabase:
    def __init__(self, table: str | Path, *, exclude_fold: int | None = None):
        with open(table, encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            if tuple(next(reader)) != TABLE_HEADER:
                raise ValueError("unexpected case table header")
            rows = [r for r in reader if exclude_fold is None or int(r[0]) != exclude_fold]
        self.scope = "全部训练病例" if exclude_fold is None else f"训练病例中排除当前病例所在的 1/{K_FOLDS}"
        self.labels = [int(r[-1]) for r in rows]
        db = sqlite3.connect(":memory:", check_same_thread=False)
        cols = ", ".join(f"{c} REAL" for c in COLUMNS.values())
        db.execute(f"CREATE TABLE cases ({cols}, {OUTCOME} INTEGER)")
        db.executemany(f"INSERT INTO cases VALUES ({', '.join('?' * (len(COLUMNS) + 1))})",
                       [[float(v) if v else None for v in r[1:-1]] + [int(r[-1])] for r in rows])
        db.execute("CREATE TABLE fields (column_name TEXT, indicator TEXT)")
        db.executemany("INSERT INTO fields VALUES (?, ?)", [(c, n) for n, c in COLUMNS.items()]
                       + [(OUTCOME, "临床妊娠结局（1 妊娠，0 未妊娠）")])
        db.commit()
        db.execute("PRAGMA query_only = 1")
        db.set_authorizer(_read_only)
        self.db = db
        self._lock = threading.Lock()      # one connection, used from the service's worker threads

    def sql(self, statement: str) -> dict:
        with self._lock:
            return self._sql(statement)

    def _sql(self, statement: str) -> dict:
        deadline = time.monotonic() + SQL_TIMEOUT_S
        self.db.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
        try:
            cursor = self.db.execute(statement)
            rows = cursor.fetchmany(SQL_MAX_ROWS + 1)
        except sqlite3.Error as exc:
            message = str(exc)
            if "interrupted" in message:
                message = f"查询超过 {SQL_TIMEOUT_S:g} 秒被中止，请缩小范围或简化查询"
            raise ValueError(f"SQL 执行失败：{message}") from None
        finally:
            self.db.set_progress_handler(None, 0)
        columns = [d[0] for d in cursor.description or ()]
        truncated = len(rows) > SQL_MAX_ROWS
        rows = [[_jsonable(v) for v in r] for r in rows[:SQL_MAX_ROWS]]
        return {"检索范围": self.scope, "columns": columns, "rows": rows, "返回行数": len(rows),
                "已截断": truncated,
                "说明": f"只读查询；表 cases（每行一个训练病例，无病例编号）和 fields（列名与指标对照）；"
                      f"最多返回 {SQL_MAX_ROWS} 行，请尽量用聚合（COUNT、AVG、GROUP BY）。"}


def _read_only(action: int, *args) -> int:
    allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}
    allowed.add(getattr(sqlite3, "SQLITE_RECURSIVE", 33))
    return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY


def _jsonable(value):
    if isinstance(value, float):
        return round(value, 4) if math.isfinite(value) else None
    return value


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, required=True, help="the 1132-case training CSV (with IDs)")
    parser.add_argument("--out", type=Path, required=True, help="case table to write (no IDs)")
    args = parser.parse_args(argv)
    print(build_table(args.source, args.out))


if __name__ == "__main__":
    main()
