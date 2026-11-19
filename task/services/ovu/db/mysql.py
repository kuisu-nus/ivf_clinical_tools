"""The MySQL case database: the schema built from the config, loading, and the read-only engine.

Layout (``<db>`` is ``mysql.database`` in db.yaml, default ``ovu``):
  <db>_base.cases_all   every training case with its ``cv_fold``; only the admin account reads it
  <db>_base.fields      column name, MySQL type and meaning of every column
  <db>.cases            view of all training cases, without ``cv_fold``        -> reader account <reader>
  <db>_f<k>.cases       the same without fold k (for training an agent on fold k) -> reader account <reader>_f<k>
  each view database also has a view ``fields``

Each reader account may only SELECT from its own view database, so the tools cannot reach
the base table, another fold or anything else on the server. The column types come from
data.yaml (int -> INT, float -> DOUBLE, category -> VARCHAR, diagnosis categories -> TINYINT),
and each column carries its Chinese name as a COMMENT.

Environment variables (passwords never go into a config file):
  OVU_MYSQL_HOST, OVU_MYSQL_PORT        server (default 127.0.0.1:3306)
  OVU_MYSQL_ADMIN_USER, OVU_MYSQL_ADMIN_PASSWORD
                                        account that creates the databases, views and reader
                                        accounts (only ``python -m ovu.db.build`` uses it)
  OVU_MYSQL_PASSWORD                    password of the reader accounts (the tools use it)
"""

from __future__ import annotations

import os
import threading
from typing import Any

try:
    import pymysql
except ImportError:                      # only needed to talk to the server; the schema is built without it
    pymysql = None


def _require() -> None:
    if pymysql is None:
        raise SystemExit("需要 pymysql：python -m pip install --user pymysql")

TIMEOUT_ERRORS = (3024, 1969, 1317)      # MySQL max_execution_time, MariaDB max_statement_time, interrupted


def settings() -> dict[str, Any]:
    return {"host": os.environ.get("OVU_MYSQL_HOST", "127.0.0.1"), "port": int(os.environ.get("OVU_MYSQL_PORT", "3306")),
            "admin_user": os.environ.get("OVU_MYSQL_ADMIN_USER", ""), "admin_password": os.environ.get("OVU_MYSQL_ADMIN_PASSWORD", ""),
            "reader_password": os.environ.get("OVU_MYSQL_PASSWORD", "")}


def names(spec: dict[str, Any], fold: int | None = None) -> tuple[str, str]:
    """(view database, reader account) for all cases, or for all cases without ``fold``."""
    db, reader = spec["mysql"]["database"], spec["mysql"]["reader"]
    return (db, reader) if fold is None else (f"{db}_f{fold}", f"{reader}_f{fold}")


def _quote(text: str) -> str:
    return "'" + text.replace("\\", "\\\\").replace("'", "''") + "'"


def schema(spec: dict[str, Any], folds: list[int]) -> list[str]:
    """Every statement that creates the databases, the base table and the views (no data, no accounts)."""
    base = f"{spec['mysql']['database']}_base"
    columns = ",\n  ".join(f"`{c['name']}` {c['mysql_type']} NULL COMMENT {_quote(c['description'][:1000])}" for c in spec["columns"])
    listed = ", ".join(f"`{c['name']}`" for c in spec["columns"])
    out = [f"CREATE DATABASE IF NOT EXISTS `{base}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci",
           f"DROP TABLE IF EXISTS `{base}`.`cases_all`",
           f"CREATE TABLE `{base}`.`cases_all` (\n  `cv_fold` TINYINT NULL COMMENT '交叉验证折号（工具不可见）',\n  {columns}\n)"
           " ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='获卵数预测：训练集病例，每行一个促排周期，不含任何编号'",
           f"DROP TABLE IF EXISTS `{base}`.`fields`",
           f"CREATE TABLE `{base}`.`fields` (`column_name` VARCHAR(64), `type` VARCHAR(32), `description` TEXT)"
           " ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"]
    for fold in [None, *folds]:
        view_db, _ = names(spec, fold)
        where = "" if fold is None else f" WHERE `cv_fold` IS NULL OR `cv_fold` <> {int(fold)}"
        out += [f"CREATE DATABASE IF NOT EXISTS `{view_db}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci",
                f"CREATE OR REPLACE SQL SECURITY DEFINER VIEW `{view_db}`.`{spec['table']}` AS SELECT {listed} FROM `{base}`.`cases_all`{where}",
                f"CREATE OR REPLACE SQL SECURITY DEFINER VIEW `{view_db}`.`fields` AS SELECT * FROM `{base}`.`fields`"]
    return out


def accounts(spec: dict[str, Any], folds: list[int], password: str) -> list[str]:
    """The reader accounts, each allowed to SELECT from its own view database only."""
    host = spec["mysql"]["reader_host"]
    out = []
    for fold in [None, *folds]:
        view_db, reader = names(spec, fold)
        out += [f"CREATE USER IF NOT EXISTS {_quote(reader)}@{_quote(host)} IDENTIFIED BY {_quote(password)}",
                f"ALTER USER {_quote(reader)}@{_quote(host)} IDENTIFIED BY {_quote(password)}",
                f"GRANT SELECT ON `{view_db}`.* TO {_quote(reader)}@{_quote(host)}"]
    return out


def admin_connection(config: dict[str, Any]) -> Any:
    _require()
    if not config["admin_user"]:
        raise SystemExit("没有设置 OVU_MYSQL_ADMIN_USER / OVU_MYSQL_ADMIN_PASSWORD，无法建库导入。"
                         "只想生成建表语句可以加 --schema-only。")
    return pymysql.connect(host=config["host"], port=config["port"], user=config["admin_user"], password=config["admin_password"],
                           charset="utf8mb4", autocommit=True, connect_timeout=10)


def load(spec: dict[str, Any], values: list[list[Any]], folds_of_rows: list[int | None], folds: list[int],
         config: dict[str, Any]) -> list[str]:
    """Create everything and load the training cases. Returns notes for the person running it."""
    notes = []
    base = f"{spec['mysql']['database']}_base"
    connection = admin_connection(config)
    try:
        with connection.cursor() as cursor:
            for statement in schema(spec, folds)[:5]:                # base database and tables first
                cursor.execute(statement)
            placeholders = ", ".join(["%s"] * (len(spec["columns"]) + 1))
            listed = ", ".join(["`cv_fold`"] + [f"`{c['name']}`" for c in spec["columns"]])
            rows = [[fold, *record] for fold, record in zip(folds_of_rows, values)]
            for start in range(0, len(rows), 1000):
                cursor.executemany(f"INSERT INTO `{base}`.`cases_all` ({listed}) VALUES ({placeholders})", rows[start:start + 1000])
            cursor.executemany(f"INSERT INTO `{base}`.`fields` VALUES (%s, %s, %s)",
                               [(c["name"], c["mysql_type"], c["description"]) for c in spec["columns"]])
            for statement in schema(spec, folds)[5:]:
                cursor.execute(statement)
            if config["reader_password"]:
                try:
                    for statement in accounts(spec, folds, config["reader_password"]):
                        cursor.execute(statement)
                    cursor.execute("FLUSH PRIVILEGES")
                except pymysql.err.MySQLError as exc:
                    notes.append(f"只读账号没有建成（{exc.args[-1]}）：请让数据库管理员执行 schema.sql 末尾注释里的账号语句。")
            else:
                notes.append("没有设置 OVU_MYSQL_PASSWORD，没有创建只读账号；设置后重跑，或让管理员执行 schema.sql 末尾注释里的账号语句。")
    finally:
        connection.close()
    return notes


class MySQLEngine:
    """Answers the tools' queries through a reader account that only sees its own view database."""
    dialect = "MySQL"

    def __init__(self, spec: dict[str, Any], *, hide_fold: int | None = None, config: dict[str, Any] | None = None):
        self.spec = spec
        self.config = config or settings()
        self.database, self.user = names(spec, hide_fold)
        self._lock = threading.Lock()
        self._connection: Any = None
        _require()
        self._connect()

    def _connect(self) -> None:
        timeout = float(self.spec["timeout_seconds"])
        connection = pymysql.connect(host=self.config["host"], port=self.config["port"], user=self.user,
                                     password=self.config["reader_password"], database=self.database, charset="utf8mb4",
                                     autocommit=True, connect_timeout=10, read_timeout=int(timeout) + 30)
        with connection.cursor() as cursor:
            cursor.execute("SET SESSION TRANSACTION READ ONLY")
            try:
                cursor.execute(f"SET SESSION max_execution_time = {int(timeout * 1000)}")       # MySQL
            except pymysql.err.MySQLError:
                cursor.execute(f"SET SESSION max_statement_time = {timeout:g}")                 # MariaDB
        self._connection = connection

    def type_of(self, column: dict[str, Any]) -> str:
        return column["mysql_type"]

    def _run(self, statement: str, limit: int | None) -> tuple[list[str], list[tuple]]:
        with self._lock:
            try:
                self._connection.ping(reconnect=False)
            except pymysql.err.MySQLError:
                self._connect()
            with self._connection.cursor() as cursor:
                cursor.execute(statement)
                rows = cursor.fetchmany(limit) if limit else cursor.fetchall()
                return [d[0] for d in cursor.description or ()], list(rows)

    def all_records(self) -> list[list[Any]]:
        listed = ", ".join(f"`{c['name']}`" for c in self.spec["columns"])
        _, rows = self._run(f"SELECT {listed} FROM `{self.spec['table']}`", None)
        return [list(row) for row in rows]

    def execute(self, statement: str, limit: int) -> tuple[list[str], list[tuple]]:
        text = statement.strip().rstrip(";").strip()
        first = text.split(None, 1)[0].upper() if text else ""
        if first not in ("SELECT", "WITH") or ";" in text:
            raise ValueError("SQL 执行失败：只允许一条 SELECT 语句")
        try:
            return self._run(text, limit)
        except pymysql.err.MySQLError as exc:
            code = exc.args[0] if exc.args else None
            if code in TIMEOUT_ERRORS:
                raise ValueError(f"SQL 执行失败：查询超过 {self.spec['timeout_seconds']:g} 秒被中止，请缩小范围或简化查询") from None
            raise ValueError(f"SQL 执行失败：{exc.args[-1] if exc.args else exc}") from None
