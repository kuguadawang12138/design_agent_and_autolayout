# -*- coding: utf-8 -*-
"""只读查看正式库。按表拉数据，或对这张表执行一条查询语句。"""

from __future__ import annotations

import re
from datetime import date, datetime
from decimal import Decimal
from typing import Any

_TABLE_NAME = re.compile(r"^[A-Za-z0-9_]+$")
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|replace|grant|revoke|"
    r"rename|call|load|into|outfile|dumpfile|lock|unlock|handler)\b",
    re.IGNORECASE,
)
_ALLOWED_HEAD = re.compile(r"^(select|with|show|describe|desc|explain)\b", re.IGNORECASE)
_LIMIT_MAX = 500


def _clean_sql(sql: str) -> str:
    text = str(sql or "").strip()
    if text.endswith(";"):
        text = text[:-1].strip()
    return text


def prepare_read(table: str, sql: str = "", limit: int = 100) -> tuple[str, int]:
    """收成一条只读语句。写入、多语句、没点到这张表的都拒绝。"""
    name = str(table or "").strip()
    if not _TABLE_NAME.fullmatch(name):
        raise ValueError("表名只允许字母、数字和下划线")
    try:
        size = int(limit)
    except (TypeError, ValueError):
        size = 100
    size = min(_LIMIT_MAX, max(1, size))
    text = _clean_sql(sql)
    if not text:
        return f"SELECT * FROM `{name}`", size
    if ";" in text:
        raise ValueError("一次只执行一条语句，不要写分号")
    if not _ALLOWED_HEAD.match(text):
        raise ValueError("只允许 SELECT、WITH、SHOW、DESCRIBE、EXPLAIN")
    if _FORBIDDEN.search(text):
        raise ValueError("这条语句包含修改数据的内容，已拒绝")
    if not re.search(rf"\b{re.escape(name)}\b", text, re.IGNORECASE):
        raise ValueError(f"语句里没有表 {name}")
    return text, size


def _cell(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        return f"<二进制 {len(value)} 字节>"
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, str) and len(value) > 4000:
        return value[:4000] + "…"
    return value


def query_table(table: str, sql: str = "", limit: int = 100) -> dict[str, Any]:
    """查询一张表。二进制内容只返回长度，避免把 PDF 整段传出去。"""
    statement, size = prepare_read(table, sql, limit)
    from footprint_db import connect

    conn = connect()
    try:
        name = str(table).strip()
        with conn.cursor() as cur:
            cur.execute("SHOW TABLES")
            names = {str(row[0]) for row in cur.fetchall()}
            if name not in names:
                raise ValueError(f"没有这张表：{name}")
            cur.execute(statement)
            rows = cur.fetchmany(size)
            columns = [item[0] for item in (cur.description or [])]
        conn.rollback()
    finally:
        conn.close()
    return {
        "ok": True,
        "table": str(table).strip(),
        "sql": statement,
        "columns": columns,
        "rows": [dict(zip(columns, (_cell(item) for item in row))) for row in rows],
        "count": len(rows),
        "limit": size,
    }
