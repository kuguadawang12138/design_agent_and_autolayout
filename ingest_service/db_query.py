# -*- coding: utf-8 -*-
"""按表分页查看。顺序固定用主键，翻页不会重复，也不会每次换一批。"""

from __future__ import annotations

import re
from datetime import date, datetime
from decimal import Decimal

_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PAGE_SIZE_MAX = 500


def _cell(value):
    if isinstance(value, (bytes, bytearray)):
        return f"[二进制 {len(value)} 字节]"
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    return value


def _ident(name: str, label: str) -> str:
    text = str(name or "").strip()
    if not _NAME.fullmatch(text):
        raise ValueError(f"{label}只能包含字母、数字和下划线")
    return text


def _columns(cur, table: str) -> set[str]:
    cur.execute(f"SHOW COLUMNS FROM `{table}`")
    return {str(row[0]) for row in cur.fetchall()}


def _primary_key(cur, table: str) -> list[str]:
    cur.execute(f"SHOW KEYS FROM `{table}` WHERE Key_name = 'PRIMARY'")
    rows = sorted(cur.fetchall(), key=lambda item: int(item[3]))
    names = [str(item[4]) for item in rows if item[4]]
    if not names:
        raise ValueError(f"表 {table} 没有主键，不能稳定分页")
    return names


def list_rows(
    table: str,
    project_code: str | None = None,
    refdes: str | None = None,
    page: int = 1,
    page_size: int = 20,
) -> dict:
    """按固定主键顺序返回一页。同一条件再次请求同一页，结果相同。"""
    from footprint_db import connect

    name = _ident(table, "表名")
    try:
        page_no = int(page)
        size = int(page_size)
    except (TypeError, ValueError) as exc:
        raise ValueError("页码和每页条数必须是整数") from exc
    if page_no < 1:
        raise ValueError("页码从 1 开始")
    if size < 1 or size > _PAGE_SIZE_MAX:
        raise ValueError(f"每页条数要在 1 到 {_PAGE_SIZE_MAX} 之间")
    code = str(project_code or "").strip()
    ref = str(refdes or "").replace("@", "").strip().upper()
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute("SHOW TABLES LIKE %s", (name,))
            if cur.fetchone() is None:
                raise ValueError(f"表不存在：{name}")
            columns = _columns(cur, name)
            keys = _primary_key(cur, name)
            where = []
            params: list = []
            if code:
                if "project_code" not in columns:
                    raise ValueError(f"表 {name} 没有 project_code，不能按项目编码过滤")
                where.append("project_code = %s")
                params.append(code)
            if ref:
                if "refdes" not in columns:
                    raise ValueError(f"表 {name} 没有 refdes，不能按位号过滤")
                where.append("UPPER(REPLACE(TRIM(refdes), '@', '')) = %s")
                params.append(ref)
            clause = (" WHERE " + " AND ".join(where)) if where else ""
            order = ", ".join(f"`{key}` ASC" for key in keys)
            cur.execute(f"SELECT COUNT(*) FROM `{name}`{clause}", params)
            total = int(cur.fetchone()[0])
            offset = (page_no - 1) * size
            cur.execute(
                f"SELECT * FROM `{name}`{clause} ORDER BY {order} LIMIT %s OFFSET %s",
                [*params, size, offset],
            )
            names = [item[0] for item in (cur.description or [])]
            records = cur.fetchall()
        conn.rollback()
    finally:
        conn.close()
    rows = [{names[i]: _cell(record[i]) for i in range(len(names))} for record in records]
    total_pages = (total + size - 1) // size if total else 0
    return {
        "ok": True,
        "table": name,
        "project_code": code or None,
        "refdes": ref or None,
        "page": page_no,
        "page_size": size,
        "total": total,
        "total_pages": total_pages,
        "order_by": keys,
        "columns": names,
        "row_count": len(rows),
        "rows": rows,
    }
