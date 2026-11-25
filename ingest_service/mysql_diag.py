# -*- coding: utf-8 -*-
"""MySQL 异常诊断（死锁等），便于日志定位表与操作。"""

from __future__ import annotations

import re
from contextlib import contextmanager
from typing import Iterator

from agent_log import log_line

_DEADLOCK_SECTION = re.compile(
    r"LATEST DETECTED DEADLOCK[\s\S]*?(?=\n-{3,}|\Z)",
    re.MULTILINE,
)


def _errno(exc: BaseException) -> int | None:
    args = getattr(exc, "args", None) or ()
    if args and isinstance(args[0], int):
        return int(args[0])
    return None


def _extract_deadlock_snippet(status_text: str) -> str:
    m = _DEADLOCK_SECTION.search(status_text or "")
    if not m:
        return (status_text or "")[:2000]
    return m.group(0).strip()[:3500]


def log_mysql_exception(
    conn,
    exc: BaseException,
    *,
    operation: str,
    detail: str = "",
    sql_hint: str = "",
) -> None:
    """
    将 pymysql 错误写入中文日志；1213 死锁时附加 SHOW ENGINE INNODB STATUS 片段。
    """
    errno = _errno(exc)
    base = f"操作={operation}"
    if detail:
        base += f" | {detail}"
    if sql_hint:
        base += f" | SQL摘要={sql_hint[:500]}"
    msg = str(exc).strip()

    if errno == 1213:
        extra = ""
        try:
            with conn.cursor() as cur:
                cur.execute("SHOW ENGINE INNODB STATUS")
                row = cur.fetchone()
                if row and len(row) >= 3:
                    extra = _extract_deadlock_snippet(str(row[2]))
        except Exception as diag_exc:
            extra = f"（无法读取 INNODB STATUS：{diag_exc}）"
        log_line(
            "数据库死锁",
            result=(
                f"【1213 Deadlock】{base} | pymysql={msg} | "
                f"说明=两事务互相等待行锁，常见于并发写 footprint_library/project_pdf/ingest_failure/process；"
                f"详见下方 INNODB 片段"
            ),
            level="error",
        )
        if extra:
            log_line("数据库死锁", detail="INNODB_STATUS", result=extra, level="error")
        return

    if errno in (1205, 1206):
        log_line(
            "数据库锁等待",
            result=f"【{errno}】{base} | pymysql={msg} | 说明=锁等待超时，可能有长事务占用",
            level="error",
        )
        return

    log_line(
        "数据库错误",
        result=f"【errno={errno or '?'}】{base} | pymysql={msg}",
        level="error",
    )


def commit_or_log(conn, *, operation: str, detail: str = "", sql_hint: str = "") -> None:
    """commit；失败时打诊断日志再抛出。"""
    try:
        conn.commit()
    except Exception as commit_exc:
        log_mysql_exception(conn, commit_exc, operation=operation, detail=detail, sql_hint=sql_hint)
        raise


@contextmanager
def db_transaction(conn, *, operation: str, detail: str = "", sql_hint: str = "") -> Iterator[None]:
    """包住一段写库。语句或提交出现死锁（1213）时立刻读 INNODB STATUS 并打中文日志，然后原样抛出。"""
    try:
        yield
        conn.commit()
    except Exception as exc:
        log_mysql_exception(conn, exc, operation=operation, detail=detail, sql_hint=sql_hint)
        try:
            conn.rollback()
        except Exception:
            pass
        raise
