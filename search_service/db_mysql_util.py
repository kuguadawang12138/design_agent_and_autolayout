# -*- coding: utf-8 -*-
"""MySQL 异常中文说明（含死锁定位，不做自动重试）。"""

from __future__ import annotations

import re
import traceback

from agent_log import log_line


def describe_mysql_error(
    exc: BaseException,
    *,
    operation: str,
    table: str,
    keys: str = "",
) -> str:
    errno = None
    msg = str(exc)
    args = getattr(exc, "args", None)
    if args and len(args) >= 1:
        try:
            errno = int(args[0])
        except (TypeError, ValueError):
            pass
        if len(args) >= 2:
            msg = str(args[1])
    key_part = f" 键={keys}" if keys else ""
    base = f"操作={operation} 表={table}{key_part} errno={errno} 原文={msg}"
    if errno == 1213:
        hint = (
            "【InnoDB 死锁】两路及以上事务互相等待行锁。"
            "常见争用表：footprint_library、project_pdf、ingest_failure、process。"
        )
        return f"{hint} | {base}"
    if errno == 1205:
        return f"【锁等待超时】{base}"
    return f"【数据库异常】{base}"


def log_mysql_error(
    exc: BaseException,
    *,
    operation: str,
    table: str,
    keys: str = "",
    level: str = "error",
    include_trace: bool = True,
) -> str:
    text = describe_mysql_error(exc, operation=operation, table=table, keys=keys)
    trace = ""
    if include_trace:
        trace = traceback.format_exc(limit=8)
        trace = re.sub(r"\s+", " ", trace.replace("\n", " | "))[:1200]
    log_line("数据库", detail=trace or "-", result=text, level=level)
    return text
