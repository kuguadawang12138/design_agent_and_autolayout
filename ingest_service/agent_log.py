# -*- coding: utf-8 -*-
"""单行结构化日志：时间 | session_id | action | detail | result。

控制台照常打印，同时按天写入入库服务下的 logs 目录。
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

_TZ = ZoneInfo("Asia/Shanghai")


def _utc_ts() -> str:
    return datetime.now(_TZ).strftime("%Y-%m-%d %H:%M:%S")


def _log_dir() -> Path:
    path = Path(__file__).resolve().parent / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


class _DailyFileHandler(logging.Handler):
    """一天一个文件：logs/年-月-日.log。"""

    def __init__(self) -> None:
        super().__init__()
        self._day = ""
        self._stream = None

    def emit(self, record: logging.LogRecord) -> None:
        try:
            day = datetime.now(_TZ).strftime("%Y-%m-%d")
            if day != self._day or self._stream is None:
                if self._stream is not None:
                    self._stream.close()
                self._day = day
                path = _log_dir() / f"{day}.log"
                self._stream = path.open("a", encoding="utf-8")
            self._stream.write(self.format(record) + "\n")
            self._stream.flush()
        except Exception:
            self.handleError(record)


def get_logger(name: str = "pcb_agent") -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        formatter = logging.Formatter("%(message)s")
        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        daily = _DailyFileHandler()
        daily.setFormatter(formatter)
        log.addHandler(stream)
        log.addHandler(daily)
        log.setLevel(logging.INFO)
        log.propagate = False
    return log


def log_line(
    action: str,
    *,
    session_id: str | None = None,
    detail: str = "",
    result: str = "",
    level: str = "info",
    logger: logging.Logger | None = None,
) -> None:
    """打印一行日志，便于 grep。入库没有会话号时不印这一项。"""
    log = logger or get_logger()
    sid = str(session_id or "").strip()
    parts = [f"time={_utc_ts()}"]
    if sid and sid != "-":
        parts.append(f"session_id={sid}")
    parts.append(f"action={action}")
    if detail:
        parts.append(f"detail={detail}")
    if result:
        parts.append(f"result={result}")
    line = " | ".join(parts)
    getattr(log, level, log.info)(line)
