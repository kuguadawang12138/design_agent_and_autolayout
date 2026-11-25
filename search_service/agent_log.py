# -*- coding: utf-8 -*-
"""单行结构化日志（检索服务）：时间 | 会话 | 动作 | 详情 | 结果。"""

from __future__ import annotations

import contextvars
import logging
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

_TZ = ZoneInfo("Asia/Shanghai")
# 一次检索里各处打印都读这个值，避免日志里会话号是横杠。
_SESSION_ID: contextvars.ContextVar[str] = contextvars.ContextVar("pcb_log_session", default="")


def _local_ts() -> str:
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


def get_logger(name: str = "pcb_search") -> logging.Logger:
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


def use_log_session(session_id: str | None) -> contextvars.Token:
    """当前线程以及从这里复制出去的线程，打印日志都带上这个会话号。"""
    return _SESSION_ID.set(str(session_id or "").strip())


def log_line(
    action: str,
    *,
    session_id: str | None = None,
    detail: str = "",
    result: str = "",
    level: str = "info",
    logger: logging.Logger | None = None,
) -> None:
    """打印一行日志，便于 grep（动作/结果建议中文）。"""
    log = logger or get_logger()
    sid = str(session_id or "").strip() or _SESSION_ID.get() or "-"
    parts = [f"时间={_local_ts()}", f"会话={sid}", f"动作={action}"]
    if detail:
        parts.append(f"详情={detail}")
    if result:
        parts.append(f"结果={result}")
    line = " | ".join(parts)
    getattr(log, level, log.info)(line)
