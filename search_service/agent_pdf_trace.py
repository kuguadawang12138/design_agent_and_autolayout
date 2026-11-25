# -*- coding: utf-8 -*-
"""PDF 接口分步耗时：写日志 + 响应头 X-Perf-Detail，供 benchmark 脚本对比。"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

from agent_log import log_line

# 日志用中文；HTTP 头仍用英文键名便于脚本解析
_ROUTE_LABEL_CN: dict[str, str] = {
    "region": "高亮区域",
    "file": "PDF文件",
    "preview": "PNG预览",
    "component_file": "位号PDF",
    "component_preview": "位号预览",
}

_STEP_LABEL_CN: dict[str, str] = {
    "db_connect": "连接数据库",
    "get_row_db_pdf": "读取PDF记录",
    "minio_pdf": "MinIO拉取PDF",
    "minio_json": "MinIO拉取器件JSON",
    "db_fetch_component": "读取器件行",
    "build_region_stored": "库内坐标算高亮框",
    "render_pdf": "渲染PDF",
    "render_png": "渲染PNG",
    "encrypt_pdf": "加密PDF",
    "response_bytes": "组装响应",
}


class PdfPerfTrace:
    def __init__(self, *, route: str, detail: str = "") -> None:
        self.route = route
        self.detail = detail
        self._t0 = time.perf_counter()
        self._steps: list[tuple[str, int]] = []

    @contextmanager
    def step(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._steps.append((name, int((time.perf_counter() - t0) * 1000)))

    def add(self, name: str, ms: int) -> None:
        self._steps.append((name, max(0, int(ms))))

    def total_ms(self) -> int:
        return int((time.perf_counter() - self._t0) * 1000)

    def as_dict(self) -> dict:
        steps = {k: v for k, v in self._steps}
        steps_line = " ".join(
            f"{_STEP_LABEL_CN.get(k, k)}={v}ms" for k, v in self._steps
        )
        return {
            "route": self.route,
            "detail": self.detail,
            "total_ms": self.total_ms(),
            "steps": steps,
            "steps_line": steps_line,
        }

    def header_value(self) -> str:
        """HTTP 头（ASCII），脚本可解析。"""
        compact = "|".join(f"{k}:{v}" for k, v in self._steps)
        return f"total={self.total_ms()};{compact}"

    def log_line(self, *, session_id: str | None = None) -> None:
        d = self.as_dict()
        route_cn = _ROUTE_LABEL_CN.get(self.route, self.route)
        log_line(
            "PDF接口耗时",
            session_id=session_id,
            detail=f"路由={route_cn} {self.detail}".strip(),
            result=f"总耗时={d['total_ms']}ms | {d['steps_line']}",
        )

    def apply_headers(self, response) -> None:
        try:
            response.headers["X-Perf-Detail"] = self.header_value()
            response.headers["X-Perf-Total-Ms"] = str(self.total_ms())
        except Exception:
            pass
