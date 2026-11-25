# -*- coding: utf-8 -*-
"""检索服务性能日志：一次请求 / 一次对话，各模块耗时汇总到单行。"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

from agent_log import log_line

# 对话流水线步骤名 → 日志中文
_STEP_LABEL_CN: dict[str, str] = {
    "init_session": "加载会话",
    "parse_intent": "理解意图",
    "resolve_json": "定位位号JSON",
    "similar_footprint": "形态IoU检索",
    "similar_parse": "解析查询JSON",
    "similar_sql": "查同脚数候选",
    "similar_minio": "MinIO拉取形态",
    "similar_iou": "计算IoU",
    "fetch_fields": "拉取候选字段",
    "fetch_minio_json": "MinIO拉取器件JSON",
    "llm_rank": "LLM排序",
    "pdf_match": "匹配PDF",
    "pdf_format": "格式化PDF链接",
    "pdf_highlight": "PDF高亮",
    "pdf_minio_pdf": "MinIO拉取PDF",
    "pdf_minio_json": "MinIO拉取高亮JSON",
    "pdf_format_minio": "MinIO拉取PDF链接",
    "reply": "生成回复",
    "clarification": "追问澄清",
    "chat_memory": "写入对话记忆",
    "pdf_pdf_match": "匹配PDF",
    "pdf_pdf_format": "格式化PDF链接",
}


class StepTimer:
    """按步骤累计毫秒，最终一行输出。"""

    __slots__ = ("_steps",)

    def __init__(self) -> None:
        self._steps: dict[str, int] = {}

    def add(self, name: str, ms: int) -> None:
        if ms < 0:
            ms = 0
        self._steps[name] = self._steps.get(name, 0) + ms

    def merge(self, other: dict[str, int] | None) -> None:
        if not other:
            return
        for key, ms in other.items():
            self.add(key, int(ms))

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, int((time.perf_counter() - t0) * 1000))

    def total_ms(self) -> int:
        return sum(self._steps.values())

    def format_steps(self) -> str:
        if not self._steps:
            return "-"
        return " ".join(
            f"{_STEP_LABEL_CN.get(k, k)}={v}ms" for k, v in self._steps.items()
        )

    def log(
        self,
        action: str,
        *,
        session_id: str | None = None,
        detail: str = "",
        extra: str = "",
        total_ms: int | None = None,
        level: str = "info",
    ) -> None:
        """单行：total + 各 step。"""
        total = self.total_ms() if total_ms is None else total_ms
        parts = [f"总耗时={total}ms", self.format_steps()]
        if extra:
            parts.append(extra)
        log_line(
            action,
            session_id=session_id,
            detail=detail,
            result=" | ".join(parts),
            level=level,
        )


def merge_step_ms(state: dict, name: str, ms: int) -> dict[str, int]:
    """LangGraph state 内累计步骤耗时。"""
    steps = dict(state.get("_step_ms") or {})
    steps[name] = steps.get(name, 0) + max(0, int(ms))
    return steps
