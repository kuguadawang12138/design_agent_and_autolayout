# -*- coding: utf-8 -*-
"""二开 HTTP 接口并发控制：接口① BRD 串行队列；接口②③ 并发上限。"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Callable, TypeVar

T = TypeVar("T")


class BrdNextQueueFull(Exception):
    """等待拉取 BRD 的请求已达上限。"""


class BrdNextQueue:
    """
    接口 GET /api/v1/brd/next：全局串行下发（避免多客户端抢同一文件），
    最多 max_waiting 个请求排队，超出返回 503。
    """

    def __init__(self, max_waiting: int = 10) -> None:
        self._max = max(1, int(max_waiting))
        self._waiting = 0
        self._counter_lock = asyncio.Lock()
        self._serve_lock = asyncio.Lock()

    async def run(self, fn: Callable[[], T | None]) -> T | None:
        async with self._counter_lock:
            if self._waiting >= self._max:
                raise BrdNextQueueFull(f"BRD 拉取队列已满（上限 {self._max}），请稍后重试")
            self._waiting += 1
        try:
            async with self._serve_lock:
                return await asyncio.to_thread(fn)
        finally:
            async with self._counter_lock:
                self._waiting -= 1


class IngestConcurrencyGate:
    """接口 POST /api/v1/ingest 与 /failure：允许多线程并发，上限可配置。"""

    def __init__(self, max_concurrent: int = 32) -> None:
        self._sem = asyncio.Semaphore(max(1, int(max_concurrent)))

    async def run(self, fn: Callable[[], Any]) -> Any:
        async with self._sem:
            if inspect.iscoroutinefunction(fn):
                return await fn()
            return await asyncio.to_thread(fn)


_brd_queue: BrdNextQueue | None = None
_ingest_gate: IngestConcurrencyGate | None = None


def init_erkai_queues(*, brd_max_waiting: int = 10, ingest_max_concurrent: int = 32) -> None:
    global _brd_queue, _ingest_gate
    _brd_queue = BrdNextQueue(max_waiting=brd_max_waiting)
    _ingest_gate = IngestConcurrencyGate(max_concurrent=ingest_max_concurrent)


def brd_queue() -> BrdNextQueue:
    if _brd_queue is None:
        init_erkai_queues()
    assert _brd_queue is not None
    return _brd_queue


def ingest_gate() -> IngestConcurrencyGate:
    if _ingest_gate is None:
        init_erkai_queues()
    assert _ingest_gate is not None
    return _ingest_gate
