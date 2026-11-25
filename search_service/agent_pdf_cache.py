# -*- coding: utf-8 -*-
"""进程内 PDF/仿射缓存，避免每次 preview 从 MySQL 拉整份 BLOB 并重算校准。"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np

_CACHE_TTL_SEC = 600
_MAX_ENTRIES = 16
_lock = threading.Lock()
_pdf_cache: dict[tuple[str, bool], tuple[float, bytes, str, float]] = {}
_align_cache: dict[tuple[str, bool], tuple[float, np.ndarray | None, str, dict | None]] = {}


def _evict_if_needed(store: dict) -> None:
    if len(store) <= _MAX_ENTRIES:
        return
    oldest_key = min(store.keys(), key=lambda k: store[k][0])
    store.pop(oldest_key, None)


def get_pdf_bytes(project_code: str, top_side: bool, loader) -> tuple[bytes, str, float]:
    """loader() -> (bytes, file_name, pdf_scale)。"""
    key = (str(project_code or "").strip(), bool(top_side))
    now = time.time()
    with _lock:
        hit = _pdf_cache.get(key)
        if hit and now - hit[0] < _CACHE_TTL_SEC:
            from agent_log import log_line

            side = "top" if key[1] else "bottom"
            log_line(
                "MinIO",
                result=f"【PDF缓存命中】项目={key[0]} 面={side} 未从MinIO拉取",
            )
            return hit[1], hit[2], hit[3]
    content, name, scale = loader()
    with _lock:
        _pdf_cache[key] = (now, content, name, float(scale or 0))
        _evict_if_needed(_pdf_cache)
    return content, name, float(scale or 0)


def get_affine(
    project_code: str,
    top_side: bool,
    compute,
) -> tuple[np.ndarray | None, str, dict | None]:
    """compute() -> (matrix, note, row_meta_without_blob)。"""

    key = (str(project_code or "").strip(), bool(top_side))
    now = time.time()
    with _lock:
        hit = _align_cache.get(key)
        if hit and now - hit[0] < _CACHE_TTL_SEC:
            return hit[1], hit[2], hit[3]
    matrix, note, row_meta = compute()
    with _lock:
        _align_cache[key] = (now, matrix, note, row_meta)
        _evict_if_needed(_align_cache)
    return matrix, note, row_meta


def invalidate_project(project_code: str) -> None:
    code = str(project_code or "").strip()
    with _lock:
        for key in list(_pdf_cache.keys()):
            if key[0] == code:
                _pdf_cache.pop(key, None)
        for key in list(_align_cache.keys()):
            if key[0] == code:
                _align_cache.pop(key, None)
