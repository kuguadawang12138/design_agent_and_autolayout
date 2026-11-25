# -*- coding: utf-8 -*-
"""BRD 落盘目录所在磁盘占用检测（OA 历史回补节流）。"""

from __future__ import annotations

import shutil
from pathlib import Path


def disk_usage_ratio_for_path(path: Path | str) -> float:
    """返回 path 所在卷已用空间占比 0~1。"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(str(p.resolve()))
    total = float(usage.total)
    if total <= 0:
        return 0.0
    return float(usage.used) / total
