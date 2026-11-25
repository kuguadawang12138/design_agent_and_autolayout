# -*- coding: utf-8 -*-
"""正反面 top_side：对外 JSON 用 boolean，库内 TINYINT(1)，URL 路径用 true/false。"""

from __future__ import annotations

from typing import Any


def parse_top_side(raw: Any, *, default: bool = True) -> bool:
    """解析明工/设计 JSON 等来源：true/1=正面，false/0=反面。"""
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return int(raw) != 0
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in ("1", "true", "top", "yes", "t", "front", "正面"):
            return True
        if s in ("0", "false", "bottom", "no", "b", "f", "back", "反面", "反"):
            return False
    return default


def top_side_from_db(raw: Any) -> bool:
    """读库：兼容 TINYINT、历史 VARCHAR yes/no。"""
    if raw is None:
        return True
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return int(raw) != 0
    return parse_top_side(str(raw))


def top_side_to_db(is_top: bool) -> int:
    return 1 if is_top else 0


def top_side_url_token(is_top: bool) -> str:
    return "true" if is_top else "false"


def parse_top_side_url_token(raw: str) -> bool:
    text = str(raw or "").strip()
    if not text:
        raise ValueError("top_side 不能为空")
    return parse_top_side(text, default=True)


def top_side_label(is_top: bool) -> str:
    """展示用：top / bottom。"""
    return "top" if is_top else "bottom"
