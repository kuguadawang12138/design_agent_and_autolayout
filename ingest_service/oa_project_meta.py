# -*- coding: utf-8 -*-
"""
OA 项目四字段：直接写入 MySQL project_profile（不再落盘 JSON）。

字段：customer_code、designer_name、designer_seniority、finish_time
来源：OA project_info（Customer_Code / Design_Engineer / experience / Completion_time）
"""

from __future__ import annotations

from typing import Any

from agent_log import log_line
from footprint_db import connect, fetch_project_profile, upsert_project_profile
from json_fields import _as_float


def _experience_text_to_score(text: str) -> float | None:
    """OA experience 文本 → 资历分（与 LLM 排序字段一致，0~1）。"""
    t = str(text or "").strip().lower()
    if not t:
        return None
    mapping = {
        "专家": 0.95,
        "资深": 0.9,
        "高级": 0.8,
        "中高": 0.7,
        "中级": 0.55,
        "初级": 0.35,
        "入门": 0.25,
    }
    for key, score in mapping.items():
        if key in t:
            return score
    return _as_float(text)


def normalize_oa_project_info(raw: dict | None) -> dict[str, Any]:
    """将 OA project_info 转为入库用 project_info。"""
    raw = raw if isinstance(raw, dict) else {}

    def pick(*keys: str):
        for key in keys:
            for variant in (key, key.lower(), key.upper()):
                val = raw.get(variant)
                if val not in (None, ""):
                    return val
        return None

    exp = pick("experience", "Experience", "工程师资历")
    seniority = _as_float(exp)
    if seniority is None:
        seniority = _experience_text_to_score(str(exp or ""))

    finish = pick("Completion_time", "completion_time", "finish_time", "complete_time")
    finish_str = str(finish).strip() if finish not in (None, "") else None
    if finish_str:
        finish_str = finish_str.replace("/", "-")[:19]

    return {
        "customer_code": pick("Customer_Code", "customer_code", "customerCode"),
        "designer_name": pick("Design_Engineer", "designer_name", "designer"),
        "designer_seniority": seniority,
        "finish_time": finish_str,
    }


def save_project_meta(project_code: str, oa_info: dict | None) -> None:
    """OA 拉取项目信息后写入 project_profile。"""
    normalized = normalize_oa_project_info(oa_info)
    if not any(normalized.values()):
        log_line(
            "oa_meta_skip",
            detail=f"project={project_code}",
            result="OA 项目信息无四字段，未写库",
            level="warning",
        )
        return
    conn = connect()
    try:
        upsert_project_profile(conn, project_code, normalized)
        log_line("oa_meta_save", detail=f"project={project_code}", result="project_profile")
    finally:
        conn.close()


def load_project_meta(project_code: str) -> dict | None:
    """从 project_profile 读取（兼容旧 ingest 调用名）。"""
    conn = connect()
    try:
        return fetch_project_profile(conn, project_code)
    finally:
        conn.close()


def delete_project_meta(project_code: str) -> bool:
    """项目 profile 按唯一键删后插维护；此处保留空实现兼容旧调用。"""
    return False


def consume_project_meta(project_code: str) -> dict | None:
    """历史接口：数据已在库内，不再读删 JSON 文件。"""
    return load_project_meta(project_code)
