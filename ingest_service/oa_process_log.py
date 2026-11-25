# -*- coding: utf-8 -*-
"""OA→BRD→明工 全流程：字段校验、中文单行日志、process 表入库。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_log import log_line
from footprint_db import connect, upsert_process_record

# 接口②文档四字段（ThirdResource 4.2）
OA_DETAIL_FIELDS: tuple[tuple[str, ...], str] = (
    (("Customer_Code", "customer_code", "customerCode"), "客户编码"),
    (("Design_Engineer", "design_engineer", "designer_name", "designer"), "设计工程师"),
    (("experience", "Experience", "工程师资历"), "工程师资历"),
    (("Completion_time", "completion_time", "finish_time", "complete_time"), "设计完成时间"),
)


def _pick_field(raw: dict, keys: tuple[str, ...]) -> Any:
    for key in keys:
        for variant in (key, key.lower(), key.upper()):
            val = raw.get(variant)
            if val not in (None, ""):
                return val
    return None


def audit_project_detail_fields(info: dict) -> tuple[bool, list[str]]:
    """检查接口②四字段是否均有值。"""
    missing: list[str] = []
    for keys, label in OA_DETAIL_FIELDS:
        if _pick_field(info, keys) in (None, ""):
            missing.append(label)
    return (len(missing) == 0, missing)


@dataclass
class OaProjectFlow:
    """单个项目 OA 拉取与解压状态（用于日志与 process 表）。"""

    project_code: str
    info_http_ok: bool = False
    info_http_note: str = ""
    detail_fields_ok: bool = False
    detail_missing: list[str] = field(default_factory=list)
    download_ok: bool = False
    download_note: str = ""
    brd_ok: bool = False
    brd_note: str = ""

    def detail_text(self) -> str:
        if not self.info_http_ok:
            return f"接口②请求失败：{self.info_http_note or '未知错误'}"
        if self.detail_fields_ok:
            return "接口②项目明细正常（客户编码/设计工程师/工程师资历/设计完成时间齐全）"
        miss = "、".join(self.detail_missing) if self.detail_missing else "未知"
        return f"接口②项目明细不完整，缺少：{miss}"

    def download_text(self) -> str:
        if not self.info_http_ok:
            return "接口②下载参数：未请求"
        if self.download_ok:
            return f"接口②下载参数正常（{self.download_note}）"
        return f"接口②下载参数异常：{self.download_note or '缺少 docId/fileId 或 download_path 为空'}"

    def brd_text(self) -> str:
        if self.brd_ok:
            return f"接口③解压 BRD 成功（{self.brd_note}）"
        if self.brd_note:
            return f"接口③解压 BRD 失败：{self.brd_note}"
        return "接口③解压 BRD：未执行"

    def one_line(self) -> str:
        code = self.project_code
        return (
            f"【BRD 流程】项目编码={code} | {self.detail_text()} | {self.download_text()} | {self.brd_text()}"
        )


def log_oa_interface1(*, start_date: str, end_date: str, ok: bool, count: int, note: str = "") -> None:
    """接口①批次日志（每个时间窗一行）。"""
    if ok:
        msg = (
            f"【接口①查询项目编码】开始={start_date} 结束={end_date}（结束日不含当天）| "
            f"请求成功 | 共 {count} 个项目编码"
        )
        if count == 0 and note:
            msg += f" | {note}"
    else:
        msg = (
            f"【接口①查询项目编码】开始={start_date} 结束={end_date}（结束日不含当天）| "
            f"请求失败 | {note or '未知原因'}"
        )
    log_line("OA流程", result=msg, level="info" if ok else "error")


def log_oa_project_flow(flow: OaProjectFlow) -> None:
    level = "info" if flow.brd_ok else "warning"
    log_line("OA流程", result=flow.one_line(), level=level)


def persist_oa_flow(flow: OaProjectFlow, *, enabled: bool = True) -> None:
    """写入 process 表（OA 阶段字段；明工 parse 字段由入库回调更新）。"""
    if not enabled:
        return
    code = str(flow.project_code or "").strip()
    if not code:
        return
    conn = connect()
    try:
        upsert_process_record(
            conn,
            project_code=code,
            oa_info_ok=1 if flow.info_http_ok and flow.detail_fields_ok else 0 if flow.info_http_ok else None,
            oa_info_note=_join_note(flow.info_http_note, flow.detail_missing),
            oa_download_ok=1 if flow.download_ok else 0 if flow.info_http_ok else None,
            oa_download_note=flow.download_note if flow.info_http_ok else flow.info_http_note,
            oa_brd_ok=1 if flow.brd_ok else 0 if flow.brd_note else None,
            oa_brd_note=flow.brd_note or None,
        )
    finally:
        conn.close()


def persist_parse_status(
    project_code: str,
    *,
    ok: bool,
    note: str = "",
) -> None:
    """明工 JSON/PDF 入库结果写入 process 表。"""
    code = str(project_code or "").strip()
    if not code:
        return
    conn = connect()
    try:
        upsert_process_record(
            conn,
            project_code=code,
            parse_ok=1 if ok else 0,
            parse_note=(note or "")[:512] or None,
        )
    finally:
        conn.close()


def _join_note(http_note: str, missing: list[str]) -> str | None:
    parts: list[str] = []
    if http_note:
        parts.append(http_note)
    if missing:
        parts.append("缺少：" + "、".join(missing))
    text = "；".join(parts).strip()
    return text[:512] if text else None
