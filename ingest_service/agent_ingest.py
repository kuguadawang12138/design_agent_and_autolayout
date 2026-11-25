# -*- coding: utf-8 -*-
"""入库编排：二开 JSON+PDF 回调 → 校验 → 按唯一键有则删后插、无则插 → 形态 fpgeom + PDF。"""

from __future__ import annotations

import json

from agent_log import log_line
from agent_pdf import PdfScaleError, pdf_file_top_side, store_board_pdf
from footprint_db import (
    ERROR_CODES,
    connect,
    fetch_component_row,
    fetch_project_profile,
    normalize_refdes,
    pending_project_status,
    pin_count_too_small,
    save_pending_component_json,
    record_failure,
    upsert_footprint_from_content,
    upsert_project_profile,
)
from footprint_json import decode_design_json_bytes, loads_design_json
from json_fields import extract_meta, validate_component_json
from oa_project_meta import consume_project_meta
from oa_process_log import persist_parse_status
from settings import get_agent_paths
from top_side_util import parse_top_side, top_side_label


def validate_json_payload(payload: dict, filename: str) -> tuple[bool, str, dict | None]:
    """入库前字段校验；形态阶段 pin 无法还原铜箔时该位号仍不入库。"""
    valid, reason, meta = validate_component_json(payload, filename)
    if not valid:
        return False, reason, meta
    if not meta.get("project_code"):
        return False, "缺少 project_code", None
    return True, "", meta


def stage_brd(
    project_code: str,
    content: bytes,
    filename: str | None = None,
    *,
    content_type: str | None = None,
) -> dict:
    """暂存 BRD 到 data/brd/{project_code}.brd；支持 OA 返回的 zip 及多层嵌套压缩包。"""
    from oa_brd_extract import stage_brd_bytes_to_dir

    paths = get_agent_paths()
    brd_dir = paths["brd"]
    code = str(project_code or "").strip()
    if not code:
        raise ValueError("project_code 不能为空")
    target = stage_brd_bytes_to_dir(
        code,
        content,
        brd_dir,
        filename,
        content_type=content_type,
    )
    from oa_brd_disk import disk_usage_ratio_for_path

    ratio = disk_usage_ratio_for_path(brd_dir)
    log_line(
        "OA流程",
        result=(
            f"【BRD 落盘】project_code={code} 文件={target.name} "
            f"大小={target.stat().st_size}B BRD目录磁盘占用={ratio * 100:.1f}%"
        ),
    )
    return {"project_code": code, "brd_path": str(target), "file_name": target.name}


def serve_next_brd() -> dict | None:
    """二开拉 BRD：仅 data/brd 根目录下 *.brd 文件。"""
    brd_dir = get_agent_paths()["brd"]
    brd_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(brd_dir.glob("*.brd"), key=lambda p: p.stat().st_mtime)
    if not files:
        return None
    path = files[0]
    content = path.read_bytes()
    project_code = path.stem
    path.unlink(missing_ok=True)
    log_line("二开接口", result=f"【拉取 BRD】已下发并删除本地文件 project_code={project_code} 文件={path.name}")
    return {"project_code": project_code, "file_name": path.name, "content": content, "size": len(content)}


def report_ingest_failure(
    *,
    project_code: str | None,
    message: str,
    refdes: str | None = None,
) -> dict:
    """二开解析失败上报 → ingest_failure（唯一键 project_code + refdes）。"""
    code = str(project_code or "").strip()
    if not code:
        raise ValueError("project_code 不能为空")
    detail = str(message or "").strip()[:4000]
    conn = connect()
    try:
        record_failure(
            conn,
            error_code=ERROR_CODES["SCRIPT"],
            detail=detail[:2000],
            project_code=code,
            refdes=refdes or "",
        )
    finally:
        conn.close()
    try:
        persist_parse_status(code, ok=False, note=detail[:512])
    except Exception as exc:
        log_line(
            "二开入库",
            result=f"【process 表未更新】project_code={code} 原因={exc}",
            level="warning",
        )
    log_line(
        "二开入库",
        result=f"【BRD 流程】项目编码={code} | 二开失败上报：{detail[:200]}",
        level="warning",
    )
    return {"ok": True}


def _pdf_warn_ingest_failure(warn: str) -> tuple[bool, str | None]:
    """
    PDF 告警是否写入 ingest_failure。
    返回 (是否写库, error_code)。校准点不足等由定时任务补刷，不写失败表。
    """
    text = str(warn or "")
    if not text:
        return False, None
    if "校准点不足" in text or "未完成仿射校准" in text or "PDF 已入库但未完成仿射校准" in text:
        return False, None
    if "未找到：" in text or "未匹配到 Eda_ 位号" in text:
        return False, None
    if "位号坐标索引 fail" in text and "校准" not in text:
        return False, None
    if "单器件校准" in text or "单点校准" in text or ("仅" in text and "个器件" in text):
        return True, ERROR_CODES["WARN_PDF"]
    return True, ERROR_CODES["PDF"]


def _failure_detail_db(filename: str, reason: str, *, refdes: str = "") -> str:
    """写入 ingest_failure 的简短摘要（不含原始 JSON，详情见日志）。"""
    parts = [f"file={filename}"]
    if refdes:
        parts.append(f"refdes={refdes}")
    parts.append(f"reason={str(reason or '').strip()}")
    return " | ".join(parts)


def _log_ingest_failure(
    *,
    project_code: str,
    filename: str,
    reason: str,
    refdes: str = "",
) -> None:
    """完整失败原因打到结构化日志，便于排查 shapes 等问题。"""
    log_line(
        "二开入库",
        result=(
            f"【入库失败详情】project_code={project_code} refdes={refdes or '-'} "
            f"file={filename} 原因={str(reason or '').strip()[:500]}"
        ),
        level="warning",
    )


def ingest_json_pdf_batch(
    *,
    project_code: str | None,
    json_blobs: list[tuple[str, bytes]],
    pdf_blobs: list[tuple[str, bytes]] | None = None,
    project_info: dict | None = None,
) -> dict:
    """二开入库：按各表唯一键逐条写入（已存在则删后插，不存在则直接插入）。"""
    conn = connect()
    ok_count = 0
    fail_count = 0
    skip_count = 0
    pdf_count = 0
    errors = []
    qualified: list[dict] = []
    code = str(project_code or "").strip()
    response: dict = {"ok": True}
    parsed_items: list[tuple[str, bytes, dict]] = []
    status_cache: dict[str, int | None] = {}
    try:
        def _pending_status(project: str) -> int | None:
            key = str(project or "").strip()
            if not key:
                return None
            if key not in status_cache:
                try:
                    status_cache[key] = pending_project_status(conn, key)
                except Exception as exc:
                    log_line(
                        "二开入库",
                        result=(
                            f"【待完成项目表不可用】project_code={key} {type(exc).__name__}，按已完成板继续"
                        ),
                        level="warning",
                    )
                    status_cache[key] = None
            return status_cache[key]

        for filename, content in json_blobs:
            try:
                text = decode_design_json_bytes(content)
                payload, how = loads_design_json(text)
                if how == "escape_repair":
                    log_line(
                        "二开入库",
                        result=f"【JSON 转义补救】file={filename} 原文解析失败后才处理转义",
                        level="warning",
                    )
            except Exception as exc:
                err = str(exc)
                item_code = code or ""
                _log_ingest_failure(project_code=item_code or "?", filename=filename, reason=err)
                if item_code and _pending_status(item_code) != 0:
                    record_failure(
                        conn,
                        error_code=ERROR_CODES["JSON_FORMAT"],
                        detail=_failure_detail_db(filename, err),
                        project_code=item_code,
                        refdes="",
                    )
                elif not code:
                    log_line(
                        "二开入库",
                        result="【入库失败】JSON 无法解析且未提供 project_code，未写入 ingest_failure（表单请传 project_code）",
                        level="warning",
                    )
                fail_count += 1
                errors.append({"file": filename, "error": err})
                continue
            valid, reason, meta = validate_json_payload(payload, filename)
            if not valid:
                item_code = code or (meta or {}).get("project_code") or ""
                ref = (meta or {}).get("refdes") or ""
                if pin_count_too_small((meta or {}).get("pin_count")):
                    skip_count += 1
                    log_line(
                        "二开入库",
                        result=(
                            f"【脚数不超过6不入库】project_code={item_code or '-'} refdes={ref or '-'} "
                            f"脚数={(meta or {}).get('pin_count')} 文件={filename}"
                        ),
                    )
                    continue
                _log_ingest_failure(project_code=item_code or "?", filename=filename, reason=reason, refdes=ref)
                if item_code and _pending_status(item_code) != 0:
                    record_failure(
                        conn,
                        error_code=ERROR_CODES["JSON_FIELD"],
                        detail=_failure_detail_db(filename, reason, refdes=ref),
                        project_code=item_code,
                        refdes=ref,
                    )
                fail_count += 1
                errors.append({"file": filename, "error": reason})
                continue
            if not code:
                code = meta["project_code"]
            parsed_items.append((filename, content, meta))

        if _pending_status(code) == 0:
            for filename, content, meta in parsed_items:
                ref = normalize_refdes(meta.get("refdes"))
                pins = int(meta.get("pin_count") or 0)
                try:
                    saved = save_pending_component_json(
                        conn,
                        project_code=code,
                        refdes=ref,
                        pin_count=pins,
                        json_text=decode_design_json_bytes(content),
                        source_file=filename,
                    )
                    if not saved:
                        skip_count += 1
                        continue
                    ok_count += 1
                    log_line(
                        "二开入库",
                        result=(
                            f"【待完成JSON】project_code={code} refdes={ref} 脚数={pins} "
                            f"写入表=pending_json 文件={filename}"
                        ),
                    )
                except Exception as exc:
                    fail_count += 1
                    errors.append({"file": filename, "refdes": ref, "error": str(exc)})
                    log_line(
                        "二开入库",
                        result=(
                            f"【待完成JSON失败】project_code={code} refdes={ref} "
                            f"{type(exc).__name__}: {exc}"
                        ),
                        level="warning",
                    )
            log_line(
                "二开入库",
                result=(
                    f"【待完成状态】project_code={code} 状态=0，不入器件表，不处理 PDF，"
                    f"JSON 写入 pending_json。成功={ok_count} 失败={fail_count} 跳过={skip_count}。"
                    f"脚数不超过 6 的不入库，其余由 pending_iou_worker.py 计算后删除"
                ),
            )
            response.update(
                {
                    "project_code": code,
                    "ok_count": ok_count,
                    "fail_count": fail_count,
                    "skip_count": skip_count,
                    "pdf_count": 0,
                    "errors": errors,
                    "pending_status": 0,
                }
            )
            if ok_count == 0 and fail_count == 0 and skip_count > 0:
                response["ok"] = True
                response["message"] = f"脚数不超过 6 的 {skip_count} 条未入库"
            elif ok_count == 0:
                response["ok"] = False
                response["message"] = errors[0]["error"] if errors else "待完成项目没有可保存的器件 JSON"
            elif fail_count > 0:
                response["ok"] = True
                response["message"] = f"待完成 JSON 已写入，部分失败 {fail_count} 条"
            else:
                response["ok"] = True
                response["message"] = "待完成项目状态为 0，JSON 已写入 pending_json，未入器件表"
            return response

        if _pending_status(code) == -1:
            log_line(
                "二开入库",
                result=f"【待完成状态】project_code={code} 状态=-1，按已完成板进入器件表",
            )

        oa_meta: dict = {}
        if code:
            oa_meta = fetch_project_profile(conn, code) or {}
            if oa_meta:
                log_line(
                    "二开入库",
                    result=f"【合并 OA 属性】project_code={code} 数据={json.dumps(oa_meta, ensure_ascii=False, default=str)[:200]}",
                )

        if code and parsed_items and oa_meta:
            upsert_project_profile(conn, code, oa_meta)

        for filename, content, meta in parsed_items:
            merged_info = {
                **oa_meta,
                **(project_info or {}),
                "project_code": code,
                "board_name": meta.get("board_name"),
            }
            result = upsert_footprint_from_content(conn, content, source_file=filename, project_info=merged_info)
            if result.get("skipped"):
                skip_count += 1
                continue
            if not result.get("success"):
                msg = result.get("message") or "形态入库失败"
                ref = meta.get("refdes") or ""
                _log_ingest_failure(project_code=code, filename=filename, reason=msg, refdes=ref)
                record_failure(
                    conn,
                    error_code=ERROR_CODES["FPGEOM"],
                    detail=_failure_detail_db(filename, msg, refdes=ref),
                    project_code=code,
                    refdes=ref,
                )
                fail_count += 1
                errors.append({"file": filename, "refdes": ref, "error": msg})
                continue
            ok_count += 1
            row = fetch_component_row(conn, code, meta["refdes"]) or {}
            qualified.append({**meta, "bbox": row})
        pdf_list = [(n, c) for n, c in (pdf_blobs or []) if c and n.lower().endswith(".pdf")]
        if not pdf_list:
            log_line(
                "二开入库",
                result=f"【PDF】project_code={code or '-'} 本批没有 PDF，器件入库不依赖 PDF",
            )
        for filename, content in pdf_list:
            side = pdf_file_top_side(filename)
            if side is None:
                only_sides = {parse_top_side(q.get("top_side")) for q in qualified}
                if len(only_sides) == 1:
                    side = next(iter(only_sides))
                    log_line(
                        "二开入库",
                        result=(
                            f"【PDF 面别】file={filename} 文件名未标明正反面，"
                            f"本批器件均为 {top_side_label(side)}，按该面校准"
                        ),
                    )
                else:
                    reason = "PDF 文件名无法区分 top/bottom，且本批器件含两面，拒绝拿全部器件去两份 PDF 上搜索"
                    _log_ingest_failure(project_code=code or "?", filename=filename, reason=reason)
                    if code:
                        record_failure(
                            conn,
                            error_code=ERROR_CODES["PDF"],
                            detail=_failure_detail_db(filename, reason),
                            project_code=code,
                            refdes="",
                        )
                    fail_count += 1
                    errors.append({"file": filename, "error": reason})
                    continue
            side_items = [q for q in qualified if parse_top_side(q.get("top_side")) == side]
            sample = side_items[0] if side_items else None
            log_line(
                "二开入库",
                result=(
                    f"【PDF 校准】file={filename} 面={top_side_label(side)} "
                    f"仅使用该面器件 {len(side_items)} 个"
                ),
            )
            if not code:
                reason = "无 project_code，无法绑定 PDF"
                _log_ingest_failure(project_code="UNKNOWN", filename=filename, reason=reason)
                record_failure(
                    conn,
                    error_code=ERROR_CODES["PDF"],
                    detail=_failure_detail_db(filename, reason),
                    project_code="UNKNOWN",
                    refdes="",
                )
                response["message"] = "部分 PDF 未能绑定：缺少 project_code"
                continue
            try:
                pdf_warn = store_board_pdf(
                    conn,
                    project_code=code,
                    top_side=side,
                    file_name=filename,
                    content=content,
                    calibration_items=side_items,
                    sample_item=sample,
                    finish_time=(oa_meta or {}).get("finish_time"),
                )
                pdf_count += 1
                if pdf_warn:
                    write_fail, err_code = _pdf_warn_ingest_failure(pdf_warn)
                    if write_fail and err_code:
                        fail_count += 1
                        errors.append({"file": filename, "error": pdf_warn, "kind": "pdf_warn"})
                        record_failure(
                            conn,
                            error_code=err_code,
                            detail=_failure_detail_db(filename, pdf_warn),
                            project_code=code,
                            refdes="",
                        )
                        _log_ingest_failure(project_code=code, filename=filename, reason=pdf_warn)
                    else:
                        log_line(
                            "二开入库",
                            result=f"【PDF 告警不写库】project={code} file={filename} {pdf_warn}",
                            level="warning",
                        )
                    if "message" not in response:
                        response["message"] = pdf_warn
            except PdfScaleError as exc:
                err = str(exc)
                _log_ingest_failure(project_code=code, filename=filename, reason=err)
                record_failure(
                    conn,
                    error_code=ERROR_CODES["PDF"],
                    detail=_failure_detail_db(filename, err),
                    project_code=code,
                    refdes="",
                )
                if "message" not in response:
                    response["message"] = f"PDF 比例校准失败：{exc}"
            except Exception as exc:
                err = str(exc)
                from mysql_diag import log_mysql_exception

                log_mysql_exception(
                    conn,
                    exc,
                    operation="ingest_pdf",
                    detail=f"project_code={code} file={filename}",
                    sql_hint="store_board_pdf / project_pdf",
                )
                _log_ingest_failure(project_code=code, filename=filename, reason=err)
                record_failure(
                    conn,
                    error_code=ERROR_CODES["PDF"],
                    detail=_failure_detail_db(filename, err),
                    project_code=code,
                    refdes="",
                )
                if "message" not in response:
                    response["message"] = f"PDF 入库失败：{exc}"
        if code:
            if fail_count == 0 and (ok_count > 0 or skip_count > 0):
                persist_parse_status(
                    code,
                    ok=True,
                    note=f"二开入库成功，器件 {ok_count} 条，跳过 {skip_count} 条，PDF {pdf_count} 份",
                )
                log_line(
                    "二开入库",
                    result=(
                        f"【BRD 流程】项目编码={code} | 二开入库成功："
                        f"器件 {ok_count} 条，跳过 {skip_count} 条，PDF {pdf_count} 份"
                    ),
                )
            elif fail_count > 0:
                err0 = errors[0].get("error") if errors else "部分或全部入库失败"
                persist_parse_status(code, ok=False, note=str(err0)[:512])
                log_line(
                    "二开入库",
                    result=f"【BRD 流程】项目编码={code} | 二开入库失败：{str(err0)[:200]}",
                    level="warning",
                )
    finally:
        conn.close()

    response.update(
        {
            "project_code": code,
            "ok_count": ok_count,
            "fail_count": fail_count,
            "skip_count": skip_count,
            "pdf_count": pdf_count,
            "errors": errors,
        }
    )
    if ok_count == 0 and fail_count == 0 and skip_count > 0:
        response["ok"] = True
        response["message"] = f"脚数不超过 6 的 {skip_count} 条未入库"
    elif ok_count == 0:
        response["ok"] = False
        response["message"] = errors[0]["error"] if errors else "入库全部失败"
    elif fail_count > 0:
        response["ok"] = True
        response["message"] = f"部分失败 {fail_count} 条，详见 errors"
    else:
        response["ok"] = True
    if code and ok_count > 0:
        consume_project_meta(code)
    return response
