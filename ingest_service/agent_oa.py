# -*- coding: utf-8 -*-
"""OA 三接口客户端（泛微 E8）：项目编码 / 项目信息 / BRD 下载。"""

from __future__ import annotations

from urllib.parse import parse_qs, urlparse

import httpx

from agent_ingest import stage_brd
from agent_log import log_line
from footprint_db import ERROR_CODES, connect, record_failure
from oa_process_log import (
    OaProjectFlow,
    audit_project_detail_fields,
    log_oa_interface1,
    log_oa_project_flow,
    persist_oa_flow,
)
from oa_project_meta import save_project_meta
from settings import get_oa_config


class OaDownloadParamsEmpty(Exception):
    """接口②无有效 download_path / docId+fileId，不应调用接口③。"""


def _normalize_date(value: str) -> str:
    """OA 接受 yyyy-MM-dd 或 yyyy/MM/dd。"""
    text = str(value or "").strip()
    return text.replace("/", "-")


def _envelope_records(data) -> list[dict]:
    if not isinstance(data, dict):
        return []
    if data.get("code") not in (None, 0, "0") and "data" not in data:
        raise RuntimeError(f"OA 返回错误：{data.get('msg') or data}")
    payload = data.get("data") if "data" in data else data
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        return [payload]
    return []


def _oa_http_error_message(resp: httpx.Response, *, api: str) -> str:
    """按 ThirdResource 文档将 HTTP/OA code 转为中文说明。"""
    try:
        data = resp.json()
        if isinstance(data, dict):
            msg = str(data.get("msg") or data.get("message") or "").strip()
            code = data.get("code")
            if msg:
                return f"{api}失败（HTTP {resp.status_code}，code={code}）：{msg}"
    except Exception:
        pass
    text = (resp.text or "").strip()[:200]
    return f"{api}失败（HTTP {resp.status_code}）{text}"


def _raise_oa_network_error(exc: Exception, *, api: str) -> None:
    """连接失败/超时等转为中文说明（系统维护、网络不通）。"""
    if isinstance(exc, httpx.ConnectError):
        raise RuntimeError(
            f"{api}无法连接 OA 服务器（连接被拒绝或网络不通，可能系统维护中）"
        ) from exc
    if isinstance(exc, httpx.TimeoutException):
        raise RuntimeError(f"{api}请求超时（超过 {get_oa_config()['timeout_sec']} 秒，OA 可能维护或过载）") from exc
    if isinstance(exc, httpx.HTTPError):
        raise RuntimeError(f"{api}网络异常：{exc}") from exc
    raise exc


def _get_raw(path: str, params: dict | None = None):
    cfg = get_oa_config()
    if not cfg["enabled"]:
        raise RuntimeError("OA 未启用，请在 config/footprint_sim.yaml 中设置 oa.enabled=true")
    url = cfg["base"] + path
    try:
        resp = httpx.get(url, params=params or {}, timeout=cfg["timeout_sec"])
    except Exception as exc:
        _raise_oa_network_error(exc, api="OA")
    resp.raise_for_status()
    return resp.json()


def parse_download_path(download_path: str | None) -> dict[str, str] | None:
    """从 download_path 解析 docId / fileId。"""
    text = str(download_path or "").strip()
    if not text:
        return None
    if text.startswith("?"):
        text = "http://oa.local/" + text
    parsed = urlparse(text)
    query = parse_qs(parsed.query)
    out: dict[str, str] = {}
    for key in ("docId", "docid", "doc_id"):
        vals = query.get(key)
        if vals and vals[0]:
            out["docId"] = vals[0]
            break
    for key in ("fileId", "fileid", "file_id"):
        vals = query.get(key)
        if vals and vals[0]:
            out["fileId"] = vals[0]
            break
    if out.get("docId") and out.get("fileId"):
        return out
    return None


def _pick_download_path(info: dict) -> str | None:
    """从接口②记录中取 download_path（兼容大小写/别名及嵌套字段）。"""

    def _looks_like_download(text: str) -> bool:
        low = text.lower()
        return "docid" in low and "fileid" in low

    for key in (
        "download_path",
        "Download_Path",
        "downloadPath",
        "download_url",
        "downloadUrl",
    ):
        val = info.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()

    stack: list = [info]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            stack.extend(cur.values())
        elif isinstance(cur, list):
            stack.extend(cur)
        elif isinstance(cur, str) and _looks_like_download(cur):
            return cur.strip()
    return None


def _raw_download_path_field(info: dict) -> str | None:
    for key in ("download_path", "Download_Path", "downloadPath"):
        if key in info:
            val = info.get(key)
            return "" if val is None else str(val)
    return None


def inspect_download_params(info: dict) -> tuple[bool, str, str | None, str | None]:
    """检查 download_path / docId / fileId。"""
    raw = _raw_download_path_field(info)
    if raw is not None and raw.strip() == "":
        return False, "download_path 为空", None, None
    path = _pick_download_path(info)
    parsed = parse_download_path(path)
    if parsed:
        d, f = parsed["docId"], parsed["fileId"]
        return True, f"docId={d} fileId={f}", d, f
    doc = info.get("docId") or info.get("docid") or info.get("doc_id")
    fid = info.get("fileId") or info.get("fileid") or info.get("file_id")
    if doc and fid:
        d, f = str(doc).strip(), str(fid).strip()
        return True, f"docId={d} fileId={f}", d, f
    if raw is not None:
        return False, "download_path 无法解析 docId 与 fileId", None, None
    return False, "缺少 download_path 与 docId/fileId", None, None


def ensure_download_params(info: dict, project_code: str) -> None:
    """download_path 为空或无法解析 docId/fileId 时抛出 OaDownloadParamsEmpty。"""
    ok, note, _, _ = inspect_download_params(info)
    if not ok:
        raise OaDownloadParamsEmpty(f"项目 {project_code} 下载参数为空（{note}）")


def resolve_brd_doc_ids(info: dict, project_code: str) -> tuple[str, str]:
    """接口③必填 docId + fileId；调用前应先 ensure_download_params。"""
    ensure_download_params(info, project_code)
    _, _, doc, fid = inspect_download_params(info)
    return str(doc).strip(), str(fid).strip()


def _extract_codes(data) -> list[str]:
    codes: list[str] = []
    for record in _envelope_records(data):
        value = record.get("project_code") or record.get("code")
        if value:
            codes.append(str(value).strip())
    if not codes and isinstance(data, list):
        for item in data:
            if isinstance(item, str) and item.strip():
                codes.append(item.strip())
            elif isinstance(item, dict):
                value = item.get("project_code") or item.get("code")
                if value:
                    codes.append(str(value).strip())
    seen: set[str] = set()
    ordered: list[str] = []
    for code in codes:
        if code and code not in seen:
            seen.add(code)
            ordered.append(code)
    return ordered


def fetch_project_codes(
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[str]:
    """接口①：按完成时间区间查 project_code 列表。"""
    cfg = get_oa_config()
    params: dict[str, str] = {}
    if start_date:
        params["start_date"] = _normalize_date(start_date)
    if end_date:
        params["end_date"] = _normalize_date(end_date)
    start_s = params.get("start_date", "-")
    end_s = params.get("end_date", "-")
    url = cfg["base"] + cfg["project_code_path"]
    try:
        resp = httpx.get(url, params=params or {}, timeout=cfg["timeout_sec"])
        if resp.status_code == 400:
            log_oa_interface1(
                start_date=start_s,
                end_date=end_s,
                ok=False,
                count=0,
                note="日期为空或格式非法，或开始日期不早于结束日期",
            )
            raise RuntimeError("接口①失败：start_date/end_date 必填且开始须早于结束")
        if resp.status_code == 403:
            log_oa_interface1(start_date=start_s, end_date=end_s, ok=False, count=0, note="IP 不在白名单")
            raise RuntimeError("接口①失败：IP 不在白名单")
        if resp.status_code == 404:
            log_oa_interface1(
                start_date=start_s,
                end_date=end_s,
                ok=True,
                count=0,
                note="该日期区间无项目数据",
            )
            return []
        if resp.status_code >= 500:
            msg = _oa_http_error_message(resp, api="接口①")
            log_oa_interface1(start_date=start_s, end_date=end_s, ok=False, count=0, note=msg)
            resp.raise_for_status()
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and str(data.get("code")) in ("404",):
            log_oa_interface1(
                start_date=start_s,
                end_date=end_s,
                ok=True,
                count=0,
                note=str(data.get("msg") or "该日期区间无项目数据"),
            )
            return []
        if isinstance(data, dict) and data.get("code") not in (None, 0, "0") and "data" not in data:
            msg = str(data.get("msg") or data)
            log_oa_interface1(start_date=start_s, end_date=end_s, ok=False, count=0, note=msg)
            raise RuntimeError(f"接口①失败：{msg}")
        codes = _extract_codes(data)
        log_oa_interface1(start_date=start_s, end_date=end_s, ok=True, count=len(codes))
        return codes
    except httpx.ConnectError as exc:
        msg = "接口①无法连接 OA（连接被拒绝或网络不通，可能系统维护中）"
        log_oa_interface1(start_date=start_s, end_date=end_s, ok=False, count=0, note=msg)
        raise RuntimeError(msg) from exc
    except httpx.TimeoutException as exc:
        msg = f"接口①请求超时（超过 {cfg['timeout_sec']} 秒，OA 可能维护或过载）"
        log_oa_interface1(start_date=start_s, end_date=end_s, ok=False, count=0, note=msg)
        raise RuntimeError(msg) from exc
    except httpx.HTTPError as exc:
        log_oa_interface1(start_date=start_s, end_date=end_s, ok=False, count=0, note=str(exc))
        raise RuntimeError(f"接口①网络异常：{exc}") from exc


def fetch_project_info(project_code: str) -> dict:
    """接口②：项目信息（含 download_path）。"""
    cfg = get_oa_config()
    url = cfg["base"] + cfg["project_info_path"]
    try:
        resp = httpx.get(url, params={"project_code": project_code}, timeout=cfg["timeout_sec"])
    except Exception as exc:
        _raise_oa_network_error(exc, api="接口②")
    if resp.status_code == 400:
        raise RuntimeError("接口②失败：project_code 不能为空")
    if resp.status_code == 403:
        raise RuntimeError("接口②失败：IP 不在白名单")
    if resp.status_code >= 500:
        raise RuntimeError(_oa_http_error_message(resp, api="接口②"))
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict) and data.get("code") not in (None, 0, "0") and not _envelope_records(data):
        raise RuntimeError(f"接口②失败：{data.get('msg') or data}")
    records = _envelope_records(data)
    info: dict = {"project_code": project_code}
    if records:
        info.update(records[0])
    elif isinstance(data, dict):
        info.update(data)
    info["project_code"] = project_code
    return info


def download_brd(
    project_code: str,
    doc_id: str | None = None,
    file_id: str | None = None,
    *,
    record_failures: bool = True,
) -> dict:
    """接口③：下载 BRD 并暂存到 data/brd/ 供明工拉取。"""
    cfg = get_oa_config()
    if not doc_id or not file_id:
        raise ValueError("缺少 docId 或 fileId，请检查接口② download_path")
    params = {"docId": str(doc_id).strip(), "fileId": str(file_id).strip()}
    url = cfg["base"] + cfg["brd_path"]
    try:
        resp = httpx.get(url, params=params, timeout=cfg["timeout_sec"])
    except Exception as exc:
        _raise_oa_network_error(exc, api="接口③")
    if resp.status_code == 400:
        raise ValueError(f"接口③失败：docId/fileId 无效（{_oa_http_error_message(resp, api='接口③')}）")
    if resp.status_code == 403:
        raise ValueError(f"接口③失败：{_oa_http_error_message(resp, api='接口③')}")
    if resp.status_code == 404:
        raise ValueError("接口③失败：附件不存在（file not found）")
    resp.raise_for_status()
    content = resp.content
    ctype = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
    if "application/json" in ctype:
        try:
            data = resp.json()
        except Exception as exc:
            raise ValueError(f"接口③返回 JSON 无法解析：{exc}") from exc
        msg = str(data.get("msg") if isinstance(data, dict) else data)[:200]
        raise ValueError(f"接口③未返回文件流：{msg}")
    if not content:
        if record_failures:
            conn = connect()
            try:
                record_failure(
                    conn,
                    error_code=ERROR_CODES["SCRIPT"],
                    detail=f"OA BRD 下载为空 project={project_code}",
                    project_code=project_code,
                    refdes="",
                )
            finally:
                conn.close()
        raise ValueError("接口③下载内容为空")
    cd = resp.headers.get("content-disposition") or ""
    filename = f"{project_code}.brd"
    if "filename*=" in cd.lower():
        part = cd.lower().split("filename*=")[-1].split("''")[-1].strip().strip('" ')
        if part:
            filename = part
    elif "filename=" in cd:
        filename = cd.split("filename=")[-1].strip('" ').split(";")[0].strip()
    try:
        manifest = stage_brd(project_code, content, filename, content_type=ctype)
    except Exception as exc:
        raise ValueError(f"解压 BRD 失败：{exc}") from exc
    # 已完成板解出 BRD 后，待完成记录若仍是 0，改成 -1。二开再回来才允许进器件表。
    try:
        from footprint_db import mark_pending_brd_extracted

        conn = connect()
        try:
            if mark_pending_brd_extracted(conn, project_code):
                log_line(
                    "OA流程",
                    result=f"【待完成状态】project_code={project_code} 已解出完成板 BRD，状态 0 改为 -1",
                )
        finally:
            conn.close()
    except Exception as exc:
        log_line(
            "OA流程",
            result=(
                f"【待完成状态未改】project_code={project_code} {type(exc).__name__}: {exc}"
            ),
            level="warning",
        )
    return manifest


def run_oa_pipeline(
    project_code: str,
    *,
    doc_id: str | None = None,
    file_id: str | None = None,
    save_meta: bool = True,
    record_failures: bool = True,
) -> dict:
    """串联 OA：可选写 project_profile + 按 download_path 下载 BRD 暂存。"""
    code = str(project_code or "").strip()
    flow = OaProjectFlow(project_code=code)
    info: dict | None = None
    try:
        info = fetch_project_info(code)
        flow.info_http_ok = True
    except Exception as exc:
        flow.info_http_note = str(exc).strip()
        log_oa_project_flow(flow)
        persist_oa_flow(flow, enabled=save_meta)
        raise

    ok4, missing = audit_project_detail_fields(info)
    flow.detail_fields_ok = ok4
    flow.detail_missing = missing

    dl_ok, dl_note, doc_parsed, fid_parsed = inspect_download_params(info)
    flow.download_ok = dl_ok
    flow.download_note = dl_note

    if save_meta and ok4:
        save_project_meta(code, info)

    if doc_id and file_id:
        doc, fid = str(doc_id).strip(), str(file_id).strip()
    elif dl_ok and doc_parsed and fid_parsed:
        doc, fid = doc_parsed, fid_parsed
    else:
        doc, fid = None, None

    if not dl_ok:
        log_oa_project_flow(flow)
        persist_oa_flow(flow, enabled=save_meta)
        raise OaDownloadParamsEmpty(f"项目 {code} 下载参数为空（{dl_note}）")

    try:
        brd = download_brd(
            code,
            doc_id=doc,
            file_id=fid,
            record_failures=record_failures,
        )
        flow.brd_ok = True
        flow.brd_note = brd.get("file_name") or f"{code}.brd"
        log_oa_project_flow(flow)
        persist_oa_flow(flow, enabled=save_meta)
        return {"project_code": code, "project_info": info, "brd": brd}
    except Exception as exc:
        flow.brd_ok = False
        flow.brd_note = str(exc).strip()
        log_oa_project_flow(flow)
        persist_oa_flow(flow, enabled=save_meta)
        raise
