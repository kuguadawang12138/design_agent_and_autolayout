# -*- coding: utf-8 -*-
"""PDF HTTP 接口（兼容 DAY002 前端）：基于 project_pdf 单表，单页 PNG 预览 + 区域定位。"""

from __future__ import annotations

import re
import time
from functools import wraps
from pathlib import Path
from typing import Callable, TypeVar
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from agent_log import log_line
from agent_timing import StepTimer

F = TypeVar("F", bound=Callable)

from agent_pdf import (
    pdf_refdes_search_terms,
    render_highlighted_pdf,
    silk_page_index,
    silk_refdes_rect,
)
from agent_pdf import highlight_region_from_stored
from agent_pdf_cache import get_pdf_bytes
from agent_pdf_trace import PdfPerfTrace
from agent_pdf import component_highlight_download_name, component_highlight_urls, stamp_pdf_region_highlight
from footprint_db import connect, fetch_component_row, fetch_pdf
from top_side_util import (
    parse_top_side_url_token,
    top_side_from_db,
    top_side_label,
    top_side_url_token,
)

router = APIRouter(prefix="/api/pdf", tags=["pdf"])


def _log_pdf_route(route: str, detail: str = ""):
    """PDF 路由：单行耗时；handler 可设 response._perf_timer = StepTimer() 输出子步骤。"""

    def deco(fn: F) -> F:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            timer = StepTimer()
            t0 = time.perf_counter()
            try:
                out = fn(*args, **kwargs)
                sub = getattr(out, "_perf_timer", None)
                if isinstance(sub, StepTimer):
                    timer.merge(sub._steps)
                return out
            finally:
                wall = int((time.perf_counter() - t0) * 1000)
                log_line(
                    "PDF接口",
                    detail=f"路由={route} {detail}".strip(),
                    result=f"总耗时={wall}ms {timer.format_steps()}",
                )

        return wrapper  # type: ignore

    return deco

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    fitz = None


def _require_fitz():
    if fitz is None:
        raise RuntimeError("未安装 PyMuPDF，请执行: pip install pymupdf")


def _pdf_url_prefix(project_code: str, top_side: bool) -> str:
    """板级 PDF 路径：project_code + top_side（URL 段 true/false）。"""
    pc = quote(str(project_code or "").strip(), safe="")
    ts = top_side_url_token(top_side)
    return f"/api/pdf/{pc}/{ts}"


def _parse_path_top_side(raw: str) -> bool:
    try:
        return parse_top_side_url_token(raw)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


def _get_row(conn, project_code: str, top_side: bool) -> dict:
    is_top = bool(top_side)
    code = str(project_code or "").strip()

    def _load_blob() -> tuple[bytes, str, float]:
        row = fetch_pdf(conn, code, is_top)
        if not row:
            raise HTTPException(404, "PDF 不存在")
        return bytes(row["file_content"]), str(row["file_name"]), float(row.get("pdf_scale") or 0)

    content, file_name, pdf_scale = get_pdf_bytes(code, is_top, _load_blob)
    page_size = _first_page_size(content)
    return {
        "project_code": code,
        "top_side": is_top,
        "side": top_side_label(is_top),
        "file_name": file_name,
        "file_content": content,
        "pdf_scale": pdf_scale,
        "page": page_size,
    }


def _trace_minio(trace: PdfPerfTrace, acc) -> None:
    """把本段 MinIO 拉取写进 PDF 耗时行，并另打一行对象数/字节/最慢。"""
    parts: list[str] = []
    for purpose, step in (("pdf", "minio_pdf"), ("source_json", "minio_json")):
        sl = acc.slice(purpose)
        if sl.count <= 0:
            continue
        trace.add(step, sl.ms)
        parts.append(sl.format(purpose))
    if parts:
        log_line("MinIO", result="【拉取汇总】" + " ".join(parts))


def _load_row_traced(trace: PdfPerfTrace, conn, project_code: str, top_side: bool) -> dict:
    from minio_store import measure_fetches

    with measure_fetches() as acc:
        with trace.step("get_row_db_pdf"):
            row = _get_row(conn, project_code, top_side)
    _trace_minio(trace, acc)
    return row


def _first_page_size(content: bytes) -> dict | None:
    _require_fitz()
    doc = fitz.open(stream=content, filetype="pdf")
    try:
        if doc.page_count < 1:
            return None
        page = doc[silk_page_index(doc)]
        return {"width": float(page.rect.width), "height": float(page.rect.height)}
    finally:
        doc.close()


def _needles(query: str) -> list[str]:
    text = (query or "").strip()
    if not text:
        return []
    parts: list[str] = []
    upper = text.upper()
    if re.match(r"^[A-Z]{1,6}\d+[A-Z0-9]*$", upper):
        parts.extend(pdf_refdes_search_terms(upper))
    parts.append(text)
    for token in re.split(r"[\s,/|]+", text):
        token = token.strip()
        if len(token) >= 2 and token.lower() not in {p.lower() for p in parts}:
            parts.append(token)
            tu = token.upper()
            if re.match(r"^[A-Z]{1,6}\d+[A-Z0-9]*$", tu):
                for term in pdf_refdes_search_terms(tu):
                    if term not in parts:
                        parts.insert(0, term)
    out: list[str] = []
    seen: set[str] = set()
    for p in parts:
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out[:8]


def _snippet(text: str, query: str, radius: int = 90) -> str:
    raw = text or ""
    needle = (query or "").strip()
    if not needle:
        return raw[:160].replace("\n", " ")
    lower = raw.lower()
    idx = lower.find(needle.lower())
    if idx < 0:
        return raw[:160].replace("\n", " ")
    start = max(0, idx - radius)
    end = min(len(raw), idx + len(needle) + radius)
    piece = raw[start:end].replace("\n", " ")
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(raw) else ""
    return f"{prefix}{piece}{suffix}"


def _inflate_rect(region: dict, pad: float, bounds: tuple) -> dict:
    x0, y0, x1, y1 = region["x0"], region["y0"], region["x1"], region["y1"]
    bx0, by0, bx1, by1 = bounds
    return {
        "x0": max(bx0, x0 - pad),
        "y0": max(by0, y0 - pad),
        "x1": min(bx1, x1 + pad),
        "y1": min(by1, y1 + pad),
    }


def search_rects(content: bytes, query: str) -> list[dict]:
    """在带 Eda_ 位号的那一页上搜索。多页 PDF 的装配图可能在第 1 页。"""
    _require_fitz()
    doc = fitz.open(stream=content, filetype="pdf")
    try:
        if doc.page_count < 1:
            return []
        page = doc[silk_page_index(doc)]
        rects = []
        seen = set()
        for needle in _needles(query):
            for rect in page.search_for(needle) or []:
                key = (round(rect.x0, 1), round(rect.y0, 1), round(rect.x1, 1), round(rect.y1, 1))
                if key in seen:
                    continue
                seen.add(key)
                rects.append({"x0": rect.x0, "y0": rect.y0, "x1": rect.x1, "y1": rect.y1, "query": needle})
        return rects
    finally:
        doc.close()


def render_page_png(
    content: bytes,
    *,
    query: str | None = None,
    zoom: float | None = None,
    region: dict | None = None,
    regions: list | None = None,
    highlight: bool = True,
    clip: bool = False,
    view: dict | None = None,
) -> bytes:
    _require_fitz()
    doc = fitz.open(stream=content, filetype="pdf")
    try:
        page = doc[silk_page_index(doc)]
        boxes = [item for item in (regions or []) if item]
        if region and highlight and not boxes:
            stamp_pdf_region_highlight(page, region)
        elif boxes and highlight:
            for box in boxes:
                stamp_pdf_region_highlight(page, box)
        elif query:
            for needle in _needles(query):
                for rect in page.search_for(needle) or []:
                    annot = page.add_highlight_annot(rect)
                    annot.set_colors(stroke=(1, 0.85, 0.2))
                    annot.update()
        if zoom is None:
            zoom = 1.05 if page.rect.width > 900 else 1.6
        zoom = max(0.2, min(float(zoom), 24.0))
        clip_rect = None
        if view:
            clip_rect = fitz.Rect(view["x0"], view["y0"], view["x1"], view["y1"]) & page.rect
            if clip_rect.is_empty or clip_rect.width < 0.5 or clip_rect.height < 0.5:
                clip_rect = None
            else:
                longest = max(clip_rect.width, clip_rect.height, 1.0)
                zoom = min(zoom, 4096.0 / longest)
        elif clip and (region or boxes):
            clip_src = region if region and not boxes else _union_region(boxes or [region])
            padded = _inflate_rect(clip_src, 24, (page.rect.x0, page.rect.y0, page.rect.x1, page.rect.y1))
            clip_rect = fitz.Rect(padded["x0"], padded["y0"], padded["x1"], padded["y1"])
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip_rect, alpha=False)
        return pix.tobytes("png")
    finally:
        doc.close()


def pdf_public_dict(
    row: dict,
    *,
    highlight_q: str | None = None,
    region: dict | None = None,
    session_id: str | None = None,
) -> dict:
    code = row["project_code"]
    is_top = bool(row["top_side"])
    base = _pdf_url_prefix(code, is_top)
    q = quote(str(highlight_q or "").strip())
    qpart = f"&q={q}" if q else ""
    sid = quote(str(session_id or "").strip())
    sidpart = f"&sid={sid}" if sid else ""
    highlighted = f"{base}/file?highlight=1{qpart}{sidpart}" if q else f"{base}/file?highlight=1{sidpart}"
    # 整板查看走高亮 PDF。clip 只给列表上的小图，不再按视口重画。
    thumb = f"{base}/preview?highlight=1{qpart}&clip=1" if q else ""
    return {
        "project_code": code,
        "top_side": is_top,
        "file_name": row["file_name"],
        "source_file_name": row["file_name"],
        "file_url": highlighted,
        "preview_url": highlighted,
        "clip_url": thumb,
        "region_url": f"{base}/region",
        "highlight_file_url": highlighted,
        "component_ref": highlight_q,
        "side": row.get("side") or top_side_label(is_top),
        "title": row.get("file_name"),
        "highlight_region": region,
    }


def resolve_region(
    conn,
    row: dict,
    query: str | None = None,
    *,
    trace: PdfPerfTrace | None = None,
) -> dict:
    ref = str(query or "").strip().upper()
    content = row["file_content"]
    pdf_rect = None
    source = "none"
    comp = None
    page_info = row.get("page")
    scale = float(row.get("pdf_scale") or 0)
    if ref and row.get("project_code"):
        if trace:
            from minio_store import measure_fetches

            with measure_fetches() as acc:
                with trace.step("db_fetch_component"):
                    comp = fetch_component_row(conn, row["project_code"], ref)
            _trace_minio(trace, acc)
        else:
            comp = fetch_component_row(conn, row["project_code"], ref)
    if comp and ref:
        if trace:
            with trace.step("build_region_stored"):
                pdf_rect = highlight_region_from_stored(comp, scale)
        else:
            pdf_rect = highlight_region_from_stored(comp, scale)
        if pdf_rect:
            source = "stored"
    pub = pdf_public_dict(row, highlight_q=ref or None, region=pdf_rect)
    payload = {
        "project_code": row["project_code"],
        "top_side": row["top_side"],
        "page": page_info,
        "component_ref": ref or None,
        "side": row.get("side"),
        "source": source,
        "pdf_rect": pdf_rect,
        "preview_url": pub["preview_url"],
        "clip_url": pub["clip_url"],
    }
    if comp:
        bbox = _normalize_bbox_from_comp(comp)
        payload["pcb"] = {
            "x": comp.get("pcb_x"),
            "y": comp.get("pcb_y"),
            "w": (bbox["maxx"] - bbox["minx"]) if bbox else None,
            "h": (bbox["maxy"] - bbox["miny"]) if bbox else None,
        }
    return payload


def _split_refdes_query(query: str | None) -> list[str]:
    """高亮参数里的位号。逗号隔开时一张 PDF 画多个框。"""
    refs: list[str] = []
    for part in re.split(r"[,，]+", str(query or "")):
        ref = part.replace("@", "").strip().upper()
        if ref and ref not in refs:
            refs.append(ref)
    return refs


def _union_region(regions: list[dict]) -> dict:
    boxes = [item for item in regions if item]
    return {
        "x0": min(float(item["x0"]) for item in boxes),
        "y0": min(float(item["y0"]) for item in boxes),
        "x1": max(float(item["x1"]) for item in boxes),
        "y1": max(float(item["y1"]) for item in boxes),
        "page_no": int((boxes[0].get("page_no") or 1)),
    }


def collect_highlight_regions(conn, row: dict, query: str | None) -> list[dict]:
    """同一面 PDF 上，每个过了 IoU 的位号各算一个高亮框。"""
    regions: list[dict] = []
    for ref in _split_refdes_query(query):
        info = resolve_region(conn, row, query=ref)
        rect = info.get("pdf_rect")
        if rect:
            regions.append(rect)
    return regions


def _normalize_bbox_from_comp(comp: dict) -> dict | None:
    minx = comp.get("bbox_min_x")
    maxx = comp.get("bbox_max_x")
    miny = comp.get("bbox_min_y")
    maxy = comp.get("bbox_max_y")
    if None in (minx, miny, maxx, maxy):
        return None
    return {"minx": float(minx), "miny": float(miny), "maxx": float(maxx), "maxy": float(maxy)}


def format_match_pdf(conn, match: dict | None, session_id: str | None = None) -> dict | None:
    if not match or not match.get("project_code"):
        return None
    code = match["project_code"]
    is_top = bool(match.get("top_side", True))
    base = _pdf_url_prefix(code, is_top)
    if match.get("highlight_baked"):
        ref = match.get("refdes") or ""
        urls = component_highlight_urls(code, ref, session_id)
        return {
            "project_code": code,
            "top_side": is_top,
            "file_name": match.get("file_name") or f"{code}_{ref}_hit.pdf",
            "source_file_name": match.get("file_name"),
            "file_url": urls["file_url"],
            "preview_url": urls["preview_url"],
            "clip_url": urls["clip_url"],
            "highlight_file_url": urls["highlight_file_url"],
            "region_url": f"{base}/region",
            "component_ref": ref,
            "side": top_side_label(is_top),
            "title": match.get("file_name"),
            "highlight_found": True,
            "highlight_baked": True,
        }
    row = _get_row(conn, code, is_top)
    region = match.get("highlight_region")
    pub = pdf_public_dict(row, highlight_q=match.get("refdes"), region=region, session_id=session_id)
    pub["highlight_found"] = bool(match.get("highlight_found"))
    return pub


@router.get("")
@_log_pdf_route("list")
def pdf_list(limit: int = Query(100, ge=1, le=500)):
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT project_code, top_side, file_name
                FROM project_pdf ORDER BY id DESC LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()
        out = []
        for code, side_raw, name in rows:
            is_top = top_side_from_db(side_raw)
            meta = {
                "project_code": code,
                "top_side": is_top,
                "side": top_side_label(is_top),
                "file_name": name,
            }
            item = pdf_public_dict(meta)
            item["title"] = name
            out.append(item)
        return out
    finally:
        conn.close()


@router.get("/search")
@_log_pdf_route("search", "q=...")
def pdf_search(q: str = Query(..., min_length=1), top_n: int = Query(20, ge=1, le=50)):
    conn = connect()
    hits = []
    try:
        with conn.cursor() as cur:
            t_sql = time.perf_counter()
            cur.execute("SELECT project_code, top_side, file_name, file_content FROM project_pdf")
            rows = cur.fetchall()
            sql_ms = int((time.perf_counter() - t_sql) * 1000)
        from minio_store import measure_fetches, resolve_stored_bytes

        t_loop = time.perf_counter()
        with measure_fetches() as acc:
            for code, side_raw, name, content in rows:
                content = resolve_stored_bytes(content, purpose="pdf")
                is_top = top_side_from_db(side_raw)
                base = _pdf_url_prefix(code, is_top)
                _require_fitz()
                doc = fitz.open(stream=content, filetype="pdf")
                try:
                    if doc.page_count < 1:
                        continue
                    texts = [(doc[i].get_text("text") or "") for i in range(doc.page_count)]
                    text = "\n".join(texts)
                    if q.lower() not in text.lower():
                        continue
                    rects = search_rects(content, q)
                    hits.append(
                        {
                            "project_code": code,
                            "top_side": is_top,
                            "title": name,
                            "file_name": name,
                            "hit_count": max(len(rects), 1),
                            "snippet": _snippet(text, q),
                            "preview_url": f"{base}/file?highlight=1&q={quote(q)}",
                            "highlight_file_url": f"{base}/file?highlight=1&q={quote(q)}",
                            "clip_url": f"{base}/preview?highlight=1&q={quote(q)}&clip=1",
                            "file_url": f"{base}/file?highlight=1&q={quote(q)}",
                        }
                    )
                finally:
                    doc.close()
        loop_ms = int((time.perf_counter() - t_loop) * 1000)
        pdf_sl = acc.slice("pdf")
        scan_ms = max(0, loop_ms - pdf_sl.ms)
        log_line(
            "PDF检索",
            detail=f"关键词={q}",
            result=(
                f"查库={sql_ms}ms 【MinIO拉取PDF】{pdf_sl.format('pdf')} "
                f"全文扫描={scan_ms}ms 命中={len(hits)}"
            ),
        )
        hits.sort(key=lambda x: x["hit_count"], reverse=True)
        return {"query": q, "return_count": len(hits[:top_n]), "items": hits[:top_n]}
    finally:
        conn.close()


@router.get("/component/{project_code}/{refdes}/file")
def component_highlight_pdf(project_code: str, refdes: str, sid: str | None = None):
    """从 project_pdf 读取板级 PDF，按库内坐标高亮后返回。"""
    conn = connect()
    try:
        comp = fetch_component_row(conn, project_code, refdes)
    finally:
        conn.close()
    is_top = top_side_from_db(comp.get("top_side")) if comp else True
    side = "true" if is_top else "false"
    return _pdf_file_response(project_code, side, highlight=True, q=refdes, sid=sid)


@router.get("/component/{project_code}/{refdes}/preview")
def component_highlight_preview(project_code: str, refdes: str, sid: str | None = None):
    """位号地址上的小图。整板不再从这里出 PNG。"""
    conn = connect()
    try:
        comp = fetch_component_row(conn, project_code, refdes)
    finally:
        conn.close()
    is_top = top_side_from_db(comp.get("top_side")) if comp else True
    side = "true" if is_top else "false"
    return _pdf_thumb_response(project_code, side, q=refdes, sid=sid)


@router.get("/{project_code}/{top_side}")
@_log_pdf_route("detail")
def pdf_detail(project_code: str, top_side: str):
    is_top = _parse_path_top_side(top_side)
    conn = connect()
    try:
        row = _get_row(conn, project_code, is_top)
        return pdf_public_dict(row)
    finally:
        conn.close()


@router.get("/{project_code}/{top_side}/region")
def pdf_region(project_code: str, top_side: str, q: str | None = None):
    is_top = _parse_path_top_side(top_side)
    trace = PdfPerfTrace(route="region", detail=f"项目={project_code} 位号={q or '-'}")
    conn = connect()
    try:
        with trace.step("db_connect"):
            pass
        row = _load_row_traced(trace, conn, project_code, is_top)
        payload = resolve_region(conn, row, query=q, trace=trace)
        trace.log_line()
        return payload
    finally:
        conn.close()


def _pdf_file_response(
    project_code: str,
    top_side: str,
    *,
    highlight: bool = False,
    q: str | None = None,
    sid: str | None = None,
) -> Response:
    """板级 PDF 下载/高亮（供路由与 component 复用，勿直接调用带 Query 的路由函数）。"""
    is_top = _parse_path_top_side(top_side)
    trace = PdfPerfTrace(
        route="file",
        detail=f"项目={project_code} 高亮={int(highlight)} 位号={q or '-'}",
    )
    conn = connect()
    try:
        row = _load_row_traced(trace, conn, project_code, is_top)
        data = row["file_content"]
        if highlight:
            with trace.step("build_region_stored"):
                regions = collect_highlight_regions(conn, row, q)
            with trace.step("render_pdf"):
                data = render_highlighted_pdf(data, regions=regions)
            filename = quote(Path(row["file_name"]).stem + "_hit.pdf")
        else:
            filename = quote(row["file_name"])
        if str(sid or "").strip():
            from pdf_crypto import encrypt_for_view

            with trace.step("encrypt_pdf"):
                data = encrypt_for_view(data, sid, row["file_name"])
            log_line(
                "PDF返回",
                result=f"类型=PDF 已加密 字节={len(data)} 文件={row['file_name']}",
            )
        else:
            log_line(
                "PDF返回",
                result=f"类型=PDF 未加密 字节={len(data)} 文件={row['file_name']}",
            )
        trace.log_line()
        resp = Response(
            content=data,
            media_type="application/pdf",
            headers={"Content-Disposition": f"inline; filename*=UTF-8''{filename}"},
        )
        trace.apply_headers(resp)
        return resp
    finally:
        conn.close()


def _pdf_thumb_response(
    project_code: str,
    top_side: str,
    *,
    q: str | None = None,
    sid: str | None = None,
) -> Response:
    """列表缩略图。没有高亮框时不把整板画成 PNG，改回高亮 PDF。"""
    is_top = _parse_path_top_side(top_side)
    ref = str(q or "").strip().upper()
    trace = PdfPerfTrace(route="preview", detail=f"项目={project_code} 位号={ref or '-'} 缩略图=1")
    conn = connect()
    try:
        row = _load_row_traced(trace, conn, project_code, is_top)
        with trace.step("build_region_stored"):
            regions = collect_highlight_regions(conn, row, q)
        if not regions:
            log_line("PDF预览", result=f"无高亮框，缩略图改为高亮 PDF 项目={project_code}")
            conn.close()
            conn = None
            return _pdf_file_response(project_code, top_side, highlight=True, q=q, sid=sid)
        with trace.step("render_png"):
            png = render_page_png(row["file_content"], regions=regions, highlight=True, clip=True)
        log_line(
            "PDF预览",
            result=f"类型=PNG 缩略图 字节={len(png)} 项目={project_code} 位号={ref or '-'}",
        )
        trace.log_line()
        resp = Response(content=png, media_type="image/png")
        trace.apply_headers(resp)
        return resp
    finally:
        if conn is not None:
            conn.close()


@router.get("/{project_code}/{top_side}/file")
def pdf_file(
    project_code: str,
    top_side: str,
    highlight: bool = Query(False),
    q: str | None = None,
    sid: str | None = None,
):
    return _pdf_file_response(project_code, top_side, highlight=highlight, q=q, sid=sid)


@router.get("/{project_code}/{top_side}/preview")
def pdf_preview(
    project_code: str,
    top_side: str,
    q: str | None = None,
    highlight: bool = True,
    clip: bool = False,
    sid: str | None = None,
    zoom: float | None = Query(None),
    vx0: float | None = None,
    vy0: float | None = None,
    vx1: float | None = None,
    vy1: float | None = None,
):
    """带 clip 且没有视口参数时才是缩略图。放大、整板都返回高亮 PDF。"""
    del highlight
    viewport = zoom is not None or None not in (vx0, vy0, vx1, vy1)
    if clip and not viewport:
        return _pdf_thumb_response(project_code, top_side, q=q, sid=sid)
    log_line(
        "PDF预览",
        result=(
            f"整版改为高亮 PDF 项目={project_code} 位号={q or '-'} "
            f"忽略 zoom={zoom if zoom is not None else '-'} 视口={int(viewport)}"
        ),
    )
    return _pdf_file_response(project_code, top_side, highlight=True, q=q, sid=sid)
