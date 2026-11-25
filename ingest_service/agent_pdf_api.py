# -*- coding: utf-8 -*-
"""PDF HTTP 接口（兼容 DAY002 前端）：基于 project_pdf 单表，单页 PNG 预览 + 区域定位。"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from agent_pdf import (
    _calibration_anchors,
    compute_board_affine_matrix,
    compute_pdf_region,
    pdf_refdes_search_terms,
    render_highlighted_pdf,
    silk_page_index,
    silk_refdes_rect,
)
from agent_pdf import component_highlight_download_name, component_highlight_urls, stamp_pdf_region_highlight
from footprint_db import connect, fetch_component_highlight_blobs, fetch_component_row, fetch_pdf
from top_side_util import (
    parse_top_side_url_token,
    top_side_from_db,
    top_side_label,
    top_side_url_token,
)

router = APIRouter(prefix="/api/pdf", tags=["pdf"])

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
    row = fetch_pdf(conn, project_code, top_side)
    if not row:
        raise HTTPException(404, "PDF 不存在")
    is_top = row["top_side"]
    content = row["file_content"]
    page_size = _first_page_size(content)
    return {
        "project_code": row["project_code"],
        "top_side": is_top,
        "side": top_side_label(is_top),
        "file_name": row["file_name"],
        "file_content": content,
        "pdf_scale": row["pdf_scale"],
        "page": page_size,
    }


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
    highlight: bool = True,
    clip: bool = False,
    view: dict | None = None,
) -> bytes:
    _require_fitz()
    doc = fitz.open(stream=content, filetype="pdf")
    try:
        page = doc[silk_page_index(doc)]
        if region and highlight:
            stamp_pdf_region_highlight(page, region)
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
        elif clip and region:
            padded = _inflate_rect(region, 24, (page.rect.x0, page.rect.y0, page.rect.x1, page.rect.y1))
            clip_rect = fitz.Rect(padded["x0"], padded["y0"], padded["x1"], padded["y1"])
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip_rect, alpha=False)
        return pix.tobytes("png")
    finally:
        doc.close()


def pdf_public_dict(row: dict, *, highlight_q: str | None = None, region: dict | None = None) -> dict:
    code = row["project_code"]
    is_top = bool(row["top_side"])
    base = _pdf_url_prefix(code, is_top)
    q = quote(str(highlight_q or "").strip())
    qpart = f"&q={q}" if q else ""
    preview = f"{base}/preview?highlight=1{qpart}"
    return {
        "project_code": code,
        "top_side": is_top,
        "file_name": row["file_name"],
        "source_file_name": row["file_name"],
        "file_url": f"{base}/file",
        "preview_url": preview,
        "clip_url": preview + "&clip=1",
        "region_url": f"{base}/region",
        "highlight_file_url": f"{base}/file?highlight=1{qpart}" if q else f"{base}/file?highlight=1",
        "component_ref": highlight_q,
        "side": row.get("side") or top_side_label(is_top),
        "title": row.get("file_name"),
        "highlight_region": region,
    }


def resolve_region(conn, row: dict, query: str | None = None) -> dict:
    ref = str(query or "").strip().upper()
    content = row["file_content"]
    pdf_rect = None
    source = "none"
    comp = None
    page_info = row.get("page")
    if ref and row.get("project_code"):
        comp = fetch_component_row(conn, row["project_code"], ref)
    if comp and ref:
        anchors = _calibration_anchors(conn, row["project_code"], row.get("top_side"))
        matrix, _note = compute_board_affine_matrix(content, anchors)
        if matrix is not None:
            bbox = {
                "bbox_min_x": comp.get("bbox_min_x"),
                "bbox_min_y": comp.get("bbox_min_y"),
                "bbox_max_x": comp.get("bbox_max_x"),
                "bbox_max_y": comp.get("bbox_max_y"),
            }
            pdf_rect = compute_pdf_region(
                matrix=matrix,
                pcb_x=comp.get("pcb_x"),
                pcb_y=comp.get("pcb_y"),
                bbox=bbox,
                comp=comp,
                pdf_bytes=content,
                refdes=ref,
            )
            if pdf_rect:
                source = "align"
    if not pdf_rect and ref:
        silk = silk_refdes_rect(content, ref)
        if silk:
            pdf_rect = silk
            source = "silk"
        else:
            rects = search_rects(content, ref)
            if rects:
                pdf_rect = rects[0]
                source = "silk"
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


def _normalize_bbox_from_comp(comp: dict) -> dict | None:
    minx = comp.get("bbox_min_x")
    maxx = comp.get("bbox_max_x")
    miny = comp.get("bbox_min_y")
    maxy = comp.get("bbox_max_y")
    if None in (minx, miny, maxx, maxy):
        return None
    return {"minx": float(minx), "miny": float(miny), "maxx": float(maxx), "maxy": float(maxy)}


def format_match_pdf(conn, match: dict | None) -> dict | None:
    if not match or not match.get("project_code"):
        return None
    code = match["project_code"]
    is_top = bool(match.get("top_side", True))
    base = _pdf_url_prefix(code, is_top)
    if match.get("highlight_baked"):
        ref = match.get("refdes") or ""
        urls = component_highlight_urls(code, ref)
        return {
            "project_code": code,
            "top_side": is_top,
            "file_name": match.get("file_name") or f"{code}_{ref}_hit.pdf",
            "source_file_name": match.get("file_name"),
            "file_url": urls["highlight_file_url"],
            "preview_url": match.get("preview_url") or urls["preview_url"],
            "clip_url": match.get("clip_url") or urls["clip_url"],
            "highlight_file_url": match.get("highlight_file_url") or urls["highlight_file_url"],
            "region_url": f"{base}/region",
            "component_ref": ref,
            "side": top_side_label(is_top),
            "title": match.get("file_name"),
            "highlight_found": True,
            "highlight_baked": True,
        }
    row = _get_row(conn, code, is_top)
    region = match.get("highlight_region")
    pub = pdf_public_dict(row, highlight_q=match.get("refdes"), region=region)
    pub["highlight_found"] = bool(match.get("highlight_found"))
    return pub


@router.get("")
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
def pdf_search(q: str = Query(..., min_length=1), top_n: int = Query(20, ge=1, le=50)):
    conn = connect()
    hits = []
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT project_code, top_side, file_name, file_content FROM project_pdf")
            rows = cur.fetchall()
        for code, side_raw, name, content in rows:
            from minio_store import resolve_stored_bytes

            content = resolve_stored_bytes(content)
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
                        "preview_url": f"{base}/preview?q={quote(q)}",
                        "file_url": f"{base}/file",
                    }
                )
            finally:
                doc.close()
        hits.sort(key=lambda x: x["hit_count"], reverse=True)
        return {"query": q, "return_count": len(hits[:top_n]), "items": hits[:top_n]}
    finally:
        conn.close()


@router.get("/component/{project_code}/{refdes}/file")
def component_highlight_pdf(project_code: str, refdes: str):
    """直接返回位号级已烘焙高亮 PDF。"""
    conn = connect()
    try:
        blob = fetch_component_highlight_blobs(conn, project_code, refdes)
        if not blob:
            raise HTTPException(404, "位号高亮 PDF 不存在，请先烘焙")
        filename = quote(component_highlight_download_name(project_code, refdes))
        return Response(
            content=blob["highlight_pdf"],
            media_type="application/pdf",
            headers={"Content-Disposition": f"inline; filename*=UTF-8''{filename}"},
        )
    finally:
        conn.close()


@router.get("/component/{project_code}/{refdes}/preview")
def component_highlight_preview(project_code: str, refdes: str):
    """直接返回位号级已烘焙预览 PNG。"""
    conn = connect()
    try:
        blob = fetch_component_highlight_blobs(conn, project_code, refdes)
        if not blob or not blob.get("highlight_preview"):
            raise HTTPException(404, "位号预览图不存在，请先烘焙")
        return Response(content=blob["highlight_preview"], media_type="image/png")
    finally:
        conn.close()


@router.get("/{project_code}/{top_side}")
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
    conn = connect()
    try:
        row = _get_row(conn, project_code, is_top)
        return resolve_region(conn, row, query=q)
    finally:
        conn.close()


@router.get("/{project_code}/{top_side}/file")
def pdf_file(
    project_code: str,
    top_side: str,
    highlight: bool = Query(False),
    q: str | None = None,
):
    is_top = _parse_path_top_side(top_side)
    conn = connect()
    try:
        row = _get_row(conn, project_code, is_top)
        data = row["file_content"]
        if highlight:
            info = resolve_region(conn, row, query=q)
            region = info.get("pdf_rect")
            data = render_highlighted_pdf(data, region)
            filename = quote(Path(row["file_name"]).stem + "_hit.pdf")
        else:
            filename = quote(row["file_name"])
        return Response(
            content=data,
            media_type="application/pdf",
            headers={"Content-Disposition": f"inline; filename*=UTF-8''{filename}"},
        )
    finally:
        conn.close()


@router.get("/{project_code}/{top_side}/preview")
def pdf_preview(
    project_code: str,
    top_side: str,
    q: str | None = None,
    highlight: bool = True,
    clip: bool = False,
    zoom: float | None = Query(None, ge=0.2, le=24),
    vx0: float | None = None,
    vy0: float | None = None,
    vx1: float | None = None,
    vy1: float | None = None,
):
    is_top = _parse_path_top_side(top_side)
    conn = connect()
    try:
        row = _get_row(conn, project_code, is_top)
        region = None
        if highlight or clip:
            info = resolve_region(conn, row, query=q)
            region = info.get("pdf_rect")
            if info.get("source") == "none":
                region = None
        view = None
        if None not in (vx0, vy0, vx1, vy1):
            view = {"x0": vx0, "y0": vy0, "x1": vx1, "y1": vy1}
        png = render_page_png(
            row["file_content"],
            query=q if not region else None,
            region=region,
            highlight=highlight and bool(region),
            clip=clip and bool(region) and view is None,
            zoom=zoom,
            view=view,
        )
        return Response(content=png, media_type="image/png")
    finally:
        conn.close()
