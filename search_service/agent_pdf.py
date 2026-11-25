# -*- coding: utf-8 -*-
"""PDF 存储与高亮：入库只存 PDF；展示时用 Eda_ 位号校准仿射矩阵 + footprint bbox 框选。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from agent_board_align import (
    affine_from_points,
    board_footprint_to_pdf_rect,
    body_size_mm,
    pcb_bbox_to_pdf_rect,
)
from agent_log import log_line
from footprint_db import (
    fetch_component_row,
    fetch_components_for_side,
    fetch_pdf,
    update_component_pdf_refdes,
    upsert_project_pdf,
)


class PdfScaleError(RuntimeError):
    """PDF 校准失败（Eda_ 位号不足等）。"""


def pdf_refdes_search_terms(refdes: str) -> list[str]:
    """PDF 丝印搜索词：先去掉位号里的 @，再优先 Eda_ 前缀。"""
    ref = str(refdes or "").replace("@", "").strip().upper()
    if not ref:
        return []
    raw = [f"Eda_{ref}", f"EDA_{ref}", f"Eda-{ref}", ref]
    seen: set[str] = set()
    out: list[str] = []
    for term in raw:
        if term not in seen:
            seen.add(term)
            out.append(term)
    return out


def _norm_silk(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "")).upper()


def _refdes_suffix(term: str) -> str:
    """去掉 Eda_ 前缀后看位号本体，例如 Eda_U10 -> U10。"""
    t = _norm_silk(term)
    for prefix in ("EDA_", "EDA-"):
        if t.startswith(prefix):
            return t[len(prefix) :]
    return t


def _whole_term(text: str, term: str) -> bool:
    """整段文字里要出现完整词。Eda_U1 不能算进 Eda_U10，U10 不能算进 U100。"""
    hay = _norm_silk(text)
    needle = _norm_silk(term)
    if not needle:
        return False
    edge = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
    suffix = _refdes_suffix(term)
    suffix_has_digit = bool(re.search(r"\d", suffix))
    start = 0
    while True:
        index = hay.find(needle, start)
        if index < 0:
            return False
        end = index + len(needle)
        before = "" if index == 0 else hay[index - 1]
        after = "" if end >= len(hay) else hay[end]
        if (not before or before not in edge) and (not after or after not in edge):
            # U10 后面不能再接数字，否则会把 U100 当成 U10
            if suffix_has_digit and end < len(hay) and hay[end] in "0123456789":
                start = index + 1
                continue
            return True
        start = index + 1


def _word_runs(page):
    """按行把挨着的单词拼起来，供全字匹配。"""
    import pymupdf as fitz

    words = []
    for item in page.get_text("words") or []:
        words.append((fitz.Rect(item[0], item[1], item[2], item[3]), _norm_silk(item[4])))
    words.sort(key=lambda pair: (round((pair[0].y0 + pair[0].y1) / 2.0, 0), pair[0].x0))
    lines: list[list] = []
    for rect, text in words:
        if not text:
            continue
        cy = (rect.y0 + rect.y1) / 2.0
        if lines and abs(cy - lines[-1][-1][2]) <= 2.5 and rect.x0 - lines[-1][-1][0].x1 < 8:
            lines[-1].append((rect, text, cy))
        else:
            lines.append([(rect, text, cy)])
    return lines


def _exact_window(lines, term: str):
    """同一行上若干单词拼起来，必须刚好等于要找的位号。"""
    needle = _norm_silk(term)
    for line in lines:
        count = len(line)
        for start in range(count):
            acc = ""
            rect = None
            for index in range(start, count):
                acc += line[index][1]
                rect = line[index][0] if rect is None else rect | line[index][0]
                if acc == needle:
                    return rect
                if len(acc) > len(needle) or not needle.startswith(acc):
                    break
    return None


def _rect_from_hit_words(page, hit, term: str):
    """只在命中点附近拼词，文本必须刚好等于要找的位号。"""
    import pymupdf as fitz

    needle = _norm_silk(term)
    height = max(float(hit.y1 - hit.y0), 1.0)
    band = fitz.Rect(hit.x0 - 3, hit.y0 - height, hit.x1 + 3, hit.y1 + height)
    words = [
        (fitz.Rect(*item[0:4]), _norm_silk(item[4]))
        for item in page.get_text("words", clip=band) or []
        if str(item[4] or "").strip()
    ]
    words.sort(key=lambda pair: pair[0].x0)
    count = len(words)
    for start in range(count):
        acc = ""
        rect = None
        for index in range(start, count):
            acc += words[index][1]
            rect = words[index][0] if rect is None else rect | words[index][0]
            if acc == needle:
                return rect
            if len(acc) > len(needle) or not needle.startswith(acc):
                break
    return None


def _search_one_term(page, lines, term: str):
    """先全字拼词。拼不到再在 search_for 邻近收窄，必须完整等于 term。"""
    found = _exact_window(lines, term)
    if found is not None:
        return found
    for hit in page.search_for(term) or []:
        rect = _rect_from_hit_words(page, hit, term)
        if rect is not None:
            return rect
    return None


def _search_refdes_rect(page, refdes: str):
    """先找完整的 Eda_位号。找不到再用短位号等兜底。"""
    lines = _word_runs(page)
    for term in pdf_refdes_search_terms(refdes):
        rect = _search_one_term(page, lines, term)
        if rect is not None:
            return rect
    return None


def _rect_center(rect) -> tuple[float, float]:
    return (float(rect.x0 + rect.x1) / 2.0, float(rect.y0 + rect.y1) / 2.0)


def _normalize_bbox(bbox: dict | None) -> dict | None:
    if not bbox:
        return None
    minx = float(bbox.get("bbox_min_x") or bbox.get("minx") or 0)
    miny = float(bbox.get("bbox_min_y") or bbox.get("miny") or 0)
    maxx = float(bbox.get("bbox_max_x") or bbox.get("maxx") or 0)
    maxy = float(bbox.get("bbox_max_y") or bbox.get("maxy") or 0)
    if maxx <= minx or maxy <= miny:
        return None
    return {"minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy}


def _pair_distance_scale(src_a, src_b, dst_a, dst_b) -> float:
    """两点的 PDF 距离 / 板坐标距离。重合或结果为 0 时返回 0。"""
    vx = float(src_b[0]) - float(src_a[0])
    vy = float(src_b[1]) - float(src_a[1])
    wx = float(dst_b[0]) - float(dst_a[0])
    wy = float(dst_b[1]) - float(dst_a[1])
    pcb = float(np.hypot(vx, vy))
    pdf = float(np.hypot(wx, wy))
    if pcb < 1e-9 or pdf < 1e-12:
        return 0.0
    return pdf / pcb


def silk_page_index(doc) -> int:
    """带 Eda_ 位号的那一页。装配图常在第 1 页，位号图在后面；单页 PDF 仍是第 1 页。"""
    if (getattr(doc, "page_count", 0) or 0) <= 1:
        return 0
    best_i = 0
    best_n = -1
    for i in range(getattr(doc, "page_count", 0) or 0):
        text = doc[i].get_text("text") or ""
        n = text.lower().count("eda_")
        if n > best_n:
            best_n = n
            best_i = i
    return best_i if best_n > 0 else 0


def calibrate_pdf_scale(pdf_bytes: bytes, anchors: list[dict]) -> tuple[np.ndarray | None, float, str]:
    """随机找两个能在 PDF 上对上的位号算比例。这一对算出 0 就再换两个。"""
    import random

    import pymupdf as fitz

    if not pdf_bytes:
        return None, 0.0, "PDF 内容为空"
    candidates = [
        a for a in (anchors or [])
        if a.get("refdes") and a.get("pcb_x") is not None and a.get("pcb_y") is not None
    ]
    random.shuffle(candidates)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if doc.page_count < 1:
        doc.close()
        return None, 0.0, "PDF 无页面"
    page_i = silk_page_index(doc)
    page = doc[page_i]
    hits: list[tuple[str, list[float], list[float]]] = []
    scanned = 0
    zero_pairs = 0
    try:
        for item in candidates:
            scanned += 1
            ref = str(item.get("refdes") or "").replace("@", "").strip().upper()
            if not ref:
                continue
            rect = _search_refdes_rect(page, ref)
            if rect is None:
                continue
            cx, cy = _rect_center(rect)
            hits.append((ref, [float(item["pcb_x"]), float(item["pcb_y"])], [cx, cy]))
            if len(hits) < 2:
                continue
            (ref_a, src_a, dst_a), (ref_b, src_b, dst_b) = hits[-2], hits[-1]
            scale = _pair_distance_scale(src_a, src_b, dst_a, dst_b)
            if scale <= 0:
                zero_pairs += 1
                log_line(
                    "PDF比例",
                    result=f"【比例为0换一对】{ref_a},{ref_b} 已放弃 {zero_pairs} 对，继续随机找",
                    level="warning",
                )
                hits.clear()
                continue
            matrix = affine_from_points(
                np.array([src_a, src_b], dtype=np.float64),
                np.array([dst_a, dst_b], dtype=np.float64),
            )
            note = (
                f"页={page_i + 1} 随机两点={ref_a},{ref_b} 已查看={scanned} "
                f"放弃0比例={zero_pairs} pdf_scale={scale:.12g}"
            )
            return matrix, scale, note
    finally:
        doc.close()
    if len(hits) == 1:
        return None, 0.0, f"只命中 1 个位号 {hits[0][0]}，凑不齐两个点，pdf_scale=0 已查看={scanned}"
    return None, 0.0, f"没有凑出比例非 0 的两个位号 已查看={scanned} 放弃0比例={zero_pairs}"


def compute_board_affine_matrix(pdf_bytes: bytes, anchors: list[dict]) -> tuple[np.ndarray | None, str]:
    """用随机两个命中位号求 PCB→PDF 相似变换。"""
    matrix, _scale, note = calibrate_pdf_scale(pdf_bytes, anchors)
    return matrix, note


_MILS_PER_MM = 39.3700787


def _component_body_wh(comp: dict | None, bbox: dict | None) -> tuple[float, float]:
    """封装宽高（与库内 pcb 坐标同单位）。"""
    box = _normalize_bbox(bbox)
    if box:
        return box["maxx"] - box["minx"], box["maxy"] - box["miny"]
    if not comp:
        return 4.0 * _MILS_PER_MM, 3.0 * _MILS_PER_MM
    w_mm, h_mm = body_size_mm(comp.get("packagesymbol"), comp.get("pin_count"))
    units = str(comp.get("units") or "mils").lower()
    if units in ("mm", "millimeter", "millimetre"):
        return w_mm, h_mm
    return w_mm * _MILS_PER_MM, h_mm * _MILS_PER_MM


def _eda_center_on_pdf(pdf_bytes: bytes, refdes: str) -> tuple[float, float] | None:
    import pymupdf as fitz

    if not pdf_bytes or not refdes:
        return None
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        if doc.page_count < 1:
            return None
        rect = _search_refdes_rect(doc[silk_page_index(doc)], refdes)
        if rect is None:
            return None
        return _rect_center(rect)
    finally:
        doc.close()


def _rect_from_center(cx: float, cy: float, w: float, h: float) -> dict:
    hw, hh = w / 2.0, h / 2.0
    return {"x0": cx - hw, "y0": cy - hh, "x1": cx + hw, "y1": cy + hh}


def _center_region_on_pdf_refdes(region: dict, eda_center: tuple[float, float]) -> dict:
    """保持 PDF 框宽高，平移到 PyMuPDF 搜到的 Eda_ 丝印中心（位置以 PDF 为准）。"""
    w = max(float(region["x1"] - region["x0"]), 0.5)
    h = max(float(region["y1"] - region["y0"]), 0.5)
    out = _rect_from_center(float(eda_center[0]), float(eda_center[1]), w, h)
    out["page_no"] = int(region.get("page_no") or 1)
    return out


def _pdf_size_from_matrix(matrix: np.ndarray, pcb_x: float, pcb_y: float, bw: float, bh: float) -> tuple[float, float]:
    """把板坐标宽高映射到 PDF 尺寸（含旋转）；pcb_x/y 为封装原点/中心，bw/bh 为全宽全高。"""
    from agent_board_align import apply_affine

    o = apply_affine(matrix, np.array([[pcb_x, pcb_y]], dtype=np.float64))[0]
    wx = apply_affine(matrix, np.array([[pcb_x + bw * 0.5, pcb_y]], dtype=np.float64))[0]
    hy = apply_affine(matrix, np.array([[pcb_x, pcb_y + bh * 0.5]], dtype=np.float64))[0]
    pw = 2.0 * float(np.hypot(wx[0] - o[0], wx[1] - o[1]))
    ph = 2.0 * float(np.hypot(hy[0] - o[0], hy[1] - o[1]))
    return max(pw, 2.0), max(ph, 2.0)


def stamp_pdf_region_highlight(page, region: dict) -> None:
    """在 PDF 页上绘制轴对齐高亮框（与 DAY002ori 一致，不用四边形/平行四边形）。"""
    import pymupdf as fitz

    rect = fitz.Rect(region["x0"], region["y0"], region["x1"], region["y1"])
    shape = page.new_shape()
    shape.draw_rect(rect)
    shape.finish(
        color=(0.92, 0.18, 0.12),
        fill=(1.0, 0.82, 0.18),
        fill_opacity=0.14,
        stroke_opacity=0.72,
        width=0.9,
    )
    shape.commit()


def compute_pdf_region(
    *,
    matrix: np.ndarray | None,
    pcb_x: float | None,
    pcb_y: float | None,
    bbox: dict | None,
    comp: dict | None = None,
    pdf_bytes: bytes | None = None,
    refdes: str | None = None,
) -> dict | None:
    """
    PCB→PDF 高亮框：
    1) 两点（及以上）相似/仿射：PCB→PDF，用库内 bbox 算 PDF 宽高；
    2) PyMuPDF 搜索 Eda_{refdes} 得丝印中心，框以该中心对齐（大小来自形态外接矩形变换）。
    """
    ref = str(refdes or (comp or {}).get("refdes") or "").strip().upper()
    eda_center = _eda_center_on_pdf(pdf_bytes, ref) if pdf_bytes and ref else None

    region: dict | None = None
    if matrix is not None and pcb_x is not None and pcb_y is not None:
        box = _normalize_bbox(bbox)
        px, py = float(pcb_x), float(pcb_y)
        if box:
            region = board_footprint_to_pdf_rect(matrix, px, py, box)
        else:
            bw, bh = _component_body_wh(comp, bbox)
            region = board_footprint_to_pdf_rect(
                matrix,
                px,
                py,
                {"minx": 0.0, "miny": 0.0, "maxx": bw, "maxy": bh},
            )

    if region is None and eda_center:
        bw, bh = _component_body_wh(comp, bbox)
        if matrix is not None and pcb_x is not None and pcb_y is not None:
            pw, ph = _pdf_size_from_matrix(matrix, float(pcb_x), float(pcb_y), bw, bh)
            region = _rect_from_center(eda_center[0], eda_center[1], pw, ph)
        else:
            region = _rect_from_center(eda_center[0], eda_center[1], 28.0, 18.0)
    elif region and eda_center:
        region = _center_region_on_pdf_refdes(region, eda_center)

    return region


def silk_refdes_rect(pdf_bytes: bytes, refdes: str) -> dict | None:
    """PDF 上 Eda_ 位号丝印矩形（比全文搜索更准）。"""
    import pymupdf as fitz

    if not pdf_bytes:
        return None
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        if doc.page_count < 1:
            return None
        rect = _search_refdes_rect(doc[silk_page_index(doc)], refdes)
        if rect is None:
            return None
        return {"x0": rect.x0, "y0": rect.y0, "x1": rect.x1, "y1": rect.y1, "query": refdes}
    finally:
        doc.close()


def _guess_top_side(filename: str, index: int, total: int) -> bool:
    """从 PDF 文件名推断正反面；无标记时默认正面。"""
    name = filename.lower()
    if any(k in name for k in ("bottom", "back", "反面", "_no", "-no", "bot")):
        return False
    if any(k in name for k in ("top", "正面", "_yes", "-yes")):
        return True
    if total == 2:
        return index == 0
    return True


def _estimate_pdf_scale(matrix: np.ndarray, pcb_x: float, pcb_y: float, bbox: dict) -> float:
    """由仿射矩阵与封装 bbox 估算 PDF点/板mil 比例（写入 project_pdf.pdf_scale）。"""
    box = _normalize_bbox(bbox)
    if not box:
        return 0.0
    bw = max(box["maxx"] - box["minx"], 0.01)
    bh = max(box["maxy"] - box["miny"], 0.01)
    pw, ph = _pdf_size_from_matrix(matrix, float(pcb_x), float(pcb_y), bw, bh)
    return float((pw / bw + ph / bh) / 2.0)


def highlight_region_from_stored(comp: dict, pdf_scale: float) -> dict | None:
    """用库内 pdf_refdes_x/y（丝印中心）+ bbox 范围 × project_pdf.pdf_scale 生成高亮框。"""
    ax = comp.get("pdf_refdes_x")
    ay = comp.get("pdf_refdes_y")
    if ax is None or ay is None:
        return None
    scale = float(pdf_scale or 0)
    if scale <= 0:
        return None
    box = _normalize_bbox(
        {
            "bbox_min_x": comp.get("bbox_min_x"),
            "bbox_min_y": comp.get("bbox_min_y"),
            "bbox_max_x": comp.get("bbox_max_x"),
            "bbox_max_y": comp.get("bbox_max_y"),
        }
    )
    if box:
        pw = (box["maxx"] - box["minx"]) * scale
        ph = (box["maxy"] - box["miny"]) * scale
    else:
        bw, bh = _component_body_wh(comp, None)
        pw, ph = bw * scale, bh * scale
    pw, ph = max(pw, 8.0), max(ph, 8.0)
    return _rect_from_center(float(ax), float(ay), pw, ph)


def index_refdes_pdf_position(conn, *, project_code: str, refdes: str, pdf_bytes: bytes) -> bool:
    """PyMuPDF 搜索 Eda_ 位号，坐标写入 footprint_library。"""
    center = _eda_center_on_pdf(pdf_bytes, refdes)
    if not center:
        return False
    update_component_pdf_refdes(
        conn,
        project_code=project_code,
        refdes=refdes,
        pdf_refdes_x=center[0],
        pdf_refdes_y=center[1],
    )
    return True


def store_board_pdf(
    conn,
    *,
    project_code: str,
    top_side: bool,
    file_name: str,
    content: bytes,
    calibration_items: list[dict] | None = None,
    sample_item: dict | None = None,
) -> str | None:
    """板级 PDF 入库：始终写入 project_pdf；再算仿射/pdf_scale 并索引各位号坐标。"""
    if not content:
        raise PdfScaleError("PDF 内容为空")
    items = list(calibration_items or [])
    if not items and sample_item:
        items = [sample_item]
    anchors = [
        {"refdes": item.get("refdes"), "pcb_x": item.get("pcb_x"), "pcb_y": item.get("pcb_y")}
        for item in items
        if item.get("refdes")
    ]
    warn: str | None = None
    matrix, scale, note = calibrate_pdf_scale(content, anchors)
    if matrix is None or scale <= 0:
        upsert_project_pdf(
            conn,
            project_code=project_code,
            top_side=top_side,
            file_name=file_name,
            content=content,
            pdf_scale=0.0,
        )
        log_line(
            "PDF存储",
            detail=f"项目={project_code} 面={top_side} 文件={file_name}",
            result=f"【校准未完成】{note} pdf_scale=0",
            level="warning",
        )
        stat = index_project_pdf_refdes(conn, project_code=project_code, top_side=top_side)
        warn = f"PDF 已入库但未完成仿射校准：{note}"
        if stat.get("fail"):
            warn += f"；位号坐标索引 fail={stat['fail']}"
        return warn
    upsert_project_pdf(
        conn,
        project_code=project_code,
        top_side=top_side,
        file_name=file_name,
        content=content,
        pdf_scale=scale,
    )
    log_line(
        "PDF存储",
        detail=f"项目={project_code} 面={top_side} 文件={file_name}",
        result=f"{note} pdf_scale={scale:.12g}",
    )
    index_project_pdf_refdes(conn, project_code=project_code, top_side=top_side)
    return warn


def _inflate_region(region: dict, pad: float, page_rect) -> dict:
    x0, y0, x1, y1 = region["x0"], region["y0"], region["x1"], region["y1"]
    px0, py0, px1, py1 = page_rect
    return {
        "x0": max(px0, x0 - pad),
        "y0": max(py0, y0 - pad),
        "x1": min(px1, x1 + pad),
        "y1": min(py1, y1 + pad),
    }


def bake_component_highlight_files(
    pdf_bytes: bytes,
    region: dict,
    *,
    refdes: str,
    project_code: str,
    board_file_name: str,
    pad: float = 24.0,
    zoom: float = 2.0,
) -> tuple[bytes | None, bytes | None]:
    """从板级 PDF + 高亮区域烘焙：裁切高亮 PDF + 预览 PNG（仅第一页）。"""
    import pymupdf as fitz

    if not pdf_bytes or not region:
        return None, None
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if doc.page_count < 1:
        doc.close()
        return None, None
    idx = silk_page_index(doc)
    page = doc[idx]
    stamp_pdf_region_highlight(page, region)
    clip = _inflate_region(region, pad, (page.rect.x0, page.rect.y0, page.rect.x1, page.rect.y1))
    clip_rect = fitz.Rect(clip["x0"], clip["y0"], clip["x1"], clip["y1"])
    if clip_rect.is_empty or clip_rect.width < 0.5 or clip_rect.height < 0.5:
        doc.close()
        return None, None

    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip_rect, alpha=False)
    preview_png = pix.tobytes("png")

    out_doc = fitz.open()
    out_page = out_doc.new_page(width=clip_rect.width, height=clip_rect.height)
    out_page.show_pdf_page(out_page.rect, doc, idx, clip=clip_rect)
    clip_pdf = out_doc.tobytes()
    out_doc.close()
    doc.close()

    return clip_pdf, preview_png


def component_highlight_download_name(project_code: str, refdes: str) -> str:
    """下载响应用文件名（不入库）。"""
    ref = str(refdes or "").strip().upper() or "REF"
    code = str(project_code or "").strip() or "project"
    return f"{code}_{ref}_hit.pdf"


def index_single_refdes_pdf_position(
    conn,
    *,
    project_code: str,
    refdes: str,
    top_side: bool | None = None,
    cache: dict | None = None,
) -> bool:
    """为单个位号索引 PDF 丝印坐标（不写 BLOB）。"""
    from top_side_util import parse_top_side

    comp = fetch_component_row(conn, project_code, refdes)
    if not comp:
        return False
    is_top = parse_top_side(top_side if top_side is not None else comp.get("top_side"))
    key = (project_code, is_top)
    cache = cache if cache is not None else {}
    if key not in cache:
        row = fetch_pdf(conn, project_code, is_top)
        cache[key] = row
    row = cache.get(key)
    if not row:
        return False
    return index_refdes_pdf_position(
        conn,
        project_code=project_code,
        refdes=refdes,
        pdf_bytes=row["file_content"],
    )


def index_project_pdf_refdes(conn, *, project_code: str, top_side: bool | None) -> dict:
    """项目某面全部位号：PyMuPDF 搜 Eda_ 写 pdf_refdes_x/y。"""
    from top_side_util import parse_top_side

    is_top = parse_top_side(top_side)
    rows = fetch_components_for_side(conn, project_code, is_top)
    pdf_cache: dict = {}
    ok = 0
    fail = 0
    for row in rows:
        ref = row.get("refdes")
        if not ref:
            continue
        try:
            if index_single_refdes_pdf_position(
                conn,
                project_code=project_code,
                refdes=ref,
                top_side=is_top,
                cache=pdf_cache,
            ):
                ok += 1
            else:
                fail += 1
        except Exception as exc:
            fail += 1
            log_line(
                "PDF位号索引",
                detail=f"项目={project_code} 位号={ref}",
                result=str(exc),
                level="warning",
            )
    log_line(
        "PDF位号索引",
        detail=f"项目={project_code} 面={is_top}",
        result=f"成功={ok} 失败={fail}",
    )
    return {"ok": ok, "fail": fail}


def bake_project_component_pdfs(conn, *, project_code: str, top_side: bool | None) -> dict:
    """兼容旧脚本名：现为索引 PDF 位号坐标。"""
    return index_project_pdf_refdes(conn, project_code=project_code, top_side=top_side)


def component_highlight_urls(project_code: str, refdes: str, session_id: str | None = None) -> dict:
    """位号地址：整板是高亮 PDF，preview 只留缩略图。"""
    from urllib.parse import quote

    code = quote(str(project_code or "").strip())
    ref = quote(str(refdes or "").strip().upper())
    base = f"/api/pdf/component/{code}/{ref}"
    sid = quote(str(session_id or "").strip())
    sidq = f"?sid={sid}" if sid else ""
    highlighted = f"{base}/file{sidq}"
    return {
        "highlight_file_url": highlighted,
        "file_url": highlighted,
        "preview_url": highlighted,
        "clip_url": f"{base}/preview",
    }


def _calibration_anchors(conn, project_code: str, top_side: bool | None) -> list[dict]:
    from top_side_util import parse_top_side

    rows = fetch_components_for_side(conn, project_code, parse_top_side(top_side))
    return [
        {
            "refdes": r.get("refdes"),
            "pcb_x": r.get("pcb_x"),
            "pcb_y": r.get("pcb_y"),
        }
        for r in rows
        if r.get("refdes") and r.get("pcb_x") is not None and r.get("pcb_y") is not None
    ]


def _new_pdf_align_cache() -> dict:
    """单次 attach 批处理内复用：同一 project+side 只算一次仿射矩阵。"""
    return {}


def _resolve_pdf_match(
    conn,
    *,
    project_code: str,
    refdes: str,
    top_side: bool | None,
    cache: dict | None = None,
    skip_baked_shortcut: bool = False,
) -> dict | None:
    from top_side_util import parse_top_side

    is_top = parse_top_side(top_side)
    key = (project_code, is_top)
    cache = cache if cache is not None else _new_pdf_align_cache()
    comp = fetch_component_row(conn, project_code, refdes) or {}
    row = fetch_pdf(conn, project_code, is_top)
    if not row:
        return None
    region = highlight_region_from_stored(comp, row.get("pdf_scale") or 0)
    note = "stored_anchor"
    if not region and not skip_baked_shortcut:
        if key not in cache:
            anchors = _calibration_anchors(conn, project_code, is_top)
            matrix, note = compute_board_affine_matrix(row["file_content"], anchors)
            cache[key] = {"row": row, "matrix": matrix, "note": note}
        entry = cache[key]
        matrix = entry.get("matrix")
        note = entry.get("note") or note
        if matrix is None:
            log_line(
                "PDF高亮",
                detail=f"项目={project_code} 位号={refdes}",
                result=f"【跳过】{note}",
                level="warning",
            )
            return None
        bbox = {
            "bbox_min_x": comp.get("bbox_min_x"),
            "bbox_min_y": comp.get("bbox_min_y"),
            "bbox_max_x": comp.get("bbox_max_x"),
            "bbox_max_y": comp.get("bbox_max_y"),
        }
        region = compute_pdf_region(
            matrix=matrix,
            pcb_x=comp.get("pcb_x"),
            pcb_y=comp.get("pcb_y"),
            bbox=bbox,
            comp=comp,
            pdf_bytes=row["file_content"],
            refdes=refdes,
        )
    if not region:
        log_line(
            "PDF高亮",
            detail=f"项目={project_code} 位号={refdes}",
            result="【无高亮框】bbox 或库内坐标无效",
            level="warning",
        )
    return {
        "project_code": project_code,
        "refdes": refdes,
        "top_side": row["top_side"],
        "file_name": row["file_name"],
        "highlight_region": region,
        "highlight_found": bool(region),
        "highlight_baked": False,
        "align_note": note,
    }


def match_pdf_for_item(conn, *, project_code: str, refdes: str, top_side: bool | None) -> dict | None:
    return _resolve_pdf_match(conn, project_code=project_code, refdes=refdes, top_side=top_side)


def match_pdf_for_items_batch(conn, items: list[dict]) -> list[dict | None]:
    """批量高亮：同项目同面 PDF/仿射矩阵只算一次。"""
    cache = _new_pdf_align_cache()
    out: list[dict | None] = []
    for item in items or []:
        out.append(
            _resolve_pdf_match(
                conn,
                project_code=item["project_code"],
                refdes=item["refdes"],
                top_side=item.get("top_side"),
                cache=cache,
            )
        )
    return out


def render_highlighted_pdf(
    pdf_bytes: bytes,
    region: dict | None = None,
    regions: list | None = None,
) -> bytes:
    """在板级 PDF 唯一页上画框后导出，不改原文件。多个框画在同一页。"""
    import pymupdf as fitz

    import time

    from agent_log import log_line

    boxes = [item for item in (regions or []) if item]
    if region and region not in boxes:
        boxes.append(region)
    t_open = time.perf_counter()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    open_ms = int((time.perf_counter() - t_open) * 1000)
    if doc.page_count < 1:
        doc.close()
        log_line("PDF打开", result=f"打开={open_ms}ms 页数=0 字节={len(pdf_bytes)}")
        return pdf_bytes
    t_page = time.perf_counter()
    page_index = silk_page_index(doc)
    page = doc[page_index]
    log_line(
        "PDF打开",
        result=(
            f"打开={open_ms}ms 选页={int((time.perf_counter() - t_page) * 1000)}ms "
            f"字节={len(pdf_bytes)} 页数={doc.page_count} "
            f"选中页={page_index + 1} 宽={page.rect.width:.1f} 高={page.rect.height:.1f}"
        ),
    )
    if boxes:
        t_cover = time.perf_counter()
        covered, white_count = _cover_and_highlight(page, boxes)
        log_line(
            "PDF遮挡",
            result=(
                f"遮挡={int((time.perf_counter() - t_cover) * 1000)}ms "
                f"高亮框={len(boxes)} 留空并集={covered} 白底块={white_count} "
                f"方式=并集外白底一次提交 未栅格化"
            ),
        )
    t_save = time.perf_counter()
    data = doc.tobytes(deflate=True)
    doc.close()
    log_line(
        "PDF导出",
        result=f"写出={int((time.perf_counter() - t_save) * 1000)}ms 字节={len(data)}",
    )
    return data


def _keep_square(region: dict) -> tuple[float, float, float, float]:
    """保留区以高亮框为中心。面积是「长边正方形的两倍」再乘 10。"""
    import math

    x0 = float(region["x0"])
    y0 = float(region["y0"])
    x1 = float(region["x1"])
    y1 = float(region["y1"])
    long_edge = max(x1 - x0, y1 - y0, 1.0)
    side = long_edge * math.sqrt(20.0)
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    half = side / 2.0
    return cx - half, cy - half, cx + half, cy + half


def _white_outside(page_rect, holes: list) -> list:
    """页面去掉留空并集后，剩下的若干白底矩形。重叠处仍算留空。"""
    import pymupdf as fitz

    if not holes:
        return [page_rect]
    xs = {float(page_rect.x0), float(page_rect.x1)}
    for hole in holes:
        xs.add(float(hole.x0))
        xs.add(float(hole.x1))
    marks = sorted(x for x in xs if float(page_rect.x0) - 0.01 <= x <= float(page_rect.x1) + 0.01)
    whites = []
    for index in range(len(marks) - 1):
        x0, x1 = marks[index], marks[index + 1]
        if x1 - x0 < 0.2:
            continue
        covered: list[tuple[float, float]] = []
        for hole in holes:
            if float(hole.x0) >= x1 - 1e-6 or float(hole.x1) <= x0 + 1e-6:
                continue
            top = max(float(hole.y0), float(page_rect.y0))
            bottom = min(float(hole.y1), float(page_rect.y1))
            if bottom - top < 0.2:
                continue
            covered.append((top, bottom))
        covered.sort()
        merged: list[list[float]] = []
        for top, bottom in covered:
            if merged and top <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], bottom)
            else:
                merged.append([top, bottom])
        cursor = float(page_rect.y0)
        for top, bottom in merged:
            if top - cursor > 0.2:
                whites.append(fitz.Rect(x0, cursor, x1, top))
            cursor = max(cursor, bottom)
        if float(page_rect.y1) - cursor > 0.2:
            whites.append(fitz.Rect(x0, cursor, x1, float(page_rect.y1)))
    return whites


def _cover_and_highlight(page, regions: list[dict]) -> tuple[int, int]:
    """留空取各器件的并集，白底和高亮框只提交一次。"""
    import pymupdf as fitz

    rect = page.rect
    holes = []
    for region in regions:
        x0, y0, x1, y1 = _keep_square(region)
        hole = fitz.Rect(x0, y0, x1, y1) & rect
        if hole.is_empty or hole.width < 1 or hole.height < 1:
            continue
        holes.append(hole)
    whites = [] if any(
        hole.width >= rect.width - 1 and hole.height >= rect.height - 1 for hole in holes
    ) else _white_outside(rect, holes)
    shape = page.new_shape()
    for white in whites:
        shape.draw_rect(white)
        shape.finish(color=None, fill=(1, 1, 1), fill_opacity=1, width=0, closePath=True)
    for region in regions:
        shape.draw_rect(fitz.Rect(float(region["x0"]), float(region["y0"]), float(region["x1"]), float(region["y1"])))
        shape.finish(
            color=(0.92, 0.18, 0.12),
            fill=(1.0, 0.82, 0.18),
            fill_opacity=0.14,
            stroke_opacity=0.72,
            width=0.9,
            closePath=True,
        )
    shape.commit(overlay=True)
    return len(holes), len(whites)
