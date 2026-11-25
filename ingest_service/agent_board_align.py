# -*- coding: utf-8 -*-
"""PCB 板坐标 ↔ PDF 仿射变换（与 DAY002 board_align 一致）。"""

from __future__ import annotations

import numpy as np


def affine_from_points(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """src/dst 为 Nx2。2 点相似变换，3 点及以上仿射。返回 2x3。"""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2:
        raise ValueError("src 与 dst 必须是同样形状的 Nx2 点列")
    n = src.shape[0]
    if n < 2:
        raise ValueError("至少需要 2 对点")
    if n == 2:
        p1, p2 = src
        q1, q2 = dst
        v = p2 - p1
        w = q2 - q1
        v2 = float(v[0] * v[0] + v[1] * v[1])
        if v2 < 1e-12:
            raise ValueError("源点重合，无法求变换")
        scale = float(np.hypot(w[0], w[1]) / np.sqrt(v2))
        ang = float(np.arctan2(w[1], w[0]) - np.arctan2(v[1], v[0]))
        c, s = np.cos(ang) * scale, np.sin(ang) * scale
        r = np.array([[c, -s], [s, c]], dtype=np.float64)
        t = q1 - r @ p1
        return np.column_stack([r, t])
    rows: list[list[float]] = []
    b: list[float] = []
    for (x, y), (xp, yp) in zip(src, dst):
        rows.append([x, y, 1.0, 0.0, 0.0, 0.0])
        b.append(xp)
        rows.append([0.0, 0.0, 0.0, x, y, 1.0])
        b.append(yp)
    sol, *_ = np.linalg.lstsq(np.asarray(rows), np.asarray(b), rcond=None)
    a, bb, tx, c, d, ty = sol
    return np.array([[a, bb, tx], [c, d, ty]], dtype=np.float64)


def apply_affine(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, 2)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    return (np.asarray(matrix, dtype=np.float64) @ np.hstack([pts, ones]).T).T


def _region_from_mapped_corners(mapped: np.ndarray, *, page_no: int = 1) -> dict:
    """板坐标矩形四角经变换后，取 PDF 轴对齐外接矩形（与 DAY002ori pcb_bbox_to_pdf_rect 一致）。"""
    x0, y0 = mapped.min(axis=0)
    x1, y1 = mapped.max(axis=0)
    return {
        "page_no": page_no,
        "x0": float(x0),
        "y0": float(y0),
        "x1": float(x1),
        "y1": float(y1),
    }


def pcb_bbox_to_pdf_rect(matrix: np.ndarray, x: float, y: float, w: float, h: float) -> dict:
    """板坐标矩形 (x,y,w,h) → PDF 轴对齐外接矩形。"""
    corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float64)
    mapped = apply_affine(matrix, corners)
    return _region_from_mapped_corners(mapped)


def body_size_mm(package_symbol: str | None, pin_count: int | None) -> tuple[float, float]:
    """与 DAY002ori board_align 一致：无 footprint bbox 时的典型封装尺寸（mm）。"""
    pkg = str(package_symbol or "").upper()
    pins = int(pin_count or 0)
    if "0402" in pkg:
        return 1.0, 0.5
    if "0603" in pkg:
        return 1.6, 0.8
    if "0805" in pkg:
        return 2.0, 1.2
    if "1206" in pkg:
        return 3.2, 1.6
    if "SOT" in pkg:
        return 3.0, 1.6
    if "SOP" in pkg or "SOIC" in pkg:
        return 6.0, 4.0
    if "TSSOP" in pkg:
        return 6.5, 4.4
    if "DIP" in pkg:
        return 10.0, 7.0
    if "BGA" in pkg or pins >= 200:
        return 19.0, 19.0
    if pins >= 100:
        return 16.0, 16.0
    if pins >= 48:
        return 12.0, 12.0
    if pins >= 16:
        return 8.0, 6.0
    if pins >= 8:
        return 5.0, 4.0
    return 4.0, 3.0


def board_footprint_to_pdf_rect(
    matrix: np.ndarray,
    pcb_x: float,
    pcb_y: float,
    bbox: dict,
) -> dict | None:
    """器件板坐标原点 + footprint bbox → PDF 高亮矩形。"""
    minx = float(bbox.get("minx") or bbox.get("bbox_min_x") or 0)
    miny = float(bbox.get("miny") or bbox.get("bbox_min_y") or 0)
    maxx = float(bbox.get("maxx") or bbox.get("bbox_max_x") or 0)
    maxy = float(bbox.get("maxy") or bbox.get("bbox_max_y") or 0)
    if maxx <= minx or maxy <= miny:
        return None
    corners = np.array(
        [
            [pcb_x + minx, pcb_y + miny],
            [pcb_x + maxx, pcb_y + miny],
            [pcb_x + maxx, pcb_y + maxy],
            [pcb_x + minx, pcb_y + maxy],
        ],
        dtype=np.float64,
    )
    mapped = apply_affine(matrix, corners)
    return _region_from_mapped_corners(mapped, page_no=1)
