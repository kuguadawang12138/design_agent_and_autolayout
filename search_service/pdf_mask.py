# -*- coding: utf-8 -*-
"""返回 PDF 时，高亮框周围留一块清晰区域，其余打码。不改库里的原文件。"""

from __future__ import annotations

import math


def keep_square(region: dict) -> dict:
    """长边 L 的正方形面积是 L²，保留区域面积取它的两倍，仍是正方形。"""
    x0 = float(region["x0"])
    y0 = float(region["y0"])
    x1 = float(region["x1"])
    y1 = float(region["y1"])
    long_edge = max(x1 - x0, y1 - y0, 1.0)
    side = long_edge * math.sqrt(2.0)
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    half = side / 2.0
    return {
        "x0": cx - half,
        "y0": cy - half,
        "x1": cx + half,
        "y1": cy + half,
        "page_no": int(region.get("page_no") or 1),
    }


def mask_page_outside(page, regions: list[dict] | None) -> None:
    """整页打成马赛克，再把每个高亮框对应的保留方块贴回原图。"""
    import numpy as np
    import pymupdf

    boxes = [keep_square(item) for item in (regions or []) if item]
    if not boxes:
        return
    pix = page.get_pixmap(alpha=False)
    channels = int(pix.n)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, channels).copy()
    block = 18
    small = arr[::block, ::block]
    mosaic = np.repeat(np.repeat(small, block, axis=0), block, axis=1)
    mosaic = mosaic[: arr.shape[0], : arr.shape[1]]
    scale_x = pix.w / max(float(page.rect.width), 1.0)
    scale_y = pix.h / max(float(page.rect.height), 1.0)
    for box in boxes:
        x0 = max(0, int((box["x0"] - page.rect.x0) * scale_x))
        y0 = max(0, int((box["y0"] - page.rect.y0) * scale_y))
        x1 = min(pix.w, int((box["x1"] - page.rect.x0) * scale_x))
        y1 = min(pix.h, int((box["y1"] - page.rect.y0) * scale_y))
        if x1 > x0 and y1 > y0:
            mosaic[y0:y1, x0:x1] = arr[y0:y1, x0:x1]
    out = pymupdf.Pixmap(pymupdf.csRGB, pix.w, pix.h, mosaic.tobytes(), False)
    page.insert_image(page.rect, pixmap=out, overlay=True)
