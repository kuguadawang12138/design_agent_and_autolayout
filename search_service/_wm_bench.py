# -*- coding: utf-8 -*-
"""临时：比较 logo 叠到 PDF 的速度，跑完删除。"""
import time
from pathlib import Path

import pymupdf

ROOT = Path(r"d:\260805_pcbdesign")
LOGO = ROOT / "公司logo"
PDFS = [
    ROOT / "DAY002" / "1154918_top.pdf",
    ROOT / "DAY002" / "1154918_bottom.pdf",
    ROOT / "DAY002" / "1159932_top.pdf",
]
OUT = ROOT / "DAY002" / "_wm_preview.png"


def logo_rect(page):
    box = page.rect
    h = max(28.0, box.height * 0.045)
    w = h * (360 / 115)
    return pymupdf.Rect(box.x1 - w - 18, box.y1 - h - 14, box.x1 - 18, box.y1 - 14)


def stamp_svg(page, svg_doc):
    page.show_pdf_page(logo_rect(page), svg_doc, 0, overlay=True)


def stamp_png(page, png):
    page.insert_image(logo_rect(page), stream=png, overlay=True, keep_proportion=True)


def stamp_eps(page, eps_doc):
    page.show_pdf_page(logo_rect(page), eps_doc, 0, overlay=True)


def once(pdf_path, kind):
    doc = pymupdf.open(pdf_path)
    page = doc[0]
    t0 = time.perf_counter()
    if kind == "svg":
        src = pymupdf.open(LOGO / "公司logo.svg")
        stamp_svg(page, src)
        src.close()
    elif kind == "png":
        stamp_png(page, PNG)
    elif kind == "eps":
        src = pymupdf.open(LOGO / "公司logo.eps")
        stamp_eps(page, src)
        src.close()
    data = doc.tobytes(garbage=3, deflate=True)
    ms = (time.perf_counter() - t0) * 1000
    doc.close()
    return ms, len(data)


svg = pymupdf.open(LOGO / "公司logo.svg")
pix = svg[0].get_pixmap(matrix=pymupdf.Matrix(4, 4), alpha=True)
PNG = pix.tobytes("png")
svg.close()
print("png_bytes", len(PNG))

for kind in ("svg", "png", "eps"):
    times = []
    try:
        for pdf in PDFS:
            for _ in range(3):
                ms, size = once(pdf, kind)
                times.append((pdf.name, round(ms, 1), size))
        avg = sum(item[1] for item in times) / len(times)
        print(kind, "avg_ms", round(avg, 1), times)
    except Exception as exc:
        print(kind, "FAIL", type(exc).__name__, exc)

doc = pymupdf.open(PDFS[0])
page = doc[0]
stamp_png(page, PNG)
pix = page.get_pixmap(matrix=pymupdf.Matrix(0.35, 0.35), alpha=False)
pix.save(str(OUT))
print("preview", OUT, pix.width, pix.height)
doc.close()
