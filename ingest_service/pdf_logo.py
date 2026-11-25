# -*- coding: utf-8 -*-
"""在每个器件中心贴一块淡水印。

用入库服务根目录的 logo2.ai。宽 40 点，旋转 45 度，透明度 0.22。
底是透明的，不挡住器件。有坐标就用坐标，没有再在 PDF 里找 Eda_ 位号。
"""

from __future__ import annotations

import math
import re
from pathlib import Path

_MARK = "device-logo"
_LOGO_WIDTH = 40.0
_OPACITY = 0.22


def _logo_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "公司logo"


def resolve_logo_path(logo_path: str | Path | None = None) -> Path:
    """指定路径优先。默认用入库服务根目录的 logo2.ai。"""
    if logo_path:
        path = Path(logo_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"水印文件不存在：{path}")
        return path.resolve()
    here = Path(__file__).resolve().parent
    for item in (here / "logo2.ai", _logo_dir() / "logo2.ai"):
        if item.is_file():
            return item.resolve()
    raise FileNotFoundError("未找到 logo2.ai，请放在入库服务根目录")


def has_company_logo(pdf_bytes: bytes | None) -> bool:
    """看 PDF 关键字里有没有按器件贴过水印。"""
    if not pdf_bytes:
        return False
    import pymupdf

    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        return False
    try:
        keywords = str((doc.metadata or {}).get("keywords") or "")
        return _MARK in keywords
    finally:
        doc.close()


def device_centers_on_page(page) -> list[tuple[float, float]]:
    """这一页上每个 Eda_ 位号文字的中心。"""
    centers: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    for block in (page.get_text("dict") or {}).get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines") or []:
            text = "".join(span.get("text") or "" for span in line.get("spans") or [])
            if "eda_" not in text.lower():
                continue
            x0, y0, x1, y1 = line["bbox"]
            key = (round((x0 + x1) / 2.0, 1), round((y0 + y1) / 2.0, 1))
            if key in seen:
                continue
            seen.add(key)
            centers.append(key)
    return centers


def _silk_page_index(doc) -> int:
    if doc.page_count <= 1:
        return 0
    best_i = 0
    best_n = -1
    for index in range(doc.page_count):
        count = (doc[index].get_text("text") or "").lower().count("eda_")
        if count > best_n:
            best_n = count
            best_i = index
    return best_i if best_n > 0 else 0


def _form_name(doc, page) -> tuple[int, str]:
    kind, value = doc.xref_get_key(page.xref, "Resources")
    text = doc.xref_object(int(value.split()[0])) if kind == "xref" else value
    match = re.search(r"/XObject\s*<<[^>]*?/(\w+)\s+(\d+)\s+0\s+R", text, re.S)
    if not match:
        raise RuntimeError("logo 没有变成可重复使用的矢量对象")
    return int(match.group(2)), match.group(1)


def _as_transparent_group(doc, form_xref: int) -> None:
    """整块标志先画实，再按 0.22 叠到板上。标志内部的不透明不会盖住器件。"""
    doc.xref_set_key(
        form_xref,
        "Group",
        "<< /S /Transparency /CS /DeviceRGB /I false >>",
    )


def _matrix(cx: float, cy: float, logo_w: float, logo_h: float) -> str:
    """把 logo 缩放到 40 点宽，绕自身中心转 45 度，再放到器件中心。"""
    scale = _LOGO_WIDTH / logo_w
    angle = math.radians(45.0)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    a = scale * cos_a
    b = scale * sin_a
    c = -scale * sin_a
    d = scale * cos_a
    ox = logo_w / 2.0
    oy = logo_h / 2.0
    e = cx - (a * ox + c * oy)
    f = cy - (b * ox + d * oy)
    return f"{a:.6f} {b:.6f} {c:.6f} {d:.6f} {e:.3f} {f:.3f}"


def _merge_resource(doc, page, key: str, entry: str) -> None:
    """往页面资源的某个字典里加一项。原来没有这个字典就新建。"""
    kind, value = doc.xref_get_key(page.xref, "Resources")
    if kind == "xref":
        res_xref = int(value.split()[0])
        res = doc.xref_object(res_xref)
    else:
        res_xref = 0
        res = value
    found = re.search(rf"/{key}\s*<<", res)
    if found:
        res = res[: found.end()] + f" {entry} " + res[found.end() :]
    else:
        res = res.rstrip()
        if res.endswith(">>"):
            res = res[:-2].rstrip() + f" /{key} << {entry} >> >>"
        else:
            res = f"<< /{key} << {entry} >> >>"
    if res_xref:
        doc.update_object(res_xref, res)
    else:
        doc.xref_set_key(page.xref, "Resources", res)


def _add_xobject(doc, page, form_xref: int, form_name: str) -> None:
    """把 logo 矢量对象挂进页面资源。页面原来没有 XObject 时也要补上。"""
    _merge_resource(doc, page, "XObject", f"/{form_name} {form_xref} 0 R")


def _append_draws(doc, page, stream: bytes, form_xref: int, form_name: str, gs_xref: int) -> None:
    xref = doc.get_new_xref()
    doc.update_object(xref, "<< /Length 0 >>")
    doc.update_stream(xref, stream)
    contents = list(page.get_contents() or [])
    contents.append(xref)
    doc.xref_set_key(page.xref, "Contents", "[ " + " ".join(f"{item} 0 R" for item in contents) + " ]")
    _merge_resource(doc, page, "ExtGState", f"/GSlogo {gs_xref} 0 R")
    _add_xobject(doc, page, form_xref, form_name)


def stamp_company_logo(
    pdf_bytes: bytes,
    logo_path: str | Path | None = None,
    centers: list[tuple[float, float]] | None = None,
) -> bytes:
    """在位号页的每个器件中心贴一块水印。没有中心点就在该页查找。"""
    import pymupdf

    if not pdf_bytes or has_company_logo(pdf_bytes):
        return pdf_bytes
    src = pymupdf.open(resolve_logo_path(logo_path))
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        if doc.page_count < 1 or src.page_count < 1:
            return pdf_bytes
        page_index = _silk_page_index(doc)
        page = doc[page_index]
        points = list(centers) if centers else device_centers_on_page(page)
        if not points:
            return pdf_bytes
        logo = src[0]
        holder = doc.new_page(width=float(logo.rect.width), height=float(logo.rect.height))
        holder.show_pdf_page(holder.rect, src, 0)
        form_xref, form_name = _form_name(doc, holder)
        _as_transparent_group(doc, form_xref)
        gs_xref = doc.get_new_xref()
        doc.update_object(
            gs_xref,
            f"<< /Type /ExtGState /CA {_OPACITY} /ca {_OPACITY} >>",
        )
        target = doc[page_index]
        to_pdf = ~target.transformation_matrix
        parts = ["/GSlogo gs"]
        logo_w = float(logo.rect.width)
        logo_h = float(logo.rect.height)
        for cx, cy in points:
            point = pymupdf.Point(float(cx), float(cy)) * to_pdf
            parts.append(f"q {_matrix(point.x, point.y, logo_w, logo_h)} cm /{form_name} Do Q")
        _append_draws(
            doc,
            target,
            ("q " + "\n".join(parts) + " Q\n").encode("ascii"),
            form_xref,
            form_name,
            gs_xref,
        )
        doc.delete_page(holder.number)
        info = dict(doc.metadata or {})
        keywords = str(info.get("keywords") or "")
        if _MARK not in keywords:
            info["keywords"] = (keywords + " " + _MARK).strip()
            doc.set_metadata(info)
        return doc.tobytes(deflate=True, garbage=3)
    finally:
        doc.close()
        src.close()
