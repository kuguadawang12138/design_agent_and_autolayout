# -*- coding: utf-8 -*-
"""给库里已经有的板级 PDF 在每个器件中心贴水印，写回原记录。

水印固定用入库服务根目录的 logo2.ai。数据库只看 config/deploy_env.yaml。
已经按器件贴过的跳过。不加密。
在入库服务目录执行：

    python watermark_existing_pdfs.py
"""

from __future__ import annotations

import argparse


def _centers(conn, pdf_bytes: bytes, project_code: str, is_top: bool) -> list[tuple[float, float]] | None:
    """库里有 PDF 坐标就用坐标。缺的位号再在 PDF 上找。一张都没有就交给贴图函数自己找。"""
    import pymupdf

    from agent_pdf import _rect_center, _search_refdes_rect
    from pdf_logo import _silk_page_index
    from top_side_util import top_side_to_db

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT refdes, pdf_refdes_x, pdf_refdes_y
            FROM footprint_library
            WHERE project_code = %s AND top_side = %s
            """,
            (project_code, top_side_to_db(is_top)),
        )
        rows = cur.fetchall()
    known: list[tuple[float, float]] = []
    missing: list[str] = []
    for refdes, x_pos, y_pos in rows:
        if x_pos is not None and y_pos is not None:
            known.append((float(x_pos), float(y_pos)))
        elif refdes:
            missing.append(str(refdes))
    if known and not missing:
        return known
    if not missing and not known:
        return None
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        if doc.page_count < 1:
            return known or None
        page = doc[_silk_page_index(doc)]
        for refdes in missing:
            rect = _search_refdes_rect(page, refdes)
            if rect is not None:
                known.append(_rect_center(rect))
    finally:
        doc.close()
    return known or None


def main() -> None:
    argparse.ArgumentParser(description="给库内 PDF 的每个器件中心贴水印").parse_args()

    from agent_log import log_line
    from settings import bootstrap, get_mysql, load_deploy_env

    info = bootstrap()
    mysql = get_mysql()
    from pdf_logo import resolve_logo_path

    try:
        logo = resolve_logo_path()
    except FileNotFoundError as exc:
        log_line("PDF水印", result=f"【无法开始】{exc}", level="error")
        raise SystemExit(1) from exc
    log_line(
        "PDF水印",
        result=(
            f"【开始】env={load_deploy_env()} 配置={info['config_path']} "
            f"数据库={mysql.get('host')}:{mysql.get('port')}/{mysql.get('database')} 水印={logo}"
        ),
    )

    from footprint_db import connect, upsert_project_pdf
    from minio_store import resolve_stored_bytes
    from pdf_logo import has_company_logo, stamp_company_logo
    from top_side_util import top_side_from_db

    conn = connect()
    done: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT project_code, top_side, file_name, file_content, pdf_scale FROM project_pdf"
            )
            rows = cur.fetchall()
        total = len(rows)
        log_line("PDF水印", result=f"【待处理】库内 PDF 共 {total} 个")
        for index, (project_code, top_side, file_name, stored, pdf_scale) in enumerate(rows, start=1):
            label = f"{project_code} {file_name}"
            try:
                raw = resolve_stored_bytes(stored)
                if has_company_logo(raw):
                    skipped.append(label)
                    log_line("PDF水印", result=f"【跳过】{index}/{total} {label} 已有水印")
                    continue
                is_top = top_side_from_db(top_side)
                centers = _centers(conn, raw, str(project_code), is_top)
                stamped = stamp_company_logo(raw, logo_path=logo, centers=centers)
                upsert_project_pdf(
                    conn,
                    project_code=project_code,
                    top_side=top_side_from_db(top_side),
                    file_name=file_name,
                    content=stamped,
                    pdf_scale=float(pdf_scale or 0),
                )
                done.append(label)
                log_line("PDF水印", result=f"【完成】{index}/{total} {label}")
            except Exception as exc:
                failed.append(f"{label} {exc}")
                log_line(
                    "PDF水印",
                    result=f"【失败】{index}/{total} {label} {exc}",
                    level="error",
                )
    finally:
        conn.close()
    failed_text = "；".join(failed) if failed else "无"
    log_line(
        "PDF水印",
        result=(
            f"【结束】共 {len(done) + len(skipped) + len(failed)} 个，"
            f"新贴 {len(done)} 个，已有水印跳过 {len(skipped)} 个，失败 {len(failed)} 个。"
            f"失败明细：{failed_text}"
        ),
        level="error" if failed else "info",
    )


if __name__ == "__main__":
    main()
