# -*- coding: utf-8 -*-
"""
把本地 PDF 录入 project_pdf（MinIO 开启时只在库里存路径），并按该面已有器件做校准。

在 ingest_service 目录执行:
  python run_ingest_local_pdf.py --pdf D:\\pdfs\\1139093_TOP.pdf
  python run_ingest_local_pdf.py --pdf D:\\pdfs\\1139093_BOTTOM.pdf --project 1139093
  python run_ingest_local_pdf.py --dir D:\\pdfs
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from agent_log import log_line
from agent_pdf import pdf_file_top_side, store_board_pdf
from footprint_db import connect, fetch_components_for_side, fetch_project_profile
from settings import bootstrap
from top_side_util import top_side_label

_CODE_SIDE = re.compile(r"^(?P<code>.+?)[_\-](?P<side>top|bottom|bot)$", re.IGNORECASE)


def _project_from_name(path: Path, override: str | None) -> str:
    if override:
        return override.strip()
    match = _CODE_SIDE.match(path.stem)
    if match:
        return match.group("code").strip()
    return path.stem.strip()


def _ingest_one(path: Path, project_code: str | None) -> int:
    if not path.is_file():
        print(f"[跳过] 不是文件: {path}")
        return 1
    if path.stat().st_size == 0:
        print(f"[跳过] 0KB 不入库: {path}")
        log_line("本地PDF", result=f"【跳过】file={path.name} 大小为 0KB", level="warning")
        return 1
    side = pdf_file_top_side(path.name)
    code = _project_from_name(path, project_code)
    if side is None:
        print(f"[失败] 文件名看不出 top/bottom: {path.name}（请使用 *_TOP.pdf 或 *_BOTTOM.pdf）")
        return 1
    if not code:
        print(f"[失败] 无法从文件名得到项目编码: {path.name}")
        return 1
    content = path.read_bytes()
    conn = connect()
    try:
        profile = fetch_project_profile(conn, code) or {}
        rows = fetch_components_for_side(conn, code, side)
        items = [
            {
                "refdes": row.get("refdes"),
                "pcb_x": row.get("pcb_x"),
                "pcb_y": row.get("pcb_y"),
                "bbox_min_x": row.get("bbox_min_x"),
                "bbox_min_y": row.get("bbox_min_y"),
                "bbox_max_x": row.get("bbox_max_x"),
                "bbox_max_y": row.get("bbox_max_y"),
                "bbox": row,
            }
            for row in rows
            if row.get("refdes")
        ]
        log_line(
            "本地PDF",
            result=(
                f"【开始录入】project={code} 面={top_side_label(side)} "
                f"file={path.name} 该面器件={len(items)} bytes={len(content)}"
            ),
        )
        warn = store_board_pdf(
            conn,
            project_code=code,
            top_side=side,
            file_name=path.name,
            content=content,
            calibration_items=items,
            sample_item=items[0] if items else None,
            finish_time=profile.get("finish_time"),
        )
    finally:
        conn.close()
    if warn:
        print(f"[已入库有告警] {code} {top_side_label(side)} {path.name} | {warn}")
        return 0
    print(f"[已入库] {code} {top_side_label(side)} {path.name} 器件={len(items)}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="录入本地板级 PDF")
    parser.add_argument("--pdf", action="append", default=[], help="单个 PDF，可重复")
    parser.add_argument("--dir", help="目录下全部 .pdf")
    parser.add_argument("--project", help="指定项目编码；不传则从文件名里 *_TOP / *_BOTTOM 前面截取")
    args = parser.parse_args()

    files: list[Path] = [Path(p) for p in args.pdf]
    if args.dir:
        folder = Path(args.dir)
        files.extend(sorted(folder.glob("*.pdf")))
        files.extend(sorted(folder.glob("*.PDF")))
    # 去重并保持顺序
    seen: set[str] = set()
    ordered: list[Path] = []
    for path in files:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(path)
    if not ordered:
        parser.error("请提供 --pdf 或 --dir")

    bootstrap()
    failed = 0
    for path in ordered:
        failed += _ingest_one(path, args.project)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
