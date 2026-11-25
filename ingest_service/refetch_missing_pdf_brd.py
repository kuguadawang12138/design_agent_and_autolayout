# -*- coding: utf-8 -*-
"""正面或反面 PDF 缺任意一份时，重新从 OA 把 BRD 拉下来。

每项目应有正面、反面各一份。缺一面也算缺，两面都在的不下载。
BRD 落到 data/brd，供二开再解析。

用法（在 ingest_service 目录）：

    python refetch_missing_pdf_brd.py
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

_TZ = ZoneInfo("Asia/Shanghai")


def _say(text: str) -> None:
    """只打到控制台，给 nohup 看。不写日志文件。"""
    now = datetime.now(_TZ).strftime("%Y-%m-%d %H:%M:%S")
    print(f"{now} {text}", flush=True)


def list_projects_missing_pdf(conn) -> list[dict]:
    """缺正面或反面 PDF 的项目。file_content 为空也算缺。"""
    sql = """
        SELECT codes.project_code,
               MAX(CASE WHEN pdf.top_side = 1 AND pdf.file_content IS NOT NULL AND pdf.file_content <> '' THEN 1 ELSE 0 END) AS has_top,
               MAX(CASE WHEN pdf.top_side = 0 AND pdf.file_content IS NOT NULL AND pdf.file_content <> '' THEN 1 ELSE 0 END) AS has_bottom
        FROM (
            SELECT project_code COLLATE utf8mb4_unicode_ci AS project_code
            FROM footprint_library
            UNION
            SELECT project_code COLLATE utf8mb4_unicode_ci
            FROM project_profile
            UNION
            SELECT project_code COLLATE utf8mb4_unicode_ci
            FROM project_pdf
        ) codes
        LEFT JOIN project_pdf pdf
            ON pdf.project_code COLLATE utf8mb4_unicode_ci = codes.project_code
        WHERE codes.project_code IS NOT NULL AND codes.project_code <> ''
        GROUP BY codes.project_code
        HAVING has_top = 0 OR has_bottom = 0
        ORDER BY codes.project_code
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
    out = []
    for project_code, has_top, has_bottom in rows:
        missing = []
        if not has_top:
            missing.append("正面")
        if not has_bottom:
            missing.append("反面")
        out.append(
            {
                "project_code": str(project_code).strip(),
                "missing": missing,
            }
        )
    return out


def main() -> None:
    from settings import bootstrap

    info = bootstrap()
    started = time.perf_counter()
    _say(f"[缺PDF拉BRD] 开始 配置={info['config_path']}")
    _say("[缺PDF拉BRD] 规则 正面或反面缺任意一份就拉取，两面都有的不拉取")

    from agent_oa import run_oa_pipeline
    from footprint_db import connect

    pulled: list[str] = []
    failed: list[str] = []

    conn = connect()
    try:
        projects = list_projects_missing_pdf(conn)
    finally:
        conn.close()

    total = len(projects)
    _say(f"[缺PDF拉BRD] 清单 缺PDF的项目={total} 个")
    for index, item in enumerate(projects, start=1):
        code = item["project_code"]
        missing = "、".join(item["missing"])
        present = "、".join(side for side in ("正面", "反面") if side not in item["missing"]) or "无"
        _say(f"[缺PDF拉BRD] 清单 {index}/{total} 项目={code} 缺={missing} 已有={present}")

    for index, item in enumerate(projects, start=1):
        code = item["project_code"]
        missing = "、".join(item["missing"])
        head = f"[缺PDF拉BRD] {index}/{total} 项目={code} 缺={missing}"
        t0 = time.perf_counter()
        _say(f"{head} 结果=开始下载")
        try:
            result = run_oa_pipeline(code, save_meta=True, record_failures=True)
            brd = (result or {}).get("brd") or {}
            file_name = str(brd.get("file_name") or "-")
            brd_path = str(brd.get("brd_path") or "")
            size = "-"
            if brd_path:
                path = Path(brd_path)
                if path.is_file():
                    size = str(path.stat().st_size)
            spent = int((time.perf_counter() - t0) * 1000)
            pulled.append(f"项目={code} 缺={missing} 文件={file_name} 路径={brd_path or '-'} 字节={size}")
            _say(
                f"{head} 结果=成功 耗时={spent}ms 文件={file_name} "
                f"路径={brd_path or '-'} 字节={size}"
            )
        except Exception as exc:
            spent = int((time.perf_counter() - t0) * 1000)
            reason = f"{type(exc).__name__}: {exc}"
            failed.append(f"项目={code} 缺={missing} 原因={reason}")
            _say(f"{head} 结果=失败 耗时={spent}ms 原因={reason}")

    elapsed = int((time.perf_counter() - started) * 1000)
    _say(
        f"[缺PDF拉BRD] 结束 清单={total} 成功={len(pulled)} "
        f"失败={len(failed)} 耗时={elapsed}ms"
    )
    if pulled:
        _say("[缺PDF拉BRD] 成功明细")
        for index, item in enumerate(pulled, start=1):
            _say(f"[缺PDF拉BRD] 成功 {index}. {item}")
    if failed:
        _say("[缺PDF拉BRD] 失败明细")
        for index, item in enumerate(failed, start=1):
            _say(f"[缺PDF拉BRD] 失败 {index}. {item}")


if __name__ == "__main__":
    main()
