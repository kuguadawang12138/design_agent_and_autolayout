# -*- coding: utf-8 -*-
"""定时补刷 footprint_library.pdf_refdes_x/y（单线程，依赖 project_pdf + PyMuPDF）。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from agent_log import log_line
from agent_pdf import index_single_refdes_pdf_position
from footprint_db import (
    connect,
    insert_pdf_refdes_backfill_log,
    list_components_missing_pdf_refdes,
)
from top_side_util import parse_top_side, top_side_label


@dataclass
class PdfRefdesBackfillResult:
    batch_key: str
    pending_total: int
    ok_count: int
    fail_count: int
    message: str

    def to_dict(self) -> dict:
        return {
            "batch_key": self.batch_key,
            "pending_total": self.pending_total,
            "ok_count": self.ok_count,
            "fail_count": self.fail_count,
            "message": self.message,
        }


def run_pdf_refdes_backfill_once(*, limit: int | None = None, force: bool = False) -> PdfRefdesBackfillResult:
    """扫描要重算 PDF 位号坐标的行。force 时已经有坐标的也重算。"""
    batch_key = datetime.now().strftime("%Y%m%d%H%M%S")
    conn = connect()
    try:
        pending = list_components_missing_pdf_refdes(conn, limit=limit, force=force)
        n = len(pending)
        log_line(
            "PDF位号补刷",
            result=f"【待补刷清单】共 {n} 条 force={int(force)}（只按器件 top_side 对应那一面 PDF）",
        )
        for row in pending:
            is_top = parse_top_side(row.get("top_side"))
            log_line(
                "PDF位号补刷",
                result=(
                    f"【待补刷】project_code={row.get('project_code')} "
                    f"refdes={row.get('refdes')} 面={top_side_label(is_top)}"
                ),
            )

        ok_count = 0
        fail_count = 0
        pdf_cache: dict = {}
        for row in pending:
            code = str(row.get("project_code") or "").strip()
            ref = str(row.get("refdes") or "").strip().upper()
            is_top = parse_top_side(row.get("top_side"))
            if not code or not ref:
                fail_count += 1
                continue
            try:
                if index_single_refdes_pdf_position(
                    conn,
                    project_code=code,
                    refdes=ref,
                    top_side=is_top,
                    cache=pdf_cache,
                ):
                    ok_count += 1
                else:
                    fail_count += 1
                    insert_pdf_refdes_backfill_log(
                        conn,
                        batch_key=batch_key,
                        row_kind="detail",
                        project_code=code,
                        refdes=ref,
                        top_side=is_top,
                        detail="PDF 上未匹配到位号丝印",
                    )
            except Exception as exc:
                fail_count += 1
                insert_pdf_refdes_backfill_log(
                    conn,
                    batch_key=batch_key,
                    row_kind="detail",
                    project_code=code,
                    refdes=ref,
                    top_side=is_top,
                    detail=str(exc)[:512],
                )

        summary_detail = f"共{n}条"
        insert_pdf_refdes_backfill_log(
            conn,
            batch_key=batch_key,
            row_kind="summary",
            pending_total=n,
            ok_count=ok_count,
            fail_count=fail_count,
            detail=summary_detail,
        )
        msg = f"待刷={n} 成功={ok_count} 失败={fail_count} batch={batch_key}"
        log_line("PDF位号补刷", result=f"【补刷完成】{msg}")
        return PdfRefdesBackfillResult(
            batch_key=batch_key,
            pending_total=n,
            ok_count=ok_count,
            fail_count=fail_count,
            message=msg,
        )
    finally:
        conn.close()
