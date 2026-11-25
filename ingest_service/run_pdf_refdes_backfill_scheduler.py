# -*- coding: utf-8 -*-
"""
PDF 位号坐标补刷定时任务（单线程，默认每 15 分钟）。

用法:
  python run_pdf_refdes_backfill_scheduler.py
  python run_pdf_refdes_backfill_scheduler.py --once
  python run_pdf_refdes_backfill_scheduler.py --once --force
"""

from __future__ import annotations

import argparse
import signal
import sys

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger

from agent_log import log_line
from pdf_refdes_backfill_jobs import run_pdf_refdes_backfill_once
from settings import bootstrap


def main() -> None:
    parser = argparse.ArgumentParser(description="PDF pdf_refdes_x/y 补刷调度")
    parser.add_argument("--once", action="store_true", help="只执行一轮后退出")
    parser.add_argument("--force", action="store_true", help="坐标已经有值的也按新规则重算")
    parser.add_argument("--interval-minutes", type=int, default=15, help="定时间隔（分钟）")
    args = parser.parse_args()

    bootstrap()

    if args.once:
        r = run_pdf_refdes_backfill_once(force=args.force)
        print(r.to_dict())
        sys.exit(0)

    sched = BlockingScheduler()

    def _job() -> None:
        try:
            run_pdf_refdes_backfill_once()
        except Exception as exc:
            log_line("PDF位号补刷", result=f"【补刷异常】{exc}", level="error")

    sched.add_job(
        _job,
        IntervalTrigger(minutes=max(1, int(args.interval_minutes))),
        id="pdf_refdes_backfill",
        max_instances=1,
        coalesce=True,
    )

    def _stop(signum, frame) -> None:
        del signum, frame
        sched.shutdown(wait=False)

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    log_line(
        "PDF位号补刷",
        result=f"【调度启动】interval={args.interval_minutes}min 单线程 max_instances=1",
    )
    _job()
    sched.start()


if __name__ == "__main__":
    main()
