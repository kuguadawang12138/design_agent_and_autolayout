# -*- coding: utf-8 -*-
"""
OA 定时拉 BRD（在 ingest_service 目录运行）。

用法:
  python run_oa_scheduler.py
  python run_oa_scheduler.py --once incremental
  python run_oa_scheduler.py --once history
  python run_oa_scheduler.py --start-date 2020-08-15 --end-date 2020-08-16
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger

from agent_log import log_line
from oa_sync_jobs import (
    incremental_recent_range,
    run_history_backfill_loop,
    run_history_chunk,
    run_incremental_recent,
    sync_project_codes,
)
from settings import (
    bootstrap,
    find_config_path,
    get_agent_paths,
    get_oa_scheduler_config,
    load_deploy_env,
)


def _history_finished(result) -> bool:
    return bool(getattr(result, "history_complete", False)) or (
        result.skipped and "历史回补已完成" in (result.message or "")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="OA BRD 定时拉取")
    parser.add_argument("--once", choices=("incremental", "history"), help="只执行一次后退出")
    parser.add_argument("--start-date", help="OA 接口① start_date（含），与 --end-date 联测 OA")
    parser.add_argument("--end-date", help="OA 接口① end_date（不含当天，联测用）")
    args = parser.parse_args()

    bootstrap()

    if args.start_date or args.end_date:
        if not (args.start_date and args.end_date):
            parser.error("请同时提供 --start-date 与 --end-date")
        sched = get_oa_scheduler_config()
        r = sync_project_codes(
            args.start_date,
            args.end_date,
            max_projects=sched.get("max_projects_per_run"),
            job_name="cli_test",
            dry_run=True,
        )
        print(r.to_dict())
        sys.exit(0 if r.fail_count == 0 else 1)

    sched = get_oa_scheduler_config()
    hb = sched.get("history_backfill") or {}
    cfg_path = find_config_path()
    env = load_deploy_env()
    tz = str(sched["timezone"])
    run_day = datetime.now(ZoneInfo(tz)).date()
    inc_start, inc_end = incremental_recent_range(run_day)
    log_line(
        "OA流程",
        result=(
            f"【调度配置】环境={env} 配置文件={cfg_path} | "
            f"历史 YAML={hb.get('start_date')}~{hb.get('end_date')} chunk_days={hb.get('chunk_days', 1)} | "
            f"增量 enabled={sched.get('incremental_enabled')} "
            f"每{sched.get('incremental_interval_days')}天 拉前{sched.get('incremental_lookback_days')}天 | "
            f"增量示例窗口（按今天 {run_day}）{inc_start}~{inc_end}（不含 end，不读历史库表）"
        ),
    )

    if args.once == "incremental":
        r = run_incremental_recent()
        print(r.to_dict())
        sys.exit(0 if r.fail_count == 0 and not r.skipped else 1)
    if args.once == "history":
        r = run_history_chunk()
        print(r.to_dict())
        sys.exit(0 if r.fail_count == 0 and not r.skipped else 1)

    scheduler = BlockingScheduler(timezone=tz)
    stop_event = threading.Event()
    incremental_on = bool(sched.get("incremental_enabled"))

    def _request_shutdown(reason: str) -> None:
        stop_event.set()
        log_line("OA流程", result=f"【调度停止】{reason}")
        if scheduler.running:
            scheduler.shutdown(wait=False)

    def _run_incremental_safe() -> None:
        try:
            run_incremental_recent()
        except Exception as exc:
            log_line("OA流程", result=f"【增量任务异常】{exc}", level="error")

    def _run_history_loop() -> None:
        try:
            r = run_history_backfill_loop(stop_check=stop_event.is_set)
            if r and _history_finished(r):
                if incremental_on:
                    log_line("OA流程", result="【历史回补】YAML 计划已跑完，增量定时任务继续运行")
                    return
                _request_shutdown("YAML 历史回补计划已全部跑完，且未开增量，进程退出")
        except Exception as exc:
            log_line("OA流程", result=f"【历史回补异常】{exc}", level="error")

    def _on_signal(signum, _frame):
        name = "SIGINT(Ctrl+C)" if signum == signal.SIGINT else f"信号{signum}"
        _request_shutdown(f"收到 {name}，正在停止定时任务")

    signal.signal(signal.SIGINT, _on_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _on_signal)

    history_thread: threading.Thread | None = None
    if sched.get("history_enabled"):
        history_thread = threading.Thread(target=_run_history_loop, name="oa_history_backfill", daemon=True)
        history_thread.start()
        hist_desc = (
            f"连续回补 chunk_days={hb.get('chunk_days', 1)} "
            f"YAML={hb.get('start_date')}~{hb.get('end_date')}（走 oa_sync_progress）"
        )
    else:
        hist_desc = "disabled"

    inc_desc = "disabled"
    if incremental_on:
        interval_days = max(1, int(sched.get("incremental_interval_days", 2) or 2))
        hour, minute = 2, 0
        cron = str(sched.get("incremental_cron") or "0 2 * * *").split()
        if len(cron) >= 2:
            try:
                minute = int(cron[0])
                hour = int(cron[1])
            except ValueError:
                pass
        now = datetime.now(ZoneInfo(tz))
        start_at = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if start_at <= now:
            start_at = start_at + timedelta(days=1)
        scheduler.add_job(
            _run_incremental_safe,
            IntervalTrigger(days=interval_days, start_date=start_at, timezone=tz),
            id="oa_incremental_recent",
            name="OA 增量拉取",
            replace_existing=True,
        )
        inc_desc = (
            f"每{interval_days}天 {hour:02d}:{minute:02d} "
            f"按运行日拉前{sched.get('incremental_lookback_days')}天（常年不关）"
        )

    if not sched.get("history_enabled") and not incremental_on:
        print("oa_scheduler：history 与 incremental 均未启用，无任务可跑")
        sys.exit(1)

    brd_dir = get_agent_paths()["brd"]
    log_line(
        "OA流程",
        result=f"【调度启动】历史={hist_desc} 增量={inc_desc} BRD目录={brd_dir}",
    )
    print(f"[{datetime.now().isoformat()}] OA scheduler history={hist_desc} incremental={inc_desc} brd={brd_dir}")

    try:
        if incremental_on:
            scheduler.start()
        elif history_thread is not None:
            while history_thread.is_alive() and not stop_event.is_set():
                history_thread.join(timeout=1.0)
    except (KeyboardInterrupt, SystemExit):
        stop_event.set()
    if history_thread is not None and history_thread.is_alive():
        history_thread.join(timeout=3.0)
    log_line("OA流程", result="【调度结束】进程已退出")


if __name__ == "__main__":
    main()