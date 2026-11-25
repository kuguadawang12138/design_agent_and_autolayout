# -*- coding: utf-8 -*-
"""OA 拉取 BRD 定时任务：按时间窗查 project_code → 下载 BRD 到 data/brd/（与二开 /api/v1/brd/next 同目录）。"""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from agent_log import log_line
from agent_oa import OaDownloadParamsEmpty, fetch_project_codes, run_oa_pipeline
from footprint_db import (
    OA_SYNC_JOB_HISTORY,
    connect,
    fetch_oa_sync_done_through,
    upsert_oa_sync_done_through,
)
from oa_plan_dates import format_yaml_plan_log
from settings import (
    get_agent_paths,
    get_oa_config,
    get_oa_scheduler_config,
    resolve_history_plan_range,
)


@dataclass
class JobResult:
    job: str
    start_date: str
    end_date: str
    total_codes: int = 0
    ok_count: int = 0
    fail_count: int = 0
    skip_count: int = 0
    failures: list[dict] = field(default_factory=list)
    skipped: bool = False
    message: str = ""
    history_complete: bool = False

    def to_dict(self) -> dict:
        return {
            "job": self.job,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "total_codes": self.total_codes,
            "ok_count": self.ok_count,
            "fail_count": self.fail_count,
            "skip_count": self.skip_count,
            "failures": self.failures,
            "skipped": self.skipped,
            "message": self.message,
            "history_complete": self.history_complete,
        }


def _oa_ready() -> None:
    cfg = get_oa_config()
    if not cfg["enabled"]:
        raise RuntimeError("OA 未启用，请在 config/footprint_sim.yaml 设置 oa.enabled=true")


def week_range_before_sunday(sunday: date) -> tuple[str, str]:
    """自然周 Sun~Sat；OA end_date 不含当天。"""
    if sunday.weekday() != 6:
        raise ValueError(f"增量周任务 reference 应为周日，当前={sunday.isoformat()}")
    saturday = sunday - timedelta(days=1)
    week_sunday = saturday - timedelta(days=6)
    start = week_sunday.isoformat()
    end = (saturday + timedelta(days=1)).isoformat()
    return start, end


def parse_project_codes_arg(raw: str) -> list[str]:
    """解析命令行项目编码：单个或多个，多个用 & 分隔。"""
    seen: set[str] = set()
    ordered: list[str] = []
    for part in str(raw or "").split("&"):
        code = part.strip()
        if not code or code in seen:
            continue
        seen.add(code)
        ordered.append(code)
    return ordered


def _run_oa_pipeline_batch(
    codes: list[str],
    *,
    job_name: str,
    dry_run: bool,
    start_date: str = "-",
    end_date: str = "-",
) -> JobResult:
    """对给定 project_code 列表逐个跑 OA 接口②③（项目之间不额外 sleep）。"""
    result = JobResult(job=job_name, start_date=start_date, end_date=end_date, total_codes=len(codes))
    for code in codes:
        try:
            run_oa_pipeline(
                code,
                save_meta=not dry_run,
                record_failures=not dry_run,
            )
            result.ok_count += 1
        except OaDownloadParamsEmpty as exc:
            result.skip_count += 1
            result.failures.append(
                {"project_code": code, "error": str(exc).strip(), "phase": "empty_download"},
            )
        except Exception as exc:
            result.fail_count += 1
            err = str(exc).strip() or exc.__class__.__name__
            log_line(
                "OA流程",
                result=f"【项目拉取失败】project_code={code} {err}",
                level="warning",
            )
            result.failures.append(
                {
                    "project_code": code,
                    "error": err,
                    "phase": "pipeline",
                    "trace": traceback.format_exc(limit=3),
                }
            )
    return result


def retry_oa_projects(projects_arg: str, *, delay_sec: float | None = None) -> JobResult:
    """
    失败项目重试：OA 接口② + ③，写 project_profile、process、ingest_failure（如有）、data/brd/。
    projects_arg：一个或多个项目编码，& 分隔。
    delay_sec 保留兼容旧脚本，已不再用于项目间 sleep。
    """
    del delay_sec
    _oa_ready()
    codes = parse_project_codes_arg(projects_arg)
    if not codes:
        return JobResult(
            job="oa_retry",
            start_date="-",
            end_date="-",
            skipped=True,
            message="未提供有效项目编码（多个请用 & 分隔）",
        )
    log_line(
        "OA流程",
        result=f"【失败重试】开始 | 项目编码={' & '.join(codes)} | 共 {len(codes)} 个",
    )
    result = _run_oa_pipeline_batch(
        codes,
        job_name="oa_retry",
        dry_run=False,
    )
    result.message = (
        f"失败重试 | 共 {result.total_codes} 项 | 成功 {result.ok_count} | "
        f"跳过 {result.skip_count} | 失败 {result.fail_count}"
    )
    log_line("OA流程", result=f"【批次汇总】{result.message}", level="warning" if result.fail_count else "info")
    _append_job_report(result)
    return result


def sync_project_codes(
    start_date: str,
    end_date: str,
    *,
    max_projects: int | None = None,
    job_name: str = "oa_sync",
    dry_run: bool = False,
) -> JobResult:
    """拉取时间窗内 project_code 并逐个下载 BRD 到 agent_paths.brd。dry_run=True 不写 project_profile/ingest_failure。"""
    _oa_ready()
    result = JobResult(job=job_name, start_date=start_date, end_date=end_date)
    try:
        codes = fetch_project_codes(start_date=start_date, end_date=end_date)
    except Exception as exc:
        result.message = f"接口①拉取项目编码失败：{exc}"
        result.fail_count = 1
        result.failures.append({"project_code": "", "error": result.message, "phase": "list_codes"})
        log_line("OA流程", result=f"【接口①失败】{result.message}", level="error")
        return result

    if max_projects and max_projects > 0:
        codes = codes[: max_projects]

    result = _run_oa_pipeline_batch(
        codes,
        job_name=job_name,
        dry_run=dry_run,
        start_date=start_date,
        end_date=end_date,
    )
    result.message = (
        f"任务={job_name} 日期={start_date}~{end_date} | "
        f"共 {result.total_codes} 项 | BRD 成功 {result.ok_count} | "
        f"跳过 {result.skip_count} | 失败 {result.fail_count}"
    )
    log_line("OA流程", result=f"【批次汇总】{result.message}", level="warning" if result.fail_count else "info")
    _append_job_report(result)
    return result


def incremental_recent_range(reference: date | None = None, lookback_days: int | None = None) -> tuple[str, str]:
    """增量：仅按「本次运行日」往前数 N 天。不用 history_backfill 起止，也不读 oa_sync_progress。"""
    from zoneinfo import ZoneInfo

    sched = get_oa_scheduler_config()
    if reference is None:
        tz = ZoneInfo(str(sched.get("timezone") or "Asia/Shanghai"))
        from datetime import datetime

        reference = datetime.now(tz).date()
    days = int(lookback_days if lookback_days is not None else sched.get("incremental_lookback_days", 2))
    if days < 1:
        days = 1
    start = (reference - timedelta(days=days)).isoformat()
    end = reference.isoformat()
    return start, end


def run_incremental_recent(*, reference: date | None = None) -> JobResult:
    """定时增量：每次触发都用当天日期算窗口，常驻调度不会随历史回补结束而停。"""
    start, end = incremental_recent_range(reference)
    sched = get_oa_scheduler_config()
    from zoneinfo import ZoneInfo
    from datetime import datetime

    run_day = reference or datetime.now(ZoneInfo(str(sched.get("timezone") or "Asia/Shanghai"))).date()
    log_line(
        "OA流程",
        result=(
            f"【增量窗口】运行日={run_day} OA接口① {start}~{end}（end 不含当天） "
            f"lookback={sched.get('incremental_lookback_days')}天 | 不读历史YAML/oa_sync_progress"
        ),
    )
    return sync_project_codes(
        start,
        end,
        max_projects=sched.get("max_projects_per_run"),
        job_name="incremental_recent",
    )


def run_incremental_week(*, reference: date | None = None) -> JobResult:
    """兼容旧名字，实际走 incremental_recent。"""
    return run_incremental_recent(reference=reference)


def _parse_date(text: str) -> date:
    return date.fromisoformat(str(text).strip()[:10])


def _done_through_in_yaml_plan(done_through: str | None, plan_start: date, plan_end: date) -> bool:
    """库内 done_through 是否落在当前 YAML 计划闭区间内。"""
    if not done_through:
        return False
    dt = _parse_date(done_through)
    return plan_start <= dt <= plan_end


def _legacy_state_path() -> Path:
    raw = "data/oa/history_backfill_state.yaml"
    root = get_agent_paths()["root"]
    path = Path(str(raw).replace("\\", "/"))
    if not path.is_absolute():
        path = root / path
    return path


def _import_legacy_yaml_done_through(conn, job_key: str, plan_start: date) -> str | None:
    """一次性：从旧版 YAML 的 next_start 推算 done_through。"""
    path = _legacy_state_path()
    if not path.is_file():
        return None
    state = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(state, dict):
        return None
    next_start = state.get("next_start")
    if not next_start:
        return None
    ns = _parse_date(str(next_start))
    if ns <= plan_start:
        return None
    done = (ns - timedelta(days=1)).isoformat()
    upsert_oa_sync_done_through(conn, job_key, done)
    log_line("oa_sync_progress_import", detail=f"job={job_key} from={path}", result=f"done_through={done}")
    return done


def run_history_backfill_loop(
    *,
    stop_check=None,
    disk_threshold: float | None = None,
    disk_wait_sec: float | None = None,
) -> JobResult | None:
    """
    历史回补连续跑：未跑完 YAML 计划前按 chunk_days 一片接一片拉取；
    BRD 目录所在磁盘占用超过阈值则等待后重试（默认 70% / 120 秒）。
    """
    import time

    from oa_brd_disk import disk_usage_ratio_for_path

    sched = get_oa_scheduler_config()
    hb = sched.get("history_backfill") or {}
    threshold = float(disk_threshold if disk_threshold is not None else hb.get("disk_usage_max_ratio", 0.70))
    wait_sec = float(disk_wait_sec if disk_wait_sec is not None else hb.get("disk_wait_seconds", 120))
    brd_dir = get_agent_paths()["brd"]
    last: JobResult | None = None

    while True:
        if stop_check and stop_check():
            log_line("OA流程", result="【历史回补】收到停止信号，连续回补退出")
            return last

        ratio = disk_usage_ratio_for_path(brd_dir)
        if ratio > threshold:
            log_line(
                "OA流程",
                result=(
                    f"【历史回补】BRD 目录所在磁盘占用 {ratio * 100:.1f}% "
                    f"> {threshold * 100:.0f}%，等待 {wait_sec:.0f} 秒后继续"
                ),
                level="warning",
            )
            slept = 0.0
            while slept < wait_sec:
                if stop_check and stop_check():
                    return last
                step = min(2.0, wait_sec - slept)
                time.sleep(step)
                slept += step
            continue

        last = run_history_chunk()
        if getattr(last, "history_complete", False) or (
            last.skipped and "历史回补已完成" in (last.message or "")
        ):
            return last
        if last.skipped and not getattr(last, "history_complete", False):
            return last
        from oa_api_pace import sleep_oa_api_interval

        sleep_oa_api_interval(reason="本片日期拉取完成，下一自然日")


def run_history_chunk(*, force: bool = False) -> JobResult:
    """
    历史回补：每次推进 chunk_days（默认 1 天）。
    进度仅写 oa_sync_progress.done_through（已做完的最后日期，含）。
    """
    sched = get_oa_scheduler_config()
    hb = sched.get("history_backfill") or {}
    if not hb.get("enabled", True) and not force:
        return JobResult(
            job="history_chunk",
            start_date="-",
            end_date="-",
            skipped=True,
            message="history_backfill.enabled=false",
        )

    job_key = str(hb.get("job_key") or OA_SYNC_JOB_HISTORY).strip()
    plan_start, plan_end = resolve_history_plan_range(hb)
    if plan_start > plan_end:
        msg = f"YAML 历史回补日期非法：start_date={plan_start} 须早于 end_date={plan_end}"
        log_line("OA流程", result=f"【历史回补配置】{msg}", level="error")
        return JobResult(
            job="history_chunk",
            start_date=plan_start.isoformat(),
            end_date=plan_end.isoformat(),
            skipped=True,
            message=msg,
        )
    year = plan_start.year
    chunk_days = max(1, int(hb.get("chunk_days", 1)))

    conn = connect()
    try:
        done_through = fetch_oa_sync_done_through(conn, job_key)
        if done_through is None:
            done_through = _import_legacy_yaml_done_through(conn, job_key, plan_start)

        raw_done = done_through
        if done_through and not _done_through_in_yaml_plan(done_through, plan_start, plan_end):
            log_line(
                "OA流程",
                result=(
                    f"【历史回补】库内 done_through={done_through} 不在 YAML 计划 "
                    f"{plan_start}~{plan_end} 内，忽略该进度，从 YAML 起点执行"
                ),
                level="warning",
            )
            done_through = None

        if done_through and _parse_date(done_through) >= plan_end:
            log_line(
                "OA流程",
                result=(
                    f"【历史回补】YAML 计划 {plan_start}~{plan_end} 已全部完成，"
                    f"库内 done_through={done_through}"
                ),
            )
            return JobResult(
                job=f"history_{year}",
                start_date=plan_start.isoformat(),
                end_date=plan_end.isoformat(),
                skipped=True,
                message=f"历史回补已完成 done_through={done_through}",
                history_complete=True,
            )

        if done_through:
            chunk_start = _parse_date(done_through) + timedelta(days=1)
        else:
            chunk_start = plan_start
        if chunk_start < plan_start:
            chunk_start = plan_start

        if chunk_start > plan_end:
            upsert_oa_sync_done_through(conn, job_key, plan_end.isoformat())
            return JobResult(
                job=f"history_{year}",
                start_date=plan_start.isoformat(),
                end_date=plan_end.isoformat(),
                skipped=True,
                message=f"历史回补已完成 done_through={plan_end.isoformat()}",
                history_complete=True,
            )

        chunk_end_inclusive = min(plan_end, chunk_start + timedelta(days=chunk_days - 1))
        oa_start = chunk_start.isoformat()
        oa_end = (chunk_end_inclusive + timedelta(days=1)).isoformat()

        log_line(
            "OA流程",
            result=(
                f"【历史回补】{format_yaml_plan_log(plan_start, plan_end, oa_start, oa_end)} | "
                f"库内进度 done_through={raw_done or '无'}"
            ),
        )

        max_p = hb.get("max_projects_per_run") or sched.get("max_projects_per_run")

        result = sync_project_codes(
            oa_start,
            oa_end,
            max_projects=max_p,
            job_name=f"history_{year}",
        )

        # 本片日期跑完即推进进度（与单个项目成败无关；YAML 计划区间以 oa_sync_progress 为准）
        upsert_oa_sync_done_through(conn, job_key, chunk_end_inclusive.isoformat())
        result.message += (
            f" | done_through={chunk_end_inclusive.isoformat()}（本片已推进，"
            f"成功={result.ok_count} 跳过={result.skip_count} 失败={result.fail_count}）"
        )
        if chunk_end_inclusive >= plan_end:
            result.history_complete = True
            result.message += " | 历史回补计划已全部跑完"
        return result
    finally:
        conn.close()


def _append_job_report(result: JobResult) -> None:
    logs = get_agent_paths()["logs"]
    logs.mkdir(parents=True, exist_ok=True)
    path = logs / "oa_scheduler_jobs.jsonl"
    row = {"time": datetime.now().isoformat(timespec="seconds"), **result.to_dict()}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
