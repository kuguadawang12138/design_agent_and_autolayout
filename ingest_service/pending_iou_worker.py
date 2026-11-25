# -*- coding: utf-8 -*-
"""用多线程计算待完成板 JSON 和库中器件的 IoU。

从 pending_json 取出脚数超过 6 的器件，和 footprint_library 里同脚数的其它器件比轮廓 IoU，
结果写入 pending_iou。算完一条就从 pending_json 删掉这一条。
脚数不超过 6 的不计算，表里已有的会删掉。
算失败的项目编码和位号写入 pending_json_block，下一轮起跳过，JSON 仍留在 pending_json。

每个线程自己连数据库。线程内部不再开进程池，避免和 8 个线程叠在一起。

脚本启动后一直运行，每 10 秒看一次 pending_json。没有新器件就等待下一轮。
只有 Ctrl+C 或结束进程才会停。正在计算的这一批会收尾，还没开始的会取消。

用法（在 ingest_service 目录，先执行 python setup_db.py）：

    python pending_iou_worker.py
    python pending_iou_worker.py --threads 8
    python pending_iou_worker.py --config config/footprint_sim_st.yaml
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


DEFAULT_THREADS = 8
POLL_SECONDS = 10


def _as_text(value) -> str:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8")
    return str(value or "")


def _run_one(job: dict) -> str:
    """算一个位号。成功返回 ok，失败返回 fail。失败时 JSON 留着，并记入异常表。"""
    from agent_log import log_line
    from footprint_db import (
        connect,
        delete_pending_component_json,
        mark_pending_json_block,
        precompute_pending_iou,
    )

    code = job["project_code"]
    ref = job["refdes"]
    pins = job["pin_count"]
    log_line(
        "待完成IOU",
        result=(
            f"【开始】project_code={code} refdes={ref} 脚数={pins} "
            f"来源={job.get('source_file') or '-'} 表=pending_json"
        ),
    )
    conn = connect()
    try:
        written = precompute_pending_iou(
            conn,
            _as_text(job["json_text"]),
            source_file=job.get("source_file") or "pending.json",
            parallel=False,
        )
        delete_pending_component_json(conn, job["id"])
        log_line(
            "待完成IOU",
            result=(
                f"【完成】project_code={code} refdes={ref} 写入 pending_iou={written} "
                f"已从 pending_json 删除"
            ),
        )
        return "ok"
    except Exception as exc:
        try:
            conn.rollback()
        except Exception:
            pass
        reason = f"{type(exc).__name__}: {exc}"
        try:
            mark_pending_json_block(conn, code, ref, reason)
            blocked = "已记入 pending_json_block，下轮跳过"
        except Exception as mark_exc:
            blocked = f"异常表未写入 {type(mark_exc).__name__}"
        log_line(
            "待完成IOU",
            result=(
                f"【失败】project_code={code} refdes={ref} {reason} JSON保留 {blocked}"
            ),
            level="warning",
        )
        return "fail"
    finally:
        conn.close()


def _one_round(threads: int) -> tuple[int, int, int]:
    """看一遍表。返回 (成功条数, 失败条数, 已删除的不超过 6 脚条数)。"""
    from agent_log import log_line
    from footprint_db import PIN_COUNT_STORE_MAX_SKIP, connect, list_pending_json_for_iou

    conn = connect()
    try:
        jobs, removed, blocked = list_pending_json_for_iou(conn)
    finally:
        conn.close()

    if not jobs:
        log_line(
            "待完成IOU",
            result=(
                f"【检查】待计算=0 异常跳过={blocked} "
                f"脚数不超过{PIN_COUNT_STORE_MAX_SKIP}已删除={removed} "
                f"{POLL_SECONDS}秒后再看"
            ),
        )
        return 0, 0, removed

    log_line(
        "待完成IOU",
        result=(
            f"【本轮开始】待计算={len(jobs)} 异常跳过={blocked} 线程={threads} "
            f"脚数不超过{PIN_COUNT_STORE_MAX_SKIP}已删除={removed}"
        ),
    )
    ok_count = 0
    fail_count = 0
    pool = ThreadPoolExecutor(max_workers=threads)
    try:
        futures = [pool.submit(_run_one, job) for job in jobs]
        for fut in as_completed(futures):
            if fut.result() == "ok":
                ok_count += 1
            else:
                fail_count += 1
    except KeyboardInterrupt:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)
    log_line(
        "待完成IOU",
        result=(
            f"【本轮完成】成功={ok_count} 失败={fail_count} 异常跳过={blocked} "
            f"脚数不超过{PIN_COUNT_STORE_MAX_SKIP}已删除={removed}"
        ),
    )
    return ok_count, fail_count, removed


def main() -> None:
    parser = argparse.ArgumentParser(description="持续计算待完成板 JSON 与库中器件的 IoU")
    parser.add_argument("--config", default=None, help="YAML 配置路径")
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS, help="线程数，默认 8")
    args = parser.parse_args()

    from settings import bootstrap

    info = bootstrap(args.config)
    print(f"配置: {info['config_path']}")

    from agent_log import log_line

    threads = max(1, int(args.threads or DEFAULT_THREADS))
    log_line(
        "待完成IOU",
        result=(
            f"【启动】持续运行 线程={threads} 每{POLL_SECONDS}秒检查 pending_json "
            f"结果写入 pending_iou。Ctrl+C 或结束进程才停"
        ),
    )
    try:
        while True:
            _one_round(threads)
            time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        log_line("待完成IOU", result="【停止】收到 Ctrl+C，脚本退出")
        print("已停止")


if __name__ == "__main__":
    main()
