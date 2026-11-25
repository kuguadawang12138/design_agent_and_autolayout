# -*- coding: utf-8 -*-
"""用多线程计算待完成板 JSON 和库中其它器件的 IoU。

JSON 在 pending_component_json，结果写入 pending_iou。默认 8 个线程。
每个线程单独连库，线程内部不再开进程池。
用法（在 ingest_service 目录，先执行 python setup_db.py）：

    python compute_pending_iou.py
    python compute_pending_iou.py --threads 8
    python compute_pending_iou.py --config config/footprint_sim_st.yaml
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed


def _compute_one(job: dict) -> tuple[str, str]:
    from footprint_db import (
        claim_pending_json,
        connect,
        finish_pending_json,
        precompute_pending_iou,
    )

    label = f"{job['project_code']}/{job['refdes']}"
    conn = connect()
    try:
        text = claim_pending_json(conn, int(job["id"]))
        if not text:
            return "skip", label
        try:
            written = precompute_pending_iou(
                conn,
                text,
                source_file=job.get("file_name") or "pending.json",
                parallel=False,
            )
        except Exception as exc:
            conn.rollback()
            finish_pending_json(conn, int(job["id"]), ok=False)
            return "fail", f"{label}: {type(exc).__name__}: {exc}"
        finish_pending_json(conn, int(job["id"]), ok=True)
        return "ok", f"{label} 写入={written}"
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="计算待完成板 JSON 与库中器件的 IoU")
    parser.add_argument("--config", default=None, help="YAML 配置路径")
    parser.add_argument("--threads", type=int, default=8, help="线程数，默认 8")
    args = parser.parse_args()
    threads = max(1, int(args.threads or 8))

    from settings import bootstrap

    info = bootstrap(args.config)
    print(f"配置: {info['config_path']}")
    print(f"线程数: {threads}")

    from footprint_db import connect, list_pending_json_jobs, reset_stuck_pending_json

    conn = connect()
    try:
        stuck = reset_stuck_pending_json(conn)
        jobs = list_pending_json_jobs(conn)
    finally:
        conn.close()
    if stuck:
        print(f"上次中断的计算中记录已退回待计算 {stuck} 条")
    print(f"待计算 {len(jobs)} 条")
    if not jobs:
        return

    ok = 0
    failed: list[str] = []
    skipped = 0
    with ThreadPoolExecutor(max_workers=threads) as pool:
        futures = [pool.submit(_compute_one, job) for job in jobs]
        for fut in as_completed(futures):
            try:
                kind, text = fut.result()
            except Exception as exc:
                failed.append(f"{type(exc).__name__}: {exc}")
                print(f"失败 {type(exc).__name__}: {exc}")
                continue
            if kind == "ok":
                ok += 1
                print(f"完成 {text}")
            elif kind == "skip":
                skipped += 1
            else:
                failed.append(text)
                print(f"失败 {text}")

    print(f"IoU 已写入 {ok} 条")
    print(f"跳过 {skipped} 条")
    print(f"失败 {len(failed)} 条")


if __name__ == "__main__":
    main()
