# -*- coding: utf-8 -*-
"""历史回补：跑完一天再跑下一天前的等待（yaml oa_api_interval_sec，默认 10 秒）。"""

from __future__ import annotations

import time


def oa_api_interval_sec() -> float:
    from settings import get_oa_scheduler_config

    sched = get_oa_scheduler_config()
    raw = sched.get("oa_api_interval_sec", sched.get("delay_sec_between_projects", 10))
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return 10.0


def sleep_oa_api_interval(*, reason: str = "") -> None:
    sec = oa_api_interval_sec()
    if sec <= 0:
        return
    from agent_log import log_line

    note = f"等待 {sec:.0f} 秒"
    if reason:
        note += f"（{reason}）"
    log_line("OA流程", result=f"【接口间隔】{note}")
    time.sleep(sec)
