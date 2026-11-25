# -*- coding: utf-8 -*-
"""YAML 历史回补日期与 OA 接口①参数的对应关系。"""

from __future__ import annotations

from datetime import date, timedelta


def oa_api_end_date(last_inclusive_day: date) -> str:
    """YAML 终止日（含）→ OA 接口① end_date（不含当天）。"""
    return (last_inclusive_day + timedelta(days=1)).isoformat()


def format_yaml_plan_log(plan_start: date, plan_end: date, oa_start: str, oa_end: str) -> str:
    """中文日志：区分 YAML 日历日与 OA 接口参数。"""
    return (
        f"YAML 从 {plan_start} 到 {plan_end}（起止日均含）；"
        f"本片接口① start_date={oa_start} end_date={oa_end}（OA 的 end 不含当天，故为终止日+1）"
    )
