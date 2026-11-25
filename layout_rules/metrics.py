# -*- coding: utf-8 -*-
"""
软指标（连续值）注册表 —— 奖励函数的原材料。

每个指标只返回一个 float，方向约定写在 docstring 里；把它变成惩罚值的形状（hinge / inverse……）
和权重由 YAML 规则决定，不在这里硬编码。

签名约定：
    placement 级： fn(cand: Candidate, ctx: PlacementContext, **params) -> float
    layout 级：    fn(placements: list[Component], ctx: PlacementContext, **params) -> float
"""

from __future__ import annotations

import math
from typing import Callable

import shapely

from .model import Candidate, Component, PlacementContext

MetricFn = Callable[..., float]

_PLACEMENT_METRICS: dict[str, MetricFn] = {}
_LAYOUT_METRICS: dict[str, MetricFn] = {}


def placement_metric(name: str):
    def deco(fn: MetricFn) -> MetricFn:
        _PLACEMENT_METRICS[name] = fn
        return fn
    return deco


def layout_metric(name: str):
    def deco(fn: MetricFn) -> MetricFn:
        _LAYOUT_METRICS[name] = fn
        return fn
    return deco


def get_metric(name: str, scope: str) -> MetricFn:
    table = _PLACEMENT_METRICS if scope == "placement" else _LAYOUT_METRICS
    if name not in table:
        raise KeyError(f"未注册的 {scope} 指标: {name}（可用: {sorted(table)}）")
    return table[name]


def list_metrics() -> dict[str, list[str]]:
    return {"placement": sorted(_PLACEMENT_METRICS), "layout": sorted(_LAYOUT_METRICS)}


# --------------------------------------------------------------------------- #
# placement 级指标
# --------------------------------------------------------------------------- #

@placement_metric("dist_to_pin_mm")
def metric_dist_to_pin(cand: Candidate, ctx: PlacementContext, fallback_mm: float = 30.0) -> float:
    """到所服务电源引脚的曼哈顿距离（越小越好）。无引脚时返回 fallback。"""
    _, d = ctx.nearest_pin(cand.comp.net_pwr, cand.x, cand.y, manhattan=True)
    return d if math.isfinite(d) else fallback_mm


@placement_metric("pad_to_via_mm")
def metric_pad_to_via(cand: Candidate, ctx: PlacementContext, net: str = "pwr",
                      new_via_offset_mm: float = 0.6) -> float:
    """
    电容到同网最近过孔的距离（越小越好）。若没有可借用的过孔，视为需在焊盘旁新打过孔，
    返回 new_via_offset_mm（典型 0402 焊盘边到过孔中心距离）。
    """
    target = cand.comp.net_pwr if net == "pwr" else cand.comp.net_gnd
    _, d = ctx.nearest_via(target, cand.x, cand.y)
    return d if math.isfinite(d) else new_via_offset_mm


@placement_metric("pwr_gnd_via_gap_mm")
def metric_pwr_gnd_via_gap(cand: Candidate, ctx: PlacementContext, fallback_mm: float = 1.0) -> float:
    """电容所用电源过孔与地过孔的间距（越小回路面积越小）。取各自最近过孔近似。"""
    pv, _ = ctx.nearest_via(cand.comp.net_pwr, cand.x, cand.y)
    gv, _ = ctx.nearest_via(cand.comp.net_gnd, cand.x, cand.y)
    if pv is None or gv is None:
        return fallback_mm
    return math.hypot(pv.x - gv.x, pv.y - gv.y)


@placement_metric("loop_inductance_est_nH")
def metric_loop_inductance(cand: Candidate, ctx: PlacementContext) -> float:
    """
    解析近似的安装环路电感（越小越好）。占位公式：
        L ≈ k_trace * (d_pad_via_pwr + d_pad_via_gnd) + k_via * via_span + k_gap * gap_pwr_gnd
    系数在 PlacementContext.inductance_coeffs 中，必须由 SI 团队按叠层校准后才能作主奖励
    （docs/08 §4.5，docs/10 §2.2 三层校准链）。
    """
    k = ctx.inductance_coeffs
    d_pwr = metric_pad_to_via(cand, ctx, net="pwr")
    d_gnd = metric_pad_to_via(cand, ctx, net="gnd")
    gap = metric_pwr_gnd_via_gap(cand, ctx)
    span = ctx.board.thickness_mm
    pv, _ = ctx.nearest_via(cand.comp.net_pwr, cand.x, cand.y)
    if pv is not None:
        span = pv.span_mm
    return (k.get("nH_per_mm_trace", 0.8) * (d_pwr + d_gnd)
            + k.get("nH_per_mm_via", 1.0) * span
            + k.get("nH_per_mm_via_gap", 0.3) * gap)


@placement_metric("channel_block_ratio")
def metric_channel_block_ratio(cand: Candidate, ctx: PlacementContext) -> float:
    """候选外形落在预留走线通道内的面积比例（0~1，越小越好）。"""
    if not ctx.board.channels:
        return 0.0
    area = cand.polygon.area or 1e-9
    return min(1.0, sum(cand.polygon.intersection(c).area for c in ctx.board.channels) / area)


@placement_metric("local_occupancy_ratio")
def metric_local_occupancy(cand: Candidate, ctx: PlacementContext, radius_mm: float = 3.0) -> float:
    """候选周围 2r 见方窗口内、同面已被器件占用的面积比例（过高说明把区域堵死）。用方窗代替圆以省掉 buffer。"""
    window = shapely.box(cand.x - radius_mm, cand.y - radius_mm, cand.x + radius_mm, cand.y + radius_mm)
    occ = 0.0
    for o in ctx.nearby_components(window):
        if o.side == cand.side:
            occ += window.intersection(o.polygon).area
    if ctx.chip.side == cand.side:
        occ += window.intersection(ctx.chip.polygon).area
    return min(1.0, occ / (window.area or 1e-9))


# --------------------------------------------------------------------------- #
# layout 级指标（整组集合属性）
# --------------------------------------------------------------------------- #

@layout_metric("coverage_uniformity")
def metric_coverage_uniformity(placements: list[Component], ctx: PlacementContext, bins: int = 8) -> float:
    """
    电容围绕主芯片的角度分布均匀性（归一化熵，0~1，越大越均匀）。
    只统计带 net_pwr 的电容。少于 2 颗时返回 1.0（无从谈均匀）。
    """
    caps = [c for c in placements if c.net_pwr]
    if len(caps) < 2:
        return 1.0
    hist = [0] * bins
    for c in caps:
        ang = math.atan2(c.y - ctx.chip.y, c.x - ctx.chip.x) % (2 * math.pi)
        hist[min(bins - 1, int(ang / (2 * math.pi) * bins))] += 1
    n = len(caps)
    ent = -sum((h / n) * math.log(h / n) for h in hist if h > 0)
    return ent / math.log(min(bins, n)) if min(bins, n) > 1 else 1.0


@layout_metric("freq_band_order")
def metric_freq_band_order(placements: list[Component], ctx: PlacementContext) -> float:
    """
    “小容值靠内、大容值靠外”的符合度（0~1，越大越符合）。
    用 log(C) 与到芯片中心距离的 Spearman 相关系数映射到 [0,1]。
    """
    caps = [c for c in placements if c.net_pwr and c.value_uF]
    if len(caps) < 3:
        return 1.0
    xs = [math.log10(c.value_uF) for c in caps]
    ys = [math.hypot(c.x - ctx.chip.x, c.y - ctx.chip.y) for c in caps]

    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    n = len(caps)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    vy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if vx == 0 or vy == 0:
        return 1.0
    rho = cov / (vx * vy)
    return (rho + 1.0) / 2.0


@layout_metric("mean_dist_to_pin_mm")
def metric_mean_dist_to_pin(placements: list[Component], ctx: PlacementContext) -> float:
    """整组电容到各自电源引脚曼哈顿距离的均值（评测用，对应 docs/08 §7.2 几何指标）。"""
    ds = []
    for c in placements:
        if not c.net_pwr:
            continue
        _, d = ctx.nearest_pin(c.net_pwr, c.x, c.y, manhattan=True)
        if math.isfinite(d):
            ds.append(d)
    return sum(ds) / len(ds) if ds else 0.0
