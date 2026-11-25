# -*- coding: utf-8 -*-
"""
检查器（布尔判定）注册表。

每个检查器只实现「机制」，所有阈值通过 params 传入（来自 YAML / Allegro 导出）。
返回 CheckResult：
    ok      是否通过
    amount  违反量（mm、个数等，用于 warning 级规则转惩罚；通过时为 0）
    detail  可读说明（进报告 / 解释）

签名约定：
    placement 级： fn(cand: Candidate, ctx: PlacementContext, **params) -> CheckResult
    layout 级：    fn(placements: list[Component], ctx: PlacementContext, **params) -> CheckResult
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import shapely
from shapely.geometry import Point

from .model import Candidate, Component, PlacementContext, opposite_side


@dataclass
class CheckResult:
    ok: bool
    amount: float = 0.0
    detail: str = ""

    @staticmethod
    def passed(detail: str = "") -> "CheckResult":
        return CheckResult(True, 0.0, detail)

    @staticmethod
    def failed(amount: float, detail: str) -> "CheckResult":
        return CheckResult(False, float(amount), detail)


CheckFn = Callable[..., CheckResult]

_PLACEMENT_CHECKS: dict[str, CheckFn] = {}
_LAYOUT_CHECKS: dict[str, CheckFn] = {}


def placement_check(name: str):
    def deco(fn: CheckFn) -> CheckFn:
        _PLACEMENT_CHECKS[name] = fn
        return fn
    return deco


def layout_check(name: str):
    def deco(fn: CheckFn) -> CheckFn:
        _LAYOUT_CHECKS[name] = fn
        return fn
    return deco


def get_check(name: str, scope: str) -> CheckFn:
    table = _PLACEMENT_CHECKS if scope == "placement" else _LAYOUT_CHECKS
    if name not in table:
        raise KeyError(f"未注册的 {scope} 检查器: {name}（可用: {sorted(table)}）")
    return table[name]


def list_checks() -> dict[str, list[str]]:
    return {"placement": sorted(_PLACEMENT_CHECKS), "layout": sorted(_LAYOUT_CHECKS)}


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #

def _nearest_pin(cand: Candidate, ctx: PlacementContext, net: str | None):
    """返回 (最近引脚, 曼哈顿距离)。net 为空时取候选器件的 net_pwr。"""
    return ctx.nearest_pin(net or cand.comp.net_pwr, cand.x, cand.y, manhattan=True)


def _nearest_via(cand: Candidate, ctx: PlacementContext, net: str | None):
    return ctx.nearest_via(net, cand.x, cand.y)


# --------------------------------------------------------------------------- #
# 几何不变量（invariant）
# --------------------------------------------------------------------------- #

@placement_check("no_overlap")
def check_no_overlap(cand: Candidate, ctx: PlacementContext, clearance_mm: float = 0.0) -> CheckResult:
    """与同面已摆放器件（含主芯片）不重叠；clearance_mm>0 时按外扩后判定。"""
    poly = cand.polygon.buffer(clearance_mm) if clearance_mm > 0 else cand.polygon
    others = list(ctx.nearby_components(cand.polygon, clearance_mm))
    if ctx.chip.side == cand.side:
        others.append(ctx.chip)
    worst = 0.0
    who = ""
    for o in others:
        if o.side != cand.side or o.refdes == cand.comp.refdes:
            continue
        if poly.intersects(o.polygon):
            a = poly.intersection(o.polygon).area
            if a > worst or not who:
                worst, who = a, o.refdes
    if who:
        return CheckResult.failed(max(worst, 1e-6), f"与 {who} 重叠 {worst:.3f} mm²")
    return CheckResult.passed()


@placement_check("inside_board")
def check_inside_board(cand: Candidate, ctx: PlacementContext, margin_mm: float = 0.0) -> CheckResult:
    """器件整体在板框内（margin_mm 为板边装配距离）。"""
    region = ctx.board.inner(margin_mm)
    if region.contains(cand.polygon):
        return CheckResult.passed()
    out = cand.polygon.difference(region).area
    return CheckResult.failed(out, f"越出板框/板边距离 {margin_mm}mm，越界面积 {out:.3f} mm²")


@placement_check("not_in_keepout")
def check_not_in_keepout(cand: Candidate, ctx: PlacementContext) -> CheckResult:
    for k in ctx.board.keepouts:
        if k.applies_to_side(cand.side) and cand.polygon.intersects(k.polygon):
            a = cand.polygon.intersection(k.polygon).area
            return CheckResult.failed(a, f"进入禁布区 {k.name or '(未命名)'} {a:.3f} mm²")
    return CheckResult.passed()


@placement_check("height_limit")
def check_height_limit(cand: Candidate, ctx: PlacementContext) -> CheckResult:
    for h in ctx.board.height_regions:
        if h.applies_to_side(cand.side) and cand.polygon.intersects(h.polygon):
            if cand.comp.z_height_mm > h.max_height_mm:
                over = cand.comp.z_height_mm - h.max_height_mm
                return CheckResult.failed(over, f"限高区 {h.name or ''} 允许 {h.max_height_mm}mm，器件 {cand.comp.z_height_mm}mm")
    return CheckResult.passed()


@placement_check("side_allowed")
def check_side_allowed(cand: Candidate, ctx: PlacementContext, allowed: list[str] | None = None) -> CheckResult:
    """面限制。allowed 为空时表示不限制。"""
    if allowed and cand.side not in allowed:
        return CheckResult.failed(1.0, f"面 {cand.side} 不在允许范围 {allowed}")
    return CheckResult.passed()


@placement_check("in_room")
def check_in_room(cand: Candidate, ctx: PlacementContext) -> CheckResult:
    """若器件或主芯片指定了 room，则电容必须落在该 room 内。"""
    room = cand.comp.room or ctx.chip.room
    if not room:
        return CheckResult.passed("未定义 room")
    poly = ctx.board.rooms.get(room)
    if poly is None:
        return CheckResult.passed(f"room {room} 无几何定义，跳过")
    if poly.contains(cand.polygon):
        return CheckResult.passed()
    out = cand.polygon.difference(poly).area
    return CheckResult.failed(out, f"越出 room {room} {out:.3f} mm²")


@placement_check("via_clearance")
def check_via_clearance(cand: Candidate, ctx: PlacementContext, min_mm: float = 0.15,
                        ignore_same_net: bool = True) -> CheckResult:
    """器件外形与他网过孔的最小距离（贴片器件焊盘不能压他网过孔）。"""
    same = {cand.comp.net_pwr, cand.comp.net_gnd}
    worst = None
    for v in ctx.nearby_vias(cand.polygon, min_mm):
        if ignore_same_net and v.net in same:
            continue
        d = cand.polygon.distance(v.point) - v.diameter_mm / 2.0
        if d < min_mm and (worst is None or d < worst[0]):
            worst = (d, v)
    if worst is not None:
        d, v = worst
        return CheckResult.failed(min_mm - d, f"距他网过孔({v.net}) {d:.3f}mm < {min_mm}mm")
    return CheckResult.passed()


@placement_check("no_th_pin_under")
def check_no_th_pin_under(cand: Candidate, ctx: PlacementContext) -> CheckResult:
    """对面器件的通孔引脚不得穿到本器件下方（Allegro DFA: TH pins under SMD）。"""
    for o in list(ctx.placed) + [ctx.chip]:
        if o.side != opposite_side(cand.side):
            continue
        for p in o.pins:
            if p.through_hole and cand.polygon.contains(Point(p.x, p.y)):
                return CheckResult.failed(1.0, f"{o.refdes}.{p.name} 通孔引脚位于器件下方")
    return CheckResult.passed()


# --------------------------------------------------------------------------- #
# 参数化硬规则（parametric_hard）
# --------------------------------------------------------------------------- #

@placement_check("dfa_pkg_spacing")
def check_dfa_pkg_spacing(cand: Candidate, ctx: PlacementContext, level: str = "minimum") -> CheckResult:
    """
    按 DFA PkgToPkg 表检查同面器件间距。level = minimum（error）| preferred（warning）。
    真实 DFA 还区分 side/end 方向，这里先用整体外形距离；方向细化留给 V1。
    """
    lookup = ctx.dfa.minimum if level == "minimum" else ctx.dfa.preferred
    worst = None
    search = max(ctx.dfa.default_preferred_mm, ctx.dfa.default_min_mm, 2.0)
    others = list(ctx.nearby_components(cand.polygon, search))
    if ctx.chip.side == cand.side:
        others.append(ctx.chip)
    for o in others:
        if o.side != cand.side or o.refdes == cand.comp.refdes:
            continue
        need = lookup(cand.comp.dfa_class, o.dfa_class)
        d = cand.polygon.distance(o.polygon)
        if d < need and (worst is None or (need - d) > worst[0]):
            worst = (need - d, o, d, need)
    if worst is not None:
        short, o, d, need = worst
        return CheckResult.failed(short, f"与 {o.refdes} 间距 {d:.3f}mm < DFA {level} {need}mm")
    return CheckResult.passed()


@placement_check("max_dist_to_pin")
def check_max_dist_to_pin(cand: Candidate, ctx: PlacementContext, max_mm: float = 5.0,
                          value_uF_max: float | None = None, value_uF_min: float | None = None) -> CheckResult:
    """到所服务电源引脚的曼哈顿距离上限；可按容值区间限定适用对象。"""
    v = cand.comp.value_uF
    if value_uF_max is not None and (v is None or v > value_uF_max):
        return CheckResult.passed("容值不在适用区间")
    if value_uF_min is not None and (v is None or v < value_uF_min):
        return CheckResult.passed("容值不在适用区间")
    pin, d = _nearest_pin(cand, ctx, None)
    if pin is None:
        return CheckResult.passed("主芯片无该电源网络引脚，跳过")
    if d > max_mm:
        return CheckResult.failed(d - max_mm, f"到 {pin.net} 引脚 {pin.name} 距离 {d:.2f}mm > {max_mm}mm")
    return CheckResult.passed()


@placement_check("preferred_side")
def check_preferred_side(cand: Candidate, ctx: PlacementContext, side: str = "BOTTOM",
                         value_uF_max: float | None = 0.1, relative_to_chip: bool = True) -> CheckResult:
    """小容值电容优先放在主芯片背面（relative_to_chip=True 时 side 指“相对芯片的对面”）。"""
    v = cand.comp.value_uF
    if value_uF_max is not None and (v is None or v > value_uF_max):
        return CheckResult.passed("容值不在适用区间")
    want = opposite_side(ctx.chip.side) if relative_to_chip else side
    if cand.side != want:
        return CheckResult.failed(1.0, f"建议放在 {want} 面，当前 {cand.side}")
    return CheckResult.passed()


@placement_check("pad_to_via_max")
def check_pad_to_via_max(cand: Candidate, ctx: PlacementContext, max_mm: float = 1.0,
                         net: str = "pwr") -> CheckResult:
    """电容中心到同网最近过孔距离上限（近似“焊盘到过孔走线长度”）。net = pwr | gnd。"""
    target = cand.comp.net_pwr if net == "pwr" else cand.comp.net_gnd
    via, d = _nearest_via(cand, ctx, target)
    if via is None:
        return CheckResult.passed("无同网过孔，视为需新增过孔（由 metrics 计惩罚）")
    if d > max_mm:
        return CheckResult.failed(d - max_mm, f"到 {target} 最近过孔 {d:.2f}mm > {max_mm}mm")
    return CheckResult.passed()


@placement_check("via_density_limit")
def check_via_density_limit(cand: Candidate, ctx: PlacementContext, radius_mm: float = 3.0,
                            max_count: int = 12, new_vias: int = 2) -> CheckResult:
    """候选位置半径 radius_mm 内过孔数（含本电容将新增的过孔）不超过上限，保护平面完整性。"""
    window = shapely.box(cand.x - radius_mm, cand.y - radius_mm, cand.x + radius_mm, cand.y + radius_mm)
    n = sum(1 for v in ctx.nearby_vias(window)
            if math.hypot(v.x - cand.x, v.y - cand.y) <= radius_mm) + new_vias
    if n > max_count:
        return CheckResult.failed(n - max_count, f"半径 {radius_mm}mm 内过孔 {n} > {max_count}")
    return CheckResult.passed()


@placement_check("orientation_radial")
def check_orientation_radial(cand: Candidate, ctx: PlacementContext, tol_deg: float = 15.0) -> CheckResult:
    """电容长轴朝向应与“候选位置 → 最近电源引脚”的方向一致（±tol）。"""
    pin, _ = _nearest_pin(cand, ctx, None)
    if pin is None:
        return CheckResult.passed("无目标引脚")
    want = math.degrees(math.atan2(pin.y - cand.y, pin.x - cand.x))
    # 长轴方向：width 是长边时长轴为 rot，否则为 rot+90
    axis = cand.rot_deg if cand.comp.width_mm >= cand.comp.height_mm else cand.rot_deg + 90.0
    diff = abs((want - axis + 90.0) % 180.0 - 90.0)
    if diff > tol_deg:
        return CheckResult.failed(diff - tol_deg, f"朝向偏差 {diff:.1f}° > {tol_deg}°")
    return CheckResult.passed()


@placement_check("not_in_channel")
def check_not_in_channel(cand: Candidate, ctx: PlacementContext, max_overlap_ratio: float = 0.0) -> CheckResult:
    """不占用预留走线通道（可允许一定比例）。"""
    if not ctx.board.channels:
        return CheckResult.passed("无通道定义")
    area = cand.polygon.area or 1e-9
    ratio = sum(cand.polygon.intersection(c).area for c in ctx.board.channels) / area
    if ratio > max_overlap_ratio:
        return CheckResult.failed(ratio - max_overlap_ratio, f"占用走线通道 {ratio:.0%}")
    return CheckResult.passed()


# --------------------------------------------------------------------------- #
# layout 级检查（整组器件）
# --------------------------------------------------------------------------- #

@layout_check("every_power_net_has_small_cap")
def check_every_power_net_has_small_cap(placements: list[Component], ctx: PlacementContext,
                                        value_uF_max: float = 0.1) -> CheckResult:
    """主芯片每个电源网络至少有一颗 ≤ value_uF_max 的去耦电容。"""
    missing = []
    for net in ctx.power_nets():
        ok = any(c.net_pwr == net and c.value_uF is not None and c.value_uF <= value_uF_max for c in placements)
        if not ok:
            missing.append(net)
    if missing:
        return CheckResult.failed(len(missing), f"电源网络缺少小容值电容: {missing}")
    return CheckResult.passed()


@layout_check("max_caps_per_gnd_via")
def check_max_caps_per_gnd_via(placements: list[Component], ctx: PlacementContext,
                               max_share: int = 2, assign_radius_mm: float = 1.5) -> CheckResult:
    """近似：每颗电容归属最近的地过孔（半径内），单个地过孔共享电容数不超过 max_share。"""
    counts: dict[int, int] = {}
    for c in placements:
        if not c.net_gnd:
            continue
        best_i, best_d = None, math.inf
        for i, v in enumerate(ctx.vias):
            if v.net != c.net_gnd:
                continue
            d = math.hypot(v.x - c.x, v.y - c.y)
            if d < best_d:
                best_i, best_d = i, d
        if best_i is not None and best_d <= assign_radius_mm:
            counts[best_i] = counts.get(best_i, 0) + 1
    over = {i: n for i, n in counts.items() if n > max_share}
    if over:
        return CheckResult.failed(sum(n - max_share for n in over.values()),
                                  f"{len(over)} 个地过孔被 >{max_share} 颗电容共享")
    return CheckResult.passed()
