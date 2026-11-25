# -*- coding: utf-8 -*-
"""
最小 PDM（统一中间表示）数据模型 —— 只覆盖去耦电容摆放任务需要的对象。

单位约定：全部为 mm，角度为度（逆时针为正）。面（side）取 "TOP" / "BOTTOM"。
这里的对象是 docs/05 §3.1 里「PDM」在首战场景下的一个子集，将来 Allegro Extracta / ODB++
解析器的产物应能直接映射到这些 dataclass；KiCad 侧可用 NeurPCB 的 kicad_bridge 映射。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import cached_property
from typing import Iterable, Optional

import numpy as np
import shapely
from shapely.geometry import Polygon, Point
from shapely.strtree import STRtree

SIDE_TOP = "TOP"
SIDE_BOTTOM = "BOTTOM"


def opposite_side(side: str) -> str:
    return SIDE_BOTTOM if side == SIDE_TOP else SIDE_TOP


def rect_polygon(cx: float, cy: float, w: float, h: float, rot_deg: float = 0.0) -> Polygon:
    """以 (cx, cy) 为中心、长 w 宽 h、旋转 rot_deg 的矩形（直接算四角坐标，比 affinity 快一个量级）。"""
    hw, hh = w / 2.0, h / 2.0
    if rot_deg:
        a = math.radians(rot_deg)
        c, s = math.cos(a), math.sin(a)
    else:
        c, s = 1.0, 0.0
    pts = []
    for dx, dy in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)):
        pts.append((cx + dx * c - dy * s, cy + dx * s + dy * c))
    return Polygon(pts)


def expanded_bbox(geom, pad: float):
    """geom 的包围盒外扩 pad 后的矩形（用于空间索引查询，避免昂贵的 buffer）。"""
    minx, miny, maxx, maxy = geom.bounds
    return shapely.box(minx - pad, miny - pad, maxx + pad, maxy + pad)


# --------------------------------------------------------------------------- #
# 板级对象
# --------------------------------------------------------------------------- #

@dataclass
class KeepoutRegion:
    """Package Keepout：禁止摆放器件的区域，可限定面。sides 为空表示两面都禁。"""
    polygon: Polygon
    sides: tuple[str, ...] = ()
    name: str = ""

    def applies_to_side(self, side: str) -> bool:
        return not self.sides or side in self.sides


@dataclass
class HeightRegion:
    """限高区：区域内器件高度不得超过 max_height_mm。"""
    polygon: Polygon
    max_height_mm: float
    sides: tuple[str, ...] = ()
    name: str = ""

    def applies_to_side(self, side: str) -> bool:
        return not self.sides or side in self.sides


@dataclass
class Board:
    outline: Polygon                                  # 板框（Route Keepin / Package Keepin 取其一）
    keepouts: list[KeepoutRegion] = field(default_factory=list)
    height_regions: list[HeightRegion] = field(default_factory=list)
    rooms: dict[str, Polygon] = field(default_factory=dict)   # room 名 → 多边形
    channels: list[Polygon] = field(default_factory=list)     # 预留走线通道（可选）
    layer_count: int = 4
    board_type: str = "digital"                       # 供规则 applies_to 匹配
    speed_class: str = "default"                      # 例如 ">=1Gbps"
    thickness_mm: float = 1.6
    _inner_cache: dict[float, Polygon] = field(default_factory=dict, repr=False)

    def inner(self, margin_mm: float) -> Polygon:
        """板框内缩 margin_mm 的区域（缓存，避免每个候选都做一次 buffer）。"""
        if margin_mm <= 0:
            return self.outline
        poly = self._inner_cache.get(margin_mm)
        if poly is None:
            poly = self.outline.buffer(-margin_mm)
            self._inner_cache[margin_mm] = poly
        return poly


# --------------------------------------------------------------------------- #
# 器件 / 引脚 / 过孔
# --------------------------------------------------------------------------- #

@dataclass
class Pin:
    name: str
    x: float                     # 板级绝对坐标
    y: float
    net: str
    kind: str = "signal"         # power / gnd / signal / nc
    through_hole: bool = False


@dataclass
class Component:
    refdes: str
    x: float
    y: float
    width_mm: float              # 封装外形（含 DFA_BOUND 余量更好）
    height_mm: float             # 封装外形另一边（不是元件高度）
    rot_deg: float = 0.0
    side: str = SIDE_TOP
    z_height_mm: float = 1.0     # 元件实际高度（限高检查用）
    dfa_class: str = "DEFAULT"   # Allegro DFA_DEV_CLASS
    room: Optional[str] = None
    pins: list[Pin] = field(default_factory=list)
    # 去耦电容专用属性
    value_uF: Optional[float] = None
    net_pwr: Optional[str] = None
    net_gnd: Optional[str] = None
    fixed: bool = False

    @cached_property
    def polygon(self) -> Polygon:
        return rect_polygon(self.x, self.y, self.width_mm, self.height_mm, self.rot_deg)

    def moved(self, x: float, y: float, rot_deg: float, side: str) -> "Component":
        """返回放到新位置的副本（引脚坐标不在此处重算，去耦任务不需要）。"""
        c = Component(
            refdes=self.refdes, x=x, y=y,
            width_mm=self.width_mm, height_mm=self.height_mm,
            rot_deg=rot_deg, side=side, z_height_mm=self.z_height_mm,
            dfa_class=self.dfa_class, room=self.room, pins=self.pins,
            value_uF=self.value_uF, net_pwr=self.net_pwr, net_gnd=self.net_gnd,
            fixed=self.fixed,
        )
        return c


@dataclass
class Via:
    x: float
    y: float
    net: str
    diameter_mm: float = 0.45
    span_mm: float = 1.6          # 过孔长度（到目标平面层深，电感估算用）

    @property
    def point(self) -> Point:
        return Point(self.x, self.y)


# --------------------------------------------------------------------------- #
# 候选动作
# --------------------------------------------------------------------------- #

@dataclass
class Candidate:
    """一个待评估的摆放动作：把 comp 放到 (x, y, rot, side)。"""
    comp: Component
    x: float
    y: float
    rot_deg: float
    side: str

    @cached_property
    def placed(self) -> Component:
        return self.comp.moved(self.x, self.y, self.rot_deg, self.side)

    @property
    def polygon(self) -> Polygon:
        return self.placed.polygon


# --------------------------------------------------------------------------- #
# 摆放上下文（RL 的「环境」在规则层看到的东西）
# --------------------------------------------------------------------------- #

@dataclass
class DfaSpacingTable:
    """
    Allegro DFA PkgToPkg Spacing 的极简版：按 (classA, classB) 查最小 / 推荐间距。
    真实项目应从 Constraint Manager 导出的 CSV 直接加载，这里只放机制。
    """
    min_mm: dict[tuple[str, str], float] = field(default_factory=dict)
    preferred_mm: dict[tuple[str, str], float] = field(default_factory=dict)
    default_min_mm: float = 0.2
    default_preferred_mm: float = 0.5

    def _lookup(self, table: dict, a: str, b: str, default: float) -> float:
        return table.get((a, b), table.get((b, a), default))

    def minimum(self, a: str, b: str) -> float:
        return self._lookup(self.min_mm, a, b, self.default_min_mm)

    def preferred(self, a: str, b: str) -> float:
        return self._lookup(self.preferred_mm, a, b, self.default_preferred_mm)


@dataclass
class PlacementContext:
    board: Board
    chip: Component                                  # 主芯片（已固定）
    placed: list[Component] = field(default_factory=list)   # 已摆放的其他器件（含本组已放电容）
    vias: list[Via] = field(default_factory=list)
    dfa: DfaSpacingTable = field(default_factory=DfaSpacingTable)
    # 环路电感估算系数（占位，需 SI 团队校准，见 docs/08 §4.5）
    inductance_coeffs: dict[str, float] = field(
        default_factory=lambda: {"nH_per_mm_trace": 0.8, "nH_per_mm_via": 1.0, "nH_per_mm_via_gap": 0.3}
    )

    # ---- 空间索引：每次 placed / vias 变化后调用 refresh() ----
    _placed_tree: Optional[STRtree] = field(default=None, repr=False)
    _placed_polys: list[Polygon] = field(default_factory=list, repr=False)
    _via_tree: Optional[STRtree] = field(default=None, repr=False)
    _via_points: list[Point] = field(default_factory=list, repr=False)
    _vias_by_net: dict[str, list[Via]] = field(default_factory=dict, repr=False)
    _via_xy_by_net: dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    _via_max_radius: float = field(default=0.0, repr=False)
    _pins_by_net: dict[str, list[Pin]] = field(default_factory=dict, repr=False)
    _pin_xy_by_net: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    def refresh(self) -> None:
        self._placed_polys = [c.polygon for c in self.placed]
        self._placed_tree = STRtree(self._placed_polys) if self._placed_polys else None
        self._refresh_vias()
        self._refresh_pins()

    def _refresh_vias(self) -> None:
        self._via_points = [v.point for v in self.vias]
        self._via_tree = STRtree(self._via_points) if self._via_points else None
        by_net: dict[str, list[Via]] = {}
        for v in self.vias:
            by_net.setdefault(v.net, []).append(v)
        self._vias_by_net = by_net
        self._via_xy_by_net = {n: np.array([(v.x, v.y) for v in vs], dtype=float) for n, vs in by_net.items()}
        self._via_max_radius = max((v.diameter_mm / 2.0 for v in self.vias), default=0.0)

    def _refresh_pins(self) -> None:
        by_net: dict[str, list[Pin]] = {}
        for p in self.chip.pins:
            by_net.setdefault(p.net, []).append(p)
        self._pins_by_net = by_net
        self._pin_xy_by_net = {n: np.array([(p.x, p.y) for p in ps], dtype=float) for n, ps in by_net.items()}

    # ---- 最近邻查询（numpy 向量化，供检查器 / 指标复用）----
    def nearest_via(self, net: Optional[str], x: float, y: float) -> tuple[Optional[Via], float]:
        """返回 (最近同网过孔, 欧氏距离)；无则 (None, inf)。"""
        if not net:
            return None, math.inf
        if not self._via_xy_by_net and self.vias:
            self._refresh_vias()
        xy = self._via_xy_by_net.get(net)
        if xy is None or len(xy) == 0:
            return None, math.inf
        d = np.hypot(xy[:, 0] - x, xy[:, 1] - y)
        i = int(d.argmin())
        return self._vias_by_net[net][i], float(d[i])

    def nearest_pin(self, net: Optional[str], x: float, y: float, manhattan: bool = True) -> tuple[Optional[Pin], float]:
        """返回 (主芯片上最近的该网引脚, 距离)；无则 (None, inf)。"""
        if not net:
            return None, math.inf
        if not self._pin_xy_by_net and self.chip.pins:
            self._refresh_pins()
        xy = self._pin_xy_by_net.get(net)
        if xy is None or len(xy) == 0:
            return None, math.inf
        if manhattan:
            d = np.abs(xy[:, 0] - x) + np.abs(xy[:, 1] - y)
        else:
            d = np.hypot(xy[:, 0] - x, xy[:, 1] - y)
        i = int(d.argmin())
        return self._pins_by_net[net][i], float(d[i])

    def add_placed(self, comp: Component) -> None:
        self.placed.append(comp)
        self.refresh()

    def add_via(self, via: Via) -> None:
        self.vias.append(via)
        self._refresh_vias()

    def nearby_vias(self, geom, buffer_mm: float = 0.0) -> Iterable[Via]:
        """返回与 geom（外扩 buffer_mm + 最大过孔半径）包围盒相交的过孔。"""
        if self._via_tree is None:
            if self.vias:
                self._refresh_vias()
            if self._via_tree is None:
                return []
        pad = buffer_mm + self._via_max_radius
        query = expanded_bbox(geom, pad) if pad > 0 else geom
        idxs = self._via_tree.query(query)
        return [self.vias[int(i)] for i in idxs]

    def nearby_components(self, poly: Polygon, buffer_mm: float = 0.0) -> Iterable[Component]:
        """返回与 poly（外扩 buffer_mm）包围盒相交的已摆放器件。"""
        if self._placed_tree is None:
            if self._placed_polys or self.placed:
                self.refresh()
            if self._placed_tree is None:
                return []
        query = expanded_bbox(poly, buffer_mm) if buffer_mm > 0 else poly
        idxs = self._placed_tree.query(query)
        return [self.placed[int(i)] for i in idxs]

    # ---- 便捷查询 ----
    def chip_pins(self, net: Optional[str] = None, kind: Optional[str] = None) -> list[Pin]:
        out = []
        for p in self.chip.pins:
            if net is not None and p.net != net:
                continue
            if kind is not None and p.kind != kind:
                continue
            out.append(p)
        return out

    def vias_of_net(self, net: str) -> list[Via]:
        if not self._vias_by_net and self.vias:
            self._refresh_vias()
        return self._vias_by_net.get(net, [])

    def is_clear_of_components(self, geom, clearance_mm: float = 0.0, exclude_refdes: str | None = None) -> bool:
        """
        geom（如新过孔圆）与所有已摆放器件外形保持 clearance_mm 以上（过孔穿透两面，不分面）。
        主芯片不在此列：BGA 背面扇出过孔本来就在芯片投影之下。
        """
        for o in self.nearby_components(geom, clearance_mm):
            if exclude_refdes and o.refdes == exclude_refdes:
                continue
            if geom.distance(o.polygon) < clearance_mm:
                return False
        return True

    def power_nets(self) -> list[str]:
        seen: dict[str, None] = {}
        for p in self.chip.pins:
            if p.kind == "power":
                seen.setdefault(p.net, None)
        return list(seen.keys())
