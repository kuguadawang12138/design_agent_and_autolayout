# -*- coding: utf-8 -*-
"""
单个模块的确定性摆放。

输入是「主芯片 + 外围器件」的 JSON（坐标单位 mil）。主芯片不动，
外围器件按网络归到芯片某一侧，能排进一排或一列就只排一排：

- 上下边的器件竖放（长轴垂直于边），左右边的器件横放，少占沿边长度
- 同一排 / 列的放置原点共线
- 放置原点落在 5 mil 格点上
- 器件外框离主芯片外框至少 50 mil
- 这一排的外框跨度不超过主芯片对角线 × (1 + DIAG_SLACK)
- 跨度中心尽量落在芯片中心；若引脚重心偏离不超过 SYMM_WINDOW，则改跟引脚重心，飞线更短

论文 arXiv:2502.14012 的三步是「初值 → 全局微调（允许重叠）→ 合法化」。
这里把第二步收成沿边的一维微调：比较「严格居中」和「贴引脚重心」，
再用 ±5 mil 的单步滑动看飞线还能不能短。哪一版合法且更短，就用哪一版。

运行：
    .\\.venv\\Scripts\\python.exe data_layout\\place_u34.py
    .\\.venv\\Scripts\\python.exe data_layout\\place_u34.py data_layout\\U34_Comps.json
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
SRC = HERE / "U34_Comps.json"
OUT_BEFORE = HERE / "U34_before.png"
OUT_AFTER = HERE / "U34_after.png"
OUT_COMPARE = HERE / "U34_compare.png"
OUT_JSON = HERE / "U34_placed.json"

# 可调阈值。apply_units() 会按文件单位改写成 mil 或 mm。
# mil：格点 5、净空 50。mm：格点 0.1，净空仍是 50 mil（1.27 mm）。
GRID = 5.0
CLEARANCE = 50.0
ALONG_GAP = 0.0            # 同一排里器件外框之间的最小缝
RING_GAP = 5.0             # 排不下、退到下一圈时，两圈外框之间的缝
DIAG_SLACK = 0.08          # 一排跨度允许超出对角线的比例
SYMM_WINDOW = 50.0         # 引脚重心偏离芯片中心在这个范围内，就跟着重心走
UNIT_NAME = "mil"
POWER_HEADS = (
    "GND", "AGND", "DGND", "PGND",
    "VSS", "VDD", "AVDD", "DVDD",
    "VCC", "AVCC", "DVCC",
    "POS", "PVDD", "VPP", "VBUS",
)


def apply_units(units: str) -> None:
    """按 JSON 的 units 切换格点和净空。摆放公式不变，只换标尺。"""
    global GRID, CLEARANCE, RING_GAP, SYMM_WINDOW, UNIT_NAME
    name = (units or "mils").strip().lower()
    if name in ("mm", "millimeter", "millimeters"):
        UNIT_NAME = "mm"
        GRID = 0.1
        mil = 0.0254
        CLEARANCE = 50.0 * mil
        RING_GAP = 5.0 * mil
        SYMM_WINDOW = 50.0 * mil
        return
    UNIT_NAME = "mil"
    GRID = 5.0
    CLEARANCE = 50.0
    RING_GAP = 5.0
    SYMM_WINDOW = 50.0


def fmt_num(value: float) -> str:
    return f"{value:.2f}" if UNIT_NAME == "mm" else f"{value:.0f}"


def is_power_net(name: str) -> bool:
    u = (name or "").upper()
    return any(u == h or u.startswith(h) for h in POWER_HEADS)


def is_gnd_net(name: str) -> bool:
    u = (name or "").upper()
    return u.startswith("GND") or u.startswith("VSS") or u.startswith("AGND")


def snap_grid(value: float, grid: float | None = None) -> float:
    """四舍五入到格点。默认读当前单位的 GRID，负坐标同样适用。"""
    grid = GRID if grid is None else grid
    return math.floor(value / grid + 0.5) * grid


def snap_outward(value: float, sign: float, grid: float | None = None) -> float:
    """往芯片外侧推到格点上，保证净空不会因为取整变小。"""
    grid = GRID if grid is None else grid
    scaled = value / grid
    if sign > 0:
        n = math.ceil(scaled - 1e-9)
    else:
        n = math.floor(scaled + 1e-9)
    return n * grid


def rot_xy(x: float, y: float, deg: float) -> tuple[float, float]:
    """逆时针旋转，和 Allegro 的 rotation 一致。"""
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return x * c - y * s, x * s + y * c


def mst_edges(points: list[tuple[float, float]]) -> list[tuple[int, int, float]]:
    n = len(points)
    if n <= 1:
        return []
    in_tree = [False] * n
    dist = [float("inf")] * n
    parent = [-1] * n
    dist[0] = 0.0
    edges = []
    for _ in range(n):
        u = min((i for i in range(n) if not in_tree[i]), key=lambda i: dist[i])
        in_tree[u] = True
        if parent[u] >= 0:
            edges.append((parent[u], u, dist[u]))
        ux, uy = points[u]
        for v in range(n):
            if in_tree[v]:
                continue
            d = math.hypot(ux - points[v][0], uy - points[v][1])
            if d < dist[v]:
                dist[v] = d
                parent[v] = u
    return edges


@dataclass
class Pin:
    number: str
    net: str
    lx: float
    ly: float


@dataclass
class Part:
    ref: str
    layer: str
    x: float
    y: float
    rot: float
    # 旋转 0° 时，原点到外框四边的距离
    nx: float
    px: float
    ny: float
    py: float
    pins: list[Pin]
    kind: str
    side: str = ""
    ring: int = -1
    priority: int = 0

    def nets(self) -> set[str]:
        return {p.net for p in self.pins if p.net}


@dataclass
class ChipPin:
    number: str
    net: str
    x: float
    y: float
    side: str

    def along(self, side: str) -> float:
        return self.y if side in ("L", "R") else self.x


@dataclass
class Item:
    part: Part
    side: str
    anchor: float       # 排序用，桥接电阻会稍稍靠外
    aim: float          # 电气上想靠近的沿边坐标
    rot: float
    back: float         # 沿边反方向的外框半边
    front: float
    inward: float       # 朝向芯片的外框半边
    outward: float
    priority: int
    face: set[str] = field(default_factory=set)
    owner: tuple = ()
    min_ring: int = 0
    ring: int = 0
    along: float = 0.0


def world_margins(part: Part, rot: float) -> tuple[float, float, float, float]:
    """返回 (左, 右, 下, 上)，都是从放置原点量到外框。"""
    r = int(round(rot)) % 360
    nx, px, ny, py = part.nx, part.px, part.ny, part.py
    if r == 0:
        return nx, px, ny, py
    if r == 180:
        return px, nx, py, ny
    if r == 90:
        # (x, y) -> (-y, x)
        return py, ny, nx, px
    if r == 270:
        # (x, y) -> (y, -x)
        return ny, py, px, nx
    raise ValueError(f"只支持 90° 整数倍，得到 {rot}")


def local_extents(rot: float, left: float, right: float, bottom: float, top: float) -> tuple[float, float, float, float]:
    """由某个旋转角下的世界外框，反推 0° 时的四边。"""
    r = int(round(rot)) % 360
    if r == 0:
        return left, right, bottom, top
    if r == 180:
        return right, left, top, bottom
    if r == 90:
        return bottom, top, right, left
    if r == 270:
        return top, bottom, left, right
    raise ValueError(f"只支持 90° 整数倍，得到 {rot}")


def aabb_of(part: Part) -> tuple[float, float, float, float]:
    left, right, bottom, top = world_margins(part, part.rot)
    return part.x - left, part.y - bottom, part.x + right, part.y + top


def pin_world(part: Part) -> list[tuple[Pin, float, float]]:
    out = []
    for p in part.pins:
        wx, wy = rot_xy(p.lx, p.ly, part.rot)
        out.append((p, part.x + wx, part.y + wy))
    return out


def _kind(ref: str) -> str:
    if ref.startswith("U"):
        return "ic"
    if ref.startswith("C"):
        return "cap"
    if ref.startswith("R"):
        return "res"
    if ref.startswith("Y"):
        return "xtal"
    if ref.startswith("FB"):
        return "fb"
    return "other"


def load_design(path: Path) -> tuple[dict[str, Part], str | None]:
    """返回 (器件, 文件里写明的主芯片)。旧版 U34 JSON 没有主芯片字段。"""
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    raw0 = (data.get("components") or [{}])[0]
    if "symbols" in data and "symbol" in raw0:
        apply_units(str(data.get("units") or "mils"))
        return _load_xfer(data), data.get("mainComp") or None
    apply_units("mils")
    return _load_legacy(data), None


def _extents_from_symbol_bbox(bb: dict) -> tuple[float, float, float, float]:
    """封装 bbox 是相对原点、旋转 0° 的外框。"""
    return (
        max(0.0, -float(bb["x1"])),
        max(0.0, float(bb["x2"])),
        max(0.0, -float(bb["y1"])),
        max(0.0, float(bb["y2"])),
    )


def _load_xfer(data: dict) -> dict[str, Part]:
    symbols = data["symbols"]
    parts: dict[str, Part] = {}
    for raw in data["components"]:
        sym = symbols.get(raw["symbol"])
        if sym is None:
            raise RuntimeError(f"{raw['ref']} 的封装 {raw['symbol']} 不在 symbols 里")
        nx, px, ny, py = _extents_from_symbol_bbox(sym["bbox"])
        ox = float(raw["origin"]["x"])
        oy = float(raw["origin"]["y"])
        rot = float(raw["rotation"])
        pins = []
        for p in raw["pins"]:
            lx, ly = rot_xy(float(p["x"]) - ox, float(p["y"]) - oy, -rot)
            pins.append(Pin(str(p["pin"]), p.get("net") or "", lx, ly))
        ref = raw["ref"]
        parts[ref] = Part(ref, raw["layer"].upper(), ox, oy, rot, nx, px, ny, py, pins, _kind(ref))
    return parts


def _load_legacy(data: dict) -> dict[str, Part]:
    parts: dict[str, Part] = {}
    for raw in data["components"]:
        ox = float(raw["origin"]["x"])
        oy = float(raw["origin"]["y"])
        rot = float(raw["rotation"])
        bb = raw["bBox"]
        x1, y1, x2, y2 = float(bb["x1"]), float(bb["y1"]), float(bb["x2"]), float(bb["y2"])
        nx, px, ny, py = local_extents(rot, ox - x1, x2 - ox, oy - y1, y2 - oy)
        pins = []
        for p in raw["pins"]:
            dx = float(p["pin-xy"]["x"]) - ox
            dy = float(p["pin-xy"]["y"]) - oy
            lx, ly = rot_xy(dx, dy, -rot)
            pins.append(Pin(str(p["pinNumber"]), p.get("pinNetName") or "", lx, ly))
        ref = raw["refName"]
        part = Part(ref, raw["Layer"].upper(), ox, oy, rot, nx, px, ny, py, pins, _kind(ref))
        left, right, bottom, top = world_margins(part, rot)
        tol = 0.01 if UNIT_NAME == "mm" else 0.2
        if max(abs(ox - left - x1), abs(ox + right - x2), abs(oy - bottom - y1), abs(oy + top - y2)) > tol:
            raise RuntimeError(f"{ref} 的外框和旋转对不上")
        parts[ref] = part
    return parts


def detect_main(parts: dict[str, Part], hint: str) -> str:
    if hint in parts:
        return hint
    return max(parts.values(), key=lambda p: (len(p.pins), (p.nx + p.px) * (p.ny + p.py))).ref


def chip_pins(part: Part) -> list[ChipPin]:
    pins = []
    for p, x, y in pin_world(part):
        dx, dy = x - part.x, y - part.y
        if abs(dx) >= abs(dy):
            side = "R" if dx > 0 else "L"
        else:
            side = "T" if dy > 0 else "B"
        pins.append(ChipPin(p.number, p.net, x, y, side))
    return pins


def _rot_pair(part: Part, side: str) -> tuple[int, int]:
    """上下竖放、左右横放：长轴指向芯片外侧，沿边只占短边。"""
    long_x = (part.nx + part.px) >= (part.ny + part.py)
    if side in ("L", "R"):
        return (0, 180) if long_x else (90, 270)
    return (90, 270) if long_x else (0, 180)


def _outward(pin: Pin, rot: float, side: str) -> float:
    wx, wy = rot_xy(pin.lx, pin.ly, rot)
    if side == "R":
        return wx
    if side == "L":
        return -wx
    if side == "T":
        return wy
    return -wy


def choose_rot(part: Part, side: str, face: set[str]) -> float:
    a, b = _rot_pair(part, side)
    pins = [p for p in part.pins if p.net in face] or list(part.pins)
    if not pins:
        return float(a)

    def score(rot: float) -> float:
        return sum(_outward(p, rot, side) for p in pins) / len(pins)

    return float(a if score(a) <= score(b) else b)


def make_item(part: Part, side: str, anchor: float, aim: float, priority: int, face: set[str], owner: tuple) -> Item:
    rot = choose_rot(part, side, face)
    left, right, bottom, top = world_margins(part, rot)
    if side in ("L", "R"):
        back, front = bottom, top
        inward, outward = (right, left) if side == "L" else (left, right)
    else:
        back, front = left, right
        inward, outward = (top, bottom) if side == "B" else (bottom, top)
    return Item(
        part=part, side=side, anchor=anchor, aim=aim, rot=rot,
        back=back, front=front, inward=inward, outward=outward,
        priority=priority, face=set(face), owner=owner,
    )


def _span(items: list[Item], centers: list[float]) -> float:
    if not items:
        return 0.0
    lo = centers[0] - items[0].back
    hi = centers[-1] + items[-1].front
    return hi - lo


def _span_mid(items: list[Item], centers: list[float]) -> float:
    lo = centers[0] - items[0].back
    hi = centers[-1] + items[-1].front
    return (lo + hi) / 2.0


def _repair(items: list[Item], centers: list[float]) -> list[float]:
    """从中间向两端推开，保证外框不重叠，并且仍在格点上。"""
    cs = list(centers)
    if len(cs) <= 1:
        return [snap_grid(cs[0])] if cs else []
    mean = sum(cs) / len(cs)
    mid_i = min(range(len(cs)), key=lambda i: abs(cs[i] - mean))
    for i in range(mid_i + 1, len(cs)):
        min_c = cs[i - 1] + items[i - 1].front + ALONG_GAP + items[i].back
        if cs[i] < min_c - 1e-6:
            cs[i] = math.ceil((min_c - 1e-9) / GRID) * GRID
    for i in range(mid_i - 1, -1, -1):
        max_c = cs[i + 1] - items[i + 1].back - ALONG_GAP - items[i].front
        if cs[i] > max_c + 1e-6:
            cs[i] = math.floor((max_c + 1e-9) / GRID) * GRID
    return cs


def place_centers(items: list[Item], axis: float, mode: str) -> list[float]:
    """按 anchor 已经排好的顺序，挤成一排，再把整排挪到目标中心。"""
    if not items:
        return []
    cs: list[float] = []
    for i, it in enumerate(items):
        if i == 0:
            cs.append(0.0)
        else:
            cs.append(cs[-1] + items[i - 1].front + ALONG_GAP + it.back)
    mid = _span_mid(items, cs)
    pin_mid = sum(it.aim for it in items) / len(items)
    if mode == "pins":
        target = pin_mid
    elif mode == "center":
        target = axis
    else:
        target = pin_mid if abs(pin_mid - axis) <= SYMM_WINDOW else axis
    cs = [c + (target - mid) for c in cs]
    cs = [snap_grid(c) for c in cs]
    cs = _repair(items, cs)
    mid = _span_mid(items, cs)
    cs = [c + snap_grid(target - mid) for c in cs]
    return cs


def fit_row(items: list[Item], axis: float, limit: float, mode: str) -> tuple[list[Item], list[float], list[Item]]:
    """尽量全放进一排。放不下就先拿掉电源件，再拿掉离中心最远的。"""
    kept = sorted(items, key=lambda it: (it.anchor, it.priority, it.part.ref))
    dropped: list[Item] = []
    while kept:
        centers = place_centers(kept, axis, mode)
        if _span(kept, centers) <= limit + 0.5 or len(kept) == 1:
            for it, c in zip(kept, centers):
                it.along = c
            return kept, centers, dropped
        victim = max(kept, key=lambda it: (it.priority, abs(it.aim - axis), it.part.ref))
        kept.remove(victim)
        dropped.append(victim)
    return [], [], dropped


def pack_side(
    items: list[Item], axis: float, limit: float, mode: str, inner_limit: float | None = None,
) -> list[list[Item]]:
    """第 0 圈用对角线长度。退到外圈后改用芯片边长，避免伸进相邻边的角落。"""
    pool = list(items)
    rings: list[list[Item]] = []
    ring_idx = 0
    while pool and ring_idx < 10:
        eligible = [it for it in pool if it.min_ring <= ring_idx]
        later = [it for it in pool if it.min_ring > ring_idx]
        if not eligible:
            ring_idx += 1
            continue
        ring_limit = limit if ring_idx == 0 else (inner_limit if inner_limit is not None else limit)
        kept, _centers, dropped = fit_row(eligible, axis, ring_limit, mode)
        for it in kept:
            it.ring = ring_idx
        rings.append(kept)
        pool = dropped + later
        ring_idx += 1
    return rings


def apply_rings(
    side: str,
    rings: list[list[Item]],
    box: tuple[float, float, float, float],
    extra: float,
) -> None:
    x1, y1, x2, y2 = box
    sign = {"R": 1.0, "L": -1.0, "T": 1.0, "B": -1.0}[side]
    edge = {"R": x2, "L": x1, "T": y2, "B": y1}[side]
    cursor = edge
    first = True
    for ring in rings:
        if not ring:
            continue
        inward = max(it.inward for it in ring)
        gap = CLEARANCE + extra if first else RING_GAP
        radial = snap_outward(cursor + sign * (gap + inward), sign)
        for it in ring:
            p = it.part
            p.rot = it.rot
            p.side = side
            p.ring = it.ring
            p.priority = it.priority
            if side in ("L", "R"):
                p.x = radial
                p.y = it.along
            else:
                p.x = it.along
                p.y = radial
        outward = max(it.outward for it in ring)
        cursor = radial + sign * outward
        first = False


def overlap_pairs(parts: dict[str, Part], main: str) -> list[tuple[str, str]]:
    refs = [p for p in parts.values()]
    bad = []
    for i in range(len(refs)):
        a = aabb_of(refs[i])
        for j in range(i + 1, len(refs)):
            b = aabb_of(refs[j])
            if a[0] < b[2] - 1e-3 and b[0] < a[2] - 1e-3 and a[1] < b[3] - 1e-3 and b[1] < a[3] - 1e-3:
                bad.append((refs[i].ref, refs[j].ref))
    return bad


def chip_clearance(parts: dict[str, Part], main: str) -> float:
    chip = aabb_of(parts[main])
    best = float("inf")
    for p in parts.values():
        if p.ref == main:
            continue
        gap = _box_gap(aabb_of(p), chip)
        best = min(best, gap)
    return best


def _box_gap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    gap_x = max(a[0] - b[2], b[0] - a[2])
    gap_y = max(a[1] - b[3], b[1] - a[3])
    if gap_x >= 0 and gap_y >= 0:
        if gap_x > 0 and gap_y > 0:
            return math.hypot(gap_x, gap_y)
        return max(gap_x, gap_y)
    if gap_x >= 0:
        return gap_x
    if gap_y >= 0:
        return gap_y
    return max(gap_x, gap_y)


def _matching_supply_pin(part: Part, side: str, pins: list[ChipPin]) -> ChipPin | None:
    cands = [p for p in pins if p.side == side and p.net in part.nets() and is_power_net(p.net)]
    if not cands:
        return None
    live = [p for p in cands if not is_gnd_net(p.net)]
    return (live or cands)[0]


def build_items(parts: dict[str, Part], main: str, pins: list[ChipPin]) -> dict[str, list[Item]]:
    by_net: dict[str, list[ChipPin]] = defaultdict(list)
    for p in pins:
        if p.net:
            by_net[p.net].append(p)
    others = [p for p in parts.values() if p.ref != main]
    chip = parts[main]
    box = aabb_of(chip)
    axis = _axes(box)

    direct: dict[str, list[ChipPin]] = {}
    for p in others:
        hits: list[ChipPin] = []
        for net in p.nets():
            if net and not is_power_net(net):
                hits.extend(by_net.get(net, []))
        if hits:
            direct[p.ref] = hits

    owned: dict[str, list[ChipPin]] = dict(direct)
    inherited: set[str] = set()
    changed = True
    guard = 0
    while changed and guard < 8:
        changed = False
        guard += 1
        for p in others:
            if p.ref in owned:
                continue
            for net in p.nets():
                if not net or is_power_net(net):
                    continue
                for q in others:
                    if q.ref in owned and net in q.nets():
                        owned[p.ref] = owned[q.ref]
                        inherited.add(p.ref)
                        changed = True
                        break

    by_side: dict[str, list[Item]] = {"L": [], "R": [], "T": [], "B": []}

    def add_signal(part: Part, hits: list[ChipPin], priority: int) -> None:
        counts: dict[str, int] = defaultdict(int)
        for h in hits:
            counts[h.side] += 1
        side = max(counts, key=lambda s: (counts[s], s))
        side_hits = [h for h in hits if h.side == side]
        alongs = [h.along(side) for h in side_hits]
        aim = sum(alongs) / len(alongs)
        # 两端都接到这颗芯片、本身又是两脚器件：排到这一对引脚的外侧，给中间的电容 / 晶振让位
        bridge = priority == 0 and part.kind != "xtal" and len(part.pins) <= 2 and len(side_hits) >= 2
        anchor = (max(alongs) + 0.01) if bridge else aim
        face = {h.net for h in side_hits if h.net and not is_power_net(h.net)}
        if not face:
            face = {h.net for h in side_hits if h.net}
        owner = (side_hits[0].number, side_hits[0].net, side)
        by_side[side].append(make_item(part, side, anchor, aim, priority, face, owner))

    for p in others:
        if p.ref in direct:
            add_signal(p, direct[p.ref], 0)
        elif p.ref in inherited:
            add_signal(p, owned[p.ref], 1)

    # 只接电源 / 地的（去耦、磁珠）。轮流分到有电源脚的边，避免全挤在最空的一侧。
    limit = _limit(box)
    supply = [p for p in others if p.ref not in owned]
    supply_count = {s: 0 for s in "LRTB"}
    for part in sorted(supply, key=lambda p: (0 if p.kind == "cap" else 1, p.ref)):
        trials = []
        for side in "LRTB":
            pin = _matching_supply_pin(part, side, pins)
            if pin is None:
                continue
            face = {n for n in part.nets() if n and not is_gnd_net(n)} or set(part.nets())
            item = make_item(part, side, pin.along(side), pin.along(side), 2, face, (pin.number, pin.net, side))
            span = 0.0
            if by_side[side]:
                ordered = sorted(by_side[side], key=lambda it: (it.anchor, it.priority, it.part.ref))
                span = _span(ordered, place_centers(ordered, axis[side], "center"))
            remain = limit - span - (item.back + item.front)
            trials.append((remain, side, item))
        trials.sort(key=lambda t: (supply_count[t[1]], -t[0]))
        placed = False
        for _remain, side, item in trials:
            trial = by_side[side] + [item]
            _kept, _cs, dropped = fit_row(trial, axis[side], limit, "center")
            if all(it.part.ref != part.ref for it in dropped):
                by_side[side].append(item)
                supply_count[side] += 1
                placed = True
                break
        if not placed and trials:
            _remain, side, item = trials[0]
            item.min_ring = 1
            by_side[side].append(item)
        elif not placed:
            # 没有任何电源脚可跟：放到离原始坐标最近的一侧，避免漏件
            dx, dy = part.x - chip.x, part.y - chip.y
            side = ("R" if dx > 0 else "L") if abs(dx) >= abs(dy) else ("T" if dy > 0 else "B")
            face = set(part.nets())
            item = make_item(part, side, axis[side], axis[side], 2, face, ("", "", side))
            by_side[side].append(item)

    missing = [p.ref for p in others if all(p.ref != it.part.ref for items in by_side.values() for it in items)]
    if missing:
        raise RuntimeError(f"有器件没有归边: {missing}")
    return by_side


def _axes(box: tuple[float, float, float, float]) -> dict[str, float]:
    x1, y1, x2, y2 = box
    return {"L": (y1 + y2) / 2.0, "R": (y1 + y2) / 2.0, "T": (x1 + x2) / 2.0, "B": (x1 + x2) / 2.0}


def _limit(box: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return math.hypot(x2 - x1, y2 - y1) * (1.0 + DIAG_SLACK)


def place_module(parts: dict[str, Part], main: str, mode: str) -> dict:
    pins = chip_pins(parts[main])
    box = aabb_of(parts[main])
    axis = _axes(box)
    limit = _limit(box)
    by_side = build_items(parts, main, pins)
    extra = {s: 0.0 for s in "LRTB"}
    rings: dict[str, list[list[Item]]] = {}

    def repack() -> None:
        for side, items in by_side.items():
            side_len = (box[2] - box[0]) if side in ("T", "B") else (box[3] - box[1])
            rings[side] = pack_side(items, axis[side], limit, mode, inner_limit=side_len)
            apply_rings(side, rings[side], box, extra[side])

    repack()
    for _ in range(30):
        pairs = overlap_pairs(parts, main)
        gap = chip_clearance(parts, main)
        if not pairs and gap >= CLEARANCE - max(GRID * 0.05, 1e-6):
            break
        if pairs:
            victims = {_pick_victim(parts, a, b, box) for a, b in pairs}
            bumped = False
            for ref in victims:
                item = _find_item(by_side, ref)
                if item is None:
                    continue
                item.min_ring = max(item.min_ring, item.ring + 1)
                bumped = True
            if not bumped:
                break
        else:
            extra[_closest_side(parts, main)] += GRID
        repack()

    # 论文第二步的一维版：沿排滑动一格，飞线变短才收下。中心最多漂 1 格。
    slides = _slide(parts, by_side, limit, box, main)
    side_info = {}
    for side, ring_list in rings.items():
        flat = [it for ring in ring_list for it in ring]
        if not flat:
            continue
        centers = [it.along for it in flat if it.ring == flat[0].ring]
        # 每圈分开报跨度，第一圈是判定对象
        first = [it for it in flat if it.ring == 0]
        if first:
            cs = [it.along for it in first]
            span = _span(first, cs)
            mid = _span_mid(first, cs)
        else:
            span, mid = 0.0, axis[side]
        side_info[side] = {
            "span": span,
            "mid": mid,
            "axis": axis[side],
            "rings": len(ring_list),
            "refs": [(it.part.ref, it.ring, it.along) for it in flat],
        }
    return {
        "main": main,
        "mode": mode,
        "box": box,
        "limit": limit,
        "diag": math.hypot(box[2] - box[0], box[3] - box[1]),
        "sides": side_info,
        "slides": slides,
        "owners": {
            it.part.ref: it.owner
            for items in by_side.values() for it in items
        },
    }


def _pick_victim(parts: dict[str, Part], a: str, b: str, box: tuple[float, float, float, float]) -> str:
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0

    def key(ref: str) -> tuple:
        p = parts[ref]
        if p.ring < 0:
            return (-1, 0, 0.0)
        return (p.priority, p.ring, abs(p.x - cx) + abs(p.y - cy))

    return a if key(a) >= key(b) else b


def _find_item(by_side: dict[str, list[Item]], ref: str) -> Item | None:
    for items in by_side.values():
        for it in items:
            if it.part.ref == ref:
                return it
    return None


def _closest_side(parts: dict[str, Part], main: str) -> str:
    chip = aabb_of(parts[main])
    worst_side = "T"
    worst_gap = float("inf")
    for p in parts.values():
        if p.ref == main or not p.side:
            continue
        gap = _box_gap(aabb_of(p), chip)
        if gap < worst_gap:
            worst_gap = gap
            worst_side = p.side
    return worst_side


def _slide(parts: dict[str, Part], by_side: dict[str, list[Item]], limit: float, box, main: str) -> int:
    """每个器件尝试沿排 ±1 格。重叠、净空、跨度变了，或者整排中心漂出 1 格，都不收。"""
    locked = {}
    for side, items in by_side.items():
        groups: dict[int, list[Item]] = defaultdict(list)
        for it in items:
            groups[it.ring].append(it)
        for ring, group in groups.items():
            ordered = sorted(group, key=lambda it: it.along)
            locked[(side, ring)] = _span_mid(ordered, [it.along for it in ordered])
    moves = 0
    base = signal_hpwl(parts)
    for _pass in range(2):
        improved = False
        for side, items in by_side.items():
            for it in sorted(items, key=lambda s: s.part.ref):
                for delta in (GRID, -GRID):
                    old = it.along
                    it.along = old + delta
                    _write_along(it)
                    if _slide_ok(parts, it, items, side, limit, box, main, locked) and signal_hpwl(parts) < base - max(GRID * 0.05, 1e-6):
                        base = signal_hpwl(parts)
                        moves += 1
                        improved = True
                        break
                    it.along = old
                    _write_along(it)
        if not improved:
            break
    return moves


def _write_along(it: Item) -> None:
    if it.side in ("L", "R"):
        it.part.y = it.along
    else:
        it.part.x = it.along


def _slide_ok(parts, moved: Item, items: list[Item], side: str, limit: float, box, main: str, locked: dict) -> bool:
    if _hits_any(parts, moved.part.ref):
        return False
    if _box_gap(aabb_of(moved.part), aabb_of(parts[main])) < CLEARANCE - max(GRID * 0.02, 1e-6):
        return False
    rings: dict[int, list[Item]] = defaultdict(list)
    for it in items:
        rings[it.ring].append(it)
    for ring, group in rings.items():
        ordered = sorted(group, key=lambda it: it.along)
        cs = [it.along for it in ordered]
        if _span(ordered, cs) > limit + GRID * 0.2:
            return False
        if abs(_span_mid(ordered, cs) - locked[(side, ring)]) > GRID * 1.05:
            return False
        for i in range(1, len(ordered)):
            need = ordered[i - 1].front + ALONG_GAP + ordered[i].back
            if cs[i] < cs[i - 1] + need - 1e-3:
                return False
    return True


def _hits_any(parts: dict[str, Part], ref: str) -> bool:
    """只查刚挪过的那一颗，结果和全表两两检查这一颗是一样的。"""
    a = aabb_of(parts[ref])
    for p in parts.values():
        if p.ref == ref:
            continue
        b = aabb_of(p)
        if a[0] < b[2] - 1e-3 and b[0] < a[2] - 1e-3 and a[1] < b[3] - 1e-3 and b[1] < a[3] - 1e-3:
            return True
    return False


def net_points(parts: dict[str, Part]) -> dict[str, list[tuple[str, float, float]]]:
    acc: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    for p in parts.values():
        for pin, x, y in pin_world(p):
            if not pin.net:
                continue
            acc[pin.net][p.ref].append((x, y))
    out: dict[str, list[tuple[str, float, float]]] = {}
    for net, by_ref in acc.items():
        pts = []
        for ref, xy in by_ref.items():
            pts.append((ref, sum(t[0] for t in xy) / len(xy), sum(t[1] for t in xy) / len(xy)))
        out[net] = pts
    return out


def signal_hpwl(parts: dict[str, Part]) -> float:
    total = 0.0
    for net, pts in net_points(parts).items():
        if is_power_net(net) or len(pts) < 2:
            continue
        xs = [p[1] for p in pts]
        ys = [p[2] for p in pts]
        total += (max(xs) - min(xs)) + (max(ys) - min(ys))
    return total


def signal_mst(parts: dict[str, Part]) -> float:
    total = 0.0
    for net, pts in net_points(parts).items():
        if is_power_net(net) or len(pts) < 2:
            continue
        total += sum(e[2] for e in mst_edges([(x, y) for _r, x, y in pts]))
    return total


def measure(parts: dict[str, Part], info: dict) -> dict:
    main = info["main"]
    off = 0
    orient_bad = []
    extra = 0
    for p in parts.values():
        if p.ref == main:
            continue
        if abs(p.x / GRID - round(p.x / GRID)) > 1e-6 or abs(p.y / GRID - round(p.y / GRID)) > 1e-6:
            off += 1
        if p.ring > 0:
            extra += 1
        left, right, bottom, top = world_margins(p, p.rot)
        wide = (left + right) >= (bottom + top) - 0.5
        if p.side in ("L", "R") and not wide:
            orient_bad.append(p.ref)
        if p.side in ("T", "B") and wide and abs((left + right) - (bottom + top)) > 1.0:
            orient_bad.append(p.ref)
    sym = 0.0
    for side, s in info["sides"].items():
        sym = max(sym, abs(s["mid"] - s["axis"]))
    return {
        "hpwl": signal_hpwl(parts),
        "mst": signal_mst(parts),
        "overlap": len(overlap_pairs(parts, main)),
        "clearance": chip_clearance(parts, main),
        "off_grid": off,
        "extra_rings": extra,
        "orient_bad": orient_bad,
        "sym": sym,
        "slides": info.get("slides", 0),
        "mode": info["mode"],
        "part_gap": min_part_gap(parts, main),
    }


def _score(m: dict) -> tuple:
    """越小越好。先合法，再比对称，外圈数量和飞线放在后面。"""
    return (
        m["overlap"],
        0 if m["clearance"] >= CLEARANCE - max(GRID * 0.05, 1e-6) else 1,
        m["off_grid"],
        len(m["orient_bad"]),
        round(m["sym"], 1),
        m["extra_rings"],
        round(m["hpwl"], 1),
    )


def min_part_gap(parts: dict[str, Part], main: str) -> float:
    refs = [p for p in parts.values() if p.ref != main]
    best = float("inf")
    for i, a in enumerate(refs):
        box_a = aabb_of(a)
        for b in refs[i + 1:]:
            best = min(best, _box_gap(box_a, aabb_of(b)))
    return best


def snapshot(parts: dict[str, Part]) -> dict[str, tuple]:
    return {
        p.ref: (p.x, p.y, p.rot, p.layer, p.side, p.ring, p.priority)
        for p in parts.values()
    }


def restore(parts: dict[str, Part], snap: dict[str, tuple]) -> None:
    for ref, rec in snap.items():
        p = parts[ref]
        p.x, p.y, p.rot, p.layer, p.side, p.ring, p.priority = rec


def _setup_font() -> None:
    for path in (
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyh.ttf",
        r"C:\Windows\Fonts\simhei.ttf",
    ):
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            name = font_manager.FontProperties(fname=path).get_name()
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


def _style(part: Part) -> dict:
    color = {
        "ic": "#8B95A1",
        "res": "#4C78A8",
        "cap": "#F2A03D",
        "xtal": "#59A14F",
        "fb": "#B07AA1",
    }.get(part.kind, "#76B7B2")
    hatch = "//" if part.layer == "BOTTOM" else None
    lw = 1.3 if part.ring > 0 else 0.6
    return {"facecolor": color, "edgecolor": "#1F2933", "hatch": hatch, "lw": lw, "alpha": 0.92}


def _net_color(net: str):
    if is_gnd_net(net):
        return (0.45, 0.45, 0.45, 0.35)
    if is_power_net(net):
        return (0.85, 0.35, 0.15, 0.45)
    palette = [
        "#1B4F72", "#117A65", "#6C3483", "#922B21", "#1A5276",
        "#196F3D", "#7D3C98", "#9A7D0A", "#1F618D", "#4A235A",
    ]
    return palette[abs(hash(net)) % len(palette)]


def _draw_on(ax, parts: dict[str, Part], main: str, title: str, subtitle: str, rules: dict | None) -> None:
    ax.set_aspect("equal")
    crowded = len(parts) > 80
    for net, pts in net_points(parts).items():
        if len(pts) < 2:
            continue
        # 器件一多，地网和电源网会把图涂满，信号网仍然画
        if crowded and is_power_net(net):
            continue
        xs = [(x, y) for _r, x, y in pts]
        color = _net_color(net)
        power = is_power_net(net)
        for i, j, _d in mst_edges(xs):
            ax.plot(
                [xs[i][0], xs[j][0]], [xs[i][1], xs[j][1]],
                color=color, lw=0.6 if power else 1.0,
                ls=(0, (2, 2)) if power else "solid",
                zorder=1, solid_capstyle="round",
            )
    label_all = len(parts) <= 60
    for p in parts.values():
        x1, y1, x2, y2 = aabb_of(p)
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, zorder=2, **_style(p)))
        if label_all or p.ref == main:
            ax.text(
                p.x, p.y, p.ref, ha="center", va="center",
                fontsize=6.2 if label_all else 8, color="#111", zorder=4,
                bbox={"boxstyle": "round,pad=0.05", "fc": "white", "ec": "none", "alpha": 0.7},
            )
    chip = parts[main]
    pxs = []
    pys = []
    for _pin, x, y in pin_world(chip):
        pxs.append(x)
        pys.append(y)
    if pxs:
        ax.plot(pxs, pys, "o", ms=1.2 if len(pxs) > 200 else 2.0, color="#1F2933", zorder=5)

    if rules:
        x1, y1, x2, y2 = rules["box"]
        ax.plot([x1, x2], [y1, y2], ls="--", color="#C0392B", lw=0.8, zorder=0)
        ax.plot([x1, x2], [y2, y1], ls="--", color="#C0392B", lw=0.8, zorder=0)
        limit = rules["limit"]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        # 四边各画一条允许长度。偏移按净空缩放，mm 不能再用 18 mil 那种常数。
        tick = max(CLEARANCE * 0.36, GRID)
        half = limit / 2.0
        mark = [
            ([cx - half, cx + half], [y2 + tick, y2 + tick]),
            ([cx - half, cx + half], [y1 - tick, y1 - tick]),
            ([x1 - tick, x1 - tick], [cy - half, cy + half]),
            ([x2 + tick, x2 + tick], [cy - half, cy + half]),
        ]
        for xs, ys in mark:
            ax.plot(xs, ys, color="#E67E22", lw=1.4, solid_capstyle="butt", zorder=0)
        ax.text(
            cx, y2 + tick * 1.55, f"允许 {fmt_num(limit)} {UNIT_NAME}",
            color="#E67E22", ha="center", va="bottom", fontsize=7,
        )

    xs, ys = [], []
    for p in parts.values():
        a, b, c, d = aabb_of(p)
        xs += [a, c]
        ys += [b, d]
    pad = CLEARANCE * 2.0
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_xlabel(f"X（{UNIT_NAME}）")
    ax.set_ylabel(f"Y（{UNIT_NAME}）")
    ax.set_title(f"{title}\n{subtitle}", fontsize=9, pad=8)
    ax.grid(True, ls=":", lw=0.4, color="#bbb")


def _legend(ax) -> None:
    handles = [
        Rectangle((0, 0), 1, 1, facecolor="#8B95A1", edgecolor="#1F2933", label="主芯片"),
        Rectangle((0, 0), 1, 1, facecolor="#4C78A8", edgecolor="#1F2933", label="电阻"),
        Rectangle((0, 0), 1, 1, facecolor="#F2A03D", edgecolor="#1F2933", label="电容（顶层）"),
        Rectangle((0, 0), 1, 1, facecolor="#F2A03D", edgecolor="#1F2933", hatch="//", label="电容（底层）"),
        Rectangle((0, 0), 1, 1, facecolor="#59A14F", edgecolor="#1F2933", label="晶振"),
        Rectangle((0, 0), 1, 1, facecolor="#B07AA1", edgecolor="#1F2933", label="磁珠"),
        plt.Line2D([0], [0], color="#1B4F72", lw=1.2, label="信号网"),
        plt.Line2D([0], [0], color="#D35400", lw=1.0, ls=(0, (2, 2)), label="电源网"),
        plt.Line2D([0], [0], color="#C0392B", lw=0.8, ls="--", label="芯片对角线"),
        plt.Line2D([0], [0], color="#E67E22", lw=1.4, label="一排允许长度"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=7, framealpha=0.92)


def draw(parts: dict[str, Part], path: Path, title: str, subtitle: str, main: str, rules: dict | None) -> None:
    _setup_font()
    fig, ax = plt.subplots(figsize=(13.2, 11.2), dpi=140)
    _draw_on(ax, parts, main, title, subtitle, rules)
    _legend(ax)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def draw_compare(panels: list[tuple[dict[str, Part], str, str]], path: Path, main: str, rules: dict) -> None:
    _setup_font()
    fig, axes = plt.subplots(1, len(panels), figsize=(8.4 * len(panels), 8.6), dpi=130)
    if len(panels) == 1:
        axes = [axes]
    for ax, (snap_parts, title, subtitle) in zip(axes, panels):
        _draw_on(ax, snap_parts, main, title, subtitle, rules)
    _legend(axes[-1])
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _subtitle(m: dict, info: dict) -> str:
    spans = []
    for side in "T B L R":
        s = info["sides"].get(side)
        if not s:
            continue
        name = {"T": "上", "B": "下", "L": "左", "R": "右"}[side]
        spans.append(f"{name}{fmt_num(s['span'])}")
    return (
        f"半周长 {fmt_num(m['hpwl'])} {UNIT_NAME}    飞线 {fmt_num(m['mst'])}    重叠 {m['overlap']}"
        f"    离芯片 {fmt_num(m['clearance'])}    器件缝 {m['part_gap']:.2f}"
        f"    中心偏离 {fmt_num(m['sym'])}\n"
        f"跨度 {' / '.join(spans)} {UNIT_NAME}，允许 {fmt_num(info['limit'])}"
        f"（对角线 {fmt_num(info['diag'])} 的 {1 + DIAG_SLACK:.0%}）"
        f"    格点 {GRID:g} {UNIT_NAME}"
    )


MODE_NAME = {
    "auto": "自动（偏离不超过对称窗就跟引脚重心）",
    "center": "严格居中",
    "pins": "贴引脚重心",
}


def dump_placed(parts: dict[str, Part], info: dict, metrics: dict, path: Path) -> None:
    rows = []
    for p in parts.values():
        if p.ref == info["main"]:
            continue
        rows.append({
            "ref": p.ref,
            "layer": p.layer,
            "x": snap_grid(p.x),
            "y": snap_grid(p.y),
            "rotation": p.rot,
            "side": p.side,
            "ring": p.ring,
            "owner_pin": list(info["owners"].get(p.ref, ())),
        })
    rows.sort(key=lambda r: r["ref"])
    doc = {
        "method": info["mode"],
        "unit": UNIT_NAME,
        "grid": GRID,
        "clearance": CLEARANCE,
        "diag_slack": DIAG_SLACK,
        "sym_window": SYMM_WINDOW,
        "metrics": {
            "signal_hpwl": round(metrics["hpwl"], 3),
            "signal_mst": round(metrics["mst"], 3),
            "overlap": metrics["overlap"],
            "clearance": round(metrics["clearance"], 3),
            "symmetry_error": round(metrics["sym"], 3),
            "extra_ring_parts": metrics["extra_rings"],
            "slides": metrics["slides"],
        },
        "components": rows,
    }
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")


def _clone_placed(parts: dict[str, Part], snap: dict[str, tuple]) -> dict[str, Part]:
    """比较图要用两套坐标同时画，所以按快照复制一份零件字典。"""
    import copy
    cloned = copy.deepcopy(parts)
    restore(cloned, snap)
    return cloned


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    paths = [Path(a) for a in sys.argv[1:]] or [SRC]
    for path in paths:
        run_file(path)


def run_file(path: Path) -> None:
    hint = path.stem.split("_")[0]
    parts, named_main = load_design(path)
    main_ref = named_main if named_main in parts else detect_main(parts, hint)
    out_dir = path.parent / "out"
    out_dir.mkdir(exist_ok=True)
    stem = path.stem
    out_before = out_dir / f"{stem}_before.png"
    out_after = out_dir / f"{stem}_after.png"
    out_compare = out_dir / f"{stem}_compare.png"
    out_json = out_dir / f"{stem}_placed.json"

    print(f"\n######## {path.name}  主芯片 {main_ref}  {len(parts)} 颗  单位 {UNIT_NAME}  格点 {GRID:g}  净空 {CLEARANCE:g}")
    original = snapshot(parts)
    before_hpwl = signal_hpwl(parts)
    before_mst = signal_mst(parts)
    before_ov = overlap_pairs(parts, main_ref)
    box = aabb_of(parts[main_ref])
    diag = math.hypot(box[2] - box[0], box[3] - box[1])
    rules = {"box": box, "limit": diag * (1.0 + DIAG_SLACK), "diag": diag}

    draw(
        parts, out_before,
        f"摆放前：{main_ref}",
        f"半周长 {fmt_num(before_hpwl)} {UNIT_NAME}    飞线 {fmt_num(before_mst)}    "
        f"重叠 {len(before_ov)} 处    格点 {GRID:g} {UNIT_NAME}",
        main_ref, rules,
    )

    results = []
    for mode in ("center", "pins", "auto"):
        restore(parts, original)
        info = place_module(parts, main_ref, mode)
        metrics = measure(parts, info)
        results.append((mode, snapshot(parts), info, metrics))
        print(f"\n=== {MODE_NAME[mode]} ===")
        print(
            f"半周长 {metrics['hpwl']:.1f}  飞线 {metrics['mst']:.1f}  "
            f"重叠 {metrics['overlap']}  净空 {metrics['clearance']:.2f}  "
            f"器件缝 {metrics['part_gap']:.2f}  偏离 {metrics['sym']:.2f}  "
            f"外圈 {metrics['extra_rings']}  出格 {metrics['off_grid']}  "
            f"朝向错 {metrics['orient_bad']}  滑动 {metrics['slides']}  评分 {_score(metrics)}"
        )
        for side in "T B L R":
            s = info["sides"].get(side)
            if not s:
                continue
            refs = s["refs"]
            if len(refs) > 18:
                names = f"{len(refs)} 颗，其中外圈 {sum(1 for _r, ring, _a in refs if ring)} 颗"
            else:
                names = ", ".join(f"{ref}{'*' if ring else ''}" for ref, ring, _a in refs)
            print(
                f"  {side} 跨度 {s['span']:.2f} / {info['limit']:.2f}  "
                f"中心 {s['mid']:.2f} (芯片 {s['axis']:.2f})  {names}"
            )

    winner = min(results, key=lambda r: _score(r[3]))
    mode, snap, info, metrics = winner
    restore(parts, snap)
    print(f"\n选用：{MODE_NAME[mode]}")
    print(f"相对原始坐标，信号半周长 {fmt_num(before_hpwl)} -> {fmt_num(metrics['hpwl'])} {UNIT_NAME}")

    draw(
        parts, out_after,
        f"摆放后：{MODE_NAME[mode]}",
        _subtitle(metrics, info),
        main_ref, rules,
    )
    panels = []
    for m, s, inf, met in results:
        if m == "auto":
            continue
        mark = "（选用）" if m == mode else ""
        panels.append((
            _clone_placed(parts, s),
            f"{MODE_NAME[m]}{mark}",
            _subtitle(met, inf),
        ))
    draw_compare(panels, out_compare, main_ref, rules)
    dump_placed(parts, info, metrics, out_json)
    print("图:", out_before)
    print("图:", out_after)
    print("图:", out_compare)


if __name__ == "__main__":
    main()
