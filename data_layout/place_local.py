# -*- coding: utf-8 -*-
"""
小模块先局部摆好，再当成一块放回主芯片旁边。不改 place_u34.py，图写到 out_local。

对照三种做法：
  radial  现有规则：靠上/下的器件竖放，靠左/右的横放，一颗一颗排
  free    放开朝向：每颗自己选 0/90/180/270，哪样让信号脚更朝芯片就用哪样，仍是一颗一颗排
  photo   先认出晶振小组，按参考图摆成一块（反馈电阻靠近管脚、两颗负载电容并排且长边顺着芯片边、晶振在最外），再塞回原位，旁边的器件让开

运行：
    .\\.venv\\Scripts\\python.exe data_layout\\place_local.py data_layout\\U34_Comps.json
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import place_u34 as base

CRYSTAL_WORDS = ("OSC", "XTAL", "X32K", "X48M", "32KHZ", "48MHZ")
HERE = Path(__file__).resolve().parent


def is_crystal_net(name: str) -> bool:
    u = (name or "").upper()
    return any(k in u for k in CRYSTAL_WORDS)


def is_crystal_part(part: base.Part) -> bool:
    if part.kind == "xtal":
        return True
    ref = part.ref
    return len(ref) > 1 and ref[0] in "XY" and ref[1].isdigit()


def crystal_members(parts: dict[str, base.Part], main: str) -> list[base.Part]:
    crystals = [p for p in parts.values() if p.ref != main and is_crystal_part(p)]
    if not crystals:
        return []
    nets = set()
    for crystal in crystals:
        nets |= {n for n in crystal.nets() if is_crystal_net(n)}
    members = []
    for part in parts.values():
        if part.ref == main:
            continue
        if is_crystal_part(part) or (nets and part.nets() & nets):
            members.append(part)
    return members


def group_of(members: list[base.Part], nets: set[str]) -> dict:
    """把一颗晶振和只连到它那对振荡网的电容、电阻配在一起。"""
    crystal = next(p for p in members if is_crystal_part(p) and (p.nets() & nets))
    caps = [p for p in members if p.kind == "cap" and len(p.nets() & nets) == 1 and not is_crystal_part(p)]
    feedback = [
        p for p in members
        if p.kind == "res" and nets <= p.nets() | {n for n in p.nets() if not is_crystal_net(n)}
        and len(p.nets() & nets) >= 2
    ]
    # 上面的条件写复杂了，电阻只要两端都是这对振荡网即可
    feedback = [p for p in members if p.kind == "res" and len(p.nets() & nets) >= 2]
    return {"crystal": crystal, "caps": caps, "feedback": feedback[:1], "nets": nets}


def split_groups(members: list[base.Part]) -> list[dict]:
    crystals = [p for p in members if is_crystal_part(p)]
    groups = []
    for crystal in crystals:
        nets = {n for n in crystal.nets() if is_crystal_net(n)}
        if len(nets) < 2:
            continue
        related = [crystal] + [
            p for p in members
            if p.ref != crystal.ref and not is_crystal_part(p) and (p.nets() & nets)
        ]
        groups.append(group_of(related, nets))
    return groups


def _long_x(part: base.Part) -> bool:
    return (part.nx + part.px) >= (part.ny + part.py)


def rots_long_along(part: base.Part, side: str, along_edge: bool) -> tuple[int, int]:
    """along_edge=True：长轴顺着芯片边。False：长轴指向芯片外侧。"""
    horizontal_edge = side in ("T", "B")
    want_horizontal = along_edge if horizontal_edge else not along_edge
    if want_horizontal:
        return (0, 180) if _long_x(part) else (90, 270)
    return (90, 270) if _long_x(part) else (0, 180)


def pin_world_at(part: base.Part, rot: float, x: float, y: float, net: str) -> tuple[float, float] | None:
    hits = []
    for pin in part.pins:
        if pin.net != net:
            continue
        wx, wy = base.rot_xy(pin.lx, pin.ly, rot)
        hits.append((x + wx, y + wy))
    if not hits:
        return None
    return (sum(p[0] for p in hits) / len(hits), sum(p[1] for p in hits) / len(hits))


def choose_flip(part: base.Part, rots: tuple[int, int], score) -> float:
    return float(min(rots, key=score))


def inward_extent(part: base.Part, rot: float, side: str) -> tuple[float, float]:
    """朝向芯片的那一边、背向芯片的那一边。"""
    left, right, bottom, top = base.world_margins(part, rot)
    if side == "T":
        return bottom, top
    if side == "B":
        return top, bottom
    if side == "R":
        return left, right
    return right, left


def along_extent(part: base.Part, rot: float, side: str) -> tuple[float, float]:
    """沿边负方向、正方向的外框。"""
    left, right, bottom, top = base.world_margins(part, rot)
    if side in ("T", "B"):
        return left, right
    return bottom, top


def put(part: base.Part, side: str, along: float, outward: float, rot: float) -> None:
    part.rot = rot
    part.side = side
    part.ring = 0
    if side in ("T", "B"):
        part.x, part.y = along, outward
    else:
        part.x, part.y = outward, along


def osc_pins(pins: list[base.ChipPin], nets: set[str]) -> list[base.ChipPin]:
    return [p for p in pins if p.net in nets]


def side_of(pins: list[base.ChipPin]) -> str:
    counts: dict[str, int] = {}
    for pin in pins:
        counts[pin.side] = counts.get(pin.side, 0) + 1
    return max(counts, key=lambda s: counts[s])


def gap_part() -> float:
    return 10.0 if base.UNIT_NAME == "mil" else 0.25


def local_length(parts: dict[str, base.Part], nets: set[str]) -> float:
    total = 0.0
    for net, pts in base.net_points(parts).items():
        if net not in nets or len(pts) < 2:
            continue
        total += sum(e[2] for e in base.mst_edges([(x, y) for _r, x, y in pts]))
    return total


def _snap_out(edge: float, sign: float, inward: float, gap: float) -> float:
    return base.snap_outward(edge + sign * (gap + inward), sign)


def layout_group(
    parts: dict[str, base.Part],
    group: dict,
    side: str,
    edge: float,
    mid: float,
    left_net: str,
    right_net: str,
    style: str,
) -> None:
    """style: photo / radial_stack / crystal_near / row。坐标是沿边 + 外侧。"""
    sign = {"T": 1.0, "B": -1.0, "R": 1.0, "L": -1.0}[side]
    crystal: base.Part = group["crystal"]
    caps: list[base.Part] = list(group["caps"])
    feedback = group["feedback"][0] if group["feedback"] else None
    left_cap = next((c for c in caps if left_net in c.nets()), None)
    right_cap = next((c for c in caps if right_net in c.nets()), None)
    gap = gap_part()

    def face_along(part: base.Part, rots: tuple[int, int], net: str, want_negative: bool) -> float:
        def score(rot: int) -> float:
            # 先放到 mid 上估引脚的沿边坐标，只用来选翻转
            pos = pin_world_at(part, rot, 0.0, 0.0, net)
            if pos is None:
                return 0.0
            along = pos[0] if side in ("T", "B") else pos[1]
            return along if want_negative else -along
        return choose_flip(part, rots, score)

    tangential = style == "photo"
    if style == "radial_stack":
        tangential = False

    if style == "row":
        # 四颗顺着边排成一排，长轴朝外，贴近管脚中点
        row = [p for p in (left_cap, crystal, right_cap, feedback) if p is not None]
        cursor_edge = edge
        along_cursor = None
        centers = []
        rots = []
        for part in row:
            rot = face_along(part, rots_long_along(part, side, False), left_net if left_net in part.nets() else right_net, True)
            back, front = along_extent(part, rot, side)
            inward, _out = inward_extent(part, rot, side)
            if along_cursor is None:
                along_cursor = mid
                center_along = mid
            else:
                center_along = along_cursor
            centers.append(center_along)
            rots.append(rot)
            along_cursor = center_along + front + gap + back
        # 把整排中心挪回 mid
        if centers:
            # 重新按宽度从中点向两侧排
            widths = []
            for part, rot in zip(row, rots):
                back, front = along_extent(part, rot, side)
                widths.append(back + front)
            total = sum(widths) + gap * (len(row) - 1)
            cursor = mid - total / 2.0
            inward_max = 0.0
            sized = []
            for part, rot, width in zip(row, rots, widths):
                back, front = along_extent(part, rot, side)
                inward, outward = inward_extent(part, rot, side)
                sized.append((part, rot, cursor + back, inward, outward))
                cursor += width + gap
                inward_max = max(inward_max, inward)
            outward = _snap_out(edge, sign, inward_max, base.CLEARANCE)
            for part, rot, along, _inw, _outw in sized:
                put(part, side, base.snap_grid(along), outward, rot)
        return

    if style == "crystal_near":
        # 晶振贴近管脚，两颗电容分在左右，反馈电阻再往外
        rot_c = 0.0
        best = None
        for rot in (0, 90, 180, 270):
            # 暂放在原点，看 OSC 左网的引脚是否在沿边负方向
            pos_l = pin_world_at(crystal, rot, 0.0, 0.0, left_net)
            pos_r = pin_world_at(crystal, rot, 0.0, 0.0, right_net)
            if pos_l is None or pos_r is None:
                continue
            along_l = pos_l[0] if side in ("T", "B") else pos_l[1]
            along_r = pos_r[0] if side in ("T", "B") else pos_r[1]
            outward_l = pos_l[1] if side in ("T", "B") else pos_l[0]
            # 左网在左，并且信号脚更朝芯片（朝外坐标更小）
            key = (0 if along_l < along_r else 1, abs(outward_l))
            if best is None or key < best[0]:
                best = (key, rot)
        rot_c = best[1] if best else 0.0
        inw, outw = inward_extent(crystal, rot_c, side)
        c_out = _snap_out(edge, sign, inw, base.CLEARANCE)
        put(crystal, side, base.snap_grid(mid), c_out, rot_c)
        crystal_outer = c_out + sign * outw
        # 电容放在晶振左右，沿边中心与晶振的左右两侧对齐
        back_c, front_c = along_extent(crystal, rot_c, side)
        cap_out_edge = crystal_outer
        for cap, sign_along in ((left_cap, -1.0), (right_cap, 1.0)):
            if cap is None:
                continue
            rot = face_along(cap, rots_long_along(cap, side, False), left_net if cap is left_cap else right_net, sign_along < 0)
            back, front = along_extent(cap, rot, side)
            inward, outward = inward_extent(cap, rot, side)
            along = mid + sign_along * (front_c + gap + back)
            cout = _snap_out(edge, sign, inward, base.CLEARANCE)
            # 与晶振同一深度会重叠，改放到晶振外侧若左右放不下；这里左右放，深度与晶振中心相同会重叠
            # 左右放时沿边已经躲开晶振，深度可以贴近芯片
            put(cap, side, base.snap_grid(along), cout, rot)
            cap_out_edge = max(cap_out_edge, cout + sign * outward) if sign > 0 else min(cap_out_edge, cout + sign * outward)
        if feedback is not None:
            rot = face_along(feedback, rots_long_along(feedback, side, True), left_net, True)
            inward, outward = inward_extent(feedback, rot, side)
            prev = cap_out_edge if sign > 0 else cap_out_edge
            # prev 是已占用的外侧边界。sign>0 时 cap_out_edge 是最大的外侧坐标
            fout = _snap_out(prev, sign, inward, gap) if sign > 0 else _snap_out(prev, sign, inward, gap)
            put(feedback, side, base.snap_grid(mid), fout, rot)
        return

    # photo / radial_stack：从芯片向外是 电阻 -> 两颗电容并排 -> 晶振
    along_edge = tangential
    cursor = edge
    first = True

    def stack_one(part: base.Part, rot: float, along: float) -> float:
        nonlocal cursor, first
        inward, outward = inward_extent(part, rot, side)
        gap_here = base.CLEARANCE if first else gap
        center = _snap_out(cursor, sign, inward, gap_here)
        put(part, side, base.snap_grid(along), center, rot)
        body_outer = center + sign * outward
        if sign > 0:
            cursor = max(cursor, body_outer) if not first else body_outer
        else:
            cursor = min(cursor, body_outer) if not first else body_outer
        first = False
        return body_outer

    if feedback is not None:
        rot_r = face_along(feedback, rots_long_along(feedback, side, along_edge), left_net, True)
        stack_one(feedback, rot_r, mid)
    # 两颗电容同一深度。信号脚朝中间（掉头），到晶振和反馈电阻的线才短。
    # 左电容的振荡脚在右侧，右电容的振荡脚在左侧。
    cap_pair = [(left_cap, left_net, False), (right_cap, right_net, True)]
    cap_pair = [item for item in cap_pair if item[0] is not None]
    if cap_pair:
        prepared = []
        for cap, net, want_neg in cap_pair:
            rot = face_along(cap, rots_long_along(cap, side, along_edge), net, want_neg)
            back, front = along_extent(cap, rot, side)
            inward, outward = inward_extent(cap, rot, side)
            prepared.append((cap, rot, back, front, inward, outward))
        total = sum(p[2] + p[3] for p in prepared) + gap * (len(prepared) - 1)
        cursor_along = mid - total / 2.0
        inward_max = max(p[4] for p in prepared)
        gap_here = base.CLEARANCE if first else gap
        center = _snap_out(cursor, sign, inward_max, gap_here)
        outer_body = center
        for cap, rot, back, front, _inw, outward in prepared:
            along = cursor_along + back
            put(cap, side, base.snap_grid(along), center, rot)
            cursor_along += back + front + gap
            body_outer = center + sign * outward
            outer_body = max(outer_body, body_outer) if sign > 0 else min(outer_body, body_outer)
        cursor = outer_body
        first = False
    # 晶振在最外，左网在左
    best = None
    for rot in (0, 90, 180, 270):
        pos_l = pin_world_at(crystal, rot, 0.0, 0.0, left_net)
        pos_r = pin_world_at(crystal, rot, 0.0, 0.0, right_net)
        if pos_l is None or pos_r is None:
            continue
        along_l = pos_l[0] if side in ("T", "B") else pos_l[1]
        along_r = pos_r[0] if side in ("T", "B") else pos_r[1]
        out_l = pos_l[1] if side in ("T", "B") else pos_l[0]
        out_r = pos_r[1] if side in ("T", "B") else pos_r[1] if False else pos_r[0]
        # 朝芯片的方向：T 侧希望信号脚的局部 y 更小（放上之后更靠近芯片）
        key = (0 if along_l < along_r else 1, out_l + out_r)
        if best is None or key < best[0]:
            best = (key, rot)
    rot_y = best[1] if best else 0.0
    stack_one(crystal, rot_y, mid)


def push_aside(parts: dict[str, base.Part], main: str, refs: set[str]) -> None:
    """晶振块占住的位置，同一侧的其他器件从块的两边向外排开，不再叠在同一点。"""
    gap = gap_part()
    if not refs:
        return
    boxes = [base.aabb_of(parts[ref]) for ref in refs]
    block = (
        min(b[0] for b in boxes), min(b[1] for b in boxes),
        max(b[2] for b in boxes), max(b[3] for b in boxes),
    )
    side = parts[next(iter(refs))].side

    def along_center(part: base.Part) -> float:
        return part.y if side in ("L", "R") else part.x

    def along_half(part: base.Part) -> float:
        box = base.aabb_of(part)
        if side in ("L", "R"):
            return (box[3] - box[1]) / 2.0
        return (box[2] - box[0]) / 2.0

    def set_along(part: base.Part, value: float) -> None:
        if side in ("L", "R"):
            part.y = value
        else:
            part.x = value

    block_lo = block[1] if side in ("L", "R") else block[0]
    block_hi = block[3] if side in ("L", "R") else block[2]
    block_c = (block_lo + block_hi) / 2.0
    others = [p for p in parts.values() if p.ref not in refs and p.ref != main and p.side == side]
    _pack_against_block(others, side, block_lo, block_hi, block_c, gap, along_center, along_half, set_along)
    _fit_diagonal(
        parts, main, refs | {p.ref for p in others}, refs, side,
        along_center, along_half, set_along, gap,
    )


def _pack_against_block(others, side, block_lo, block_hi, block_c, gap, along_center, along_half, set_along) -> None:
    """左右两侧都贴着晶振块排，不留 R162 那种空档。"""
    left = sorted([p for p in others if along_center(p) < block_c], key=along_center, reverse=True)
    right = sorted([p for p in others if along_center(p) >= block_c], key=along_center)
    edge = block_lo
    for part in left:
        half = along_half(part)
        set_along(part, base.snap_grid(edge - gap - half))
        edge = along_center(part) - along_half(part)
    edge = block_hi
    for part in right:
        half = along_half(part)
        set_along(part, base.snap_grid(edge + gap + half))
        edge = along_center(part) + along_half(part)


def _fit_diagonal(parts, main, side_refs, frozen, side, along_center, along_half, set_along, gap) -> None:
    """整排外框不许超出对角线允许长度。左边超了就整排往右边的空位挪。晶振块本身不拆。"""
    chip = base.aabb_of(parts[main])
    diag = ((chip[2] - chip[0]) ** 2 + (chip[3] - chip[1]) ** 2) ** 0.5
    limit = diag * (1.0 + base.DIAG_SLACK)
    axis = ((chip[1] + chip[3]) / 2.0) if side in ("L", "R") else ((chip[0] + chip[2]) / 2.0)
    window_lo = axis - limit / 2.0
    window_hi = axis + limit / 2.0

    def span():
        boxes = [base.aabb_of(p) for p in parts.values() if p.ref in side_refs]
        if side in ("L", "R"):
            return min(b[1] for b in boxes), max(b[3] for b in boxes)
        return min(b[0] for b in boxes), max(b[2] for b in boxes)

    lo, hi = span()
    delta = 0.0
    if lo < window_lo:
        delta = window_lo - lo
    if hi + delta > window_hi:
        delta = window_hi - hi
    delta = base.snap_grid(delta)
    if abs(delta) >= base.GRID / 2:
        for part in parts.values():
            if part.ref in side_refs:
                set_along(part, base.snap_grid(along_center(part) + delta))
    lo, hi = span()
    # 整排比对角线还长：把超出去那一头的器件挪到还有空的一头，贴着排
    for _ in range(12):
        lo, hi = span()
        if lo >= window_lo - 1.0 and hi <= window_hi + 1.0:
            break
        overflow_left = lo < window_lo - 1.0
        members = [p for p in parts.values() if p.ref in side_refs and p.ref not in frozen and p.ref != main]
        if overflow_left:
            victim = min(members, key=along_center)
            # 放到当前最右边的外侧
            rightmost = max(members, key=along_center)
            set_along(victim, base.snap_grid(along_center(rightmost) + along_half(rightmost) + gap + along_half(victim)))
        else:
            victim = max(members, key=along_center)
            leftmost = min(members, key=along_center)
            set_along(victim, base.snap_grid(along_center(leftmost) - along_half(leftmost) - gap - along_half(victim)))


def apply_style(parts: dict[str, base.Part], main: str, style: str) -> float:
    pins = base.chip_pins(parts[main])
    chip = base.aabb_of(parts[main])
    members = crystal_members(parts, main)
    total = 0.0
    for group in split_groups(members):
        nets = group["nets"]
        found = osc_pins(pins, nets)
        if len(found) < 2:
            continue
        side = side_of(found)
        found.sort(key=lambda p: p.along(side))
        left_net, right_net = found[0].net, found[-1].net
        mid = sum(p.along(side) for p in found) / len(found)
        edge = {"T": chip[3], "B": chip[1], "R": chip[2], "L": chip[0]}[side]
        layout_group(parts, group, side, edge, mid, left_net, right_net, style)
        total += local_length(parts, nets)
    refs = {p.ref for p in members}
    push_aside(parts, main, refs)
    return total


def choose_rot_free(part: base.Part, side: str, face: set[str]) -> float:
    """不再强制上下竖、左右横。信号脚更朝芯片的那个朝向优先，一样近就选更少占边的。"""
    pins = [p for p in part.pins if p.net in face] or list(part.pins)
    best_rot = 0
    best_key = None
    for rot in (0, 90, 180, 270):
        if pins:
            acc = 0.0
            for pin in pins:
                wx, wy = base.rot_xy(pin.lx, pin.ly, rot)
                if side == "R":
                    acc += wx
                elif side == "L":
                    acc += -wx
                elif side == "T":
                    acc += wy
                else:
                    acc += -wy
            outward = acc / len(pins)
        else:
            outward = 0.0
        back, front = along_extent(part, float(rot), side)
        key = (outward, back + front)
        if best_key is None or key < best_key:
            best_key = key
            best_rot = rot
    return float(best_rot)


def _draw(parts, path, title, subtitle, main, rules) -> None:
    base._setup_font()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12.5, 10.5), dpi=130)
    base._draw_on(ax, parts, main, title, subtitle, rules)
    base._legend(ax)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _zoom(parts, path, main, refs: set[str], title: str) -> None:
    if not refs:
        return
    base._setup_font()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(figsize=(8.2, 7.2), dpi=140)
    xs, ys = [], []
    for ref in refs:
        box = base.aabb_of(parts[ref])
        xs += [box[0], box[2]]
        ys += [box[1], box[3]]
    pad = 80.0 if base.UNIT_NAME == "mil" else 2.0
    # 把主芯片靠近这一组的那一段也画进来
    chip = base.aabb_of(parts[main])
    ax.set_aspect("equal")
    for net, pts in base.net_points(parts).items():
        if not is_crystal_net(net) or len(pts) < 2:
            continue
        xy = [(x, y) for _r, x, y in pts]
        for i, j, _d in base.mst_edges(xy):
            ax.plot([xy[i][0], xy[j][0]], [xy[i][1], xy[j][1]], color="#3DDC97", lw=1.2, zorder=1)
    for part in parts.values():
        box = base.aabb_of(part)
        if box[2] < min(xs) - pad or box[0] > max(xs) + pad or box[3] < min(ys) - pad or box[1] > max(ys) + pad:
            if part.ref != main:
                continue
        ax.add_patch(Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], zorder=2, **base._style(part)))
        if part.ref == main or part.ref in refs:
            ax.text(part.x, part.y, part.ref, ha="center", va="center", fontsize=8, color="#111", zorder=4,
                    bbox={"boxstyle": "round,pad=0.04", "fc": "white", "ec": "none", "alpha": 0.75})
    ax.set_xlim(min(min(xs), chip[0]) - pad, max(max(xs), chip[2]) + pad)
    # 只取晶振附近，不要整颗芯片
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(f"X（{base.UNIT_NAME}）")
    ax.set_ylabel(f"Y（{base.UNIT_NAME}）")
    ax.grid(True, ls=":", lw=0.4, color="#bbb")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def run_file(path: Path) -> None:
    hint = path.stem.split("_")[0]
    parts, named = base.load_design(path)
    main = named if named in parts else base.detect_main(parts, hint)
    original = base.snapshot(parts)
    out_dir = path.parent / "out_local"
    out_dir.mkdir(exist_ok=True)
    stem = path.stem
    box = base.aabb_of(parts[main])
    diag = ((box[2] - box[0]) ** 2 + (box[3] - box[1]) ** 2) ** 0.5
    rules = {"box": box, "limit": diag * (1.0 + base.DIAG_SLACK), "diag": diag}
    members = crystal_members(parts, main)
    refs = {p.ref for p in members}
    print(f"\n######## {path.name}  晶振相关 {sorted(refs) or '无'}")

    # 现有
    base.restore(parts, original)
    base.place_module(parts, main, "center")
    len_radial = local_length(parts, {n for p in members for n in p.nets() if is_crystal_net(n)})
    ov_radial = len(base.overlap_pairs(parts, main))
    snap_radial = base.snapshot(parts)
    print(f"现有竖横  振荡线 {len_radial:.1f}  重叠 {ov_radial}  全板半周长 {base.signal_hpwl(parts):.1f}")

    # 放开朝向
    saved = base.choose_rot
    base.choose_rot = choose_rot_free
    base.restore(parts, original)
    try:
        base.place_module(parts, main, "center")
    finally:
        base.choose_rot = saved
    len_free = local_length(parts, {n for p in members for n in p.nets() if is_crystal_net(n)})
    ov_free = len(base.overlap_pairs(parts, main))
    snap_free = base.snapshot(parts)
    # 放开朝向相对现有规则，实际改了几颗的旋转
    n_rot = 0
    for ref, rec_r in snap_radial.items():
        rec_f = snap_free[ref]
        if abs(rec_r[2] - rec_f[2]) > 0.1:
            n_rot += 1
    print(f"放开朝向  振荡线 {len_free:.1f}  重叠 {ov_free}  全板半周长 {base.signal_hpwl(parts):.1f}  旋转改变 {n_rot} 颗")

    # 几种局部摆法，按振荡线长短选，同时单独留下照片那种
    scores = {}
    snaps = {}
    for style in ("photo", "radial_stack", "crystal_near", "row"):
        base.restore(parts, snap_radial)
        length = apply_style(parts, main, style)
        scores[style] = length
        snaps[style] = base.snapshot(parts)
        ov = len(base.overlap_pairs(parts, main))
        print(f"局部 {style:14} 振荡线 {length:.1f}  重叠 {ov}  全板半周长 {base.signal_hpwl(parts):.1f}")

    best = min(scores, key=scores.get)
    import place_nocross as nc
    base.restore(parts, snap_radial)
    print("现有规则", nc._cross_text(parts))
    base.restore(parts, snaps["photo"])
    print("照片摆法", nc._cross_text(parts), "重叠", len(base.overlap_pairs(parts, main)))
    chip = base.aabb_of(parts[main])
    limit = (((chip[2] - chip[0]) ** 2 + (chip[3] - chip[1]) ** 2) ** 0.5) * (1.0 + base.DIAG_SLACK)
    axis = (chip[0] + chip[2]) / 2.0
    top = [p for p in parts.values() if p.side == "T"]
    if top:
        boxes = [base.aabb_of(p) for p in top]
        lo, hi = min(b[0] for b in boxes), max(b[2] for b in boxes)
        print(f"上边跨度 {hi - lo:.1f} / 允许 {limit:.1f}  左缘 {lo:.1f}（允许 {axis - limit/2:.1f}）  右缘 {hi:.1f}（允许 {axis + limit/2:.1f}）")
    print(f"局部线长最短的是 {best}")

    names = {
        "radial": "现有：上下竖、左右横",
        "free": "放开朝向，仍一颗颗排",
        "photo": "小模块按参考图摆成一块",
        "best": f"局部搜索最短：{best}",
    }
    panels = []
    for key, snap, subtitle in (
        ("radial", snap_radial, f"振荡线 {len_radial:.0f}"),
        ("free", snap_free, f"振荡线 {len_free:.0f}"),
        ("photo", snaps["photo"], f"振荡线 {scores['photo']:.0f}"),
    ):
        base.restore(parts, snap)
        _draw(parts, out_dir / f"{stem}_{key}.png", names[key], subtitle, main, rules)
        if key == "photo":
            _zoom(parts, out_dir / f"{stem}_photo_zoom.png", main, refs, "晶振小组放大（按参考图）")
        panels.append((base._clone_placed(parts, snap), names[key], subtitle))
    # 搜索结果若不是照片那种，再加一栏
    if best != "photo":
        base.restore(parts, snaps[best])
        _draw(parts, out_dir / f"{stem}_search.png", names["best"], f"振荡线 {scores[best]:.0f}", main, rules)
        _zoom(parts, out_dir / f"{stem}_search_zoom.png", main, refs, f"晶振小组放大（{best}）")
        panels.append((base._clone_placed(parts, snaps[best]), names["best"], f"振荡线 {scores[best]:.0f}"))
    base.draw_compare(panels, out_dir / f"{stem}_compare.png", main, rules)
    print("图:", out_dir / f"{stem}_compare.png")
    print("放大:", out_dir / f"{stem}_photo_zoom.png")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    paths = [Path(a) for a in sys.argv[1:]] or [
        HERE / "U34_Comps.json",
        HERE / "data" / "U1_Group_20261110_105225.json",
    ]
    for path in paths:
        run_file(path)


if __name__ == "__main__":
    main()
