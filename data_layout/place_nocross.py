# -*- coding: utf-8 -*-
"""
在 place_u34.py 之外多一版约束，不改原来的脚本，也不覆盖 data/out。

1. 信号飞线不交叉。电源网和地网不计入：它们走平面，线段必然到处交叉。
   「飞线交叉」指两根信号网的直线段相交，而且不共用端点。
2. 晶振整组（晶振 + 共用振荡网的电容/电阻）优先放到高速引脚最少的一侧，
   并靠到这一侧里离高速引脚更远的那一头。

高速引脚：网络名里带 PCIE / USB / SATA / HDMI / MIPI / LVDS / DDR / DQS /
NVME / SERDES / EDP，或带 _DATA / _DQS / _DM 的存储器数据。
晶振自己的 OSC、XTAL、X32K、X48M 不算要躲开的高速脚。

运行：
    .\\.venv\\Scripts\\python.exe data_layout\\place_nocross.py data_layout\\data\\U1_Group_20261110_105225.json
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import place_u34 as base

HERE = Path(__file__).resolve().parent
OUT_NAME = "out_nocross"

# 这些词出现在网络名里，就当成高速。晶振网另外排除。
HS_WORDS = (
    "PCIE", "USB", "SATA", "HDMI", "MIPI", "LVDS", "DDR", "DQS",
    "NVME", "SERDES", "SGMII", "XAUI", "EDP", "SLVS", "MGT",
)
CRYSTAL_WORDS = ("OSC", "XTAL", "X32K", "X48M", "32KHZ", "48MHZ")

# 本轮晶振组成员，供消交叉时整组不要拆开
CRYSTAL_REFS: set[str] = set()
MOVE_NOTES: list[str] = []


def _is_crystal_net(name: str) -> bool:
    u = (name or "").upper()
    return any(k in u for k in CRYSTAL_WORDS)


def is_hs_net(name: str) -> bool:
    """高速信号。电源、地、晶振自己的脚都不算。"""
    u = (name or "").upper()
    if not u or base.is_power_net(u) or _is_crystal_net(u):
        return False
    if any(k in u for k in HS_WORDS):
        return True
    if "_DATA" in u or "DQS" in u or "_DM" in u or u.endswith("_DM"):
        return True
    return False


def _is_crystal_part(part: base.Part) -> bool:
    if part.kind == "xtal":
        return True
    ref = part.ref
    return len(ref) > 1 and ref[0] in "XY" and ref[1].isdigit()


def hs_counts(pins: list[base.ChipPin]) -> dict[str, int]:
    counts = {s: 0 for s in "LRTB"}
    for pin in pins:
        if is_hs_net(pin.net):
            counts[pin.side] += 1
    return counts


def _crystal_groups(parts: dict[str, base.Part], main: str) -> list[list[base.Part]]:
    crystals = [p for p in parts.values() if p.ref != main and _is_crystal_part(p)]
    groups = []
    used: set[str] = set()
    for crystal in crystals:
        nets = {n for n in crystal.nets() if _is_crystal_net(n)}
        members = [crystal]
        used.add(crystal.ref)
        for part in parts.values():
            if part.ref == main or part.ref in used:
                continue
            if nets and (part.nets() & nets):
                members.append(part)
                used.add(part.ref)
        # 晶振放中间，电容靠它，电阻放外侧
        rank = {"cap": 0, "xtal": 1, "res": 2, "fb": 3}
        members.sort(key=lambda p: (0 if _is_crystal_part(p) else 1, rank.get(p.kind, 4), p.ref))
        # 把晶振插到组的中间
        xtal = [p for p in members if _is_crystal_part(p)]
        others = [p for p in members if not _is_crystal_part(p)]
        mid = len(others) // 2
        ordered = others[:mid] + xtal + others[mid:]
        groups.append(ordered)
    return groups


def _find_item(by_side: dict[str, list[base.Item]], ref: str) -> base.Item | None:
    for items in by_side.values():
        for it in items:
            if it.part.ref == ref:
                return it
    return None


def relocate_crystals(
    by_side: dict[str, list[base.Item]],
    parts: dict[str, base.Part],
    pins: list[base.ChipPin],
    main: str,
) -> None:
    """把晶振整组挪到高速引脚最少的一侧，并靠到离这些引脚更远的一头。"""
    global CRYSTAL_REFS, MOVE_NOTES
    CRYSTAL_REFS = set()
    MOVE_NOTES = []
    groups = _crystal_groups(parts, main)
    if not groups:
        MOVE_NOTES.append("这块板没有晶振")
        return
    counts = hs_counts(pins)
    natural_votes: dict[str, int] = defaultdict(int)
    for group in groups:
        for part in group:
            old = _find_item(by_side, part.ref)
            if old is not None and _is_crystal_part(part):
                natural_votes[old.side] += 1
    natural = max(natural_votes, key=lambda s: (natural_votes[s], s)) if natural_votes else "T"
    # 高速脚一样少时，留在晶振引脚原来的那一侧，避免无故拉长振荡线
    quiet = min("LRTB", key=lambda s: (counts[s], 0 if s == natural else 1, s))
    alongs = {
        s: [p.along(s) for p in pins if p.side == s]
        for s in "LRTB"
    }
    hs_along = {
        s: [p.along(s) for p in pins if p.side == s and is_hs_net(p.net)]
        for s in "LRTB"
    }
    names = {"L": "左", "R": "右", "T": "上", "B": "下"}
    for group in groups:
        for part in group:
            CRYSTAL_REFS.add(part.ref)
    MOVE_NOTES.append(
        "高速引脚 "
        + " ".join(f"{names[s]}{counts[s]}" for s in "LRTB")
        + f"，高速脚最少的是{names[quiet]}侧"
    )
    if quiet == natural:
        MOVE_NOTES.append(f"晶振本来就在{names[quiet]}侧，振荡脚也在这边，位置保持不动")
        return

    def pin_along(part: base.Part) -> float:
        nets = {n for n in part.nets() if _is_crystal_net(n)} or set(part.nets())
        if quiet in ("L", "R"):
            vals = [p.y for p in pins if p.net in nets]
        else:
            vals = [p.x for p in pins if p.net in nets]
        return sum(vals) / len(vals) if vals else 0.0

    ordered: list[base.Part] = []
    notes = []
    for group in sorted(groups, key=lambda g: pin_along(g[len(g) // 2])):
        old_sides = []
        chunk = sorted(group, key=pin_along)
        for part in chunk:
            old = _find_item(by_side, part.ref)
            old_sides.append(old.side if old else "?")
            if old is not None:
                by_side[old.side] = [it for it in by_side[old.side] if it.part.ref != part.ref]
        ordered.extend(chunk)
        notes.append((",".join(p.ref for p in chunk), "".join(old_sides)))

    span = alongs[quiet]
    lo = min(span) if span else 0.0
    hi = max(span) if span else 0.0
    hs = hs_along[quiet]
    if hs:
        mean = sum(hs) / len(hs)
        end = "lo" if abs(lo - mean) >= abs(hi - mean) else "hi"
    else:
        end = "lo"
    axis = (lo + hi) / 2.0
    n = len(ordered)
    for i, part in enumerate(ordered):
        # 锚点落在引脚范围外面，整组连在一起，不会插进别的器件中间
        anchor = (hi + 10.0 + i) if end == "hi" else (lo - 10.0 - (n - 1 - i))
        face = {net for net in part.nets() if _is_crystal_net(net)} or set(part.nets())
        item = base.make_item(part, quiet, anchor, axis, 0, face, ("", next(iter(face), ""), quiet))
        by_side[quiet].append(item)
    where = "高端" if end == "hi" else "低端"
    for refs, old_sides in notes:
        MOVE_NOTES.append(f"  {refs}：原边 {old_sides} -> {names[quiet]}侧{where}")


def signal_segments(parts: dict[str, base.Part]) -> list[tuple[str, str, tuple[float, float], tuple[float, float]]]:
    """每条信号网一棵最小生成树，电源和地不画进交叉判断。"""
    segs = []
    for net, pts in base.net_points(parts).items():
        if base.is_power_net(net) or len(pts) < 2:
            continue
        xy = [(x, y) for _ref, x, y in pts]
        for i, j, _d in base.mst_edges(xy):
            segs.append((pts[i][0], pts[j][0], xy[i], xy[j]))
    return segs


def _proper_cross(a, b, c, d) -> bool:
    # 包围盒先挡掉大部分不相交的线段
    if max(a[0], b[0]) < min(c[0], d[0]) or max(c[0], d[0]) < min(a[0], b[0]):
        return False
    if max(a[1], b[1]) < min(c[1], d[1]) or max(c[1], d[1]) < min(a[1], b[1]):
        return False

    def orient(p, q, r) -> float:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    o1, o2 = orient(a, b, c), orient(a, b, d)
    o3, o4 = orient(c, d, a), orient(c, d, b)
    return o1 * o2 < 0 and o3 * o4 < 0


def count_crossings(parts: dict[str, base.Part], crystal_only: bool | None = None) -> int:
    """crystal_only=True 只数晶振网，False 只数其余信号，None 全部信号。"""
    segs = []
    for net, pts in base.net_points(parts).items():
        if base.is_power_net(net) or len(pts) < 2:
            continue
        crystal = _is_crystal_net(net)
        if crystal_only is True and not crystal:
            continue
        if crystal_only is False and crystal:
            continue
        xy = [(x, y) for _ref, x, y in pts]
        for i, j, _d in base.mst_edges(xy):
            segs.append((pts[i][0], pts[j][0], xy[i], xy[j]))
    n = 0
    for i in range(len(segs)):
        ra, rb, a, b = segs[i]
        for j in range(i + 1, len(segs)):
            rc, rd, c, d = segs[j]
            if len({ra, rb, rc, rd}) < 4:
                continue
            if _proper_cross(a, b, c, d):
                n += 1
    return n


def _along(part: base.Part) -> float:
    return part.y if part.side in ("L", "R") else part.x


def _set_along(part: base.Part, value: float) -> None:
    if part.side in ("L", "R"):
        part.y = value
    else:
        part.x = value


def uncross(parts: dict[str, base.Part], main: str) -> int:
    """同一圈里交换相邻器件的位置，交叉减少才留下。晶振组不拆。"""
    swaps = 0
    current = count_crossings(parts)
    for _pass in range(2):
        changed = False
        for side in "LRTB":
            rings: dict[int, list[base.Part]] = defaultdict(list)
            for part in parts.values():
                if part.ref != main and part.side == side and part.ring >= 0:
                    rings[part.ring].append(part)
            for members in rings.values():
                members.sort(key=_along)
                i = 0
                while i < len(members) - 1:
                    a, b = members[i], members[i + 1]
                    a_in, b_in = a.ref in CRYSTAL_REFS, b.ref in CRYSTAL_REFS
                    if a_in or b_in:
                        i += 1
                        continue
                    oa, ob = _along(a), _along(b)
                    _set_along(a, ob)
                    _set_along(b, oa)
                    if base.overlap_pairs(parts, main) or count_crossings(parts) >= current:
                        _set_along(a, oa)
                        _set_along(b, ob)
                        i += 1
                        continue
                    current = count_crossings(parts)
                    members[i], members[i + 1] = b, a
                    swaps += 1
                    changed = True
                    if swaps >= 30:
                        return swaps
                if swaps >= 30:
                    return swaps
        if not changed:
            break
    return swaps


def _cross_text(parts: dict[str, base.Part]) -> str:
    """全部信号交叉，再拆开：普通信号之间、晶振线之间、晶振线和别的信号线。"""
    tagged: list[tuple[bool, str, str, tuple[float, float], tuple[float, float]]] = []
    for net, pts in base.net_points(parts).items():
        if base.is_power_net(net) or len(pts) < 2:
            continue
        crystal = _is_crystal_net(net)
        xy = [(x, y) for _ref, x, y in pts]
        for i, j, _d in base.mst_edges(xy):
            tagged.append((crystal, pts[i][0], pts[j][0], xy[i], xy[j]))
    total = xtal_xtal = other_other = mixed = 0
    for i in range(len(tagged)):
        fa, ra, rb, a, b = tagged[i]
        for j in range(i + 1, len(tagged)):
            fb, rc, rd, c, d = tagged[j]
            if len({ra, rb, rc, rd}) < 4:
                continue
            if not _proper_cross(a, b, c, d):
                continue
            total += 1
            if fa and fb:
                xtal_xtal += 1
            elif fa or fb:
                mixed += 1
            else:
                other_other += 1
    return (
        f"信号交叉 {total}（信号之间 {other_other}，"
        f"晶振线之间 {xtal_xtal}，晶振线和信号线 {mixed}）"
    )


def _draw(parts, path, title, subtitle, main, rules, marks: list[str]) -> None:
    base._setup_font()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(13.2, 11.2), dpi=120)
    base._draw_on(ax, parts, main, title, subtitle, rules)
    for ref in marks:
        part = parts.get(ref)
        if part is None:
            continue
        ax.scatter([part.x], [part.y], s=80, facecolors="none", edgecolors="#C0392B", linewidths=1.4, zorder=6)
        ax.text(part.x, part.y, ref, color="#C0392B", fontsize=8, ha="center", va="bottom", zorder=7)
    base._legend(ax)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def run_file(path: Path) -> None:
    hint = path.stem.split("_")[0]
    parts, named = base.load_design(path)
    main = named if named in parts else base.detect_main(parts, hint)
    out_dir = path.parent / OUT_NAME
    out_dir.mkdir(exist_ok=True)
    stem = path.stem
    original = base.snapshot(parts)
    box = base.aabb_of(parts[main])
    diag = (box[2] - box[0]) ** 2 + (box[3] - box[1]) ** 2
    diag **= 0.5
    rules = {"box": box, "limit": diag * (1.0 + base.DIAG_SLACK), "diag": diag}

    print(f"\n######## {path.name}  {main}  {len(parts)} 颗  {base.UNIT_NAME}  格点 {base.GRID:g}")
    before_cross = _cross_text(parts)
    print("摆放前", before_cross)
    _draw(
        parts, out_dir / f"{stem}_before.png",
        f"摆放前：{main}",
        before_cross,
        main, rules, [],
    )

    # 现有居中，当作对照，不写回原来的 out 目录
    saved_build, saved_slide = base.build_items, base._slide
    base.restore(parts, original)
    info_v1 = base.place_module(parts, main, "center")
    v1_cross = _cross_text(parts)
    snap_v1 = base.snapshot(parts)
    print("现有居中", v1_cross, f"半周长 {base.signal_hpwl(parts):.1f}")

    def build_items_v2(parts_map, main_ref, pins):
        by_side = saved_build(parts_map, main_ref, pins)
        relocate_crystals(by_side, parts_map, pins, main_ref)
        return by_side

    base.restore(parts, original)
    base.build_items = build_items_v2
    try:
        info_v2 = base.place_module(parts, main, "center")
    finally:
        base.build_items = saved_build
    print("晶振：")
    for line in MOVE_NOTES:
        print(line)
    pre_swap = _cross_text(parts)
    swaps = uncross(parts, main)
    v2_cross = _cross_text(parts)
    metrics = base.measure(parts, info_v2)
    print(f"挪晶振后、消交叉前 {pre_swap}")
    print(f"相邻交换 {swaps} 次后 {v2_cross}")
    print(
        f"半周长 {metrics['hpwl']:.1f}  重叠 {metrics['overlap']}  "
        f"净空 {metrics['clearance']:.2f}  偏离 {metrics['sym']:.2f}  外圈 {metrics['extra_rings']}"
    )
    marks = sorted(CRYSTAL_REFS)
    subtitle = f"{v2_cross}    半周长 {base.fmt_num(metrics['hpwl'])} {base.UNIT_NAME}    重叠 {metrics['overlap']}"
    _draw(
        parts, out_dir / f"{stem}_after.png",
        f"不交叉 + 晶振远离高速：{main}",
        subtitle,
        main, rules, [ref for ref in marks if _is_crystal_part(parts[ref])],
    )
    panels = [
        (base._clone_placed(parts, snap_v1), "现有居中", v1_cross),
        (base._clone_placed(parts, base.snapshot(parts)), "本版", v2_cross),
    ]
    # _clone 会按快照恢复，第二份必须在 clone 之前拍好。上面 snapshot 是当前 v2，OK。
    base.draw_compare(panels, out_dir / f"{stem}_compare.png", main, rules)
    print("图:", out_dir / f"{stem}_before.png")
    print("图:", out_dir / f"{stem}_after.png")
    print("图:", out_dir / f"{stem}_compare.png")
    _ = info_v1


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    paths = [Path(a) for a in sys.argv[1:]] or [
        HERE / "data" / "U44_Group_20261110_105314.json",
        HERE / "data" / "U1_Group_20261110_105225.json",
        HERE / "U34_Comps.json",
    ]
    for path in paths:
        run_file(path)


if __name__ == "__main__":
    main()
