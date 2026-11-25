# -*- coding: utf-8 -*-
"""
端到端演示：合成一个 BGA 场景，用规则集驱动「候选生成 → 掩码 → 惩罚打分 → 贪心摆放 → 终局评估」。

这就是 docs/08 §6 要求的 B2「规则驱动启发式基线」的骨架：
  - 候选生成器 + 掩码     = RL 的动作空间（docs/08 §4.2）
  - 每步惩罚 / 终局惩罚    = RL 的奖励函数负项（docs/08 §4.4）
  - evaluate 报告          = 规则引擎事后检查 / 评测（docs/08 §7）
换成 RL 时只需把「选惩罚最小的候选」替换成「策略网络在合法候选上采样」。

运行：
    python -m layout_rules.demo_decap            # 打印过程与报告
    python -m layout_rules.demo_decap --json out.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict

from shapely.geometry import Point, box

from .compiler import compile_ruleset
from .loader import load_ruleset
from .model import (
    Board, Candidate, Component, DfaSpacingTable, HeightRegion, KeepoutRegion,
    Pin, PlacementContext, Via, SIDE_BOTTOM, SIDE_TOP, opposite_side,
)


# --------------------------------------------------------------------------- #
# 1. 合成场景
# --------------------------------------------------------------------------- #

def build_scene() -> tuple[PlacementContext, list[Component]]:
    board = Board(
        outline=box(0, 0, 60, 60),
        keepouts=[KeepoutRegion(box(0, 0, 10, 10), name="安装孔区")],
        height_regions=[HeightRegion(box(48, 0, 60, 60), max_height_mm=0.7, sides=(SIDE_TOP,), name="结构限高")],
        channels=[box(20, 44, 40, 47)],
        layer_count=6,
        board_type="digital",
        speed_class=">=1Gbps",
        thickness_mm=1.6,
    )

    # 主芯片：15×15 BGA，1.0mm pitch，TOP 面
    cx, cy, n, pitch = 30.0, 30.0, 15, 1.0
    pins: list[Pin] = []
    vias: list[Via] = []
    for i in range(n):
        for j in range(n):
            x = cx + (i - (n - 1) / 2) * pitch
            y = cy + (j - (n - 1) / 2) * pitch
            r = max(abs(i - 7), abs(j - 7))
            if r <= 2:
                net, kind = ("VDD_CORE", "power") if (i + j) % 2 == 0 else ("GND", "gnd")
            elif r <= 4:
                net, kind = ("VDD_IO", "power") if (i + j) % 2 == 0 else ("GND", "gnd")
            else:
                net, kind = (f"SIG_{i}_{j}", "signal")
            pins.append(Pin(name=f"{chr(65 + i)}{j + 1}", x=x, y=y, net=net, kind=kind))
            if kind in ("power", "gnd"):
                # dog-bone 扇出过孔：斜向 0.5mm
                vias.append(Via(x=x + 0.5, y=y + 0.5, net=net, diameter_mm=0.45, span_mm=1.6))
    chip = Component("U1", cx, cy, 15.0, 15.0, side=SIDE_TOP, z_height_mm=1.2,
                     dfa_class="BGA", pins=pins, fixed=True)

    placed = [
        Component("J1", 30.0, 55.0, 40.0, 6.0, side=SIDE_TOP, z_height_mm=8.0, dfa_class="CONN", fixed=True),
        Component("U2", 12.0, 45.0, 8.0, 8.0, side=SIDE_BOTTOM, z_height_mm=1.0, dfa_class="QFN", fixed=True),
    ]

    dfa = DfaSpacingTable(
        min_mm={("0402", "0402"): 0.2, ("0402", "BGA"): 0.5, ("0603", "BGA"): 0.5, ("0805", "BGA"): 0.8,
                ("0402", "CONN"): 1.0, ("0603", "CONN"): 1.0, ("0805", "CONN"): 1.0},
        preferred_mm={("0402", "0402"): 0.4, ("0402", "BGA"): 1.0, ("0603", "BGA"): 1.0, ("0805", "BGA"): 1.5},
        default_min_mm=0.25, default_preferred_mm=0.6,
    )

    ctx = PlacementContext(board=board, chip=chip, placed=placed, vias=vias, dfa=dfa)
    ctx.refresh()

    def cap(ref, value, pkg, net):
        w, h, z = {"0402": (1.0, 0.5, 0.5), "0603": (1.6, 0.8, 0.8), "0805": (2.0, 1.25, 1.2)}[pkg]
        return Component(ref, 0, 0, w, h, side=SIDE_TOP, z_height_mm=z, dfa_class=pkg,
                         value_uF=value, net_pwr=net, net_gnd="GND")

    caps = (
        [cap(f"C{i}", 0.1, "0402", "VDD_CORE") for i in range(1, 7)]
        + [cap(f"C{i}", 0.1, "0402", "VDD_IO") for i in range(7, 11)]
        + [cap(f"C{i}", 1.0, "0603", "VDD_CORE") for i in range(11, 13)]
        + [cap(f"C{i}", 10.0, "0805", "VDD_IO") for i in range(13, 15)]
    )
    return ctx, caps


# --------------------------------------------------------------------------- #
# 2. 候选生成器（几何 + 规则，非学习）
# --------------------------------------------------------------------------- #

def generate_candidates(comp: Component, ctx: PlacementContext, pitch: float = 1.0,
                        ring_mm: float = 8.0) -> list[Candidate]:
    """
    docs/08 §4.2：
      1) 芯片背面对应区域按 fanout 间隙生成阵列候选；
      2) 芯片四周环形区域按网格生成候选；
      3) × 2 旋转 × 2 面。
    """
    chip = ctx.chip
    half = chip.width_mm / 2.0
    out: list[Candidate] = []

    # 背面阵列：落在扇出过孔之间的空隙（过孔在 pin+0.5 处，因此候选放在 pin 位置附近的整数格）
    back = opposite_side(chip.side)
    xs = [chip.x - half + 0.5 + k * pitch for k in range(int(chip.width_mm / pitch))]
    ys = [chip.y - half + 0.5 + k * pitch for k in range(int(chip.height_mm / pitch))]
    for x in xs:
        for y in ys:
            for rot in (0.0, 90.0):
                out.append(Candidate(comp, x, y, rot, back))

    # 四周环带（两面都生成）
    lo, hi = half + 1.0, half + ring_mm
    steps = int((2 * hi) / pitch) + 1
    for i in range(steps):
        for j in range(steps):
            dx = -hi + i * pitch
            dy = -hi + j * pitch
            if max(abs(dx), abs(dy)) < lo:
                continue
            for side in (SIDE_TOP, SIDE_BOTTOM):
                for rot in (0.0, 90.0):
                    out.append(Candidate(comp, chip.x + dx, chip.y + dy, rot, side))
    return out


# --------------------------------------------------------------------------- #
# 3. 贪心基线（B2）
# --------------------------------------------------------------------------- #

def place_greedy(ctx: PlacementContext, caps: list[Component], compiled, verbose: bool = True) -> list[dict]:
    log: list[dict] = []
    # docs/08 §4.3 决策顺序：容值从小到大
    order = sorted(caps, key=lambda c: (c.value_uF or 0.0, c.refdes))
    t_start = time.perf_counter()
    for cap in order:
        t0 = time.perf_counter()
        cands = generate_candidates(cap, ctx)
        m = compiled.mask(cands, ctx)
        legal = m.legal_indices
        entry = {
            "refdes": cap.refdes, "value_uF": cap.value_uF, "net": cap.net_pwr,
            "candidates": len(cands), "after_must": m.n_after_must, "after_should": m.n_after_should,
            "degraded": m.degraded,
        }
        if not legal:
            entry["result"] = "无合法候选，需人工处理"
            log.append(entry)
            if verbose:
                print(f"[{cap.refdes}] 无合法候选！must 后剩 {m.n_after_must}")
            continue
        best_i, best_p = None, None
        for i in legal:
            p = compiled.step_penalty(cands[i], ctx, m.degraded)
            if best_p is None or p.total < best_p.total:
                best_i, best_p = i, p
        chosen = cands[best_i]
        placed = chosen.placed
        ctx.add_placed(placed)
        via_notes = _add_vias_if_needed(placed, ctx)
        dt = time.perf_counter() - t0
        if via_notes:
            entry["via_notes"] = via_notes
        entry.update({
            "x": round(chosen.x, 3), "y": round(chosen.y, 3), "rot": chosen.rot_deg, "side": chosen.side,
            "penalty": round(best_p.total, 4),
            "penalty_terms": {k: round(v, 4) for k, v in sorted(best_p.terms.items(), key=lambda kv: -kv[1])},
            "elapsed_ms": round(dt * 1000, 1),
        })
        log.append(entry)
        if verbose:
            top = ", ".join(f"{k}={v:.2f}" for k, v in list(entry["penalty_terms"].items())[:3])
            deg = f" 降级:{m.degraded}" if m.degraded else ""
            print(f"[{cap.refdes} {cap.value_uF}uF {cap.net_pwr}] 候选 {len(cands)} → must {m.n_after_must}"
                  f" → should {m.n_after_should}{deg} | 选 ({chosen.x:.1f},{chosen.y:.1f},{chosen.rot_deg:.0f}°,{chosen.side})"
                  f" 惩罚 {best_p.total:.2f} [{top}] {dt*1000:.0f}ms")
    if verbose:
        print(f"共 {len(order)} 颗，耗时 {time.perf_counter() - t_start:.2f}s")
    return log


def _add_vias_if_needed(cap: Component, ctx: PlacementContext, borrow_mm: float = 1.0,
                        via_d: float = 0.45, comp_clear: float = 0.15, via_clear: float = 0.5) -> list[str]:
    """
    若 borrow_mm 内没有可借用的同网过孔，则为电容新打过孔（电源 / 地各一）。

    过孔位置本身也是“动作”的一部分（docs/08 §1.4 via-in-pad / 共享过孔），这里用最简策略：
    依次尝试 长轴两端 → 短轴两端 的位置，要求
      - 与所有已摆放器件外形距离 ≥ comp_clear（过孔穿透两面，不分面；主芯片投影下允许）
      - 与已有过孔中心距离 ≥ via_clear
    找不到合法位置则放弃并返回说明（真实系统里应作为惩罚项或触发人工处理）。
    """
    notes: list[str] = []
    long_axis = math.radians(cap.rot_deg if cap.width_mm >= cap.height_mm else cap.rot_deg + 90.0)
    long_off = max(cap.width_mm, cap.height_mm) / 2.0 + 0.4
    short_off = min(cap.width_mm, cap.height_mm) / 2.0 + 0.4
    attempts = [
        (long_axis, long_off, 1.0), (long_axis, long_off, -1.0),
        (long_axis + math.pi / 2, short_off, 1.0), (long_axis + math.pi / 2, short_off, -1.0),
    ]
    used: list[tuple[float, float]] = []
    for net in (cap.net_pwr, cap.net_gnd):
        if not near_via(cap, ctx, net, borrow_mm):
            placed = False
            for ang, off, sign in attempts:
                x = cap.x + sign * off * math.cos(ang)
                y = cap.y + sign * off * math.sin(ang)
                if any(math.hypot(x - ux, y - uy) < via_clear for ux, uy in used):
                    continue
                circle = Point(x, y).buffer(via_d / 2.0, quad_segs=4)
                if not ctx.is_clear_of_components(circle, comp_clear, exclude_refdes=cap.refdes):
                    continue
                if any(math.hypot(v.x - x, v.y - y) < via_clear for v in ctx.nearby_vias(circle, via_clear)):
                    continue
                ctx.add_via(Via(x, y, net=net, diameter_mm=via_d, span_mm=ctx.board.thickness_mm))
                used.append((x, y))
                placed = True
                break
            if not placed:
                notes.append(f"{cap.refdes}: 无法为 {net} 新增过孔")
    return notes


def near_via(cap: Component, ctx: PlacementContext, net: str | None, radius_mm: float) -> bool:
    if not net:
        return True
    _, d = ctx.nearest_via(net, cap.x, cap.y)
    return d <= radius_mm


# --------------------------------------------------------------------------- #
# 4. 主流程
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="去耦电容规则驱动摆放演示（B2 基线骨架）")
    ap.add_argument("--rules", default=None, help="规则 YAML 路径，默认 rules/decap_v0.yaml")
    ap.add_argument("--json", default=None, help="把过程与报告写入 JSON 文件")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    ruleset = load_ruleset(args.rules)
    ctx, caps = build_scene()
    compiled = compile_ruleset(ruleset, ctx.board)
    if not args.quiet:
        print(compiled.describe())
        print("-" * 78)

    log = place_greedy(ctx, caps, compiled, verbose=not args.quiet)
    placed_caps = [c for c in ctx.placed if c.value_uF is not None]

    term = compiled.terminal_penalty(placed_caps, ctx)
    report = compiled.evaluate(placed_caps, ctx)

    if not args.quiet:
        print("-" * 78)
        print(f"终局惩罚 {term.total:.3f}: " + ", ".join(f"{k}={v:.3f}" for k, v in term.terms.items()))
        print("-" * 78)
        print("事后全量检查：")
        print(report.summary())
        print("-" * 78)
        print(f"硬约束违反 = {report.hard_violations}（docs/08 §7.2 要求 = 0）")

    if args.json:
        payload = {
            "ruleset": {"name": ruleset.name, "version": ruleset.version, "active_rules": [r.id for r in compiled.rules]},
            "placements": log,
            "terminal_penalty": {"total": term.total, "terms": term.terms},
            "report": {
                "hard_violations": report.hard_violations,
                "soft_violations": report.soft_violations,
                "violations": [asdict(v) for v in report.violations],
                "metrics": report.metrics,
                "skipped_rules": report.skipped_rules,
            },
        }
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        if not args.quiet:
            print(f"已写入 {args.json}")
    return 0 if report.hard_violations == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
