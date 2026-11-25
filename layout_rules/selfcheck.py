# -*- coding: utf-8 -*-
"""
规则自检：docs/05 §2.4「每条规则上线前必须配正例 / 负例」的最小实现（Rule2DRC 式按执行结果判定）。

  1. 规则文件通过 schema 校验（kind/enforcement 搭配合法、阈值来源已声明……）
  2. 所有 check / metric 名都能在注册表里找到
  3. 每条规则 `tests:` 里引用的夹具按预期 pass / fail
  4. 所有 metric 在样例上能算出有限值

运行：python -m layout_rules.selfcheck
"""

from __future__ import annotations

import math
import sys

from shapely.geometry import box

from .checks import get_check, list_checks
from .compiler import compile_ruleset
from .loader import load_ruleset
from .metrics import get_metric, list_metrics
from .model import (
    Board, Candidate, Component, DfaSpacingTable, HeightRegion, KeepoutRegion,
    Pin, PlacementContext, Via, SIDE_BOTTOM, SIDE_TOP,
)


# --------------------------------------------------------------------------- #
# 夹具
# --------------------------------------------------------------------------- #

def base_context() -> PlacementContext:
    board = Board(
        outline=box(0, 0, 40, 40),
        keepouts=[KeepoutRegion(box(0, 0, 5, 5), name="K1")],
        height_regions=[HeightRegion(box(35, 0, 40, 40), max_height_mm=0.6, name="H1")],
        channels=[box(0, 30, 40, 32)],
        board_type="digital", speed_class=">=1Gbps", layer_count=6,
    )
    chip = Component("U1", 20, 20, 10, 10, side=SIDE_TOP, z_height_mm=1.5, dfa_class="BGA", pins=[
        Pin("A1", 16, 16, "VDD", "power"),
        Pin("A2", 17, 16, "GND", "gnd"),
        Pin("B1", 24, 24, "VDD_IO", "power"),
    ])
    vias = [
        Via(15.5, 15.5, "VDD"),
        Via(16.5, 15.5, "GND"),
        Via(30.0, 30.0, "SIG"),
    ]
    # R1 放在芯片（x 15~25）右侧 5mm 处，避免夹具与芯片本身相碰
    placed = [Component("R1", 30, 20, 2.0, 1.0, side=SIDE_TOP, z_height_mm=0.5, dfa_class="0402")]
    ctx = PlacementContext(board=board, chip=chip, placed=placed, vias=vias,
                           dfa=DfaSpacingTable(default_min_mm=0.2, default_preferred_mm=0.5))
    ctx.refresh()
    return ctx


def cap(value_uF: float = 0.1, z: float = 0.5, net: str = "VDD") -> Component:
    return Component("Cx", 0, 0, 1.0, 0.5, side=SIDE_TOP, z_height_mm=z, dfa_class="0402",
                     value_uF=value_uF, net_pwr=net, net_gnd="GND")


def fixtures() -> dict[str, Candidate]:
    c = cap()
    tall = cap(z=1.0)
    return {
        "overlap_pos": Candidate(c, 20, 10, 0, SIDE_TOP),
        "overlap_neg": Candidate(c, 30, 20, 0, SIDE_TOP),          # 压在 R1 上
        "board_pos": Candidate(c, 20, 10, 0, SIDE_TOP),
        "board_neg": Candidate(c, 39.9, 20, 0, SIDE_TOP),          # 越出板框
        "keepout_pos": Candidate(c, 20, 10, 0, SIDE_TOP),
        "keepout_neg": Candidate(c, 2.5, 2.5, 0, SIDE_TOP),        # 进禁布区
        "height_pos": Candidate(c, 37, 20, 0, SIDE_TOP),           # 0.5 ≤ 0.6
        "height_neg": Candidate(tall, 37, 20, 0, SIDE_TOP),        # 1.0 > 0.6
        "via_clear_pos": Candidate(c, 20, 10, 0, SIDE_TOP),
        "via_clear_neg": Candidate(c, 30, 30, 0, SIDE_TOP),        # 压他网过孔 SIG
        "dfa_min_pos": Candidate(c, 30, 22, 0, SIDE_TOP),          # 与 R1 间距 1.25mm
        "dfa_min_neg": Candidate(c, 30, 20.85, 0, SIDE_TOP),       # 与 R1 间距 0.1mm < 0.2mm
        "dist_pos": Candidate(c, 15, 17, 0, SIDE_TOP),             # 到 A1 曼哈顿 2mm
        "dist_neg": Candidate(c, 5, 30, 0, SIDE_TOP),              # 25mm
    }


# --------------------------------------------------------------------------- #
# 自检
# --------------------------------------------------------------------------- #

def main() -> int:
    failures: list[str] = []
    ruleset = load_ruleset(strict=True)
    print(f"规则文件校验通过：{ruleset.name} v{ruleset.version}，共 {len(ruleset.rules)} 条")

    # 2. 注册表解析
    compiled = compile_ruleset(ruleset, None)
    print(compiled.describe())
    print(f"已注册检查器: {list_checks()}")
    print(f"已注册指标  : {list_metrics()}")

    # 3. 规则自带测试
    ctx = base_context()
    fx = fixtures()
    n_tests = 0
    for r in ruleset.rules:
        if not r.tests or not r.check:
            continue
        fn = get_check(r.check, r.scope)
        for t in r.tests:
            n_tests += 1
            name, expect = t["fixture"], t["expect"]
            if name not in fx:
                failures.append(f"{r.id}: 夹具 {name} 不存在")
                continue
            res = fn(fx[name], ctx, **r.params)
            got = "pass" if res.ok else "fail"
            mark = "OK " if got == expect else "BAD"
            if got != expect:
                failures.append(f"{r.id}/{name}: 期望 {expect}，实际 {got}（{res.detail}）")
            print(f"  [{mark}] {r.id:<16} {name:<14} → {got:<4} {res.detail}")

    # 4. 指标可算
    sample = fx["dist_pos"]
    for name in list_metrics()["placement"]:
        v = get_metric(name, "placement")(sample, ctx)
        if not math.isfinite(v):
            failures.append(f"指标 {name} 返回非有限值 {v}")
        print(f"  metric {name:<26} = {v:.4f}")
    layout_sample = [fx["dist_pos"].placed, fx["dfa_min_pos"].placed]
    for name in list_metrics()["layout"]:
        v = get_metric(name, "layout")(layout_sample, ctx)
        if not math.isfinite(v):
            failures.append(f"layout 指标 {name} 返回非有限值 {v}")
        print(f"  layout metric {name:<19} = {v:.4f}")

    print("-" * 70)
    if failures:
        print(f"自检失败 {len(failures)} 项：")
        for f in failures:
            print("  - " + f)
        return 1
    print(f"自检通过：{n_tests} 个规则测试用例、{len(list_metrics()['placement']) + len(list_metrics()['layout'])} 个指标")
    return 0


if __name__ == "__main__":
    sys.exit(main())
