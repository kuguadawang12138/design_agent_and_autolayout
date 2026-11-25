# -*- coding: utf-8 -*-
"""
造几组模块假数据，格式和 U34_Comps.json 相同，用来检查摆放是不是只认这一块板。

    .\\.venv\\Scripts\\python.exe data_layout\\make_cases.py

坐标单位 mil。外围器件的初始位置故意散开、旋转打乱，摆放前的图应该是乱的。
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "cases"


def rot_xy(x: float, y: float, deg: float) -> tuple[float, float]:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return x * c - y * s, x * s + y * c


def component(ref: str, ox: float, oy: float, rot: float, lw: float, lh: float,
              pins: list[tuple[str, str, float, float]], layer: str = "TOP") -> dict:
    """pins 是 0° 时相对原点的坐标。外框按旋转后的轴对齐盒写出。"""
    corners = [rot_xy(sx * lw / 2, sy * lh / 2, rot) for sx in (-1, 1) for sy in (-1, 1)]
    xs = [ox + x for x, _y in corners]
    ys = [oy + y for _x, y in corners]
    world_pins = []
    for num, net, lx, ly in pins:
        wx, wy = rot_xy(lx, ly, rot)
        world_pins.append({
            "pinNumber": str(num),
            "region": "",
            "order": "0",
            "pinNetName": net,
            "pin-xy": {"x": round(ox + wx, 3), "y": round(oy + wy, 3)},
            "bBox": {"x1": 0, "y1": 0, "x2": 0, "y2": 0},
        })
    return {
        "refName": ref,
        "region": "",
        "order": "0",
        "rotation": float(rot),
        "Layer": layer,
        "origin": {"x": round(ox, 3), "y": round(oy, 3)},
        "bBox": {
            "x1": round(min(xs), 3), "y1": round(min(ys), 3),
            "x2": round(max(xs), 3), "y2": round(max(ys), 3),
        },
        "pins": world_pins,
    }


def axial(net_a: str, net_b: str, length: float) -> list[tuple[str, str, float, float]]:
    d = length * 0.36
    return [("1", net_a, d, 0.0), ("2", net_b, -d, 0.0)]


def qfp(ref: str, cx: float, cy: float, body: float, n_side: int, pitch: float,
        net_at) -> dict:
    """四边等脚。net_at(side, index) 给出网络名。脚必须落在该边的中段，避免被判到邻边。"""
    span = (n_side - 1) * pitch
    half = body * 0.42
    pins = []
    num = 1
    for side, sx, sy, axis in (
        ("R", half, 0.0, "y"),
        ("T", 0.0, half, "x"),
        ("L", -half, 0.0, "y"),
        ("B", 0.0, -half, "x"),
    ):
        for i in range(n_side):
            along = -span / 2 + i * pitch
            if axis == "y":
                lx, ly = sx, sy + along
            else:
                lx, ly = sx + along, sy
            pins.append((str(num), net_at(side, i), lx, ly))
            num += 1
    return component(ref, cx, cy, 0, body, body, pins)


def scatter(rng: random.Random, cx: float, cy: float) -> tuple[float, float, float]:
    ang = rng.uniform(0, 2 * math.pi)
    dist = rng.uniform(700, 1400)
    rot = rng.choice((0, 90, 180, 270))
    return cx + dist * math.cos(ang), cy + dist * math.sin(ang), rot


def write(name: str, comps: list[dict]) -> None:
    path = OUT / name
    path.write_text(
        json.dumps({"components": comps}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(path, len(comps))


def case_qfn() -> None:
    """小 QFN：电源网用 +3V3 这种不带 VCC 前缀的名字，位号用普通 R/C/FB。"""
    rng = random.Random(1)
    cx = cy = 5000.0
    body = 200.0

    def net_at(side: str, i: int) -> str:
        table = {
            ("L", 0): "GPIO0", ("L", 1): "GPIO1", ("L", 2): "GND", ("L", 3): "+3V3",
            ("R", 0): "GPIO2", ("R", 1): "GPIO3", ("R", 2): "GND", ("R", 3): "+3V3",
            ("T", 0): "GPIO4", ("T", 1): "SWDIO", ("T", 2): "SWCLK", ("T", 3): "NRST",
            ("B", 0): "UART_TX", ("B", 1): "UART_RX", ("B", 2): "GND", ("B", 3): "+3V3",
        }
        return table[(side, i)]

    comps = [qfp("U1", cx, cy, body, 4, 32, net_at)]
    specs = [
        ("R1", "GPIO0", "LED0", "TOP"),
        ("R2", "GPIO1", "BTN", "TOP"),
        ("R3", "UART_TX", "TX_EXT", "TOP"),
        ("R4", "NRST", "+3V3", "TOP"),
        ("C1", "+3V3", "GND", "TOP"),
        ("C2", "+3V3", "GND", "BOTTOM"),
        ("FB1", "+3V3", "VSYS", "TOP"),
    ]
    for ref, a, b, layer in specs:
        x, y, rot = scatter(rng, cx, cy)
        lw, lh = (110.0, 44.0) if ref.startswith("FB") else ((80.0, 30.0) if ref.startswith("C") else (112.0, 52.0))
        comps.append(component(ref, x, y, rot, lw, lh, axial(a, b, lw), layer))
    write("synth_qfn.json", comps)


def case_ldo() -> None:
    """SOT-23-5 稳压：输入电容、输出电容、反馈分压。主芯片脚数很少。"""
    rng = random.Random(2)
    cx, cy = 3000.0, 3000.0
    # 左三脚 VIN/GND/EN，右两脚 FB/VOUT。脚的 |x| 大于 |y|，才会被分到左右边。
    pins = [
        ("1", "VIN", -50.0, -24.0),
        ("2", "GND", -50.0, 0.0),
        ("3", "EN", -50.0, 24.0),
        ("4", "FB", 50.0, -16.0),
        ("5", "VOUT", 50.0, 16.0),
    ]
    comps = [component("U2", cx, cy, 0, 140.0, 90.0, pins)]
    specs = [
        ("C10", "VIN", "GND", 80.0, 30.0),
        ("C11", "VOUT", "GND", 90.0, 40.0),
        ("R10", "VOUT", "FB", 112.0, 52.0),
        ("R11", "FB", "GND", 112.0, 52.0),
        ("C12", "EN", "GND", 60.0, 30.0),
    ]
    for ref, a, b, lw, lh in specs:
        x, y, rot = scatter(rng, cx, cy)
        comps.append(component(ref, x, y, rot, lw, lh, axial(a, b, lw)))
    write("synth_ldo.json", comps)


def case_xtal() -> None:
    """晶振位号用 X1，不依赖 Y 前缀。时钟脚在上边，其余边各有一颗电阻。"""
    rng = random.Random(3)
    cx = cy = 4000.0

    def net_at(side: str, i: int) -> str:
        if side == "T":
            return ["XI", "XO", "GND", "VDD"][i]
        if side == "B":
            return ["VDD", "GND", "BOOT", "UART"][i]
        if side == "L":
            return ["IO0", "IO1", "GND", "VDD"][i]
        return ["IO2", "IO3", "NRST", "GND"][i]

    comps = [qfp("U3", cx, cy, 280.0, 4, 36, net_at)]
    x, y, rot = scatter(rng, cx, cy)
    # 晶振四脚：信号在同一头，地在另一头
    xtal_pins = [
        ("1", "XI", -30.0, 40.0),
        ("2", "GND", -30.0, -40.0),
        ("3", "XO", 30.0, 40.0),
        ("4", "GND", 30.0, -40.0),
    ]
    comps.append(component("X1", x, y, rot, 170.0, 130.0, xtal_pins))
    specs = [
        ("C20", "XI", "GND", 70.0, 40.0),
        ("C21", "XO", "GND", 70.0, 40.0),
        ("R20", "XI", "XO", 112.0, 52.0),
        ("R21", "BOOT", "VDD", 112.0, 52.0),
        ("R22", "NRST", "VDD", 112.0, 52.0),
        ("R23", "IO0", "SIG0", 112.0, 52.0),
        ("C22", "VDD", "GND", 80.0, 30.0),
    ]
    for ref, a, b, lw, lh in specs:
        x, y, rot = scatter(rng, cx, cy)
        comps.append(component(ref, x, y, rot, lw, lh, axial(a, b, lw)))
    write("synth_xtal.json", comps)


def case_crowded() -> None:
    """下边 14 颗串联电阻，短边加起来超过对角线，必须退出第二圈。"""
    rng = random.Random(4)
    cx = cy = 6000.0
    n = 12

    def net_at(side: str, i: int) -> str:
        if side == "B":
            return f"SIG{i}"
        if i % 4 == 0:
            return "VDD"
        if i % 4 == 1:
            return "GND"
        return f"{side}{i}"

    comps = [qfp("U4", cx, cy, 480.0, n, 24, net_at)]
    for i in range(14):
        # 前 12 颗各接一个下边信号脚，多出来的两颗跟前两颗共用网络，仍然归下边
        sig = f"SIG{i % 12}"
        x, y, rot = scatter(rng, cx, cy)
        comps.append(component(
            f"R{100 + i}", x, y, rot, 112.0, 52.0, axial(sig, f"OUT{i}", 112.0),
        ))
    # 再放两颗去耦，确认电源件不会被这 14 颗挤丢
    for ref in ("C30", "C31"):
        x, y, rot = scatter(rng, cx, cy)
        comps.append(component(ref, x, y, rot, 80.0, 30.0, axial("VDD", "GND", 80.0)))
    write("synth_crowded.json", comps)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    case_qfn()
    case_ldo()
    case_xtal()
    case_crowded()


if __name__ == "__main__":
    main()
