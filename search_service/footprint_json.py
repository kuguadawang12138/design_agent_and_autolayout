# -*- coding: utf-8 -*-
"""从设计部 JSON 还原封装铜箔几何（参数化 pad + shapes/void 图元）。"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from arc_iou import arc_from_three, loop_from_points

# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------


@dataclass
class ShapeIsland:
    """单个外轮廓岛 + 孔洞（均为 footprint 坐标系下的折线点列）。"""

    outline: np.ndarray  # (N, 2)
    holes: list[np.ndarray] = field(default_factory=list)
    # 圆弧+直线。IoU 用这份，折线点列仍留给栅格和面积。
    outline_edges: list | None = None
    hole_edges: list | None = None


@dataclass
class PinGeometry:
    pin_number: str
    padstack: str
    x: float
    y: float
    rotation_deg: float
    figure_type: str
    islands: list[ShapeIsland] = field(default_factory=list)


@dataclass
class FootprintModel:
    name: str
    units: str = "mils"
    project_code: str = ""
    project_name: str | None = None
    refdes: str = ""
    layers: int | None = None
    devicetype: str = ""
    packagesymbol: str = ""
    value: str = ""
    top_side: bool = True
    refdes_x: float | None = None
    refdes_y: float | None = None
    pins: list[PinGeometry] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def pin_count(self) -> int:
        return len(self.pins)

    def bounds(self) -> tuple[float, float, float, float]:
        xs: list[float] = []
        ys: list[float] = []
        for pin in self.pins:
            for isl in pin.islands:
                for arr in [isl.outline, *isl.holes]:
                    if len(arr):
                        xs.extend(arr[:, 0].tolist())
                        ys.extend(arr[:, 1].tolist())
        if not xs:
            return 0.0, 0.0, 0.0, 0.0
        return min(xs), min(ys), max(xs), max(ys)

    def centroid(self) -> np.ndarray:
        pts: list[list[float]] = []
        for pin in self.pins:
            pts.append([pin.x, pin.y])
        if not pts:
            return np.zeros(2)
        return np.mean(np.array(pts, dtype=float), axis=0)


# ---------------------------------------------------------------------------
# 封装族解析（检索硬过滤预留）
# ---------------------------------------------------------------------------

FAMILY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "BGA": ("bga", "csp", "lga"),
    "QFN": ("qfn", "dfn", "vdfn"),
    "QFP": ("qfp", "lqfp", "tqfp", "mqfp"),
    "WSON": ("wson", "dson", "son"),
    "SOP": ("sop", "soic", "ssop", "tssop", "msop"),
    "PASSIVE": ("0201", "0402", "0603", "0805", "1206", "c0201", "c0402", "r0201"),
    "CONNECTOR": ("conn", "header", "usb", "hdmi", "fpc"),
}


def parse_package_family(name: str) -> str:
    """从封装名启发式解析族，未识别返回 UNKNOWN。"""
    lower = name.lower()
    for family, keys in FAMILY_KEYWORDS.items():
        if any(k in lower for k in keys):
            return family
    return "UNKNOWN"


def hard_filter_pin_count(query: FootprintModel, candidate: FootprintModel, exact: bool = True) -> bool:
    """引脚数硬过滤：默认必须相等。"""
    if exact:
        return query.pin_count == candidate.pin_count
    if query.pin_count <= 0 or candidate.pin_count <= 0:
        return False
    rel = abs(query.pin_count - candidate.pin_count) / max(query.pin_count, candidate.pin_count)
    return rel <= 0.05


def hard_filter_package_family(
    query_name: str,
    candidate_name: str,
    *,
    enabled: bool = False,
) -> bool:
    """封装族硬过滤（默认关闭，待规则定稿后开启）。"""
    if not enabled:
        return True
    qf = parse_package_family(query_name)
    cf = parse_package_family(candidate_name)
    if qf == "UNKNOWN" or cf == "UNKNOWN":
        return True
    return qf == cf


# ---------------------------------------------------------------------------
# JSON 读取与格式归一
# ---------------------------------------------------------------------------

_JSON_LITERALS = frozenset({"true", "false", "null"})


class FootprintLoadError(Exception):
    """封装 JSON 加载失败。"""

    def __init__(self, path: Path | str, message: str) -> None:
        self.path = Path(path)
        self.message = message
        super().__init__(f"{self.path.name}: {message}")


def _normalize_design_json_text(text: str) -> str:
    """修正设计部导出中常见的非标准 JSON（如 \"units\":mils）。"""
    def _quote_ident(match: re.Match[str]) -> str:
        val = match.group(2)
        if val.lower() in _JSON_LITERALS:
            return match.group(0)
        return f'"{match.group(1)}":"{val}"{match.group(3)}'

    return re.sub(
        r'"([^"\\]+)"\s*:\s*([A-Za-z_][A-Za-z0-9_\-]*)\s*([,}\]])',
        _quote_ident,
        text,
    )


def _parse_json_text(text: str, *, source: Path | str) -> dict[str, Any]:
    """解析设计部 JSON 文本（文件或上传内容共用）。"""
    src = Path(source)
    text = text.strip()
    if not text:
        raise FootprintLoadError(src, "空内容")

    data: Any = None
    last_err: json.JSONDecodeError | None = None
    for candidate in (text, _normalize_design_json_text(text)):
        try:
            data = json.loads(candidate)
            last_err = None
            break
        except json.JSONDecodeError as exc:
            last_err = exc

    if last_err is not None:
        if text.startswith('"shapes"') or text.startswith("shapes"):
            wrapped = "{" + text.rstrip(",") + "}"
            try:
                frag = json.loads(_normalize_design_json_text(wrapped))
            except json.JSONDecodeError as exc:
                raise FootprintLoadError(
                    src,
                    f"JSON 解析失败（line {exc.lineno}, col {exc.colno}）: {exc.msg}",
                ) from exc
            data = {
                "units": "mils",
                "pins": [
                    {
                        "pin_number": 1,
                        "padstack": src.stem,
                        "x": 0,
                        "y": 0,
                        "rotation": 0,
                        "figure_type": "SHAPE",
                        "shape_name": src.stem,
                        **frag,
                    }
                ],
            }
        else:
            raise FootprintLoadError(
                src,
                f"JSON 解析失败（line {last_err.lineno}, col {last_err.colno}）: {last_err.msg}",
            ) from last_err

    if not isinstance(data, dict):
        raise FootprintLoadError(src, "JSON 顶层必须是对象")
    return data


def _parse_json_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig")
    return _parse_json_text(text, source=path)


def _parse_optional_int(raw: Any) -> int | None:
    if raw is None or raw == "":
        return None
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return None


def _to_float(raw: Any, default: float = 0.0) -> float:
    """数值字段兼容字符串形式，如 \"29.53\"。"""
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _get_first_raw(data: dict[str, Any], *keys: str) -> Any:
    """按优先级取第一个非空字段（如顶层 X / x）。"""
    for key in keys:
        if key not in data:
            continue
        val = data[key]
        if val is None:
            continue
        if isinstance(val, str) and val.strip() == "":
            continue
        return val
    return None


def _pin_silent_skip_geometry(figure_type: str, w: float, h: float, pin: dict[str, Any]) -> bool:
    """
    无有效可绘尺寸时静默跳过（如 NTH 孔位 width/height=0），不记告警。
    仅当 width/height 等参数有正数时才期望生成铜箔轮廓。
    """
    ft = figure_type.upper()
    if ft in ("NULL", "", "APERTURE"):
        return True
    if ft in ("SHAPE", "FLASH"):
        raw = pin.get("shapes")
        if raw is None or raw == []:
            return True
        return False
    if w <= 0:
        return True
    if ft not in ("CIRCLE", "SQUARE") and h <= 0:
        return True
    return False


def _resolve_pin_number(item: dict[str, Any]) -> str:
    """pin_number 为空时用 padstack 作为 pin 标识。"""
    pin_number = str(item.get("pin_number", "")).strip()
    if pin_number:
        return pin_number
    padstack = str(item.get("padstack", "")).strip()
    return padstack or "?"


def _parse_top_side(raw: Any) -> bool:
    """1/true=正面(Top)，0/false=反面(Bottom)。"""
    from top_side_util import parse_top_side

    return parse_top_side(raw)


def _resolve_project_code(data: dict[str, Any]) -> tuple[str, str | None]:
    """
    项目编码：可为纯数字/编码，或「1188009画板子」合并串。
    也支持 project_code + project_name 分开填。
    """
    code = str(data.get("project_code", "")).strip()
    name = str(data.get("project_name", "")).strip() or None
    if not code and name:
        code = name
    return code, name


def _void_to_holes(void_raw: Any, warnings: list[str]) -> list[np.ndarray]:
    """
    void 与 shape 同级：单层图元数组 [{type:segment|arc|polygon}, ...]。
    多个 polygon 图元表示多个孔；无孔为 []。
    若误写为 [[...]] 双层数组，兼容解析并提示。
    """
    if not void_raw or not isinstance(void_raw, list):
        return []

    # 误写：[[{...}]] 或 [[{...}], [{...}]]
    if void_raw and isinstance(void_raw[0], list):
        warnings.append("void 应为与 shape 同级的单层图元数组，检测到双层数组")
        holes: list[np.ndarray] = []
        for part in void_raw:
            if isinstance(part, list):
                hp = _primitives_to_polygon(part, warnings)
                if len(hp) >= 3:
                    holes.append(hp)
        return holes

    if not isinstance(void_raw[0], dict):
        return []

    # 标准单层：polygon 图元各自成孔；segment/arc 连续拼成一个孔
    holes: list[np.ndarray] = []
    chunk: list[dict] = []
    for prim in void_raw:
        if not isinstance(prim, dict):
            continue
        t = str(prim.get("type", "")).lower()
        if t == "polygon":
            if chunk:
                hp = _primitives_to_polygon(chunk, warnings)
                if len(hp) >= 3:
                    holes.append(hp)
                chunk = []
            hp = _primitives_to_polygon([prim], warnings)
            if len(hp) >= 3:
                holes.append(hp)
        else:
            chunk.append(prim)
    if chunk:
        hp = _primitives_to_polygon(chunk, warnings)
        if len(hp) >= 3:
            holes.append(hp)
    return holes


def _normalize_shape_islands(raw: Any) -> list[dict[str, Any]]:
    """统一为 [{shape:[], void:[]}, ...]；仅支持数组格式。"""
    if raw is None:
        return []
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        shape = item.get("shape", item.get("outline", []))
        void = item.get("void", item.get("voids", []))
        out.append({"shape": shape or [], "void": void or []})
    return out


def _pin_record_to_island_defs(
    pin: dict[str, Any],
    *,
    pin_number: str,
    warnings: list[str],
) -> list[dict[str, Any]]:
    if "shapes" not in pin:
        return []
    raw = pin["shapes"]
    if isinstance(raw, dict):
        warnings.append(
            f"pin {pin_number}: shapes 格式错误，必须为数组 [{{\"shape\":[], \"void\":[]}}]，当前为对象"
        )
        return []
    if not isinstance(raw, list):
        warnings.append(
            f"pin {pin_number}: shapes 格式错误，必须为数组 [{{\"shape\":[], \"void\":[]}}]"
        )
        return []
    return _normalize_shape_islands(raw)


def _build_footprint_model(data: dict[str, Any], *, name: str) -> FootprintModel:
    """从已解析 dict 构建 FootprintModel。"""
    units = str(data.get("units", "mils")).strip() or "mils"
    project_code, project_name = _resolve_project_code(data)
    if not project_code:
        raise ValueError("缺少必填字段 project_code（项目/板子编码）")

    refdes_raw = data.get("refdes")
    if refdes_raw is None or str(refdes_raw).strip() == "":
        raise ValueError("缺少必填字段 refdes（位号，单个字符串如 U1）")
    if isinstance(refdes_raw, list):
        raise ValueError("refdes 只支持单个字符串，不支持数组")
    refdes = str(refdes_raw).replace("@", "").strip().upper()

    packagesymbol = str(data.get("packagesymbol", "")).strip()
    display_name = packagesymbol or refdes or name

    refdes_x: float | None = None
    refdes_y: float | None = None
    raw_x = _get_first_raw(data, "X", "x")
    raw_y = _get_first_raw(data, "Y", "y")
    if raw_x is not None:
        refdes_x = _to_float(raw_x)
    if raw_y is not None:
        refdes_y = _to_float(raw_y)

    model = FootprintModel(
        name=display_name,
        units=units,
        project_code=project_code,
        project_name=project_name,
        refdes=refdes,
        layers=_parse_optional_int(data.get("layers")),
        devicetype=str(data.get("devicetype", "")).strip(),
        packagesymbol=packagesymbol,
        value=str(data.get("value", "")).strip(),
        top_side=_parse_top_side(data.get("top_side", 1)),
        refdes_x=refdes_x,
        refdes_y=refdes_y,
    )

    pins_raw = data.get("pins", [])
    if not pins_raw:
        model.warnings.append("缺少 pins[]，无法还原完整封装")
        return model

    for item in pins_raw:
        if not isinstance(item, dict):
            continue
        pin_number = _resolve_pin_number(item)
        padstack = str(item.get("padstack", ""))
        x = _to_float(item.get("x", 0))
        y = _to_float(item.get("y", 0))
        rotation = _to_float(item.get("rotation", 0))
        figure_type = str(item.get("figure_type", "RECTANGLE")).upper()
        offset_x = _to_float(item.get("offset_x", 0))
        offset_y = _to_float(item.get("offset_y", 0))
        w = _to_float(item.get("width", 0))
        h = _to_float(item.get("height", 0), w)
        silent_skip = _pin_silent_skip_geometry(figure_type, w, h, item)

        local_islands = _build_local_islands(item, figure_type, model.warnings)
        placed: list[ShapeIsland] = []
        for loc in local_islands:
            outline = _transform_points(loc.outline, x, y, rotation, offset_x, offset_y)
            holes = [
                _transform_points(h, x, y, rotation, offset_x, offset_y) for h in loc.holes
            ]
            if len(outline) >= 3:
                placed.append(
                    ShapeIsland(
                        outline=outline,
                        holes=holes,
                        outline_edges=_transform_edges(
                            loc.outline_edges, x, y, rotation, offset_x, offset_y
                        )
                        if loc.outline_edges
                        else None,
                        hole_edges=[
                            _transform_edges(loop, x, y, rotation, offset_x, offset_y)
                            for loop in loc.hole_edges
                        ]
                        if loc.hole_edges
                        else None,
                    )
                )

        if not placed and not silent_skip and figure_type not in ("NULL", ""):
            if not any(w.startswith(f"pin {pin_number}:") for w in model.warnings):
                model.warnings.append(f"pin {pin_number}: 未生成有效铜箔轮廓")

        model.pins.append(
            PinGeometry(
                pin_number=pin_number,
                padstack=padstack,
                x=x,
                y=y,
                rotation_deg=rotation,
                figure_type=figure_type,
                islands=placed,
            )
        )

    return model


def load_footprint_json(path: Path | str) -> FootprintModel:
    """加载设计部 JSON 文件，还原为 FootprintModel。"""
    p = Path(path)
    try:
        data = _parse_json_file(p)
    except FootprintLoadError:
        raise
    except OSError as exc:
        raise FootprintLoadError(p, f"读取文件失败: {exc}") from exc
    try:
        return _build_footprint_model(data, name=p.stem)
    except ValueError as exc:
        raise FootprintLoadError(p, str(exc)) from exc


def load_footprint_json_from_bytes(
    content: bytes | str,
    *,
    source_name: str = "upload.json",
) -> FootprintModel:
    """从上传字节/文本解析封装 JSON（无需落盘）。"""
    src = Path(source_name)
    if isinstance(content, bytes):
        text = content.decode("utf-8-sig")
    else:
        text = content
    data = _parse_json_text(text, source=src)
    try:
        return _build_footprint_model(data, name=src.stem)
    except ValueError as exc:
        raise FootprintLoadError(src, str(exc)) from exc


# ---------------------------------------------------------------------------
# 几何：参数化 + 图元折线化
# ---------------------------------------------------------------------------


def _transform_points(
    pts: np.ndarray,
    tx: float,
    ty: float,
    rot_deg: float,
    offset_x: float,
    offset_y: float,
) -> np.ndarray:
    if len(pts) == 0:
        return pts
    rad = math.radians(rot_deg)
    c, s = math.cos(rad), math.sin(rad)
    out = np.empty_like(pts)
    for i, (xl, yl) in enumerate(pts):
        ox, oy = xl + offset_x, yl + offset_y
        out[i, 0] = tx + ox * c - oy * s
        out[i, 1] = ty + ox * s + oy * c
    return out


def _circle_edges(r: float) -> list:
    return [
        ("A", 0.0, 0.0, float(r), 0.0, math.pi),
        ("A", 0.0, 0.0, float(r), math.pi, 2 * math.pi),
    ]


def _rect_edges(w: float, h: float) -> list:
    hw, hh = w / 2.0, h / 2.0
    return [
        ("L", -hw, -hh, hw, -hh),
        ("L", hw, -hh, hw, hh),
        ("L", hw, hh, -hw, hh),
        ("L", -hw, hh, -hw, -hh),
    ]


def _swap_xy_edge(edge: tuple) -> tuple:
    if edge[0] == "L":
        return ("L", edge[2], edge[1], edge[4], edge[3])
    return ("A", edge[2], edge[1], edge[3], math.pi / 2 - edge[4], math.pi / 2 - edge[5])


def _oblong_edges(w: float, h: float, along_x: bool) -> list:
    if not along_x:
        return [_swap_xy_edge(edge) for edge in _oblong_edges(h, w, True)]
    radius = h / 2.0
    if w <= h + 1e-9:
        return _circle_edges(w / 2.0)
    straight = w / 2.0 - radius
    return [
        ("L", -straight, -radius, straight, -radius),
        ("A", straight, 0.0, radius, -math.pi / 2, math.pi / 2),
        ("L", straight, radius, -straight, radius),
        ("A", -straight, 0.0, radius, math.pi / 2, math.pi / 2 + math.pi),
    ]


def _rounded_rect_edges(w: float, h: float, r: float) -> list:
    hw, hh = w / 2.0, h / 2.0
    r = min(r, hw, hh)
    if r <= 1e-9:
        return _rect_edges(w, h)
    corners = [
        (hw - r, hh - r, 0.0, math.pi / 2),
        (-hw + r, hh - r, math.pi / 2, math.pi),
        (-hw + r, -hh + r, math.pi, 3 * math.pi / 2),
        (hw - r, -hh + r, 3 * math.pi / 2, 2 * math.pi),
    ]
    edges = []
    for index, (cx, cy, a0, a1) in enumerate(corners):
        edges.append(("A", cx, cy, r, a0, a1))
        ncx, ncy, na0, _na1 = corners[(index + 1) % 4]
        x1 = cx + r * math.cos(a1)
        y1 = cy + r * math.sin(a1)
        x2 = ncx + r * math.cos(na0)
        y2 = ncy + r * math.sin(na0)
        if abs(x1 - x2) > 1e-9 or abs(y1 - y2) > 1e-9:
            edges.append(("L", x1, y1, x2, y2))
    return edges


def _parametric_edges(pin: dict[str, Any], figure_type: str, w: float, h: float) -> list | None:
    ft = figure_type.upper()
    if ft == "CIRCLE":
        return _circle_edges(w / 2.0)
    if ft in ("SQUARE", "RECTANGLE"):
        return _rect_edges(w, h)
    if ft in ("OBLONG_X", "OBLONG_Y"):
        return _oblong_edges(w, h, ft == "OBLONG_X")
    if ft == "ROUNDED_RECTANGLE":
        radius = _to_float(pin.get("corner_radius", 0), min(w, h) * 0.25)
        if radius <= 0:
            radius = min(w, h) * 0.25
        return _rounded_rect_edges(w, h, min(radius, w / 2.0, h / 2.0))
    return None


def _transform_edges(edges, tx, ty, rot_deg, offset_x, offset_y):
    if not edges:
        return edges
    rad = math.radians(rot_deg)
    c, s = math.cos(rad), math.sin(rad)

    def xf(x: float, y: float) -> tuple[float, float]:
        x += offset_x
        y += offset_y
        return tx + x * c - y * s, ty + x * s + y * c

    out = []
    for edge in edges:
        if edge[0] == "L":
            x1, y1 = xf(edge[1], edge[2])
            x2, y2 = xf(edge[3], edge[4])
            out.append(("L", x1, y1, x2, y2))
        else:
            cx, cy = xf(edge[1], edge[2])
            out.append(("A", cx, cy, edge[3], edge[4] + rad, edge[5] + rad))
    return out


def _edge_ends(edge: tuple) -> tuple[tuple[float, float], tuple[float, float]]:
    if edge[0] == "L":
        return (edge[1], edge[2]), (edge[3], edge[4])
    _kind, cx, cy, radius, a0, a1 = edge
    return (
        (cx + radius * math.cos(a0), cy + radius * math.sin(a0)),
        (cx + radius * math.cos(a1), cy + radius * math.sin(a1)),
    )


def _reverse_edge(edge: tuple) -> tuple:
    if edge[0] == "L":
        return ("L", edge[3], edge[4], edge[1], edge[2])
    return ("A", edge[1], edge[2], edge[3], edge[5], edge[4])


def _order_edges(edges: list) -> list:
    if not edges:
        return []
    remaining = list(edges)
    ordered = [remaining.pop(0)]
    tol = 1e-3
    guard = 0
    while remaining and guard < 10000:
        guard += 1
        end = _edge_ends(ordered[-1])[1]
        found = False
        for index, edge in enumerate(remaining):
            start, stop = _edge_ends(edge)
            if _near(start, end, tol):
                ordered.append(edge)
                remaining.pop(index)
                found = True
                break
            if _near(stop, end, tol):
                ordered.append(_reverse_edge(edge))
                remaining.pop(index)
                found = True
                break
        if found:
            continue
        start = _edge_ends(ordered[0])[0]
        for index, edge in enumerate(remaining):
            a, b = _edge_ends(edge)
            if _near(b, start, tol):
                ordered.insert(0, edge)
                remaining.pop(index)
                found = True
                break
            if _near(a, start, tol):
                ordered.insert(0, _reverse_edge(edge))
                remaining.pop(index)
                found = True
                break
        if not found:
            break
    return ordered


def _arc_prim_edge(prim: dict, warnings: list[str]) -> tuple:
    sx, sy = _to_float(prim["start_x"]), _to_float(prim["start_y"])
    ex, ey = _to_float(prim["end_x"]), _to_float(prim["end_y"])
    if prim.get("center_x") is not None and prim.get("radius") is not None:
        cx = _to_float(prim["center_x"])
        cy = _to_float(prim["center_y"])
        radius = _to_float(prim["radius"])
        cw = int(prim.get("cw", 1))
        a0 = math.atan2(sy - cy, sx - cx)
        a1 = math.atan2(ey - cy, ex - cx)
        if abs(sx - ex) < 1e-6 and abs(sy - ey) < 1e-6:
            return ("A", cx, cy, radius, a0, a0 + 2 * math.pi)
        if cw:
            if a1 >= a0:
                a1 -= 2 * math.pi
        elif a1 <= a0:
            a1 += 2 * math.pi
        return ("A", cx, cy, radius, a0, a1)
    if prim.get("mid_x") is not None:
        return arc_from_three(
            sx, sy, _to_float(prim["mid_x"]), _to_float(prim["mid_y"]), ex, ey
        )
    warnings.append("arc 缺少 center 或 mid 点，退化为直线")
    return ("L", sx, sy, ex, ey)


def _primitives_to_edges(primitives: list[dict], warnings: list[str]) -> list | None:
    if not primitives:
        return None
    if len(primitives) == 1 and str(primitives[0].get("type", "")).lower() == "polygon":
        return loop_from_points(_parse_points_str(primitives[0].get("points", ""))) or None
    raw = []
    for prim in primitives:
        kind = str(prim.get("type", "")).lower()
        if kind == "segment":
            raw.append(
                (
                    "L",
                    _to_float(prim["x1"]),
                    _to_float(prim["y1"]),
                    _to_float(prim["x2"]),
                    _to_float(prim["y2"]),
                )
            )
        elif kind == "arc":
            raw.append(_arc_prim_edge(prim, warnings))
        elif kind == "polygon":
            return loop_from_points(_parse_points_str(prim.get("points", ""))) or None
    ordered = _order_edges(raw)
    return ordered or None


def _void_to_hole_edges(void_raw: Any, warnings: list[str]) -> list:
    if not void_raw or not isinstance(void_raw, list):
        return []
    if isinstance(void_raw[0], list):
        holes = []
        for part in void_raw:
            if isinstance(part, list):
                edges = _primitives_to_edges(part, warnings)
                if edges:
                    holes.append(edges)
        return holes
    if not isinstance(void_raw[0], dict):
        return []
    holes = []
    chunk: list[dict] = []
    for prim in void_raw:
        if not isinstance(prim, dict):
            continue
        kind = str(prim.get("type", "")).lower()
        if kind == "polygon":
            if chunk:
                edges = _primitives_to_edges(chunk, warnings)
                if edges:
                    holes.append(edges)
                chunk = []
            edges = _primitives_to_edges([prim], warnings)
            if edges:
                holes.append(edges)
        else:
            chunk.append(prim)
    if chunk:
        edges = _primitives_to_edges(chunk, warnings)
        if edges:
            holes.append(edges)
    return holes


def _island_with_edges(outline, holes, outline_edges, hole_edges) -> ShapeIsland:
    if not outline_edges and len(outline) >= 3:
        outline_edges = loop_from_points(outline)
    if not hole_edges:
        hole_edges = [loop_from_points(hole) for hole in holes if len(hole) >= 3]
    return ShapeIsland(
        outline=outline,
        holes=holes,
        outline_edges=outline_edges or None,
        hole_edges=hole_edges or None,
    )


def _build_local_islands(
    pin: dict[str, Any],
    figure_type: str,
    warnings: list[str],
) -> list[ShapeIsland]:
    ft = figure_type.upper()
    if ft == "DONUT":
        w = _to_float(pin.get("width", 0))
        if w <= 0:
            return []
        inner_r = _to_float(pin.get("inner_diameter", w / 2), w / 2) / 2.0
        return [
            _island_with_edges(
                _circle_points(w / 2.0, 48),
                [_circle_points(inner_r, 32)],
                _circle_edges(w / 2.0),
                [_circle_edges(inner_r)],
            )
        ]

    if ft in ("SHAPE", "FLASH"):
        pin_number = str(pin.get("pin_number", ""))
        defs = _pin_record_to_island_defs(pin, pin_number=pin_number, warnings=warnings)
        out: list[ShapeIsland] = []
        for d in defs:
            outline = _primitives_to_polygon(d.get("shape", []), warnings)
            holes = _void_to_holes(d.get("void"), warnings)
            if len(outline) >= 3:
                out.append(
                    _island_with_edges(
                        outline,
                        holes,
                        _primitives_to_edges(d.get("shape", []), warnings),
                        _void_to_hole_edges(d.get("void"), warnings),
                    )
                )
        return out

    if ft in ("NULL", "APERTURE"):
        return []

    w = _to_float(pin.get("width", 0))
    h = _to_float(pin.get("height", 0), w)
    if w <= 0:
        return []

    poly = _parametric_polygon(pin, ft, w, h)
    if len(poly) < 3:
        return []
    return [_island_with_edges(poly, [], _parametric_edges(pin, ft, w, h), [])]


def _parametric_polygon(pin: dict[str, Any], figure_type: str, w: float, h: float) -> np.ndarray:
    ft = figure_type.upper()
    hw, hh = w / 2.0, h / 2.0

    if ft == "CIRCLE":
        return _circle_points(hw, 48)
    if ft == "SQUARE":
        return _rect_points(w, h)
    if ft == "RECTANGLE":
        return _rect_points(w, h)
    if ft in ("OBLONG_X", "OBLONG_Y"):
        return _oblong_points(w, h, ft == "OBLONG_X")
    if ft == "ROUNDED_RECTANGLE":
        r = _to_float(pin.get("corner_radius", 0), min(w, h) * 0.25)
        if r <= 0:
            r = min(w, h) * 0.25
        return _rounded_rect_points(w, h, min(r, hw, hh))
    if ft == "OCTAGON":
        return _regular_polygon(8, hw, hh)
    if ft in ("N_SIDED_POLYGON", "PENTAGON"):
        sides = int(pin.get("sides", 5 if ft == "PENTAGON" else 6) or 6)
        return _regular_polygon(sides, hw, hh)
    if ft == "TRIANGLE":
        return _regular_polygon(3, hw, hh)
    if ft == "DIAMOND":
        return np.array([[0, hh], [hw, 0], [0, -hh], [-hw, 0]], dtype=float)
    if ft == "CROSS":
        t = min(w, h) * 0.3
        return np.array(
            [
                [-t, hh],
                [t, hh],
                [t, t],
                [hw, t],
                [hw, -t],
                [t, -t],
                [t, -hh],
                [-t, -hh],
                [-t, -t],
                [-hw, -t],
                [-hw, t],
                [-t, t],
            ],
            dtype=float,
        )
    if ft in ("HEXAGON_X", "HEXAGON_Y"):
        return _regular_polygon(6, hw, hh)

    return _rect_points(w, h)


def _circle_points(r: float, n: int) -> np.ndarray:
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.column_stack([r * np.cos(angles), r * np.sin(angles)])


def _rect_points(w: float, h: float) -> np.ndarray:
    hw, hh = w / 2.0, h / 2.0
    return np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]], dtype=float)


def _oblong_points(w: float, h: float, along_x: bool) -> np.ndarray:
    """长圆（stadium）：长边方向用圆角矩形近似。"""
    if along_x:
        return _rounded_rect_points(w, h, h / 2.0)
    pts = _rounded_rect_points(h, w, w / 2.0)
    return np.column_stack([pts[:, 1], pts[:, 0]])


def _rounded_rect_points(w: float, h: float, r: float) -> np.ndarray:
    hw, hh = w / 2.0, h / 2.0
    r = min(r, hw, hh)
    pts: list[list[float]] = []
    corners = [
        (hw - r, hh - r, 0, math.pi / 2),
        (-hw + r, hh - r, math.pi / 2, math.pi),
        (-hw + r, -hh + r, math.pi, 3 * math.pi / 2),
        (hw - r, -hh + r, 3 * math.pi / 2, 2 * math.pi),
    ]
    for cx, cy, a0, a1 in corners:
        pts.extend(_arc_sample(cx, cy, r, r, a0, a1, 8))
    return np.array(pts, dtype=float)


def _regular_polygon(sides: int, hw: float, hh: float) -> np.ndarray:
    angles = np.linspace(0, 2 * math.pi, sides, endpoint=False)
    return np.column_stack([hw * np.cos(angles), hh * np.sin(angles)])


def _arc_sample(
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    a0: float,
    a1: float,
    steps: int,
) -> list[list[float]]:
    if a1 < a0:
        a1 += 2 * math.pi
    ts = np.linspace(a0, a1, max(steps, 2))
    return [[cx + rx * math.cos(t), cy + ry * math.sin(t)] for t in ts]


def _primitives_to_polygon(primitives: list[dict], warnings: list[str]) -> np.ndarray:
    """将 segment/arc/polygon 图元链成闭合折线。"""
    if not primitives:
        return np.array([], dtype=float).reshape(0, 2)

    if len(primitives) == 1 and primitives[0].get("type") == "polygon":
        return _parse_points_str(primitives[0].get("points", ""))

    edges: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for prim in primitives:
        t = str(prim.get("type", "")).lower()
        if t == "segment":
            p0 = (_to_float(prim["x1"]), _to_float(prim["y1"]))
            p1 = (_to_float(prim["x2"]), _to_float(prim["y2"]))
            edges.append((p0, p1))
        elif t == "arc":
            pts = _arc_to_points(prim, warnings)
            for i in range(len(pts) - 1):
                edges.append((tuple(pts[i]), tuple(pts[i + 1])))
        elif t == "polygon":
            poly = _parse_points_str(prim.get("points", ""))
            if len(poly) >= 3:
                return poly

    return _chain_edges(edges)


def _parse_points_str(s: str) -> np.ndarray:
    pts: list[list[float]] = []
    for part in str(s).split(";"):
        part = part.strip()
        if not part:
            continue
        xy = part.split(",")
        if len(xy) >= 2:
            pts.append([_to_float(xy[0]), _to_float(xy[1])])
    return np.array(pts, dtype=float) if pts else np.array([], dtype=float).reshape(0, 2)


def _arc_to_points(prim: dict, warnings: list[str]) -> list[list[float]]:
    sx, sy = _to_float(prim["start_x"]), _to_float(prim["start_y"])
    ex, ey = _to_float(prim["end_x"]), _to_float(prim["end_y"])

    if prim.get("center_x") is not None and prim.get("radius") is not None:
        cx = _to_float(prim["center_x"])
        cy = _to_float(prim["center_y"])
        r = _to_float(prim["radius"])
        cw = int(prim.get("cw", 1))
        a0 = math.atan2(sy - cy, sx - cx)
        a1 = math.atan2(ey - cy, ex - cx)
        if cw:
            if a1 >= a0:
                a1 -= 2 * math.pi
        else:
            if a1 <= a0:
                a1 += 2 * math.pi
        steps = max(8, int(abs(a1 - a0) / (math.pi / 18)))
        if abs(sx - ex) < 1e-6 and abs(sy - ey) < 1e-6:
            # 整圆
            angles = np.linspace(0, 2 * math.pi, 32, endpoint=False)
            return [[cx + r * math.cos(a), cy + r * math.sin(a)] for a in angles] + [[sx, sy]]
        ts = np.linspace(a0, a1, steps + 1)
        return [[cx + r * math.cos(t), cy + r * math.sin(t)] for t in ts]

    if prim.get("mid_x") is not None:
        mx, my = _to_float(prim["mid_x"]), _to_float(prim["mid_y"])
        # 三点定弧 — 简化：采样 start→mid→end
        return [[sx, sy], [mx, my], [ex, ey]]

    warnings.append("arc 缺少 center 或 mid 点，退化为直线")
    return [[sx, sy], [ex, ey]]


def _chain_edges(edges: list[tuple[tuple[float, float], tuple[float, float]]]) -> np.ndarray:
    if not edges:
        return np.array([], dtype=float).reshape(0, 2)
    tol = 1e-3
    remaining = edges[:]
    start, end = remaining.pop(0)
    chain = [start, end]
    while remaining:
        found = False
        for i, (a, b) in enumerate(remaining):
            if _near(b, chain[-1], tol):
                chain.append(a)
                remaining.pop(i)
                found = True
                break
            if _near(a, chain[-1], tol):
                chain.append(b)
                remaining.pop(i)
                found = True
                break
            if _near(a, chain[0], tol):
                chain.insert(0, b)
                remaining.pop(i)
                found = True
                break
            if _near(b, chain[0], tol):
                chain.insert(0, a)
                remaining.pop(i)
                found = True
                break
        if not found:
            break
    if _near(chain[0], chain[-1], tol) and len(chain) > 2:
        chain.pop()
    return np.array(chain, dtype=float)


def _near(a: tuple[float, float], b: tuple[float, float], tol: float) -> bool:
    return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol


# ---------------------------------------------------------------------------
# 尺寸归一（供 .fpgeom 缓存） / SVG 导出
# ---------------------------------------------------------------------------


def _clone_model_geometry(
    model: FootprintModel,
    *,
    name: str | None = None,
    units: str | None = None,
) -> FootprintModel:
    """深拷贝铜箔几何。"""
    cloned = FootprintModel(
        name=name or model.name,
        units=units or model.units,
        warnings=list(model.warnings),
    )
    for pin in model.pins:
        islands = [
            ShapeIsland(outline=isl.outline.copy(), holes=[h.copy() for h in isl.holes])
            for isl in pin.islands
        ]
        cloned.pins.append(
            PinGeometry(
                pin_number=pin.pin_number,
                padstack=pin.padstack,
                x=pin.x,
                y=pin.y,
                rotation_deg=pin.rotation_deg,
                figure_type=pin.figure_type,
                islands=islands,
            )
        )
    return cloned


def _rotate_points_90(points: np.ndarray, k: int) -> np.ndarray:
    """绕原点逆时针旋转 k*90°。"""
    if len(points) == 0:
        return points
    k %= 4
    if k == 0:
        return points.copy()
    x, y = points[:, 0], points[:, 1]
    if k == 1:
        return np.column_stack([y, -x])
    if k == 2:
        return np.column_stack([-x, -y])
    return np.column_stack([-y, x])


def normalize_footprint_for_iou(model: FootprintModel) -> FootprintModel:
    """尺寸归一：bbox 中心移到原点，最长边缩放到 1.0。"""
    min_x, min_y, max_x, max_y = model.bounds()
    if max_x <= min_x and max_y <= min_y:
        return _clone_model_geometry(model, units="normalized")

    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    extent = max(max_x - min_x, max_y - min_y, 1e-9)
    shift = np.array([cx, cy], dtype=float)

    out = FootprintModel(name=model.name, units="normalized", warnings=list(model.warnings))
    for pin in model.pins:
        islands: list[ShapeIsland] = []
        for isl in pin.islands:
            outline = (isl.outline - shift) / extent
            holes = [(h - shift) / extent for h in isl.holes if len(h)]
            islands.append(ShapeIsland(outline=outline, holes=holes))
        out.pins.append(
            PinGeometry(
                pin_number=pin.pin_number,
                padstack=pin.padstack,
                x=(pin.x - cx) / extent,
                y=(pin.y - cy) / extent,
                rotation_deg=pin.rotation_deg,
                figure_type=pin.figure_type,
                islands=islands,
            )
        )
    return out


def render_footprint_svg(
    model: FootprintModel,
    out_path: Path | str,
    *,
    width_px: int = 1200,
    height_px: int = 1200,
    pad_color: str = "#58a6ff",
    hole_color: str = "#0d1117",
    title: str | None = None,
) -> None:
    """导出还原 SVG。"""
    out_path = Path(out_path)
    min_x, min_y, max_x, max_y = model.bounds()
    bw = max(max_x - min_x, 1.0)
    bh = max(max_y - min_y, 1.0)
    margin = max(bw, bh) * 0.08 + 10.0
    vx, vy = min_x - margin, min_y - margin
    vw, vh = bw + 2 * margin, bh + 2 * margin

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>\n',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width_px}" height="{height_px}" ',
        f'viewBox="{vx:.2f} {vy:.2f} {vw:.2f} {vh:.2f}">\n',
        f'<rect x="{vx:.2f}" y="{vy:.2f}" width="{vw:.2f}" height="{vh:.2f}" fill="#0d1117"/>\n',
    ]
    if title is None:
        title = f"{model.name} | {model.pin_count} pins | {model.units}"
    lines.append(
        f'<text x="{vx + vw*0.02:.2f}" y="{vy + vh*0.06:.2f}" fill="#e6edf3" '
        f'font-family="Segoe UI,Arial,sans-serif" font-size="{vh*0.04:.1f}">{title}</text>\n'
    )

    for pin in model.pins:
        for isl in pin.islands:
            d = _poly_to_svg_path(isl.outline)
            lines.append(f'<path d="{d}" fill="{pad_color}" fill-rule="evenodd" stroke="#1f6feb" stroke-width="{max(vw,vh)*0.001:.3f}"/>\n')
            for hole in isl.holes:
                dh = _poly_to_svg_path(hole)
                lines.append(f'<path d="{dh}" fill="{hole_color}"/>\n')

    if model.warnings:
        warn = " | ".join(model.warnings[:3])
        lines.append(
            f'<text x="{vx + vw*0.02:.2f}" y="{vy + vh*0.96:.2f}" fill="#f0883e" '
            f'font-size="{vh*0.028:.1f}">{warn}</text>\n'
        )

    lines.append("</svg>\n")
    out_path.write_text("".join(lines), encoding="utf-8")


def _poly_to_svg_path(poly: np.ndarray) -> str:
    if len(poly) == 0:
        return ""
    parts = [f"M {poly[0,0]:.4f} {poly[0,1]:.4f}"]
    for i in range(1, len(poly)):
        parts.append(f"L {poly[i,0]:.4f} {poly[i,1]:.4f}")
    parts.append("Z")
    return " ".join(parts)
