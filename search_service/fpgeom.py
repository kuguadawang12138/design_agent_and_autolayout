# -*- coding: utf-8 -*-
"""
.fpgeom — 铜箔多边形缓存（msgpack 二进制）。

还原 JSON 时按原始坐标写入（不缩放、不平移）。
比对 IoU 时再临时把外接框中心移到原点、最长边缩放到 1，不写回缓存。
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import msgpack
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

from arc_iou import loop_from_points, make_shape, rotate_shape, shapes_iou
from footprint_json import (
    FootprintModel,
    _rotate_points_90,
    load_footprint_json,
    parse_package_family,
)

FPGEOM_MAGIC = b"FPGM"
FPGEOM_VERSION = 1
_HEADER = struct.Struct("4sB")


def _edges_from_raw(raw) -> list | None:
    if not raw:
        return None
    return [tuple(edge) for edge in raw]


def _hole_edges_from_raw(raw) -> list | None:
    if not raw:
        return None
    return [[tuple(edge) for edge in loop] for loop in raw]


def _island_to_dict(isl: "FpGeomIsland") -> dict[str, Any]:
    item: dict[str, Any] = {
        "outline": isl.outline.astype(np.float32).tolist(),
        "holes": [h.astype(np.float32).tolist() for h in isl.holes],
    }
    if isl.outline_edges:
        item["outline_edges"] = [list(edge) for edge in isl.outline_edges]
    if isl.hole_edges:
        item["hole_edges"] = [[list(edge) for edge in loop] for loop in isl.hole_edges]
    return item


@dataclass
class FpGeomIsland:
    outline: np.ndarray
    holes: list[np.ndarray] = field(default_factory=list)
    # 圆弧+直线的原始边。没有时 IoU 用轮廓折线。
    outline_edges: list | None = None
    hole_edges: list | None = None


@dataclass
class FpGeomCache:
    name: str
    pin_count: int
    units_src: str
    source_file: str = ""
    project_code: str = ""
    refdes: str = ""
    islands: list[FpGeomIsland] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # original=原始坐标；normalized=2026-10 之前按最长边缩放过的旧缓存
    coord_space: str = "normalized"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "pin_count": self.pin_count,
            "units_src": self.units_src,
            "source_file": self.source_file,
            "project_code": self.project_code,
            "refdes": self.refdes,
            "warnings": self.warnings,
            "coord_space": self.coord_space,
            "islands": [_island_to_dict(isl) for isl in self.islands],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FpGeomCache:
        islands: list[FpGeomIsland] = []
        for item in data.get("islands", []):
            outline = np.array(item.get("outline", []), dtype=np.float32)
            holes = [np.array(h, dtype=np.float32) for h in item.get("holes", [])]
            if len(outline) >= 3:
                islands.append(
                    FpGeomIsland(
                        outline=outline,
                        holes=holes,
                        outline_edges=_edges_from_raw(item.get("outline_edges")),
                        hole_edges=_hole_edges_from_raw(item.get("hole_edges")),
                    )
                )
        return cls(
            name=str(data.get("name", "")),
            pin_count=int(data.get("pin_count", 0)),
            units_src=str(data.get("units_src", "mils")),
            source_file=str(data.get("source_file", "")),
            project_code=str(data.get("project_code", "")),
            refdes=str(data.get("refdes", "")),
            islands=islands,
            warnings=list(data.get("warnings", [])),
            coord_space=str(data.get("coord_space") or "normalized"),
        )


def build_fpgeom_from_footprint(
    model: FootprintModel,
    *,
    source_file: str = "",
) -> FpGeomCache:
    """写入原始轮廓。归一化只在 best_geom_iou 里临时做，不落盘。"""
    islands: list[FpGeomIsland] = []
    for pin in model.pins:
        for isl in pin.islands:
            if len(isl.outline) < 3:
                continue
            holes = [h.astype(np.float32) for h in isl.holes if len(h) >= 3]
            islands.append(
                FpGeomIsland(
                    outline=isl.outline.astype(np.float32),
                    holes=holes,
                    outline_edges=list(isl.outline_edges) if isl.outline_edges else None,
                    hole_edges=[list(loop) for loop in isl.hole_edges] if isl.hole_edges else None,
                )
            )
    return FpGeomCache(
        name=model.name,
        pin_count=model.pin_count,
        units_src=model.units,
        source_file=source_file or f"{model.name}.json",
        project_code=model.project_code,
        refdes=model.refdes,
        islands=islands,
        warnings=list(model.warnings),
        coord_space="original",
    )


def save_fpgeom(cache: FpGeomCache, path: Path | str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(fpgeom_to_bytes(cache))


def fpgeom_to_bytes(cache: FpGeomCache) -> bytes:
    """序列化为 .fpgeom 二进制（供文件或数据库 BYTEA）。"""
    body = msgpack.packb(cache.to_dict(), use_bin_type=True)
    return _HEADER.pack(FPGEOM_MAGIC, FPGEOM_VERSION) + body


def fpgeom_from_bytes(data: bytes) -> FpGeomCache:
    """从二进制加载 FpGeomCache。"""
    if len(data) < _HEADER.size:
        raise ValueError("fpgeom 数据过短")
    magic, version = _HEADER.unpack_from(data)
    if magic != FPGEOM_MAGIC:
        raise ValueError("非 fpgeom 格式")
    if version != FPGEOM_VERSION:
        raise ValueError(f"不支持的 fpgeom 版本: {version}")
    payload = msgpack.unpackb(data[_HEADER.size :], raw=False)
    if not isinstance(payload, dict):
        raise ValueError("fpgeom 内容无效")
    cache = FpGeomCache.from_dict(payload)
    return cache


def load_fpgeom(path: Path | str) -> FpGeomCache:
    """读取 .fpgeom 文件。"""
    p = Path(path)
    cache = fpgeom_from_bytes(p.read_bytes())
    if not cache.name:
        cache.name = p.stem
    return cache


def _rotate_islands(islands: list[FpGeomIsland], k90: int) -> list[FpGeomIsland]:
    k90 %= 4
    if k90 == 0:
        return [
            FpGeomIsland(outline=isl.outline.copy(), holes=[h.copy() for h in isl.holes])
            for isl in islands
        ]
    out: list[FpGeomIsland] = []
    for isl in islands:
        outline = _rotate_points_90(isl.outline, k90).astype(np.float32)
        holes = [_rotate_points_90(h, k90).astype(np.float32) for h in isl.holes]
        out.append(FpGeomIsland(outline=outline, holes=holes))
    return out


# 长度单位换算到 mil。面积比较前统一用 mils²，避免 mm 与 mils 直接比。
_MILS_PER_UNIT = {
    "mil": 1.0,
    "mils": 1.0,
    "thou": 1.0,
    "th": 1.0,
    "inch": 1000.0,
    "in": 1000.0,
    "inches": 1000.0,
    "mm": 1000.0 / 25.4,
    "millimeter": 1000.0 / 25.4,
    "millimeters": 1000.0 / 25.4,
    "millimetre": 1000.0 / 25.4,
    "millimetres": 1000.0 / 25.4,
    "cm": 1000.0 / 2.54,
    "centimeter": 1000.0 / 2.54,
    "centimeters": 1000.0 / 2.54,
    "centimetre": 1000.0 / 2.54,
    "um": 1000.0 / 25400.0,
    "µm": 1000.0 / 25400.0,
    "micron": 1000.0 / 25400.0,
    "microns": 1000.0 / 25400.0,
    "m": 1000.0 / 0.0254,
    "meter": 1000.0 / 0.0254,
    "metre": 1000.0 / 0.0254,
}


def mils_per_unit(units: str | None) -> float:
    """1 个该单位等于多少 mil。认不出的单位按 mils 计。"""
    key = (units or "mils").strip().lower()
    return _MILS_PER_UNIT.get(key, 1.0)


def islands_copper_area_mils2(islands: list[Any], units: str | None) -> float:
    """按真实轮廓并集算面积（mils²）。重叠区域只计一次，孔洞从面积里扣除。

    坐标仍是封装原始单位，这里先换算到 mil 再算面积。
    面积只从这份原始轮廓算。2026-10 之前的 fpgeom 缓存按最长边缩放过，不能拿来反推面积。
    """
    scale = mils_per_unit(units)
    polys: list[Polygon] = []
    for isl in islands:
        outline = np.asarray(getattr(isl, "outline"), dtype=float)
        if outline.ndim != 2 or len(outline) < 3:
            continue
        holes: list[list[list[float]]] = []
        for hole in getattr(isl, "holes", []) or []:
            arr = np.asarray(hole, dtype=float)
            if arr.ndim == 2 and len(arr) >= 3:
                holes.append((arr * scale).tolist())
        try:
            poly = Polygon((outline * scale).tolist(), holes)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty:
                continue
            polys.append(poly)
        except Exception:
            continue
    if not polys:
        return 0.0
    merged = unary_union(polys)
    return float(merged.area)


def footprint_copper_area_mils2(model: FootprintModel) -> float:
    """封装铜箔真实面积（mils²），用还原后的岛轮廓，不是外接矩形。"""
    islands = [isl for pin in model.pins for isl in pin.islands]
    return islands_copper_area_mils2(islands, model.units)


def _islands_to_shapely(islands: list[FpGeomIsland]) -> Polygon:
    polys: list[Polygon] = []
    for isl in islands:
        try:
            holes = [h.tolist() for h in isl.holes if len(h) >= 3]
            poly = Polygon(isl.outline.tolist(), holes)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty:
                continue
            polys.append(poly)
        except Exception:
            continue
    if not polys:
        return Polygon()
    merged = unary_union(polys)
    if merged.geom_type == "Polygon":
        return merged
    if merged.geom_type == "MultiPolygon":
        return unary_union(list(merged.geoms))
    return Polygon()


def shapely_iou(a: Polygon, b: Polygon) -> float:
    if a.is_empty and b.is_empty:
        return 1.0
    if a.is_empty or b.is_empty:
        return 0.0
    inter = a.intersection(b).area
    union = a.union(b).area
    if union <= 0:
        return 0.0
    return float(inter / union)


def _normalize_islands_for_iou(islands: list[FpGeomIsland]) -> list[FpGeomIsland]:
    """比对前临时归一：外接框中心移到原点，最长边缩放到 1。不写回缓存。"""
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")
    for isl in islands:
        outline = isl.outline
        if outline.ndim != 2 or len(outline) == 0:
            continue
        min_x = min(min_x, float(outline[:, 0].min()))
        min_y = min(min_y, float(outline[:, 1].min()))
        max_x = max(max_x, float(outline[:, 0].max()))
        max_y = max(max_y, float(outline[:, 1].max()))
    if min_x == float("inf"):
        return islands
    extent = max(max_x - min_x, max_y - min_y, 1e-9)
    shift = np.array([(min_x + max_x) / 2.0, (min_y + max_y) / 2.0], dtype=np.float64)
    out: list[FpGeomIsland] = []
    for isl in islands:
        outline = ((isl.outline.astype(np.float64) - shift) / extent).astype(np.float32)
        holes = [((h.astype(np.float64) - shift) / extent).astype(np.float32) for h in isl.holes]
        out.append(FpGeomIsland(outline=outline, holes=holes))
    return out


def _islands_for_iou(cache: FpGeomCache) -> list[FpGeomIsland]:
    """旧缓存已经缩放过，直接比；新缓存是原始坐标，比之前再归一。"""
    if cache.coord_space == "original":
        return _normalize_islands_for_iou(cache.islands)
    return cache.islands


def geometry_for_iou(cache: FpGeomCache):
    """栅格用的并集几何只建一次。归一化在这里做，不写回缓存。"""
    return _islands_to_shapely(_islands_for_iou(cache))


def _iou_norm_params(islands: list[FpGeomIsland]) -> tuple[np.ndarray, float] | None:
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")
    for isl in islands:
        outline = isl.outline
        if outline.ndim != 2 or len(outline) == 0:
            continue
        min_x = min(min_x, float(outline[:, 0].min()))
        min_y = min(min_y, float(outline[:, 1].min()))
        max_x = max(max_x, float(outline[:, 0].max()))
        max_y = max(max_y, float(outline[:, 1].max()))
    if min_x == float("inf"):
        return None
    extent = max(max_x - min_x, max_y - min_y, 1e-9)
    shift = np.array([(min_x + max_x) / 2.0, (min_y + max_y) / 2.0], dtype=np.float64)
    return shift, extent


def _norm_edges(edges: list, shift: np.ndarray, extent: float) -> list:
    sx = float(shift[0])
    sy = float(shift[1])
    inv = 1.0 / extent
    out = []
    for edge in edges:
        if edge[0] == "L":
            _, x1, y1, x2, y2 = edge
            out.append(("L", (x1 - sx) * inv, (y1 - sy) * inv, (x2 - sx) * inv, (y2 - sy) * inv))
        else:
            _, cx, cy, radius, a0, a1 = edge
            out.append(("A", (cx - sx) * inv, (cy - sy) * inv, radius * inv, a0, a1))
    return out


def shape_for_iou(cache: FpGeomCache) -> dict:
    """比对用的圆弧图形只建一次。有原始边就用边，旧缓存退回轮廓折线。"""
    outers: list = []
    holes: list = []
    has_edges = any(isl.outline_edges for isl in cache.islands)
    if has_edges and cache.coord_space == "original":
        params = _iou_norm_params(cache.islands)
        if params is None:
            return make_shape([], [])
        shift, extent = params
        for isl in cache.islands:
            if isl.outline_edges:
                outers.append(_norm_edges(isl.outline_edges, shift, extent))
            elif len(isl.outline) >= 3:
                outers.append(_norm_edges(loop_from_points(isl.outline), shift, extent))
            for loop in isl.hole_edges or []:
                holes.append(_norm_edges(loop, shift, extent))
            if not isl.hole_edges:
                for hole in isl.holes:
                    if len(hole) >= 3:
                        holes.append(_norm_edges(loop_from_points(hole), shift, extent))
        return make_shape(outers, holes)
    for isl in _islands_for_iou(cache):
        outers.append(loop_from_points(isl.outline))
        for hole in isl.holes:
            if len(hole) >= 3:
                holes.append(loop_from_points(hole))
    return make_shape(outers, holes)


def pack_iou_shape(shape: dict) -> bytes:
    return msgpack.packb(shape, use_bin_type=True)


def unpack_iou_shape(blob: bytes) -> dict:
    data = msgpack.unpackb(blob, raw=False)
    data["outers"] = [[tuple(edge) for edge in loop] for loop in data.get("outers", [])]
    data["holes"] = [[tuple(edge) for edge in loop] for loop in data.get("holes", [])]
    data["obox"] = [tuple(box) for box in data.get("obox", [])]
    data["hbox"] = [tuple(box) for box in data.get("hbox", [])]
    return data


_QUERY_SLOT: dict[str, Any] = {"blob": b"", "geom": None}


def cached_query_geom(query_blob: bytes):
    """同一进程里同一份 query 图形只还原一次，后面的候选直接复用。"""
    slot = _QUERY_SLOT
    if slot["geom"] is None or slot["blob"] != query_blob:
        slot["blob"] = query_blob
        slot["geom"] = unpack_iou_shape(query_blob)
    return slot["geom"]


def best_iou_rotations(query_shape, cand_shape) -> tuple[float, int]:
    """候选图形只建一次，四个方向旋转同一份边，再和这一份 query 比。"""
    best_iou, best_angle = -1.0, 0
    for k in range(4):
        rotated = cand_shape if k == 0 else rotate_shape(cand_shape, k)
        iou = shapes_iou(query_shape, rotated)
        if iou > best_iou:
            best_iou, best_angle = iou, k * 90
    return best_iou, best_angle


def best_geom_iou(a: FpGeomCache, b: FpGeomCache) -> tuple[float, int]:
    return best_iou_rotations(shape_for_iou(a), shape_for_iou(b))


# 落盘：魔数 RSTR + uint16 高 + uint16 宽 + 按行 packbits 的 0/1。只存 0°。
# 边长像素取 64/128/256/512/1024。最大 1024，约 128KB。
RASTER_SIZE_MIN = 64
RASTER_SIZE_MAX = 1024
RASTER_SIZE = RASTER_SIZE_MAX
# 最长边（mils）落在这些上限以内时用对应档。200 mil 约 5mm，两脚小电阻电容在这一档。
# 1200 mil 起直接用 1024，不到 1200 又超过 800 的用 512。
_RASTER_EDGE_LIMITS = (
    (200.0, 64),
    (450.0, 128),
    (800.0, 256),
)
_RASTER_EDGE_1024 = 1200.0
_RASTER_MAGIC = b"RSTR"
_RASTER_HEADER = struct.Struct("<4sHH")


def _iter_polygons(geom):
    if geom is None or geom.is_empty:
        return
    kind = geom.geom_type
    if kind == "Polygon":
        yield geom
        return
    if kind in ("MultiPolygon", "GeometryCollection"):
        for part in geom.geoms:
            yield from _iter_polygons(part)


def _fill_ring(grid: np.ndarray, pts: np.ndarray, value: int) -> None:
    """扫描线填充。像素中心与原来的 contains 判断一致。"""
    size = grid.shape[0]
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) >= 2 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    if len(pts) < 3:
        return
    px = (pts[:, 0] + 0.5) * size - 0.5
    py = (pts[:, 1] + 0.5) * size - 0.5
    x1 = np.roll(px, -1)
    y1 = np.roll(py, -1)
    y_min = np.minimum(py, y1)
    y_max = np.maximum(py, y1)
    active = (y_max - y_min) > 1e-12
    if not np.any(active):
        return
    x0 = px[active]
    y0 = py[active]
    x1 = x1[active]
    y1 = y1[active]
    y_min = y_min[active]
    y_max = y_max[active]
    row0 = max(int(np.floor(y_min.min())), 0)
    row1 = min(int(np.ceil(y_max.max())), size - 1)
    if row1 < row0:
        return
    scan = np.arange(row0, row1 + 1, dtype=np.float64)
    cross = (y_min[:, None] <= scan[None, :]) & (scan[None, :] < y_max[:, None])
    span = (y1 - y0)[:, None]
    x_at = np.where(cross, x0[:, None] + (scan[None, :] - y0[:, None]) / span * (x1 - x0)[:, None], np.nan)
    for offset, row in enumerate(range(row0, row1 + 1)):
        xs = x_at[:, offset]
        xs = xs[np.isfinite(xs)]
        if xs.size < 2:
            continue
        xs.sort()
        for index in range(0, xs.size - 1, 2):
            left = int(np.ceil(xs[index]))
            right = int(np.floor(xs[index + 1]))
            if left <= right:
                grid[row, max(left, 0) : min(right, size - 1) + 1] = value


def rasterize_geometry(geom, size: int = RASTER_SIZE) -> np.ndarray:
    """把已经归一（中心在原点、最长边为 1）的几何画成 uint8 栅格。"""
    grid = np.zeros((size, size), dtype=np.uint8)
    for poly in _iter_polygons(geom):
        _fill_ring(grid, np.asarray(poly.exterior.coords), 1)
        for ring in poly.interiors:
            _fill_ring(grid, np.asarray(ring.coords), 0)
    return grid


def cache_longest_edge_mils(cache: FpGeomCache) -> float | None:
    """原始坐标下的外接框最长边，换算成 mil。旧的归一化缓存量不出真实边长。"""
    if cache.coord_space != "original":
        return None
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")
    for isl in cache.islands:
        outline = isl.outline
        if outline.ndim != 2 or len(outline) == 0:
            continue
        min_x = min(min_x, float(outline[:, 0].min()))
        min_y = min(min_y, float(outline[:, 1].min()))
        max_x = max(max_x, float(outline[:, 0].max()))
        max_y = max(max_y, float(outline[:, 1].max()))
    if min_x == float("inf"):
        return None
    edge = max(max_x - min_x, max_y - min_y) * mils_per_unit(cache.units_src)
    if edge <= 1e-6:
        return None
    return float(edge)


def choose_raster_size(edge_mils: float | None, pin_count: int = 0) -> int:
    """有边长就按最长边分档；没有边长再按脚数。结果只可能是 64/128/256/512/1024。"""
    if edge_mils is not None and edge_mils > 0:
        if edge_mils >= _RASTER_EDGE_1024:
            return RASTER_SIZE_MAX
        for limit, size in _RASTER_EDGE_LIMITS:
            if edge_mils <= limit:
                return size
        return 512
    pins = int(pin_count or 0)
    if pins <= 2:
        return 64
    if pins <= 16:
        return 128
    if pins <= 64:
        return 256
    if pins <= 256:
        return 512
    return RASTER_SIZE_MAX


def raster_from_cache(cache: FpGeomCache, size: int | None = None) -> np.ndarray:
    if size is None:
        size = choose_raster_size(cache_longest_edge_mils(cache), cache.pin_count)
    return rasterize_geometry(geometry_for_iou(cache), int(size))


def resize_raster(grid: np.ndarray, size: int) -> np.ndarray:
    """把正方形栅格拉到同一边长再比。2 的幂之间用整数倍放大或缩小。"""
    image = np.ascontiguousarray(grid, dtype=np.uint8)
    height, width = image.shape
    if height == size and width == size:
        return image
    if height == width and size > height and size % height == 0:
        factor = size // height
        return np.repeat(np.repeat(image, factor, axis=0), factor, axis=1)
    if height == width and height > size and height % size == 0:
        factor = height // size
        blocks = image.reshape(size, factor, size, factor)
        return (blocks.max(axis=(1, 3)) > 0).astype(np.uint8)
    rows = np.linspace(0, max(height - 1, 0), size).astype(np.int32)
    cols = np.linspace(0, max(width - 1, 0), size).astype(np.int32)
    return image[rows][:, cols]


def _pack_grid(grid: np.ndarray) -> np.ndarray:
    return np.packbits(np.ascontiguousarray(grid, dtype=np.uint8), axis=1)


def raster_query_packs(grid: np.ndarray) -> list[np.ndarray]:
    """query 的四个方向各压成一行 8 像素一组的位图，候选保持 0°。"""
    packs: list[np.ndarray] = []
    for turn in range(4):
        rotated = grid if turn == 0 else np.rot90(grid, -turn)
        packs.append(_pack_grid(rotated))
    return packs


def raster_to_bytes(arr: np.ndarray) -> bytes:
    """uint8 的 0/1 按行 packbits 后写入。1024×1024 为 128KB，另加 8 字节头。"""
    grid = np.ascontiguousarray(arr, dtype=np.uint8)
    if grid.ndim != 2:
        raise ValueError("栅格必须是二维数组")
    packed = _pack_grid(grid)
    return _RASTER_HEADER.pack(_RASTER_MAGIC, grid.shape[0], grid.shape[1]) + packed.tobytes()


def _raster_header(data: bytes) -> tuple[int, int, bytes]:
    if len(data) < _RASTER_HEADER.size:
        raise ValueError("栅格数据过短")
    magic, height, width = _RASTER_HEADER.unpack_from(data)
    if magic != _RASTER_MAGIC:
        raise ValueError("非栅格格式")
    raw = data[_RASTER_HEADER.size :]
    packed_width = (int(width) + 7) // 8
    need = int(height) * packed_width
    if len(raw) != need:
        raise ValueError("栅格长度与宽高不符")
    return int(height), int(width), raw


def raster_packed_from_bytes(data: bytes) -> np.ndarray:
    """读出按行压缩的位图，不展开成 1024×1024。"""
    height, width, raw = _raster_header(data)
    packed_width = (width + 7) // 8
    return np.frombuffer(raw, dtype=np.uint8).reshape((height, packed_width)).copy()


def raster_from_bytes(data: bytes) -> np.ndarray:
    height, width, raw = _raster_header(data)
    packed_width = (width + 7) // 8
    packed = np.frombuffer(raw, dtype=np.uint8).reshape((height, packed_width))
    return np.unpackbits(packed, axis=1, count=width)


def raster_packed_for_query(data: bytes, query_size: int) -> np.ndarray:
    """候选和 query 边长不同时，先缩放到 query 的边长再压位。"""
    height, width, _raw = _raster_header(data)
    if height == query_size and width == query_size:
        return raster_packed_from_bytes(data)
    return _pack_grid(resize_raster(raster_from_bytes(data), query_size))


def best_packed_raster_iou(packs: list[np.ndarray], cand_packed: np.ndarray) -> tuple[float, int]:
    """在压缩位图上数 1 的个数。四个方向已经准备好，不再展开整幅图。"""
    cand = np.ascontiguousarray(cand_packed, dtype=np.uint8)
    best_iou, best_angle = -1.0, 0
    for turn, packed in enumerate(packs):
        inter = int(np.bitwise_count(np.bitwise_and(packed, cand)).sum(dtype=np.int64))
        union = int(np.bitwise_count(np.bitwise_or(packed, cand)).sum(dtype=np.int64))
        if union <= 0:
            iou = 1.0 if inter == 0 else 0.0
        else:
            iou = inter / union
        if iou > best_iou:
            best_iou, best_angle = iou, turn * 90
    return best_iou, best_angle


def best_raster_iou(query_grid: np.ndarray, cand_grid: np.ndarray) -> tuple[float, int]:
    """query 转四个方向，候选保持 0°。"""
    return best_packed_raster_iou(raster_query_packs(query_grid), _pack_grid(cand_grid))


def load_fpgeoms_from_dir(fpgeom_dir: Path | str) -> tuple[list[FpGeomCache], list[dict]]:
    d = Path(fpgeom_dir)
    caches: list[FpGeomCache] = []
    errors: list[dict] = []
    for path in sorted(d.glob("*.fpgeom")):
        if path.parent != d:
            continue
        try:
            caches.append(load_fpgeom(path))
        except Exception as exc:
            errors.append(
                {"file": path.name, "error_type": type(exc).__name__, "error": str(exc)}
            )
    return caches, errors


def _compare_fpgeom_pair(task: tuple[FpGeomCache, FpGeomCache]) -> dict:
    """单对 fpgeom IoU（供多进程调用）。"""
    ca, cb = task
    iou, angle = best_geom_iou(ca, cb)
    return {
        "a": ca.name,
        "b": cb.name,
        "pin_count": ca.pin_count,
        "iou": round(iou, 4),
        "best_rotation_deg": angle,
    }


def compare_fpgeoms_in_dir(
    fpgeom_dir: Path | str,
    *,
    workers: int = 1,
) -> tuple[list[dict], list[dict]]:
    """目录内 .fpgeom 两两几何 IoU（pin 数相同才比，支持多进程）。"""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    caches, load_errors = load_fpgeoms_from_dir(fpgeom_dir)
    pair_tasks: list[tuple[FpGeomCache, FpGeomCache]] = []
    for i, ca in enumerate(caches):
        for cb in caches[i + 1 :]:
            if ca.pin_count != cb.pin_count:
                continue
            pair_tasks.append((ca, cb))

    results: list[dict] = []
    n_workers = max(1, workers)
    if n_workers > 1 and len(pair_tasks) > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(_compare_fpgeom_pair, task) for task in pair_tasks]
            for fut in as_completed(futures):
                results.append(fut.result())
    else:
        for task in pair_tasks:
            results.append(_compare_fpgeom_pair(task))
    return results, load_errors
