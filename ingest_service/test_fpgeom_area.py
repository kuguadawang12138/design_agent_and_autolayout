# -*- coding: utf-8 -*-
"""面积与几何缓存的本地用例。不连数据库、不连 MinIO。

在 ingest_service 目录运行：

    python -m unittest test_fpgeom_area
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import numpy as np

from backfill_copper_area import area_if_missing
from footprint_json import (
    FootprintModel,
    PinGeometry,
    ShapeIsland,
    normalize_footprint_for_iou,
)
from footprint_db import assign_iou_ranks
from fpgeom import (
    RASTER_SIZE_MAX,
    FpGeomCache,
    best_geom_iou,
    best_raster_iou,
    build_fpgeom_from_footprint,
    choose_raster_size,
    footprint_copper_area_mils2,
    fpgeom_from_bytes,
    fpgeom_to_bytes,
    geometry_for_iou,
    islands_copper_area_mils2,
    raster_from_bytes,
    raster_from_cache,
    raster_to_bytes,
    rasterize_geometry,
    resize_raster,
)


def _square(x0: float, y0: float, width: float, height: float | None = None) -> np.ndarray:
    height = width if height is None else height
    return np.array(
        [
            [x0, y0],
            [x0 + width, y0],
            [x0 + width, y0 + height],
            [x0, y0 + height],
        ],
        dtype=np.float64,
    )


def _model(parts: list[tuple[np.ndarray, list[np.ndarray]]], units: str = "mils") -> FootprintModel:
    pins: list[PinGeometry] = []
    for index, (outline, holes) in enumerate(parts, start=1):
        pins.append(
            PinGeometry(
                pin_number=str(index),
                padstack="",
                x=float(outline[:, 0].mean()),
                y=float(outline[:, 1].mean()),
                rotation_deg=0.0,
                figure_type="rect",
                islands=[ShapeIsland(outline=outline, holes=list(holes))],
            )
        )
    return FootprintModel(name="T", units=units, project_code="P", refdes="U1", pins=pins)


def _longest_edge(cache: FpGeomCache) -> float:
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")
    for isl in cache.islands:
        min_x = min(min_x, float(isl.outline[:, 0].min()))
        min_y = min(min_y, float(isl.outline[:, 1].min()))
        max_x = max(max_x, float(isl.outline[:, 0].max()))
        max_y = max(max_y, float(isl.outline[:, 1].max()))
    return max(max_x - min_x, max_y - min_y)


class CopperAreaTests(unittest.TestCase):
    def test_square_area_uses_real_outline(self) -> None:
        model = _model([(_square(100.0, 200.0, 10.0), [])])
        self.assertAlmostEqual(footprint_copper_area_mils2(model), 100.0, places=6)

    def test_overlap_counted_once(self) -> None:
        no_overlap = _model([(_square(0, 0, 10), []), (_square(20, 0, 10), [])])
        full = _model([(_square(0, 0, 10), []), (_square(0, 0, 10), [])])
        partial = _model([(_square(0, 0, 10), []), (_square(5, 0, 10), [])])
        self.assertAlmostEqual(footprint_copper_area_mils2(no_overlap), 200.0, places=6)
        self.assertAlmostEqual(footprint_copper_area_mils2(full), 100.0, places=6)
        self.assertAlmostEqual(footprint_copper_area_mils2(partial), 150.0, places=6)

    def test_hole_subtracted(self) -> None:
        outer = _square(0, 0, 10)
        hole = _square(3, 3, 4)
        model = _model([(outer, [hole])])
        self.assertAlmostEqual(footprint_copper_area_mils2(model), 84.0, places=6)

    def test_mm_matches_mils(self) -> None:
        mils_per_mm = 1000.0 / 25.4
        in_mm = _model([(_square(0, 0, 1.0), [])], units="mm")
        in_mils = _model([(_square(0, 0, mils_per_mm), [])], units="mils")
        self.assertAlmostEqual(
            footprint_copper_area_mils2(in_mm),
            footprint_copper_area_mils2(in_mils),
            places=4,
        )


class FpgeomOriginalTests(unittest.TestCase):
    def test_saved_cache_keeps_original_size(self) -> None:
        model = _model([(_square(100.0, 200.0, 40.0, 10.0), [])])
        cache = build_fpgeom_from_footprint(model)
        self.assertEqual(cache.coord_space, "original")
        self.assertAlmostEqual(_longest_edge(cache), 40.0, places=4)
        self.assertNotAlmostEqual(_longest_edge(cache), 1.0, places=3)
        restored = fpgeom_from_bytes(fpgeom_to_bytes(cache))
        self.assertEqual(restored.coord_space, "original")
        self.assertAlmostEqual(_longest_edge(restored), 40.0, places=4)
        np.testing.assert_allclose(restored.islands[0].outline, cache.islands[0].outline)

    def test_same_shape_iou_after_shift_and_rotation(self) -> None:
        left = build_fpgeom_from_footprint(_model([(_square(100, 200, 40, 10), [])]))
        shifted = build_fpgeom_from_footprint(_model([(_square(-30, 8, 40, 10), [])]))
        iou, angle = best_geom_iou(left, shifted)
        self.assertGreater(iou, 0.99)
        self.assertEqual(angle, 0)

        rotated = _square(0, 0, 10, 40)
        turned = build_fpgeom_from_footprint(_model([(rotated, [])]))
        iou_rot, angle_rot = best_geom_iou(left, turned)
        self.assertGreater(iou_rot, 0.99)
        self.assertIn(angle_rot, (90, 270))

    def test_legacy_normalized_cache_still_matches(self) -> None:
        model = _model([(_square(100.0, 200.0, 40.0, 10.0), [])])
        fresh = build_fpgeom_from_footprint(model)
        legacy_body = build_fpgeom_from_footprint(normalize_footprint_for_iou(model)).to_dict()
        legacy_body.pop("coord_space")
        legacy = FpGeomCache.from_dict(legacy_body)
        self.assertEqual(legacy.coord_space, "normalized")
        self.assertAlmostEqual(_longest_edge(legacy), 1.0, places=4)
        iou, _angle = best_geom_iou(legacy, fresh)
        self.assertGreater(iou, 0.99)


class BackfillDecisionTests(unittest.TestCase):
    def test_only_null_area_is_filled(self) -> None:
        self.assertEqual(area_if_missing(None, 84.0), 84.0)
        self.assertIsNone(area_if_missing(12.5, 84.0))
        self.assertIsNone(area_if_missing(0.0, 84.0))

    def test_script_does_not_log_or_print_geometry_rewrite(self) -> None:
        text = Path(__file__).with_name("backfill_copper_area.py").read_text(encoding="utf-8")
        self.assertNotIn("\ufffd", text)
        self.assertNotIn("log_line", text)
        self.assertNotIn("delete_object", text)
        self.assertNotIn("save_object", text)
        self.assertNotIn("init_schema", text)
        for line in text.splitlines():
            if re.search(r"(^|[^\w])print\s*\(", line):
                self.assertNotIn("fpgeom", line.lower())

    def test_area_is_not_taken_from_scaled_cache(self) -> None:
        model = _model([(_square(0, 0, 10), [])])
        raw = footprint_copper_area_mils2(model)
        scaled = build_fpgeom_from_footprint(normalize_footprint_for_iou(model))
        from_scaled = islands_copper_area_mils2(scaled.islands, "mils")
        self.assertAlmostEqual(raw, 100.0, places=6)
        self.assertAlmostEqual(from_scaled, 1.0, places=4)
        self.assertNotAlmostEqual(from_scaled, raw, places=3)


class RasterAndRankTests(unittest.TestCase):
    def test_raster_size_follows_edge_then_pins(self) -> None:
        self.assertEqual(choose_raster_size(40, 2), 64)
        self.assertEqual(choose_raster_size(200, 80), 64)
        self.assertEqual(choose_raster_size(201, 2), 128)
        self.assertEqual(choose_raster_size(450, 2), 128)
        self.assertEqual(choose_raster_size(800, 2), 256)
        self.assertEqual(choose_raster_size(1199, 2), 512)
        self.assertEqual(choose_raster_size(1200, 2), 1024)
        self.assertEqual(choose_raster_size(2000, 2), 1024)
        self.assertEqual(choose_raster_size(0, 2), 64)
        self.assertEqual(choose_raster_size(None, 16), 128)
        self.assertEqual(choose_raster_size(None, 64), 256)
        self.assertEqual(choose_raster_size(None, 256), 512)
        self.assertEqual(choose_raster_size(None, 300), 1024)
        self.assertEqual(RASTER_SIZE_MAX, 1024)

    def test_raster_stores_one_orientation(self) -> None:
        cache = build_fpgeom_from_footprint(_model([(_square(100, 200, 40, 10), [])]))
        grid = raster_from_cache(cache)
        raw = raster_to_bytes(grid)
        restored = raster_from_bytes(raw)
        self.assertEqual(grid.shape, (64, 64))
        self.assertEqual(restored.shape, (64, 64))
        packed = 8 + 64 * ((64 + 7) // 8)
        self.assertEqual(len(raw), packed)
        self.assertGreater(int(grid.sum()), 0)

        large = raster_from_cache(
            build_fpgeom_from_footprint(_model([(_square(0, 0, 2000, 800), [])]))
        )
        self.assertEqual(large.shape, (RASTER_SIZE_MAX, RASTER_SIZE_MAX))

    def test_raster_iou_survives_resize(self) -> None:
        cache = build_fpgeom_from_footprint(_model([(_square(100, 200, 40, 10), [])]))
        coarse = raster_from_cache(cache)
        fine = rasterize_geometry(geometry_for_iou(cache), RASTER_SIZE_MAX)
        iou, angle = best_raster_iou(fine, resize_raster(coarse, RASTER_SIZE_MAX))
        self.assertGreater(iou, 0.9)
        self.assertEqual(angle, 0)
        iou_down, angle_down = best_raster_iou(coarse, resize_raster(fine, 64))
        self.assertGreater(iou_down, 0.9)
        self.assertEqual(angle_down, 0)

    def test_raster_iou_shift_and_rotation(self) -> None:
        wide = raster_from_cache(build_fpgeom_from_footprint(_model([(_square(100, 200, 40, 10), [])])))
        shifted = raster_from_cache(build_fpgeom_from_footprint(_model([(_square(-30, 8, 40, 10), [])])))
        iou, angle = best_raster_iou(wide, shifted)
        self.assertGreater(iou, 0.9)
        self.assertEqual(angle, 0)

        tall = raster_from_cache(build_fpgeom_from_footprint(_model([(_square(0, 0, 10, 40), [])])))
        iou_rot, angle_rot = best_raster_iou(wide, tall)
        self.assertGreater(iou_rot, 0.9)
        self.assertIn(angle_rot, (90, 270))

    def test_rank_keeps_only_high_scores_descending(self) -> None:
        ranked = assign_iou_ranks(
            [
                {"project_code": "A", "refdes": "U1", "iou": 0.79},
                {"project_code": "B", "refdes": "U2", "iou": 0.91},
                {"project_code": "C", "refdes": "U3", "iou": 0.80},
                {"project_code": "D", "refdes": "U4", "iou": 0.95},
            ]
        )
        self.assertEqual([row["refdes"] for row in ranked], ["U4", "U2", "U3"])
        self.assertEqual([row["rank_no"] for row in ranked], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
