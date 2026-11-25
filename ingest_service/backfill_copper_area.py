# -*- coding: utf-8 -*-
"""把 footprint_library 里面积为空的记录，按 MinIO 上的原始 JSON 算出来写回去。

已有面积不改。用法（在 ingest_service 目录）：

    python backfill_copper_area.py
    python backfill_copper_area.py --config config/footprint_sim_st.yaml
"""

from __future__ import annotations

import argparse
from typing import Any


def area_if_missing(existing: Any, computed: float) -> float | None:
    """已有面积保持不动，只补空值。返回 None 表示这一列不更新。"""
    if existing is not None:
        return None
    return float(computed)


def main() -> None:
    parser = argparse.ArgumentParser(description="补齐 footprint_library 中为空的铜箔面积（mils²）")
    parser.add_argument("--config", default=None, help="YAML 配置路径")
    args = parser.parse_args()

    from settings import bootstrap

    info = bootstrap(args.config)
    print(f"配置: {info['config_path']}")

    from footprint_db import connect, load_footprint_json_from_bytes, resolve_object_finish_time
    from fpgeom import footprint_copper_area_mils2
    from minio_store import resolve_stored_bytes

    filled = 0
    still_empty = 0
    failed: list[str] = []
    done = 0

    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, project_code, refdes, source_json, fpgeom, copper_area
                FROM footprint_library
                """
            )
            rows = cur.fetchall()

        for row in rows:
            row_id, project_code, refdes, source_json, stored_fpgeom, copper_area = row
            label = f"{project_code}/{refdes}"
            model = None
            if source_json:
                try:
                    raw = resolve_stored_bytes(source_json)
                    model = load_footprint_json_from_bytes(
                        raw, source_name=f"{project_code}_{refdes}.json"
                    )
                except Exception as exc:
                    model = None
                    if copper_area is None:
                        failed.append(f"{label}: {type(exc).__name__}")
                        still_empty += 1

            if model is not None and copper_area is None:
                try:
                    new_area = area_if_missing(copper_area, footprint_copper_area_mils2(model))
                    if new_area is not None:
                        with conn.cursor() as cur:
                            cur.execute(
                                """
                                UPDATE footprint_library
                                SET copper_area = %s
                                WHERE id = %s AND copper_area IS NULL
                                """,
                                (new_area, row_id),
                            )
                            filled += cur.rowcount
                except Exception as exc:
                    failed.append(f"{label}: {type(exc).__name__}")
                    still_empty += 1
            elif copper_area is None and model is None and not source_json:
                failed.append(f"{label}: 没有原始 JSON")
                still_empty += 1

            if model is not None:
                _rewrite_stored_geometry(
                    conn,
                    row_id,
                    project_code,
                    refdes,
                    stored_fpgeom,
                    model,
                    resolve_object_finish_time,
                )

            done += 1
            if done % 50 == 0:
                conn.commit()
        conn.commit()
    finally:
        conn.close()

    print(f"面积为空已写入 {filled} 条")
    print(f"面积仍为空 {still_empty} 条")
    for item in failed:
        print(f"  {item}")


def _rewrite_stored_geometry(
    conn,
    row_id: int,
    project_code,
    refdes,
    stored_fpgeom,
    model,
    resolve_finish_time,
) -> None:
    """用原始坐标覆盖库里已有的几何缓存。不打印、不写业务日志。失败时把原对象放回去。"""
    try:
        from minio.error import S3Error

        from fpgeom import build_fpgeom_from_footprint, fpgeom_to_bytes
        from minio_store import _client, _finish_date, get_object_bytes, object_key_of, put_object

        cache = build_fpgeom_from_footprint(model, source_file=f"{project_code}_{refdes}.json")
        data = fpgeom_to_bytes(cache)
        key = object_key_of(stored_fpgeom)
        client, bucket = _client()
        if not key:
            day = _finish_date(resolve_finish_time(conn, project_code, None))
            safe_ref = str(refdes or "").strip().upper() or "NA"
            safe_code = str(project_code or "").strip() or "NA"
            key = f"{day.year:04d}/{day.month:02d}/{day.day:02d}/{safe_code}_{safe_ref}.fpgeom"

        previous = None
        try:
            client.stat_object(bucket, key)
        except S3Error as exc:
            if getattr(exc, "code", "") not in ("NoSuchKey", "NoSuchObject"):
                return
        else:
            previous = get_object_bytes(key)
            client.remove_object(bucket, key)

        try:
            put_object(key, data, "application/octet-stream")
        except Exception:
            if previous is not None:
                put_object(key, previous, "application/octet-stream")
            return

        with conn.cursor() as cur:
            cur.execute(
                "UPDATE footprint_library SET fpgeom = %s, geom_islands = %s WHERE id = %s",
                (key, len(cache.islands), row_id),
            )
    except Exception:
        return


if __name__ == "__main__":
    main()
