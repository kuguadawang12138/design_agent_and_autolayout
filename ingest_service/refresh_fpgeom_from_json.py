# -*- coding: utf-8 -*-
"""用库里已有的原始 JSON 重写 fpgeom。

旧文件只有折线点。这里按 JSON 重新生成，圆和圆弧写成边，再覆盖原来的对象。
面积、栅格不动。

用法（在 ingest_service 目录）：

    python refresh_fpgeom_from_json.py
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

_TZ = ZoneInfo("Asia/Shanghai")


def _say(text: str) -> None:
    """只打到控制台，给 nohup 看。不写日志文件。"""
    now = datetime.now(_TZ).strftime("%Y-%m-%d %H:%M:%S")
    print(f"{now} {text}", flush=True)


def _edge_counts(cache) -> tuple[int, int, int]:
    """有边的岛数、圆弧条数、直线条数。"""
    islands = 0
    arcs = 0
    lines = 0
    for isl in cache.islands:
        loops = []
        if isl.outline_edges:
            loops.append(isl.outline_edges)
            islands += 1
        loops.extend(isl.hole_edges or [])
        for loop in loops:
            for edge in loop:
                if edge[0] == "A":
                    arcs += 1
                else:
                    lines += 1
    return islands, arcs, lines


def rewrite_fpgeom_from_model(
    conn,
    row_id: int,
    project_code: str,
    refdes: str,
    stored_fpgeom: Any,
    model,
) -> dict[str, Any]:
    """覆盖这一条的 fpgeom。失败时把刚删掉的对象放回去。"""
    from minio.error import S3Error

    from footprint_db import resolve_object_finish_time
    from fpgeom import build_fpgeom_from_footprint, fpgeom_to_bytes
    from minio_store import _client, _finish_date, get_object_bytes, object_key_of, put_object

    cache = build_fpgeom_from_footprint(model, source_file=f"{project_code}_{refdes}.json")
    data = fpgeom_to_bytes(cache)
    key = object_key_of(stored_fpgeom)
    client, bucket = _client()
    if not key:
        day = _finish_date(resolve_object_finish_time(conn, project_code, None))
        safe_ref = str(refdes or "").strip().upper() or "NA"
        safe_code = str(project_code or "").strip() or "NA"
        key = f"{day.year:04d}/{day.month:02d}/{day.day:02d}/{safe_code}_{safe_ref}.fpgeom"

    previous = None
    try:
        client.stat_object(bucket, key)
    except S3Error as exc:
        if getattr(exc, "code", "") not in ("NoSuchKey", "NoSuchObject"):
            raise
    else:
        previous = get_object_bytes(key)
        client.remove_object(bucket, key)

    try:
        put_object(key, data, "application/octet-stream")
    except Exception:
        if previous is not None:
            put_object(key, previous, "application/octet-stream")
        raise

    with conn.cursor() as cur:
        cur.execute(
            "UPDATE footprint_library SET fpgeom = %s, geom_islands = %s WHERE id = %s",
            (key, len(cache.islands), row_id),
        )
    edged, arcs, lines = _edge_counts(cache)
    return {
        "key": key,
        "islands": len(cache.islands),
        "edged_islands": edged,
        "arcs": arcs,
        "lines": lines,
        "bytes": len(data),
        "pins": int(getattr(model, "pin_count", 0) or 0),
    }


def main() -> None:
    from settings import bootstrap

    info = bootstrap()
    started = time.perf_counter()
    _say(f"[重写fpgeom] 开始 配置={info['config_path']}")

    from footprint_db import connect, load_footprint_json_from_bytes
    from minio_store import resolve_stored_bytes

    rewritten = 0
    skipped: list[str] = []
    failed: list[str] = []
    done = 0

    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, project_code, refdes, source_json, fpgeom
                FROM footprint_library
                ORDER BY id
                """
            )
            rows = cur.fetchall()
        total = len(rows)
        _say(f"[重写fpgeom] 清单 器件={total} 条，逐条用 JSON 重写 fpgeom")

        for row_id, project_code, refdes, source_json, stored_fpgeom in rows:
            done += 1
            code = str(project_code or "").strip()
            ref = str(refdes or "").strip()
            head = f"[重写fpgeom] {done}/{total} id={row_id} 项目={code} 位号={ref}"
            json_key = str(source_json or "").strip()
            old_key = str(stored_fpgeom or "").strip()
            if not json_key:
                skipped.append(f"id={row_id} 项目={code} 位号={ref} 原因=没有原始JSON")
                _say(f"{head} 结果=跳过 原因=没有原始JSON 旧fpgeom={old_key or '-'}")
                continue
            t0 = time.perf_counter()
            try:
                raw = resolve_stored_bytes(source_json)
                model = load_footprint_json_from_bytes(
                    raw, source_name=f"{code}_{ref}.json"
                )
                saved = rewrite_fpgeom_from_model(
                    conn, row_id, code, ref, stored_fpgeom, model
                )
                rewritten += 1
                spent = int((time.perf_counter() - t0) * 1000)
                _say(
                    f"{head} 结果=成功 脚数={saved['pins']} 岛={saved['islands']} "
                    f"带边的岛={saved['edged_islands']} 圆弧={saved['arcs']} 直线={saved['lines']} "
                    f"字节={saved['bytes']} 耗时={spent}ms "
                    f"JSON={json_key} 旧fpgeom={old_key or '-'} 新fpgeom={saved['key']}"
                )
            except Exception as exc:
                spent = int((time.perf_counter() - t0) * 1000)
                reason = f"{type(exc).__name__}: {exc}"
                failed.append(f"id={row_id} 项目={code} 位号={ref} 原因={reason}")
                _say(
                    f"{head} 结果=失败 耗时={spent}ms 原因={reason} "
                    f"JSON={json_key} 旧fpgeom={old_key or '-'}"
                )
            if done % 20 == 0:
                conn.commit()
                _say(
                    f"[重写fpgeom] 进度 {done}/{total} 成功={rewritten} "
                    f"跳过={len(skipped)} 失败={len(failed)}"
                )
        conn.commit()
    finally:
        conn.close()

    elapsed = int((time.perf_counter() - started) * 1000)
    _say(
        f"[重写fpgeom] 结束 总数={done} 成功={rewritten} "
        f"跳过={len(skipped)} 失败={len(failed)} 耗时={elapsed}ms"
    )
    if skipped:
        _say("[重写fpgeom] 跳过明细")
        for index, item in enumerate(skipped, start=1):
            _say(f"[重写fpgeom] 跳过 {index}. {item}")
    if failed:
        _say("[重写fpgeom] 失败明细")
        for index, item in enumerate(failed, start=1):
            _say(f"[重写fpgeom] 失败 {index}. {item}")


if __name__ == "__main__":
    main()
