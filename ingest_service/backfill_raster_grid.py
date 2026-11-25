# -*- coding: utf-8 -*-
"""把还没有栅格的器件，按已保存的 fpgeom 写入 footprint_raster。

只补 footprint_raster 里没有的项目编码+位号。已有记录不改。
用法（在 ingest_service 目录，先执行 python setup_db.py）：

    python backfill_raster_grid.py
    python backfill_raster_grid.py --config config/footprint_sim_st.yaml
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="把缺失的器件栅格写入 footprint_raster")
    parser.add_argument("--config", default=None, help="YAML 配置路径")
    args = parser.parse_args()

    from settings import bootstrap

    info = bootstrap(args.config)
    print(f"配置: {info['config_path']}")

    from agent_log import log_line
    from footprint_db import connect, normalize_refdes, pin_count_too_small
    from fpgeom import fpgeom_from_bytes, raster_from_cache, raster_to_bytes
    from minio_store import resolve_stored_bytes

    filled = 0
    failed: list[str] = []
    done = 0

    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT l.project_code, l.refdes, l.fpgeom, l.pin_count
                FROM footprint_library l
                LEFT JOIN footprint_raster r
                  ON r.project_code = l.project_code
                 AND REPLACE(r.refdes, '@', '') = REPLACE(l.refdes, '@', '')
                WHERE r.id IS NULL
                """
            )
            rows = cur.fetchall()
        log_line("栅格", result=f"【栅格回填】待写入 footprint_raster 的器件={len(rows)}")

        for project_code, refdes, stored_fpgeom, pin_count in rows:
            code = project_code or ""
            ref = normalize_refdes(refdes)
            label = f"{code}/{ref}"
            if pin_count_too_small(pin_count):
                log_line(
                    "栅格",
                    result=(
                        f"【脚数不超过6不入库】project_code={code} refdes={ref} "
                        f"脚数={pin_count} 表=footprint_raster"
                    ),
                )
                continue
            try:
                raw = resolve_stored_bytes(stored_fpgeom)
                cache = fpgeom_from_bytes(raw)
                grid = raster_from_cache(cache)
                blob = raster_to_bytes(grid)
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO footprint_raster
                            (project_code, refdes, raster_size, raster_grid)
                        VALUES (%s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE
                            raster_size=VALUES(raster_size),
                            raster_grid=VALUES(raster_grid),
                            updated_at=CURRENT_TIMESTAMP
                        """,
                        (code, ref, int(grid.shape[0]), blob),
                    )
                    filled += 1
                log_line(
                    "栅格",
                    result=(
                        f"【栅格回填】project_code={code} refdes={ref} "
                        f"边长={int(grid.shape[0])} 字节={len(blob)} 表=footprint_raster"
                    ),
                )
            except Exception as exc:
                failed.append(f"{label}: {type(exc).__name__}")
                log_line(
                    "栅格",
                    result=f"【栅格回填失败】project_code={code} refdes={ref} {type(exc).__name__}",
                    level="warning",
                )
            done += 1
            if done % 50 == 0:
                conn.commit()
        conn.commit()
    finally:
        conn.close()

    log_line("栅格", result=f"【栅格回填结束】写入={filled} 失败={len(failed)}")
    print(f"栅格为空已写入 {filled} 条")
    print(f"栅格仍为空 {len(failed)} 条")
    for item in failed:
        print(f"  {item}")


if __name__ == "__main__":
    main()
