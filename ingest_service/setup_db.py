# -*- coding: utf-8 -*-
"""创建 footprint_sim 数据库并建表（MySQL）。"""

from __future__ import annotations

import argparse

from settings import bootstrap, get_mysql, resolve_schema_path


def ensure_database() -> None:
    import pymysql

    mysql = get_mysql()
    dbname = str(mysql.get("database", "footprint_sim"))
    conn = pymysql.connect(
        host=str(mysql.get("host", "localhost")),
        port=int(mysql.get("port", 3306)),
        user=str(mysql.get("user", "root")),
        password=str(mysql.get("password", "root")),
        charset="utf8mb4",
        autocommit=True,
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"CREATE DATABASE IF NOT EXISTS `{dbname}` "
                "DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
            print(f"数据库就绪: {dbname}")
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="初始化 MySQL 封装库（建表+迁移仅此入口，服务运行时不再自动 init_schema）"
    )
    parser.add_argument("--config", default=None, help="YAML 配置路径")
    args = parser.parse_args()

    info = bootstrap(args.config)
    print(f"配置: {info['config_path']}")
    print(f"schema: {info['schema_path']}")

    from footprint_db import connect, init_schema

    ensure_database()
    with connect() as conn:
        init_schema(conn, schema_path=resolve_schema_path())
    print(
        "表结构就绪: project_profile, footprint_library, project_pdf, ingest_failure, "
        "oa_sync_progress, process, pending_project, pending_iou, pending_json, "
        "pending_json_block, footprint_raster"
    )


if __name__ == "__main__":
    main()
