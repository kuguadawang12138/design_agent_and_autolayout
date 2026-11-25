# -*- coding: utf-8 -*-
"""
二开入库/失败上报联调脚本（单文件，在 ingest_service 目录运行）。

示例:
  python test_erkai_ingest.py --base http://127.0.0.1:8000
  python test_erkai_ingest.py --base http://192.168.x.x:8000 --project TEST_FAIL_001
"""

from __future__ import annotations

import argparse
import json
import sys

import httpx

from footprint_db import connect
from settings import bootstrap


def _print_failures(conn, project_code: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT project_code, refdes, error_code, detail, updated_at
            FROM ingest_failure WHERE project_code = %s
            """,
            (project_code,),
        )
        rows = cur.fetchall()
    if not rows:
        print(f"[数据库] ingest_failure 中无 project_code={project_code} 的记录")
        return
    print(f"[数据库] ingest_failure 共 {len(rows)} 条：")
    for row in rows:
        print(f"  refdes={row[1]!r} code={row[2]} detail={row[3][:200]!r} 更新={row[4]}")


def main() -> int:
    parser = argparse.ArgumentParser(description="测试二开入库 400 与 ingest_failure 写入")
    parser.add_argument("--base", default="http://127.0.0.1:8000", help="入库服务根地址")
    parser.add_argument("--project", default="TEST_ERKAI_FAIL", help="测试用 project_code")
    args = parser.parse_args()

    bootstrap()
    base = args.base.rstrip("/")
    code = args.project.strip()
    client = httpx.Client(timeout=60.0)

    print("=== 1. 二开失败上报 POST /api/v1/ingest/failure ===")
    r1 = client.post(
        f"{base}/api/v1/ingest/failure",
        data={
            "project_code": code,
            "message": "脚本模拟：BRD 解析失败",
            "refdes": "",
        },
    )
    print(f"HTTP {r1.status_code} {r1.text[:300]}")

    print("\n=== 2. 二开入库 POST /api/v1/ingest（故意传非法 JSON，应 HTTP 400 且写 ingest_failure）===")
    bad_json = b"{ this is not valid json"
    r2 = client.post(
        f"{base}/api/v1/ingest",
        data={"project_code": code},
        files=[("json_files", ("bad.json", bad_json, "application/json"))],
    )
    print(f"HTTP {r2.status_code}")
    try:
        print(json.dumps(r2.json(), ensure_ascii=False, indent=2))
    except Exception:
        print(r2.text[:500])

    print("\n=== 3. 查询 ingest_failure ===")
    conn = connect()
    try:
        _print_failures(conn, code)
    finally:
        conn.close()

    print(
        "\n说明：若步骤 2 为 400 但库中无记录，通常是未传表单 project_code 且 JSON 内也无法解析 project_code。"
        "正式联调请始终在表单传 project_code。"
    )
    return 0 if r2.status_code == 400 else 1


if __name__ == "__main__":
    sys.exit(main())
