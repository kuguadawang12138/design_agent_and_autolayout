# -*- coding: utf-8 -*-
"""
按项目编码重跑 OA 接口②（项目明细）+ 接口③（下载/解压 BRD），并写入数据库与 data/brd/。

多个项目编码用英文 & 分隔（读取时也按 & 拆分）。

用法（在 ingest_service 目录）:
  python run_oa_retry_projects.py --projects 1165551
  python run_oa_retry_projects.py --projects "PRJ001&PRJ002&PRJ003"
"""

from __future__ import annotations

import argparse
import json
import sys

from oa_sync_jobs import parse_project_codes_arg, retry_oa_projects
from settings import bootstrap, find_config_path, load_deploy_env


def main() -> None:
    parser = argparse.ArgumentParser(description="重试失败项目的 OA 接口②③")
    parser.add_argument(
        "--projects",
        "-p",
        required=True,
        help="项目编码；多个用 & 分隔，例如 1165551 或 PRJ001&PRJ002",
    )
    args = parser.parse_args()

    bootstrap()
    codes = parse_project_codes_arg(args.projects)
    if not codes:
        print("错误：未解析到有效项目编码（多个请用 & 分隔）", file=sys.stderr)
        sys.exit(2)

    env = load_deploy_env()
    cfg_path = find_config_path()
    print(f"环境={env} 配置={cfg_path} 项目数={len(codes)} 列表={' & '.join(codes)}")

    result = retry_oa_projects(args.projects)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    # 有失败则非 0；仅「下载参数为空」跳过时可视为部分成功，仍返回 1 便于脚本感知
    sys.exit(0 if result.fail_count == 0 else 1)


if __name__ == "__main__":
    main()
