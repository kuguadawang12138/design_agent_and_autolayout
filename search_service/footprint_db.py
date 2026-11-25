# -*- coding: utf-8 -*-
"""封装库 MySQL 读写（fpgeom + 元数据）。"""

from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fpgeom import (
    FPGEOM_VERSION,
    FpGeomCache,
    best_iou_rotations,
    best_packed_raster_iou,
    build_fpgeom_from_footprint,
    cached_query_geom,
    footprint_copper_area_mils2,
    fpgeom_from_bytes,
    fpgeom_to_bytes,
    pack_iou_shape,
    shape_for_iou,
    raster_from_cache,
    raster_packed_for_query,
    raster_query_packs,
    raster_to_bytes,
)
from footprint_json import load_footprint_json, load_footprint_json_from_bytes

DEFAULT_DSN = "mysql://root:root@localhost:3306/footprint_sim"


@dataclass
class FootprintRow:
    id: int
    name: str
    pin_count: int
    fpgeom: bytes
    project_code: str
    refdes: str
    bbox_min_x: float
    bbox_min_y: float
    bbox_max_x: float
    bbox_max_y: float


def get_dsn() -> str:
    try:
        from settings import get_dsn as _yaml_dsn

        return _yaml_dsn()
    except Exception:
        return DEFAULT_DSN


def connect(dsn: str | None = None):
    import pymysql

    from settings import get_mysql

    mysql = get_mysql()
    return pymysql.connect(
        host=str(mysql.get("host", "localhost")),
        port=int(mysql.get("port", 3306)),
        user=str(mysql.get("user", "root")),
        password=str(mysql.get("password", "root")),
        database=str(mysql.get("database", "footprint_sim")),
        charset="utf8mb4",
        autocommit=False,
    )


def _migrate_object_path_columns(conn) -> None:
    """与入库库一致：三个大字段改为 VARCHAR 路径。"""
    specs = (
        ("footprint_library", "fpgeom", "VARCHAR(512) NOT NULL COMMENT 'MinIO 路径：年/月/日/项目_位号.fpgeom'"),
        ("footprint_library", "source_json", "VARCHAR(512) NULL COMMENT 'MinIO 路径：年/月/日/项目_位号.json'"),
        ("project_pdf", "file_content", "VARCHAR(512) NOT NULL COMMENT 'MinIO 路径：年/月/日/项目_top或bottom.pdf'"),
    )
    from agent_log import log_line

    with conn.cursor() as cur:
        for table, col, decl in specs:
            cur.execute(f"SHOW TABLES LIKE '{table}'")
            if not cur.fetchone():
                continue
            cur.execute(f"SHOW COLUMNS FROM `{table}` LIKE %s", (col,))
            row = cur.fetchone()
            if not row:
                continue
            col_type = str(row[1]).lower()
            if col_type.startswith("varchar"):
                continue
            cur.execute(f"ALTER TABLE `{table}` MODIFY COLUMN `{col}` {decl}")
            log_line("数据库", result=f"【字段改类型】{table}.{col} -> VARCHAR(512)")
    conn.commit()


def init_schema(conn, schema_path: Path | None = None) -> None:
    """仅由 setup_db.py / 运维脚本调用；运行中的检索/入库服务不要调用（避免并发 DDL 与旧库误迁移）。"""
    if schema_path is None:
        from settings import resolve_schema_path

        schema_path = resolve_schema_path()
    schema_path = schema_path.resolve()
    if not schema_path.is_file():
        raise FileNotFoundError(f"找不到 schema.sql: {schema_path}")
    sql = schema_path.read_text(encoding="utf-8")
    with conn.cursor() as cur:
        for stmt in _split_sql_statements(sql):
            cur.execute(stmt)
    conn.commit()
    _migrate_footprint_library(conn)
    _migrate_object_path_columns(conn)
    _migrate_oa_sync_progress_table(conn)
    _migrate_process_table(conn)


PROJECT_PROFILE_FIELDS = (
    "customer_code",
    "designer_name",
    "designer_seniority",
    "finish_time",
)


def normalize_refdes(raw) -> str:
    """位号不带 @。入库和检索都先去掉再比较。"""
    return str(raw or "").replace("@", "").strip().upper()


_REFDES_EQ = "REPLACE(refdes, '@', '') = %s"


def _take_created_at(cur, table: str, where_sql: str, params: tuple):
    """读出该唯一键第一次的创建时间，并删掉旧行。没有旧行则返回 None。"""
    cur.execute(
        f"SELECT created_at FROM {table} WHERE {where_sql} ORDER BY created_at ASC LIMIT 1",
        params,
    )
    row = cur.fetchone()
    if not row:
        return None
    created = row[0]
    cur.execute(f"DELETE FROM {table} WHERE {where_sql}", params)
    return created


def _extract_project_profile(meta: dict | None, info: dict | None = None) -> dict[str, Any]:
    """从 JSON/OA 元数据提取项目级四字段（不写 footprint_library）。空值不覆盖已有完成时间等字段。"""
    out: dict[str, Any] = {}
    for src in (info or {}, meta or {}):
        if not isinstance(src, dict):
            continue
        for key in PROJECT_PROFILE_FIELDS:
            val = src.get(key)
            if val not in (None, ""):
                out[key] = val
    return out


def upsert_project_profile(conn, project_code: str, info: dict | None) -> None:
    """按 project_code 唯一键 UPSERT。"""
    code = str(project_code or "").strip()
    if not code:
        return
    payload = _extract_project_profile(info)
    if not payload:
        return
    cols = list(payload.keys())
    placeholders = ", ".join(["%s"] * len(cols))
    updates = ", ".join(f"{c}=VALUES({c})" for c in cols)
    sql = f"""
        INSERT INTO project_profile (project_code, {', '.join(cols)})
        VALUES (%s, {placeholders})
        ON DUPLICATE KEY UPDATE {updates}, updated_at=CURRENT_TIMESTAMP
    """
    with conn.cursor() as cur:
        cur.execute(sql, [code] + [payload[c] for c in cols])
    from mysql_diag import commit_or_log

    commit_or_log(
        conn,
        operation="upsert_project_profile",
        detail=f"project_code={code}",
        sql_hint="project_profile",
    )


def fetch_project_profile(conn, project_code: str) -> dict | None:
    code = str(project_code or "").strip()
    if not code:
        return None
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT customer_code, designer_name, designer_seniority, finish_time
            FROM project_profile WHERE project_code = %s LIMIT 1
            """,
            (code,),
        )
        row = cur.fetchone()
    if not row:
        return None
    return dict(zip(PROJECT_PROFILE_FIELDS, row))


def resolve_object_finish_time(conn, project_code: str, explicit: Any = None) -> Any:
    """MinIO 年/月/日目录：先按项目编码取 project_profile.finish_time，没有再用传入值。都没有则返回 None，由对象键落到当天。"""
    code = str(project_code or "").strip()
    if code:
        prof = fetch_project_profile(conn, code) or {}
        finish = prof.get("finish_time")
        if finish not in (None, ""):
            return finish
    if explicit not in (None, ""):
        return explicit
    return None


def _merge_project_profile(conn, row: dict) -> dict:
    """查询结果合并项目属性（接口字段与旧版 footprint 行一致）。"""
    if not row:
        return row
    code = row.get("project_code")
    prof = fetch_project_profile(conn, code) if code else None
    if prof:
        row.update(prof)
    return row


def _migrate_footprint_library(conn) -> None:
    """旧库增量迁移。"""
    with conn.cursor() as cur:
        cur.execute("SHOW TABLES LIKE 'footprint_library'")
        if cur.fetchone():
            cur.execute("SHOW COLUMNS FROM footprint_library")
            existing = {row[0] for row in cur.fetchall()}
            adds = [
                ("layers", "INT NULL COMMENT '板子层数'"),
                ("devicetype", "VARCHAR(256) NULL"),
                ("packagesymbol", "VARCHAR(256) NULL"),
                ("value", "VARCHAR(256) NULL"),
                ("pcb_x", "DOUBLE NULL"),
                ("pcb_y", "DOUBLE NULL"),
                ("top_side", "VARCHAR(16) NULL"),
                ("customer_code", "VARCHAR(64) NULL"),
                ("designer_name", "VARCHAR(128) NULL"),
                ("designer_seniority", "DOUBLE NULL"),
                ("finish_time", "DATETIME NULL"),
                ("pdf_refdes_x", "DOUBLE NULL COMMENT 'PDF丝印中心X'"),
                ("pdf_refdes_y", "DOUBLE NULL COMMENT 'PDF丝印中心Y'"),
                (
                    "copper_area",
                    "DOUBLE NULL COMMENT '铜箔真实面积（mils²，形状并集，已换算单位）'",
                ),
                (
                    "raster_grid",
                    "MEDIUMBLOB NULL COMMENT '1024x1024 的 0 度栅格，按行 packbits，约 128KB'",
                ),
            ]
            for col, decl in adds:
                if col not in existing:
                    cur.execute(f"ALTER TABLE footprint_library ADD COLUMN {col} {decl}")
            # MySQL 5.7：JSON/TEXT 列统一为 MEDIUMTEXT
            for col, comment in (
                ("source_json", "原始 JSON 归档（上传文件 UTF-8 原文）"),
                ("warnings", "解析告警（JSON 数组字符串）"),
            ):
                if col not in existing:
                    continue
                cur.execute("SHOW COLUMNS FROM footprint_library LIKE %s", (col,))
                col_row = cur.fetchone()
                if not col_row:
                    continue
                col_type = str(col_row[1]).lower()
                if "json" in col_type or col_type == "text":
                    cur.execute(
                        f"ALTER TABLE footprint_library MODIFY COLUMN {col} MEDIUMTEXT NULL COMMENT %s",
                        (comment,),
                    )
            for drop_col in ("highlight_pdf", "highlight_preview"):
                if drop_col in existing:
                    cur.execute(f"ALTER TABLE footprint_library DROP COLUMN {drop_col}")
            if "highlight_pdf_name" in existing:
                cur.execute("ALTER TABLE footprint_library DROP COLUMN highlight_pdf_name")
            # 已撤掉向量库：旧列 text_snapshot 不再需要
            if "text_snapshot" in existing:
                cur.execute("ALTER TABLE footprint_library DROP COLUMN text_snapshot")
            # source_json 已是 MinIO 路径，文件名在路径里，不再单独存 source_file
            if "source_file" in existing:
                cur.execute("ALTER TABLE footprint_library DROP COLUMN source_file")
            if "project_name" in existing:
                cur.execute(
                    "UPDATE footprint_library SET name=COALESCE(name, project_name) WHERE project_name IS NOT NULL"
                )
        cur.execute("SHOW TABLES LIKE 'project_pdf'")
        if cur.fetchone():
            cur.execute("SHOW COLUMNS FROM project_pdf")
            pdf_cols = {row[0] for row in cur.fetchall()}
            if "refdes" in pdf_cols or "file_hash" in pdf_cols or "highlight_region" in pdf_cols:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS project_pdf_new (
                        id BIGINT AUTO_INCREMENT PRIMARY KEY,
                        project_code VARCHAR(128) NOT NULL,
                        top_side VARCHAR(16) NOT NULL DEFAULT 'yes',
                        file_name VARCHAR(256) NOT NULL,
                        file_content LONGBLOB NOT NULL,
                        pdf_scale DOUBLE NOT NULL DEFAULT 0 COMMENT 'PDF↔板坐标比例（入库必填）',
                        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                        UNIQUE KEY uk_pdf_project_side (project_code, top_side)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                    """
                )
                cur.execute(
                    """
                    INSERT INTO project_pdf_new (project_code, top_side, file_name, file_content, pdf_scale, created_at)
                    SELECT project_code, top_side, MAX(file_name), MAX(file_content), MAX(pdf_scale), MIN(created_at)
                    FROM project_pdf GROUP BY project_code, top_side
                    ON DUPLICATE KEY UPDATE file_content=VALUES(file_content)
                    """
                )
                cur.execute("DROP TABLE project_pdf")
                cur.execute("RENAME TABLE project_pdf_new TO project_pdf")
            elif "pdf_scale" not in pdf_cols:
                cur.execute(
                    "ALTER TABLE project_pdf ADD COLUMN pdf_scale DOUBLE NOT NULL DEFAULT 0 "
                    "COMMENT 'PDF↔板坐标比例（入库必填）'"
                )
            else:
                cur.execute("UPDATE project_pdf SET pdf_scale=0 WHERE pdf_scale IS NULL")
                try:
                    cur.execute(
                        "ALTER TABLE project_pdf MODIFY COLUMN pdf_scale DOUBLE NOT NULL "
                        "COMMENT 'PDF↔板坐标比例（入库必填）'"
                    )
                except Exception:
                    pass
        cur.execute("SHOW TABLES LIKE 'ingest_failure'")
        if cur.fetchone():
            cur.execute("SHOW COLUMNS FROM ingest_failure")
            fail_cols = {row[0] for row in cur.fetchall()}
            if "detail" not in fail_cols and "reason" in fail_cols:
                cur.execute("ALTER TABLE ingest_failure CHANGE COLUMN reason detail TEXT NOT NULL")
            for drop_col in ("batch_no", "session_id", "raw_payload", "file_name"):
                if drop_col in fail_cols:
                    cur.execute(f"ALTER TABLE ingest_failure DROP COLUMN {drop_col}")
            if "refdes" in fail_cols:
                cur.execute("UPDATE ingest_failure SET refdes='' WHERE refdes IS NULL")
                try:
                    cur.execute("ALTER TABLE ingest_failure MODIFY refdes VARCHAR(64) NOT NULL DEFAULT ''")
                except Exception:
                    pass
        _migrate_top_side_to_boolean(conn)
        _migrate_project_profile_table(conn)
    conn.commit()


def _migrate_project_profile_table(conn) -> None:
    """project_profile 表 + 从 footprint_library 迁出四字段。"""
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS project_profile (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                project_code VARCHAR(128) NOT NULL,
                customer_code VARCHAR(64) NULL,
                designer_name VARCHAR(128) NULL,
                designer_seniority DOUBLE NULL,
                finish_time DATETIME NULL,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                UNIQUE KEY uk_project_profile (project_code)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
        )
        cur.execute("SHOW TABLES LIKE 'footprint_library'")
        if not cur.fetchone():
            return
        cur.execute("SHOW COLUMNS FROM footprint_library")
        fp_cols = {row[0] for row in cur.fetchall()}
        legacy = [c for c in PROJECT_PROFILE_FIELDS if c in fp_cols]
        if legacy:
            cur.execute(
                """
                INSERT INTO project_profile (
                    project_code, customer_code, designer_name, designer_seniority, finish_time
                )
                SELECT project_code,
                       MAX(customer_code), MAX(designer_name),
                       MAX(designer_seniority), MAX(finish_time)
                FROM footprint_library
                GROUP BY project_code
                ON DUPLICATE KEY UPDATE
                    customer_code=COALESCE(VALUES(customer_code), project_profile.customer_code),
                    designer_name=COALESCE(VALUES(designer_name), project_profile.designer_name),
                    designer_seniority=COALESCE(VALUES(designer_seniority), project_profile.designer_seniority),
                    finish_time=COALESCE(VALUES(finish_time), project_profile.finish_time)
                """
            )
            for col in legacy:
                cur.execute(f"ALTER TABLE footprint_library DROP COLUMN {col}")


def _migrate_top_side_to_boolean(conn) -> None:
    """VARCHAR yes/no → TINYINT(1) 布尔面。"""
    from top_side_util import top_side_from_db, top_side_to_db

    def _col_type(table: str, col: str) -> str:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DATA_TYPE FROM information_schema.COLUMNS
                WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s AND COLUMN_NAME = %s
                """,
                (table, col),
            )
            row = cur.fetchone()
        return (row[0] or "").lower() if row else ""

    for table in ("footprint_library", "project_pdf"):
        with conn.cursor() as cur:
            cur.execute(f"SHOW TABLES LIKE %s", (table,))
            if not cur.fetchone():
                continue
        if _col_type(table, "top_side") not in ("varchar", "char", "text"):
            continue
        tmp = "top_side_bool_mig"
        with conn.cursor() as cur:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {tmp} TINYINT(1) NULL")
            cur.execute(f"SELECT project_code, refdes, top_side FROM {table}" if table == "footprint_library" else f"SELECT id, top_side FROM {table}")
            rows = cur.fetchall()
        updates = []
        if table == "footprint_library":
            for code, ref, ts in rows:
                updates.append((top_side_to_db(top_side_from_db(ts)), code, ref))
            with conn.cursor() as cur:
                for val, code, ref in updates:
                    cur.execute(
                        f"UPDATE {table} SET {tmp}=%s WHERE project_code=%s AND refdes=%s",
                        (val, code, ref),
                    )
        else:
            with conn.cursor() as cur:
                for rid, ts in rows:
                    cur.execute(
                        f"UPDATE {table} SET {tmp}=%s WHERE id=%s",
                        (top_side_to_db(top_side_from_db(ts)), rid),
                    )
        with conn.cursor() as cur:
            cur.execute(f"ALTER TABLE {table} DROP COLUMN top_side")
            cur.execute(f"UPDATE {table} SET {tmp}=1 WHERE {tmp} IS NULL")
            cur.execute(
                f"ALTER TABLE {table} CHANGE COLUMN {tmp} top_side TINYINT(1) NOT NULL DEFAULT 1 "
                f"COMMENT '1=正面 0=反面'"
            )


def _split_sql_statements(sql: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    for line in sql.splitlines():
        stripped = line.strip()
        if stripped.startswith("--"):
            continue
        buf.append(line)
        if stripped.endswith(";"):
            parts.append("\n".join(buf).rstrip(";\n\r\t "))
            buf = []
    if buf:
        parts.append("\n".join(buf))
    return [p for p in parts if p.strip()]


def _raw_bytes_from_content(content: bytes | str) -> bytes:
    if isinstance(content, bytes):
        return content
    return content.encode("utf-8")


def _source_json_archive_text(raw_bytes: bytes) -> str:
    """入库归档：保留上传 JSON 文件 UTF-8 原文，不做 dumps 重排。"""
    return raw_bytes.decode("utf-8-sig", errors="replace")


def _load_source_json(raw_bytes: bytes) -> dict[str, Any]:
    text = raw_bytes.decode("utf-8-sig")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            from footprint_json import _normalize_design_json_text

            data = json.loads(_normalize_design_json_text(text))
        except json.JSONDecodeError:
            return {"raw": text[:5000]}
    return data if isinstance(data, dict) else {"raw": data}


def _split_model_warnings(warnings: list[str]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """将解析 warnings 拆成字段级 / pin 级，供 API 与日志使用。"""
    field_issues: list[dict[str, str]] = []
    pin_issues: list[dict[str, str]] = []
    for msg in warnings:
        if msg.startswith("pin ") and ": " in msg:
            head, issue = msg.split(": ", 1)
            pin_number = head.removeprefix("pin ").strip() or "?"
            pin_issues.append({"pin_number": pin_number, "issue": issue})
        else:
            field_issues.append({"field": "-", "issue": msg})
    return field_issues, pin_issues


def _format_log_issues(
    field_issues: list[dict[str, str]],
    pin_issues: list[dict[str, str]],
    *,
    max_chars: int = 4000,
) -> str:
    """拼成单行日志；过长则截断并标注总数。"""
    parts: list[str] = []
    for item in field_issues:
        parts.append(item["issue"])
    for item in pin_issues:
        parts.append(f"pin {item['pin_number']}:{item['issue']}")
    if not parts:
        return "无"
    text = " | ".join(parts)
    total = len(parts)
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}...（共{total}条）"


def _build_library_message_success(action: str, row: dict[str, Any]) -> str:
    op = "新增入库" if action == "inserted" else "更新入库"
    return (
        f"入库成功（{op}），项目 {row['project_code']} 位号 {row['refdes']}，"
        f"封装 {row['name']}，共 {row['pin_count']} 个 pin，铜箔已全部正常生成。"
    )


def _build_library_message_failure(
    row: dict[str, Any],
    field_issues: list[dict[str, str]],
    pin_issues: list[dict[str, str]],
) -> str:
    base = (
        f"入库失败，项目 {row.get('project_code', '?')} 位号 {row.get('refdes', '?')}，"
        f"封装 {row.get('name', '?')}，共 {row.get('pin_count', 0)} 个 pin"
    )
    details: list[str] = []
    for item in field_issues:
        details.append(item["issue"])
    for item in pin_issues:
        details.append(f"pin {item['pin_number']}：{item['issue']}")
    if not details:
        return f"{base}，存在未通过校验的问题。"
    detail_text = "；".join(details[:50])
    if len(details) > 50:
        detail_text += f"；等共 {len(details)} 条问题"
    return f"{base}，存在问题：{detail_text}"


def _log_library_failure(
    log,
    *,
    project_code: str,
    refdes: str,
    source_file: str,
    pin_count: int,
    field_issues: list[dict[str, str]],
    pin_issues: list[dict[str, str]],
) -> None:
    issue_text = _format_log_issues(field_issues, pin_issues)
    log.warning(
        "library FAIL project=%s refdes=%s file=%s db=skip pins=%d field_issues=%d pin_issues=%d",
        project_code,
        refdes,
        source_file,
        pin_count,
        len(field_issues),
        len(pin_issues),
    )
    log.warning(
        "library FAIL detail project=%s refdes=%s file=%s %s",
        project_code,
        refdes,
        source_file,
        issue_text,
    )
    for item in field_issues:
        log.warning(
            "library field_issue project=%s refdes=%s file=%s %s",
            project_code,
            refdes,
            source_file,
            item["issue"],
        )
    for item in pin_issues:
        log.warning(
            "library pin_issue project=%s refdes=%s file=%s pin=%s %s",
            project_code,
            refdes,
            source_file,
            item["pin_number"],
            item["issue"],
        )


def build_row_from_content(
    content: bytes | str,
    *,
    source_file: str = "upload.json",
    project_info: dict | None = None,
) -> dict[str, Any]:
    """从上传 JSON 内容构建入库行（解析 + fpgeom + 业务元数据）。"""
    raw_bytes = _raw_bytes_from_content(content)
    model = load_footprint_json_from_bytes(raw_bytes, source_name=source_file)
    cache = build_fpgeom_from_footprint(model, source_file=source_file)
    min_x, min_y, max_x, max_y = model.bounds()
    source_json = _load_source_json(raw_bytes)
    info = project_info or {}
    from json_fields import extract_meta

    meta = extract_meta(source_json if isinstance(source_json, dict) else {}, source_file)
    meta.update(
        {
            "project_code": model.project_code or meta.get("project_code"),
            "board_name": model.project_name or meta.get("board_name") or info.get("board_name"),
            "refdes": model.refdes or meta.get("refdes"),
            "layers": model.layers if model.layers is not None else meta.get("layers") or info.get("layers"),
            "device_type": model.devicetype or meta.get("device_type"),
            "package_symbol": model.packagesymbol or meta.get("package_symbol"),
            "value": model.value or meta.get("value"),
            "pcb_x": model.refdes_x if model.refdes_x is not None else meta.get("pcb_x"),
            "pcb_y": model.refdes_y if model.refdes_y is not None else meta.get("pcb_y"),
            "top_side": bool(model.top_side),
        }
    )
    board_name = meta.get("board_name") or None
    project_profile = _extract_project_profile(meta, info)
    return {
        "name": board_name,
        "pin_count": model.pin_count,
        "copper_area": footprint_copper_area_mils2(model),
        "units": model.units,
        "bbox_min_x": float(min_x),
        "bbox_min_y": float(min_y),
        "bbox_max_x": float(max_x),
        "bbox_max_y": float(max_y),
        "fpgeom_version": FPGEOM_VERSION,
        "fpgeom": fpgeom_to_bytes(cache),
        "project_code": meta["project_code"] or model.project_code,
        "refdes": meta["refdes"] or model.refdes,
        "layers": meta.get("layers"),
        "devicetype": meta.get("device_type"),
        "packagesymbol": meta.get("package_symbol"),
        "value": meta.get("value"),
        "pcb_x": meta.get("pcb_x"),
        "pcb_y": meta.get("pcb_y"),
        "top_side": meta.get("top_side") if meta.get("top_side") is not None else True,
        "source_json": _source_json_archive_text(raw_bytes),
        "warnings": json.dumps(model.warnings, ensure_ascii=False),
        "geom_islands": len(cache.islands),
        "raster_grid": raster_to_bytes(_raster := raster_from_cache(cache)),
        "raster_size": int(_raster.shape[0]),
        "_meta": meta,
        "_project_profile": project_profile,
    }


def build_row_from_json(json_path: Path) -> dict[str, Any]:
    model = load_footprint_json(json_path)
    cache = build_fpgeom_from_footprint(model, source_file=json_path.name)
    min_x, min_y, max_x, max_y = model.bounds()
    raw_json = json_path.read_text(encoding="utf-8-sig")
    try:
        source_json = json.loads(raw_json)
    except json.JSONDecodeError:
        source_json = {"raw": raw_json[:5000]}
    return {
        "name": model.name,
        "pin_count": model.pin_count,
        "copper_area": footprint_copper_area_mils2(model),
        "units": model.units,
        "bbox_min_x": float(min_x),
        "bbox_min_y": float(min_y),
        "bbox_max_x": float(max_x),
        "bbox_max_y": float(max_y),
        "fpgeom_version": FPGEOM_VERSION,
        "fpgeom": fpgeom_to_bytes(cache),
        "project_code": model.project_code,
        "refdes": model.refdes,
        "source_json": raw_json,
        "warnings": json.dumps(model.warnings, ensure_ascii=False),
        "geom_islands": len(cache.islands),
        "raster_grid": raster_to_bytes(_raster := raster_from_cache(cache)),
        "raster_size": int(_raster.shape[0]),
    }


def _pin_count_too_small(pin_count) -> bool:
    """脚数已知且不超过 6 不入库。未知脚数不拦截。"""
    if pin_count is None or pin_count == "":
        return False
    try:
        return int(pin_count) <= 6
    except (TypeError, ValueError):
        return False


def upsert_footprint(conn, row: dict[str, Any]) -> str:
    """按 project_code+refdes 唯一键：已存在则先删后插，否则直接插入。MinIO 对象同样先删再传。"""
    from minio_store import object_key_of, save_object

    code = str(row.get("project_code") or "").strip()
    ref = normalize_refdes(row.get("refdes"))
    row["refdes"] = ref
    if _pin_count_too_small(row.get("pin_count")):
        from agent_log import log_line

        log_line(
            "器件入库",
            result=f"【脚数不超过6不入库】project_code={code} refdes={ref} 脚数={row.get('pin_count')}",
        )
        return "skipped"
    old_fp = old_json = None
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT fpgeom, source_json FROM footprint_library WHERE project_code=%s AND {_REFDES_EQ} LIMIT 1",
            (code, ref),
        )
        found = cur.fetchone()
        if found:
            old_fp, old_json = object_key_of(found[0]), object_key_of(found[1])
    finish = resolve_object_finish_time(conn, code, row.get("_finish_time"))
    fp = row.get("fpgeom")
    if object_key_of(fp) is None:
        raw = bytes(fp) if isinstance(fp, (bytes, bytearray)) else str(fp or "").encode("utf-8")
        row["fpgeom"] = save_object(
            raw,
            finish_time=finish,
            filename=f"{code}_{ref}.fpgeom",
            content_type="application/octet-stream",
            old_key=old_fp,
        )
    else:
        row["fpgeom"] = object_key_of(fp)
    sj = row.get("source_json")
    if object_key_of(sj) is None:
        raw_text = sj if isinstance(sj, str) else bytes(sj or b"").decode("utf-8", errors="replace")
        row["source_json"] = save_object(
            raw_text.encode("utf-8"),
            finish_time=finish,
            filename=f"{code}_{ref}.json",
            content_type="application/json; charset=utf-8",
            old_key=old_json,
        )
    else:
        row["source_json"] = object_key_of(sj)
    from mysql_diag import db_transaction

    with db_transaction(
        conn,
        operation="upsert_footprint",
        detail=f"project_code={code} refdes={ref}",
        sql_hint="footprint_library",
    ):
        with conn.cursor() as cur:
            created = _take_created_at(
                cur,
                "footprint_library",
                f"project_code = %s AND {_REFDES_EQ}",
                (code, ref),
            )
            cols = [
                "name", "pin_count", "copper_area", "units",
                "bbox_min_x", "bbox_min_y", "bbox_max_x", "bbox_max_y",
                "fpgeom_version", "fpgeom",
                "project_code", "refdes",
                "layers", "devicetype", "packagesymbol", "value",
                "pcb_x", "pcb_y", "top_side",
                "source_json", "warnings", "geom_islands",
            ]
            if created is not None:
                row["created_at"] = created
                cols.append("created_at")
            names = ", ".join(cols)
            holders = ", ".join(f"%({c})s" for c in cols)
            cur.execute(f"INSERT INTO footprint_library ({names}) VALUES ({holders})", row)
    _save_footprint_raster(conn, code, ref, row.get("raster_size"), row.get("raster_grid"))
    return "updated" if created is not None else "inserted"


def _save_footprint_raster(conn, project_code: str, refdes: str, raster_size, raster_grid) -> None:
    """栅格按项目编码+位号写入 footprint_raster。失败不影响器件行。"""
    if not raster_grid or not raster_size:
        return
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO footprint_raster (project_code, refdes, raster_size, raster_grid)
                VALUES (%s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    raster_size=VALUES(raster_size),
                    raster_grid=VALUES(raster_grid),
                    updated_at=CURRENT_TIMESTAMP
                """,
                (project_code, refdes, int(raster_size), raster_grid),
            )
        conn.commit()
        from agent_log import log_line

        log_line(
            "栅格",
            result=(
                f"【栅格写入】project_code={project_code} refdes={refdes} "
                f"边长={int(raster_size)} 字节={len(raster_grid)} 表=footprint_raster"
            ),
        )
    except Exception as exc:
        try:
            conn.rollback()
        except Exception:
            pass
        from agent_log import log_line

        log_line(
            "栅格",
            result=f"【栅格未写入】project_code={project_code} refdes={refdes} {type(exc).__name__}",
            level="warning",
        )


def upsert_footprint_from_content(
    conn,
    content: bytes | str,
    *,
    source_file: str = "upload.json",
    project_info: dict | None = None,
) -> dict[str, Any]:
    """
    上传 JSON 校验并入库。
    仅当全部 pin 无告警时写入数据库；否则拒绝入库。
    返回 {code, success, message}。
    """
    import logging

    log = logging.getLogger("pkg_sim")
    row = build_row_from_content(content, source_file=source_file, project_info=project_info)
    warnings: list[str] = json.loads(row["warnings"])
    field_issues, pin_issues = _split_model_warnings(warnings)
    issue_count = len(field_issues) + len(pin_issues)

    log.info(
        "library parse project=%s refdes=%s file=%s pins=%d islands=%d issue_count=%d",
        row["project_code"],
        row["refdes"],
        source_file,
        row["pin_count"],
        row["geom_islands"],
        issue_count,
    )

    if _pin_count_too_small(row.get("pin_count")):
        return {
            "code": 1,
            "success": True,
            "skipped": True,
            "pin_count": row.get("pin_count"),
            "message": "脚数不超过 6，未入库",
        }

    if issue_count > 0:
        _log_library_failure(
            log,
            project_code=row["project_code"],
            refdes=row["refdes"],
            source_file=source_file,
            pin_count=row["pin_count"],
            field_issues=field_issues,
            pin_issues=pin_issues,
        )
        return {
            "code": 0,
            "success": False,
            "message": _build_library_message_failure(row, field_issues, pin_issues),
        }

    profile = row.pop("_project_profile", None) or {}
    payload = {k: v for k, v in row.items() if not str(k).startswith("_")}
    payload["_finish_time"] = profile.get("finish_time")
    action = upsert_footprint(conn, payload)
    if action == "skipped":
        return {
            "code": 1,
            "success": True,
            "skipped": True,
            "pin_count": row.get("pin_count"),
            "message": "脚数不超过 6，未入库",
        }
    if profile:
        upsert_project_profile(conn, row["project_code"], profile)
    message = _build_library_message_success(action, row)
    log.info(
        "library OK project=%s refdes=%s file=%s action=%s pins=%d islands=%d db=committed",
        row["project_code"],
        row["refdes"],
        source_file,
        action,
        row["pin_count"],
        row["geom_islands"],
    )
    return {
        "code": 1,
        "success": True,
        "message": message,
    }


def public_similar_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """对外 results：项目编码 + 位号 + 封装 bbox + IoU。"""
    return [
        {
            "project_code": row["project_code"],
            "refdes": row["refdes"],
            "minx": row["bbox_min_x"],
            "miny": row["bbox_min_y"],
            "maxx": row["bbox_max_x"],
            "maxy": row["bbox_max_y"],
            "iou": row.get("iou"),
        }
        for row in results
    ]


def load_precomputed_similar(
    conn,
    project_code: str,
    refdes: str,
    top_k: int,
    min_iou: float | None = None,
) -> list[dict[str, Any]] | None:
    """pending_iou 里已有该项目+位号则按名次取。低于 min_iou 的不要，条数不超过 top_k。没有记录返回 None。"""
    code = str(project_code or "").strip()
    ref = normalize_refdes(refdes)
    if not code or not ref:
        return None
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM pending_iou WHERE project_code=%s AND refdes=%s LIMIT 1",
            (code, ref),
        )
        if cur.fetchone() is None:
            return None
        sql = """
            SELECT match_project_code, match_refdes, iou,
                   bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y
            FROM pending_iou
            WHERE project_code=%s AND refdes=%s
        """
        params: list[Any] = [code, ref]
        if min_iou is not None:
            sql += " AND iou >= %s"
            params.append(float(min_iou))
        sql += " ORDER BY rank_no ASC"
        if top_k is not None and int(top_k) >= 0:
            sql += " LIMIT %s"
            params.append(int(top_k))
        cur.execute(sql, params)
        raw = cur.fetchall()
    return [
        {
            "project_code": row[0] or "",
            "refdes": row[1] or "",
            "pin_count": 0,
            "bbox_min_x": float(row[3] or 0),
            "bbox_min_y": float(row[4] or 0),
            "bbox_max_x": float(row[5] or 0),
            "bbox_max_y": float(row[6] or 0),
            "iou": round(float(row[2] or 0), 4),
        }
        for row in raw
    ]


_RASTER_PACKS = None


def _init_raster_worker(blobs: list[bytes], height: int, packed_width: int) -> None:
    """每个进程只展开一次 query 的四个方向。"""
    global _RASTER_PACKS
    import numpy as np

    _RASTER_PACKS = [
        np.frombuffer(blob, dtype=np.uint8).reshape((height, packed_width)).copy()
        for blob in blobs
    ]


def _score_raster_task(
    packed_blob: bytes,
    height: int,
    packed_width: int,
    meta: dict[str, Any],
) -> dict[str, Any]:
    import numpy as np

    cand = np.frombuffer(packed_blob, dtype=np.uint8).reshape((height, packed_width))
    iou, _angle = best_packed_raster_iou(_RASTER_PACKS, cand)
    item = dict(meta)
    item["iou"] = round(float(iou), 4)
    return item


def _screen_sql(filters: dict | None) -> tuple[str, list]:
    """层数、三个器件属性和具体完成时间，在算 IoU 之前滤掉。"""
    from datetime import datetime

    filters = filters or {}
    sql = ""
    params: list[Any] = []
    layers = filters.get("layers")
    if layers not in (None, ""):
        try:
            sql += " AND l.layers = %s"
            params.append(int(layers))
        except (TypeError, ValueError):
            pass
    for column in ("devicetype", "packagesymbol", "value"):
        text = str(filters.get(column) or "").strip()
        if text:
            sql += f" AND LOWER(TRIM(l.{column})) LIKE CONCAT('%%', LOWER(%s), '%%')"
            params.append(text)
    year = filters.get("finish_year")
    if year in (None, ""):
        return sql, params
    try:
        year_i = int(year)
    except (TypeError, ValueError):
        return sql, params
    month = filters.get("finish_month")
    try:
        month_i = int(month) if month not in (None, "") else None
    except (TypeError, ValueError):
        month_i = None
    if month_i and 1 <= month_i <= 12:
        start = datetime(year_i, month_i, 1)
        end = datetime(year_i + 1, 1, 1) if month_i == 12 else datetime(year_i, month_i + 1, 1)
    else:
        start = datetime(year_i, 1, 1)
        end = datetime(year_i + 1, 1, 1)
    sql += (
        " AND EXISTS (SELECT 1 FROM project_profile p "
        "WHERE p.project_code = l.project_code "
        "AND p.finish_time >= %s AND p.finish_time < %s)"
    )
    params.extend([start, end])
    return sql, params


def _fuzzy_finish_time_range(filters: dict) -> tuple[Any, Any] | None:
    """模糊检索里的年份，对应 project_profile.finish_time。"""
    from datetime import datetime

    year = filters.get("finish_year")
    if year in (None, ""):
        return None
    try:
        year_i = int(year)
    except (TypeError, ValueError):
        return None
    month = filters.get("finish_month")
    try:
        month_i = int(month) if month not in (None, "") else None
    except (TypeError, ValueError):
        month_i = None
    if month_i and 1 <= month_i <= 12:
        start = datetime(year_i, month_i, 1)
        end = datetime(year_i + 1, 1, 1) if month_i == 12 else datetime(year_i, month_i + 1, 1)
    else:
        start = datetime(year_i, 1, 1)
        end = datetime(year_i + 1, 1, 1)
    return start, end


def _append_fuzzy_layers(filters: dict, where: list[str], params: list[Any], group_cols: list[str]) -> None:
    """层数：一个值用等于；区间用 BETWEEN。"""
    lmin = filters.get("layers_min")
    lmax = filters.get("layers_max")
    layers = filters.get("layers")
    try:
        if lmin not in (None, "") and lmax not in (None, ""):
            low, high = int(lmin), int(lmax)
            if low > high:
                low, high = high, low
            if low == high:
                where.append("l.layers = %s")
                params.append(low)
            else:
                where.append("l.layers BETWEEN %s AND %s")
                params.extend([low, high])
            group_cols.append("l.layers")
            return
        if lmin not in (None, ""):
            where.append("l.layers = %s")
            params.append(int(lmin))
            group_cols.append("l.layers")
            return
        if layers not in (None, ""):
            where.append("l.layers = %s")
            params.append(int(layers))
            group_cols.append("l.layers")
    except (TypeError, ValueError):
        pass


def _append_fuzzy_pin(filters: dict, where: list[str], params: list[Any], group_cols: list[str]) -> None:
    pin = filters.get("pin_count")
    if pin in (None, ""):
        return
    try:
        where.append("l.pin_count = %s")
        params.append(int(pin))
        group_cols.append("l.pin_count")
    except (TypeError, ValueError):
        pass


def _append_fuzzy_year(filters: dict, where: list[str], params: list[Any], group_cols: list[str]) -> None:
    bounds = _fuzzy_finish_time_range(filters)
    if not bounds:
        return
    start, end = bounds
    where.append(
        "EXISTS (SELECT 1 FROM project_profile p "
        "WHERE p.project_code = l.project_code "
        "AND p.finish_time >= %s AND p.finish_time < %s)"
    )
    params.extend([start, end])
    group_cols.append("YEAR(p.finish_time)")


_FUZZY_JOIN = (
    " LEFT JOIN project_profile p "
    "ON p.project_code COLLATE utf8mb4_unicode_ci = l.project_code COLLATE utf8mb4_unicode_ci"
)


def query_fuzzy_library(conn, filters: dict | None) -> tuple[int, list[dict[str, Any]]]:
    """按用户给出的属性做模糊条件，再按这些属性分组计数。"""
    filters = filters or {}
    where: list[str] = []
    params: list[Any] = []
    group_cols: list[str] = []
    for column in ("devicetype", "packagesymbol", "value"):
        raw = str(filters.get(column) or "").strip()
        tokens: list[str] = []
        for part in re.split(r"[-_ \u3000\uff0d\u2013\u2014]+", raw):
            token = part.strip().replace("\\", "").replace("%", "").replace("_", "")
            if len(token) < 2 or token.lower() in {item.lower() for item in tokens}:
                continue
            tokens.append(token)
        if not tokens:
            continue
        group_cols.append(f"l.{column}")
        for token in tokens:
            where.append(f"LOWER(TRIM(l.{column})) LIKE %s")
            params.append(f"%{token.lower()}%")
    _append_fuzzy_layers(filters, where, params, group_cols)
    _append_fuzzy_pin(filters, where, params, group_cols)
    _append_fuzzy_year(filters, where, params, group_cols)
    if not where or not group_cols:
        return 0, []
    clause = " AND ".join(where)
    grouped = ", ".join(group_cols)
    need_join = any("p." in col or "YEAR(p." in col for col in group_cols)
    join_sql = _FUZZY_JOIN if need_join else ""
    count_sql = (
        "SELECT COUNT(*) FROM ("
        f"SELECT {grouped} FROM footprint_library l{join_sql} "
        f"WHERE {clause} GROUP BY {grouped}"
        ") grouped_attrs"
    )
    with conn.cursor() as cur:
        cur.execute(count_sql, params)
        total = int(cur.fetchone()[0])
        if total > 200:
            return total, []
        list_sql = (
            "SELECT l.devicetype, l.packagesymbol, l.value, l.layers, l.pin_count, "
            "YEAR(p.finish_time) AS finish_year, COUNT(*) AS hit_count "
            f"FROM footprint_library l{_FUZZY_JOIN} "
            f"WHERE {clause} "
            "GROUP BY l.devicetype, l.packagesymbol, l.value, l.layers, l.pin_count, YEAR(p.finish_time) "
            "ORDER BY hit_count DESC, l.devicetype, l.packagesymbol, l.value, l.layers, "
            "l.pin_count, finish_year"
        )
        cur.execute(list_sql, params)
        columns = [item[0] for item in (cur.description or [])]
        rows = [dict(zip(columns, record)) for record in cur.fetchall()]
    return total, rows


def query_devices_by_attributes(conn, choice: dict) -> list[dict[str, Any]]:
    """按用户选定的一组属性，取出项目编码和位号。"""
    where: list[str] = []
    params: list[Any] = []
    for column in ("devicetype", "packagesymbol", "value", "layers", "pin_count"):
        value = (choice or {}).get(column)
        if value is None or value == "":
            where.append(f"l.{column} IS NULL")
            continue
        where.append(f"l.{column} = %s")
        params.append(value)
    year = (choice or {}).get("finish_year")
    if year in (None, ""):
        where.append("YEAR(p.finish_time) IS NULL")
    else:
        where.append("YEAR(p.finish_time) = %s")
        params.append(int(year))
    if not where:
        return []
    sql = (
        "SELECT l.project_code, l.refdes, l.top_side, l.devicetype, l.packagesymbol, "
        "l.value, l.layers, l.pin_count, YEAR(p.finish_time) AS finish_year, "
        "p.customer_code, p.designer_name, p.designer_seniority, p.finish_time "
        f"FROM footprint_library l{_FUZZY_JOIN} "
        f"WHERE {' AND '.join(where)} "
        "ORDER BY l.project_code, l.refdes"
    )
    with conn.cursor() as cur:
        cur.execute(sql, params)
        columns = [item[0] for item in (cur.description or [])]
        return [dict(zip(columns, record)) for record in cur.fetchall()]


def _screen_active(filters: dict | None) -> bool:
    if not filters:
        return False
    if filters.get("layers") not in (None, "") or filters.get("finish_year") not in (None, ""):
        return True
    return any(str(filters.get(key) or "").strip() for key in ("devicetype", "packagesymbol", "value"))


def query_similar_raster(
    query_cache: FpGeomCache,
    conn,
    *,
    top_k: int,
    min_iou: float,
    exclude_self: bool,
    copper_area: float | None,
    area_tolerance: float,
    timing: dict[str, Any] | None = None,
    library_filters: dict | None = None,
) -> list[dict[str, Any]]:
    """用库里存好的 0° 栅格和 query 栅格比 IoU。query 只栅格化一次，四个方向用 np.rot90。"""
    import time as _time

    sql = """
    SELECT l.id, l.name, l.pin_count, l.project_code, l.refdes,
           l.bbox_min_x, l.bbox_min_y, l.bbox_max_x, l.bbox_max_y, r.raster_grid
    FROM footprint_library l
    LEFT JOIN footprint_raster r
      ON r.project_code = l.project_code
     AND REPLACE(r.refdes, '@', '') = REPLACE(l.refdes, '@', '')
    WHERE l.pin_count = %s
    """
    params: list[Any] = [query_cache.pin_count]
    if copper_area is not None and copper_area > 0:
        tol = max(0.0, float(area_tolerance))
        params.extend([copper_area * (1.0 - tol), copper_area * (1.0 + tol)])
        sql += " AND l.copper_area BETWEEN %s AND %s"
    screen_sql, screen_params = _screen_sql(library_filters)
    sql += screen_sql
    params.extend(screen_params)
    t_sql = _time.perf_counter()
    with conn.cursor() as cur:
        cur.execute(sql, params)
        raw_rows = cur.fetchall()
    sql_ms = int((_time.perf_counter() - t_sql) * 1000)

    t_iou = _time.perf_counter()
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from settings import get_concurrency_config

    query_grid = raster_from_cache(query_cache)
    query_size = int(query_grid.shape[0])
    packs = raster_query_packs(query_grid)
    height, packed_width = packs[0].shape
    pack_blobs = [pack.tobytes() for pack in packs]
    tasks: list[tuple[bytes, dict[str, Any]]] = []
    skipped = 0
    resized = 0
    for rid, name, pin_count, project_code, refdes, bminx, bminy, bmaxx, bmaxy, raster in raw_rows:
        project_code = project_code or ""
        refdes = refdes or ""
        if exclude_self and query_cache.project_code == project_code and query_cache.refdes == refdes:
            continue
        if not raster:
            skipped += 1
            continue
        blob = raster if isinstance(raster, (bytes, bytearray)) else bytes(raster)
        if len(blob) >= 8 and blob[:4] == b"RSTR":
            src_h, src_w = struct.unpack_from("<HH", blob, 4)
            if src_h != query_size or src_w != query_size:
                resized += 1
        packed = raster_packed_for_query(blob, query_size)
        tasks.append(
            (
                packed.tobytes(),
                {
                    "id": rid,
                    "name": name,
                    "project_code": project_code,
                    "refdes": refdes,
                    "pin_count": pin_count,
                    "bbox_min_x": float(bminx or 0),
                    "bbox_min_y": float(bminy or 0),
                    "bbox_max_x": float(bmaxx or 0),
                    "bbox_max_y": float(bmaxy or 0),
                },
            )
        )

    from agent_log import log_line

    log_line(
        "栅格",
        result=(
            f"【栅格比对】project_code={query_cache.project_code} refdes={query_cache.refdes} "
            f"脚数={query_cache.pin_count} 查询边长={query_size} 库中取出={len(raw_rows)} "
            f"空栅格跳过={skipped} 缩放到查询边长={resized} 进入比对={len(tasks)}"
        ),
    )
    conc = get_concurrency_config()
    use_parallel = len(tasks) > 1 and conc.iou_pool_max_workers > 1
    used_workers = 1
    scored: list[dict[str, Any]] = []
    if use_parallel:
        used_workers = conc.iou_workers_per_request
        pool = ProcessPoolExecutor(
            max_workers=used_workers,
            initializer=_init_raster_worker,
            initargs=(pack_blobs, height, packed_width),
        )
        try:
            futures = [
                pool.submit(_score_raster_task, blob, height, packed_width, meta)
                for blob, meta in tasks
            ]
            for fut in as_completed(futures):
                scored.append(fut.result())
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
    else:
        _init_raster_worker(pack_blobs, height, packed_width)
        for blob, meta in tasks:
            scored.append(_score_raster_task(blob, height, packed_width, meta))

    scored.sort(key=lambda row: row["iou"], reverse=True)
    filtered = [row for row in scored if row["iou"] >= min_iou]
    if top_k is not None and int(top_k) >= 0:
        filtered = filtered[: int(top_k)]
    _log_iou_processes(
        used_workers,
        len(tasks),
        f"栅格边长={query_size} 空栅格跳过={skipped} 缩放={resized} 达到阈值={len(filtered)}",
    )
    if timing is not None:
        timing["similar_sql"] = sql_ms
        timing["similar_minio"] = 0
        timing["minio_fpgeom_text"] = "对象数=0 耗时=0ms"
        timing["similar_iou"] = int((_time.perf_counter() - t_iou) * 1000)
        timing["candidate_rows"] = len(raw_rows)
        timing["legacy_rows"] = skipped
        timing["iou_workers"] = used_workers
    return filtered


def find_similar_from_content(
    conn,
    content: bytes | str,
    *,
    source_file: str = "upload.json",
    top_k: int | None = None,
    min_iou: float | None = None,
    library_filters: dict | None = None,
) -> dict[str, Any]:
    """上传 JSON 与库内同 pin 记录做 IoU 检索（仅检索，不入库）。"""
    import logging
    import time

    from settings import get_similarity_config

    log = logging.getLogger("pkg_sim")
    sim_cfg = get_similarity_config()
    resolved_top_k = sim_cfg.top_k if top_k is None else top_k
    resolved_min_iou = sim_cfg.min_iou if min_iou is None else min_iou

    t0 = time.perf_counter()
    raw_bytes = _raw_bytes_from_content(content)
    model = load_footprint_json_from_bytes(raw_bytes, source_name=source_file)
    query_cache = build_fpgeom_from_footprint(model, source_file=source_file)
    query_area = footprint_copper_area_mils2(model)
    parse_ms = int((time.perf_counter() - t0) * 1000)

    box: dict[str, Any] = {}
    results: list[dict[str, Any]] | None = None
    if _screen_active(library_filters):
        results = None
    else:
        try:
            results = load_precomputed_similar(
                conn,
                model.project_code,
                model.refdes,
                resolved_top_k,
                resolved_min_iou,
            )
        except Exception as exc:
            from agent_log import log_line

            log_line(
                "形态检索",
                result=f"【预计算表】读取失败 {type(exc).__name__}，改走栅格",
                level="warning",
            )
            results = None
    if results is not None:
        taken = "全部" if int(resolved_top_k) < 0 else f"前{resolved_top_k}"
        _log_iou_processes(0, len(results), f"预计算表 直接取{taken}")
        box["similar_sql"] = int((time.perf_counter() - t0) * 1000) - parse_ms
        box["similar_iou"] = 0
        box["candidate_rows"] = len(results)
        box["iou_workers"] = 0
    else:
        results = query_similar_raster(
            query_cache,
            conn,
            top_k=resolved_top_k,
            min_iou=resolved_min_iou,
            exclude_self=sim_cfg.exclude_self,
            copper_area=query_area,
            area_tolerance=sim_cfg.area_tolerance,
            timing=box,
            library_filters=library_filters,
        )
    public_results = public_similar_results(results)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    top_iou = results[0]["iou"] if results else None
    log.info(
        "similar project=%s refdes=%s pin=%d ret=%d top_iou=%s min_iou=%s top_k=%d %dms",
        model.project_code,
        model.refdes,
        model.pin_count,
        len(public_results),
        top_iou if top_iou is not None else "-",
        resolved_min_iou,
        resolved_top_k,
        elapsed_ms,
    )
    from agent_log import log_line

    log_line(
        "形态检索",
        detail=(
            f"项目={model.project_code} 位号={model.refdes} 脚数={model.pin_count} "
            f"面积={query_area:.6g}mils² 浮动=±{sim_cfg.area_tolerance:.0%} "
            f"初筛层数={ (library_filters or {}).get('layers') or '-' } "
            f"初筛年={(library_filters or {}).get('finish_year') or '-'}"
        ),
        result=(
            f"解析JSON={parse_ms}ms 查同脚数={box.get('similar_sql', 0)}ms "
            f"库内原文={box.get('legacy_rows', 0)} "
            f"【MinIO拉取形态】{box.get('minio_fpgeom_text', '对象数=0 耗时=0ms')} "
            f"计算IoU={box.get('similar_iou', 0)}ms "
            f"候选={box.get('candidate_rows', 0)} 返回={len(public_results)}"
        ),
    )

    return {
        "results": public_results,
        "_step_ms": {
            "similar_parse": parse_ms,
            "similar_sql": int(box.get("similar_sql") or 0),
            "similar_minio": int(box.get("similar_minio") or 0),
            "similar_iou": int(box.get("similar_iou") or 0),
        },
    }


def _materialize_blob(stored: bytes | str | None, *, purpose: str = "") -> bytes:
    from minio_store import resolve_stored_bytes

    return resolve_stored_bytes(stored, purpose=purpose)


def load_candidates_by_pin_count(
    conn,
    pin_count: int,
    *,
    exclude_id: int | None = None,
    timing: dict[str, Any] | None = None,
    copper_area: float | None = None,
    area_tolerance: float | None = None,
) -> list[FootprintRow]:
    from minio_store import measure_fetches, object_key_of

    sql = """
    SELECT id, name, pin_count, fpgeom, project_code, refdes,
           bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y
    FROM footprint_library
    WHERE pin_count = %s
    """
    params: list[Any] = [pin_count]
    # 面积和脚数一起做初筛。copper_area 已是 mils²；浮动比例来自检索 yaml。
    if copper_area is not None and copper_area > 0 and area_tolerance is not None:
        tol = max(0.0, float(area_tolerance))
        params.extend([copper_area * (1.0 - tol), copper_area * (1.0 + tol)])
        sql += " AND copper_area BETWEEN %s AND %s"
    if exclude_id is not None:
        sql += " AND id <> %s"
        params.append(exclude_id)
    sql += " ORDER BY name"
    rows: list[FootprintRow] = []
    import time as _time

    t_sql = _time.perf_counter()
    with conn.cursor() as cur:
        cur.execute(sql, params)
        raw_rows = cur.fetchall()
    sql_ms = int((_time.perf_counter() - t_sql) * 1000)
    legacy_rows = 0
    with measure_fetches() as acc:
        for rid, name, pc, blob, project_code, refdes, bminx, bminy, bmaxx, bmaxy in raw_rows:
            if not object_key_of(blob):
                legacy_rows += 1
            rows.append(
                FootprintRow(
                    id=rid,
                    name=name,
                    pin_count=pc,
                    fpgeom=_materialize_blob(blob, purpose="fpgeom"),
                    project_code=project_code or "",
                    refdes=refdes or "",
                    bbox_min_x=float(bminx or 0),
                    bbox_min_y=float(bminy or 0),
                    bbox_max_x=float(bmaxx or 0),
                    bbox_max_y=float(bmaxy or 0),
                )
            )
    if timing is not None:
        timing["similar_sql"] = sql_ms
        timing["similar_minio"] = acc.slice("fpgeom").ms
        timing["minio_fpgeom_text"] = acc.slice("fpgeom").format("fpgeom")
        timing["legacy_rows"] = legacy_rows
        timing["candidate_rows"] = len(rows)
    return rows


def _score_candidate_task(
    query_wkb: bytes,
    query_project: str,
    query_refdes: str,
    row_id: int,
    name: str,
    fpgeom_bytes: bytes,
    project_code: str,
    refdes: str,
    pin_count: int,
    bbox_min_x: float,
    bbox_min_y: float,
    bbox_max_x: float,
    bbox_max_y: float,
    exclude_self: bool,
) -> dict[str, Any] | None:
    """单候选 IoU。query 只还原一次，候选图形只建一次再旋转四次。"""
    if exclude_self and query_project == project_code and query_refdes == refdes:
        return None
    cand = fpgeom_from_bytes(fpgeom_bytes)
    iou, _angle = best_iou_rotations(cached_query_geom(query_wkb), shape_for_iou(cand))
    return {
        "id": row_id,
        "name": name,
        "project_code": project_code,
        "refdes": refdes,
        "pin_count": pin_count,
        "bbox_min_x": bbox_min_x,
        "bbox_min_y": bbox_min_y,
        "bbox_max_x": bbox_max_x,
        "bbox_max_y": bbox_max_y,
        "iou": round(iou, 4),
    }


def _log_iou_processes(used: int, candidates: int, mode: str) -> None:
    from agent_log import log_line

    log_line(
        "形态检索",
        result=f"【IoU进程】方式={mode} 本次使用进程数={used} 候选={candidates}",
    )


def query_similar(
    query_cache: FpGeomCache,
    conn,
    *,
    top_k: int | None = None,
    min_iou: float | None = None,
    exclude_id: int | None = None,
    exclude_self: bool | None = None,
    parallel: bool = True,
    timing: dict[str, Any] | None = None,
    copper_area: float | None = None,
) -> list[dict[str, Any]]:
    import time as _time
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from settings import get_concurrency_config, get_similarity_config

    sim = get_similarity_config()
    conc = get_concurrency_config()
    top_k = sim.top_k if top_k is None else top_k
    min_iou = sim.min_iou if min_iou is None else min_iou
    exclude_self = sim.exclude_self if exclude_self is None else exclude_self

    box: dict[str, Any] = {}
    candidates = load_candidates_by_pin_count(
        conn,
        query_cache.pin_count,
        exclude_id=exclude_id,
        timing=box,
        copper_area=copper_area,
        area_tolerance=sim.area_tolerance,
    )
    query_blob = pack_iou_shape(shape_for_iou(query_cache))
    cached_query_geom(query_blob)
    tasks: list[tuple] = [
        (
            query_blob,
            query_cache.project_code,
            query_cache.refdes,
            row.id,
            row.name,
            row.fpgeom,
            row.project_code,
            row.refdes,
            row.pin_count,
            row.bbox_min_x,
            row.bbox_min_y,
            row.bbox_max_x,
            row.bbox_max_y,
            exclude_self,
        )
        for row in candidates
    ]

    scored: list[dict[str, Any]] = []
    t_iou = _time.perf_counter()
    use_parallel = parallel and len(tasks) > 1 and conc.iou_pool_max_workers > 1
    used_workers = 1

    if use_parallel:
        executor = None
        local_pool: ProcessPoolExecutor | None = None
        try:
            from concurrency import get_iou_executor

            executor = get_iou_executor()
            used_workers = conc.iou_pool_max_workers
        except Exception:
            used_workers = conc.iou_workers_per_request
            local_pool = ProcessPoolExecutor(
                max_workers=used_workers,
                initializer=cached_query_geom,
                initargs=(query_blob,),
            )
            executor = local_pool
        try:
            futures = [executor.submit(_score_candidate_task, *task) for task in tasks]
            for fut in as_completed(futures):
                item = fut.result()
                if item is not None:
                    scored.append(item)
        finally:
            if local_pool is not None:
                local_pool.shutdown(wait=True, cancel_futures=True)
    else:
        for task in tasks:
            item = _score_candidate_task(*task)
            if item is not None:
                scored.append(item)

    _log_iou_processes(used_workers, len(tasks), "轮廓")
    scored.sort(key=lambda x: x["iou"], reverse=True)
    filtered = [row for row in scored if row["iou"] >= min_iou]
    if timing is not None:
        timing.update(box)
        timing["similar_iou"] = int((_time.perf_counter() - t_iou) * 1000)
        timing["candidate_rows"] = len(candidates)
        timing["iou_workers"] = used_workers
    return filtered[:top_k]


# ---------------------------------------------------------------------------
# 智能体扩展：失败记录 / PDF / 批量查询
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 错误码（接口与 ingest_failure.error_code 一致）
# ---------------------------------------------------------------------------
# ERR_SCRIPT      二开脚本/BRD 解析上报（唯一键 project_code，refdes=''）
# ERR_JSON_FORMAT JSON 无法解析（唯一键 project_code + refdes）
# ERR_JSON_FIELD  JSON 缺字段/格式不合格（唯一键 project_code + refdes）
# ERR_FPGEOM      形态 fpgeom 入库失败（唯一键 project_code + refdes）
# ERR_PDF         PDF 入库失败（唯一键 project_code，refdes=''）

ERROR_CODES = {
    "SCRIPT": "ERR_SCRIPT",
    "JSON_FORMAT": "ERR_JSON_FORMAT",
    "JSON_FIELD": "ERR_JSON_FIELD",
    "FPGEOM": "ERR_FPGEOM",
    "PDF": "ERR_PDF",
}

# ingest_failure.detail 只存摘要；完整原因见服务端日志（library FAIL / ingest_fail）
DETAIL_DB_MAX = 2000


def truncate_failure_detail(text: str, *, max_len: int = DETAIL_DB_MAX) -> str:
    """截断写入 DB 的失败摘要，避免 TEXT 溢出导致整批入库中断。"""
    text = str(text or "").strip()
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    return f"{text[: max_len - 28]}...(truncated total={len(text)}chars)"


def record_failure(
    conn,
    *,
    error_code: str,
    detail: str,
    project_code: str,
    refdes: str | None = None,
) -> None:
    """失败记录 UPSERT：BRD/二开 refdes=''；JSON/形态 refdes=位号。DB 仅存摘要，失败不抛异常。"""
    import logging

    log = logging.getLogger("pkg_sim")
    code = str(project_code or "").strip()
    if not code:
        raise ValueError("record_failure 需要 project_code")
    ref = normalize_refdes(refdes)
    text = truncate_failure_detail(str(detail or "").strip() or error_code)

    def _write(detail_text: str) -> None:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ingest_failure (project_code, refdes, error_code, detail)
                VALUES (%s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    error_code=VALUES(error_code),
                    detail=VALUES(detail),
                    updated_at=CURRENT_TIMESTAMP
                """,
                (code, ref, error_code, detail_text),
            )
        conn.commit()

    try:
        _write(text)
    except Exception as exc:
        from mysql_diag import log_mysql_exception

        log_mysql_exception(
            conn,
            exc,
            operation="record_failure",
            detail=f"project_code={code} refdes={ref}",
            sql_hint="ingest_failure",
        )
        log.warning(
            "record_failure db_error project=%s refdes=%s code=%s err=%s detail_len=%d",
            code,
            ref,
            error_code,
            exc,
            len(text),
        )
        try:
            conn.rollback()
        except Exception:
            pass
        # 再试一次极短摘要，仍失败则只打日志，不中断入库批次
        mini = truncate_failure_detail(
            f"{error_code} file_ref={ref or '-'} {str(exc)[:120]}",
            max_len=500,
        )
        try:
            _write(mini)
        except Exception as exc2:
            try:
                conn.rollback()
            except Exception:
                pass
            log.warning(
                "record_failure skipped project=%s refdes=%s code=%s err=%s",
                code,
                ref,
                error_code,
                exc2,
            )


def upsert_project_pdf(
    conn,
    *,
    project_code: str,
    top_side: bool,
    file_name: str,
    content: bytes,
    pdf_scale: float = 0,
    finish_time=None,
) -> None:
    from minio_store import object_key_of, save_object
    from top_side_util import parse_top_side, top_side_label, top_side_to_db

    scale = float(pdf_scale or 0)
    side = top_side_to_db(parse_top_side(top_side))
    code = str(project_code or "").strip()
    old_key = None
    with conn.cursor() as cur:
        cur.execute(
            "SELECT file_content FROM project_pdf WHERE project_code=%s AND top_side=%s LIMIT 1",
            (code, side),
        )
        old_row = cur.fetchone()
        if old_row:
            old_key = object_key_of(old_row[0])
    finish_time = resolve_object_finish_time(conn, code, finish_time)
    path = save_object(
        content,
        finish_time=finish_time,
        filename=f"{code}_{top_side_label(bool(side))}.pdf",
        content_type="application/pdf",
        old_key=old_key,
    )
    from mysql_diag import db_transaction

    with db_transaction(
        conn,
        operation="upsert_project_pdf",
        detail=f"project_code={code} top_side={side}",
        sql_hint="project_pdf",
    ):
        with conn.cursor() as cur:
            created = _take_created_at(
                cur,
                "project_pdf",
                "project_code = %s AND top_side = %s",
                (code, side),
            )
            if created is not None:
                cur.execute(
                    """
                    INSERT INTO project_pdf (
                        project_code, top_side, file_name, file_content, pdf_scale, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (code, side, file_name, path, scale, created),
                )
            else:
                cur.execute(
                    """
                    INSERT INTO project_pdf (project_code, top_side, file_name, file_content, pdf_scale)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (code, side, file_name, path, scale),
                )


def fetch_pdf(conn, project_code: str, top_side: bool | None = None) -> dict | None:
    from top_side_util import parse_top_side, top_side_from_db, top_side_to_db

    side = top_side_to_db(parse_top_side(top_side))
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, project_code, top_side, file_name, file_content, pdf_scale
            FROM project_pdf
            WHERE project_code = %s AND top_side = %s
            LIMIT 1
            """,
            (project_code, side),
        )
        row = cur.fetchone()
    if not row:
        return None
    return {
        "id": row[0],
        "project_code": row[1],
        "top_side": top_side_from_db(row[2]),
        "file_name": row[3],
        "file_content": _materialize_blob(row[4], purpose="pdf"),
        "pdf_scale": row[5],
    }


def fetch_components_for_side(conn, project_code: str, top_side: bool) -> list[dict]:
    """同项目同面的全部器件（用于 PDF 仿射校准）。"""
    from top_side_util import parse_top_side, top_side_from_db, top_side_to_db

    side = top_side_to_db(parse_top_side(top_side))
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT refdes, pcb_x, pcb_y, bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y, top_side
            FROM footprint_library
            WHERE project_code = %s AND (top_side = %s OR top_side IS NULL)
            ORDER BY refdes
            """,
            (project_code, side),
        )
        rows = cur.fetchall()
    out: list[dict] = []
    for refdes, px, py, bminx, bminy, bmaxx, bmaxy, ts in rows:
        out.append(
            {
                "refdes": refdes,
                "pcb_x": px,
                "pcb_y": py,
                "bbox_min_x": bminx,
                "bbox_min_y": bminy,
                "bbox_max_x": bmaxx,
                "bbox_max_y": bmaxy,
                "top_side": top_side_from_db(ts),
            }
        )
    return out


def fetch_component_row(conn, project_code: str, refdes: str) -> dict | None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT f.id, f.project_code, f.name, f.refdes, f.layers, f.devicetype, f.packagesymbol, f.value,
                   f.pcb_x, f.pcb_y, f.top_side,
                   p.customer_code, p.designer_name, p.designer_seniority, p.finish_time,
                   f.pin_count, f.units, f.bbox_min_x, f.bbox_min_y, f.bbox_max_x, f.bbox_max_y,
                   f.pdf_refdes_x, f.pdf_refdes_y, f.source_json,
                   (f.pdf_refdes_x IS NOT NULL AND f.pdf_refdes_y IS NOT NULL) AS pdf_anchor_ready
            FROM footprint_library f
            LEFT JOIN project_profile p ON p.project_code = f.project_code
            WHERE f.project_code = %s AND REPLACE(f.refdes, '@', '') = %s
            LIMIT 1
            """,
            (project_code, normalize_refdes(refdes)),
        )
        row = cur.fetchone()
    if not row:
        return None
    keys = [
        "id", "project_code", "name", "refdes", "layers", "devicetype", "packagesymbol", "value",
        "pcb_x", "pcb_y", "top_side", "customer_code", "designer_name", "designer_seniority",
        "finish_time", "pin_count", "units", "bbox_min_x", "bbox_min_y", "bbox_max_x", "bbox_max_y",
        "pdf_refdes_x", "pdf_refdes_y", "source_json",
        "pdf_anchor_ready",
    ]
    from top_side_util import top_side_from_db

    out = dict(zip(keys, row))
    out["pdf_anchor_ready"] = bool(out.get("pdf_anchor_ready"))
    out["highlight_ready"] = out["pdf_anchor_ready"]
    if "top_side" in out:
        out["top_side"] = top_side_from_db(out.get("top_side"))
    out["board_name"] = out.get("name")
    if isinstance(out.get("source_json"), (str, bytes)):
        try:
            from minio_store import object_key_of, resolve_stored_bytes

            key = object_key_of(out.get("source_json"))
            text = resolve_stored_bytes(key, purpose="source_json").decode("utf-8") if key else out["source_json"]
            if isinstance(text, bytes):
                text = text.decode("utf-8")
            out["source_json"] = json.loads(text)
        except Exception:
            pass
    return out


def update_component_pdf_refdes(
    conn,
    *,
    project_code: str,
    refdes: str,
    pdf_refdes_x: float | None,
    pdf_refdes_y: float | None,
) -> None:
    """只更新丝印坐标和更新时间，创建时间保持第一次。"""
    from mysql_diag import db_transaction

    ref = normalize_refdes(refdes)
    with db_transaction(
        conn,
        operation="update_component_pdf_refdes",
        detail=f"project_code={project_code} refdes={ref}",
        sql_hint="footprint_library UPDATE pdf_refdes",
    ):
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE footprint_library
                SET pdf_refdes_x=%s, pdf_refdes_y=%s, updated_at=CURRENT_TIMESTAMP
                WHERE project_code=%s AND REPLACE(refdes, '@', '')=%s
                """,
                (pdf_refdes_x, pdf_refdes_y, project_code, ref),
            )


def update_component_highlight(
    conn,
    *,
    project_code: str,
    refdes: str,
    highlight_pdf: bytes | None = None,
    highlight_preview: bytes | None = None,
) -> None:
    """兼容旧调用：已改为坐标入库，此函数不再写 BLOB。"""
    del highlight_pdf, highlight_preview


def fetch_component_highlight_blobs(conn, project_code: str, refdes: str) -> dict | None:
    """已废弃 BLOB 预览，统一从 project_pdf 现场渲染。"""
    return None


def list_components_for_highlight_bake(
    conn,
    project_code: str | None = None,
    *,
    include_baked: bool = False,
) -> list[dict]:
    """列出可烘焙位号（项目有板级 PDF）。"""
    sql = """
        SELECT f.project_code, f.refdes, f.top_side, p.file_name
        FROM footprint_library f
        INNER JOIN project_pdf p
          ON p.project_code COLLATE utf8mb4_unicode_ci = f.project_code COLLATE utf8mb4_unicode_ci
         AND p.top_side = f.top_side
        WHERE 1=1
    """
    if not include_baked:
        sql += " AND (f.pdf_refdes_x IS NULL OR f.pdf_refdes_y IS NULL)"
    params: list = []
    if project_code:
        sql += " AND f.project_code = %s"
        params.append(project_code)
    sql += " ORDER BY f.project_code, f.refdes"
    from top_side_util import top_side_from_db

    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    return [
        {
            "project_code": r[0],
            "refdes": r[1],
            "top_side": top_side_from_db(r[2]),
            "board_pdf_name": r[3],
        }
        for r in rows
    ]


OA_SYNC_JOB_HISTORY = "history_backfill"


def _oa_sync_progress_columns(conn) -> set[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COLUMN_NAME FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'oa_sync_progress'
            """
        )
        return {row[0] for row in cur.fetchall()}


def _migrate_oa_sync_progress_table(conn) -> None:
    """旧版多列表迁移为 job_key + done_through。"""
    with conn.cursor() as cur:
        cur.execute("SHOW TABLES LIKE 'oa_sync_progress'")
        if not cur.fetchone():
            return
    cols = _oa_sync_progress_columns(conn)
    if not cols:
        return
    if "done_through" in cols and "next_start" not in cols and "plan_start" not in cols:
        return
    rows: list[tuple[str, str | None]] = []
    with conn.cursor() as cur:
        if "next_start" in cols:
            cur.execute("SELECT job_key, next_start, completed, plan_start FROM oa_sync_progress")
            for job_key, next_start, completed, plan_start in cur.fetchall():
                done: str | None = None
                if next_start is not None:
                    from datetime import date, timedelta

                    ns = next_start if isinstance(next_start, date) else date.fromisoformat(str(next_start)[:10])
                    ps = (
                        plan_start
                        if isinstance(plan_start, date)
                        else date.fromisoformat(str(plan_start)[:10])
                        if plan_start
                        else ns
                    )
                    if ns > ps:
                        done = (ns - timedelta(days=1)).isoformat()
                    elif completed:
                        done = ns.isoformat() if hasattr(ns, "isoformat") else str(ns)[:10]
                rows.append((str(job_key), done))
        elif "job_key" in cols:
            cur.execute("SELECT job_key, done_through FROM oa_sync_progress")
            rows = [(str(r[0]), str(r[1])[:10] if r[1] else None) for r in cur.fetchall()]
    with conn.cursor() as cur:
        cur.execute("DROP TABLE oa_sync_progress")
        cur.execute(
            """
            CREATE TABLE oa_sync_progress (
                job_key VARCHAR(64) NOT NULL PRIMARY KEY,
                done_through DATE NULL COMMENT '已处理完成的最后一天（含）'
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
        )
        for job_key, done in rows:
            cur.execute(
                "INSERT INTO oa_sync_progress (job_key, done_through) VALUES (%s, %s)",
                (job_key, done),
            )
    conn.commit()


def fetch_oa_sync_done_through(conn, job_key: str) -> str | None:
    """读取已处理到的最后一天（含），无记录返回 None。"""
    key = str(job_key or "").strip()
    if not key:
        return None
    with conn.cursor() as cur:
        cur.execute(
            "SELECT done_through FROM oa_sync_progress WHERE job_key = %s LIMIT 1",
            (key,),
        )
        row = cur.fetchone()
    if not row or row[0] is None:
        return None
    val = row[0]
    if hasattr(val, "isoformat"):
        return val.isoformat()[:10]
    return str(val)[:10]


def upsert_oa_sync_done_through(conn, job_key: str, done_through: str | None) -> None:
    """写入已处理到的最后一天（含）。"""
    key = str(job_key or "").strip()
    if not key:
        raise ValueError("job_key 不能为空")
    done = str(done_through)[:10] if done_through else None
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO oa_sync_progress (job_key, done_through) VALUES (%s, %s)
            ON DUPLICATE KEY UPDATE done_through = VALUES(done_through)
            """,
            (key, done),
        )
    conn.commit()


def _migrate_process_table(conn) -> None:
    """确保 process 表存在（与 schema.sql 一致）。"""
    with conn.cursor() as cur:
        cur.execute("SHOW TABLES LIKE 'process'")
        if cur.fetchone():
            return
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS `process` (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                project_code VARCHAR(128) NOT NULL,
                oa_info_ok TINYINT(1) NULL,
                oa_info_note VARCHAR(512) NULL,
                oa_download_ok TINYINT(1) NULL,
                oa_download_note VARCHAR(512) NULL,
                oa_brd_ok TINYINT(1) NULL,
                oa_brd_note VARCHAR(512) NULL,
                parse_ok TINYINT(1) NULL,
                parse_note VARCHAR(512) NULL,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                UNIQUE KEY uk_process_project (project_code),
                KEY idx_process_parse (parse_ok)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
        )
    conn.commit()


def delete_project_components(conn, project_code: str) -> int:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM footprint_library WHERE project_code = %s", (project_code,))
        deleted = cur.rowcount
        cur.execute("DELETE FROM project_pdf WHERE project_code = %s", (project_code,))
        cur.execute("DELETE FROM project_profile WHERE project_code = %s", (project_code,))
        cur.execute("DELETE FROM ingest_failure WHERE project_code = %s", (project_code,))
    conn.commit()
    return int(deleted or 0)

