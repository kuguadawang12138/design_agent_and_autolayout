# -*- coding: utf-8 -*-
"""封装库 MySQL 读写（fpgeom + 元数据）。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fpgeom import (
    FPGEOM_VERSION,
    FpGeomCache,
    best_iou_rotations,
    build_fpgeom_from_footprint,
    cached_query_geom,
    footprint_copper_area_mils2,
    fpgeom_from_bytes,
    fpgeom_to_bytes,
    pack_iou_shape,
    shape_for_iou,
    raster_from_cache,
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


def init_schema(conn, schema_path: Path | None = None) -> None:
    """仅由 setup_db.py / 运维脚本调用；运行中的入库服务不要调用（避免并发 DDL）。"""
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
    _migrate_pdf_refdes_backfill_log(conn)
    _migrate_pending_project_status(conn)


PROJECT_PROFILE_FIELDS = (
    "customer_code",
    "designer_name",
    "designer_seniority",
    "finish_time",
)


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


def _unique_row_exists(cur, table: str, where_sql: str, params: tuple) -> bool:
    """按唯一键条件判断行是否已存在。"""
    cur.execute(f"SELECT 1 FROM {table} WHERE {where_sql} LIMIT 1", params)
    return cur.fetchone() is not None


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


def upsert_project_profile(conn, project_code: str, info: dict | None) -> None:
    """按 project_code 唯一键：无则 INSERT；有则 DELETE 后 INSERT（合并旧四字段）。"""
    from mysql_diag import db_transaction

    code = str(project_code or "").strip()
    if not code:
        return
    payload = _extract_project_profile(info)
    if not payload:
        return
    existing = fetch_project_profile(conn, code) or {}
    merged: dict[str, Any] = {}
    for key in PROJECT_PROFILE_FIELDS:
        val = payload.get(key)
        if val in (None, ""):
            val = existing.get(key)
        if val not in (None, ""):
            merged[key] = val
    if not merged:
        return
    with db_transaction(
        conn,
        operation="upsert_project_profile",
        detail=f"project_code={code}",
        sql_hint="project_profile",
    ):
        with conn.cursor() as cur:
            created = _take_created_at(cur, "project_profile", "project_code = %s", (code,))
            cols = list(merged.keys())
            values: list[Any] = [code] + [merged[c] for c in cols]
            if created is not None:
                cols.append("created_at")
                values.append(created)
            placeholders = ", ".join(["%s"] * len(cols))
            cur.execute(
                f"INSERT INTO project_profile (project_code, {', '.join(cols)}) VALUES (%s, {placeholders})",
                values,
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
    prof = dict(zip(PROJECT_PROFILE_FIELDS, row))
    return _normalize_profile_values(prof)


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


def _normalize_profile_values(prof: dict[str, Any]) -> dict[str, Any]:
    """MySQL DATETIME 等转为可 JSON 序列化的值。"""
    from datetime import date, datetime

    out: dict[str, Any] = {}
    for key, val in prof.items():
        if isinstance(val, datetime):
            out[key] = val.isoformat(sep=" ", timespec="seconds")
        elif isinstance(val, date):
            out[key] = val.isoformat()
        else:
            out[key] = val
    return out


def _merge_project_profile(conn, row: dict) -> dict:
    """查询结果合并项目属性（接口字段与旧版 footprint 行一致）。"""
    if not row:
        return row
    code = row.get("project_code")
    prof = fetch_project_profile(conn, code) if code else None
    if prof:
        row.update(prof)
    return row


def _migrate_object_path_columns(conn) -> None:
    """fpgeom、source_json、file_content 改为 VARCHAR，只存 MinIO 路径。"""
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
            try:
                cur.execute(f"ALTER TABLE `{table}` MODIFY COLUMN `{col}` {decl}")
                log_line("数据库", result=f"【字段改类型】{table}.{col} -> VARCHAR(512)")
            except Exception as exc:
                log_line(
                    "数据库",
                    result=(
                        f"【字段改类型失败】{table}.{col} 仍是 {col_type}。"
                        f"列里若还是旧二进制，长度超过 512，需要先清空该列再执行 setup_db。原因={exc}"
                    ),
                    level="error",
                )
                raise
    conn.commit()


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
            # warnings 仍是文本；source_json / fpgeom 改为路径 VARCHAR，见 _migrate_object_path_columns
            for col, comment in (("warnings", "解析告警（JSON 数组字符串）"),):
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
    """归档为 UTF-8 JSON 文本（GBK 已转码，顶层转义已还原，ensure_ascii=False）。"""
    from footprint_json import normalized_json_text

    try:
        return normalized_json_text(raw_bytes)
    except Exception:
        return decode_fallback_text(raw_bytes)


def decode_fallback_text(raw_bytes: bytes) -> str:
    from footprint_json import decode_design_json_bytes

    return decode_design_json_bytes(raw_bytes)


def _load_source_json(raw_bytes: bytes) -> dict[str, Any]:
    from footprint_json import decode_design_json_bytes, loads_design_json

    text = decode_design_json_bytes(raw_bytes)
    try:
        data, _how = loads_design_json(text)
        return data
    except json.JSONDecodeError:
        return {"raw": text[:5000]}


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


def _materialize_blob(stored: bytes | str | None) -> bytes:
    from minio_store import resolve_stored_bytes

    return resolve_stored_bytes(stored)


def _offload_library_objects(row: dict[str, Any], *, old_fpgeom: str | None, old_source_json: str | None) -> None:
    """fpgeom / source_json 写入 MinIO，库内只留 VARCHAR 路径。已有对象先删再传。"""
    from minio_store import object_key_of, save_object

    finish = row.get("_finish_time")
    code = str(row.get("project_code") or "").strip()
    ref = str(row.get("refdes") or "").strip().upper()
    fp = row.get("fpgeom")
    if object_key_of(fp) is None:
        raw = bytes(fp) if isinstance(fp, (bytes, bytearray)) else str(fp or "").encode("utf-8")
        row["fpgeom"] = save_object(
            raw,
            finish_time=finish,
            filename=f"{code}_{ref}.fpgeom",
            content_type="application/octet-stream",
            old_key=old_fpgeom,
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
            old_key=old_source_json,
        )
    else:
        row["source_json"] = object_key_of(sj)


def _library_object_keys(conn, project_code: str, refdes: str) -> tuple[str | None, str | None]:
    from minio_store import object_key_of

    code = str(project_code or "").strip()
    ref = normalize_refdes(refdes)
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT fpgeom, source_json FROM footprint_library WHERE project_code=%s AND {_REFDES_EQ} LIMIT 1",
            (code, ref),
        )
        row = cur.fetchone()
    if not row:
        return None, None
    return object_key_of(row[0]), object_key_of(row[1])


# 脚数能确定且不超过 6 的器件不写入器件表、栅格、待完成 JSON 和预计算表。
PIN_COUNT_STORE_MAX_SKIP = 6
# 待完成 IoU 沿用同一门槛：不超过 6 脚不计算，表里已有的也会删掉。
PENDING_IOU_MIN_PINS = PIN_COUNT_STORE_MAX_SKIP


def pin_count_too_small(pin_count) -> bool:
    """脚数已知且不超过 6 返回 True。未知脚数不拦截。"""
    if pin_count is None or pin_count == "":
        return False
    try:
        pins = int(pin_count)
    except (TypeError, ValueError):
        return False
    return pins <= PIN_COUNT_STORE_MAX_SKIP


def _exec_ignore_missing_table(cur, sql: str, params: tuple) -> None:
    try:
        cur.execute(sql, params)
    except Exception as exc:
        code = exc.args[0] if getattr(exc, "args", None) else None
        if code == 1146:
            return
        raise


def drop_small_pin_library(conn, project_code: str, refdes: str) -> None:
    """本次入库的小脚器件不保留器件行和栅格。只删这一个项目位号。"""
    code = str(project_code or "").strip()
    ref = normalize_refdes(refdes)
    if not code or not ref:
        return
    with conn.cursor() as cur:
        _exec_ignore_missing_table(
            cur,
            f"DELETE FROM footprint_library WHERE project_code=%s AND {_REFDES_EQ}",
            (code, ref),
        )
        _exec_ignore_missing_table(
            cur,
            f"DELETE FROM footprint_raster WHERE project_code=%s AND {_REFDES_EQ}",
            (code, ref),
        )
    conn.commit()


def drop_small_pin_pending(conn, project_code: str, refdes: str) -> None:
    """小脚器件不留待完成 JSON，也不留它自己的预计算结果。"""
    code = str(project_code or "").strip()
    ref = normalize_refdes(refdes)
    if not code or not ref:
        return
    with conn.cursor() as cur:
        _exec_ignore_missing_table(
            cur,
            f"DELETE FROM pending_iou WHERE project_code=%s AND {_REFDES_EQ}",
            (code, ref),
        )
        _exec_ignore_missing_table(
            cur,
            f"DELETE FROM pending_json WHERE project_code=%s AND {_REFDES_EQ}",
            (code, ref),
        )
        _exec_ignore_missing_table(
            cur,
            f"DELETE FROM pending_json_block WHERE project_code=%s AND {_REFDES_EQ}",
            (code, ref),
        )
    conn.commit()


def upsert_footprint(conn, row: dict[str, Any]) -> str:
    """按 project_code+refdes 唯一键：已存在则先删后插，否则直接插入。创建时间沿用第一次。"""
    from mysql_diag import db_transaction

    row["refdes"] = normalize_refdes(row.get("refdes"))
    if pin_count_too_small(row.get("pin_count")):
        drop_small_pin_library(conn, row.get("project_code"), row["refdes"])
        drop_small_pin_pending(conn, row.get("project_code"), row["refdes"])
        from agent_log import log_line

        log_line(
            "器件入库",
            result=(
                f"【脚数不超过{PIN_COUNT_STORE_MAX_SKIP}不入库】"
                f"project_code={row.get('project_code')} refdes={row['refdes']} "
                f"脚数={row.get('pin_count')}"
            ),
        )
        return "skipped"
    row["_finish_time"] = resolve_object_finish_time(
        conn, row.get("project_code"), row.get("_finish_time")
    )
    old_fp, old_json = _library_object_keys(conn, row.get("project_code"), row.get("refdes"))
    _offload_library_objects(row, old_fpgeom=old_fp, old_source_json=old_json)
    code = row["project_code"]
    ref = row["refdes"]
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

    if pin_count_too_small(row.get("pin_count")):
        upsert_footprint(conn, row)
        return {
            "code": 1,
            "success": True,
            "skipped": True,
            "pin_count": row.get("pin_count"),
            "message": f"脚数不超过 {PIN_COUNT_STORE_MAX_SKIP}，未入库",
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
            "message": f"脚数不超过 {PIN_COUNT_STORE_MAX_SKIP}，未入库",
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


def find_similar_from_content(
    conn,
    content: bytes | str,
    *,
    source_file: str = "upload.json",
    top_k: int | None = None,
    min_iou: float | None = None,
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

    results = query_similar(
        query_cache,
        conn,
        top_k=resolved_top_k,
        min_iou=resolved_min_iou,
        exclude_self=sim_cfg.exclude_self,
        copper_area=query_area,
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

    return {"results": public_results}


def load_candidates_by_pin_count(
    conn,
    pin_count: int,
    *,
    exclude_id: int | None = None,
    copper_area: float | None = None,
    area_tolerance: float | None = None,
    exclude_project_code: str | None = None,
) -> list[FootprintRow]:
    sql = """
    SELECT id, name, pin_count, fpgeom, project_code, refdes,
           bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y
    FROM footprint_library
    WHERE pin_count = %s
    """
    params: list[Any] = [pin_count]
    # 面积和脚数一起做初筛。copper_area 已是 mils²。
    if copper_area is not None and copper_area > 0 and area_tolerance is not None:
        tol = max(0.0, float(area_tolerance))
        params.extend([copper_area * (1.0 - tol), copper_area * (1.0 + tol)])
        sql += " AND copper_area BETWEEN %s AND %s"
    if exclude_project_code:
        sql += " AND project_code <> %s"
        params.append(str(exclude_project_code).strip())
    if exclude_id is not None:
        sql += " AND id <> %s"
        params.append(exclude_id)
    sql += " ORDER BY name"
    rows: list[FootprintRow] = []
    with conn.cursor() as cur:
        cur.execute(sql, params)
        for rid, name, pc, blob, project_code, refdes, bminx, bminy, bmaxx, bmaxy in cur.fetchall():
            rows.append(
                FootprintRow(
                    id=rid,
                    name=name,
                    pin_count=pc,
                    fpgeom=_materialize_blob(blob),
                    project_code=project_code or "",
                    refdes=refdes or "",
                    bbox_min_x=float(bminx or 0),
                    bbox_min_y=float(bminy or 0),
                    bbox_max_x=float(bmaxx or 0),
                    bbox_max_y=float(bmaxy or 0),
                )
            )
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
    copper_area: float | None = None,
    exclude_project_code: str | None = None,
) -> list[dict[str, Any]]:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from settings import get_concurrency_config, get_similarity_config

    sim = get_similarity_config()
    conc = get_concurrency_config()
    top_k = sim.top_k if top_k is None else top_k
    min_iou = sim.min_iou if min_iou is None else min_iou
    exclude_self = sim.exclude_self if exclude_self is None else exclude_self

    candidates = load_candidates_by_pin_count(
        conn,
        query_cache.pin_count,
        exclude_id=exclude_id,
        copper_area=copper_area,
        area_tolerance=sim.area_tolerance,
        exclude_project_code=exclude_project_code,
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
    if top_k is not None and top_k < 0:
        return filtered
    return filtered[:top_k]


PENDING_IOU_MIN = 0.8


def assign_iou_ranks(rows: list[dict[str, Any]], min_iou: float = PENDING_IOU_MIN) -> list[dict[str, Any]]:
    """丢掉低于阈值的项，按分值从高到低编号，1 是最高分。"""
    kept = [dict(row) for row in rows if float(row.get("iou") or 0) >= float(min_iou)]
    kept.sort(key=lambda row: float(row["iou"]), reverse=True)
    for index, row in enumerate(kept, start=1):
        row["rank_no"] = index
    return kept


def _migrate_pending_project_status(conn) -> None:
    """给 pending_project 补 status。0=待完成，-1=已完成板已解出 BRD。"""
    with conn.cursor() as cur:
        cur.execute("SHOW TABLES LIKE 'pending_project'")
        if cur.fetchone() is None:
            return
        cur.execute("SHOW COLUMNS FROM pending_project")
        cols = {row[0] for row in cur.fetchall()}
        if "status" in cols:
            return
        cur.execute(
            "ALTER TABLE pending_project ADD COLUMN status TINYINT NOT NULL DEFAULT 0 "
            "COMMENT '0=待完成，-1=已完成板已解出BRD' AFTER project_code"
        )
    conn.commit()


def pending_project_status(conn, project_code: str) -> int | None:
    """没有记录返回 None。0=待完成，-1=已完成板已解出 BRD。"""
    code = str(project_code or "").strip()
    if not code:
        return None
    with conn.cursor() as cur:
        cur.execute(
            "SELECT status FROM pending_project WHERE project_code=%s LIMIT 1",
            (code,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    return int(row[0])


def is_pending_project(conn, project_code: str) -> bool:
    """状态仍为 0 才算待完成。-1 已经可以按完成板入库。"""
    return pending_project_status(conn, project_code) == 0


def mark_pending_project(conn, project_code: str) -> None:
    """OA 交来待完成板：按项目编码删掉旧记录再插入，状态固定为 0。"""
    code = str(project_code or "").strip()
    if not code:
        raise ValueError("project_code 不能为空")
    with conn.cursor() as cur:
        cur.execute("DELETE FROM pending_iou WHERE project_code=%s", (code,))
        cur.execute("DELETE FROM pending_json WHERE project_code=%s", (code,))
        _exec_ignore_missing_table(
            cur,
            "DELETE FROM pending_json_block WHERE project_code=%s",
            (code,),
        )
        cur.execute("DELETE FROM pending_project WHERE project_code=%s", (code,))
        cur.execute(
            "INSERT INTO pending_project (project_code, status) VALUES (%s, 0)",
            (code,),
        )
    conn.commit()


def mark_pending_brd_extracted(conn, project_code: str) -> bool:
    """拉取已完成板并解出 BRD 后，把状态从 0 改成 -1。不在表里或已经是 -1 则不动。"""
    code = str(project_code or "").strip()
    if not code:
        return False
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE pending_project
            SET status=-1
            WHERE project_code=%s AND status=0
            """,
            (code,),
        )
        changed = cur.rowcount > 0
    conn.commit()
    return changed


def replace_pending_iou(conn, project_code: str, refdes: str, matches: list[dict[str, Any]]) -> int:
    """按位号整组替换预计算 IoU。只保留 80% 及以上，最高分 rank_no=1。"""
    code = str(project_code or "").strip()
    ref = normalize_refdes(refdes)
    ranked = assign_iou_ranks(matches, PENDING_IOU_MIN)
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM pending_iou WHERE project_code=%s AND refdes=%s",
            (code, ref),
        )
        if ranked:
            cur.executemany(
                """
                INSERT INTO pending_iou (
                    project_code, refdes, match_project_code, match_refdes, iou, rank_no,
                    bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                [
                    (
                        code,
                        ref,
                        row.get("project_code") or "",
                        normalize_refdes(row.get("refdes")),
                        float(row["iou"]),
                        int(row["rank_no"]),
                        row.get("bbox_min_x"),
                        row.get("bbox_min_y"),
                        row.get("bbox_max_x"),
                        row.get("bbox_max_y"),
                    )
                    for row in ranked
                ],
            )
    conn.commit()
    return len(ranked)


def precompute_pending_iou(
    conn,
    content: bytes | str,
    *,
    source_file: str = "upload.json",
    parallel: bool = True,
) -> int:
    """待完成项目的一个位号：脚数和面积初筛后做轮廓 IoU，结果写入 pending_iou。"""
    raw_bytes = _raw_bytes_from_content(content)
    model = load_footprint_json_from_bytes(raw_bytes, source_name=source_file)
    if pin_count_too_small(model.pin_count):
        drop_small_pin_pending(conn, model.project_code, model.refdes)
        from agent_log import log_line

        log_line(
            "待完成IOU",
            result=(
                f"【脚数不超过{PIN_COUNT_STORE_MAX_SKIP}不计算】"
                f"project_code={model.project_code} refdes={model.refdes} 脚数={model.pin_count}"
            ),
        )
        return 0
    cache = build_fpgeom_from_footprint(model, source_file=source_file)
    area = footprint_copper_area_mils2(model)
    matches = query_similar(
        cache,
        conn,
        top_k=-1,
        min_iou=PENDING_IOU_MIN,
        exclude_self=True,
        copper_area=area,
        exclude_project_code=model.project_code,
        parallel=parallel,
    )
    return replace_pending_iou(conn, model.project_code, model.refdes, matches)


def save_pending_component_json(
    conn,
    *,
    project_code: str,
    refdes: str,
    pin_count: int,
    json_text: str,
    source_file: str = "",
) -> bool:
    """待完成板的器件 JSON 写入 pending_json，不进 MinIO。同一项目+位号后来的覆盖先前的。

    脚数不超过 6 时不写入，并清掉这个位号已有的待完成 JSON 和预计算。返回 False。
    """
    code = str(project_code or "").strip()
    ref = normalize_refdes(refdes)
    if not code or not ref:
        raise ValueError("pending_json 需要 project_code 和 refdes")
    if pin_count_too_small(pin_count):
        drop_small_pin_pending(conn, code, ref)
        from agent_log import log_line

        log_line(
            "二开入库",
            result=(
                f"【脚数不超过{PIN_COUNT_STORE_MAX_SKIP}不入库】"
                f"project_code={code} refdes={ref} 脚数={pin_count} 表=pending_json"
            ),
        )
        return False
    text = str(json_text or "")
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO pending_json (project_code, refdes, pin_count, source_file, json_text)
            VALUES (%s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                pin_count=VALUES(pin_count),
                source_file=VALUES(source_file),
                json_text=VALUES(json_text),
                created_at=CURRENT_TIMESTAMP
            """,
            (code, ref, int(pin_count or 0), source_file or None, text),
        )
        _exec_ignore_missing_table(
            cur,
            f"DELETE FROM pending_json_block WHERE project_code=%s AND {_REFDES_EQ}",
            (code, ref),
        )
    conn.commit()
    return True


def mark_pending_json_block(conn, project_code: str, refdes: str, reason: str = "") -> None:
    """IoU 算失败的项目位号记入异常表。下一轮脚本不再取这条 JSON。"""
    code = str(project_code or "").strip()
    ref = normalize_refdes(refdes)
    if not code or not ref:
        return
    text = str(reason or "").replace("\n", " ").strip()[:500]
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO pending_json_block (project_code, refdes, reason)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE
                reason=VALUES(reason),
                updated_at=CURRENT_TIMESTAMP
            """,
            (code, ref, text or None),
        )
    conn.commit()


def list_pending_json_for_iou(conn) -> tuple[list[dict[str, Any]], int, int]:
    """返回 (需要计算的行, 已删掉的不超过 6 脚条数, 异常表里跳过的条数)。"""
    jobs: list[dict[str, Any]] = []
    removed = 0
    with conn.cursor() as cur:
        _exec_ignore_missing_table(
            cur,
            """
            DELETE pio FROM pending_iou pio
            INNER JOIN pending_json pj
              ON pj.project_code = pio.project_code
             AND REPLACE(pj.refdes, '@', '') = REPLACE(pio.refdes, '@', '')
            WHERE pj.pin_count <= %s
            """,
            (PIN_COUNT_STORE_MAX_SKIP,),
        )
        cur.execute(
            "DELETE FROM pending_json WHERE pin_count <= %s",
            (PIN_COUNT_STORE_MAX_SKIP,),
        )
        removed = int(cur.rowcount or 0)
        conn.commit()
        blocked = 0
        try:
            cur.execute(
                """
                SELECT COUNT(*)
                FROM pending_json pj
                WHERE pj.pin_count > %s
                  AND EXISTS (
                    SELECT 1 FROM pending_json_block b
                    WHERE b.project_code = pj.project_code
                      AND REPLACE(b.refdes, '@', '') = REPLACE(pj.refdes, '@', '')
                  )
                """,
                (PIN_COUNT_STORE_MAX_SKIP,),
            )
            blocked = int(cur.fetchone()[0])
            cur.execute(
                """
                SELECT pj.id, pj.project_code, pj.refdes, pj.pin_count, pj.source_file, pj.json_text
                FROM pending_json pj
                WHERE pj.pin_count > %s
                  AND NOT EXISTS (
                    SELECT 1 FROM pending_json_block b
                    WHERE b.project_code = pj.project_code
                      AND REPLACE(b.refdes, '@', '') = REPLACE(pj.refdes, '@', '')
                  )
                ORDER BY pj.id
                """,
                (PIN_COUNT_STORE_MAX_SKIP,),
            )
        except Exception as exc:
            code = exc.args[0] if getattr(exc, "args", None) else None
            if code != 1146:
                raise
            conn.rollback()
            blocked = 0
            cur.execute(
                """
                SELECT id, project_code, refdes, pin_count, source_file, json_text
                FROM pending_json
                WHERE pin_count > %s
                ORDER BY id
                """,
                (PIN_COUNT_STORE_MAX_SKIP,),
            )
        for row_id, project_code, refdes, pin_count, source_file, json_text in cur.fetchall():
            if isinstance(json_text, (bytes, bytearray)):
                json_text = bytes(json_text).decode("utf-8")
            jobs.append(
                {
                    "id": int(row_id),
                    "project_code": project_code or "",
                    "refdes": refdes or "",
                    "pin_count": int(pin_count or 0),
                    "source_file": source_file or "",
                    "json_text": json_text or "",
                }
            )
    return jobs, removed, blocked


def delete_pending_component_json(conn, row_id: int) -> None:
    """一个位号的 IoU 算完后，从 pending_json 删掉这一行。"""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM pending_json WHERE id=%s", (int(row_id),))
    conn.commit()


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
# WARN_PDF        PDF 已入库但仅单器件校准等告警（非致命）

ERROR_CODES = {
    "SCRIPT": "ERR_SCRIPT",
    "JSON_FORMAT": "ERR_JSON_FORMAT",
    "JSON_FIELD": "ERR_JSON_FIELD",
    "FPGEOM": "ERR_FPGEOM",
    "PDF": "ERR_PDF",
    "WARN_PDF": "WARN_PDF",
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

    from mysql_diag import db_transaction

    def _write(detail_text: str) -> None:
        with db_transaction(
            conn,
            operation="record_failure",
            detail=f"project_code={code} refdes={ref} code={error_code}",
            sql_hint="ingest_failure",
        ):
            with conn.cursor() as cur:
                created = _take_created_at(
                    cur,
                    "ingest_failure",
                    f"project_code = %s AND {_REFDES_EQ}",
                    (code, ref),
                )
                if created is not None:
                    cur.execute(
                        """
                        INSERT INTO ingest_failure (project_code, refdes, error_code, detail, created_at)
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        (code, ref, error_code, detail_text, created),
                    )
                else:
                    cur.execute(
                        """
                        INSERT INTO ingest_failure (project_code, refdes, error_code, detail)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (code, ref, error_code, detail_text),
                    )

    try:
        _write(text)
    except Exception as exc:
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
    finish_time: Any = None,
) -> None:
    from mysql_diag import db_transaction
    from top_side_util import parse_top_side, top_side_to_db

    scale = float(pdf_scale or 0)
    side = top_side_to_db(parse_top_side(top_side))
    code = str(project_code or "").strip()
    from minio_store import object_key_of, save_object
    from top_side_util import top_side_label

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
    blob = save_object(
        content,
        finish_time=finish_time,
        filename=f"{code}_{top_side_label(bool(side))}.pdf",
        content_type="application/pdf",
        old_key=old_key,
    )
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
                    (code, side, file_name, blob, scale, created),
                )
            else:
                cur.execute(
                    """
                    INSERT INTO project_pdf (project_code, top_side, file_name, file_content, pdf_scale)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (code, side, file_name, blob, scale),
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
        "file_content": _materialize_blob(row[4]),
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
            WHERE project_code = %s AND top_side = %s
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
            text = resolve_stored_bytes(key).decode("utf-8") if key else out["source_json"]
            if isinstance(text, bytes):
                text = text.decode("utf-8")
            out["source_json"] = json.loads(text)
        except Exception:
            pass
    return out


def _fetch_footprint_storage_row(conn, project_code: str, refdes: str) -> dict[str, Any] | None:
    """读取 footprint_library 一行（用于按唯一键删后重插）。"""
    code = str(project_code or "").strip()
    ref = normalize_refdes(refdes)
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT name, pin_count, copper_area, units,
                   bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y,
                   fpgeom_version, fpgeom,
                   project_code, refdes,
                   layers, devicetype, packagesymbol, value,
                   pcb_x, pcb_y, top_side,
                   source_json, warnings, geom_islands,
                   pdf_refdes_x, pdf_refdes_y
            FROM footprint_library
            WHERE project_code = %s AND {_REFDES_EQ}
            LIMIT 1
            """,
            (code, ref),
        )
        row = cur.fetchone()
    if not row:
        return None
    keys = [
        "name",
        "pin_count",
        "copper_area",
        "units",
        "bbox_min_x",
        "bbox_min_y",
        "bbox_max_x",
        "bbox_max_y",
        "fpgeom_version",
        "fpgeom",
        "project_code",
        "refdes",
        "layers",
        "devicetype",
        "packagesymbol",
        "value",
        "pcb_x",
        "pcb_y",
        "top_side",
        "source_json",
        "warnings",
        "geom_islands",
        "pdf_refdes_x",
        "pdf_refdes_y",
    ]
    return dict(zip(keys, row))


def update_component_pdf_refdes(
    conn,
    *,
    project_code: str,
    refdes: str,
    pdf_refdes_x: float | None,
    pdf_refdes_y: float | None,
) -> None:
    """只更新丝印坐标和更新时间。创建时间保持第一次写入，不删行、不动 MinIO。"""
    from mysql_diag import db_transaction

    if pdf_refdes_x is None or pdf_refdes_y is None:
        return

    code = str(project_code or "").strip()
    ref = normalize_refdes(refdes)
    if not code or not ref:
        return
    with db_transaction(
        conn,
        operation="update_component_pdf_refdes",
        detail=f"project_code={code} refdes={ref}",
        sql_hint="footprint_library UPDATE pdf_refdes",
    ):
        with conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE footprint_library
                SET pdf_refdes_x=%s, pdf_refdes_y=%s, updated_at=CURRENT_TIMESTAMP
                WHERE project_code=%s AND {_REFDES_EQ}
                """,
                (pdf_refdes_x, pdf_refdes_y, code, ref),
            )


def update_project_pdf_scale(conn, project_code: str, top_side: bool, pdf_scale: float) -> None:
    """只更新比例和更新时间。创建时间、PDF 路径都不动。"""
    from mysql_diag import db_transaction
    from top_side_util import parse_top_side, top_side_to_db

    code = str(project_code or "").strip()
    side = top_side_to_db(parse_top_side(top_side))
    scale = float(pdf_scale or 0)
    if not code or scale <= 0:
        return
    with db_transaction(
        conn,
        operation="update_project_pdf_scale",
        detail=f"project_code={code} top_side={side} pdf_scale={scale:.12g}",
        sql_hint="project_pdf UPDATE pdf_scale",
    ):
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE project_pdf
                SET pdf_scale=%s, updated_at=CURRENT_TIMESTAMP
                WHERE project_code=%s AND top_side=%s
                """,
                (scale, code, side),
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
    """按 job_key 主键：无则 INSERT；有则 DELETE 后 INSERT。"""
    from mysql_diag import db_transaction

    key = str(job_key or "").strip()
    if not key:
        raise ValueError("job_key 不能为空")
    done = str(done_through)[:10] if done_through else None
    with db_transaction(
        conn,
        operation="upsert_oa_sync_done_through",
        detail=f"job_key={key} done_through={done}",
        sql_hint="oa_sync_progress",
    ):
        with conn.cursor() as cur:
            existed = _unique_row_exists(cur, "oa_sync_progress", "job_key = %s", (key,))
            if existed:
                cur.execute("DELETE FROM oa_sync_progress WHERE job_key = %s", (key,))
            cur.execute(
                "INSERT INTO oa_sync_progress (job_key, done_through) VALUES (%s, %s)",
                (key, done),
            )


def _migrate_pdf_refdes_backfill_log(conn) -> None:
    """确保 pdf_refdes_backfill_log 表存在。"""
    with conn.cursor() as cur:
        cur.execute("SHOW TABLES LIKE 'pdf_refdes_backfill_log'")
        if cur.fetchone():
            return
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS pdf_refdes_backfill_log (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                batch_key VARCHAR(32) NOT NULL,
                project_code VARCHAR(128) NOT NULL DEFAULT '',
                refdes VARCHAR(64) NOT NULL DEFAULT '',
                top_side TINYINT(1) NULL,
                pending_total INT NULL,
                ok_count INT NULL,
                fail_count INT NULL,
                row_kind VARCHAR(16) NOT NULL DEFAULT 'detail',
                detail VARCHAR(512) NULL,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                KEY idx_backfill_batch (batch_key),
                KEY idx_backfill_project (project_code, refdes)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
        )
    conn.commit()


def list_components_missing_pdf_refdes(
    conn,
    *,
    limit: int | None = None,
    force: bool = False,
) -> list[dict[str, Any]]:
    """列出要补刷坐标的位号。默认只取坐标为空的；force 时已有坐标的也重新刷。"""
    sql = """
        SELECT f.project_code, f.refdes, f.top_side
        FROM footprint_library f
        INNER JOIN project_pdf p
            ON p.project_code = f.project_code AND p.top_side = f.top_side
        WHERE 1=1
    """
    if not force:
        sql += " AND (f.pdf_refdes_x IS NULL OR f.pdf_refdes_y IS NULL)"
    sql += " ORDER BY f.project_code, f.refdes"
    if limit is not None and limit > 0:
        sql += f" LIMIT {int(limit)}"
    with conn.cursor() as cur:
        cur.execute(sql)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def insert_pdf_refdes_backfill_log(
    conn,
    *,
    batch_key: str,
    row_kind: str,
    project_code: str = "",
    refdes: str = "",
    top_side: bool | None = None,
    pending_total: int | None = None,
    ok_count: int | None = None,
    fail_count: int | None = None,
    detail: str | None = None,
) -> None:
    """写入 PDF 位号坐标补刷日志（summary 或明细）。"""
    from mysql_diag import db_transaction
    from top_side_util import top_side_to_db

    ts = top_side_to_db(top_side) if top_side is not None else None
    with db_transaction(
        conn,
        operation="insert_pdf_refdes_backfill_log",
        detail=f"batch={batch_key} project={project_code} refdes={refdes} kind={row_kind}",
        sql_hint="pdf_refdes_backfill_log",
    ):
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO pdf_refdes_backfill_log (
                    batch_key, project_code, refdes, top_side,
                    pending_total, ok_count, fail_count, row_kind, detail
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    batch_key,
                    project_code or "",
                    normalize_refdes(refdes),
                    ts,
                    pending_total,
                    ok_count,
                    fail_count,
                    row_kind,
                    (detail or "")[:512] if detail else None,
                ),
            )


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


def fetch_process_record(conn, project_code: str) -> dict[str, Any] | None:
    code = str(project_code or "").strip()
    if not code:
        return None
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT oa_info_ok, oa_info_note, oa_download_ok, oa_download_note,
                   oa_brd_ok, oa_brd_note, parse_ok, parse_note
            FROM `process` WHERE project_code = %s LIMIT 1
            """,
            (code,),
        )
        row = cur.fetchone()
    if not row:
        return None
    keys = (
        "oa_info_ok",
        "oa_info_note",
        "oa_download_ok",
        "oa_download_note",
        "oa_brd_ok",
        "oa_brd_note",
        "parse_ok",
        "parse_note",
    )
    return dict(zip(keys, row))


def upsert_process_record(
    conn,
    project_code: str,
    *,
    oa_info_ok: int | None = None,
    oa_info_note: str | None = None,
    oa_download_ok: int | None = None,
    oa_download_note: str | None = None,
    oa_brd_ok: int | None = None,
    oa_brd_note: str | None = None,
    parse_ok: int | None = None,
    parse_note: str | None = None,
) -> None:
    """按 project_code 唯一键：无则 INSERT；有则 DELETE 后 INSERT（合并旧流程字段）。创建时间沿用第一次。"""
    from mysql_diag import db_transaction

    code = str(project_code or "").strip()
    if not code:
        return
    base: dict[str, Any] = {
        "project_code": code,
        "oa_info_ok": None,
        "oa_info_note": None,
        "oa_download_ok": None,
        "oa_download_note": None,
        "oa_brd_ok": None,
        "oa_brd_note": None,
        "parse_ok": None,
        "parse_note": None,
    }
    existing = fetch_process_record(conn, code)
    if existing:
        base.update(existing)
    for key, val in (
        ("oa_info_ok", oa_info_ok),
        ("oa_info_note", oa_info_note),
        ("oa_download_ok", oa_download_ok),
        ("oa_download_note", oa_download_note),
        ("oa_brd_ok", oa_brd_ok),
        ("oa_brd_note", oa_brd_note),
        ("parse_ok", parse_ok),
        ("parse_note", parse_note),
    ):
        if val is not None:
            if key.endswith("_note") and isinstance(val, str):
                text = " ".join(val.split())
                base[key] = text if len(text) <= 512 else text[:511] + "…"
            else:
                base[key] = val
    cols = [
        "project_code",
        "oa_info_ok",
        "oa_info_note",
        "oa_download_ok",
        "oa_download_note",
        "oa_brd_ok",
        "oa_brd_note",
        "parse_ok",
        "parse_note",
    ]
    with db_transaction(
        conn,
        operation="upsert_process_record",
        detail=f"project_code={code}",
        sql_hint="process",
    ):
        with conn.cursor() as cur:
            created = _take_created_at(cur, "`process`", "project_code = %s", (code,))
            insert_cols = list(cols)
            insert_vals = [base[c] for c in cols]
            if created is not None:
                insert_cols.append("created_at")
                insert_vals.append(created)
            holders = ", ".join(["%s"] * len(insert_cols))
            cur.execute(
                f"INSERT INTO `process` ({', '.join(insert_cols)}) VALUES ({holders})",
                insert_vals,
            )


def delete_project_components(conn, project_code: str) -> int:
    """
    按 project_code 整表批量删除（仅运维脚本用手动清理）。
    正常入库应走各表唯一键上的「有则删后插、无则插」，勿调用本函数。
    """
    with conn.cursor() as cur:
        cur.execute("DELETE FROM footprint_library WHERE project_code = %s", (project_code,))
        deleted = cur.rowcount
        cur.execute("DELETE FROM project_pdf WHERE project_code = %s", (project_code,))
        cur.execute("DELETE FROM project_profile WHERE project_code = %s", (project_code,))
        cur.execute("DELETE FROM ingest_failure WHERE project_code = %s", (project_code,))
    from mysql_diag import commit_or_log

    commit_or_log(
        conn,
        operation="delete_project_components",
        detail=f"project_code={project_code} deleted_rows={deleted}",
        sql_hint="DELETE footprint_library/project_pdf/...",
    )
    return int(deleted or 0)

