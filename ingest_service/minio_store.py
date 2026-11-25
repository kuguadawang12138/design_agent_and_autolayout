# -*- coding: utf-8 -*-
"""MinIO 对象存储：库内 fpgeom / source_json / file_content 只存对象路径，内容放桶里。"""

from __future__ import annotations

import io
import re
from datetime import date, datetime
from typing import Any

from agent_log import log_line

_KEY_RE = re.compile(r"^\d{4}/\d{2}/\d{2}/.+\.(fpgeom|json|pdf)$", re.IGNORECASE)


def get_minio_settings() -> dict[str, Any]:
    from settings import get_minio_config

    return get_minio_config()


def minio_enabled() -> bool:
    cfg = get_minio_settings()
    return bool(cfg.get("enabled"))


def build_object_key(finish_time: Any, filename: str) -> str:
    """按板子完成时间分年/月/日目录，文件名用唯一键。"""
    day = _finish_date(finish_time)
    if finish_time in (None, ""):
        log_line(
            "MinIO",
            result=f"【完成时间为空】file={filename} 按当天 {day.isoformat()} 分目录",
            level="warning",
        )
    safe = str(filename or "file.bin").replace("\\", "_").replace("/", "_").strip() or "file.bin"
    return f"{day.year:04d}/{day.month:02d}/{day.day:02d}/{safe}"


def object_key_of(stored: bytes | str | None) -> str | None:
    """库里的值若已是对象路径则返回路径，二进制/JSON 原文返回 None。"""
    if stored is None:
        return None
    if isinstance(stored, bytes):
        if stored.startswith(b"FPGM") or stored.startswith(b"%PDF") or stored[:1] in (b"{", b"["):
            return None
        try:
            text = stored.decode("utf-8").strip()
        except UnicodeDecodeError:
            return None
    else:
        text = str(stored).strip()
        if text.startswith("{") or text.startswith("["):
            return None
    if _KEY_RE.match(text):
        return text
    return None


def delete_object(key: str) -> bool:
    """删除对象。不存在视为未删除。"""
    if not key:
        return False
    client, bucket = _client()
    from minio.error import S3Error

    try:
        client.stat_object(bucket, key)
    except S3Error as exc:
        if getattr(exc, "code", "") in ("NoSuchKey", "NoSuchObject"):
            return False
        raise
    client.remove_object(bucket, key)
    log_line("MinIO", result=f"【删除】bucket={bucket} key={key}")
    return True


def save_object(
    data: bytes,
    *,
    finish_time: Any,
    filename: str,
    content_type: str,
    old_key: str | None = None,
) -> str:
    """
    按唯一文件名写入。
    目标已存在，或库里还有另一条旧路径：先删除再上传；都没有则直接上传。
    """
    if not minio_enabled():
        raise RuntimeError("fpgeom、source_json、file_content 只存 MinIO 路径，请在 yaml 打开 minio.enabled")
    key = build_object_key(finish_time, filename)
    removed_old = False
    if old_key and old_key != key:
        removed_old = delete_object(old_key)
    replaced = delete_object(key)
    put_object(key, data, content_type)
    if replaced or removed_old:
        log_line("MinIO", result=f"【更新】先删除再上传 key={key} 旧路径={old_key or key}")
    else:
        log_line("MinIO", result=f"【新增】直接上传 key={key}")
    return key


def get_object_bytes(key: str) -> bytes:
    client, bucket = _client()
    resp = client.get_object(bucket, key)
    try:
        data = resp.read()
    finally:
        resp.close()
        resp.release_conn()
    return data


def resolve_stored_bytes(stored: bytes | str | None) -> bytes:
    """路径则从 MinIO 取回；否则视为库内仍保存的原始字节。"""
    key = object_key_of(stored)
    if not key:
        if isinstance(stored, str):
            return stored.encode("utf-8")
        return bytes(stored or b"")
    if not minio_enabled():
        raise RuntimeError(f"库内是 MinIO 路径 {key}，但 yaml minio.enabled 未打开")
    return get_object_bytes(key)


def put_object(key: str, data: bytes, content_type: str) -> str:
    client, bucket = _client()
    client.put_object(bucket, key, io.BytesIO(data), length=len(data), content_type=content_type)
    return key


def _finish_date(finish_time: Any) -> date:
    if isinstance(finish_time, datetime):
        return finish_time.date()
    if isinstance(finish_time, date):
        return finish_time
    text = str(finish_time or "").strip()
    if len(text) >= 10:
        try:
            return date.fromisoformat(text[:10])
        except ValueError:
            pass
    return date.today()


def _client():
    cfg = get_minio_settings()
    if not cfg.get("enabled"):
        raise RuntimeError("MinIO 未启用")
    from minio import Minio

    endpoint = str(cfg.get("endpoint") or "").strip()
    bucket = str(cfg.get("bucket") or "").strip()
    if not endpoint or not bucket:
        raise RuntimeError("minio.endpoint 或 minio.bucket 未配置")
    client = Minio(
        endpoint,
        access_key=str(cfg.get("access_key") or ""),
        secret_key=str(cfg.get("secret_key") or ""),
        secure=bool(cfg.get("secure", False)),
    )
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
        log_line("MinIO", result=f"【建桶】bucket={bucket}")
    return client, bucket
