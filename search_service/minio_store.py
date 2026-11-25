# -*- coding: utf-8 -*-
"""MinIO 对象存储：库内 fpgeom / source_json / file_content 只存对象路径，内容放桶里。"""

from __future__ import annotations

import io
import re
import threading
import time
from contextlib import contextmanager
from datetime import date, datetime
from typing import Any, Iterator

from agent_log import log_line

_KEY_RE = re.compile(r"^\d{4}/\d{2}/\d{2}/.+\.(fpgeom|json|pdf)$", re.IGNORECASE)
_fetch_tls = threading.local()


class _FetchSlice:
    """同一用途（形态 / JSON / PDF）的拉取累计。"""

    __slots__ = ("count", "nbytes", "ms", "slowest_ms", "slowest_key")

    def __init__(self) -> None:
        self.count = 0
        self.nbytes = 0
        self.ms = 0
        self.slowest_ms = 0
        self.slowest_key = ""

    def add(self, key: str, nbytes: int, ms: int) -> None:
        self.count += 1
        self.nbytes += nbytes
        self.ms += ms
        if ms >= self.slowest_ms:
            self.slowest_ms = ms
            self.slowest_key = key

    def format(self, purpose: str) -> str:
        if self.count <= 0:
            return f"用途={purpose} 对象数=0 字节=0 耗时=0ms"
        slow = f"{self.slowest_key} {self.slowest_ms}ms" if self.slowest_key else "-"
        return (
            f"用途={purpose} 对象数={self.count} 字节={self.nbytes} "
            f"耗时={self.ms}ms 最慢={slow}"
        )


class FetchAcc:
    """一批 MinIO 拉取的耗时。批量时不逐条打日志，由调用方打汇总。"""

    __slots__ = ("by_purpose",)

    def __init__(self) -> None:
        self.by_purpose: dict[str, _FetchSlice] = {}

    def add(self, purpose: str, key: str, nbytes: int, ms: int) -> None:
        name = purpose or "-"
        sl = self.by_purpose.get(name)
        if sl is None:
            sl = _FetchSlice()
            self.by_purpose[name] = sl
        sl.add(key, nbytes, ms)

    def slice(self, purpose: str) -> _FetchSlice:
        return self.by_purpose.get(purpose) or _FetchSlice()

    @property
    def count(self) -> int:
        return sum(sl.count for sl in self.by_purpose.values())

    @property
    def ms(self) -> int:
        return sum(sl.ms for sl in self.by_purpose.values())


@contextmanager
def measure_fetches() -> Iterator[FetchAcc]:
    """统计这段代码里的 MinIO 拉取。期间不逐条打印，避免形态候选刷屏。"""
    prev = getattr(_fetch_tls, "acc", None)
    acc = FetchAcc()
    _fetch_tls.acc = acc
    try:
        yield acc
    finally:
        _fetch_tls.acc = prev


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

    t0 = time.perf_counter()
    try:
        client.stat_object(bucket, key)
    except S3Error as exc:
        if getattr(exc, "code", "") in ("NoSuchKey", "NoSuchObject"):
            return False
        raise
    client.remove_object(bucket, key)
    ms = int((time.perf_counter() - t0) * 1000)
    log_line("MinIO", result=f"【删除】bucket={bucket} key={key} 耗时={ms}ms")
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
    t0 = time.perf_counter()
    key = build_object_key(finish_time, filename)
    removed_old = False
    if old_key and old_key != key:
        removed_old = delete_object(old_key)
    replaced = delete_object(key)
    put_object(key, data, content_type)
    ms = int((time.perf_counter() - t0) * 1000)
    if replaced or removed_old:
        log_line("MinIO", result=f"【更新】先删除再上传 key={key} 旧路径={old_key or key} 字节={len(data)} 耗时={ms}ms")
    else:
        log_line("MinIO", result=f"【新增】直接上传 key={key} 字节={len(data)} 耗时={ms}ms")
    return key


def get_object_bytes(key: str, *, purpose: str = "") -> bytes:
    """从 MinIO 读取对象。批量统计中只累计；单次拉取直接打一行耗时。"""
    client, bucket = _client()
    t0 = time.perf_counter()
    resp = client.get_object(bucket, key)
    try:
        data = resp.read()
    finally:
        resp.close()
        resp.release_conn()
    ms = int((time.perf_counter() - t0) * 1000)
    acc = getattr(_fetch_tls, "acc", None)
    if isinstance(acc, FetchAcc):
        acc.add(purpose, key, len(data), ms)
    else:
        log_line(
            "MinIO",
            result=f"【拉取】用途={purpose or '-'} key={key} 字节={len(data)} 耗时={ms}ms",
        )
    return data


def resolve_stored_bytes(stored: bytes | str | None, *, purpose: str = "") -> bytes:
    """路径则从 MinIO 取回；否则视为库内仍保存的原始字节。"""
    key = object_key_of(stored)
    if not key:
        if isinstance(stored, str):
            return stored.encode("utf-8")
        return bytes(stored or b"")
    if not minio_enabled():
        raise RuntimeError(f"库内是 MinIO 路径 {key}，但 yaml minio.enabled 未打开")
    return get_object_bytes(key, purpose=purpose)


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
