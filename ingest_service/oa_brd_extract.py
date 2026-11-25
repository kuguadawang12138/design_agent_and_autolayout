# -*- coding: utf-8 -*-
"""从 OA 接口③返回的 zip/嵌套压缩包或裸 BRD 中解析出 .brd 文件（对齐 DAY002ori oa_client）。"""

from __future__ import annotations

import io
import os
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Any

ARCHIVE_SUFFIXES = (
    ".zip",
    ".7z",
    ".rar",
    ".tar",
    ".tgz",
    ".tar.gz",
    ".tar.bz2",
    ".tbz2",
    ".gz",
    ".bz2",
)

# 分卷：zip 的 .z01/.z02…，以及 .001/.002…
_SPLIT_VOLUME = __import__("re").compile(r"\.(z\d{2,}|\d{3})$", __import__("re").IGNORECASE)


def _brief_errors(errors: list[str], limit: int = 240) -> str:
    """失败原因压短，写进 process 表时不超过 512 字。"""
    parts: list[str] = []
    for item in errors[:4]:
        text = " ".join(str(item).split())
        if len(text) > 60:
            text = text[:59] + "…"
        if text:
            parts.append(text)
    if len(errors) > 4:
        parts.append(f"另有{len(errors) - 4}条")
    text = "；".join(parts)
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _safe_name(name: str | None, fallback: str) -> str:
    text = (name or fallback).replace("\\", "/").rsplit("/", 1)[-1].strip()
    text = "".join("-" if ch in '<>:"|?*' else ch for ch in text)
    return text or fallback


def _is_split_volume_name(name: str | None) -> bool:
    low = (name or "").lower()
    return bool(_SPLIT_VOLUME.search(low))


def _looks_like_archive_name(name: str | None) -> bool:
    low = (name or "").lower()
    if low.endswith(ARCHIVE_SUFFIXES) or _is_split_volume_name(low):
        return True
    # 无扩展名但文件名以 _zip 结尾
    stem = low.rsplit("/", 1)[-1]
    return stem.endswith("_zip") or stem.endswith("-zip")


def _archive_magic(content: bytes | None) -> str | None:
    if not content or len(content) < 2:
        return None
    if content[:2] == b"PK":
        return "zip"
    if content[:6] == b"7z\xbc\xaf'\x1c":
        return "7z"
    if content[:4] == b"Rar!":
        return "rar"
    if content[:2] == b"\x1f\x8b":
        return "gzip"
    if len(content) >= 3 and content[:3] == b"BZh":
        return "bzip2"
    return None


def _archive_hint(name: str | None, content_type: str | None = None, content: bytes | None = None) -> bool:
    low_type = (content_type or "").lower()
    if _looks_like_archive_name(name):
        return True
    if any(k in low_type for k in ("zip", "rar", "7z", "tar", "compressed", "gzip")):
        return True
    return _archive_magic(content) is not None


def _write_file_safe(root: Path, member_name: str, content: bytes) -> Path | None:
    if not content:
        return None
    root = root.resolve()
    clean = member_name.replace("\\", "/").lstrip("/")
    if not clean:
        return None
    target = (root / clean).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return target


def _extract_zip_bytes(content: bytes, dest_dir: Path) -> list[Path]:
    written: list[Path] = []
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            with zf.open(info) as fh:
                path = _write_file_safe(dest_dir, info.filename, fh.read())
            if path:
                written.append(path)
    return written


def _extract_tar_bytes(content: bytes, dest_dir: Path) -> list[Path]:
    written: list[Path] = []
    with tarfile.open(fileobj=io.BytesIO(content), mode="r:*") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            fh = tf.extractfile(member)
            if fh is None:
                continue
            path = _write_file_safe(dest_dir, member.name, fh.read())
            if path:
                written.append(path)
    return written


def _extract_7z_bytes(content: bytes, filename: str, dest_dir: Path) -> list[Path]:
    """7z / 分卷：优先 py7zr，否则调用本机 7z。RAR 不走 py7zr。"""
    dest_dir.mkdir(parents=True, exist_ok=True)
    if str(filename or "").lower().endswith(".rar") or _archive_magic(content) == "rar":
        return _extract_via_7z_cli(content, filename, dest_dir)
    try:
        import py7zr

        with py7zr.SevenZipFile(io.BytesIO(content), mode="r") as archive:
            archive.extractall(path=dest_dir)
        return [p for p in dest_dir.rglob("*") if p.is_file()]
    except ImportError:
        pass
    except Exception:
        if not _is_split_volume_name(filename):
            raise
    return _extract_via_7z_cli(content, filename, dest_dir)


def _windows_program_dirs() -> list[Path]:
    """Windows 默认安装目录。服务进程的 PATH 里经常没有 7-Zip / WinRAR。"""
    roots: list[Path] = []
    seen: set[str] = set()
    for key in ("ProgramW6432", "ProgramFiles", "ProgramFiles(x86)"):
        value = os.environ.get(key)
        if value and value not in seen:
            seen.add(value)
            roots.append(Path(value))
    return roots


def _first_existing(paths: list[Path]) -> str | None:
    for path in paths:
        if path.is_file():
            return str(path)
    return None


def _find_unrar() -> str | None:
    """先看 PATH，再看 WinRAR 默认目录里的 UnRAR.exe / Rar.exe。"""
    for name in ("unrar", "unrar-free", "UnRAR"):
        found = shutil.which(name)
        if found:
            return found
    extras: list[Path] = []
    for root in _windows_program_dirs():
        extras.extend(
            [
                root / "WinRAR" / "UnRAR.exe",
                root / "WinRAR" / "Unrar.exe",
                root / "WinRAR" / "Rar.exe",
            ]
        )
    return _first_existing(extras)


def _find_7z() -> str | None:
    """先看 PATH，再看 7-Zip 默认目录。本机常见是 C:\\Program Files\\7-Zip\\7z.exe。"""
    for name in ("7z", "7za", "7zz"):
        found = shutil.which(name)
        if found:
            return found
    return _first_existing([root / "7-Zip" / "7z.exe" for root in _windows_program_dirs()])


def _extract_via_7z_cli(content: bytes, filename: str, dest_dir: Path) -> list[Path]:
    import subprocess

    exe = _find_7z()
    if not exe:
        raise RuntimeError("未安装 py7zr，且找不到 7z/7zz 命令，也无法在 Program Files\\7-Zip 找到 7z.exe")
    dest_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(filename).suffix or ".bin"
    with tempfile.NamedTemporaryFile(prefix="oa_arc_", suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    try:
        proc = subprocess.run(
            [exe, "x", "-y", f"-o{dest_dir}", str(tmp_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()[:300]
            raise RuntimeError(detail or f"7z 退出码 {proc.returncode}")
    finally:
        tmp_path.unlink(missing_ok=True)
    return [p for p in dest_dir.rglob("*") if p.is_file()]


def _clear_extract_dir(dest_dir: Path) -> None:
    """换下一种解压方式前清掉上一次没解完的文件。"""
    if not dest_dir.exists():
        return
    for child in dest_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


def _list_extracted_files(dest_dir: Path) -> list[Path]:
    return [p for p in dest_dir.rglob("*") if p.is_file()]


def _extract_rar_via_rarfile(archive_path: Path, dest_dir: Path, tool: str) -> list[Path]:
    """rarfile 自己不解 RAR，只是去调系统里的 unrar / unar / bsdtar。"""
    import rarfile

    rarfile.UNRAR_TOOL = tool
    with rarfile.RarFile(str(archive_path)) as rf:
        rf.extractall(path=str(dest_dir))
    return _list_extracted_files(dest_dir)


def _run_extract_cli(argv: list[str], dest_dir: Path) -> list[Path]:
    import subprocess

    proc = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()[:300]
        raise RuntimeError(detail or f"{argv[0]} 退出码 {proc.returncode}")
    return _list_extracted_files(dest_dir)


def _extract_rar_bytes(content: bytes, filename: str, dest_dir: Path) -> list[Path]:
    """
    RAR 不能用 py7zr。
    依次试：rarfile、unrar、unar、bsdtar、7z/7zz。
    Windows 上 PATH 没有时，再找 Program Files\\WinRAR 和 Program Files\\7-Zip。
    """
    from agent_log import log_line

    dest_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(filename).suffix or ".rar"
    with tempfile.NamedTemporaryFile(prefix="oa_rar_", suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        archive_path = Path(tmp.name)
    errors: list[str] = []
    try:
        unrar = _find_unrar()
        unar = shutil.which("unar")
        bsdtar = shutil.which("bsdtar")
        seven = _find_7z()
        steps: list[tuple[str, Any]] = []
        for tool_name, tool_path in (("unrar", unrar), ("unar", unar), ("bsdtar", bsdtar)):
            if tool_path:
                steps.append(
                    (
                        f"rarfile/{Path(tool_path).name}",
                        lambda p=tool_path: _extract_rar_via_rarfile(archive_path, dest_dir, p),
                    )
                )
        if unrar:
            steps.append(
                (
                    Path(unrar).name,
                    lambda exe=unrar: _run_extract_cli(
                        [exe, "x", "-o+", str(archive_path), str(dest_dir) + "/"],
                        dest_dir,
                    ),
                )
            )
        if unar:
            steps.append(
                (
                    "unar",
                    lambda exe=unar: _run_extract_cli(
                        [exe, "-force-overwrite", "-output-directory", str(dest_dir), str(archive_path)],
                        dest_dir,
                    ),
                )
            )
        if bsdtar:
            steps.append(
                (
                    "bsdtar",
                    lambda exe=bsdtar: _run_extract_cli(
                        [exe, "-xf", str(archive_path), "-C", str(dest_dir)],
                        dest_dir,
                    ),
                )
            )
        if seven:
            steps.append((f"7z:{seven}", lambda: _extract_via_7z_cli(content, filename, dest_dir)))
        if not steps:
            log_line(
                "BRD解压",
                result="【没有 RAR 工具】已查 PATH 的 unrar/unar/bsdtar/7z，以及 WinRAR、7-Zip 默认目录",
                level="warning",
            )
            raise RuntimeError("没有可解 RAR 的程序")
        for name, fn in steps:
            _clear_extract_dir(dest_dir)
            try:
                written = fn()
                if written:
                    log_line(
                        "BRD解压",
                        result=f"【RAR解压成功】file={filename} 工具={name} 文件数={len(written)}",
                    )
                    return written
                errors.append(f"{name}=解出 0 个文件")
            except ImportError:
                errors.append(f"{name}=未安装 rarfile")
            except Exception as exc:
                errors.append(f"{name}={exc}")
        raise RuntimeError("RAR 解压失败：" + _brief_errors(errors))
    finally:
        archive_path.unlink(missing_ok=True)


def _preferred_archive_kind(filename: str, content: bytes) -> str | None:
    """按后缀判断先试哪一种；后缀认不出再用文件头。"""
    low = (filename or "").lower()
    if low.endswith((".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar")):
        return "tar"
    if low.endswith(".zip") or low.endswith("_zip") or low.endswith("-zip"):
        return "zip"
    if low.endswith(".7z"):
        return "7z"
    if low.endswith(".rar"):
        return "rar"
    if _is_split_volume_name(low):
        return "7z"
    magic = _archive_magic(content)
    if magic in ("gzip", "bzip2"):
        return "tar"
    if magic in ("zip", "7z", "rar"):
        return magic
    return None


def _extract_archive_bytes(content: bytes, filename: str, dest_dir: Path) -> list[Path]:
    """先按后缀解一种，失败后再把 zip / 7z / rar / tar 都试一遍。"""
    from agent_log import log_line

    if not content:
        log_line("BRD解压", result=f"【跳过解压】file={filename} 大小为 0KB", level="warning")
        raise ValueError(f"文件大小为 0KB，不进行解压：{filename}")

    catalog: list[tuple[str, Any]] = [
        ("zip", lambda: _extract_zip_bytes(content, dest_dir)),
        ("7z", lambda: _extract_7z_bytes(content, filename, dest_dir)),
        ("rar", lambda: _extract_rar_bytes(content, filename, dest_dir)),
        ("tar", lambda: _extract_tar_bytes(content, dest_dir)),
    ]
    prefer = _preferred_archive_kind(filename, content)
    if prefer:
        attempts = [item for item in catalog if item[0] == prefer]
        attempts.extend(item for item in catalog if item[0] != prefer)
    else:
        attempts = catalog
    log_line(
        "BRD解压",
        result=f"【解压顺序】file={filename} 后缀优先={prefer or '未知'} 随后={','.join(name for name, _fn in attempts)}",
    )
    errors: list[str] = []
    for kind, fn in attempts:
        try:
            written = fn()
            if written:
                log_line("BRD解压", result=f"【解压成功】file={filename} 方式={kind} 文件数={len(written)}")
                return written
            errors.append(f"{kind}=解出 0 个文件")
        except Exception as exc:
            errors.append(f"{kind}={exc}")
    split_note = ""
    if _is_split_volume_name(filename):
        split_note = "；这是分卷（z01/001 等），单个分卷通常解不出内容，需要同一套全部分卷"
    msg = f"压缩包无法解压：{Path(filename).name}{split_note}；{_brief_errors(errors)}"
    log_line("BRD解压", result=f"【压缩包无法解压】{msg}", level="warning")
    raise ValueError(msg)


def _unpack_nested_archives(root: Path, max_depth: int = 8) -> list[Path]:
    """递归解压目录内嵌套的 zip/tar，直到没有新的压缩包。"""
    written: list[Path] = []
    seen: set[Path] = set()
    for _ in range(max_depth):
        pending: list[Path] = []
        for p in root.rglob("*"):
            if not p.is_file() or p.resolve() in seen:
                continue
            if p.stat().st_size == 0:
                from agent_log import log_line

                log_line("BRD解压", result=f"【跳过解压】file={p.name} 大小为 0KB", level="warning")
                seen.add(p.resolve())
                continue
            head = p.read_bytes()[:8]
            if _archive_hint(p.name, content=head):
                pending.append(p)
        if not pending:
            break
        for path in pending:
            seen.add(path.resolve())
            out_dir = path.with_suffix("")
            out_dir.mkdir(parents=True, exist_ok=True)
            try:
                written.extend(_extract_archive_bytes(path.read_bytes(), path.name, out_dir))
            except Exception as exc:
                from agent_log import log_line

                log_line(
                    "BRD解压",
                    result=f"【嵌套压缩包跳过】file={path.name} {exc}",
                    level="warning",
                )
                continue
    return written


def _find_brd_file(root: Path, project_code: str | None = None) -> Path | None:
    root = Path(root)
    code = str(project_code or "").strip().lower()
    candidates = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".brd"]
    if not candidates:
        return None

    def rank(path: Path) -> tuple[int, int, int, str]:
        stem = path.stem.lower()
        exact = 0 if code and stem == code else 1
        contains = 0 if code and code in stem else 1
        return (exact, contains, len(path.relative_to(root).parts), path.as_posix().lower())

    return sorted(candidates, key=rank)[0]


def _store_payload_to_dir(
    content: bytes,
    filename: str | None,
    dest_dir: Path,
    project_code: str,
    content_type: str | None = None,
) -> list[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = _safe_name(filename, f"{project_code}.brd")
    written: list[Path] = []
    if _archive_hint(name, content_type, content):
        archive_dir = dest_dir / Path(name).stem
        archive_dir.mkdir(parents=True, exist_ok=True)
        written.extend(_extract_archive_bytes(content, name, archive_dir))
        written.extend(_unpack_nested_archives(archive_dir))
        return written

    if not name.lower().endswith(".brd"):
        stem = Path(name).stem or project_code
        name = f"{stem}.brd"
    path = dest_dir / name
    path.write_bytes(content)
    written.append(path)
    return written


def resolve_brd_path(
    project_code: str,
    content: bytes,
    filename: str | None = None,
    *,
    content_type: str | None = None,
    work_dir: Path | None = None,
) -> Path:
    """
    在 work_dir 下解包并找到 .brd；返回该文件路径（尚未重命名为项目编码）。
    """
    if not content:
        raise ValueError("文件大小为 0KB，不进行解压")
    code = str(project_code or "").strip()
    if not code:
        raise ValueError("project_code 不能为空")
    unpack_root = Path(work_dir) if work_dir else Path(".")
    unpack_root.mkdir(parents=True, exist_ok=True)
    files = _store_payload_to_dir(content, filename, unpack_root, code, content_type)
    brd = _find_brd_file(unpack_root, code)
    if brd is None:
        sample = "、".join(p.name for p in list(files)[:3] if p.exists())
        tail = f"，如 {sample}" if sample else ""
        raise ValueError(f"未找到.brd，解出{len(files)}个文件{tail}")
    return brd


def stage_brd_bytes_to_dir(
    project_code: str,
    content: bytes,
    brd_dir: Path,
    filename: str | None = None,
    *,
    content_type: str | None = None,
    keep_unpack: bool = False,
) -> Path:
    """
    将 OA 下载结果落盘为 {project_code}.brd（位于 brd_dir 根目录）。
    压缩包在系统临时目录解压，只保留 .brd，解压中间文件立即删除。
    """
    del keep_unpack  # 始终不保留解压目录
    code = str(project_code or "").strip()
    brd_dir = Path(brd_dir)
    brd_dir.mkdir(parents=True, exist_ok=True)
    target = brd_dir / f"{code}.brd"
    legacy_unpack = brd_dir / "_oa_unpack"
    if legacy_unpack.exists():
        shutil.rmtree(legacy_unpack, ignore_errors=True)
    with tempfile.TemporaryDirectory(prefix="oa_brd_unpack_") as tmp:
        resolved = resolve_brd_path(
            code,
            content,
            filename,
            content_type=content_type,
            work_dir=Path(tmp),
        )
        shutil.copy2(resolved, target)
    return target
