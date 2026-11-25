# -*- coding: utf-8 -*-
"""会话池：当前板 JSON 落盘 + 位号索引；关闭会话或 TTL 到期后释放。"""

from __future__ import annotations

import json
import re
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from agent_log import log_line
from json_fields import extract_meta, validate_component_json
from settings import get_agent_paths

_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_cleanup_lock = threading.Lock()


def sessions_root() -> Path:
    root = get_agent_paths()["sessions"]
    root.mkdir(parents=True, exist_ok=True)
    return root


def _session_dir(session_id: str) -> Path:
    value = str(session_id or "").strip()
    if not _SESSION_ID_RE.fullmatch(value):
        raise ValueError("session_id 格式无效")
    root = sessions_root().resolve()
    target = (root / value).resolve()
    if target.parent != root:
        raise ValueError("session_id 路径无效")
    return target


def _safe_name(name: str, used: set[str]) -> str:
    raw = (name or "part.json").replace("\\", "/").rsplit("/", 1)[-1]
    if not raw.lower().endswith(".json"):
        raw = f"{raw}.json"
    stem = raw[:-5]
    out = raw
    n = 1
    while out.lower() in used:
        out = f"{stem}_{n}.json"
        n += 1
    used.add(out.lower())
    return out


def _iter_objects(filename: str, content: bytes) -> list[tuple[str, dict, bytes]]:
    data = json.loads(content.decode("utf-8-sig"))
    if isinstance(data, dict) and isinstance(data.get("records"), list):
        rows = [x for x in data["records"] if isinstance(x, dict)]
        wrap = True
    elif isinstance(data, list):
        rows = [x for x in data if isinstance(x, dict)]
        wrap = True
    elif isinstance(data, dict):
        return [(filename or "part.json", data, content)]
    else:
        raise ValueError("JSON 根节点必须是对象或数组")
    if not rows:
        raise ValueError("JSON 里没有器件对象")
    out = []
    stem = (filename or "part.json").replace("\\", "/").rsplit("/", 1)[-1]
    if stem.lower().endswith(".json"):
        stem = stem[:-5]
    for i, row in enumerate(rows, start=1):
        meta = extract_meta(row, filename or "")
        ref = meta.get("refdes") or f"P{i}"
        name = f"{stem}_{ref}.json" if wrap else (filename or f"{ref}.json")
        raw = json.dumps(row, ensure_ascii=False).encode("utf-8")
        out.append((name, row, raw))
    return out


def validate_board_component(payload: dict, filename: str) -> tuple[bool, str, dict | None]:
    """当前板 JSON 格式校验（不入库、不做形态比对）。"""
    return validate_component_json(payload, filename)


def upload_board_json(files: list[tuple[str, bytes]]) -> dict:
    """接收当前板全部 JSON：仅格式校验 + 暂存会话，不做 IoU/向量检索。"""
    if not files:
        raise ValueError("请上传至少一个 .json 文件")
    parsed: list[tuple[str, str, dict, bytes]] = []
    errors: list[dict] = []
    for filename, content in files:
        name = filename or "part.json"
        if not name.lower().endswith(".json"):
            err = "仅接受 .json 文件"
            errors.append({"file": name, "error": err})
            log_line("板级JSON校验", detail=f"文件={name}", result=err)
            continue
        if not content:
            err = "空文件"
            errors.append({"file": name, "error": err})
            log_line("板级JSON校验", detail=f"文件={name}", result=err)
            continue
        try:
            objects = _iter_objects(name, content)
        except Exception as exc:
            err = str(exc)
            errors.append({"file": name, "error": err})
            log_line("板级JSON校验", detail=f"文件={name}", result=err)
            continue
        for stored_name, payload, raw in objects:
            ok, reason, meta = validate_board_component(payload, stored_name)
            if not ok:
                errors.append({"file": stored_name, "error": reason})
                log_line("板级JSON校验", detail=f"文件={stored_name}", result=reason)
                continue
            parsed.append((stored_name, name, meta, raw))
    if errors:
        summary = errors[0]["error"] if len(errors) == 1 else f"共 {len(errors)} 个文件格式不合格"
        return {
            "ok": False,
            "message": summary,
            "errors": errors,
            "valid_count": len(parsed),
            "fail_count": len(errors),
        }
    if not parsed:
        return {"ok": False, "message": "没有可解析的器件 JSON", "errors": errors}
    session_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S") + "-" + uuid4().hex[:8]
    folder = _session_dir(session_id)
    folder.mkdir(parents=True, exist_ok=True)
    used: set[str] = set()
    items = []
    for stored_name, original_name, meta, raw in parsed:
        stored = _safe_name(stored_name, used)
        (folder / stored).write_bytes(raw)
        items.append({"file_name": stored, "original_name": original_name, **meta})
    index = {
        "session_id": session_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "component_count": len(items),
        "items": items,
        "errors": [],
        "refdes_list": [x["refdes"] for x in items if x.get("refdes")],
    }
    (folder / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    log_line(
        "板级JSON上传",
        session_id=session_id,
        detail=f"上传文件数={len(files)} 器件数={len(items)}",
        result=f"位号示例={index['refdes_list'][:8]}",
    )
    return {
        "ok": True,
        "session_id": session_id,
        "component_count": len(items),
        "refdes_list": index["refdes_list"],
    }


def create_chat_session(refdes: str | None = None) -> str:
    """对话开始时建一个空会话，只记下位号。不写聊天记录。"""
    session_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S") + "-" + uuid4().hex[:8]
    folder = _session_dir(session_id)
    folder.mkdir(parents=True, exist_ok=True)
    ref = str(refdes or "").replace("@", "").strip().upper() or None
    index = {
        "session_id": session_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "component_count": 0,
        "items": [],
        "errors": [],
        "refdes_list": [ref] if ref else [],
        "target_refdes": ref,
    }
    (folder / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    log_line("会话创建", session_id=session_id, result=f"位号={ref or '-'}")
    return session_id


def pick_component_json(files: list[tuple[str, bytes]], refdes: str) -> tuple[bytes, str]:
    """从上传的 JSON 里取出要检索的那一个器件。只有一个对象时直接用。"""
    ref = str(refdes or "").replace("@", "").strip().upper()
    found: list[tuple[bytes, str, str]] = []
    for filename, content in files:
        name = filename or "part.json"
        try:
            objects = _iter_objects(name, content)
        except Exception as exc:
            raise ValueError(f"器件内容无法解析：{exc}") from exc
        for stored_name, payload, raw in objects:
            meta = extract_meta(payload, stored_name)
            part_ref = str(meta.get("refdes") or "").replace("@", "").strip().upper()
            found.append((raw, stored_name, part_ref))
    if not found:
        raise ValueError("没有可解析的器件 JSON")
    if ref:
        matched = [item for item in found if item[2] == ref]
        if matched:
            raw, stored_name, _part_ref = matched[0]
            return raw, stored_name
    if len(found) == 1:
        raw, stored_name, _part_ref = found[0]
        return raw, stored_name
    raise ValueError("请上传该位号对应的器件 JSON")


def create_session(files: list[tuple[str, bytes]]) -> dict:
    """兼容旧调用：等同 upload_board_json，校验失败时抛 ValueError。"""
    result = upload_board_json(files)
    if not result.get("ok"):
        msg = result.get("message") or "JSON 格式校验失败"
        raise ValueError(msg)
    return result


def save_fuzzy_ready(session_id: str, ready: dict | None) -> None:
    """低于 200 组时先记下组合，等用户确认后再进入选择。"""
    path = _session_dir(session_id) / "index.json"
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    data["fuzzy_ready"] = ready or None
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def save_fuzzy_choices(session_id: str, choices: list | None) -> None:
    """记下低于 200 组时给用户选的属性组合。选完或重新查就清掉。"""
    path = _session_dir(session_id) / "index.json"
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    data["fuzzy_choices"] = choices or None
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def save_fuzzy_empty_count(session_id: str, count: int) -> None:
    """记下连续查不到分组的次数。查到结果后清零。"""
    path = _session_dir(session_id) / "index.json"
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    data["fuzzy_empty_count"] = int(count)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def save_search_intent(session_id: str, intent: dict | None, pending: dict | None) -> None:
    """记下已经确认的初筛条件，以及还没确认完的那一轮。"""
    path = _session_dir(session_id) / "index.json"
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    data["search_intent"] = intent or {}
    data["pending_confirm"] = pending
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def set_chat_refdes(session_id: str, refdes: str | None) -> None:
    """定位到位号后写回会话。没有位号时保持空。"""
    path = _session_dir(session_id) / "index.json"
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    ref = str(refdes or "").replace("@", "").strip().upper() or None
    data["target_refdes"] = ref
    data["refdes_list"] = [ref] if ref else []
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_session(session_id: str) -> dict:
    path = _session_dir(session_id) / "index.json"
    if not path.exists():
        raise FileNotFoundError(f"会话不存在：{session_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def remember_last_query(session_id: str, *, refdes: str | None, top_n: int | None) -> None:
    """记住上一次查询的位号/条数，便于用户后续只说「再给我 20 个」。"""
    path = _session_dir(session_id) / "index.json"
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    data["last_query"] = {"refdes": refdes, "top_n": top_n}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


_CHAT_HISTORY_MAX = 40


def append_chat_turn(session_id: str, *, user: str, assistant: str) -> None:
    """追加一轮对话到会话记忆（供下一轮意图理解）。"""
    path = _session_dir(session_id) / "index.json"
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    hist = list(data.get("chat_history") or [])
    now = datetime.now(timezone.utc).isoformat()
    if (user or "").strip():
        hist.append({"role": "user", "content": user.strip(), "at": now})
    if (assistant or "").strip():
        hist.append({"role": "assistant", "content": assistant.strip(), "at": now})
    if len(hist) > _CHAT_HISTORY_MAX:
        hist = hist[-_CHAT_HISTORY_MAX:]
    data["chat_history"] = hist
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def recent_chat_history(session: dict, *, limit: int = 12) -> list[dict]:
    """最近若干条对话，供 LLM 意图 prompt 使用。"""
    rows = list(session.get("chat_history") or [])
    if limit > 0:
        rows = rows[-limit:]
    return [{"role": x.get("role"), "content": x.get("content")} for x in rows if x.get("content")]


def delete_session(session_id: str) -> bool:
    """关闭会话，释放磁盘缓存。"""
    folder = _session_dir(session_id)
    if not folder.exists():
        return False
    shutil.rmtree(folder)
    log_line("会话删除", session_id=session_id, result="【已释放磁盘缓存】")
    return True


def llm_summary(session: dict) -> list[dict]:
    out = []
    for item in session.get("items") or []:
        out.append(
            {
                "refdes": item.get("refdes"),
                "package_symbol": item.get("package_symbol"),
                "pin_count": item.get("pin_count"),
                "device_type": item.get("device_type"),
                "layers": item.get("layers"),
                "value": item.get("value"),
                "top_side": item.get("top_side"),
            }
        )
    return out


def resolve_refdes(session_id: str, refdes: str) -> dict:
    session = load_session(session_id)
    want = (refdes or "").replace("@", "").strip().upper()
    hits = [
        x for x in (session.get("items") or [])
        if (x.get("refdes") or "").replace("@", "").upper() == want
    ]
    if not hits:
        raise ValueError(f"会话中找不到位号 {want}")
    if len(hits) > 1:
        names = ", ".join(x["file_name"] for x in hits)
        raise ValueError(f"位号 {want} 对应多个文件：{names}")
    hit = hits[0]
    path = _session_dir(session_id) / hit["file_name"]
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return {
        "file_name": hit["file_name"],
        "bytes": path.read_bytes(),
        "payload": payload,
        "meta": dict(hit),
    }


def cleanup_expired_sessions(ttl_days: int = 7) -> int:
    """TTL 到期自动清理（默认 7 天）。"""
    root = sessions_root()
    now = datetime.now(timezone.utc)
    removed = 0
    with _cleanup_lock:
        for folder in list(root.iterdir()):
            if not folder.is_dir() or not _SESSION_ID_RE.fullmatch(folder.name):
                continue
            index_path = folder / "index.json"
            try:
                created_raw = json.loads(index_path.read_text(encoding="utf-8")).get("created_at", "")
                created_at = datetime.fromisoformat(created_raw)
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=timezone.utc)
            except Exception:
                created_at = datetime.fromtimestamp(folder.stat().st_mtime, tz=timezone.utc)
            if (now - created_at).total_seconds() <= ttl_days * 86400:
                continue
            shutil.rmtree(folder)
            removed += 1
            log_line("会话过期清理", session_id=folder.name, result="【TTL到期已删除】")
    return removed
