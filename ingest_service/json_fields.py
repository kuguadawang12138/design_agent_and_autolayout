# -*- coding: utf-8 -*-
"""设计部 JSON 顶层字段提取（不含 pins[] 几何）。"""

from __future__ import annotations

from typing import Any


def _pick(payload: dict | None, *keys) -> Any:
    if not isinstance(payload, dict):
        return None
    lower = {str(k).strip().lower(): v for k, v in payload.items()}
    for key in keys:
        if key in payload and payload[key] not in (None, ""):
            return payload[key]
        lk = str(key).strip().lower()
        if lk in lower and lower[lk] not in (None, ""):
            return lower[lk]
    return None


def _as_str(value) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    return text or None


def _as_int(value) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def _as_float(value) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def normalize_project_code(raw) -> str | None:
    text = _as_str(raw)
    if not text:
        return None
    return text


def extract_refdes(payload: dict | None, filename: str = "") -> str | None:
    raw = _pick(payload, "refdes", "ref", "component_ref", "designator", "位号")
    text = _as_str(raw)
    if text:
        return text.replace("@", "").strip().upper()
    stem = (filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    if stem.lower().endswith(".json"):
        stem = stem[:-5]
    if "_" in stem:
        tail = stem.rsplit("_", 1)[-1].replace("@", "").strip().upper()
        if tail and tail[0].isalpha() and any(ch.isdigit() for ch in tail):
            return tail
    return None


def extract_top_side(payload: dict | None) -> bool:
    """器件 JSON 正反面：布尔或 0/1（与明工/设计部字段一致）。"""
    from top_side_util import parse_top_side

    raw = _pick(payload, "top_side", "topSide", "side", "正反面")
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return True
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return int(raw) != 0
    return parse_top_side(raw)


def extract_meta(payload: dict | None, filename: str = "") -> dict:
    payload = payload if isinstance(payload, dict) else {}
    pkg = _as_str(_pick(payload, "packagesymbol", "package_symbol", "package", "封装名称"))
    if not pkg:
        pkg = _as_str(_pick(payload, "name"))
    layers = _as_int(_pick(payload, "layers", "board_layers", "层数"))
    pin_count = _as_int(_pick(payload, "pin_count"))
    pins = payload.get("pins") or payload.get("pads")
    if pin_count is None and isinstance(pins, (list, tuple)):
        pin_count = len(pins)
    return {
        "project_code": normalize_project_code(_pick(payload, "project_code", "source_board", "项目编码")),
        "board_name": _as_str(_pick(payload, "project_name", "board_name", "板子名称")),
        "refdes": extract_refdes(payload, filename),
        "units": _as_str(_pick(payload, "units", "unit")) or "mils",
        "layers": layers,
        "device_type": _as_str(_pick(payload, "devicetype", "device_type", "器件类型")),
        "package_symbol": pkg,
        "value": _as_str(_pick(payload, "value", "参数值")),
        "top_side": extract_top_side(payload),
        "pin_count": pin_count,
        "model": _as_str(_pick(payload, "model", "型号")),
        "brand": _as_str(_pick(payload, "brand", "品牌")),
        "pcb_x": _as_float(_pick(payload, "X", "x")),
        "pcb_y": _as_float(_pick(payload, "Y", "y")),
        "customer_code": _as_str(_pick(payload, "customer_code", "customerCode", "客户编码")),
        "designer_name": _as_str(_pick(payload, "designer_name", "designer", "设计工程师")),
        "designer_seniority": _as_float(_pick(payload, "designer_seniority", "seniority_score", "工程师资历")),
        "finish_time": _as_str(_pick(payload, "finish_time", "complete_time", "完成时间")),
        "source_file": filename,
    }


def validate_component_json(payload: dict | None, filename: str = "") -> tuple[bool, str, dict | None]:
    """器件 JSON 格式校验（入库 / 会话上传共用）。"""
    if not isinstance(payload, dict):
        return False, "JSON 必须是对象", None
    meta = extract_meta(payload, filename)
    if not meta.get("refdes"):
        return False, "缺少 refdes（位号）", None
    units = meta.get("units")
    if not units:
        return False, "缺少 units（长度单位）", None
    pins = payload.get("pins") or payload.get("pads")
    if not isinstance(pins, list) or len(pins) == 0:
        return False, "缺少 pins[] 或为空", None
    for i, pin in enumerate(pins):
        if not isinstance(pin, dict):
            return False, f"pins[{i}] 必须是对象", None
        padstack = pin.get("padstack") or pin.get("pad_stack")
        if not padstack:
            return False, f"pins[{i}] 缺少 padstack", None
        for key in ("x", "y", "rotation"):
            if key not in pin and key.upper() not in pin:
                return False, f"pins[{i}] 缺少 {key}", None
    return True, "", meta


def query_fields_from_meta(meta: dict | None) -> dict:
    meta = meta or {}
    return {
        "package_symbol": meta.get("package_symbol"),
        "pins": meta.get("pin_count"),
        "device_type": meta.get("device_type"),
        "model": meta.get("model"),
        "brand": meta.get("brand"),
        "value": meta.get("value"),
        "layers": meta.get("layers"),
        "component_ref": meta.get("refdes"),
        "project_code": meta.get("project_code"),
        "top_side": meta.get("top_side"),
    }
