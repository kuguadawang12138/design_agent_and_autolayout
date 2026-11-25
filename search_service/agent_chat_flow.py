# -*- coding: utf-8 -*-
"""对话先定位位号并返回 session_id；检索在上传 JSON 的接口里做。

流式文案只说到哪一步、第几批、处理了多少，不写内部算法细节。
器件信息由模型按五档打分（温度 0）。项目属性用公式，先换成百分。
总分 = 0.9×IoU百分 + 0.05×模型分 + 0.05×项目百分，按总分从高到低排。
"""

from __future__ import annotations

import contextvars
import json
import queue
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Iterator

from agent_llm_rerank import (
    LlmRankError,
    _parse_json_object,
    call_llm_chat,
    score_device_batch,
)
from agent_log import log_line, use_log_session
from agent_rerank import fetch_candidate_fields
from agent_session import (
    append_chat_turn,
    create_chat_session,
    load_session,
    save_fuzzy_choices,
    save_fuzzy_ready,
    save_fuzzy_empty_count,
    save_search_intent,
    set_chat_refdes,
)
from footprint_db import (
    connect,
    fetch_project_profile,
    find_similar_from_content,
    query_devices_by_attributes,
    query_fuzzy_library,
)
from json_fields import extract_meta
from settings import get_agent_config, get_llm_config

BATCH_SIZE = 20
SCORE_THREADS = 2
# 三项都先换成 0～100，再乘权重，满分 100。剩下的一成由模型和项目对半。
IOU_WEIGHT = 0.9
DEVICE_WEIGHT = 0.05
PROJECT_WEIGHT = 0.05
PROJECT_RAW_MAX = 20.0
_REFDES_IN_TEXT = re.compile(r"[A-Za-z]{1,6}\d+[A-Za-z0-9]*")
_FUZZY_SPLIT = re.compile(r"[-_ \u3000\uff0d\u2013\u2014]+")
_FUZZY_TEXT_FIELDS = ("devicetype", "packagesymbol", "value")
FUZZY_RESULT_LIMIT = 200
_FUZZY_NOTE = "本次查询为模糊查询不能保证准确率如果需要进一步请提供位号以进行精确查询"
_FUZZY_TEMPLATE_KEYS: tuple[tuple[str, str], ...] = (
    ("devicetype", "器件类型"),
    ("packagesymbol", "封装名"),
    ("value", "元件参数值"),
    ("pin_count", "pin数"),
    ("finish_year", "年份"),
    ("layers", "层数"),
)
_FUZZY_REFINE_ASK = "请补充更多信息，我需要查得更详细。"
_AGREE_WORDS = {"对", "是", "可以", "没错", "确认", "好", "行", "就这样", "嗯", "好的", "没问题"}
_ASSISTANT_SENTENCE = "我是辅助 EDA 设计的小助手。"
# 意图只认这些字段。没提到的留空，不要多写字段，避免把没说过的内容回给用户。
_ASSISTANT_SYSTEM = (
    "你是辅助 EDA 设计的小助手。只根据用户这句话填写下面这个 JSON，不要增加字段，不要编造用户没说的内容。\n"
    "mode：问你是谁、你好、你能做什么，填 chat。能确定一个位号就填 refdes。"
    "没有位号，但说到了封装、值、器件类型、层数、pin 脚数或完成年份，填 fuzzy。什么都没说到，填 ask。\n"
    "refdes：只要位号本身，例如 U4、C12、R101。没有就 null。多个位号分不清就 null，mode 填 ask。没有位号也可以查询，不要为此编造位号。\n"
    "time_focus：只有用户说到最近、近些天、越新、完成时间早晚，才填 1。指定了某年或某年某月时填 0。都没提到填 0。\n"
    "finish_year、finish_month：只有说到具体某年或某年某月才填数字，例如 2025 年填 year=2025、month 留 null；"
    "2025年3月填 year=2025、month=3。只说最近、近些天时不要填年月。有具体年月时 time_focus 必须是 0。\n"
    "layers：用户只说了一个层数时填整数，同时 layers_min、layers_max 都填同一个数。没说就都为 null。\n"
    "layers_min、layers_max：用户说层数区间时填，例如 4 到 8 层填 min=4、max=8。单值时不要乱填区间。\n"
    "pin_count：用户说到 pin 数、脚数、引脚个数时填整数。没说到就 null。\n"
    "finish_year：用户说到哪一年完成、哪年的板子时填四位年份，对应项目完成时间。没说到就 null。\n"
    "devicetype、packagesymbol、value：用户说到器件类型、封装、值、阻值、容值时，把你的理解填进对应字段。"
    "packagesymbol 是封装名。devicetype 和 value 容易混，分不清就按你目前的猜测填上，没提到的留空字符串。\n"
    "need_confirm：用户提到了器件类型、封装、值这三项里的任何信息，填 1。"
    "或者你不确定层数、pin、年份、devicetype、packagesymbol、value 里任何一项该怎么归类，也填 1。都没提到并且你能确定，填 0。\n"
    "只输出这一个 JSON：\n"
    '{"mode":"chat或refdes或ask","refdes":null,"time_focus":0,"finish_year":null,"finish_month":null,'
    '"layers":null,"layers_min":null,"layers_max":null,"pin_count":null,'
    '"devicetype":"","packagesymbol":"","value":"","need_confirm":0}'
)
_CONFIRM_SYSTEM = (
    "你是辅助 EDA 设计的小助手。用户正在确认上一轮列出的器件信息。\n"
    "判断用户是不是觉得你列的内容对。同意、是、可以、没错、就这样，以及别的肯定说法都算认同，不要只认「对」这个字。\n"
    "认同就 need_confirm 填 0，字段保持上一轮的内容。\n"
    "用户补充或改正了，就把对应字段改掉，need_confirm 填 0。\n"
    "还是分不清属于哪一项，need_confirm 填 1，并写上你目前的猜测。\n"
    "不要新增字段，不要编造用户没说的内容。\n"
    "只输出这一个 JSON：\n"
    '{"mode":"refdes","refdes":null,"time_focus":0,"finish_year":null,"finish_month":null,'
    '"layers":null,"layers_min":null,"layers_max":null,"pin_count":null,'
    '"devicetype":"","packagesymbol":"","value":"","need_confirm":0}'
)
_SUMMARY_SYSTEM = (
    "你是辅助 EDA 设计的小助手。把用户给出的几句说明收成一段中文。"
    "只输出这段中文，不要输出 JSON，不要讨论提示词是否冲突。"
)


def _event(step: str, text: str, **extra: Any) -> dict[str, Any]:
    return {"step": step, "text": text, **extra}


def _refdes_hits(message: str) -> list[str]:
    hits: list[str] = []
    for tok in _REFDES_IN_TEXT.findall(message or ""):
        ref = tok.upper()
        if ref not in hits:
            hits.append(ref)
    return hits


def _opt_int(value) -> int | None:
    if value in (None, "", "null"):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _opt_text(value) -> str | None:
    text = str(value or "").strip()
    if not text or text.lower() == "null":
        return None
    return text


def _clean_filters(data: dict, pending: dict | None) -> dict:
    """只留下约定的初筛字段。确认轮里用户没改的，沿用上一轮。"""
    previous = pending or {}
    time_focus = 1 if _opt_int(data.get("time_focus")) == 1 else 0
    if pending and not time_focus:
        time_focus = 1 if _opt_int(previous.get("time_focus")) == 1 else 0
    year = _opt_int(data.get("finish_year"))
    month = _opt_int(data.get("finish_month"))
    layers = _opt_int(data.get("layers"))
    layers_min = _opt_int(data.get("layers_min"))
    layers_max = _opt_int(data.get("layers_max"))
    pin_count = _opt_int(data.get("pin_count"))
    device_type = _opt_text(data.get("devicetype"))
    package_symbol = _opt_text(data.get("packagesymbol"))
    value = _opt_text(data.get("value"))
    if layers is not None and layers_min is None and layers_max is None:
        layers_min = layers_max = layers
    if pending:
        year = year if year is not None else _opt_int(previous.get("finish_year"))
        month = month if month is not None else _opt_int(previous.get("finish_month"))
        layers = layers if layers is not None else _opt_int(previous.get("layers"))
        layers_min = layers_min if layers_min is not None else _opt_int(previous.get("layers_min"))
        layers_max = layers_max if layers_max is not None else _opt_int(previous.get("layers_max"))
        pin_count = pin_count if pin_count is not None else _opt_int(previous.get("pin_count"))
        device_type = device_type or _opt_text(previous.get("devicetype"))
        package_symbol = package_symbol or _opt_text(previous.get("packagesymbol"))
        value = value or _opt_text(previous.get("value"))
        if layers is not None and layers_min is None and layers_max is None:
            layers_min = layers_max = layers
    if month is not None and (month < 1 or month > 12):
        month = None
    # 有具体年月就只做初筛，完成时间不再单独加权。
    if year is not None:
        time_focus = 0
    return {
        "time_focus": time_focus,
        "finish_year": year,
        "finish_month": month,
        "layers": layers,
        "layers_min": layers_min,
        "layers_max": layers_max,
        "pin_count": pin_count,
        "devicetype": device_type,
        "packagesymbol": package_symbol,
        "value": value,
    }


def _fuzzy_tokens(text: str) -> list[str]:
    """按横杠、下划线、空格切开，只留两个字符及以上的片段。"""
    tokens: list[str] = []
    for part in _FUZZY_SPLIT.split(str(text or "").strip()):
        token = part.strip()
        if len(token) >= 2 and token.lower() not in {item.lower() for item in tokens}:
            tokens.append(token)
    return tokens


def _layers_display(filters: dict) -> str:
    """层数展示：区间或单值。"""
    lmin = filters.get("layers_min")
    lmax = filters.get("layers_max")
    if lmin is not None and lmax is not None:
        if int(lmin) != int(lmax):
            return f"{lmin}-{lmax}"
        return str(lmin)
    if filters.get("layers") is not None:
        return str(filters.get("layers"))
    return ""


def _fuzzy_template(filters: dict | None = None) -> str:
    """前端展示用模板，字段名用中文。"""
    filters = filters or {}
    lines: list[str] = []
    for key, label in _FUZZY_TEMPLATE_KEYS:
        if key == "layers":
            text = _layers_display(filters)
        elif key == "finish_year":
            year = filters.get("finish_year")
            text = "" if year is None else str(year)
        elif key == "pin_count":
            pin = filters.get("pin_count")
            text = "" if pin is None else str(pin)
        else:
            text = str(filters.get(key) or "")
        lines.append(f"{label}：{text}")
    return "\n".join(lines)


def _fuzzy_ask_text() -> str:
    return "请按下面的格式填写，没说到的留空。\n" + _fuzzy_template()


def _fuzzy_sse_fields(filters: dict) -> dict[str, Any]:
    """模糊查询各步骤带回的字段。"""
    return {
        "devicetype": filters.get("devicetype") or "",
        "packagesymbol": filters.get("packagesymbol") or "",
        "value": filters.get("value") or "",
        "pin_count": filters.get("pin_count"),
        "finish_year": filters.get("finish_year"),
        "layers": _opt_int(filters.get("layers")),
        "layers_min": filters.get("layers_min"),
        "layers_max": filters.get("layers_max"),
        "template": _fuzzy_template(filters),
    }


def _has_fuzzy_fields(filters: dict | None) -> bool:
    filters = filters or {}
    if filters.get("pin_count") is not None:
        return True
    if filters.get("finish_year") is not None:
        return True
    if filters.get("layers_min") is not None or filters.get("layers_max") is not None:
        return True
    if filters.get("layers") is not None:
        return True
    return any(_fuzzy_tokens(filters.get(key) or "") for key in _FUZZY_TEXT_FIELDS)


def _confirm_text(filters: dict) -> str:
    """反问只列出这几项，不另写别的内容。"""
    lines = [line for line in _fuzzy_template(filters).splitlines()]
    lines.append("请确认以上内容，不对的直接补充。")
    return "\n".join(lines)


def parse_refdes_message(
    message: str,
    pending: dict | None = None,
    carry: dict | None = None,
) -> dict[str, Any]:
    """整句交给模型。不先做正则。确认轮会带上上一轮记下的内容。"""
    cfg = get_llm_config()
    if not cfg.get("enabled") or not get_agent_config().get("llm_intent_enabled", True):
        return {
            "target_refdes": None,
            "need_clarification": True,
            "clarification_message": "请直接说出要查询的位号，例如 U1。",
            "filters": {},
        }
    if pending:
        prompt = (
            "上一轮记录：\n"
            + json.dumps(pending.get("filters") or pending, ensure_ascii=False)
            + f"\n用户回答：{message}"
        )
        system = _CONFIRM_SYSTEM
    else:
        prompt = f"用户消息：{message}"
        system = _ASSISTANT_SYSTEM
    try:
        text = call_llm_chat(prompt, cfg, purpose="意图", temperature=0, system_prompt=system)
        data = _parse_json_object(text, context="定位位号")
    except LlmRankError as exc:
        log_line("LLM意图", detail=str(exc), result="【失败】", level="error")
        return {
            "target_refdes": None,
            "need_clarification": True,
            "clarification_message": "请再说一下要查询的位号。",
            "filters": {},
        }
    return _intent_from_llm(data, pending, carry)


def _intent_from_llm(
    data: dict,
    pending: dict | None = None,
    carry: dict | None = None,
) -> dict[str, Any]:
    """模型结果收成位号、初筛条件和是否反问。多出来的字段丢掉。"""
    mode = str(data.get("mode") or "").strip().lower()
    filters = _clean_filters(data, pending or carry)
    if mode == "chat" and not pending:
        return {
            "target_refdes": None,
            "need_clarification": False,
            "chat_reply": True,
            "clarification_message": _ASSISTANT_SENTENCE,
            "filters": {},
        }
    target = str(data.get("refdes") or data.get("target_refdes") or "").replace("@", "").strip().upper()
    if pending and not (target and _REFDES_IN_TEXT.fullmatch(target)):
        target = str(pending.get("refdes") or "").upper()
    has_refdes = bool(target and _REFDES_IN_TEXT.fullmatch(target))
    need_confirm = _opt_int(data.get("need_confirm")) == 1
    if not pending and not carry and any(filters.get(key) for key in ("devicetype", "packagesymbol", "value")):
        need_confirm = True
    if need_confirm:
        question = _confirm_text(filters)
        return {
            "target_refdes": target if has_refdes else None,
            "need_clarification": False,
            "need_confirm": True,
            "clarification_message": question,
            "filters": filters,
        }
    if has_refdes:
        return {
            "target_refdes": target,
            "need_clarification": False,
            "clarification_message": "",
            "filters": filters,
        }
    if _has_fuzzy_fields(filters):
        return {
            "target_refdes": None,
            "need_clarification": False,
            "fuzzy_query": True,
            "clarification_message": "",
            "filters": filters,
        }
    return {
        "target_refdes": None,
        "need_clarification": True,
        "clarification_message": _fuzzy_ask_text(),
        "filters": filters,
    }


def _is_agree(message: str) -> bool:
    text = re.sub(r"[\s。！!，,]", "", str(message or ""))
    return text in _AGREE_WORDS


def _extracted_text(filters: dict) -> str:
    """确认这一步只返回本次提取到的属性。"""
    body = _fuzzy_template(filters)
    return "请确认这次提取到的属性。确认后我会列出可选组合。\n" + body


def _choice_index(message: str, count: int) -> int | None:
    """只把「1」「第2个」这类说法当成选项。对不上就返回空。"""
    text = re.sub(r"\s+", "", str(message or ""))
    matched = re.fullmatch(r"(?:选|看|第)?(\d+)(?:个|组|项|号)?", text)
    if not matched:
        return None
    number = int(matched.group(1))
    if number < 1 or number > count:
        return -1
    return number - 1


def _choice_text(rows: list[dict]) -> str:
    """属性组合放在 items 里，文字里不再逐条列出。"""
    return "\n".join(
        [
            f"按你给的条件，可以看的属性组合有 {len(rows)} 个。",
            "选中后我会给出这一组的项目编码、位号和 PDF。",
            "还想看别的组合，选完这一组之后，下一轮重新告诉我器件类型、封装名、元件参数值、pin数、年份或层数。",
            _FUZZY_NOTE,
        ]
    )


def _fuzzy_pick_events(session_id: str, message: str, choice: dict, number: int) -> Iterator[dict[str, Any]]:
    """按选定的四个属性取出项目编码、位号和 PDF。"""
    conn = connect()
    try:
        rows = query_devices_by_attributes(conn, choice)
        for row in rows:
            row["component_ref"] = row.get("refdes")
        if rows:
            _attach_highlight(conn, rows, session_id)
        shown = [_public_item(row) for row in rows]
    except Exception as exc:
        log_line("模糊查询", detail=str(exc), result="【取器件失败】", level="error")
        ask = "这一组暂时没有取出器件。请再回复一次序号。"
        append_chat_turn(session_id, user=message, assistant=ask)
        yield _event("choose", ask, reply=ask, message=ask, refdes=None, session_id=session_id)
        return
    finally:
        conn.close()
    save_fuzzy_choices(session_id, None)
    if not shown:
        ask = "这一组下面没有项目编码和位号。可以回复别的序号，或下一轮重新填写属性。"
        append_chat_turn(session_id, user=message, assistant=ask)
        log_line("模糊查询", result=f"【选中第{number}组但没有器件】")
        yield _event("choose", ask, reply=ask, message=ask, refdes=None, session_id=session_id)
        return
    reply = (
        f"已按你选的第 {number} 组，找到 {len(shown)} 个项目位号，PDF 在结果里。\n"
        "还想看其他组合的话，下一轮重新告诉我器件类型、封装名、元件参数值、pin数、年份或层数。"
    )
    append_chat_turn(session_id, user=message, assistant=reply)
    log_line("模糊查询", result=f"【打开第{number}组】器件={len(shown)}")
    yield _event(
        "done",
        reply,
        reply=reply,
        message=reply,
        refdes=None,
        items=shown,
        return_count=len(shown),
        candidate_count=len(shown),
        session_id=session_id,
    )


def _fuzzy_search_events(session_id: str, message: str, filters: dict) -> Iterator[dict[str, Any]]:
    """没有位号时，用四个字段的片段做模糊查询。超过 200 条就不返回列表。"""
    tokens = {
        key: _fuzzy_tokens(filters.get(key) or "")
        for key in _FUZZY_TEXT_FIELDS
    }
    log_line(
        "模糊查询",
        result=(
            f"层数={_layers_display(filters) or '-'} "
            f"pin={filters.get('pin_count') if filters.get('pin_count') is not None else '-'} "
            f"年={filters.get('finish_year') if filters.get('finish_year') is not None else '-'} "
            f"片段={tokens}"
        ),
    )
    conn = connect()
    try:
        total, rows = query_fuzzy_library(conn, filters)
        if total > FUZZY_RESULT_LIMIT:
            ask = (
                f"按这些属性分组后有 {total} 组，超过 {FUZZY_RESULT_LIMIT} 组。"
                + _FUZZY_REFINE_ASK
            )
            append_chat_turn(session_id, user=message, assistant=ask)
            save_fuzzy_choices(session_id, None)
            save_fuzzy_ready(session_id, None)
            save_search_intent(session_id, filters, None)
            log_line("模糊查询", result=f"【过多】{total}")
            yield _event(
                "refine",
                ask,
                reply=ask,
                message=ask,
                refdes=None,
                session_id=session_id,
                **_fuzzy_sse_fields(filters),
            )
            return
        shown = list(rows)
    except Exception as exc:
        log_line("模糊查询", detail=str(exc), result="【失败】", level="error")
        ask = "模糊查询暂时没做成，请再说一下器件类型、封装名、元件参数值、pin数、年份或层数。"
        append_chat_turn(session_id, user=message, assistant=ask)
        yield _event(
            "refine",
            ask,
            reply=ask,
            message=ask,
            refdes=None,
            session_id=session_id,
        )
        return
    finally:
        conn.close()
    if total <= 0:
        session = load_session(session_id)
        empty_count = int(session.get("fuzzy_empty_count") or 0) + 1
        save_fuzzy_empty_count(session_id, 0 if empty_count >= 3 else empty_count)
        if empty_count >= 3:
            ask = "搜索不到"
            log_line("模糊查询", result=f"【没有结果】第{empty_count}次，停止补充")
        else:
            ask = "没有查到符合这些条件的分组。请按格式重新填写，没说到的留空。\n" + _fuzzy_template()
            log_line("模糊查询", result=f"【没有结果】第{empty_count}次")
        append_chat_turn(session_id, user=message, assistant=ask)
        save_fuzzy_choices(session_id, None)
        save_fuzzy_ready(session_id, None)
        save_search_intent(session_id, filters, None)
        yield _event(
            "empty",
            ask,
            reply=ask,
            message=ask,
            refdes=None,
            session_id=session_id,
        )
        return
    if len(shown) > FUZZY_RESULT_LIMIT:
        ask = (
            f"这些属性合在一起有 {len(shown)} 组，超过 {FUZZY_RESULT_LIMIT} 组。"
            + _FUZZY_REFINE_ASK
        )
        append_chat_turn(session_id, user=message, assistant=ask)
        save_fuzzy_choices(session_id, None)
        save_fuzzy_ready(session_id, None)
        save_search_intent(session_id, filters, None)
        log_line("模糊查询", result=f"【属性组合过多】{len(shown)}")
        yield _event(
            "refine",
            ask,
            reply=ask,
            message=ask,
            refdes=None,
            session_id=session_id,
            **_fuzzy_sse_fields(filters),
        )
        return
    reply = _extracted_text(filters)
    for index, row in enumerate(shown, start=1):
        row["item"] = index
    append_chat_turn(session_id, user=message, assistant=reply)
    save_fuzzy_empty_count(session_id, 0)
    save_fuzzy_choices(session_id, None)
    save_fuzzy_ready(session_id, {"rows": shown, "total": total})
    save_search_intent(session_id, filters, None)
    log_line("模糊查询", result=f"【确认属性】分组={total} 组合={len(shown)}")
    yield _event(
        "confirm",
        reply,
        reply=reply,
        message=reply,
        refdes=None,
        session_id=session_id,
        **_fuzzy_sse_fields(filters),
    )


def fuzzy_events(message: str, session_id: str) -> Iterator[dict[str, Any]]:
    """属性查询。确认、分组、选一个、取项目位号和 PDF，都在这里判断。"""
    if not session_id:
        session_id = create_chat_session(None)
    use_log_session(session_id)
    session = load_session(session_id)
    pending = session.get("pending_confirm") if isinstance(session.get("pending_confirm"), dict) else None
    choices = session.get("fuzzy_choices") if isinstance(session.get("fuzzy_choices"), list) else []
    if choices and not pending:
        picked = _choice_index(message, len(choices))
        if picked is not None:
            if picked < 0:
                ask = f"序号要在 1 到 {len(choices)} 之间。回复序号即可，例如 1。"
                append_chat_turn(session_id, user=message, assistant=ask)
                yield _event(
                    "choose",
                    ask,
                    reply=ask,
                    message=ask,
                    refdes=None,
                    session_id=session_id,
                )
                return
            yield from _fuzzy_pick_events(session_id, message, choices[picked], picked + 1)
            return
    ready = session.get("fuzzy_ready") if isinstance(session.get("fuzzy_ready"), dict) else None
    if ready and ready.get("rows") and _is_agree(message):
        shown = list(ready.get("rows") or [])
        total = int(ready.get("total") or len(shown))
        reply = _choice_text(shown)
        append_chat_turn(session_id, user=message, assistant=reply)
        save_fuzzy_ready(session_id, None)
        save_fuzzy_choices(session_id, shown)
        log_line("模糊查询", result=f"【请选择】属性组={len(shown)}")
        yield _event(
            "choose",
            reply,
            reply=reply,
            message=reply,
            refdes=None,
            items=shown,
            return_count=len(shown),
            candidate_count=total,
            session_id=session_id,
        )
        return
    intent = parse_refdes_message(message, pending)
    filters = dict(intent.get("filters") or {})
    if not _has_fuzzy_fields(filters):
        ask = _fuzzy_ask_text()
        append_chat_turn(session_id, user=message, assistant=ask)
        log_line("模糊查询", result="【属性空白】")
        save_fuzzy_ready(session_id, None)
        save_fuzzy_choices(session_id, None)
        yield _event(
            "blank",
            ask,
            reply=ask,
            message=ask,
            refdes=None,
            session_id=session_id,
            template=_fuzzy_template(),
        )
        return
    yield from _fuzzy_search_events(session_id, message, filters)


def locate_refdes_events(message: str, session_id: str | None = None) -> Iterator[dict[str, Any]]:
    """只回复一句话，并分成两个去向：有位号走 sessions，否则走属性查询。"""
    if not session_id:
        session_id = create_chat_session(None)
    use_log_session(session_id)
    session = load_session(session_id)
    pending = session.get("pending_confirm") if isinstance(session.get("pending_confirm"), dict) else None
    intent = parse_refdes_message(message, pending)
    refdes = str(intent.get("target_refdes") or "").upper() or None
    if intent.get("chat_reply"):
        append_chat_turn(session_id, user=message, assistant=_ASSISTANT_SENTENCE)
        log_line("常规回答", result=_ASSISTANT_SENTENCE)
        yield _event(
            "done",
            _ASSISTANT_SENTENCE,
            reply=_ASSISTANT_SENTENCE,
            message=_ASSISTANT_SENTENCE,
            refdes=None,
            session_id=session_id,
        )
        return
    if refdes and not intent.get("need_confirm") and not intent.get("fuzzy_query"):
        set_chat_refdes(session_id, refdes)
        filters = dict(intent.get("filters") or {})
        save_search_intent(session_id, filters, None)
        append_chat_turn(session_id, user=message, assistant="已定位位号")
        log_line(
            "定位位号",
            result=(
                f"位号={refdes} 时间关注={filters.get('time_focus') or 0} "
                f"年={filters.get('finish_year') or '-'} 月={filters.get('finish_month') or '-'} "
                f"层数={filters.get('layers') or '-'}"
            ),
        )
        yield _event("refdes_ready", "已定位位号", refdes=refdes, session_id=session_id)
        return
    append_chat_turn(session_id, user=message, assistant=message)
    log_line("分流", result="【属性查询】")
    yield _event(
        "need_clarify",
        message,
        reply=message,
        message=message,
        refdes=None,
        session_id=session_id,
    )


def _loads_meta(content: bytes, filename: str) -> dict:
    text = content.decode("utf-8-sig")
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("器件 JSON 必须是对象")
    return extract_meta(payload, filename)


def _query_meta(conn, content: bytes, filename: str, refdes: str) -> dict:
    meta = _loads_meta(content, filename)
    if refdes:
        meta["refdes"] = refdes
    profile = fetch_project_profile(conn, meta.get("project_code") or "") or {}
    for key in ("customer_code", "designer_name", "designer_seniority", "finish_time"):
        if meta.get(key) in (None, "") and profile.get(key) not in (None, ""):
            meta[key] = profile[key]
    return meta


def seniority_grade(value) -> str | None:
    """库内资历分 1～20 映射成五档。17～20 为 A，13～16 为 B，9～12 为 C，5～8 为 D，1～4 为 E。"""
    score = _as_float(value)
    if score is None or score < 1:
        return None
    point = int(score)
    if point > 20:
        point = 20
    return "ABCDE"[(20 - point) // 4]


def _as_float(value) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _time_ord(value) -> float | None:
    if value in (None, ""):
        return None
    if hasattr(value, "timestamp"):
        try:
            return float(value.timestamp())
        except (TypeError, ValueError, OSError):
            return None
    text = str(value).strip().replace("/", "-")[:19]
    for fmt, size in (("%Y-%m-%d %H:%M:%S", 19), ("%Y-%m-%d", 10)):
        try:
            return datetime.strptime(text[:size], fmt).timestamp()
        except ValueError:
            continue
    return None


def _span(value: float | None, low: float | None, high: float | None, weight: float) -> float:
    """有值才计分。全体一样时给满分，缺值是 0。"""
    if value is None or low is None or high is None:
        return 0.0
    if high <= low:
        return weight
    ratio = (value - low) / (high - low)
    return weight * max(0.0, min(1.0, ratio))


def attach_project_scores(query_meta: dict, candidates: list[dict]) -> None:
    """客户 8、完成时间 6、资历 5、设计师 1。不调用模型。"""
    times = []
    seniors = []
    for row in candidates:
        row["_finish_ord"] = _time_ord(row.get("finish_time"))
        row["_seniority"] = _as_float(row.get("designer_seniority") or row.get("seniority_score"))
        if row["_finish_ord"] is not None:
            times.append(row["_finish_ord"])
        if row["_seniority"] is not None:
            seniors.append(row["_seniority"])
    tmin, tmax = (min(times), max(times)) if times else (None, None)
    smin, smax = (min(seniors), max(seniors)) if seniors else (None, None)
    customer_q = str(query_meta.get("customer_code") or "").strip()
    designer_q = str(query_meta.get("designer_name") or "").strip()
    for row in candidates:
        customer = 8.0 if customer_q and customer_q == str(row.get("customer_code") or "").strip() else 0.0
        designer = 1.0 if designer_q and designer_q == str(row.get("designer_name") or "").strip() else 0.0
        finish = _span(row.get("_finish_ord"), tmin, tmax, 6.0)
        senior = _span(row.get("_seniority"), smin, smax, 5.0)
        row["project_customer"] = round(customer, 4)
        row["project_finish"] = round(finish, 4)
        row["project_seniority_score"] = round(senior, 4)
        row["project_designer"] = round(designer, 4)
        row["project_score"] = round(customer + finish + senior + designer, 4)


def _iou_points(iou) -> float:
    """IoU 先留四位，再换成 0～100。1.00 是 100。"""
    value = round(float(iou or 0), 4)
    if value < 0:
        return 0.0
    if value > 1:
        return 100.0
    return round(value * 100, 4)


def _apply_total(
    row: dict,
    device_score: float,
    reason: str,
    device_score_raw: float | None = None,
    time_focus: bool = False,
) -> None:
    """平时 IoU 0.9、模型 0.05、项目 0.05。用户关心完成时间时，完成时间单独占 0.05。"""
    device = max(0.0, min(100.0, float(device_score)))
    raw = device if device_score_raw is None else max(0.0, min(100.0, float(device_score_raw)))
    row["device_score_raw"] = round(raw, 4)
    iou_points = _iou_points(row.get("iou"))
    iou_weighted = round(IOU_WEIGHT * iou_points, 4)
    if time_focus:
        finish_points = min(100.0, float(row.get("project_finish") or 0) / 6.0 * 100.0)
        other_raw = (
            float(row.get("project_customer") or 0)
            + float(row.get("project_seniority_score") or 0)
            + float(row.get("project_designer") or 0)
        )
        other_points = min(100.0, other_raw / 14.0 * 100.0)
        device_weighted = round(0.025 * device, 4)
        project_weighted = round(0.025 * other_points, 4)
        row["finish_weighted"] = round(0.05 * finish_points, 4)
        project_points = round(other_points, 4)
    else:
        project_raw = max(0.0, float(row.get("project_score") or 0))
        project_points = min(100.0, project_raw / PROJECT_RAW_MAX * 100.0)
        device_weighted = round(DEVICE_WEIGHT * device, 4)
        project_weighted = round(PROJECT_WEIGHT * project_points, 4)
        row["finish_weighted"] = 0.0
        project_points = round(project_points, 4)
    row["device_score"] = round(device, 4)
    row["device_reason"] = reason
    row["iou_points"] = iou_points
    row["project_points"] = round(project_points, 4)
    row["iou_weighted"] = iou_weighted
    row["device_weighted"] = device_weighted
    row["project_weighted"] = project_weighted
    row["total_score"] = round(
        iou_weighted + device_weighted + project_weighted + float(row.get("finish_weighted") or 0),
        4,
    )
    row["final_score"] = row["total_score"]


def _log_score_details(candidates: list[dict], time_focus: bool = False) -> None:
    """按最终名次把每一项得分打进日志。"""
    if time_focus:
        weight_text = "权重 IoU=0.9 完成时间=0.05 模型=0.025 其他项目=0.025"
    else:
        weight_text = "权重 IoU=0.9 模型=0.05 项目=0.05，三项都先换成百分再乘权重"
    log_line("候选打分", result=weight_text)
    for rank, row in enumerate(candidates, start=1):
        log_line(
            "候选打分",
            detail=f"名次={rank} 项目={row.get('project_code')} 位号={row.get('refdes')}",
            result=(
                f"IoU={round(float(row.get('iou') or 0), 4)} IoU百分={row.get('iou_points')} "
                f"IoU加权={row.get('iou_weighted')} "
                f"模型原始={row.get('device_score_raw')} 模型={row.get('device_score')} "
                f"模型加权={row.get('device_weighted')} "
                f"客户={row.get('project_customer')} 完成时间={row.get('project_finish')} "
                f"资历={row.get('project_seniority_score')} 设计师={row.get('project_designer')} "
                f"项目原始={row.get('project_score')} 项目百分={row.get('project_points')} "
                f"项目加权={row.get('project_weighted')} 完成时间加权={row.get('finish_weighted')} "
                f"总分={row.get('total_score')}"
            ),
        )


def score_and_sort(query_meta: dict, candidates: list[dict], on_progress, time_focus: bool = False) -> list[dict]:
    """两个线程，每批最多 20 条。打完再统一排序。"""
    attach_project_scores(query_meta, candidates)
    batches = [candidates[i : i + BATCH_SIZE] for i in range(0, len(candidates), BATCH_SIZE)]
    total = len(candidates)
    done = 0
    lock = threading.Lock()

    def _one(batch_no: int, batch: list[dict]) -> None:
        nonlocal done
        scored = score_device_batch(query_meta, batch)
        for index, row in enumerate(batch):
            got = scored.get(index, (0.0, "", 0.0))
            device = got[0]
            reason = got[1] if len(got) > 1 else ""
            raw = got[2] if len(got) > 2 else device
            _apply_total(row, device, reason, raw, time_focus=time_focus)
        with lock:
            done += len(batch)
            current = done
        on_progress(batch_no, len(batches), current, total)

    if batches:
        workers = min(SCORE_THREADS, len(batches))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(contextvars.copy_context().run, _one, i + 1, batch)
                for i, batch in enumerate(batches)
            ]
            for fut in as_completed(futures):
                fut.result()
    candidates.sort(
        key=lambda row: (
            -float(row.get("total_score") or 0),
            -round(float(row.get("iou") or 0), 4),
        )
    )
    _log_score_details(candidates, time_focus=time_focus)
    return candidates


def _summarize(query_meta: dict, items: list[dict]) -> str:
    lines = []
    for row in items[:8]:
        reason = str(row.get("device_reason") or "").strip()
        if reason:
            lines.append(f"{row.get('project_code')}/{row.get('refdes')}：{reason}")
    if not items:
        return "没有找到可返回的相近器件。"
    if not lines:
        top = items[0]
        return f"已按相近程度整理完成，排在前面的是 {top.get('project_code')}/{top.get('refdes')}。"
    cfg = get_llm_config()
    joined = "\n".join(lines)
    if not cfg.get("enabled"):
        return joined
    prompt = (
        "把下面几句器件说明整理成一段不超过 120 字的中文，给工程师看。"
        "不要写内部计算过程。\n"
        f"{joined}"
    )
    try:
        text = call_llm_chat(
            prompt,
            cfg,
            json_mode=False,
            purpose="整理说明",
            temperature=0.1,
            system_prompt=_SUMMARY_SYSTEM,
        ).strip()
        plain = _plain_summary(text)
        if plain:
            return plain
        log_line("整理说明", detail=text[:300], result="【不是中文说明，改用原句】", level="warning")
        return joined
    except Exception as exc:
        log_line("整理说明", detail=str(exc), result="【改用原句】", level="warning")
        return joined


def _plain_summary(text: str) -> str | None:
    """总结必须是中文句子。模型回了 JSON 或提示词冲突时丢掉。"""
    raw = str(text or "").strip()
    if not raw or raw.startswith("{") or raw.startswith("["):
        return None
    compact = raw.replace(" ", "").replace("\n", "")
    if "逻辑冲突" in compact or "只输出JSON" in compact or "不要输出JSON" in compact:
        return None
    return raw


def _side_members(row: dict) -> list[dict]:
    members = row.get("_side_members")
    if members:
        return list(members)
    return [row]


def collapse_by_project_side(candidates: list[dict]) -> list[dict]:
    """同一项目、同一面只留 IoU 最高的一条去打分，其余同面器件挂在这条上。"""
    from top_side_util import parse_top_side

    groups: dict[tuple, list[dict]] = {}
    order: list[tuple] = []
    for row in candidates or []:
        key = (str(row.get("project_code") or ""), bool(parse_top_side(row.get("top_side"))))
        if key not in groups:
            order.append(key)
            groups[key] = []
        groups[key].append(row)
    reps: list[dict] = []
    for key in order:
        members = groups[key]
        members.sort(key=lambda item: (-float(item.get("iou") or 0), str(item.get("refdes") or "")))
        rep = members[0]
        rep["_side_members"] = members
        reps.append(rep)
    return reps


def _joined_refdes(row: dict) -> str:
    refs = []
    for member in _side_members(row):
        ref = str(member.get("refdes") or "").strip()
        if ref and ref not in refs:
            refs.append(ref)
    return ",".join(refs)


def _attach_highlight(conn, items: list[dict], session_id: str | None = None) -> None:
    from agent_pdf import match_pdf_for_items_batch
    from agent_pdf_api import format_match_pdf
    from top_side_util import parse_top_side, top_side_label

    # 打分代表先定位这一面的 PDF，位号改成逗号串，打开时画出这一面全部框。
    lookup = []
    for item in items:
        joined = _joined_refdes(item)
        lookup.append(
            {
                "project_code": item.get("project_code"),
                "refdes": str(item.get("refdes") or "").strip(),
                "top_side": item.get("top_side"),
                "_joined_refdes": joined,
            }
        )
    try:
        matches = match_pdf_for_items_batch(conn, lookup)
    except Exception as exc:
        log_line("PDF高亮", detail=str(exc), result="【高亮未生成，字段留空】", level="warning")
        for item in items:
            item["pdf"] = None
        return
    for item, probe, match in zip(items, lookup, matches):
        try:
            if match:
                match["refdes"] = probe["_joined_refdes"]
            item["pdf"] = format_match_pdf(conn, match, session_id=session_id)
        except Exception as exc:
            log_line(
                "PDF高亮",
                detail=str(exc),
                result=f"【单条留空】{item.get('project_code')}/{item.get('refdes')}",
                level="warning",
            )
            item["pdf"] = None
        is_top = parse_top_side(item.get("top_side"))
        item["pdf_side"] = top_side_label(is_top)
        item["component_ref"] = probe["_joined_refdes"]


def _library_text(value) -> str | None:
    """器件表里的文本。没有或空白就返回空。"""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _member_texts(row: dict, field: str) -> list[str | None]:
    return [_library_text(member.get(field)) for member in _side_members(row)]


def _member_layers(row: dict) -> list[int | None]:
    """同面每个器件的层数，顺序和 devicetype 那几列一致。"""
    values: list[int | None] = []
    for member in _side_members(row):
        raw = member.get("layers")
        if raw is None or raw == "":
            values.append(None)
            continue
        try:
            values.append(int(raw))
        except (TypeError, ValueError):
            values.append(None)
    return values


def _query_trait_sentence(meta: dict, refdes: str) -> str:
    """查询位号的三个特性直接拼接，不调用模型。"""
    device_type = _library_text(meta.get("devicetype") or meta.get("device_type")) or ""
    package_symbol = _library_text(meta.get("packagesymbol") or meta.get("package_symbol")) or ""
    value = _library_text(meta.get("value")) or ""
    return (
        f"{refdes}的三个特性是devicetype是{device_type}、"
        f"packagesymbol是{package_symbol}、value是{value}"
    )


def _public_item(row: dict) -> dict:
    joined = row.get("component_ref") or _joined_refdes(row)
    return {
        "project_code": row.get("project_code"),
        "refdes": joined,
        "iou": row.get("iou"),
        "device_score": row.get("device_score"),
        "project_score": row.get("project_score"),
        "total_score": row.get("total_score"),
        "score": row.get("total_score"),
        "pdf": row.get("pdf"),
        "pdf_side": row.get("pdf_side"),
        "component_ref": joined,
        "highlight_count": len(_side_members(row)),
        "devicetype": _member_texts(row, "devicetype"),
        "packagesymbol": _member_texts(row, "packagesymbol"),
        "value": _member_texts(row, "value"),
        "layers": _member_layers(row),
        "customer_code": row.get("customer_code"),
        "finish_time": str(row.get("finish_time") or "")[:19] or None,
        "designer_name": row.get("designer_name"),
        "designer_seniority": seniority_grade(row.get("designer_seniority")),
    }


def search_events(
    session_id: str,
    refdes: str,
    content: bytes,
    filename: str,
    top_n: int,
) -> Iterator[dict[str, Any]]:
    """第二段：JSON 只在内存里解析。按前端 top_n 截取后再做高亮和说明。"""
    use_log_session(session_id)
    session = load_session(session_id)
    screen = session.get("search_intent") if isinstance(session.get("search_intent"), dict) else {}
    time_focus = _opt_int(screen.get("time_focus")) == 1
    refdes = str(refdes or "").replace("@", "").strip().upper()
    yield _event("parse_component", "开始检索，正在解析对应器件", refdes=refdes)
    conn = connect()
    try:
        try:
            meta = _query_meta(conn, content, filename, refdes)
            similar = find_similar_from_content(
                conn,
                content,
                source_file=filename or "upload.json",
                library_filters=screen,
            )
            raw_candidates = fetch_candidate_fields(conn, similar.get("results") or [])
            # 同一项目同一面只打分 IoU 最高的一条，这一面仍只占一条返回。
            enriched = collapse_by_project_side(raw_candidates)
        except Exception as exc:
            log_line("对话检索", session_id=session_id, detail=str(exc), result="【解析失败】", level="error")
            yield _event("need_clarify", "需要补充说明", message="器件内容无法解析")
            return
    finally:
        conn.close()

    progress: queue.Queue = queue.Queue()
    holder: dict[str, Any] = {}

    def _progress(batch_no: int, batch_total: int, done: int, total: int) -> None:
        progress.put(
            _event(
                "rank_batch",
                f"正在计算候选顺序，第{batch_no}批，已处理{done}个",
                batch_no=batch_no,
                batch_total=batch_total,
                done=done,
                total=total,
            )
        )

    def _rank() -> None:
        try:
            holder["ranked"] = score_and_sort(meta, enriched, _progress, time_focus=time_focus) if enriched else []
        except Exception as exc:
            holder["error"] = exc
        finally:
            progress.put(None)

    # 打分在另一条线程里，把当前会话号复制过去，日志才不会变成横杠。
    rank_ctx = contextvars.copy_context()
    threading.Thread(target=lambda: rank_ctx.run(_rank), daemon=True).start()
    while True:
        event = progress.get()
        if event is None:
            break
        yield event
    if holder.get("error"):
        log_line(
            "对话检索",
            session_id=session_id,
            detail=str(holder["error"]),
            result="【排序失败】",
            level="error",
        )
        yield _event("need_clarify", "需要补充说明", message="候选顺序暂时算不出来")
        return
    ranked = holder.get("ranked") or []
    yield _event("rank_done", "候选顺序计算完毕", total=len(ranked))
    # 实际条数少于 top_n 时，有多少返回多少。
    limit = int(top_n) if int(top_n) > 0 else len(ranked)
    chosen = ranked[:limit]
    yield _event("highlight", "正在提取版图，正在生成高亮")
    conn = connect()
    try:
        if chosen:
            _attach_highlight(conn, chosen, session_id)
        yield _event("summarize", "正在整理结果")
        reply = _summarize(meta, chosen)
        reply = f"{reply}\n{_query_trait_sentence(meta, refdes)}"
        items = [_public_item(row) for row in chosen]
        log_line(
            "对话检索",
            session_id=session_id,
            result=(
                f"位号={refdes} 过IoU={len(raw_candidates)} 打分={len(enriched)} "
                f"返回={len(items)} top_n={top_n}"
            ),
        )
        yield _event(
            "done",
            reply,
            refdes=refdes,
            top_n=int(top_n),
            reply=reply,
            items=items,
            return_count=len(items),
            candidate_count=len(enriched),
        )
    finally:
        conn.close()


def default_top_n(raw: int | None) -> int:
    if raw is not None and int(raw) > 0:
        return int(raw)
    return int(get_agent_config()["rerank_top_n"])
