# -*- coding: utf-8 -*-
"""LLM 判断节点：形态 IoU 缩小范围后，用大模型在候选范围内排序并返回前 N 个。

做法（写进提示词）：
  1. 把当前器件与每个形态候选的规格参数（devicetype / value / model / brand 等，有什么给什么）交给 LLM，
     让它尽可能推断各自的 型号、系列、厂商、类别（大类）、应用特性、核心参数；提取不出来的留空，不编造。
  2. 按推断结果比对相似度，权重合计 80%，优先级：型号 > 系列 > 厂商 > 类别 > 应用特性 > 核心参数。
  3. 项目属性合计 20%（有才算）：customer_code 8%（相同优先）、finish_time 6%（越晚优先）、
     designer_seniority 5%（越高优先）、designer_name 1%（相同优先）。
  4. 只输出结果：候选范围内的 project_code + refdes 排序，前 N 个（默认 10，用户可指定）。
板层数、封装符号不参与 LLM 比对。
严格只在形态候选范围内排序；LLM 未启用、调用失败或输出不可解析时直接抛 LlmRankError，不做规则兜底。
LLM 连接参数全部来自 config/footprint_sim.yaml 的 llm 段。
"""

from __future__ import annotations

import json
import re

import httpx

from agent_log import log_line
from agent_tools import list_tools
from settings import get_agent_config, get_llm_config

_JSON_RE = re.compile(r"\{[\s\S]*\}")
_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


class LlmRankError(RuntimeError):
    """LLM 判断失败（未启用 / 请求失败 / 输出不可解析 / 无有效结果）。"""


# 默认提示词；可在 yaml 的 llm.rank_prompt 覆盖，占位符 {query} / {candidates} / {top_n}
# 候选带 [序号]，模型只返回 indices，JSON 更短、不易被截断。
# 候选一行一个，字段用「键=值」紧凑排布，只列非空字段，尽量省 token。
DEFAULT_RANK_PROMPT = (
    "你是资深电子元器件工程师，按规格相似度给历史候选排序。\n"
    "字段含义：器件=器件类型/型号 值=参数值 型号 厂商 客户=客户编码 完成=项目完成日期 资历=工程师资历分 设计师。\n"
    "当前器件：{query}\n"
    "历史候选（[序号] 只能使用这些序号）：\n"
    "{candidates}\n"
    "权重：器件信息 80%（型号>系列>厂商>类别>应用特性>核心参数，提不出留空不编造）；"
    "项目属性 20%（客户相同 8%、完成越晚 6%、资历越高 5%、设计师相同 1%）。\n"
    "返回最相似的前 {top_n} 个序号；候选不足则全部返回。\n"
    "只输出一个 JSON 对象，不要思考过程、不要 markdown、不要解释：\n"
    '{"indices":[0,2,1],"reason":"一句话依据"}'
)


_REFDES_TOKEN = re.compile(r"^[A-Z]{1,6}\d+[A-Z0-9]*$", re.IGNORECASE)
_REFDES_IN_TEXT = re.compile(r"[A-Za-z]{1,6}\d+[A-Za-z0-9]*")
_TOP_N_RE = re.compile(r"(?:前|top\s*)(\d{1,3})|(\d{1,3})\s*(?:个|条|款)", re.IGNORECASE)
_FOLLOWUP_RE = re.compile(r"再|继续|更多|还有|它|这个|上一个|同一个|刚才")


def _parse_top_n_rule(text: str) -> int | None:
    """规则抽「前 N 个 / N 个 / top N」。"""
    m = _TOP_N_RE.search(text or "")
    if not m:
        return None
    n = int(m.group(1) or m.group(2) or 0)
    return n if n > 0 else None


def rule_parse_intent_fallback(message: str, session: dict) -> dict | None:
    """
    规则识别意图，能识别就不调 LLM：
      1) 消息里恰好出现 1 个已上传位号 → 直接检索（可带「前 N 个」）；
      2) 没有位号但有「再/继续/它/N 个」等续问词且有上次查询 → 沿用上次位号。
    多个位号或无法判断时返回 None，交给 LLM / 追问。
    """
    raw = (message or "").strip()
    if not raw:
        return None
    known = {(x.get("refdes") or "").upper() for x in (session.get("items") or []) if x.get("refdes")}
    if not known:
        return None

    top_n = _parse_top_n_rule(raw)

    def _make(ref: str, source: str) -> dict:
        return {
            "tool": "find_similar_history_modules",
            "target_refdes": ref,
            "top_n": top_n,
            "intent": "find_similar_history",
            "need_clarification": False,
            "clarification_message": "",
            "intent_source": source,
        }

    hits: list[str] = []
    for tok in _REFDES_IN_TEXT.findall(raw):
        u = tok.upper()
        if u in known and u not in hits:
            hits.append(u)
    if len(hits) == 1:
        return _make(hits[0], "rule_refdes")
    if len(hits) > 1:
        return None

    last = session.get("last_query") or {}
    last_ref = str(last.get("refdes") or "").strip().upper()
    if last_ref and last_ref in known and (top_n or _FOLLOWUP_RE.search(raw)):
        return _make(last_ref, "rule_last_query")
    return None


DEFAULT_INTENT_PROMPT = (
    "你是 PCB 设计助手，从用户话里判断要查哪个位号。\n"
    "当前板已上传位号（target_refdes 只能从中选，禁止编造）：{refdes_index}\n"
    "上次查询：{last_query}\n"
    "近期对话：{chat_history}\n"
    "工具：{tools}\n"
    "用户消息：{message}\n"
    "只输出 JSON，不要思考过程、不要其它文字：\n"
    '{{"tool":"find_similar_history_modules"|null,"target_refdes":"位号或null","top_n":整数或null,'
    '"need_clarification":true或false,"clarification_message":"需追问时填写否则空字符串"}}\n'
    "规则：提到位号则取大写位号且必须在列表中；找相似/历史/复用/推荐 tool=find_similar_history_modules；"
    "「前 N 个」填 top_n；位号不确定或不在列表、看不懂意图时 need_clarification=true 并礼貌追问。"
)


_INTENT_REFDES_MAX = 400
_INTENT_HISTORY_CHARS = 80


def _clean(d: dict) -> dict:
    return {k: v for k, v in d.items() if v not in (None, "")}


def _query_fields(query_meta: dict | None) -> dict:
    """当前器件交给 LLM 的规格参数：不含板层数、封装符号。"""
    meta = query_meta or {}
    return _clean(
        {
            "devicetype": meta.get("device_type") or meta.get("devicetype"),
            "value": meta.get("value"),
            "model": meta.get("model"),
            "brand": meta.get("brand"),
            "customer_code": meta.get("customer_code"),
            "designer_name": meta.get("designer_name"),
            "designer_seniority": meta.get("designer_seniority") or meta.get("seniority_score"),
            "finish_time": _fmt_time(meta.get("finish_time")),
        }
    )


def _fmt_time(value):
    if value in (None, ""):
        return None
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    return str(value)[:10]


def _candidate_brief(c: dict) -> dict:
    """形态候选交给 LLM 的规格参数：不含板层数、封装符号。"""
    return _clean(
        {
            "project_code": c.get("project_code"),
            "refdes": c.get("refdes"),
            "devicetype": c.get("devicetype") or c.get("device_type"),
            "value": c.get("value"),
            "model": c.get("model"),
            "brand": c.get("brand"),
            "customer_code": c.get("customer_code"),
            "designer_name": c.get("designer_name"),
            "designer_seniority": c.get("designer_seniority") or c.get("seniority_score"),
            "finish_time": _fmt_time(c.get("finish_time")),
        }
    )


# 紧凑「键=值」排布用的字段顺序与中文短标签（不含 project_code/refdes：序号已能唯一定位）
_BRIEF_LABELS: tuple[tuple[str, str], ...] = (
    ("devicetype", "器件"),
    ("value", "值"),
    ("model", "型号"),
    ("brand", "厂商"),
    ("customer_code", "客户"),
    ("finish_time", "完成"),
    ("designer_seniority", "资历"),
    ("designer_name", "设计师"),
)


def _brief_line(brief: dict) -> str:
    """{"devicetype":"LT8612","value":"..."} → 「器件=LT8612 值=...」，只列非空字段。"""
    parts = [f"{label}={brief[key]}" for key, label in _BRIEF_LABELS if brief.get(key) not in (None, "")]
    return " ".join(parts) or "无参数"


def _format_candidates_for_rank(candidates: list[dict]) -> str:
    lines: list[str] = []
    for i, c in enumerate(candidates):
        lines.append(f"[{i}] {_brief_line(_candidate_brief(c))}")
    return "\n".join(lines)


def build_rank_prompt(query_meta: dict, candidates: list[dict], top_n: int, template: str | None = None) -> str:
    tpl = template or DEFAULT_RANK_PROMPT
    q = _brief_line(_query_fields(query_meta))
    cands = _format_candidates_for_rank(candidates)
    return tpl.replace("{query}", q).replace("{candidates}", cands).replace("{top_n}", str(top_n))


def _balanced_json_objects(text: str) -> list[str]:
    """按花括号配对切出所有顶层 {...} 片段（思考段里夹杂多个 JSON 时用）。"""
    out: list[str] = []
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text or ""):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                out.append(text[start : i + 1])
                start = -1
    return out


def _repair_json_trailing_commas(text: str) -> str:
    """去掉 JSON 中常见的尾逗号，便于解析模型输出。"""
    prev = None
    out = text
    while prev != out:
        prev = out
        out = re.sub(r",(\s*[}\]])", r"\1", out)
    return out


def _extract_json_blob(text: str) -> str:
    text = (text or "").strip()
    fence = _FENCE_RE.search(text)
    if fence:
        return fence.group(1).strip()
    match = _JSON_RE.search(text)
    if match:
        return match.group(0)
    return text


def _parse_json_object(text: str, *, context: str = "LLM") -> dict:
    """尽量从模型输出中提取 JSON 对象；失败时记录原文便于排查。"""
    raw = text or ""
    blob = _extract_json_blob(raw)
    candidates = [blob, _repair_json_trailing_commas(blob)]
    # 思考段混入多个 JSON 时，从后往前逐个尝试（最后一个通常是最终答案）
    for piece in reversed(_balanced_json_objects(raw)):
        candidates.append(piece)
        candidates.append(_repair_json_trailing_commas(piece))
    last_exc: json.JSONDecodeError | None = None
    for item in candidates:
        try:
            data = json.loads(item)
        except json.JSONDecodeError as exc:
            last_exc = exc
            continue
        if isinstance(data, dict):
            return data
        raise LlmRankError(f"{context} 输出 JSON 根节点必须是对象")
    log_line("LLM解析", detail=raw[:4000], result="【JSON解析失败】", level="error")
    if last_exc is not None:
        raise LlmRankError(f"{context} 输出 JSON 解析失败：{last_exc}") from last_exc
    raise LlmRankError(f"{context} 输出中没有 JSON")


# 关思考的请求体字段：vLLM / SGLang / Ollama(Qwen3) 通用；其它框架可用 yaml llm.extra_body 覆盖
_THINK_OFF_BODY: dict = {"chat_template_kwargs": {"enable_thinking": False}}
# Qwen3 软开关：用户消息末尾加 /no_think 即关闭思考，对其它模型是无害的普通文本
_NO_THINK_SUFFIX = " /no_think"
# 这些顶层字段在部分严格的 OpenAI 兼容服务上会 400，出错时逐个去掉重试
_OPTIONAL_BODY_KEYS = ("response_format", "chat_template_kwargs")


def _unpack_chat_response(data: dict) -> tuple[str, str, str, dict]:
    """返回 (正文, 思考段, finish_reason, usage)。思考段兼容 reasoning_content / reasoning。"""
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content = str(msg.get("content") or "")
    reasoning = str(msg.get("reasoning_content") or msg.get("reasoning") or "")
    finish = str(choice.get("finish_reason") or "unknown")
    usage = data.get("usage") or {}
    return content, reasoning, finish, usage


def call_llm_chat(
    prompt: str,
    cfg: dict | None = None,
    *,
    json_mode: bool | None = None,
    purpose: str = "LLM",
    temperature: float | None = None,
    system_prompt: str | None = None,
) -> str:
    """
    OpenAI 兼容 /chat/completions 调用，参数来自 yaml llm 段。
    - disable_thinking=true（默认）：请求体加 enable_thinking=false 并在提示词末尾加 /no_think，
      避免思考型模型把 max_tokens 花在 reasoning 上、正文为空（finish_reason=length）。
    - 正文为空但思考段里已有 JSON 时，退而取思考段解析。
    - 每次调用打一行中文日志：耗时、输入/输出 token、finish_reason，便于定位慢与截断。
    """
    import time

    cfg = cfg or get_llm_config()
    headers = {"Content-Type": "application/json"}
    if cfg.get("api_key"):
        headers["Authorization"] = f"Bearer {cfg['api_key']}"
    for k, v in (cfg.get("extra_headers") or {}).items():
        headers[str(k)] = str(v)
    use_json = cfg.get("json_mode", True) if json_mode is None else json_mode
    think_off = bool(cfg.get("disable_thinking", True))
    user_prompt = f"{prompt}{_NO_THINK_SUFFIX}" if think_off else prompt
    # 系统提示只跟这一次调用走。要 JSON 的打分不带到「不要 JSON」的总结上。
    if system_prompt is not None:
        system = system_prompt
    elif use_json:
        system = cfg.get("system_prompt") or "你是严谨的电子元器件比对助手，只输出 JSON。"
    else:
        system = "按用户这一条的要求用中文作答。不要输出 JSON，不要讨论提示词是否冲突。"
    body: dict = {
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": cfg.get("temperature", 0.1) if temperature is None else float(temperature),
        "max_tokens": cfg.get("max_tokens", 2048),
    }
    if use_json:
        body["response_format"] = {"type": "json_object"}
    if think_off:
        body.update(json.loads(json.dumps(_THINK_OFF_BODY)))
    for k, v in (cfg.get("extra_body") or {}).items():
        body[str(k)] = v
    url = cfg["base_url"].rstrip("/") + cfg.get("chat_path", "/chat/completions")
    timeout = cfg.get("timeout_sec", 60)

    t0 = time.perf_counter()
    resp = httpx.post(url, headers=headers, json=body, timeout=timeout)
    # 400：可能是服务不认 response_format / chat_template_kwargs，逐个去掉重试
    for key in _OPTIONAL_BODY_KEYS:
        if resp.status_code != 400 or key not in body:
            continue
        body.pop(key, None)
        log_line("LLM调用", detail=f"用途={purpose}", result=f"【400 去掉 {key} 重试】", level="warning")
        resp = httpx.post(url, headers=headers, json=body, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    content, reasoning, finish, usage = _unpack_chat_response(data)
    ms = int((time.perf_counter() - t0) * 1000)
    log_line(
        "LLM调用",
        detail=f"用途={purpose} 模型={cfg['model']} 提示词字符={len(user_prompt)}",
        result=(
            f"耗时={ms}ms finish={finish} "
            f"输入token={usage.get('prompt_tokens', '-')} 输出token={usage.get('completion_tokens', '-')} "
            f"正文字符={len(content.strip())} 思考字符={len(reasoning)}"
        ),
    )
    if content.strip():
        return content
    # 正文为空：思考型模型把额度花在 reasoning 上。若思考段里已有 JSON，直接拿来用
    if reasoning and _balanced_json_objects(reasoning):
        log_line(
            "LLM调用",
            detail=f"用途={purpose}",
            result="【正文为空，改用思考段中的 JSON】",
            level="warning",
        )
        return reasoning
    hint = (
        "疑似思考型模型把 token 花在推理上；请确认 yaml llm.disable_thinking=true 对该服务生效，"
        "或通过 llm.extra_body 传该框架的关思考参数，或改用非思考（instruct）模型"
    )
    raise LlmRankError(f"LLM 返回空内容（finish_reason={finish}，思考字符={len(reasoning)}）。{hint}")


def _pairs_from_indices(indices: list, candidates: list[dict]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for raw in indices:
        try:
            idx = int(raw)
        except (TypeError, ValueError):
            continue
        if idx < 0 or idx >= len(candidates):
            continue
        c = candidates[idx]
        code = str(c.get("project_code") or "").strip()
        ref = str(c.get("refdes") or "").strip().upper()
        if code and ref:
            out.append((code, ref))
    return out


def _parse_ranked(text: str, candidates: list[dict]) -> tuple[list[tuple[str, str]], str]:
    data = _parse_json_object(text, context="LLM 排序")
    reason = str(data.get("reason") or "")

    indices = data.get("indices")
    if indices is None:
        indices = data.get("order")
    if isinstance(indices, list) and indices:
        pairs = _pairs_from_indices(indices, candidates)
        if pairs:
            return pairs, reason

    ranked = data.get("ranked") or []
    out: list[tuple[str, str]] = []
    for item in ranked:
        if isinstance(item, dict):
            code = str(item.get("project_code") or "").strip()
            ref = str(item.get("refdes") or item.get("component_ref") or "").strip().upper()
            if code and ref:
                out.append((code, ref))
        elif isinstance(item, (int, float)):
            out.extend(_pairs_from_indices([item], candidates))
        elif isinstance(item, str) and ":" in item:
            code, ref = item.split(":", 1)
            out.append((code.strip(), ref.strip().upper()))
    if out:
        return out, reason

    # 兜底：从原文抠 indices 数组（模型 JSON 半截时）
    m = re.search(r'"indices"\s*:\s*\[([\d,\s]+)\]', text or "")
    if m:
        nums = [int(x.strip()) for x in m.group(1).split(",") if x.strip().isdigit()]
        pairs = _pairs_from_indices(nums, candidates)
        if pairs:
            return pairs, reason

    raise LlmRankError("LLM 排序结果中无有效 indices 或 ranked")


def llm_rank_candidates(query_meta: dict, candidates: list[dict], top_n: int = 10) -> dict:
    """返回 {"items": 排序后的候选（仅 LLM 提到且在候选范围内的）, "reason", "prompt", "top_n"}。

    - 严格只接受形态候选范围内的 project_code+refdes，范围外一概丢弃；
    - 不追加 LLM 未提到的候选，不做规则兜底；任何失败抛 LlmRankError。
    """
    if not candidates:
        return {"items": [], "reason": "", "prompt": "", "top_n": top_n}
    cfg = get_llm_config()
    if not cfg.get("enabled"):
        raise LlmRankError("LLM 未启用，请在 config/footprint_sim.yaml 的 llm 段设置 enabled: true")
    top_n = max(1, int(top_n or 10))
    # 形态 IoU 已按相似度排好序，只把前 rank_max_candidates 个交给 LLM，控制提示词长度
    max_c = int(cfg.get("rank_max_candidates") or 0)
    if max_c > 0 and len(candidates) > max_c:
        log_line(
            "LLM排序",
            detail=f"候选数={len(candidates)}",
            result=f"【裁剪】只取形态前 {max_c} 个交给 LLM",
        )
        candidates = list(candidates[:max_c])
    prompt = build_rank_prompt(query_meta, candidates, top_n, cfg.get("rank_prompt"))
    try:
        text = call_llm_chat(prompt, cfg, purpose="排序")
    except LlmRankError:
        raise
    except Exception as exc:
        log_line("LLM排序", detail=str(exc), result="【请求失败】", level="error")
        raise LlmRankError(f"LLM 请求失败：{exc}") from exc
    text_retry = ""
    try:
        pairs, reason = _parse_ranked(text, candidates)
    except LlmRankError as exc:
        log_line("LLM排序", detail=(text or "")[:4000], result="【首次解析失败，重试】", level="warning")
        fix_prompt = (
            '只输出 JSON：{"indices":[按相似度排序的候选序号], "reason":"一句话"}。'
            "不要思考过程、不要 markdown、不要解释。序号必须来自候选列表。\n\n"
            f"{prompt}"
        )
        text_retry = call_llm_chat(fix_prompt, cfg, purpose="排序重试")
        try:
            pairs, reason = _parse_ranked(text_retry, candidates)
        except LlmRankError:
            log_line("LLM排序", detail=(text_retry or text or "")[:4000], result="【重试后仍解析失败】", level="error")
            raise LlmRankError(f"LLM 排序 JSON 解析失败：{exc}") from exc
    index = {(str(c.get("project_code")), str(c.get("refdes") or "").upper()): c for c in candidates}
    ordered: list[dict] = []
    seen = set()
    dropped = 0
    for key in pairs:
        c = index.get(key)
        if c is None:
            dropped += 1
            continue
        if key in seen:
            continue
        item = dict(c)
        item["llm_rank"] = len(ordered) + 1
        item["llm_reason"] = reason
        item["final_score"] = round(1.0 - len(ordered) / max(len(candidates), 1), 4)
        ordered.append(item)
        seen.add(key)
        if len(ordered) >= top_n:
            break
    if not ordered:
        raise LlmRankError("LLM 未返回任何候选范围内的 project_code+refdes")
    log_line(
        "LLM排序",
        detail=f"候选数={len(candidates)} 要求前={top_n}",
        result=(
            f"返回={len(ordered)} 丢弃超范围={dropped} "
            f"理由={reason[:100]}"
        ),
    )
    return {"items": ordered, "reason": reason, "prompt": prompt, "top_n": top_n}


def build_intent_prompt(
    message: str,
    session: dict,
    template: str | None = None,
    *,
    history_turns: int = 4,
) -> str:
    """
    意图提示词尽量短：位号只给名字（不给封装/类型/pin 数），对话只带最近几条且截断，
    工具只给名字和一句描述。
    """
    tpl = template or DEFAULT_INTENT_PROMPT
    refs = [str(x.get("refdes")).upper() for x in (session.get("items") or []) if x.get("refdes")]
    if len(refs) > _INTENT_REFDES_MAX:
        refdes_index = ",".join(refs[:_INTENT_REFDES_MAX]) + f"...（共{len(refs)}个）"
    else:
        refdes_index = ",".join(refs)
    last_query = json.dumps(session.get("last_query") or {}, ensure_ascii=False)
    from agent_session import recent_chat_history

    hist = [
        {"role": h.get("role"), "content": str(h.get("content") or "")[:_INTENT_HISTORY_CHARS]}
        for h in recent_chat_history(session, limit=max(0, int(history_turns)))
    ]
    chat_history = json.dumps(hist, ensure_ascii=False) if hist else "无"
    tools = json.dumps(
        [{"name": t.get("name"), "description": t.get("description")} for t in list_tools()],
        ensure_ascii=False,
    )
    return (
        tpl.replace("{message}", message or "")
        .replace("{refdes_index}", refdes_index)
        .replace("{last_query}", last_query)
        .replace("{chat_history}", chat_history)
        .replace("{tools}", tools)
    )


def llm_parse_intent(message: str, session: dict) -> dict:
    """用 LLM 从用户话里提取 tool / target_refdes / top_n；不编造位号，不确定则追问。"""
    ruled = rule_parse_intent_fallback(message, session)
    if ruled:
        log_line(
            "意图识别",
            detail=f"来源={ruled.get('intent_source')} 未调LLM",
            result=f"位号={ruled.get('target_refdes') or '-'} 条数={ruled.get('top_n') or '-'}",
        )
        return ruled

    known = {(x.get("refdes") or "").upper() for x in (session.get("items") or []) if x.get("refdes")}
    if not get_agent_config().get("llm_intent_enabled", True):
        # 规则识别不出且配置不走 LLM：直接追问
        refs = sorted(known)
        hint = "、".join(refs[:12])
        log_line("意图识别", detail="规则未识别且 llm_intent_enabled=false", result="【直接追问】")
        return {
            "tool": None,
            "target_refdes": None,
            "top_n": None,
            "intent": "clarification",
            "need_clarification": True,
            "clarification_message": (
                f"请直接说出要查询的位号（如 {refs[0] if refs else 'U1'}）。当前板已上传 {len(refs)} 个器件"
                + (f"：{hint}" if hint else "")
            ),
            "intent_source": "rule_clarify",
        }

    cfg = get_llm_config()
    if not cfg.get("enabled"):
        raise LlmRankError("LLM 未启用，请在 config/footprint_sim.yaml 的 llm 段设置 enabled: true")
    prompt = build_intent_prompt(
        message,
        session,
        cfg.get("intent_prompt"),
        history_turns=int(cfg.get("intent_history_turns") or 4),
    )
    try:
        text = call_llm_chat(prompt, cfg, purpose="意图")
        data = _parse_json_object(text, context="LLM 意图")
    except LlmRankError:
        fix_prompt = (
            "你上次输出的不是合法 JSON。请只重新输出完整 JSON 对象，字段："
            "tool, target_refdes, top_n, need_clarification, clarification_message。\n\n"
            f"{prompt}"
        )
        text = call_llm_chat(fix_prompt, cfg, purpose="意图重试")
        data = _parse_json_object(text, context="LLM 意图")

    tool = data.get("tool")
    target = str(data.get("target_refdes") or "").strip().upper() or None
    top_n_raw = data.get("top_n")
    top_n = int(top_n_raw) if top_n_raw not in (None, "") else None
    if top_n is not None and top_n <= 0:
        top_n = None
    need_clarification = bool(data.get("need_clarification"))
    clarification_message = str(data.get("clarification_message") or "").strip()

    last = session.get("last_query") or {}
    if not target and top_n and last.get("refdes"):
        target = str(last["refdes"]).upper()

    if target and target not in known:
        need_clarification = True
        clarification_message = clarification_message or f"位号 {target} 不在当前板已上传列表中，请确认要查哪个位号。"
        target = None

    if not target and not need_clarification:
        ruled2 = rule_parse_intent_fallback(message, session)
        if ruled2:
            return ruled2

    if tool == "find_similar_history_modules" and not target and not need_clarification:
        refs = sorted(known)
        hint = "、".join(refs[:12])
        need_clarification = True
        clarification_message = clarification_message or (
            f"请指定位号（refdes）。当前板已索引 {len(refs)} 个器件" + (f"：{hint}" if hint else "")
        )

    if not tool and target and not need_clarification:
        tool = "find_similar_history_modules"

    intent = "find_similar_history" if (tool == "find_similar_history_modules" or target) and target else "clarification"
    if need_clarification:
        intent = "clarification"

    log_line(
        "LLM意图",
        detail=f"工具={tool} 位号={target} 条数={top_n}",
        result=f"需追问={need_clarification}",
    )
    return {
        "tool": tool,
        "target_refdes": target,
        "top_n": top_n,
        "intent": intent,
        "need_clarification": need_clarification,
        "clarification_message": clarification_message,
        "intent_source": "llm",
    }


def llm_compose_reply(query_meta: dict, items: list[dict], reason: str) -> str | None:
    """可选：让 LLM 用一段自然语言向用户总结推荐结果（llm.reply_enabled=true 时生效）。"""
    cfg = get_llm_config()
    if not cfg.get("enabled") or not cfg.get("reply_enabled"):
        return None
    brief = [
        {
            "rank": i + 1,
            "project_code": x.get("project_code"),
            "refdes": x.get("refdes"),
            "devicetype": x.get("devicetype") or x.get("device_type"),
            "value": x.get("value"),
            "pdf": bool(x.get("pdf")),
        }
        for i, x in enumerate(items[:5])
    ]
    prompt = (
        "你是 PCB 设计助手。请用不超过 120 字的中文，向工程师总结下面的历史相似器件推荐结果，"
        "指出排第一的为什么最相关，并提醒可点击 PDF 查看板图高亮。不要输出 JSON。\n"
        f"当前器件：{json.dumps(_query_fields(query_meta), ensure_ascii=False)}\n"
        f"排序依据：{reason}\n"
        f"推荐列表：{json.dumps(brief, ensure_ascii=False)}"
    )
    try:
        return call_llm_chat(prompt, cfg, json_mode=False, purpose="回复").strip() or None
    except Exception as exc:
        log_line("LLM回复", detail=str(exc), result="【自然语言总结失败】", level="warning")
        return None


# 器件信息交给模型打档。缺型号、缺厂商不再按 0 分扣光，所以不会出现个位数和九十几同时出现。
# 项目属性用公式，不进模型。
DEVICE_SCORE_PROMPT = (
    "你是资深电子元器件工程师。只比较器件本身，不要考虑客户、完成时间、资历、设计师。\n"
    "这些候选的封装形状已经接近，分数只区分器件信息，不要从 0 打到 100。\n"
    "当前器件：{query}\n"
    "候选（只能使用这些序号）：\n{candidates}\n"
    "只能打这五档里的一个整数：\n"
    "82：器件类型、值、型号都对得上。\n"
    "74：器件类型相同，值接近，或型号是同一系列。\n"
    "66：器件类型相同，值不同。\n"
    "58：类型相关，但不是同一种器件。\n"
    "50：类型明显不同。\n"
    "某一侧没有型号或厂商时，不要因为缺字段降档，这项按相同处理。不要打这五档以外的分数。\n"
    "每个序号都要给分。只输出一个 JSON，不要解释：\n"
    '{"scores":[{"index":0,"score":74,"reason":"一句话"}]}'
)
# 模型仍可能打出个位数。收到这五档上，原始分留在日志里。
_DEVICE_BANDS = (50, 58, 66, 74, 82)

_DEVICE_LABELS: tuple[tuple[str, str], ...] = (
    ("devicetype", "器件"),
    ("value", "值"),
    ("model", "型号"),
    ("brand", "厂商"),
)


def iou_tier(iou: float) -> int:
    """1.00 单独一档。其余先留四位，再按 0.01 向下归档。返回值越大档越高。"""
    value = round(float(iou or 0), 4)
    if value >= 1:
        return 100
    return int(value * 100 + 1e-6)


def _device_line(row: dict) -> str:
    parts = [f"{label}={row[key]}" for key, label in _DEVICE_LABELS if row.get(key) not in (None, "")]
    return " ".join(parts) or "无参数"


def _parse_device_scores(text: str, count: int) -> dict[int, tuple[float, str]]:
    data = _parse_json_object(text, context="器件打分")
    rows = data.get("scores") or data.get("items") or []
    out: dict[int, tuple[float, str]] = {}
    if isinstance(rows, dict):
        rows = [{"index": k, "score": v} for k, v in rows.items()]
    for item in rows:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("index"))
        except (TypeError, ValueError):
            continue
        if index < 0 or index >= count or index in out:
            continue
        try:
            score = float(item.get("score"))
        except (TypeError, ValueError):
            score = 0.0
        raw = max(0.0, min(100.0, score))
        out[index] = (_limit_device_score(raw), str(item.get("reason") or "").strip(), raw)
    return out


def _limit_device_score(score: float) -> float:
    """落在五档上。低于 50 的收到 50，高于 82 的收到 82。"""
    try:
        value = float(score)
    except (TypeError, ValueError):
        value = 50.0
    if value < 50:
        return 50.0
    if value > 82:
        return 82.0
    return float(min(_DEVICE_BANDS, key=lambda band: abs(band - value)))


def score_device_batch(query_meta: dict, batch: list[dict]) -> dict[int, tuple]:
    """一批最多 20 条。模型只给器件信息打分。失败时这一批记 0 分，不中断整次检索。"""
    if not batch:
        return {}
    cfg = get_llm_config()
    if not cfg.get("enabled"):
        raise LlmRankError("LLM 未启用，请在 config/footprint_sim.yaml 的 llm 段设置 enabled: true")
    lines = [f"[{i}] {_device_line(_candidate_brief(row))}" for i, row in enumerate(batch)]
    prompt = (
        DEVICE_SCORE_PROMPT.replace("{query}", _device_line(_query_fields(query_meta)))
        .replace("{candidates}", "\n".join(lines))
    )
    try:
        text = call_llm_chat(
            prompt,
            cfg,
            purpose="器件打分",
            temperature=0,
            system_prompt="你是严谨的电子元器件比对助手。这一次只输出 JSON。",
        )
        return _parse_device_scores(text, len(batch))
    except Exception as exc:
        log_line("器件打分", detail=str(exc), result=f"【本批记 0 分】条数={len(batch)}", level="warning")
        return {}
