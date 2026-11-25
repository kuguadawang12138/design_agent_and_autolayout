# -*- coding: utf-8 -*-
"""LangGraph 对话编排（每一步一个节点）：

init_session → parse_intent → resolve_json → similar_footprint（IoU）
→ fetch_fields（从库取四字段）→ llm_rank（LLM 判断排序）→ pdf_highlight → reply
"""

from __future__ import annotations

import time
from typing import Any

from agent_llm_rerank import LlmRankError, llm_compose_reply, llm_parse_intent
from agent_log import log_line
from agent_timing import StepTimer
from agent_rerank import fetch_candidate_fields, rank_with_llm
from agent_session import append_chat_turn, llm_summary, load_session, remember_last_query, resolve_refdes
from footprint_db import connect, find_similar_from_content
from json_fields import query_fields_from_meta
from settings import get_agent_config


def similar_from_json(conn, content: bytes, filename: str) -> dict:
    """直接调用形态比对函数，不走 HTTP，检索永不入库。"""
    return find_similar_from_content(conn, content, source_file=filename)


def build_reply(state: dict) -> str:
    if state.get("clarification"):
        return state["clarification"]
    if state.get("llm_reply"):
        return state["llm_reply"]
    ref = state.get("target_refdes")
    similar = state.get("similar") or {}
    items = state.get("items") or []
    n_sim = len(similar.get("results") or [])
    parts = [f"已定位位号 {ref}，完成形态 IoU 检索，形态库返回 {n_sim} 组候选。"]
    if items:
        want = state.get("top_n") or len(items)
        parts.append(f"经 LLM 在候选范围内判断后返回 {len(items)} 个（要求前 {want} 个），已附 PDF 高亮信息。")
        top = items[0]
        parts.append(f"最相关：{top.get('project_code')} / {top.get('refdes')}。")
        if state.get("llm_reason"):
            parts.append(f"依据：{state['llm_reason']}")
    else:
        parts.append("暂无可展示的历史模块。")
    return "".join(parts)


def _compile_graph(conn, session_id: str):
    from langgraph.graph import END, START, StateGraph

    def _step(state: dict, name: str, **extra) -> dict:
        return {
            **state,
            **extra,
            "nodes": [*(state.get("nodes") or []), name],
            "_step_ms": state.get("_step_ms") or {},
        }

    def _run_node(state: dict, step_name: str, fn):
        t0 = time.perf_counter()
        out = fn(state)
        ms = int((time.perf_counter() - t0) * 1000)
        if isinstance(out, dict):
            steps = dict(state.get("_step_ms") or {})
            for k, v in (out.get("_step_ms") or {}).items():
                if k != step_name:
                    steps[k] = v
            steps[step_name] = ms
            out["_step_ms"] = steps
        return out

    def init_session(state: dict) -> dict:
        def _body(s: dict) -> dict:
            session = load_session(s["session_id"])
            return _step(s, "init_session", session=session, refdes_index=llm_summary(session))

        return _run_node(state, "init_session", _body)

    def node_parse_intent(state: dict) -> dict:
        """用 LLM 理解用户要查哪个位号、调用哪个 tool（不用正则抽位号）。"""
        def _body(s: dict) -> dict:
            session = s.get("session") or {}
            message = s.get("message") or ""
            try:
                intent = llm_parse_intent(message, session)
            except LlmRankError as exc:
                log_line("LLM意图", session_id=session_id, detail=str(exc), result="【失败】", level="error")
                return _step(
                    s,
                    "parse_intent",
                    clarification=f"意图理解失败：{exc}",
                    intent={"intent": "clarification", "intent_source": "llm"},
                )
            top_n = intent.get("top_n") or get_agent_config()["rerank_top_n"]
            out = _step(
                s,
                "parse_intent",
                intent=intent,
                target_refdes=intent.get("target_refdes"),
                top_n=top_n,
            )
            if intent.get("need_clarification") or not intent.get("target_refdes"):
                out["clarification"] = intent.get("clarification_message") or "请说明要查询的位号（refdes）。"
            return out

        return _run_node(state, "parse_intent", _body)

    def node_resolve_json(state: dict) -> dict:
        def _body(s: dict) -> dict:
            try:
                resolved = resolve_refdes(s["session_id"], s.get("target_refdes"))
            except (ValueError, FileNotFoundError) as exc:
                return _step(s, "resolve_json", clarification=str(exc))
            return _step(s, "resolve_json", resolved=resolved)

        return _run_node(state, "resolve_json", _body)

    def node_similar(state: dict) -> dict:
        """步骤 1：形态 IoU 检索，缩小候选范围。"""

        def _body(s: dict) -> dict:
            resolved = s.get("resolved") or {}
            try:
                similar = similar_from_json(conn, resolved["bytes"], resolved["file_name"])
            except Exception as exc:
                log_line(
                    "形态检索",
                    session_id=session_id,
                    detail=str(exc),
                    result="【IoU检索失败】",
                    level="error",
                )
                return _step(
                    s,
                    "similar_footprint",
                    clarification=f"形态检索失败：{exc}",
                    similar={"results": [], "error": str(exc)},
                )
            sub_steps = dict(similar.pop("_step_ms", None) or {})
            s["_similar_count"] = len(similar.get("results") or [])
            out = _step(s, "similar_footprint", similar=similar)
            merged = dict(out.get("_step_ms") or {})
            merged.update(sub_steps)
            out["_step_ms"] = merged
            return out

        return _run_node(state, "similar_footprint", _body)

    def node_fetch_fields(state: dict) -> dict:
        def _body(s: dict) -> dict:
            from minio_store import measure_fetches

            cands = (s.get("similar") or {}).get("results") or []
            with measure_fetches() as acc:
                enriched = fetch_candidate_fields(conn, cands)
            json_slice = acc.slice("source_json")
            if json_slice.count:
                log_line(
                    "拉取候选字段",
                    session_id=session_id,
                    result=f"【MinIO拉取器件JSON】{json_slice.format('source_json')}",
                )
            s["_candidate_fields_count"] = len(enriched)
            out = _step(s, "fetch_fields", candidates=enriched)
            merged = dict(out.get("_step_ms") or {})
            merged["fetch_minio_json"] = json_slice.ms
            out["_step_ms"] = merged
            return out

        return _run_node(state, "fetch_fields", _body)

    def node_llm_rank(state: dict) -> dict:
        def _body(s: dict) -> dict:
            meta = (s.get("resolved") or {}).get("meta") or {}
            try:
                ranked = rank_with_llm(meta, s.get("candidates") or [], top_n=s.get("top_n"))
            except LlmRankError as exc:
                log_line("LLM排序", session_id=session_id, detail=str(exc), result="【失败】", level="error")
                return _step(s, "llm_rank", clarification=f"LLM 判断失败：{exc}", items=[])
            remember_last_query(s["session_id"], refdes=s.get("target_refdes"), top_n=s.get("top_n"))
            return _step(
                s,
                "llm_rank",
                items=ranked["items"],
                llm_reason=ranked.get("reason"),
                rank_prompt=ranked.get("prompt"),
            )

        return _run_node(state, "llm_rank", _body)

    def node_pdf(state: dict) -> dict:
        def _body(s: dict) -> dict:
            from agent_pdf import match_pdf_for_items_batch
            from agent_pdf_api import format_match_pdf
            from top_side_util import parse_top_side, top_side_label

            from minio_store import measure_fetches

            items = list(s.get("items") or [])
            sub = StepTimer()
            with sub.measure("pdf_match"):
                with measure_fetches() as acc_match:
                    matches = match_pdf_for_items_batch(conn, items)
            with sub.measure("pdf_format"):
                with measure_fetches() as acc_fmt:
                    for item, match in zip(items, matches):
                        is_top = parse_top_side(item.get("top_side"))
                        item["pdf"] = format_match_pdf(conn, match)
                        item["pdf_side"] = top_side_label(is_top)
                        item["pdf_keywords"] = item.get("refdes") or item.get("component_ref")
                        item["component_ref"] = item.get("refdes") or item.get("component_ref")
                        item["score"] = item.get("final_score")
            pdf_match_sl = acc_match.slice("pdf")
            json_match_sl = acc_match.slice("source_json")
            pdf_fmt_sl = acc_fmt.slice("pdf")
            if acc_match.count or acc_fmt.count:
                log_line(
                    "PDF高亮",
                    session_id=session_id,
                    result=(
                        f"【MinIO】匹配PDF {pdf_match_sl.format('pdf')} "
                        f"匹配JSON {json_match_sl.format('source_json')} "
                        f"链接PDF {pdf_fmt_sl.format('pdf')}"
                    ),
                )
            out = _step(s, "pdf_highlight", items=items, pdf_hits=[x["pdf"] for x in items if x.get("pdf")])
            merged = dict(out.get("_step_ms") or {})
            for k, v in sub._steps.items():
                merged[f"pdf_{k}"] = v
            merged["pdf_minio_pdf"] = pdf_match_sl.ms
            merged["pdf_minio_json"] = json_match_sl.ms
            merged["pdf_format_minio"] = pdf_fmt_sl.ms
            out["_step_ms"] = merged
            return out

        return _run_node(state, "pdf_highlight", _body)

    def node_reply(state: dict) -> dict:
        def _body(s: dict) -> dict:
            meta = (s.get("resolved") or {}).get("meta") or {}
            llm_reply = None
            if s.get("items") and not s.get("clarification"):
                llm_reply = llm_compose_reply(meta, s.get("items") or [], s.get("llm_reason") or "")
            return _step(s, "reply", llm_reply=llm_reply)

        return _run_node(state, "reply", _body)

    def node_clarification(state: dict) -> dict:
        return _run_node(state, "clarification", lambda s: _step(s, "clarification"))

    def after_intent(state: dict) -> str:
        return "clarification" if state.get("clarification") or not state.get("target_refdes") else "resolve_json"

    def after_resolve(state: dict) -> str:
        return "clarification" if state.get("clarification") or not state.get("resolved") else "similar_footprint"

    def after_similar(state: dict) -> str:
        if state.get("clarification"):
            return "clarification"
        if not (state.get("similar") or {}).get("results"):
            return "reply"
        return "fetch_fields"

    def after_llm_rank(state: dict) -> str:
        return "clarification" if state.get("clarification") else "pdf_highlight"

    graph = StateGraph(dict)
    graph.add_node("init_session", init_session)
    graph.add_node("parse_intent", node_parse_intent)
    graph.add_node("resolve_json", node_resolve_json)
    graph.add_node("similar_footprint", node_similar)
    graph.add_node("fetch_fields", node_fetch_fields)
    graph.add_node("llm_rank", node_llm_rank)
    graph.add_node("pdf_highlight", node_pdf)
    graph.add_node("reply", node_reply)
    graph.add_node("clarification", node_clarification)

    graph.add_edge(START, "init_session")
    graph.add_edge("init_session", "parse_intent")
    graph.add_conditional_edges("parse_intent", after_intent, {"resolve_json": "resolve_json", "clarification": "clarification"})
    graph.add_conditional_edges("resolve_json", after_resolve, {"similar_footprint": "similar_footprint", "clarification": "clarification"})
    graph.add_conditional_edges(
        "similar_footprint",
        after_similar,
        {"fetch_fields": "fetch_fields", "clarification": "clarification", "reply": "reply"},
    )
    graph.add_edge("fetch_fields", "llm_rank")
    graph.add_conditional_edges("llm_rank", after_llm_rank, {"pdf_highlight": "pdf_highlight", "clarification": "clarification"})
    graph.add_edge("pdf_highlight", "reply")
    graph.add_edge("clarification", END)
    graph.add_edge("reply", END)
    return graph.compile()


def run_agent(session_id: str, message: str) -> dict:
    t_wall = time.perf_counter()
    conn = connect()
    try:
        app = _compile_graph(conn, session_id)
        state = app.invoke({"session_id": session_id, "message": message, "nodes": [], "_step_ms": {}})
    finally:
        conn.close()
    payload = _to_payload(state)
    steps = dict(state.get("_step_ms") or {})
    try:
        t0 = time.perf_counter()
        append_chat_turn(
            session_id,
            user=message,
            assistant=str(payload.get("reply") or payload.get("clarification") or ""),
        )
        steps["chat_memory"] = int((time.perf_counter() - t0) * 1000)
    except Exception as exc:
        log_line(
            "对话记忆",
            session_id=session_id,
            detail=str(exc),
            result="【写入会话历史失败】",
            level="warning",
        )
    wall_ms = int((time.perf_counter() - t_wall) * 1000)
    similar_n = state.get("_similar_count")
    if similar_n is None:
        similar_n = len((state.get("similar") or {}).get("results") or [])
    detail = (
        f"位号={state.get('target_refdes') or '-'} "
        f"形态候选={similar_n} 返回条数={len(state.get('items') or [])}"
    )
    perf = StepTimer()
    perf.merge(steps)
    perf.log("对话检索", session_id=session_id, detail=detail, total_ms=wall_ms)
    payload["step_ms"] = steps
    payload["response_ms"] = wall_ms
    return payload


def _to_payload(state: dict) -> dict:
    session = state.get("session") or {}
    resolved = state.get("resolved") or {}
    similar = state.get("similar") or {}
    items = state.get("items") or []
    reply = build_reply(state)
    out: dict[str, Any] = {
        "session_id": state.get("session_id"),
        "nodes": state.get("nodes") or [],
        "target_refdes": state.get("target_refdes"),
        "clarification": state.get("clarification"),
        "reply": reply,
        "refdes_index": state.get("refdes_index") or llm_summary(session),
        "intent": (state.get("intent") or {}).get("intent"),
        "mode": "iou_then_llm",
        "graph": "init_session→parse_intent→resolve_json→similar_footprint→fetch_fields→llm_rank→pdf_highlight→reply",
        "tools": ["find_similar_history_modules"],
        "llm_reason": state.get("llm_reason"),
        "top_n": state.get("top_n"),
    }
    if resolved:
        out["json_file"] = resolved.get("file_name")
        out["meta"] = resolved.get("meta")
        out["query"] = query_fields_from_meta(resolved.get("meta"))
    out["similar"] = {"results": similar.get("results") or [], "error": similar.get("error")}
    out["items"] = items
    out["pdf_hits"] = state.get("pdf_hits") or []
    out["return_count"] = len(items)
    out["candidate_count"] = len(state.get("candidates") or [])
    out["response_ms"] = 0
    return out
