# -*- coding: utf-8 -*-
"""检索后处理：形态候选 → 从库取四字段 → LLM 判断排序 → 按 project_code+refdes 取 PDF 高亮。"""

from __future__ import annotations

from agent_llm_rerank import llm_rank_candidates
from agent_pdf import match_pdf_for_items_batch
from agent_pdf_api import format_match_pdf
from footprint_db import fetch_component_row
from settings import get_agent_config


def fetch_candidate_fields(conn, candidates: list[dict]) -> list[dict]:
    """形态 IoU 候选 → 补齐库里的 layers/devicetype/packagesymbol/value 等字段。"""
    out: list[dict] = []
    for raw in candidates or []:
        code = raw.get("project_code")
        ref = str(raw.get("refdes") or "").replace("@", "").strip().upper()
        if not code or not ref:
            continue
        row = fetch_component_row(conn, code, ref)
        if not row:
            continue
        out.append(
            {
                "project_code": code,
                "board_name": row.get("name"),
                "refdes": ref,
                "component_ref": ref,
                "layers": row.get("layers"),
                "devicetype": row.get("devicetype"),
                "device_type": row.get("devicetype"),
                "packagesymbol": row.get("packagesymbol"),
                "package_symbol": row.get("packagesymbol"),
                "value": row.get("value"),
                "pin_count": row.get("pin_count"),
                "top_side": row.get("top_side") if row.get("top_side") is not None else True,
                "customer_code": row.get("customer_code"),
                "designer_name": row.get("designer_name"),
                "designer_seniority": row.get("designer_seniority"),
                "seniority_score": row.get("designer_seniority"),
                "finish_time": row.get("finish_time"),
                "iou": raw.get("iou"),
                "bbox": {
                    "minx": raw.get("minx") or row.get("bbox_min_x"),
                    "miny": raw.get("miny") or row.get("bbox_min_y"),
                    "maxx": raw.get("maxx") or row.get("bbox_max_x"),
                    "maxy": raw.get("maxy") or row.get("bbox_max_y"),
                },
            }
        )
    return out


def rank_with_llm(query_meta: dict, candidates: list[dict], top_n: int | None = None) -> dict:
    """LLM 判断（替代向量库）。top_n 未指定时用 yaml agent.rerank_top_n（默认 10）。失败抛 LlmRankError。"""
    cfg = get_agent_config()
    top_n = top_n or cfg["rerank_top_n"]
    return llm_rank_candidates(query_meta, candidates, top_n=top_n)


def attach_pdf(conn, items: list[dict]) -> list[dict]:
    """按 project_code + top_side 取板级 PDF，并按位号运行时算高亮框。"""
    matches = match_pdf_for_items_batch(conn, items)
    for item, match in zip(items, matches):
        from top_side_util import parse_top_side, top_side_label

        is_top = parse_top_side(item.get("top_side"))
        item["pdf"] = format_match_pdf(conn, match)
        item["pdf_side"] = top_side_label(is_top)
        item["pdf_keywords"] = item.get("refdes") or item.get("component_ref")
        item["component_ref"] = item.get("refdes") or item.get("component_ref")
        item["score"] = item.get("final_score")
    return items


def rerank_similar_results(
    conn,
    *,
    candidates: list[dict],
    query_meta: dict,
    query_refdes: str,
    top_n: int | None = None,
) -> dict:
    """一步完成（供脚本/旧调用）：取字段 → LLM 排序 → PDF。图里是拆成三个节点执行。"""
    enriched = fetch_candidate_fields(conn, candidates)
    ranked = rank_with_llm(query_meta, enriched, top_n=top_n)
    items = attach_pdf(conn, ranked["items"])
    return {
        "items": items,
        "return_count": len(items),
        "candidate_count": len(candidates or []),
        "pdf_hits": [x["pdf"] for x in items if x.get("pdf")],
        "mode": "iou_then_llm",
        "llm_reason": ranked.get("reason"),
        "top_n": ranked.get("top_n"),
    }
