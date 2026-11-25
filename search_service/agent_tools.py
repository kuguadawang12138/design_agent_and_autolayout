# -*- coding: utf-8 -*-
"""Agent 工具定义：供 LangGraph / 后续 tool-calling 使用。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class AgentTool:
    name: str
    description: str
    handler: Callable[..., Any]
    intent_keywords: tuple[str, ...] = ()


FIND_SIMILAR_HISTORY = AgentTool(
    name="find_similar_history_modules",
    description=(
        "当用户想查找历史板子上形态/封装相似的器件、相似设计复用、历史模块推荐时调用。"
        "输入：会话中的位号 refdes；输出：历史 project_code+refdes 列表、重排得分、PDF 高亮信息。"
    ),
    handler=None,  # 由 agent_graph 注入
    intent_keywords=("相似", "历史", "封装", "复用", "推荐", "找", "查", "模块", "类似"),
)


def detect_similar_intent(message: str) -> bool:
    text = (message or "").strip()
    if not text:
        return False
    return any(kw in text for kw in FIND_SIMILAR_HISTORY.intent_keywords)


def list_tools() -> list[dict]:
    return [
        {
            "name": FIND_SIMILAR_HISTORY.name,
            "description": FIND_SIMILAR_HISTORY.description,
            "intent_keywords": list(FIND_SIMILAR_HISTORY.intent_keywords),
        }
    ]
