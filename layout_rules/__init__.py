# -*- coding: utf-8 -*-
"""
layout_rules —— PCB 布局约束的「规则 DSL → 三种执行形态」最小实现。

设计要点（详见 docs/11-约束分层与执行策略.md）：
  1. 规范原文（知识库）不直接参与 RL 决策；
  2. 规则以 YAML DSL 表达（对象 / 检查器 / 阈值 / severity / applies_to / source）；
  3. 同一份 DSL 编译成三种形态：
       - mask     ：候选动作掩码（硬约束，模型物理上无法输出非法动作）
       - penalty  ：每步 / 终局奖励惩罚项（软约束、warning 级参数化规则）
       - evaluate ：事后全量检查报告（规则引擎 & 评测共用）
  4. 代码只写「机制」，阈值全部来自 DSL / Allegro 导出 / 历史数据挖掘 / SI 校准。

对外主要入口：
    from layout_rules import load_ruleset, compile_ruleset
"""

from .loader import load_ruleset, select_rules
from .compiler import compile_ruleset, CompiledRuleset
from .model import (
    Board,
    Component,
    Pin,
    Via,
    Candidate,
    PlacementContext,
    KeepoutRegion,
    HeightRegion,
)

__all__ = [
    "load_ruleset",
    "select_rules",
    "compile_ruleset",
    "CompiledRuleset",
    "Board",
    "Component",
    "Pin",
    "Via",
    "Candidate",
    "PlacementContext",
    "KeepoutRegion",
    "HeightRegion",
]
