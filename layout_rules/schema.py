# -*- coding: utf-8 -*-
"""
规则 DSL 的数据结构与合法性校验。

一条规则 = 元数据（出处、适用范围）+ 分类（kind）+ 执行形态（enforcement）+ 检查器/指标 + 参数。

kind（约束的本质，决定它“应该”落在哪一层）：
    invariant        几何不变量，任何板都成立（重叠、越板框、禁布……）→ 代码机制 + 掩码
    parametric_hard  规范里有明确阈值，但阈值随板类/客户变 → DSL 参数化；error 级进掩码，warning 级进惩罚
    soft             有方向无阈值（尽量近、尽量小）→ 奖励项，权重学出来
    preference       说不清的审美（对齐、朝向一致）→ 不写代码，BC + 偏好判别器学
    knowledge        原理 / 案例 → 只进 RAG 知识库

enforcement（这条规则在系统里实际怎么执行）：
    mask              候选动作掩码（tier=must 一定过滤；tier=should 过滤后若候选为空则自动降级为 penalty）
    penalty           每步惩罚项（scope=placement）
    terminal_penalty  终局惩罚项（scope=layout，整组器件的集合属性）
    eval_only         只在事后检查 / 评测报告里出现，不进训练信号
    learned           交给学习（BC / Critic_expert），代码中不实现
    none              不执行（知识型）

threshold_source（阈值从哪来，做到「代码里没有魔法数字」）：
    geometry | allegro_cm | allegro_dfa | spec_doc | data_mined | si_calibration | expert
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

KINDS = ("invariant", "parametric_hard", "soft", "preference", "knowledge")
ENFORCEMENTS = ("mask", "penalty", "terminal_penalty", "eval_only", "learned", "none")
TIERS = ("must", "should")
SEVERITIES = ("error", "warning", "info")
SCOPES = ("placement", "layout")
THRESHOLD_SOURCES = (
    "geometry", "allegro_cm", "allegro_dfa", "spec_doc",
    "data_mined", "si_calibration", "expert",
)
SHAPES = ("linear", "hinge", "quadratic", "inverse")

# 哪些 kind 允许哪些 enforcement —— 防止把软规则误做成掩码、把知识型做成惩罚
ALLOWED_ENFORCEMENT: dict[str, tuple[str, ...]] = {
    "invariant": ("mask", "eval_only"),
    "parametric_hard": ("mask", "penalty", "terminal_penalty", "eval_only"),
    "soft": ("penalty", "terminal_penalty", "eval_only"),
    "preference": ("learned", "eval_only"),
    "knowledge": ("none",),
}


@dataclass
class RuleSource:
    doc: str = ""
    section: str = ""
    page: Optional[int] = None
    version: str = ""

    def cite(self) -> str:
        parts = [p for p in (self.doc, self.section) if p]
        s = " ".join(parts)
        if self.page is not None:
            s += f" p.{self.page}"
        if self.version:
            s += f" ({self.version})"
        return s or "（无出处）"


@dataclass
class AppliesTo:
    board_type: list[str] = field(default_factory=lambda: ["*"])
    speed_class: list[str] = field(default_factory=lambda: ["*"])
    layer_count_min: Optional[int] = None
    layer_count_max: Optional[int] = None

    def matches(self, board_type: str, speed_class: str, layer_count: int) -> bool:
        if "*" not in self.board_type and board_type not in self.board_type:
            return False
        if "*" not in self.speed_class and speed_class not in self.speed_class:
            return False
        if self.layer_count_min is not None and layer_count < self.layer_count_min:
            return False
        if self.layer_count_max is not None and layer_count > self.layer_count_max:
            return False
        return True


@dataclass
class Shape:
    """软指标 → 惩罚值 的映射形状。惩罚 = weight * f(value)。"""
    type: str = "linear"
    knee: float = 0.0        # hinge：value <= knee 时惩罚为 0
    normalize: float = 1.0   # 先把 value 除以 normalize 再套形状

    def apply(self, value: float) -> float:
        v = value / self.normalize if self.normalize else value
        if self.type == "linear":
            return max(0.0, v)
        if self.type == "hinge":
            return max(0.0, v - self.knee / (self.normalize or 1.0))
        if self.type == "quadratic":
            return max(0.0, v) ** 2
        if self.type == "inverse":
            # 值越大越好（如均匀性 0~1）：惩罚 = 1 - v
            return max(0.0, 1.0 - v)
        raise ValueError(f"未知 shape.type: {self.type}")


@dataclass
class Rule:
    id: str
    name: str
    kind: str
    enforcement: str
    scope: str = "placement"
    tier: str = "must"
    severity: str = "error"
    check: Optional[str] = None          # 检查器名（checks.py 注册表）
    metric: Optional[str] = None         # 指标名（metrics.py 注册表）
    params: dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    shape: Shape = field(default_factory=Shape)
    threshold_source: str = "spec_doc"
    needs_calibration: bool = False
    source: RuleSource = field(default_factory=RuleSource)
    applies_to: AppliesTo = field(default_factory=AppliesTo)
    message: str = ""
    fix_hint: str = ""
    tests: list[dict[str, Any]] = field(default_factory=list)   # 正 / 负例夹具引用（selfcheck 用）
    notes: str = ""

    # ---- 派生属性 ----
    @property
    def is_mask(self) -> bool:
        return self.enforcement == "mask"

    @property
    def is_step_penalty(self) -> bool:
        return self.enforcement == "penalty" and self.scope == "placement"

    @property
    def is_terminal_penalty(self) -> bool:
        return self.enforcement == "terminal_penalty" or (
            self.enforcement == "penalty" and self.scope == "layout"
        )

    @property
    def executable(self) -> bool:
        """是否需要代码执行（learned / none 不需要）。"""
        return self.enforcement in ("mask", "penalty", "terminal_penalty", "eval_only")

    def validate(self) -> list[str]:
        """返回问题列表；为空表示合法。"""
        errs: list[str] = []
        if self.kind not in KINDS:
            errs.append(f"kind 非法: {self.kind}")
        if self.enforcement not in ENFORCEMENTS:
            errs.append(f"enforcement 非法: {self.enforcement}")
        if self.kind in ALLOWED_ENFORCEMENT and self.enforcement not in ALLOWED_ENFORCEMENT[self.kind]:
            errs.append(
                f"kind={self.kind} 不允许 enforcement={self.enforcement}"
                f"（允许: {ALLOWED_ENFORCEMENT[self.kind]}）"
            )
        if self.tier not in TIERS:
            errs.append(f"tier 非法: {self.tier}")
        if self.severity not in SEVERITIES:
            errs.append(f"severity 非法: {self.severity}")
        if self.scope not in SCOPES:
            errs.append(f"scope 非法: {self.scope}")
        if self.threshold_source not in THRESHOLD_SOURCES:
            errs.append(f"threshold_source 非法: {self.threshold_source}")
        if self.shape.type not in SHAPES:
            errs.append(f"shape.type 非法: {self.shape.type}")
        if self.executable and not (self.check or self.metric):
            errs.append("可执行规则必须指定 check 或 metric")
        if self.check and self.metric:
            errs.append("check 与 metric 只能指定一个")
        if self.enforcement == "mask" and self.metric:
            errs.append("mask 形态必须用 check（布尔判定），不能用 metric")
        if self.enforcement == "mask" and self.scope != "placement":
            errs.append("mask 只能作用于 placement 级（单步可判定）规则")
        if self.enforcement == "mask" and self.severity == "warning" and self.tier == "must":
            errs.append("warning 级规则不应作为 must 掩码（建议 tier=should 或改 penalty）")
        return errs


@dataclass
class Ruleset:
    name: str
    version: str
    rules: list[Rule]
    description: str = ""

    def by_id(self, rule_id: str) -> Rule:
        for r in self.rules:
            if r.id == rule_id:
                return r
        raise KeyError(rule_id)

    def validate(self) -> dict[str, list[str]]:
        problems: dict[str, list[str]] = {}
        seen: set[str] = set()
        for r in self.rules:
            errs = r.validate()
            if r.id in seen:
                errs.append("规则 id 重复")
            seen.add(r.id)
            if errs:
                problems[r.id] = errs
        return problems
