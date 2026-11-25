# -*- coding: utf-8 -*-
"""
规则编译器：把同一份 Ruleset 编译成三种执行形态。

    compiled = compile_ruleset(ruleset, board)
    m = compiled.mask(candidates, ctx)                 # 事中：动作掩码（含 must/should 分级降级）
    p = compiled.step_penalty(cand, ctx, m.degraded)   # 事中：每步惩罚
    t = compiled.terminal_penalty(placements, ctx)     # 事中：终局惩罚
    r = compiled.evaluate(placements, ctx)             # 事后：全量检查报告（规则引擎 / 评测共用）

这样“检查标准”和“生成标准”天然一致（docs/02 ADR-02），不会出现 AI 生成的布局通不过自己检查的情况。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .checks import CheckResult, get_check
from .metrics import get_metric
from .model import Board, Candidate, Component, PlacementContext
from .schema import Rule, Ruleset
from .loader import select_rules


@dataclass
class BoundRule:
    """规则 + 已解析的可调用对象。"""
    rule: Rule
    fn: Any

    def run_check(self, target, ctx: PlacementContext) -> CheckResult:
        return self.fn(target, ctx, **self.rule.params)

    def run_metric(self, target, ctx: PlacementContext) -> float:
        return float(self.fn(target, ctx, **self.rule.params))

    def penalty_of(self, target, ctx: PlacementContext) -> tuple[float, str]:
        """把该规则在 target 上的结果换算成惩罚值（>=0）。"""
        r = self.rule
        if r.check:
            res = self.run_check(target, ctx)
            return (0.0 if res.ok else r.weight * res.amount), res.detail
        val = self.run_metric(target, ctx)
        return r.weight * r.shape.apply(val), f"{r.metric}={val:.3f}"


@dataclass
class MaskResult:
    mask: list[bool]
    reasons: dict[int, list[str]] = field(default_factory=dict)   # 候选下标 → 被哪些规则拒绝
    degraded: list[str] = field(default_factory=list)             # 因候选为空而降级为惩罚的 should 规则
    n_after_must: int = 0
    n_after_should: int = 0

    @property
    def legal_indices(self) -> list[int]:
        return [i for i, ok in enumerate(self.mask) if ok]


@dataclass
class PenaltyBreakdown:
    total: float = 0.0
    terms: dict[str, float] = field(default_factory=dict)
    details: dict[str, str] = field(default_factory=dict)

    def add(self, rule_id: str, value: float, detail: str = "") -> None:
        if value:
            self.terms[rule_id] = self.terms.get(rule_id, 0.0) + value
            self.total += value
        if detail:
            self.details[rule_id] = detail


@dataclass
class Violation:
    rule_id: str
    severity: str
    refdes: str
    message: str
    source: str
    fix_hint: str
    amount: float


@dataclass
class EvalReport:
    violations: list[Violation] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    skipped_rules: list[str] = field(default_factory=list)   # learned / none 形态，不由代码执行

    @property
    def hard_violations(self) -> int:
        return sum(1 for v in self.violations if v.severity == "error")

    @property
    def soft_violations(self) -> int:
        return sum(1 for v in self.violations if v.severity != "error")

    def summary(self) -> str:
        lines = [
            f"违反项 {len(self.violations)}（error {self.hard_violations} / warning+info {self.soft_violations}）",
        ]
        for v in self.violations:
            lines.append(f"  [{v.severity.upper()}] {v.rule_id} {v.refdes}: {v.message}  ← {v.source}")
        if self.metrics:
            lines.append("指标：")
            for k, val in self.metrics.items():
                lines.append(f"  {k} = {val:.3f}")
        if self.skipped_rules:
            lines.append(f"未由代码执行的规则（learned/none）: {', '.join(self.skipped_rules)}")
        return "\n".join(lines)


class CompiledRuleset:
    def __init__(self, ruleset: Ruleset, active_rules: list[Rule]):
        self.ruleset = ruleset
        self.rules = active_rules
        self.mask_must: list[BoundRule] = []
        self.mask_should: list[BoundRule] = []
        self.step_penalties: list[BoundRule] = []
        self.terminal_penalties: list[BoundRule] = []
        self.eval_placement: list[BoundRule] = []     # 事后检查：所有 placement 级可执行规则
        self.eval_layout: list[BoundRule] = []        # 事后检查：所有 layout 级可执行规则
        self.non_executable: list[Rule] = []
        self._bind()

    # ------------------------------------------------------------------ #
    def _bind(self) -> None:
        for r in self.rules:
            if not r.executable:
                self.non_executable.append(r)
                continue
            if r.check:
                fn = get_check(r.check, r.scope)
            else:
                fn = get_metric(r.metric, r.scope)
            b = BoundRule(r, fn)
            if r.is_mask:
                (self.mask_must if r.tier == "must" else self.mask_should).append(b)
            elif r.is_step_penalty:
                self.step_penalties.append(b)
            elif r.is_terminal_penalty:
                self.terminal_penalties.append(b)
            (self.eval_placement if r.scope == "placement" else self.eval_layout).append(b)

    # ------------------------------------------------------------------ #
    # 形态 1：动作掩码
    # ------------------------------------------------------------------ #
    def mask(self, candidates: list[Candidate], ctx: PlacementContext) -> MaskResult:
        n = len(candidates)
        mask = [True] * n
        reasons: dict[int, list[str]] = {}

        def apply(bound_rules: list[BoundRule]) -> None:
            for i, cand in enumerate(candidates):
                if not mask[i]:
                    continue
                for b in bound_rules:
                    res = b.run_check(cand, ctx)
                    if not res.ok:
                        mask[i] = False
                        reasons.setdefault(i, []).append(f"{b.rule.id}: {res.detail}")
                        break

        apply(self.mask_must)
        n_must = sum(mask)
        degraded: list[str] = []
        if self.mask_should and n_must > 0:
            snapshot = list(mask)
            apply(self.mask_should)
            if sum(mask) == 0:
                # should 级把可行集砍空 → 回退到 must 结果，should 规则降级为惩罚
                mask = snapshot
                for i in list(reasons.keys()):
                    reasons[i] = [x for x in reasons[i] if not any(x.startswith(b.rule.id) for b in self.mask_should)]
                    if not reasons[i]:
                        del reasons[i]
                degraded = [b.rule.id for b in self.mask_should]
        return MaskResult(mask=mask, reasons=reasons, degraded=degraded,
                          n_after_must=n_must, n_after_should=sum(mask))

    # ------------------------------------------------------------------ #
    # 形态 2：惩罚项
    # ------------------------------------------------------------------ #
    def step_penalty(self, cand: Candidate, ctx: PlacementContext,
                     degraded: Optional[list[str]] = None) -> PenaltyBreakdown:
        out = PenaltyBreakdown()
        for b in self.step_penalties:
            v, d = b.penalty_of(cand, ctx)
            out.add(b.rule.id, v, d)
        if degraded:
            for b in self.mask_should:
                if b.rule.id in degraded:
                    v, d = b.penalty_of(cand, ctx)
                    out.add(b.rule.id, v, d)
        return out

    def terminal_penalty(self, placements: list[Component], ctx: PlacementContext) -> PenaltyBreakdown:
        out = PenaltyBreakdown()
        for b in self.terminal_penalties:
            v, d = b.penalty_of(placements, ctx)
            out.add(b.rule.id, v, d)
        return out

    # ------------------------------------------------------------------ #
    # 形态 3：事后全量检查
    # ------------------------------------------------------------------ #
    def evaluate(self, placements: list[Component], ctx: PlacementContext) -> EvalReport:
        """
        对已完成的一组摆放做全量检查。逐个器件检查时，把该器件从 ctx.placed 里摘出来，
        其余器件作为环境，避免“自己和自己重叠”。
        """
        report = EvalReport(skipped_rules=[r.id for r in self.non_executable])
        base_placed = [c for c in ctx.placed if c.refdes not in {p.refdes for p in placements}]
        for comp in placements:
            others = base_placed + [p for p in placements if p.refdes != comp.refdes]
            sub = PlacementContext(board=ctx.board, chip=ctx.chip, placed=others, vias=ctx.vias,
                                   dfa=ctx.dfa, inductance_coeffs=ctx.inductance_coeffs)
            sub.refresh()
            cand = Candidate(comp=comp, x=comp.x, y=comp.y, rot_deg=comp.rot_deg, side=comp.side)
            for b in self.eval_placement:
                r = b.rule
                if r.check:
                    res = b.run_check(cand, sub)
                    if not res.ok:
                        report.violations.append(Violation(
                            rule_id=r.id, severity=r.severity, refdes=comp.refdes,
                            message=r.message or res.detail, source=r.source.cite(),
                            fix_hint=r.fix_hint, amount=res.amount))
                else:
                    key = f"{r.metric}"
                    report.metrics[key] = report.metrics.get(key, 0.0) + b.run_metric(cand, sub) / max(1, len(placements))
        for b in self.eval_layout:
            r = b.rule
            if r.check:
                res = b.run_check(placements, ctx)
                if not res.ok:
                    report.violations.append(Violation(
                        rule_id=r.id, severity=r.severity, refdes="*",
                        message=r.message or res.detail, source=r.source.cite(),
                        fix_hint=r.fix_hint, amount=res.amount))
            else:
                report.metrics[r.metric] = b.run_metric(placements, ctx)
        return report

    # ------------------------------------------------------------------ #
    def describe(self) -> str:
        def ids(lst):
            return ", ".join(b.rule.id for b in lst) or "-"
        return "\n".join([
            f"规则集 {self.ruleset.name} v{self.ruleset.version}：生效 {len(self.rules)} 条",
            f"  掩码 must   : {ids(self.mask_must)}",
            f"  掩码 should : {ids(self.mask_should)}",
            f"  每步惩罚    : {ids(self.step_penalties)}",
            f"  终局惩罚    : {ids(self.terminal_penalties)}",
            f"  仅评测      : {', '.join(b.rule.id for b in self.eval_placement + self.eval_layout if b.rule.enforcement == 'eval_only') or '-'}",
            f"  学习/知识   : {', '.join(r.id for r in self.non_executable) or '-'}",
        ])


def compile_ruleset(ruleset: Ruleset, board: Optional[Board] = None) -> CompiledRuleset:
    """按板类筛选生效规则并绑定检查器 / 指标。board 为空时全部规则生效。"""
    if board is None:
        active = list(ruleset.rules)
    else:
        active = select_rules(ruleset.rules, board.board_type, board.speed_class, board.layer_count)
    return CompiledRuleset(ruleset, active)
