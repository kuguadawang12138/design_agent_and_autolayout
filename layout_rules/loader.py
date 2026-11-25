# -*- coding: utf-8 -*-
"""
YAML 规则文件加载、校验、按板类筛选。

规则文件格式见 rules/decap_v0.yaml；字段含义见 schema.py。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

import yaml

from .schema import AppliesTo, Rule, RuleSource, Ruleset, Shape

DEFAULT_RULES_DIR = Path(__file__).resolve().parent / "rules"


def _as_list(v: Any) -> list:
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return list(v)
    return [v]


def _parse_rule(raw: dict[str, Any]) -> Rule:
    src = raw.get("source") or {}
    app = raw.get("applies_to") or {}
    shp = raw.get("shape") or {}
    return Rule(
        id=str(raw["id"]),
        name=str(raw.get("name", raw["id"])),
        kind=str(raw.get("kind", "parametric_hard")),
        enforcement=str(raw.get("enforcement", "eval_only")),
        scope=str(raw.get("scope", "placement")),
        tier=str(raw.get("tier", "must")),
        severity=str(raw.get("severity", "error")),
        check=raw.get("check"),
        metric=raw.get("metric"),
        params=dict(raw.get("params") or {}),
        weight=float(raw.get("weight", 1.0)),
        shape=Shape(
            type=str(shp.get("type", "linear")),
            knee=float(shp.get("knee", 0.0)),
            normalize=float(shp.get("normalize", 1.0)),
        ),
        threshold_source=str(raw.get("threshold_source", "spec_doc")),
        needs_calibration=bool(raw.get("needs_calibration", False)),
        source=RuleSource(
            doc=str(src.get("doc", "")),
            section=str(src.get("section", "")),
            page=src.get("page"),
            version=str(src.get("version", "")),
        ),
        applies_to=AppliesTo(
            board_type=[str(x) for x in (_as_list(app.get("board_type")) or ["*"])],
            speed_class=[str(x) for x in (_as_list(app.get("speed_class")) or ["*"])],
            layer_count_min=app.get("layer_count_min"),
            layer_count_max=app.get("layer_count_max"),
        ),
        message=str(raw.get("message", "")),
        fix_hint=str(raw.get("fix_hint", "")),
        tests=list(raw.get("tests") or []),
        notes=str(raw.get("notes", "")),
    )


def load_ruleset(path: str | os.PathLike | None = None, strict: bool = True) -> Ruleset:
    """
    加载一个规则文件。path 为空时加载默认的 rules/decap_v0.yaml。
    strict=True 时任何校验错误都抛 ValueError，避免带病规则进入训练环境。
    """
    p = Path(path) if path is not None else DEFAULT_RULES_DIR / "decap_v0.yaml"
    with open(p, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    rules = [_parse_rule(r) for r in (raw.get("rules") or [])]
    rs = Ruleset(
        name=str(raw.get("ruleset", p.stem)),
        version=str(raw.get("version", "0")),
        rules=rules,
        description=str(raw.get("description", "")),
    )
    problems = rs.validate()
    if problems and strict:
        lines = [f"规则文件 {p} 校验失败："]
        for rid, errs in problems.items():
            for e in errs:
                lines.append(f"  [{rid}] {e}")
        raise ValueError("\n".join(lines))
    return rs


def select_rules(rules: Iterable[Rule], board_type: str, speed_class: str = "default",
                 layer_count: int = 4) -> list[Rule]:
    """按 applies_to 选出对当前板生效的规则（知识库→规则集选择器的最简实现）。"""
    return [r for r in rules if r.applies_to.matches(board_type, speed_class, layer_count)]
