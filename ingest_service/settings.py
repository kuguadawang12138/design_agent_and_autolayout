# -*- coding: utf-8 -*-
"""
统一 YAML 配置。项目根 = 本文件所在目录（与操作系统、cd 无关）。

1) 必须先有 config/deploy_env.yaml，且 env 为 prd 或 st（仅此文件决定环境，缺则无法启动）
2) 再加载 config/footprint_sim_prd.yaml 或 footprint_sim_st.yaml
3) 也可 --config 指定 footprint 配置路径（仍须存在 deploy_env.yaml）
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError("请安装 PyYAML: pip install pyyaml") from exc

# 项目根目录：任意 py 与 config/ 同级即可
ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class SimilarityConfig:
    pin_filter: bool
    min_iou: float
    top_k: int
    exclude_self: bool
    area_tolerance: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "pin_filter": self.pin_filter,
            "min_iou": self.min_iou,
            "top_k": self.top_k,
            "exclude_self": self.exclude_self,
            "area_tolerance": self.area_tolerance,
            "rule": "pin + 面积浮动初筛 → IoU 精排 → min_iou 阈值 → Top-K；exclude_self 按 project_code+refdes",
        }


@dataclass(frozen=True)
class ApiConfig:
    host: str
    port: int
    uvicorn_workers: int


@dataclass(frozen=True)
class ConcurrencyConfig:
    """API /similar 真并发与 IoU 进程池配置。"""

    total_cpu_cores: int
    reserved_cpu_cores: int
    max_concurrent_similar: int
    iou_workers_per_request: int
    iou_pool_max_workers: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_cpu_cores": self.total_cpu_cores,
            "reserved_cpu_cores": self.reserved_cpu_cores,
            "available_cpu_cores": self.available_cpu_cores(),
            "max_concurrent_similar": self.max_concurrent_similar,
            "iou_workers_per_request": self.iou_workers_per_request,
            "iou_pool_max_workers": self.iou_pool_max_workers,
            "peak_iou_processes_hint": self.iou_pool_max_workers,
        }

    def available_cpu_cores(self) -> int:
        total = self.total_cpu_cores or (os.cpu_count() or 4)
        return max(1, total - self.reserved_cpu_cores)


@dataclass(frozen=True)
class BatchConfig:
    compare_workers: int
    restore_workers: int
    benchmark_max_records: int
    benchmark_max_pairs: int


def project_root() -> Path:
    return ROOT


DEPLOY_ENV_FILENAME = "deploy_env.yaml"


def deploy_env_path() -> Path:
    return ROOT / "config" / DEPLOY_ENV_FILENAME


def load_deploy_env() -> str:
    """读取 config/deploy_env.yaml，env 必须为 prd 或 st。"""
    path = deploy_env_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"缺少 {path}，请复制 config/deploy_env.example.yaml 为 deploy_env.yaml，并设置 env: prd 或 env: st"
        )
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} 顶层必须是 YAML 对象")
    raw = data.get("env")
    if raw is None or not str(raw).strip():
        raise ValueError(f"{path} 必须填写 env: prd 或 env: st")
    text = str(raw).strip().lower()
    if text in ("prd", "prod", "production"):
        return "prd"
    if text in ("st", "test", "testing", "staging", "dev"):
        return "st"
    raise ValueError(f"{path} 中 env={raw!r} 无效，只允许 prd 或 st")


def normalize_config_env() -> str:
    return load_deploy_env()


def config_file_name(env: str) -> str:
    return f"footprint_sim_{env}.yaml"


def config_path_for_env(env: str) -> Path:
    return ROOT / "config" / config_file_name(env)


def default_config_path() -> Path:
    return config_path_for_env(load_deploy_env())


def example_config_path(env: str | None = None) -> Path:
    e = env if env in ("prd", "st") else load_deploy_env()
    return ROOT / "config" / f"footprint_sim_{e}.example.yaml"


def find_config_path(explicit: Path | str | None = None) -> Path:
    env = load_deploy_env()
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"配置文件不存在: {path}")
        return path
    primary = config_path_for_env(env)
    if not primary.is_file():
        raise FileNotFoundError(
            f"deploy_env.yaml 指定 env={env}，但未找到 {primary}。"
            f"请复制 config/footprint_sim_{env}.example.yaml → config/{config_file_name(env)}"
        )
    return primary.resolve()


# bootstrap 之后，后面的 get_mysql() 都读这一份，避免和日志里打印的不是同一个文件。
_ACTIVE_CONFIG: str | None = None


@lru_cache
def load_settings(config_path: str | None = None) -> dict[str, Any]:
    path = find_config_path(config_path or _ACTIVE_CONFIG)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"配置文件顶层必须是对象: {path}")
    return data


def _resolve_path(raw: str, *, base: Path) -> Path:
    if not raw:
        return base
    path = Path(raw.replace("\\", "/"))
    if not path.is_absolute():
        path = base / path
    return path.expanduser().resolve()


def resolve_project_root(cfg: dict[str, Any] | None = None) -> Path:
    cfg = cfg or load_settings()
    raw = str((cfg.get("paths") or {}).get("project_root") or "").strip()
    if raw:
        return _resolve_path(raw, base=ROOT)
    return ROOT


def resolve_schema_path(cfg: dict[str, Any] | None = None) -> Path:
    schema = ROOT / "schema.sql"
    if schema.is_file():
        return schema
    raise FileNotFoundError(f"找不到 schema.sql: {schema}")


def resolve_json_import_dirs(cfg: dict[str, Any] | None = None) -> list[Path]:
    cfg = cfg or load_settings()
    root = resolve_project_root(cfg)
    raw = (cfg.get("paths") or {}).get("json_import_dirs") or []
    if not isinstance(raw, list):
        return []
    return [_resolve_path(str(item), base=root) for item in raw if str(item).strip()]


def get_mysql(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or load_settings()
    mysql = cfg.get("mysql")
    if not mysql:
        raise KeyError(
            f"配置缺少 mysql 段，请检查 {find_config_path()} 中 mysql.host/port/user/password/database"
        )
    return dict(mysql)


def get_dsn(cfg: dict[str, Any] | None = None) -> str:
    mysql = get_mysql(cfg)
    return (
        f"mysql://{mysql.get('user', 'root')}:{mysql.get('password', 'root')}"
        f"@{mysql.get('host', 'localhost')}:{mysql.get('port', 3306)}/{mysql.get('database', 'footprint_sim')}"
    )


def get_similarity_config(cfg: dict[str, Any] | None = None) -> SimilarityConfig:
    sim = (cfg or load_settings()).get("similarity") or {}
    return SimilarityConfig(
        pin_filter=bool(sim.get("pin_filter", True)),
        min_iou=float(sim.get("min_iou", 0.0)),
        top_k=int(sim.get("top_k", 20)),
        exclude_self=bool(sim.get("exclude_self", True)),
        area_tolerance=float(sim.get("area_tolerance", 0.10)),
    )


def get_api_config(cfg: dict[str, Any] | None = None) -> ApiConfig:
    api = (cfg or load_settings()).get("api") or {}
    return ApiConfig(
        host=str(api.get("host", "0.0.0.0")),
        port=int(api.get("port", 8000)),
        uvicorn_workers=int(api.get("uvicorn_workers", 1)),
    )


def get_erkai_api_config(cfg: dict[str, Any] | None = None) -> dict[str, int]:
    """二开 HTTP 并发：接口① BRD 排队；接口②③ 并发上限。"""
    raw = (cfg or load_settings()).get("erkai_api") or {}
    return {
        "brd_queue_max": max(1, int(raw.get("brd_queue_max", 10))),
        "ingest_max_concurrent": max(1, int(raw.get("ingest_max_concurrent", 32))),
    }


def get_concurrency_config(cfg: dict[str, Any] | None = None) -> ConcurrencyConfig:
    raw = (cfg or load_settings()).get("concurrency") or {}
    total_cpu = int(raw.get("total_cpu_cores", 0))
    reserved = int(raw.get("reserved_cpu_cores", 0))
    max_concurrent = max(1, int(raw.get("max_concurrent_similar", 10)))
    per_request = int(raw.get("iou_workers_per_request", 0))
    pool_max = int(raw.get("iou_pool_max_workers", 0))

    available = max(1, (total_cpu or (os.cpu_count() or 4)) - reserved)
    if per_request <= 0:
        # 自动：剩余核平分给每路并发，单路上限 8
        per_request = max(1, min(8, available // max_concurrent))
    if pool_max <= 0:
        pool_max = min(available, max_concurrent * per_request)

    pool_max = max(1, pool_max)
    # Windows ProcessPoolExecutor 上限 61
    if sys.platform == "win32":
        pool_max = min(pool_max, 61)

    return ConcurrencyConfig(
        total_cpu_cores=total_cpu,
        reserved_cpu_cores=reserved,
        max_concurrent_similar=max_concurrent,
        iou_workers_per_request=per_request,
        iou_pool_max_workers=pool_max,
    )


def get_batch_config(cfg: dict[str, Any] | None = None) -> BatchConfig:
    batch = (cfg or load_settings()).get("batch") or {}
    conc = get_concurrency_config(cfg)
    default_workers = conc.iou_workers_per_request
    return BatchConfig(
        compare_workers=int(batch.get("compare_workers", default_workers)),
        restore_workers=int(batch.get("restore_workers", default_workers)),
        benchmark_max_records=int(batch.get("benchmark_max_records", 1000)),
        benchmark_max_pairs=int(batch.get("benchmark_max_pairs", 0)),
    )


def get_agent_paths(cfg: dict[str, Any] | None = None) -> dict[str, Path]:
    """智能体运行时目录（项目根 data/ 下）。"""
    cfg = cfg or load_settings()
    root = resolve_project_root(cfg)
    paths = cfg.get("agent_paths") or {}
    return {
        "root": root,
        "brd": _resolve_path(str(paths.get("brd", "data/brd")), base=root),
        "logs": _resolve_path(str(paths.get("logs", "data/logs")), base=root),
    }


def ensure_runtime_data_dirs(cfg: dict[str, Any] | None = None) -> None:
    """仅创建运行时必要目录（BRD 暂存、日志）；JSON/PDF 只入 MySQL，不落本地。"""
    paths = get_agent_paths(cfg)
    for key in ("brd", "logs"):
        paths[key].mkdir(parents=True, exist_ok=True)


def _parse_history_plan_date(raw: Any, field: str):
    """历史回补 YAML 日期，只认 YYYY-MM-DD。"""
    from datetime import date

    if raw in (None, "", False, "null", "none", "None"):
        return None
    text = str(raw).strip()[:10]
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(
            f"oa_scheduler.history_backfill.{field} 不是合法日期（需要 YYYY-MM-DD），当前值={raw!r}"
        ) from exc


def resolve_history_plan_range(hb: dict[str, Any] | None = None) -> tuple[Any, Any]:
    """
    YAML 历史回补日历区间：start_date 与 end_date 均为**含当天**的最后处理日。
    调 OA 接口①时 end_date 参数须为「终止日 + 1 天」（OA 左闭右开）。
    """
    from datetime import date

    hb = dict(hb or {})
    sd = _parse_history_plan_date(hb.get("start_date"), "start_date")
    ed = _parse_history_plan_date(hb.get("end_date"), "end_date")
    if sd and ed:
        return sd, ed
    year = int(hb.get("year") or date.today().year)
    if sd and not ed:
        return sd, date(sd.year, 12, 31)
    if ed and not sd:
        return date(ed.year, 1, 1), ed
    return date(year, 1, 1), date(year, 12, 31)


def get_oa_scheduler_config(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """OA 定时任务：history_backfill 写在当前环境 yaml 的 oa_scheduler 段。"""
    cfg = cfg or load_settings()
    sched = dict(cfg.get("oa_scheduler") or {})
    hb = dict(sched.get("history_backfill") or {})
    history_on = bool(hb.get("enabled", sched.get("history_enabled", True)))
    # 只跑增量时 history_backfill 里日期可以不配；开了历史才严格解析
    if history_on:
        ps, pe = resolve_history_plan_range(hb)
    else:
        ps, pe = resolve_history_plan_range({})
    hb["start_date"] = ps.isoformat()
    hb["end_date"] = pe.isoformat()

    if "enabled" not in hb:
        hb["enabled"] = bool(sched.get("history_enabled", True))

    max_p = sched.get("max_projects_per_run", hb.get("max_projects_per_run", 0))
    try:
        max_p_int = int(max_p) if max_p else 0
    except (TypeError, ValueError):
        max_p_int = 0

    return {
        "timezone": str(sched.get("timezone") or "Asia/Shanghai"),
        "history_enabled": bool(hb.get("enabled", True)),
        "history_backfill": hb,
        "incremental_enabled": bool(sched.get("incremental_enabled", False)),
        "incremental_lookback_days": int(sched.get("incremental_lookback_days", 2) or 2),
        "incremental_interval_days": int(sched.get("incremental_interval_days", 2) or 2),
        "incremental_cron": str(sched.get("incremental_cron") or "0 2 * * *"),
        "oa_api_interval_sec": float(
            sched.get(
                "oa_api_interval_sec",
                hb.get("oa_api_interval_sec", sched.get("delay_sec_between_projects", 10)),
            )
        ),
        "max_projects_per_run": max_p_int or None,
    }


def get_minio_config(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """对象存储。enabled=false 时三个大字段仍直接写入 MySQL。"""
    cfg = cfg or load_settings()
    raw = cfg.get("minio") or {}
    return {
        "enabled": bool(raw.get("enabled", False)),
        "endpoint": str(raw.get("endpoint") or "").strip(),
        "access_key": str(raw.get("access_key") or ""),
        "secret_key": str(raw.get("secret_key") or ""),
        "bucket": str(raw.get("bucket") or "footprint"),
        "secure": bool(raw.get("secure", False)),
    }


def get_oa_config(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or load_settings()
    oa = cfg.get("oa") or {}
    return {
        "enabled": bool(oa.get("enabled", True)),
        "base": str(oa.get("base", "http://127.0.0.1")).rstrip("/"),
        "project_code_path": str(oa.get("project_code_path", "/api/third/resource/project_code")),
        "project_info_path": str(oa.get("project_info_path", "/api/third/resource/project_info")),
        "brd_path": str(oa.get("brd_path", "/api/third/resource/download")),
        "timeout_sec": float(oa.get("timeout_sec", 60.0)),
        "allow_http_api": bool(oa.get("allow_http_api", False)),
    }


def get_agent_config(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or load_settings()
    agent = cfg.get("agent") or {}
    return {
        "session_ttl_days": int(agent.get("session_ttl_days", 7)),
        "rerank_top_n": int(agent.get("rerank_top_n", 10)),
        "llm_intent_enabled": bool(agent.get("llm_intent_enabled", False)),
    }


def get_llm_config(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """LLM 判断/回复所用大模型（OpenAI 兼容接口），全部从 yaml llm 段读取，便于外部切换模型。"""
    cfg = cfg or load_settings()
    llm = cfg.get("llm") or {}
    extra_headers = llm.get("extra_headers") or {}
    return {
        "enabled": bool(llm.get("enabled", False)),
        "base_url": str(llm.get("base_url", "http://127.0.0.1:11434/v1")),
        "chat_path": str(llm.get("chat_path", "/chat/completions")),
        "model": str(llm.get("model", "qwen2.5")),
        "api_key": str(llm.get("api_key", "")),
        "timeout_sec": float(llm.get("timeout_sec", 60)),
        "temperature": float(llm.get("temperature", 0.1)),
        "max_tokens": int(llm.get("max_tokens", 2048)),
        "json_mode": bool(llm.get("json_mode", True)),
        "system_prompt": llm.get("system_prompt"),
        "rank_prompt": llm.get("rank_prompt"),
        "intent_prompt": llm.get("intent_prompt"),
        "reply_enabled": bool(llm.get("reply_enabled", False)),
        "extra_headers": dict(extra_headers) if isinstance(extra_headers, dict) else {},
    }


def bootstrap(config_path: Path | str | None = None) -> dict[str, Path]:
    """加载 YAML。不传路径时只认 deploy_env.yaml 里的 env。"""
    global _ACTIVE_CONFIG
    path = find_config_path(config_path)
    _ACTIVE_CONFIG = str(path)
    load_settings.cache_clear()
    cfg = load_settings()
    return {
        "project_root": resolve_project_root(cfg),
        "schema_path": resolve_schema_path(cfg),
        "config_path": path,
    }
