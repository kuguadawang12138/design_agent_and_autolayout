# -*- coding: utf-8 -*-
"""对话检索 + PDF 预览 API（与入库服务分离，共用 MySQL）。"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from agent_chat_flow import default_top_n, fuzzy_events, locate_refdes_events, search_events
from agent_log import log_line, use_log_session
from agent_pdf_api import router as pdf_router
from agent_session import cleanup_expired_sessions, create_chat_session, load_session, pick_component_json
from footprint_db import connect
from settings import bootstrap, get_agent_config, get_agent_paths, get_llm_config


@asynccontextmanager
async def lifespan(app: FastAPI):
    bootstrap()
    for key, path in get_agent_paths().items():
        if key != "root":
            path.mkdir(parents=True, exist_ok=True)
    # 不在此建表/迁移；表结构请运维执行 python setup_db.py 或手工 DDL
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
    finally:
        conn.close()
    cfg = get_agent_config()
    stop = threading.Event()

    def _worker():
        cleanup_expired_sessions(cfg["session_ttl_days"])
        while not stop.wait(3600):
            cleanup_expired_sessions(cfg["session_ttl_days"])

    threading.Thread(target=_worker, daemon=True, name="session-cleanup").start()
    llm = get_llm_config()
    log_line(
        "服务启动",
        result=(
            f"【检索服务】会话TTL={cfg['session_ttl_days']}天 "
            f"LLM={'开' if llm.get('enabled') else '关'} model={llm.get('model') or '-'}"
        ),
    )
    yield
    stop.set()
    log_line("服务停止", result="【检索服务】进程退出")


app = FastAPI(title="PCB 对话检索", version="2.2.0", lifespan=lifespan)


class _HttpTimingMiddleware(BaseHTTPMiddleware):
    """每个 HTTP 接口一行耗时（与对话流水线日志配合排查）。"""

    async def dispatch(self, request: Request, call_next):
        t0 = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            ms = int((time.perf_counter() - t0) * 1000)
            path = request.url.path
            if path.startswith("/api"):
                log_line(
                    "HTTP接口",
                    detail=f"{request.method} {path}",
                    result=f"状态码={status} 耗时={ms}ms",
                )


app.add_middleware(_HttpTimingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(pdf_router)


def _ok(data: dict) -> dict:
    return data


def _fail(message: str, **extra) -> dict:
    return {"ok": False, "message": message, **extra}


@app.get("/api/health")
def health():
    llm = get_llm_config()
    return {
        "ok": True,
        "service": "search",
        "llm": {"enabled": llm["enabled"], "model": llm["model"], "base_url": llm["base_url"]},
    }


@app.post("/api/agent/sessions")
async def upload_session(
    session_id: str = Form(...),
    top_n: int | None = Form(None),
    refdes: str = Form(""),
    files: list[UploadFile] | None = File(None),
    json_files: list[UploadFile] | None = File(None),
):
    """用 chat 返回的 session_id，加上器件 JSON 和 top_n，流式返回推荐结果。"""
    uploads = [item for item in list(files or []) + list(json_files or []) if item is not None]
    if not uploads:
        return JSONResponse(_fail("请上传至少一个 JSON 文件"), status_code=400)
    try:
        session = load_session(session_id)
    except FileNotFoundError as exc:
        return JSONResponse(_fail(str(exc)), status_code=404)
    except ValueError as exc:
        return JSONResponse(_fail(str(exc)), status_code=400)
    target = str(refdes or session.get("target_refdes") or "").replace("@", "").strip().upper()
    if not target:
        return JSONResponse(_fail("请先通过对话定位位号"), status_code=400)
    blobs = [(item.filename or "part.json", await item.read()) for item in uploads]
    try:
        raw, filename = pick_component_json(blobs, target)
    except ValueError as exc:
        return JSONResponse(_fail(str(exc)), status_code=400)
    limit = default_top_n(top_n)
    use_log_session(session_id)
    return _sse(search_events(session_id, target, raw, filename, limit))


@app.get("/api/agent/sessions/{session_id}")
def get_session(session_id: str):
    try:
        session = load_session(session_id)
    except FileNotFoundError:
        return JSONResponse(_fail("会话不存在"), status_code=404)
    return _ok(
        {
            "session_id": session_id,
            "component_count": session.get("component_count"),
            "refdes_list": session.get("refdes_list") or [],
            "items": session.get("items") or [],
            "chat_history": session.get("chat_history") or [],
            "last_query": session.get("last_query"),
        }
    )


def _sse(events):
    def _gen():
        for event in events:
            yield "data: " + json.dumps(event, ensure_ascii=False, default=str) + "\n\n"

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/agent/fuzzy")
async def agent_fuzzy(message: str = Form(""), session_id: str = Form(...)):
    """属性查询。chat 返回 need_clarify 后，用同一个 session_id 把这句话传进来。"""
    if not message.strip():
        return JSONResponse(_fail("请输入检索内容"), status_code=400)
    try:
        load_session(session_id)
    except FileNotFoundError:
        return JSONResponse(_fail("会话不存在"), status_code=404)
    except ValueError as exc:
        return JSONResponse(_fail(str(exc)), status_code=400)
    use_log_session(session_id)
    return _sse(fuzzy_events(message.strip(), session_id))


@app.post("/api/agent/chat")
async def agent_chat(message: str = Form(""), session_id: str = Form("")):
    """只回复一句话：有位号是 refdes_ready，否则是 need_clarify。"""
    if not message.strip():
        return JSONResponse(_fail("请输入检索内容"), status_code=400)
    sid = str(session_id or "").strip()
    if sid:
        try:
            load_session(sid)
        except FileNotFoundError:
            return JSONResponse(_fail("会话不存在"), status_code=404)
        except ValueError as exc:
            return JSONResponse(_fail(str(exc)), status_code=400)
    else:
        sid = create_chat_session(None)
    use_log_session(sid)
    return _sse(locate_refdes_events(message.strip(), sid))
