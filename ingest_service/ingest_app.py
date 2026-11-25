# -*- coding: utf-8 -*-
"""入库 / OA / 二开 API（与检索服务分离，共用 MySQL）。"""

from __future__ import annotations

import io
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from agent_ingest import ingest_json_pdf_batch, report_ingest_failure, serve_next_brd, stage_brd
from agent_log import log_line
from agent_oa import fetch_project_codes, run_oa_pipeline
from erkai_api_queue import BrdNextQueueFull, brd_queue, ingest_gate, init_erkai_queues
from settings import bootstrap, ensure_runtime_data_dirs, get_erkai_api_config


@asynccontextmanager
async def lifespan(app: FastAPI):
    bootstrap()
    eq = get_erkai_api_config()
    init_erkai_queues(
        brd_max_waiting=eq["brd_queue_max"],
        ingest_max_concurrent=eq["ingest_max_concurrent"],
    )
    ensure_runtime_data_dirs()
    yield


app = FastAPI(title="PCB 入库与 OA", version="2.2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _ok(data: dict) -> dict:
    return data


def _fail(message: str, **extra) -> dict:
    return {"ok": False, "message": message, **extra}


@app.get("/api/health")
def health():
    return {"ok": True, "service": "ingest", "message": "入库服务正常"}


class DbQueryContext(BaseModel):
    table: str = ""
    project_code: str = ""
    refdes: str = ""


class DbQueryPage(BaseModel):
    curpage: int = 1
    pagesize: int = 50


class DbQueryBody(BaseModel):
    context: DbQueryContext
    page: DbQueryPage = Field(default_factory=DbQueryPage)


@app.post("/api/db/query")
def db_query(body: DbQueryBody):
    """按表、项目编码、位号分页。同一条件同一页每次结果相同。"""
    from db_query import list_rows

    table = body.context.table
    try:
        data = list_rows(
            table,
            project_code=body.context.project_code,
            refdes=body.context.refdes,
            page=body.page.curpage,
            page_size=body.page.pagesize,
        )
    except ValueError as exc:
        log_line("库表查询", result=f"【拒绝】表={table} {exc}", level="warning")
        return JSONResponse(_fail(str(exc)), status_code=400)
    except Exception as exc:
        log_line("库表查询", result=f"【失败】表={table} {type(exc).__name__}: {exc}", level="error")
        return JSONResponse(_fail(str(exc)), status_code=500)
    log_line(
        "库表查询",
        result=(
            f"【返回】表={data['table']} 项目={data['project_code'] or '-'} "
            f"位号={data['refdes'] or '-'} 第{data['page']}页 "
            f"每页={data['page_size']} 本页={data['row_count']} 总行={data['total']}"
        ),
    )
    return data


# ===================== 二开：/api/v1/... =====================


@app.get("/api/v1/brd/next")
async def v1_get_brd():
    """二开接口1：拉取 BRD（串行队列，最多 10 个请求排队）。"""
    try:
        item = await brd_queue().run(serve_next_brd)
    except BrdNextQueueFull as exc:
        log_line("二开接口", result=f"【拉取 BRD】{exc}", level="warning")
        return JSONResponse(_fail(str(exc)), status_code=503)
    if item is None:
        log_line("二开接口", result="【拉取 BRD】暂无待处理文件")
        return JSONResponse(_fail("暂无待处理的 BRD 文件"))
    log_line(
        "二开接口",
        result=f"【拉取 BRD】已下发 project_code={item['project_code']} 文件={item['file_name']} 大小={item['size']} 字节",
    )
    return StreamingResponse(
        io.BytesIO(item["content"]),
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{item["file_name"]}"',
            "X-Project-Code": item["project_code"],
        },
    )


@app.post("/api/v1/ingest")
async def v1_ingest(
    project_code: str | None = Form(None),
    json_files: list[UploadFile] = File(default=[]),
    pdf_files: list[UploadFile] = File(default=[]),
):
    """二开接口2：回传 JSON + PDF 入库。"""
    code = str(project_code or "").strip()
    json_blobs = [(f.filename or "part.json", await f.read()) for f in json_files]
    pdf_blobs = [(f.filename or "board.pdf", await f.read()) for f in pdf_files]
    if not json_blobs:
        log_line("二开接口", result="【入库】拒绝：未上传 JSON 文件", level="warning")
        return JSONResponse(_fail("请上传至少一个 JSON 文件"), status_code=400)
    log_line(
        "二开接口",
        result=(
            f"【入库】开始 project_code={code or '（由 JSON 内解析）'} "
            f"JSON 文件数={len(json_blobs)} PDF 文件数={len(pdf_blobs)}"
        ),
    )
    result = await ingest_gate().run(
        lambda: ingest_json_pdf_batch(
            project_code=project_code, json_blobs=json_blobs, pdf_blobs=pdf_blobs
        )
    )
    status = 200 if result.get("ok") else 400
    level = "info" if result.get("ok") else "warning"
    log_line(
        "二开接口",
        result=(
            f"【入库】结束 HTTP={status} project_code={result.get('project_code') or '-'} "
            f"成功={result.get('ok_count', 0)} 失败={result.get('fail_count', 0)} "
            f"说明={result.get('message') or '无'}"
        ),
        level=level,
    )
    return JSONResponse(result, status_code=status)


@app.post("/api/v1/ingest/failure")
async def v1_ingest_failure(
    message: str = Form(...),
    project_code: str = Form(...),
    refdes: str | None = Form(None),
):
    """二开接口3：解析/转换失败上报。"""
    if not message.strip():
        log_line("二开接口", result="【失败上报】拒绝：message 为空", level="warning")
        return JSONResponse(_fail("message 不能为空"), status_code=400)
    if not project_code.strip():
        log_line("二开接口", result="【失败上报】拒绝：project_code 为空", level="warning")
        return JSONResponse(_fail("project_code 不能为空"), status_code=400)
    try:
        await ingest_gate().run(
            lambda: report_ingest_failure(
                project_code=project_code.strip(),
                message=message.strip(),
                refdes=refdes,
            )
        )
    except ValueError as exc:
        log_line("二开接口", result=f"【失败上报】参数错误：{exc}", level="warning")
        return JSONResponse(_fail(str(exc)), status_code=400)
    log_line(
        "二开接口",
        result=(
            f"【失败上报】已写入 ingest_failure project_code={project_code.strip()} "
            f"refdes={refdes or ''} 原因={message.strip()[:120]}"
        ),
    )
    return _ok({"ok": True, "message": "失败记录已入库"})


@app.post("/api/v1/pending/brd")
async def v1_pending_brd(
    project_code: str = Form(...),
    file: UploadFile = File(...),
):
    """待完成设计的板：OA 上传 BRD 压缩包，解出 .brd 放到原 BRD 目录，并记下项目编码。"""
    code = str(project_code or "").strip()
    if not code:
        log_line("待完成入库", result="【拒绝】project_code 为空", level="warning")
        return JSONResponse(_fail("project_code 不能为空"), status_code=400)
    content = await file.read()
    if not content:
        log_line("待完成入库", result=f"【拒绝】project_code={code} 文件为空", level="warning")
        return JSONResponse(_fail("文件为空"), status_code=400)
    try:
        manifest = stage_brd(
            code,
            content,
            file.filename,
            content_type=file.content_type,
        )
    except ValueError as exc:
        log_line("待完成入库", result=f"【解压失败】project_code={code} {exc}", level="warning")
        return JSONResponse(_fail(str(exc)), status_code=400)
    except Exception as exc:
        log_line("待完成入库", result=f"【解压失败】project_code={code} {type(exc).__name__}: {exc}", level="error")
        return JSONResponse(_fail(str(exc)), status_code=500)

    from footprint_db import connect, mark_pending_project

    conn = connect()
    try:
        mark_pending_project(conn, code)
    finally:
        conn.close()
    log_line(
        "待完成入库",
        result=(
            f"【待完成项目】project_code={code} 状态=0 文件={manifest.get('file_name')} "
            f"已放入 BRD 目录，等待解析"
        ),
    )
    return _ok(
        {
            "ok": True,
            "project_code": code,
            "file_name": manifest.get("file_name"),
            "message": "BRD 已放入待解析目录，项目已记为待完成",
        }
    )


# ===================== OA（默认关闭 HTTP，请用 run_oa_scheduler.py） =====================


def _oa_http_allowed() -> bool:
    from settings import get_oa_config

    return bool(get_oa_config().get("allow_http_api"))


@app.get("/api/oa/project-codes")
def oa_project_codes():
    if not _oa_http_allowed():
        return JSONResponse(_fail("OA HTTP 已关闭，请使用 run_oa_scheduler.py"), status_code=403)
    try:
        return _ok({"items": fetch_project_codes()})
    except Exception as exc:
        return JSONResponse(_fail(str(exc)), status_code=502)


@app.post("/api/oa/pipeline")
def oa_pipeline(
    project_code: str = Form(...),
    doc_id: str | None = Form(None),
    file_id: str | None = Form(None),
):
    if not _oa_http_allowed():
        return JSONResponse(_fail("OA HTTP 已关闭，请使用 run_oa_scheduler.py"), status_code=403)
    try:
        return _ok(run_oa_pipeline(project_code, doc_id=doc_id, file_id=file_id))
    except Exception as exc:
        return JSONResponse(_fail(str(exc)), status_code=502)
