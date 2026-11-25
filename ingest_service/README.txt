PCB 入库 / OA / 二开服务（独立部署包）
==========================================

与 search_service 共用同一 MySQL。不含前端。
footprint_db：入库读写（fpgeom、project_pdf、process 等）；footprint_json：明工 JSON→FootprintModel→fpgeom（写入 footprint_library.fpgeom）。

已集成：
  - GET /api/v1/brd/next、POST /api/v1/ingest、POST /api/v1/ingest/failure（二开接口，日志为中文）
  - OA 定时拉 BRD：python run_oa_scheduler.py（见 docs/OA定时任务说明.md）
  - OA 失败项目重试（接口②+③，写库）：见下方
  - OA HTTP 手动触发默认关闭（oa.allow_http_api: false）

失败项目重试（OA 接口② + 接口③）：
  python run_oa_retry_projects.py --projects 1165551
  python run_oa_retry_projects.py --projects "PRJ001&PRJ002&PRJ003"
  多个项目编码用英文 & 分隔。

二开入库/失败联调（单脚本）：
  python test_erkai_ingest.py --base http://127.0.0.1:8000 --project TEST_ERKAI_FAIL

本地 data 目录：仅 data/brd（待二开拉取）、data/logs；JSON/PDF 只进 MySQL，不创建 data/json。

YAML 历史回补：start_date 与 end_date 为起止日（均含）；调 OA 接口①时 end_date 在代码中为「终止日+1」。
历史计划跑完后 run_oa_scheduler 自动退出；Ctrl+C 会停止调度器。

部署（本文件夹为项目根，服务器是 Ubuntu）：
  1. pip install -r requirements.txt
  2. config/deploy_env.yaml + footprint_sim_prd.yaml 或 st.yaml
  3. python setup_db.py   （建表/迁移只做这一次，服务内不再自动改表结构）
  4. python run_ingest_app.py

OA 压缩包 / RAR（Ubuntu）：
  py7zr 只能解 7z，不能解 RAR。RAR 用 Python 包 rarfile，它自己不解包，要调用系统命令。
  sudo apt-get update
  sudo apt-get install -y unrar
  若软件源没有 unrar（没开非自由源），改用：
  sudo apt-get install -y unar
  requirements.txt 里已有 rarfile。解压时先按文件后缀选一种，失败后再把 zip、7z、rar、tar 都试一遍。
  RAR 内部再依次尝试 rarfile、unrar、unar、bsdtar、7z/7zz，不会用 py7zr 去解 RAR。

默认 api.port 示例 8000（入库机与检索机分机部署时各自 :8000）。

二开并发（yaml erkai_api）：
  接口① GET /api/v1/brd/next — 串行下发，最多 10 个请求排队，满则 HTTP 503
  接口②③ POST ingest / failure — 默认最多 32 路并发（可调）；建议 api.uvicorn_workers>=4

PDF：无论仿射校准是否成功，均写入 project_pdf。
  校准点不足/未完成仿射/未找到位号等不写 ingest_failure（靠位号补刷定时任务）。
  仅单器件无法双点校准等写入 ingest_failure，错误码 WARN_PDF（非 ERR_PDF）。
  检索服务高亮仅依赖库内 pdf_refdes_x/y + pdf_scale，不再在检索时搜 PDF 兜底。
  位号坐标补刷（单线程，默认 15 分钟）：
    python run_pdf_refdes_backfill_scheduler.py
    python run_pdf_refdes_backfill_scheduler.py --once
  日志表 pdf_refdes_backfill_log（批次汇总 + 失败明细）。
