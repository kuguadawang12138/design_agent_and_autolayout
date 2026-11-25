PCB 入库 / OA / 明工二开服务（独立部署包）
==========================================
### 虚拟机安装环境说明：
1. Ubuntu24.04（noble）Python3.12 新增 PEP668 保护机制：禁止直接在系统全局 Python 用 pip 安装包，防止 pip 破坏系统自带 Python 依赖，导致 apt 等系统工具损坏。
2. 创建虚拟环境 python -m venv prdEnv
3. 激活虚拟环境 source prdEnv/bin/activate
4. 激活后命令行会前面多 (venv)，此时pip是独立环境，不受PEP668限制，pip install -r requirements.txt
5. 后续每次使用项目，都需要先执行 source prdEnv/bin/activate；退出虚拟环境输入 deactivate
6. OA解压rar压缩包：sudo apt update && sudo apt install unrar

### 部署脚本说明：

与 search_service 共用同一 MySQL（config 里 mysql 段须一致）。

部署（在本目录 ingest_service 下执行）：

  1. pip install -r requirements.txt

  2. 配置 config/（缺任一项则无法启动）
     copy config\deploy_env.example.yaml config\deploy_env.yaml
       → 填写 env: prd  或  env: st
     copy config\footprint_sim_prd.example.yaml config\footprint_sim_prd.yaml
     copy config\footprint_sim_st.example.yaml config\footprint_sim_st.yaml
       → 按本机改 mysql、oa.base 等（prd/st 各一份）

  3. python setup_db.py（建库脚本，入库服务和检索服务跑一个就行）

  4. python run_ingest_app.py
     默认 http://0.0.0.0:8000

  5. （可选）OA 定时拉 BRD（历史 + 增量同一脚本，增量配置见文末）
     python run_oa_scheduler.py
     或 python run_oa_scheduler.py --once incremental   试跑增量
     或 python run_oa_scheduler.py --once history       试跑一片历史
	 
  6.失败项目重试（OA 接口② 项目明细 + 接口③ 下载 BRD，更新 project_profile / process / data/brd）：

	 python run_oa_retry_projects.py -p "PRJ001&PRJ002&PRJ003"
  7.启动补充数据库中位号在pdf的坐标定时任务（十五分钟启动一次）：
  
	python run_pdf_refdes_backfill_scheduler.py

8.待完成板 IoU（一直跑）：

	nohup python pending_iou_worker.py >/data/logs/pending_iou.log 2>&1 &

9.补面积、补栅格（库里缺了再跑，跑完自己停）：

	nohup python backfill_copper_area.py  >/data/logs/area.log  2>&1 &
	nohup python backfill_raster_grid.py  >/data/logs/grid.log 2>&1 &

接口：GET /api/v1/brd/next、POST /api/v1/ingest、POST /api/v1/ingest/failure
BRD 目录：agent_paths.brd（默认 data/brd），与 OA 下载路径一致。

---

<span style="color:#0969da">**2025-11-18-master** PDF 位号全字匹配（U10/U100 区分）；OA 增量改为每 2 天凌晨拉前 2 天。</span>

### OA 仅增量（仍用 `run_oa_scheduler.py`）

历史回补结束后，改 `config/footprint_sim_prd.yaml`（或 st）里 **`oa_scheduler`** 段：

- `history_backfill.enabled` → `false`：不再跑历史回补线程。
- `incremental_enabled` → `true`：打开定时增量。
- `incremental_lookback_days` → `2`：**每次触发当天**为运行日，向 OA 要前 2 个自然日（`start=今天-2`，`end=今天`，`end` 不含当天）；**不读** `history_backfill.start_date/end_date`，**不读** `oa_sync_progress`。
- `incremental_interval_days` → `2`：进程常驻，**每 2 天**凌晨再跑一轮增量（历史跑完也不会停增量）。
- `incremental_cron` → `"0 2 * * *"`：触发时刻取**凌晨 2:00**（分、时同 cron；间隔仍以 `incremental_interval_days` 为准）。

试跑：`python run_oa_scheduler.py --once incremental`

### PDF 位号坐标按全字匹配重算（一次性）

补刷脚本已用新匹配逻辑，已有坐标也会按 `--force` 覆盖：

`nohup python run_pdf_refdes_backfill_scheduler.py --once --force >/data/logs/new_pdf_refdes.log 2>&1 &`
