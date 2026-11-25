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

  5. （可选）OA 定时拉 BRD
     python run_oa_scheduler.py
     或 python run_oa_scheduler.py --once history  试跑一片
	 
  6.失败项目重试（OA 接口② 项目明细 + 接口③ 下载 BRD，更新 project_profile / process / data/brd）：

	 python run_oa_retry_projects.py -p "PRJ001&PRJ002&PRJ003"
  7.启动补充数据库中位号在pdf的坐标定时任务（十五分钟启动一次）：
  
	python run_pdf_refdes_backfill_scheduler.py

8.待完成板 IoU（一直跑）：

	nohup python pending_iou_worker.py >/data/logs/pending_iou.log 2>&1 &

9.补面积、补栅格（库里缺了跑一次性的跑完停）：

	nohup python backfill_copper_area.py  >/data/logs/area.log  2>&1 &
	nohup python backfill_raster_grid.py  >/data/logs/grid.log 2>&1 &

接口：GET /api/v1/brd/next、POST /api/v1/ingest、POST /api/v1/ingest/failure
BRD 目录：agent_paths.brd（默认 data/brd），与 OA 下载路径一致。

10.补水印（一次性和面积栅格一样）：

	nohup python watermark_existing_pdfs.py >/data/logs/logo.log  2>&1 &
	
11.把现有的位号坐标更新成全字匹配的坐标没有则不动（一次性）：

	nohup python run_pdf_refdes_backfill_scheduler.py --force >/data/logs/new_pdf_refdes.log  2>&1 &
	
12.刷新现有库里存放点集的文件支持后续圆形/异形的IOU直接计算不离散化近似（一次性）：

	nohup python refresh_fpgeom_from_json.py > refresh_fpgeom.out 2>&1 &
	
13.拉取现在minIO中没有pdf的brd文件补充pdf（一次性）：

	nohup python refetch_missing_pdf_brd.py > refetch_brd.out 2>&1 &