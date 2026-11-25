# OA 定时拉 BRD 说明

## 配置环境

入库服务从 **`ingest_service/settings.py` → `find_config_path()`** 加载 YAML：

1. **`config/deploy_env.yaml`**（必填，仅一行 `env: prd` 或 `env: st`，缺则无法启动）  
2. **`config/footprint_sim_prd.yaml`** 或 **`footprint_sim_st.yaml`**（由 deploy_env 决定）

模板：`deploy_env.example.yaml` → `deploy_env.yaml`；`footprint_sim_* .example.yaml` → 对应 yaml。  
**检索服务**在 `search_service/config/` 同样两份；两台机器 **deploy_env.env 与 mysql 须一致**。

历史回补起止写在当前环境 yaml 的 **`oa_scheduler.history_backfill.start_date / end_date`**（或仅写 **`year`** 表示整年）：**起止日均为含当天**；调用 OA 接口①时 **`end_date` 参数 = YAML 终止日 + 1 天**（OA 左闭右开）。断点在表 **`oa_sync_progress.done_through`**（表示已处理到的最后日历日，含）。

**无命令行参数**运行 `run_oa_scheduler.py` 时，接口①日期**只**来自上述 YAML 计划区间内的分片。库内 `done_through` 若不在 YAML 计划区间内（例如 YAML 是 2024、库是 2025），**以 YAML 为准**从计划起点重跑，不会报错也不会按 25 年续跑。每个分片日期跑完后**无论项目成败**都会把 `done_through` 推进到该片最后一天（含）。启动日志会打印 `【调度配置】环境=… 配置文件=…`。

## 调度

- 历史回补**连续**按 **`chunk_days`**（默认 1 天）一片接一片跑，直到 YAML 计划跑完；**不再**按固定 2 小时间隔。
- **`oa_api_interval_sec`**（默认 10 秒）：仅用于**历史回补**——当天该片日期全部项目拉完后，再等待该秒数后拉下一自然日（同一天内项目连续拉，无额外间隔）。
- 每落盘一份 BRD 会打日志 **`【BRD 落盘】… BRD目录磁盘占用=xx%`**。
- **`data/brd` 所在磁盘**占用超过 **`disk_usage_max_ratio`**（默认 0.70）时，等待 **`disk_wait_seconds`**（默认 120 秒）后重试。
- 可选 **`incremental_enabled`** 按 cron 拉「刚结束的一周」。

## 启动

```bash
cd ingest_service
python run_oa_scheduler.py --once history
nohup python run_oa_scheduler.py >> data/logs/oa_scheduler.nohup.log 2>&1 &
```

计划全部跑完后进程**自动退出**（无需一直挂着）；`Ctrl+C` 会停止调度器。

### 失败项目重试（接口② + ③，写库）

对指定 **project_code** 重跑 OA，更新 `project_profile`、`process`、BRD 落盘；多个编码用 **`&`** 分隔。

```bash
cd ingest_service
python run_oa_retry_projects.py --projects 1165551
python run_oa_retry_projects.py -p "PRJ001&PRJ002"
```

### 命令行联测（不改 YAML 时间窗、不写库）

指定接口①日期区间即可；**不写 `project_profile` / `ingest_failure`**，有有效 `download_path` 时仍会把 BRD 落到 `data/brd/` 验证接口③。

```bash
python run_oa_scheduler.py --start-date 2020-08-15 --end-date 2020-08-16
```

接口② `download_path` 为 `""` 时记 **下载参数为空**（`oa_job_item_skip`），不会误调接口③ 报 404。

## 日志

- 控制台/`data/logs`：`action=OA流程` 中文单行（接口①每批一行；每个项目编码一行含②③）
- `data/logs/oa_scheduler_jobs.jsonl`：每片时间窗汇总
- `SELECT * FROM process WHERE project_code=?`：OA 四字段/下载参数/解压 BRD/明工入库 成败
- `SELECT job_key, done_through FROM oa_sync_progress;`：续跑进度

## 与明工二开

BRD：`agent_paths.brd` ↔ **GET /api/v1/brd/next**。  
项目属性：OA → **`project_profile`**；入库时按 **project_code** 合并。
