PCB 对话检索 + PDF 预览服务（独立部署包）
==========================================

与 ingest_service 共用同一 MySQL，只需能连上库即可检索已入库数据。

代码说明（为何本包仍有 footprint_db / footprint_json）：
  - 形态相似检索读库里的 fpgeom（BLOB）做 IoU，不重复解析历史 source_json。
  - 用户上传「查询用 JSON」时，检索侧仍要用 footprint_json 解析 → fpgeom，再与库内候选比对。
  - footprint_db 是本服务的 MySQL 访问层（连接、查 fpgeom/元数据/PDF、find_similar 等），与 ingest 各有一份拷贝，部署互不依赖 Python 路径。
  - 改 footprint_json 解析规则时：入库服务写库、检索服务查库前解析，两处需同步，否则新 JSON 在检索端会算错 fpgeom。

前端不在此目录维护，请使用仓库 DAY002/desktop 交给前端同事（对接本服务 :8000）。

已集成 API：
  - POST /api/agent/sessions、GET /api/agent/sessions/{id}
  - POST /api/agent/chat
  - POST /api/v1/board/json（与 sessions 等价，供 EDA/明工文档接口4）
  - GET /api/pdf/...（含位号级高亮 /component/.../file|preview）

LLM 调用策略（省 token、防「返回空内容 finish_reason=length」）：
  - 意图先走规则：消息里恰有 1 个已上传位号 / 「再给我 N 个」续问 → 不调 LLM；识别不出才调 LLM（agent.llm_intent_enabled=false 则直接追问）。
  - 排序只调 1 次 LLM：候选按「[序号] 器件= 值= 客户= 完成= 资历=」一行一个紧凑排布，只返回 indices。
  - llm.disable_thinking=true 关思考型模型的推理（Qwen3/R1 等），否则 max_tokens 会被思考耗尽、正文为空。
  - 每次调用有「动作=LLM调用」日志：耗时、输入/输出 token、finish_reason、思考字符数，可据此判断是模型慢还是提示词长。

日志：控制台单行中文，格式「时间 | 会话 | 动作 | 详情 | 结果」（北京时间），便于 grep 动作=对话检索、PDF接口 等。

部署（本文件夹为项目根）：
  1. pip install -r requirements.txt
  2. 复制 config/footprint_sim.example.yaml 为 config/footprint_sim.yaml
  3. python setup_db.py   （建表/迁移只做这一次；run_search_app 启动不会 init_schema）
  4. python run_search_app.py
