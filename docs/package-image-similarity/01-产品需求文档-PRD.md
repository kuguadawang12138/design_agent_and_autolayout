# 封装图像相似度检索 — 产品需求文档（PRD）

| 项 | 内容 |
|---|---|
| 模块代号 | PKG-IMG-SIM |
| 版本 | v0.9.2 |
| 日期 | 2026-08-31 |
| 所属项目 | PCB-Copilot（PCB 设计智能辅助平台） |
| 文档范围 | **仅本技术点**；与 GNN/RL 布局、知识库 RAG 等模块独立演进 |

---

## 1. 背景与动机

### 1.1 业务场景

工程师在 Allegro 中做新板设计时，需要查找历史上形态相近的封装。本模块用**几何 IoU**（非 SSIM 主路径）回答「哪个历史封装形状最接近」。

### 1.2 与主项目的关系

对应主 PRD **S4 — 相似设计复用** 的封装粒度子能力；与 `board-packages` xlsx 结构化检索互补。

### 1.3 方案结论

| 手段 | 角色 |
|---|---|
| SQL / pin 数 | **硬过滤** |
| **fpgeom + 几何 IoU** | **精排（主路径）** |
| BMP / SSIM | 无 JSON 时兜底，非 MVP 重点 |

### 1.4 方案演进

| 阶段 | 结论 |
|---|---|
| 图像 SSIM/ORB | Allegro 同源导出图 SSIM 虚高，**不作主指标** |
| **当前主路径** | 设计部 JSON → 形状还原 → **fpgeom 缓存** → **几何 IoU**（Shapely） |
| 历史库存储 | **MySQL 8+**：元数据 + fpgeom LONGBLOB；唯一键 project_code+refdes；**IoU 不入库** |
| 数据来源 | 设计部 JSON（顶层 project_code/refdes/units + pins[]），字段见设计部字段清单 |

---

## 2. 用户与场景

**S1 — 单封装查询，返回相似度排序（当前）**

Agent 收到设计部 JSON → 入库（或读库）→ `pin_count` 硬过滤 → 几何 IoU Top-K。

**S2 — 综合排序（Phase 1）**

在 IoU 基础上叠加 name_match、usage 等权重。

---

## 3. 功能需求

### 3.1 已实现（Phase 0.5）

| 编号 | 需求 | 验收 |
|---|---|---|
| SIM-01 | 两封装几何 IoU | Shapely 精确多边形，0/90/180/270° 取最优 |
| SIM-02 | 批量两两对比 / 1-vs-N | `restore_footprint.py` 或 `query_similar.py` |
| SIM-03 | JSON 形状还原 + fpgeom 缓存 | `.fpgeom` msgpack 二进制 |
| SIM-04 | 设计部 JSON 接口 | 字段清单 + 非标准 JSON 兼容（如 `"units":mils`） |
| SIM-12 | **MySQL 历史库** | `footprint_library` 表 + `/library` API |
| SIM-13 | Agent 入库/检索 API 雏形 | `footprint_db.py` |

### 3.2 Phase 1

| 编号 | 需求 |
|---|---|
| SIM-10 | 万级库性能优化（粗召回可选） |
| SIM-11 | 综合排序 API |
| SIM-14 | Agent 工具 `find_similar_packages(json)` 正式注册 |

---

## 4. 非功能需求

| 类别 | 要求 |
|---|---|
| 数据库 | **MySQL 8+** |
| 比对性能 | 单次 pin 过滤后候选 <500：秒级 IoU 精排可接受 |
| 可解释 | /similar 返回 project_code、refdes、封装 bbox；IoU 仅日志/CLI |
| 合规 | 内网；仅存封装符号级几何 |

---

## 5. 排序与硬过滤

### 5.1 硬过滤（已定）

| 规则 | 默认 |
|---|---|
| **引脚数** | **必须相等**（SQL `WHERE pin_count = ?` + 应用层校验） |
| 封装族 | 待定，默认不启用 |

### 5.2 精排

硬过滤后按 **几何 IoU 降序**；相似度分数**不写入数据库**。

---

## 6. 数据与存储（v0.4）

### 6.1 入库内容

| 存储 | 是否入库 | 用途 |
|---|---|---|
| **fpgeom**（归一化多边形） | ✅ BYTEA | 比对热路径 |
| pin_count、name、bbox、units | ✅ 列 + 索引 | 硬过滤、展示 |
| source_json | ✅ JSONB | 审计、重导、算法升级 |
| content_hash | ✅ 唯一索引 | 去重 |
| **IoU / 相似度** | ❌ | 查询时计算 |

### 6.2 不入库的内容

- 不把 JSON 拆成 pins/shapes 多表（检索场景无必要）
- 不预计算两两 IoU

### 6.3 流程

```
Agent 收到 JSON
  → build_fpgeom_from_footprint()
  → UPSERT footprint_library（按 content_hash 去重）
查询时
  → SELECT ... WHERE pin_count = ?
  → best_geom_iou(query_fpgeom, candidate_fpgeom)
  → Top-K
```

### 6.4 脚本（另一台机器可一键跑）

```powershell
powershell -ExecutionPolicy Bypass -File scripts/footprint_db/run_all.ps1
```

详见 `scripts/footprint_db/README.md`。

---

## 7. 还原与验证工具

```bash
# 本地目录：还原 + 写 fpgeom + 目录内两两 IoU（不入库）
python src/footprint_similarity/restore_footprint.py --input data/1190967/json --workers 8

# 数据库：相似检索
python scripts/footprint_db/query_similar.py data/1190967/json/1190967_LT86121_QFN88.json
```

---

## 8. 接口草案（Agent）

```json
{
  "query_footprint_json": "path/to/query.json",
  "top_k": 10
}
```

响应：`{ "results": [{ "project_code", "refdes", "minx", "miny", "maxx", "maxy" }, ...] }`

---

## 9. 验收标准（v0.4）

1. PostgreSQL 建表 + 导入 1190967（79）+ board-packages（7）共 86 条量级。
2. `query_similar.py` 对 QFN88 查询返回 pin=88 候选 IoU 排序。
3. fpgeom 与文件缓存格式一致，模块可 `load_fpgeom` / `fpgeom_from_bytes`。
4. 三文档与脚本同步 v0.4。

---

## 10. 里程碑

| 阶段 | 交付 |
|---|---|
| Phase 0 | JSON 还原 + fpgeom + restore CLI ✅ |
| **Phase 0.5** | **PostgreSQL 入库 + query_similar** ✅ |
| Phase 1 | HTTP API + Agent 工具 + 综合排序 |
| Phase 2 | 百万库粗召回 + 全量异形 SHAPE |
