# PKG-IMG-SIM HTTP API 接口文档

Base URL：`http://<host>:8000`  
OpenAPI：`http://<host>:8000/docs`  
数据库：**MySQL 8+**，唯一键 `project_code + refdes`  
API 版本：**0.9.2**

---

## 1. 接口分工

| 接口 | 用途 |
|------|------|
| POST `/api/v1/footprints/library` | **唯一入库入口**（UPSERT） |
| POST `/api/v1/footprints/similar` | **仅检索**（不入库） |
| GET `/api/v1/footprints/settings` | 读取 YAML 策略 |
| GET `/health` | 健康检查 |

配置：`config/footprint_sim.yaml` → **`mysql`** 段（非 PostgreSQL）

---

## 2. 检索策略（/similar）

`min_iou`、`top_k` 可在请求里传，不传则用 YAML `similarity` 默认值。

| 步骤 | 规则 |
|------|------|
| 1 | pin 硬过滤 |
| 2 | 全量几何 IoU 精排 |
| 3 | iou ≥ min_iou |
| 4 | Top-K；exclude_self 排除同 project_code+refdes |

---

## 3. POST /api/v1/footprints/library

上传器件 JSON，按 **project_code + refdes** UPSERT。

| 字段 | 类型 | 必填 |
|------|------|------|
| file | file | 是 |

**响应**

```json
{
  "action": "inserted",
  "project_code": "1190967",
  "refdes": "U1",
  "name": "QFN88-0_4-1",
  "pin_count": 89,
  "geom_islands": 89,
  "warnings": []
}
```

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/footprints/library" -F "file=@U1.json"
```

---

## 4. POST /api/v1/footprints/similar

仅检索，**不接受 save_to_library**。

| 字段 | 类型 | 必填 |
|------|------|------|
| file | file | 是 |
| min_iou | float | 否 |
| top_k | int | 否 |

**响应**

```json
{
  "results": [
    {
      "project_code": "1190967",
      "refdes": "U2",
      "minx": -210.5,
      "miny": -180.0,
      "maxx": 210.5,
      "maxy": 180.0
    }
  ]
}
```

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/footprints/similar" \
  -F "file=@封装形状还原-示例.json" \
  -F "min_iou=0.9" \
  -F "top_k=5"
```

---

## 5. CLI

| 命令 | 作用 |
|------|------|
| python run_api.py | 启动 API |
| python setup_db.py | MySQL 建库建表 |
| python query_similar.py \<json\> | CLI 检索 |
