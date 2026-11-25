-- PCB 智能体 + 形态比对 统一库（MySQL 5.7+；JSON 字段用 TEXT 存 UTF-8 字符串）
-- 五张表：project_profile、footprint_library、project_pdf、ingest_failure、oa_sync_progress

CREATE TABLE IF NOT EXISTS project_profile (
    id                  BIGINT AUTO_INCREMENT PRIMARY KEY COMMENT '自增主键',
    project_code        VARCHAR(128) NOT NULL COMMENT '项目编码',
    customer_code       VARCHAR(64) NULL COMMENT '客户编码',
    designer_name       VARCHAR(128) NULL COMMENT '设计工程师',
    designer_seniority  DOUBLE NULL COMMENT '工程师资历分',
    finish_time         DATETIME NULL COMMENT '项目完成时间',
    created_at          DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at          DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    UNIQUE KEY uk_project_profile (project_code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='项目级属性（OA 等）';

CREATE TABLE IF NOT EXISTS footprint_library (
    id                  BIGINT AUTO_INCREMENT PRIMARY KEY COMMENT '自增主键',
    name                VARCHAR(512) NULL COMMENT '板名，可为空（JSON project_name）',
    pin_count           INT NOT NULL COMMENT 'Pin 数量，形态检索初筛',
    copper_area         DOUBLE NULL COMMENT '铜箔真实面积（mils²）。按还原形状并集计算，重叠只计一次，已换算单位',
    units               VARCHAR(32) NOT NULL DEFAULT 'mils' COMMENT '长度单位',
    bbox_min_x          DOUBLE NULL COMMENT '封装 bbox 最小 X（mils）',
    bbox_min_y          DOUBLE NULL COMMENT '封装 bbox 最小 Y（mils）',
    bbox_max_x          DOUBLE NULL COMMENT '封装 bbox 最大 X（mils）',
    bbox_max_y          DOUBLE NULL COMMENT '封装 bbox 最大 Y（mils）',
    fpgeom_version      SMALLINT NOT NULL DEFAULT 1 COMMENT 'fpgeom 格式版本',
    fpgeom              VARCHAR(512) NOT NULL COMMENT 'MinIO 路径：年/月/日/项目_位号.fpgeom',
    project_code        VARCHAR(128) NOT NULL COMMENT '项目编码；重入库时整项目先删后插',
    refdes              VARCHAR(64) NOT NULL COMMENT '位号',
    layers              INT NULL COMMENT '板子层数',
    devicetype          VARCHAR(256) NULL COMMENT '器件类型/型号',
    packagesymbol       VARCHAR(256) NULL COMMENT '封装符号',
    value               VARCHAR(256) NULL COMMENT '器件值',
    pcb_x               DOUBLE NULL COMMENT '位号在板坐标 X',
    pcb_y               DOUBLE NULL COMMENT '位号在板坐标 Y',
    top_side            TINYINT(1) NOT NULL DEFAULT 1 COMMENT '1=正面 0=反面',
    source_json         VARCHAR(512) NULL COMMENT 'MinIO 路径：年/月/日/项目_位号.json',
    warnings            MEDIUMTEXT NULL COMMENT '解析告警（JSON 数组字符串）',
    geom_islands        INT NOT NULL DEFAULT 0 COMMENT '铜箔岛数量',
    pdf_refdes_x        DOUBLE NULL COMMENT 'PyMuPDF：位号 Eda_ 丝印在 PDF 第1页中心 X（点）',
    pdf_refdes_y        DOUBLE NULL COMMENT 'PyMuPDF：位号 Eda_ 丝印在 PDF 第1页中心 Y（点）',
    created_at          DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at          DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    UNIQUE KEY uk_project_refdes (project_code, refdes),
    KEY idx_pin_count (pin_count),
    KEY idx_project_code (project_code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='器件形态+元数据';

CREATE TABLE IF NOT EXISTS project_pdf (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY COMMENT '自增主键',
    project_code    VARCHAR(128) NOT NULL COMMENT '项目编码',
    top_side        TINYINT(1) NOT NULL DEFAULT 1 COMMENT '1=正面 0=反面',
    file_name       VARCHAR(256) NOT NULL COMMENT 'PDF 文件名',
    file_content    VARCHAR(512) NOT NULL COMMENT 'MinIO 路径：年/月/日/项目_top或bottom.pdf',
    pdf_scale       DOUBLE NOT NULL DEFAULT 0 COMMENT 'PCB↔PDF 比例（约 PDF点/板mil），与高亮框宽高=bbox×scale 配合 pdf_refdes_x/y',
    created_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    UNIQUE KEY uk_pdf_project_side (project_code, top_side),
    KEY idx_pdf_project (project_code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='项目 PDF（每面一份）';

CREATE TABLE IF NOT EXISTS ingest_failure (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY COMMENT '自增主键',
    project_code    VARCHAR(128) NOT NULL COMMENT '项目编码',
    refdes          VARCHAR(64) NOT NULL DEFAULT '' COMMENT '位号；JSON失败填位号；BRD/二开失败留空',
    error_code      VARCHAR(32) NOT NULL COMMENT '错误码，见 footprint_db.ERROR_CODES',
    detail          TEXT NOT NULL COMMENT '失败原因全文（含原始信息，不拆分）',
    created_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '首次记录时间',
    updated_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    UNIQUE KEY uk_fail (project_code, refdes),
    KEY idx_fail_code (error_code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='入库/解析失败';

CREATE TABLE IF NOT EXISTS oa_sync_progress (
    job_key      VARCHAR(64) NOT NULL PRIMARY KEY COMMENT '任务键，如 history_backfill',
    done_through DATE NULL COMMENT '已处理完成的最后一天（含）；NULL=尚未开始'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='OA 定时任务进度（重启续跑）';

CREATE TABLE IF NOT EXISTS `process` (
    id                  BIGINT AUTO_INCREMENT PRIMARY KEY COMMENT '自增主键',
    project_code        VARCHAR(128) NOT NULL COMMENT '项目编码',
    oa_info_ok          TINYINT(1) NULL COMMENT '接口②四字段是否齐全',
    oa_info_note        VARCHAR(512) NULL COMMENT '接口②缺字段或 HTTP/OA 错误摘要',
    oa_download_ok      TINYINT(1) NULL COMMENT 'download_path/docId/fileId 是否可用',
    oa_download_note    VARCHAR(512) NULL COMMENT '下载参数说明或 docId/fileId',
    oa_brd_ok           TINYINT(1) NULL COMMENT '接口③是否解压出 BRD',
    oa_brd_note         VARCHAR(512) NULL COMMENT 'BRD 文件名或失败原因',
    parse_ok            TINYINT(1) NULL COMMENT '明工解析/入库是否成功',
    parse_note          VARCHAR(512) NULL COMMENT '明工入库摘要或失败原因（详见 ingest_failure）',
    created_at          DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at          DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    UNIQUE KEY uk_process_project (project_code),
    KEY idx_process_parse (parse_ok)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='项目 OA→明工 全流程状态';

CREATE TABLE IF NOT EXISTS pdf_refdes_backfill_log (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY COMMENT '自增主键',
    batch_key       VARCHAR(32) NOT NULL COMMENT '批次键 YYYYMMDDHHmmss',
    project_code    VARCHAR(128) NOT NULL DEFAULT '' COMMENT '项目编码；汇总行留空',
    refdes          VARCHAR(64) NOT NULL DEFAULT '' COMMENT '位号；汇总行留空',
    top_side        TINYINT(1) NULL COMMENT '1=正面 0=反面',
    pending_total   INT NULL COMMENT '本批待刷总数（仅 summary 行）',
    ok_count        INT NULL COMMENT '成功数（仅 summary 行）',
    fail_count      INT NULL COMMENT '失败数（仅 summary 行）',
    row_kind        VARCHAR(16) NOT NULL DEFAULT 'detail' COMMENT 'summary=批次汇总 ok=成功 detail=失败明细',
    detail          VARCHAR(512) NULL COMMENT '说明或失败原因',
    created_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '记录时间',
    KEY idx_backfill_batch (batch_key),
    KEY idx_backfill_project (project_code, refdes)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='PDF 位号坐标补刷日志';

-- 2026-10-23 追加，不改上面已有建表语句。
-- pending_project：OA 提前交来、设计尚未完成的项目。
-- pending_iou：这些项目每个位号对库内器件的 IoU，只保留 >= 0.8，rank_no=1 为最高分。
-- raster_grid：器件 0° 栅格。比对时再旋转，不存四个方向。

CREATE TABLE IF NOT EXISTS pending_project (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY COMMENT '自增主键',
    project_code    VARCHAR(128) NOT NULL COMMENT '待完成设计的项目编码',
    created_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    UNIQUE KEY uk_pending_project (project_code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='待完成项目';

CREATE TABLE IF NOT EXISTS pending_iou (
    id                  BIGINT AUTO_INCREMENT PRIMARY KEY COMMENT '自增主键',
    project_code        VARCHAR(128) NOT NULL COMMENT '待完成项目编码',
    refdes              VARCHAR(64) NOT NULL COMMENT '待完成项目上的位号',
    match_project_code  VARCHAR(128) NOT NULL COMMENT '库中命中器件的项目编码',
    match_refdes        VARCHAR(64) NOT NULL COMMENT '库中命中器件的位号',
    iou                 DOUBLE NOT NULL COMMENT '形态 IoU，仅保留 >= 0.8',
    rank_no             INT NOT NULL COMMENT '同一位号内按 IoU 从高到低，1 为最高',
    bbox_min_x          DOUBLE NULL COMMENT '命中器件 bbox 最小 X（mils）',
    bbox_min_y          DOUBLE NULL COMMENT '命中器件 bbox 最小 Y（mils）',
    bbox_max_x          DOUBLE NULL COMMENT '命中器件 bbox 最大 X（mils）',
    bbox_max_y          DOUBLE NULL COMMENT '命中器件 bbox 最大 Y（mils）',
    created_at          DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '写入时间',
    UNIQUE KEY uk_pending_iou (project_code, refdes, match_project_code, match_refdes),
    KEY idx_pending_iou_top (project_code, refdes, rank_no)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='待完成项目预计算 IoU';

SET @raster_exists := (
    SELECT COUNT(*) FROM information_schema.COLUMNS
    WHERE TABLE_SCHEMA = DATABASE()
      AND TABLE_NAME = 'footprint_library'
      AND COLUMN_NAME = 'raster_grid'
);
SET @raster_sql := IF(
    @raster_exists = 0,
    'ALTER TABLE footprint_library ADD COLUMN raster_grid MEDIUMBLOB NULL COMMENT ''1024x1024 的 0 度栅格，按行 packbits，约 128KB''',
    'SELECT 1'
);
PREPARE raster_stmt FROM @raster_sql;
EXECUTE raster_stmt;
DEALLOCATE PREPARE raster_stmt;

-- 2026-10-26 追加。栅格不再写入 footprint_library.raster_grid，改按项目编码+位号存这张表。
CREATE TABLE IF NOT EXISTS footprint_raster (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY COMMENT '自增主键',
    project_code    VARCHAR(128) NOT NULL COMMENT '项目编码，对应 footprint_library.project_code',
    refdes          VARCHAR(64) NOT NULL COMMENT '位号，对应 footprint_library.refdes',
    raster_size     SMALLINT NOT NULL COMMENT '边长像素，64/128/256/512/1024',
    raster_grid     MEDIUMBLOB NOT NULL COMMENT '0度栅格，按行 packbits',
    created_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    UNIQUE KEY uk_raster_project_refdes (project_code, refdes)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='器件栅格，按项目编码+位号关联器件表';

-- 2026-10-26 追加。pending_project.status：0=待完成，-1=已完成板已解出 BRD。
SET @pending_status_exists := (
    SELECT COUNT(*) FROM information_schema.COLUMNS
    WHERE TABLE_SCHEMA = DATABASE()
      AND TABLE_NAME = 'pending_project'
      AND COLUMN_NAME = 'status'
);
SET @pending_status_sql := IF(
    @pending_status_exists = 0,
    'ALTER TABLE pending_project ADD COLUMN status TINYINT NOT NULL DEFAULT 0 COMMENT ''0=待完成，-1=已完成板已解出BRD'' AFTER project_code',
    'SELECT 1'
);
PREPARE pending_status_stmt FROM @pending_status_sql;
EXECUTE pending_status_stmt;
DEALLOCATE PREPARE pending_status_stmt;

-- 2026-10-26 追加。待完成板回传的器件 JSON 直接放这张表，不进 MinIO。
-- 脚数少于 6 的留在表里。6 脚及以上由脚本算完 IoU 后按位号删除。
CREATE TABLE IF NOT EXISTS pending_json (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY COMMENT '自增主键',
    project_code    VARCHAR(128) NOT NULL COMMENT '待完成项目编码',
    refdes          VARCHAR(64) NOT NULL COMMENT '位号',
    pin_count       INT NOT NULL DEFAULT 0 COMMENT '脚数，少于 6 不计算 IoU',
    source_file     VARCHAR(255) NULL COMMENT '二开回传的文件名',
    json_text       MEDIUMTEXT NOT NULL COMMENT '器件 JSON 原文',
    created_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '写入时间',
    UNIQUE KEY uk_pending_json (project_code, refdes)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='待完成板器件 JSON，算完 IoU 后删除';

-- 2026-11-03 追加。pending_json 里 IoU 算失败、会反复卡住的项目位号。
-- pending_iou_worker 每轮跳过这里的项目编码+位号，JSON 仍留在 pending_json。
CREATE TABLE IF NOT EXISTS pending_json_block (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY COMMENT '自增主键',
    project_code    VARCHAR(128) NOT NULL COMMENT '项目编码',
    refdes          VARCHAR(64) NOT NULL COMMENT '位号',
    reason          VARCHAR(512) NULL COMMENT '最近一次失败原因',
    created_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '首次记入时间',
    updated_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '最近一次失败时间',
    UNIQUE KEY uk_pending_json_block (project_code, refdes)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='待完成 IoU 算不出来的项目位号，脚本跳过';
