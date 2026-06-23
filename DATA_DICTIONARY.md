# 财务小工具平台数据字典

本数据字典定义了“本地优先财务小工具平台”的核心数据底座（`warehouse.sqlite`）和运行记录库（`run_index.sqlite`）的表结构、字段含义以及设计规范。

---

## 1. 财务数据仓库 (`warehouse.sqlite`)

财务数据仓库用于存储由 `report_ingestor`（报表清洗与落库工具）清洗后的标准化财务分录、导入批次信息，以及其他派生工具（如验证报告、财务指标）计算生成的业务结果。

### 1.1 导入批次表 (`import_batch`)

记录财务报表每一次执行落库的批次信息，支持全局重跑与期间追溯。

| 字段名称 | 数据类型 | 约束 | 描述/解释 | 示例值 |
| :--- | :--- | :--- | :--- | :--- |
| `batch_id` | TEXT | PRIMARY KEY | 批次唯一标识，通常为执行时的时间戳 | `"20260622_123045"` |
| `tool_id` | TEXT | | 执行落库的工具ID | `"report_ingestor"` |
| `run_id` | TEXT | | 与 `run_index.sqlite` 关联的运行ID | `"20260622_123045"` |
| `created_at` | TEXT | NOT NULL | 批次创建时间，ISO 8601 格式 | `"2026-06-22T12:30:45"` |
| `mode` | TEXT | | 写入模式，支持覆盖或追加 | `"replace"` |
| `input_dir` | TEXT | | 数据源的输入目录绝对/相对路径 | `"/path/to/input"` |
| `file_glob` | TEXT | | 数据源文件的通配符匹配规则 | `"*.xlsx"` |

---

### 1.2 标准财务数据表 (`warehouse_cleaned`)

存储所有经过清洗、标准化科目映射后的财务分录细节。这是平台的最核心业务数据源（`warehouse_cleaned`）。

| 字段名称 | 数据类型 | 约束 | 描述/解释 | 示例值 |
| :--- | :--- | :--- | :--- | :--- |
| `batch_id` | TEXT | NOT NULL | 关联的批次ID（外键关联 `import_batch.batch_id`） | `"20260622_123045"` |
| `row_hash` | TEXT | NOT NULL | 分录行的哈希校验码，由源分录关键字段SHA1生成，防重 | `"7a1bf6ec..."` |
| `源文件` | TEXT | | 原始 Excel 报表的文件名，方便追溯原始数据 | `"报表-2026年1期月报.xlsx"` |
| `来源Sheet` | TEXT | | 原始报表中的 Sheet 标签名 | `"资产负债表"` |
| `期间` | TEXT | INDEX | 财务报告期间，标准格式为 `YYYYMM` | `"202601"` |
| `年份` | TEXT | INDEX | 财务报告年度，标准格式为 `YYYY` | `"2026"` |
| `主体` | TEXT | INDEX | 从表头区域识别出的报表主体/公司/单位名称 | `"某某有限公司"` |
| `报表口径` | TEXT | | 数据范围口径，通常为“单体”或“合并” | `"单体"` |
| `报表类型` | TEXT | INDEX | 报表分类：`资产负债表`、`利润表`、`现金流量表` | `"资产负债表"` |
| `大类` | TEXT | | 财务要素大类，如：`资产`、`负债及权益`、`损益`、`现金流` | `"资产"` |
| `科目` | TEXT | | 原始文件中的会计科目名称（含空格或符号） | `"  货币资金"` |
| `科目规范` | TEXT | INDEX | 经过别名匹配和科目映射后的标准会计科目名称 | `"货币资金"` |
| `时间属性` | TEXT | INDEX | 金额的时间发生维度，如：`期末余额`、`年初余额`、`本期金额`、`本年累计金额` | `"期末余额"` |
| `金额` | REAL | | 标准化后的本币金额数值 | `1250000.50` |

* **联合主键**：`(batch_id, row_hash)`
* **常用索引**：
  * `ix_warehouse_period_stmt ON warehouse_cleaned(期间, 报表类型, 时间属性)` (加速常用科目余额表组合查询)
  * `ix_warehouse_subject ON warehouse_cleaned(科目规范)` (加速单科目跨期间透视)

---

### 1.3 报表验证结果表 (`warehouse_validation`)

由 `validation_report`（验证报告工具）生成，存储资产负债表平衡性验证、勾稽关系检查结果。

| 字段名称 | 数据类型 | 约束 | 描述/解释 | 示例值 |
| :--- | :--- | :--- | :--- | :--- |
| `tool_id` | TEXT | NOT NULL | 运行的验证工具ID | `"validation_report"` |
| `run_id` | TEXT | NOT NULL | 本次验证运行ID | `"20260622_124500"` |
| `source_tool_id` | TEXT | | 验证的数据源工具ID，通常为 `report_ingestor` | `"report_ingestor"` |
| `source_run_id` | TEXT | | 验证的源数据批次/运行ID | `"20260622_123045"` |
| `源文件` | TEXT | NOT NULL | 被验证 of Excel 报表文件名 | `"报表-2026年1期月报.xlsx"` |
| `来源Sheet` | TEXT | | 验证对应的报表 Sheet 名称 | `"资产负债表"` |
| `期间` | TEXT | NOT NULL | 对应的财务期间 `YYYYMM` | `"202601"` |
| `验证项目` | TEXT | NOT NULL | 验证的物理检查项，如“资产 vs 负债及权益”、“借贷平衡” | `"资产 = 负债 + 权益"` |
| `时间属性` | TEXT | NOT NULL | 对应的时点/时段维度 | `"期末余额"` |
| `差额` | REAL | | 勾稽平衡性差额（正常应为 0.0） | `0.0` |
| `是否平衡` | TEXT | INDEX | 平衡状态：`是` / `否` | `"是"` |
| `验证结果` | TEXT | | 详细的校验信息文本 | `"资产(100) = 负债及权益(100)"` |

* **联合主键**：`(tool_id, run_id, 源文件, 期间, 验证项目, 时间属性)`

---

### 1.4 财务指标结果表 (`warehouse_metrics`)

由 `financial_metrics`（财务指标生成工具）派生计算生成，存储各期间主体维度的比率指标。

| 字段名称 | 数据类型 | 约束 | 描述/解释 | 示例值 |
| :--- | :--- | :--- | :--- | :--- |
| `tool_id` | TEXT | NOT NULL | 指标计算工具ID | `"financial_metrics"` |
| `run_id` | TEXT | NOT NULL | 本次计算运行ID | `"20260622_125000"` |
| `source_tool_id` | TEXT | | 基础数据源工具ID | `"report_ingestor"` |
| `source_run_id` | TEXT | | 数据源的运行批次ID | `"20260622_123045"` |
| `源文件` | TEXT | NOT NULL | 被分析的源报表文件名 | `"报表-2026年1期月报.xlsx"` |
| `期间` | TEXT | NOT NULL | 对应的指标期间 `YYYYMM` | `"202601"` |
| `metrics_json` | TEXT | | 序列化的指标键值对（流动比率、资产负债率、净资产收益率等） | `{"流动比率": 1.85, "资产负债率": 0.45}` |

* **联合主键**：`(tool_id, run_id, 源文件, 期间)`

---

## 2. 平台运行索引 (`run_index.sqlite`)

平台底座的核心管理数据库，用于记录所有工具的历史运行轨迹、参数快照、产生的物理文件清单，支持工作台的可追溯性。

### 2.1 任务运行表 (`runs`)

| 字段名称 | 数据类型 | 约束 | 描述/解释 | 示例值 |
| :--- | :--- | :--- | :--- | :--- |
| `tool_id` | TEXT | PRIMARY KEY | 工具唯一标识符 | `"report_ingestor"` |
| `run_id` | TEXT | PRIMARY KEY | 运行唯一标识符，通常为执行时间戳 | `"20260622_123045"` |
| `started_at` | TEXT | | 运行启动时间，ISO 8601 / 秒精度 | `"2026-06-22T12:30:45"` |
| `finished_at` | TEXT | | 运行完成时间，ISO 8601 / 秒精度 | `"2026-06-22T12:31:02"` |
| `status` | TEXT | | 运行最终状态：`ok` (正常完成), `error` (异常中断), `cancelled` (用户取消) | `"ok"` |
| `cleaned_path` | TEXT | | 产生的主要清洗结果 Excel 文件物理绝对路径（若有） | `"L:/Web/.../output/report_ingestor/20260622_123045.xlsx"` |
| `cleaned_sqlite_path` | TEXT | | 产生的主要 SQLite 文件绝对路径（若有） | `"L:/Web/.../data/warehouse.sqlite"` |
| `cleaned_rows` | INTEGER | | 清洗行数（针对落库类/数据生成类工具） | `370` |
| `processed_files` | INTEGER | | 成功处理的原始文件个数 | `2` |
| `found_files_json` | TEXT | | 查找到的文件列表 JSON 数组 | `["L:/.../报表1.xlsx", "L:/.../报表2.xlsx"]` |
| `errors_json` | TEXT | | 运行过程中产生的 Error 信息文本列表 (JSON 数组) | `[]` |
| `warnings_json` | TEXT | | 运行中产生的 Warning 信息文本列表 (JSON 数组) | `["报表202605_无锡... 未识别到三大报表Sheet"]` |
| `artifacts_json` | TEXT | | 生成的产物列表 (JSON 数组)，包含名称、路径和类型 | `[{"name": "清洗表", "path": "...", "kind": "xlsx"}]` |
| `input_json` | TEXT | | 工具运行的配置输入快照（包含输入路径、匹配通配符、工具独立参数等） | `{"input_dir": "L:/...", "file_glob": "*.xlsx", "tool_params": {}}` |
| `output_json` | TEXT | | 工具运行的指标输出快照（包含行数、异常笔数、各派生产物路径） | `{"cleaned_rows": 370, "processed_files": 2, ...}` |
| `meta_json` | TEXT | | 平台兼容性元数据 JSON 快照 | `{"output_dir": "output", ...}` |

* **联合主键**：`(tool_id, run_id)`
* **常用索引**：
  * `idx_runs_tool_started ON runs(tool_id, started_at DESC)` (加速历史运行记录的分页查看)
  * `idx_runs_tool_finished ON runs(tool_id, finished_at DESC)`

---

## 3. 设计与数据流约束

1. **绝对路径与相对路径**：
   - 数据库存储物理路径时，使用物理绝对路径以防不同进程工作目录不一致；
   - API 提供下载和展示时，必须经过安全性校验，禁止下载外部目录的敏感文件（只能下载 `data/` 或各工具配置的 `output_dir` 下的文件）。
2. **读写仓库事务约束**：
   - 使用 SQLite 时务必开启 `WAL` 日志模式以提高并发读取性能（在 `run_index.py` 中已固化）。
   - 数据写入完毕必须立即提交事务并释放连接，避免库文件出现 `database is locked`。
