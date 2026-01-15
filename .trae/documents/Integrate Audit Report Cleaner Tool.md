# 集成“审定报表清洗工具”并支持 Rules 可视化

## 1. 平台 API 升级：支持多工具配置隔离
为了让“审定报表清洗工具”和“月度报表清洗工具”互不干扰，需要升级平台 API 以支持 `tool_id` 参数。
- **目标文件**: `financial_analyzer_web.py`
- **变更点**:
  - `/api/config` (GET/POST): 增加 `tool_id` 参数。若未传则用默认 ID。根据 ID 读写 `tools/{id}/config.json`。
  - `/api/rules` (GET/POST): 增加 `tool_id` 参数。根据 ID 读写 `tools/{id}/rules.json`。
  - `/api/config/queries...`: 同样增加 `tool_id` 隔离（每个工具的常用查询分开存）。

## 2. 创建新工具：审定报表清洗工具
- **目录**: `tools/audit_report_cleaner/`
- **核心逻辑 (`core.py`)**:
  - 迁移原脚本逻辑：文件名日期提取、智能表头定位、左右分栏提取（BS）、单栏提取（PL/CF）。
  - 输出标准化：生成与平台兼容的 `AnalysisResult` 和 sqlite 数据库。
  - 输出路径：`output/audit_report_cleaner/{timestamp}/...`
- **配置 (`config.json`)**:
  - 默认输入/输出目录配置。
  - 特有配置项（如左右分栏的关键词配置，但这版先按原脚本逻辑硬编码，配置项留空备用）。
- **规则 (`rules.json`)**:
  - 初始化一个空的标准结构文件：`{"subject_aliases": {}, "variables": {}, "metrics": []}`。
- **Web UI (`web/`)**:
  - 复用月度工具的 UI 框架。
  - **保留规则页**：直接复用可视化编辑逻辑，支持对 `tools/audit_report_cleaner/rules.json` 的读写。
  - 移除不适用的配置项（如月度工具特有的 Sheet 关键词），保留通用配置。

## 3. 更新现有工具前端
- **目标文件**: `tools/monthly_report_cleaner/web/index.html`
- **变更点**: 修改所有 API 请求（config, rules, queries），自动带上 `?tool_id=monthly_report_cleaner`。

## 4. 注册与验证
- **注册**: 在 `tools/builtin_tools.py` 注册 `audit_report_cleaner`。
- **验证**:
  - 启动 Web 服务。
  - 切换到“审定报表清洗工具”。
  - 检查配置页能否保存。
  - 检查规则页能否新增规则并保存到 `tools/audit_report_cleaner/rules.json`。
  - 运行清洗任务，检查日志和输出结果。
