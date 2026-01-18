# 财务数据分析平台开发指南（平台化版本）

本指南聚焦“平台化约束”：配置去工具化、清洗管线复用、Run Index 可追溯，以及 Web Shell + iframe 的插件式 UI 结构。

## 1. 项目架构概览

- 后端：Python（FastAPI）提供配置/运行/查询 API，负责工具调度与产物管理
- 前端：统一 Web Shell（web/index.html），以 iframe 加载工具前端（tools/<tool_id>/web）
- 工具：tools/<tool_id>/core.py 实现清洗/验证/指标等逻辑，尽量复用平台公共管线
- 可追溯：每次运行写入 Run Index（data/run_index.sqlite），支持历史 run_id 查询与导出

### 目录结构（核心）

```text
root/
├── fa_platform/
│   ├── paths.py        路径与资源定位（开发/打包兼容）
│   ├── jsonx.py        JSON 净化（必须用于 API 输出）
│   ├── webx.py         Web 辅助（SSE/工具资源发现）
│   ├── pipeline.py     清洗管线公共能力（run_dir/sqlite/artifacts）
│   └── run_index.py    Run Index（data/run_index.sqlite）
├── tools/
│   ├── builtin_tools.py
│   └── <tool_id>/
│       ├── core.py
│       ├── config.json
│       ├── rules.json（可选）
│       └── web/
│           ├── manifest.json
│           └── index.html
├── web/
│   ├── index.html
│   └── styles.css
├── financial_analyzer_core.py
├── financial_analyzer_web.py
└── build_exe.py
```

## 2. 配置平台化（AppConfig + tool_params）

平台通用配置由 `AppConfig` 承载（input/output/glob/开关等），工具特有配置统一放入：

- `cfg.tool_params[tool_id]`（一个 dict bucket）

要点：
- Web API 的 `/api/config` 会把 `tool_params[tool_id]` 展平到返回 JSON 顶层，工具页可直接用“扁平字段”渲染表单
- Web API 的保存配置会将“非 AppConfig 字段”自动写回 `tool_params[tool_id]`，避免 AppConfig 被工具字段污染

## 3. Run Index（运行索引，可追溯）

每次工具运行完成后，平台会把运行信息 upsert 到：
- `data/run_index.sqlite`

记录内容包括：
- tool_id / run_id
- cleaned_path / cleaned_sqlite_path
- cleaned_rows / processed_files / errors
- meta：输入/输出/匹配模式 + tool_params 快照（用于审计与复现）

## 4. 清洗管线框架平台化（fa_platform.pipeline）

平台提供可复用的清洗管线能力，工具实现尽量复用而不是各写一套：

- `build_run_dir(output_dir, tool_id, stamp=None) -> (ts, run_dir)`
- `write_sqlite_tables(sqlite_path, cleaned, validation=None, metrics=None)`
- `build_artifacts(cleaned_path, cleaned_sqlite_path, validation_path, metrics_path) -> List[dict]`

约定：
- 输出目录：`output/<tool_id>/<YYYYMMDD_HHMMSS>/`
- 数据目录：`data/`（SQLite + run_index.sqlite）

## 5. 开发新工具（推荐模板）

### 5.1 目录与注册
- 新建目录：`tools/<tool_id>/`
- 提供：`core.py / config.json / web/manifest.json / web/index.html`
- 在 `tools/builtin_tools.py` 中注册

### 5.2 config.json（示例）

```json
{
  "tool_id": "my_new_tool",
  "input_dir": "",
  "file_glob": "*.xlsx",
  "output_dir": "output",
  "output_basename": "清洗后的AI标准财务表.xlsx",
  "generate_validation": true,
  "generate_metrics": true,
  "tool_params": {
    "my_new_tool": {
      "some_tool_specific_key": "value"
    }
  }
}
```

### 5.3 core.py（run_analysis 约定）

- `run_analysis(cfg: AppConfig, logger=None, progress_cb=None, cancel_event=None) -> AnalysisResult`

建议输出：
- `AnalysisResult.cleaned_path`
- `AnalysisResult.cleaned_sqlite_path`
- （可选）`validation_path / metrics_path`
- `AnalysisResult.artifacts`（用于 Web 展示“可打开/可下载”的产物列表）
- `AnalysisResult.run_id`（建议与时间戳一致）

## 6. Web Shell 与 iframe 工具页约定

### 6.1 manifest.json 与 Tabs
- 工具页的 Tab 声明由 `tools/<tool_id>/web/manifest.json` 提供
- Web Shell 会渲染 Tabs，并通过 URL hash（如 `#tab=config`）驱动 iframe 内部切换
- 工具页必须监听 `hashchange` 并切换视图

### 6.2 API 输出必须 sanitize_json
所有 API 返回对象必须先做 JSON 净化，避免 Pandas/NaN/datetime 序列化异常。

## 7. 启动方式（与仓库真实入口一致）

### 7.1 启动 Web（推荐）
显式传入某个工具的 config.json：

```powershell
py -c "import financial_analyzer_web as w; raise SystemExit(w.run_web(r'.\tools\report_ingestor\config.json', port=8765))"
```

### 7.2 启动 CLI

```powershell
py -c "import financial_analyzer_core as c; raise SystemExit(c.main())" --config .\tools\report_ingestor\config.json
```

## 8. 输出与存储规范

- 输出目录：`output/<tool_id>/<YYYYMMDD_HHMMSS>/`
- 数据目录：`data/`
  - 清洗 SQLite：建议包含 `cleaned` 表，并对常用过滤列建索引
  - Run Index：`data/run_index.sqlite` 记录每次运行的路径与参数快照

## 9. 打包与发布

- 打包脚本：build_exe.py（PyInstaller）
- 资源文件：web/ 与 tools/ 下的 web 资源需被包含
- 动态依赖：必要时加入 hiddenimports 或确保可被静态导入链覆盖
- 入口文件：当前 build_exe.py 仍引用 `financial_analyzer.py`，如缺失需先调整打包入口
