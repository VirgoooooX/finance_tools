# 财务数据分析平台（Web / 命令行 / EXE）

面向财务 Excel 报表的本机分析平台：以“工具插件”方式沉淀清洗、验证、指标、查询与规则配置能力，并提供统一 Web Shell 作为操作入口。

仓库地址：https://github.com/VirgoooooX/finance_tools

## 平台概览

平台化的核心点：
- 统一入口：一个 Web Shell 管理工具切换、日志流、开始/停止与状态展示
- 插件化扩展：新增工具只需新增一个 `tools/<tool_id>/` 目录并注册
- 数据可查询：清洗结果会写入累计库（`data/warehouse.sqlite`），Web 端支持筛选/聚合/导出
- 输出可追溯：每次运行生成时间戳目录，并写入 Run Index（`data/run_index.sqlite`）

当前内置工具：
- `report_ingestor`：报表清洗与落库（可选：验证、指标、累计库查询）
- `validation_report`：基于清洗源批次生成验证报告
- `financial_metrics`：基于清洗源批次生成财务指标
- `payment_monitor`：收款进度监控

## 架构与目录结构

```text
root/
├── fa_platform/                  平台公共库（路径/JSON/Web/RunIndex/管线）
│   ├── paths.py                  路径与资源定位
│   ├── jsonx.py                  JSON 净化（NaN/datetime/pandas）
│   ├── webx.py                   Web 辅助（SSE/工具资源发现）
│   ├── pipeline.py               清洗管线公共能力（run_dir/sqlite/artifacts）
│   └── run_index.py              Run Index（data/run_index.sqlite）
├── tools/                        工具插件目录
│   ├── builtin_tools.py          内置工具注册
│   ├── report_ingestor/
│   ├── validation_report/
│   ├── financial_metrics/
│   └── payment_monitor/
├── web/                          全局 Web Shell 与样式
├── financial_analyzer_web.py     Web 服务（FastAPI）
├── financial_analyzer_core.py    配置类型/工具注册表/命令行入口函数 main()
└── build_exe.py                  PyInstaller 打包脚本
```

## 快速开始（Web）

### 环境要求
- Windows 10/11
- Python 3.11+

### 安装依赖
建议使用虚拟环境：

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -U pip
py -m pip install pandas openpyxl fastapi uvicorn
```

### 启动（推荐：显式指定配置）
选择一个工具的配置文件作为启动入口（例如 report_ingestor）：

```powershell
py -c "import financial_analyzer_web as w; raise SystemExit(w.run_web(r'.\tools\report_ingestor\config.json', port=8765))"
```

说明：
- `run_web(config_path, host='127.0.0.1', port=8765, open_browser=True)`
- `config_path` 指向某个工具的 `tools/<tool_id>/config.json`

### 启动（可选：用环境变量指定默认工具）

```powershell
$env:FA_DEFAULT_TOOL_ID = "report_ingestor"
py .\financial_analyzer_web.py
```

## 命令行模式（CLI）

CLI 入口是 `financial_analyzer_core.main()`，用 `-c` 调用：

```powershell
py -c "import financial_analyzer_core as c; raise SystemExit(c.main())" --config .\tools\report_ingestor\config.json
```

可覆盖项（覆盖配置文件中的字段）：
- `--tool-id <tool_id>`
- `--input-dir <dir>`
- `--output-dir <dir>`
- `--glob <pattern>`

示例：

```powershell
py -c "import financial_analyzer_core as c; raise SystemExit(c.main())" --config .\tools\report_ingestor\config.json --input-dir "D:\data" --glob "*.xlsx"
```

## 配置、规则与工具

### 工具配置（tools/<tool_id>/config.json）
平台通用配置字段位于顶层（`input_dir/file_glob/output_dir/...`），工具特有配置进入：
- `tool_params.<tool_id>.*`

Web 端保存配置时：
- 顶层已知字段直接写入 AppConfig
- 未知字段自动归档到 `tool_params[tool_id]`

### 工具规则（tools/<tool_id>/rules.json）
规则页编辑并保存该文件，常用于同义词、取数规则、指标公式等（视工具实现）。

## 输出与数据目录

### 输出目录（可追溯）
默认输出在 `output/<tool_id>/` 下。不同工具可继续细分为 `monthly/` 或 `<YYYYMMDD_HHMMSS>/`，每次运行会通过 Run Index 记录实际产物路径。

### 数据目录（可查询 + 可追溯）
默认落地到 `data/`：
- `data/warehouse.sqlite`：累计财务数据仓库，包含清洗表、验证结果、指标结果等
- `data/run_index.sqlite`：Run Index（记录每次运行的状态、告警、产物、输入/输出快照等）

### 运行历史与源批次
- `/api/runs?tool_id=<tool_id>`：返回指定工具自己的运行历史
- `/api/source-runs?source_tool_id=report_ingestor`：返回可供验证/指标等下游工具选择的清洗源批次
- `/api/artifacts/download?path=...`：下载 `data/` 或各工具 `output_dir` 下的产物
- `/api/artifacts/open`：在服务端本机打开产物路径；如果服务开放给局域网其他机器，打开动作仍发生在运行服务的那台机器上

## 打包成 EXE

项目提供一键打包脚本：[build_exe.py](build_exe.py)

```powershell
py build_exe.py
```

默认入口为 `financial_analyzer_web.py`，可用 `--entry` 覆盖。

打包产物：
- `dist/财务数据分析工具.exe`

运行说明：
- 双击 EXE 会启动本机 Web（默认监听 `127.0.0.1`）
- 输出默认落在 EXE 同级目录的 `output/` 与 `data/`

## 本机与局域网访问

默认建议本机运行：`host='127.0.0.1'`。如果需要让局域网其他机器访问，可以启动时使用 `host='0.0.0.0'` 并放行防火墙端口。第一阶段不包含账号/权限系统，因此只建议在可信内网临时开放。

更多开发规范见：DEVELOPMENT_GUIDE_ZH.md
