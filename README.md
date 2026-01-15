# 财务数据分析平台（Web / 命令行 / EXE）

面向财务 Excel 报表的本机分析平台：以“工具插件”方式沉淀清洗、验证、指标、查询与规则配置能力，并提供统一 Web Shell 作为操作入口。

仓库地址：<https://github.com/VirgoooooX/finance_tools>

## 目录
- [平台概览](#平台概览)
- [架构与目录结构](#架构与目录结构)
- [快速开始（Web）](#快速开始web)
- [命令行模式](#命令行模式)
- [配置、规则与工具](#配置规则与工具)
- [输出与数据目录](#输出与数据目录)
- [打包成 EXE](#打包成-exe)
- [开发与扩展](#开发与扩展)
- [常见问题](#常见问题)

## 平台概览

平台化的核心点：
- 统一入口：一个 Web Shell 管理工具切换、日志流、开始/停止与状态展示
- 插件化扩展：新增工具只需新增一个 `tools/<tool_id>/` 目录并注册
- 数据可查询：清洗结果会写入 SQLite（`data/*.sqlite`），Web 端支持筛选/聚合/导出
- 输出可追溯：每次运行生成时间戳目录，避免覆盖并便于追溯

当前内置工具：
- `monthly_report_cleaner`：月度报表清洗（可选：验证、指标）
- `audit_report_cleaner`：审定报表清洗（清洗结果 + SQLite）

## 架构与目录结构

### 架构（简图）
```text
┌──────────────────────────────┐
│ Web Shell (web/index.html)   │
│ - 工具选择/Tab/控制按钮       │
│ - SSE 日志/进度订阅与展示      │
└───────────────┬──────────────┘
                │ iframe
┌───────────────▼──────────────┐
│ Tool Web (tools/<id>/web/)   │
│ - 配置表单/查询/结果展示      │
│ - 通过 /api/* 调用后端        │
└───────────────┬──────────────┘
                │
┌───────────────▼──────────────┐
│ FastAPI (financial_analyzer_web.py)
│ - 配置读写 / 规则读写 / 运行管理
│ - 工具 Web 资源静态服务         │
└───────────────┬──────────────┘
                │
┌───────────────▼──────────────┐
│ Tools (tools/<id>/core.py)   │
│ - 清洗/验证/指标/落库/产物输出  │
└──────────────────────────────┘
```

### 目录结构（核心）
```text
root/
├── fa_platform/                  平台公共库（路径/JSON/Web 辅助）
├── tools/                        工具插件目录
│   ├── builtin_tools.py           内置工具注册
│   ├── monthly_report_cleaner/    示例工具：月度报表
│   └── audit_report_cleaner/      示例工具：审定报表
├── web/                          全局 Web Shell 与样式
├── financial_analyzer.py         启动入口（Web/命令行分发）
├── financial_analyzer_web.py     Web 服务（FastAPI）
├── financial_analyzer_core.py    配置类型/工具注册表/命令行执行
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

### 启动
默认端口 8765：

```powershell
py financial_analyzer.py --web --port 8765
```

常用参数：
- 不自动打开浏览器：`py financial_analyzer.py --web --no-browser`
- 端口传 0 自动选择空闲端口：`py financial_analyzer.py --web --port 0`

### 使用流程（建议）
1. 选择工具：顶部工具列表切换
2. 配置：填写输入目录、匹配模式（glob）、输出目录/文件名并保存
3. 扫描：确认识别到待处理 Excel 列表
4. 运行：开始运行，在“日志/结果”查看进度与输出路径
5. 查询：在“查询”页筛选/聚合，并可导出 CSV/XLSX（视工具实现）
6. 规则：在“规则”页编辑 `rules.json` 并保存

## 命令行模式

命令行以“配置文件”为入口运行工具（按配置中的 `tool_id` 决定具体工具）：

```powershell
py financial_analyzer.py --config .\tools\monthly_report_cleaner\config.json
```

常用覆盖项：

```powershell
py financial_analyzer.py --config .\tools\monthly_report_cleaner\config.json --input-dir "D:\data" --glob "*.xlsx" --output-dir "output"
```

强制指定工具（覆盖配置中的 `tool_id`）：

```powershell
py financial_analyzer.py --config .\tools\monthly_report_cleaner\config.json --tool-id audit_report_cleaner
```

## 配置、规则与工具

### 工具配置（config.json）
每个工具有独立配置文件：
- `tools/<tool_id>/config.json`

典型字段：
- `tool_id`：工具 ID（决定运行哪个工具）
- `input_dir`：输入目录
- `file_glob`：文件匹配模式（例如 `*.xlsx`）
- `output_dir`：输出目录（可写相对路径，默认会解析到应用根目录下）
- `output_basename`：输出文件名（如 `清洗结果.xlsx`）
- `saved_queries`：Web 端保存的查询条件（工具自定义使用）

Web 中“保存配置”会写回该文件，用于持久化。

### 工具规则（rules.json）
每个工具可有独立规则文件：
- `tools/<tool_id>/rules.json`

规则页用于编辑并保存该文件，常用于：
- 科目同义词、关键词匹配规则
- 变量取数规则与指标计算公式（视工具实现）

### 工具 Web（manifest.json + index.html）
工具的前端资源位于：
- `tools/<tool_id>/web/manifest.json`
- `tools/<tool_id>/web/index.html`

`manifest.json` 用于声明：
- 工具标题/图标
- Tab 列表（Web Shell 会据此渲染顶部 Tab，并通过 URL hash 驱动 iframe 内路由）

## 输出与数据目录

### 输出目录（可追溯）
默认输出在应用根目录下的 `output/`，并按工具与时间戳分层：

```text
output/
└── <tool_id>/
    └── <YYYYMMDD_HHMMSS>/
        ├── <output_basename>
        ├── <output_basename>_验证报告.xlsx      （视工具配置/实现）
        └── <output_basename>_财务指标.xlsx      （视工具配置/实现）
```

### SQLite 数据（可查询）
默认落地到应用根目录下的 `data/`：
- `data/*.sqlite`

通常会包含：
- `cleaned` 表：清洗后的明细数据（并建立常用列索引，提升 Web 查询速度）

## 打包成 EXE

项目提供一键打包脚本：[build_exe.py](build_exe.py)

```powershell
py build_exe.py
```

打包产物：
- `dist/财务数据分析工具.exe`

运行说明：
- 双击 EXE 会启动本机 Web（默认监听 `127.0.0.1`）
- 输出默认落在 EXE 同级目录的 `output/` 与 `data/`

## 开发与扩展

### 新增一个工具（概览）
1. 新建目录：`tools/<tool_id>/`
2. 实现核心逻辑：`tools/<tool_id>/core.py`，提供标准 `run_analysis(cfg, logger, progress_cb, cancel_event)`（参照现有工具）
3. 配置默认参数：`tools/<tool_id>/config.json`
4. 提供 Web：`tools/<tool_id>/web/manifest.json` + `index.html`
5. 注册到平台：更新 [builtin_tools.py](tools/builtin_tools.py)

更完整的开发规范与 Web 注意事项见：
- [DEVELOPMENT_GUIDE_ZH.md](DEVELOPMENT_GUIDE_ZH.md)
- [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md)

## 常见问题
- 端口被占用：启动时改端口，例如 `--port 9000`
- 依赖缺失：确认已安装 `pandas openpyxl fastapi uvicorn`
- JSON 中文乱码：建议用 UTF-8 保存；项目读取/保存时会尝试修复常见编码问题
