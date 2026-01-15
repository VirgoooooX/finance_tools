# 财务数据分析工具（Web/命令行）

本项目用于批量清洗财务报表（Excel）、生成验证结果与财务指标，并提供本机 Web 界面进行配置、运行、查询与规则编辑。

仓库地址：<https://github.com/VirgoooooX/finance_tools>

## 功能
- 多工具插件化：按工具切换不同清洗/规则逻辑
- Web 界面：配置、扫描文件、运行、查看日志、查询数据、编辑 rules.json
- 数据输出：每次运行产出独立时间戳目录，避免覆盖
- SQLite 缓存：清洗后的结构化数据落地到 `data/*.sqlite`，便于查询

当前内置工具：
- `monthly_report_cleaner`：月度报表清洗 +（可选）验证/指标
- `audit_report_cleaner`：审定报表清洗（输出清洗结果 + SQLite）

## 运行环境
- Windows 10/11
- Python 3.11+

依赖（手动安装即可）：
- pandas、openpyxl（读写 Excel）
- fastapi、uvicorn（Web 服务）

## 快速开始（Web）
建议使用虚拟环境：

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -U pip
py -m pip install pandas openpyxl fastapi uvicorn
```

启动 Web（默认 8765 端口）：

```powershell
py financial_analyzer.py --web --port 8765
```

常用参数：
- 不自动打开浏览器：`py financial_analyzer.py --web --no-browser`
- 自动选择空闲端口：`py financial_analyzer.py --web --port 0`

## Web 使用流程
1. 选择工具：顶部工具列表切换（不同工具有不同配置/规则页）
2. 配置：填写输入目录、文件匹配模式、输出目录/文件名等并保存
3. 扫描：确认识别到需要处理的 Excel 列表
4. 运行：开始运行，在“日志/结果”查看进度、输出路径、结果预览
5. 查询：在“查询”页对清洗数据进行筛选/聚合，并可导出
6. 规则：在“规则”页可视化编辑 `rules.json`

## 命令行模式
命令行会读取配置文件并运行对应工具（以 `tool_id` 为准）：

```powershell
py financial_analyzer.py --config .\tools\monthly_report_cleaner\config.json
```

覆盖部分配置（可选）：

```powershell
py financial_analyzer.py --config .\tools\monthly_report_cleaner\config.json --input-dir "D:\data" --glob "*.xlsx" --output-dir "output"
```

强制指定工具（可选）：

```powershell
py financial_analyzer.py --config .\tools\monthly_report_cleaner\config.json --tool-id audit_report_cleaner
```

## 配置与规则文件
- 工具配置：`tools/<tool_id>/config.json`
- 工具规则：`tools/<tool_id>/rules.json`

Web 界面里的“保存配置/保存规则”会直接写回对应文件，便于持久化。

## 输出与数据目录
- 运行输出目录：`output/<tool_id>/<YYYYMMDD_HHMMSS>/`
  - 清洗表：`<output_basename>`
  - 月度工具可额外生成：`*_验证报告.xlsx`、`*_财务指标.xlsx`
- SQLite 数据目录：`data/*.sqlite`
  - 清洗表会写入 `cleaned` 表，并建立常用列索引（用于 Web 查询）

## 打包成 EXE（可选）
项目根目录提供打包脚本 [build_exe.py](build_exe.py)：

```powershell
py build_exe.py
```

产物：
- `dist/财务数据分析工具.exe`

说明：
- 运行 EXE 会启动本机 Web（默认 `127.0.0.1`），并在控制台输出日志与端口信息
- 输出默认落在 EXE 同级目录的 `output/` 与 `data/`

## 开发
新增工具、Web 前端开发注意事项、输出规范等见：
- [DEVELOPMENT_GUIDE_ZH.md](DEVELOPMENT_GUIDE_ZH.md)
- [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md)

## 常见问题
- 端口被占用：启动时改端口，例如 `--port 9000`
- 依赖缺失：按“快速开始”安装依赖（建议用虚拟环境）
- JSON 中文乱码：建议用 UTF-8 保存；项目读取/保存时会尝试修复常见编码问题
