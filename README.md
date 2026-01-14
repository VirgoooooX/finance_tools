# 财务数据分析工具（Web/命令行）

本项目用于批量清洗财务报表、生成验证报告与财务指标，并提供本机 Web 界面。

## 功能概览
- 清洗：将原始 Excel 报表统一整理成“清洗后的AI标准财务表.xlsx”
- 验证：生成验证报告，并在 Web 里查询不平衡记录/明细
- 指标：基于 rules.json 的变量取数与公式计算生成指标结果
- Web：本机可视化界面，支持保存配置、查询、运行、查看输出、编辑 rules.json

## 运行环境
- Windows 10/11
- Python 3.11+

## 推荐安装方式（可选）
建议使用虚拟环境，避免与其他项目依赖冲突：
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 安装依赖
```powershell
py -m pip install -U pip
py -m pip install pandas openpyxl fastapi uvicorn
```

## 启动 Web 界面
```powershell
py financial_analyzer.py --web --port 8765
```

说明：
- 直接运行 `py financial_analyzer.py`（不带参数）会默认启动 Web。
- 如不想自动打开浏览器：`py financial_analyzer.py --web --no-browser`，再手动访问 `http://127.0.0.1:8765/`。
- 端口可传 0 自动选择空闲端口：`py financial_analyzer.py --web --port 0`

## Web 使用流程（建议）
1. 配置：在“配置”页设置输入目录、文件匹配模式、输出目录等，点“保存配置”
2. 扫描：点“扫描文件”确认识别到要处理的 Excel 列表
3. 运行：点“开始运行”，在“日志/结果”查看进度与输出路径
4. 查询：用“查询”页在清洗表中筛选科目/日期/报表类型等，并可导出 CSV
5. 规则：在“规则”页可视化编辑 rules.json（同义词/变量/指标/高级 JSON）

## 规则（rules.json）说明
规则文件默认位置：`config/rules.json`

主要结构：
- subject_aliases：科目同义词，用于更稳的关键词匹配
- variables：变量取数规则（关键词、报表类型、时间属性、大类等）
- metrics：指标计算公式（支持直接引用变量与 safe_div 等）

指标公式说明：
- 推荐用 `safe_div(a, b)` 做除法（分母为 0 时更稳）
- 规则页提供“插入变量 / 插入 safe_div / 校验公式 / 引用变量跳转”等编辑辅助

## 命令行模式（可选）
```powershell
py financial_analyzer.py --config financial_analyzer_config.json
```

## 配置文件说明
- 核心配置文件：`config/financial_analyzer_config.json`
- 程序运行时会读取/写入该文件
- 打包后会默认使用 `.exe` 同级目录下的 `config/financial_analyzer_config.json`（便于持久化）

## 打包成 EXE（Web 版）
项目根目录提供了自动打包脚本 [build_exe.py](file:///l:/Web/Financial%20data%20analysis/build_exe.py)：

```powershell
py build_exe.py
```

打包结果：
- 输出文件：`dist/财务数据分析工具.exe`
- 图标文件：默认使用项目根目录下的 `app_icon.ico`
- 静态资源：会自动打包 `web/` 目录
- 默认会打包 `config/` 目录（含 rules.json、financial_analyzer_config.json）

运行打包后的 EXE：
- 双击运行会启动 Web（默认监听 `127.0.0.1`）
- 如端口被占用，会自动顺延选择可用端口，并在控制台打印提示
- 运行中产生的输出默认在 `.exe` 同级目录的 `output/` 下（以及 `data/` 下的 sqlite 缓存）

## 常见问题
- 端口被占用：启动时改端口，例如 `--port 9000`
- 缺少依赖：按上面的“安装依赖”重新安装（建议使用虚拟环境）
- rules.json/界面出现中文乱码：通常是外部工具以错误编码保存 JSON。本项目会在读取/保存 rules 时自动尝试修复；也建议用 UTF-8 保存
- 终端出现 `GET /@vite/client 404`：不是本项目引用 Vite，通常是浏览器缓存/插件探测导致，可忽略
