# 财务数据分析工具（Web/桌面/命令行）

本项目用于批量清洗财务报表、生成验证报告与财务指标，并提供本机 Web 界面。

## 运行环境
- Windows 10/11
- Python 3.11+

## 安装依赖
```powershell
py -m pip install -U pip
py -m pip install pandas openpyxl fastapi uvicorn
```

可选（桌面 UI 才需要）：
```powershell
py -m pip install customtkinter
```

## 启动 Web 界面
```powershell
py financial_analyzer.py --web --port 8765
```

说明：
- 直接运行 `py financial_analyzer.py`（不带参数）会默认启动 Web。
- 如不想自动打开浏览器：`py financial_analyzer.py --web --no-browser`，再手动访问 `http://127.0.0.1:8765/`。

## 启动桌面 UI（可选）
```powershell
py financial_analyzer.py --ui
```

## 命令行模式（可选）
```powershell
py financial_analyzer.py --config financial_analyzer_config.json
```

## 打包成 EXE（Web 版）
项目根目录提供了自动打包脚本 [build_exe.py](file:///l:/Web/Financial%20data%20analysis/build_exe.py)：

```powershell
py build_exe.py
```

打包结果：
- 输出文件：`dist/财务数据分析工具.exe`
- 图标文件：默认使用项目根目录下的 `app_icon.ico`
- 静态资源：会自动打包 `web/` 目录

配置文件说明：
- 运行时会读取/写入 `financial_analyzer_config.json`
- 打包后默认使用 `.exe` 同级目录下的 `financial_analyzer_config.json`（方便持久化保存配置）

## 常见问题
- 端口被占用：启动时改端口，例如 `--port 9000`
- 缺少依赖：按上面的“安装依赖”重新安装（建议使用虚拟环境）
