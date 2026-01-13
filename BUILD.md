# 财务数据分析工具 - 打包说明 (CustomTkinter 版本)

## 1. 安装依赖

确保已安装所有必要依赖：
```bash
pip install -r requirements.txt
```

## 2. 使用 PyInstaller 打包成单文件 .exe

运行以下命令将应用程序打包：

```bash
pyinstaller --name="财务数据分析工具" \
            --noconsole \
            --onefile \
            --add-data="config.yaml;." \
            gui_ctk.py
```

### 参数详解：
- `--name`：生成的可执行文件名称。
- `--noconsole` (或 `--windowed`)：运行时不显示黑色的控制台窗口。
- `--onefile`：将所有依赖打包进一个单一的 .exe 文件。
- `--add-data="config.yaml;."`：将默认配置文件打包进程序（运行时会自动提取）。

## 3. 打包产物

打包完成后，您可以在项目根目录的 `dist/` 文件夹下找到：
- `dist/财务数据分析工具.exe`

## 4. 运行注意事项

- **首次运行**：程序会自动在当前目录下生成默认的 `config.yaml`（如果不存在）。
- **配置文件**：如果您想自定义科目匹配，可以直接编辑 exe 同目录下的 `config.yaml`。
- **环境**：打包后的 exe 可以在没有安装 Python 的 Windows 电脑上独立运行。
