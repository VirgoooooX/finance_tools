# 打包说明

## 安装依赖

```bash
pip install -r requirements.txt
```

## 打包成exe

```bash
pyinstaller --name="财务数据分析工具" --windowed --onefile --add-data="config.yaml;." gui_main.py
```

### 参数说明
- `--name` - 输出exe的名称
- `--windowed` - 不显示控制台窗口
- `--onefile` - 打包成单个exe文件
- `--add-data` - 包含配置文件（格式：源文件;目标路径）

## 打包后的文件

打包完成后，exe文件在 `dist/` 目录下：
- `dist/财务数据分析工具.exe`

## 使用方法

1. 将 `config.yaml` 文件放在exe同目录下（首次运行会自动生成）
2. 双击运行 `财务数据分析工具.exe`
3. 选择包含Excel文件的目录
4. 配置选项
5. 点击"开始处理"

## 注意事项

- 确保config.yaml文件与exe在同一目录
- 打包后的exe可以在没有Python环境的电脑上运行
- 首次运行可能需要Windows Defender权限
