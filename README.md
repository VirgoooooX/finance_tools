# 财务数据分析工具 - 使用说明

## 项目结构

```
Financial data analysis/
├── financial_analyzer.py  # 核心数据处理逻辑
├── gui_ctk.py            # CustomTkinter 图形界面主程序 (新)
├── config_manager.py     # 配置管理模块
├── config.yaml           # 配置文件
├── requirements.txt      # Python依赖包
├── BUILD.md             # 打包说明
└── README.md            # 项目文档
```

## 快速开始

### 方式1: 命令行版本（无需GUI）

```bash
python financial_analyzer.py
```

### 方式2: GUI 图形界面版本 (推荐)

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 运行 GUI 程序：
```bash
python gui_ctk.py
```

3. 在界面中：
   - 选择文件所在目录
   - 点击“开始处理”即可完成数据清洗、验证和指标计算。

## 功能特性

### 📊 数据清洗
- 自动识别所有包含 BS/PL/CF 的工作表
- 支持多种日期格式
- 标准化输出格式

### 🔍 数据验证  
- 会计恒等式验证
- 数据质量检查

### 📈 财务指标（16项）
- 流动性指标、偿债能力指标、盈利能力指标、现金流指标

### ⚙️ 配置管理
- YAML 配置文件，支持自定义科目映射和处理逻辑。

## 文档
- 详细打包步骤请参阅 [BUILD.md](BUILD.md)

## 版本说明

### v2.1 (当前版本)
- ✅ **全新界面**：使用 CustomTkinter 重写，界面更精美、现代。
- ✅ **极速运行**：后台线程执行处理，界面无卡顿。
- ✅ **配置灵活**：支持科目映射、容差调整。

### v1.0
- ✅ 命令行基础版本

## 依赖环境

- Python 3.8+
- pandas, openpyxl, PyYAML, customtkinter

## 许可证

MIT License
