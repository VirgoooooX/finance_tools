# 项目状态

## 收款进度监控工具

### 当前状态

✅ **已完成核心功能**
- OCR 识别引擎 (PaddleOCR CPU 模式)
- 信息提取器 (合同、发票、回执)
- 智能匹配引擎 (三方匹配)
- 数据库操作 (SQLite)
- 进度监控器
- Web 界面 (配置、上传、查询、Dashboard)

### OCR 配置

- **引擎**: PaddleOCR (PP-OCRv5)
- **模式**: CPU
- **识别质量**: 优秀 (置信度 0.95+)
- **处理速度**: ~18 秒/文件
- **适用场景**: 小批量处理 (< 100 文件/天)

### 已实现的功能

1. **文档识别**
   - 支持 PDF 和图片格式 (JPG/PNG)
   - 自动识别文档类型 (合同/发票/回执)
   - 提取关键信息 (金额、日期、编号等)

2. **智能匹配**
   - 三方匹配 (合同-发票-回执)
   - 模糊匹配 (名称相似度)
   - 金额容差匹配
   - 部分收款支持

3. **进度监控**
   - 实时进度跟踪
   - 收款状态统计
   - 未匹配项目提醒

4. **数据管理**
   - SQLite 数据库存储
   - 文件去重 (SHA256 哈希)
   - 结果缓存
   - 历史记录查询

5. **Web 界面**
   - 配置管理
   - 文件上传
   - 结果查询
   - Dashboard 展示

### 文件结构

```
tools/payment_monitor/
├── config.json          # 配置文件
├── core.py             # 核心处理流程
├── database.py         # 数据库操作
├── extractor.py        # 信息提取器
├── matching.py         # 智能匹配引擎
├── models.py           # 数据模型
├── monitor.py          # 进度监控器
├── ocr_engine.py       # OCR 识别引擎
├── web/
│   ├── index.html      # Web 界面
│   └── manifest.json   # 工具清单
├── FEATURES.md         # 功能说明
├── README.md           # 使用说明
└── USAGE.md            # 详细用法
```

### 测试文件

- `test_ocr.py` - OCR 功能测试
- `test_samples/` - 测试样本 (发票、回执)

### 文档

- `OCR_GUIDE.md` - OCR 使用指南
- `DEVELOPMENT_GUIDE_ZH.md` - 开发指南
- `README.md` - 项目说明

### 下一步

如需进一步优化或添加功能，可以：

1. **性能优化**
   - 如果处理量大 (> 100 文件/天)，考虑购买 NVIDIA GPU
   - 预期加速 5-10x (18 秒 -> 2-4 秒/文件)

2. **功能扩展**
   - 添加更多文档类型支持
   - 优化匹配算法
   - 增强 Dashboard 功能
   - 添加导出功能

3. **集成测试**
   - 端到端测试
   - 性能测试
   - 压力测试

### 依赖

核心依赖:
```
paddleocr
pillow
pymupdf
openpyxl
fastapi
uvicorn
```

安装:
```bash
pip install paddleocr pillow pymupdf openpyxl fastapi uvicorn
```

### 运行

启动 Web 服务:
```bash
python financial_analyzer_web.py
```

访问: http://localhost:8000

### 测试

测试 OCR 功能:
```bash
python test_ocr.py
```

### 配置

配置文件: `tools/payment_monitor/config.json`

关键配置项:
- `ocr_use_gpu`: GPU 加速 (默认 false)
- `ocr_lang`: 识别语言 (默认 "ch")
- `tolerance`: 金额容差 (默认 0.01)
- `name_similarity_threshold`: 名称相似度阈值 (默认 0.8)

### 性能

当前配置 (CPU 模式):
- 处理速度: ~18 秒/文件
- 识别质量: 优秀 (0.95+)
- 内存占用: ~500MB
- 适合: 小批量处理

GPU 模式 (可选):
- 处理速度: ~2-4 秒/文件
- 识别质量: 优秀 (0.95+)
- 需要: NVIDIA GPU + CUDA
- 适合: 中大批量处理

### 已清理的内容

以下内容已清理，不再使用:
- ❌ EasyOCR 相关依赖和脚本
- ❌ AMD GPU / DirectML 相关配置
- ❌ 性能测试脚本和报告
- ❌ 临时测试文件

### 保留的内容

以下内容保留，继续使用:
- ✅ PaddleOCR CPU 模式
- ✅ 核心功能代码
- ✅ Web 界面
- ✅ 测试脚本 (test_ocr.py)
- ✅ 测试样本
- ✅ 文档

---

最后更新: 2026-03-22
