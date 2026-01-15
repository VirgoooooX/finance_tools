# 财务数据分析平台开发指南

本指南详细介绍了项目结构、共享资源以及在平台内开发新工具的最佳实践。

## 1. 项目架构概览

本项目是一个 **Python + Web** 的混合应用。
- **后端 (Backend)**: Python (FastAPI) 负责核心逻辑、文件处理 (Pandas/SQLite) 和工具管理。
- **前端 (Frontend)**: 一个统一的 Web Shell (`web/index.html`)，通过 `iframe` 加载各个独立的工具。
- **模块化 (Modularity)**: 通过 `tools/` 目录下的“工具 (Tools)”插件来扩展功能。

### 目录结构
```text
root/
├── fa_platform/             # [共享] 核心平台工具库
│   ├── paths.py             # 路径处理与资源定位
│   ├── webx.py              # Web 服务器助手与 SSE
│   └── jsonx.py             # JSON 净化 (处理 NaN/Date 等)
├── tools/                   # [模块] 工具插件目录
│   ├── audit_report_cleaner/# 示例工具
│   │   ├── web/             # 工具专属的前端资源
│   │   ├── core.py          # 工具核心逻辑
│   │   ├── config.json      # 工具配置
│   │   └── __init__.py      # 注册入口
├── web/                     # [共享] 全局 Web 资源
│   ├── styles.css           # 全局 CSS 变量与样式
│   └── index.html           # 主应用 Shell
├── financial_analyzer_core.py # 应用核心 (配置, 注册表)
├── financial_analyzer_web.py  # Web 服务器 (FastAPI)
└── financial_analyzer.py      # 启动入口
```

---

## 2. 共享资源 (开发必读)

### 2.1 共享 CSS (`web/styles.css`)
所有工具都应使用共享的 CSS 变量以保持 UI 一致性。
**关键变量:**
- **颜色**: `--primary` (深蓝), `--accent` (蓝色), `--success` (绿色), `--danger` (红色), `--bg-body`, `--bg-surface`.
- **布局**: `--radius-sm`, `--radius-md`, `--shadow-sm`.
- **字体**: `--font-sans` (Inter/System UI), `--font-mono` (JetBrains Mono).

**常用类:**
- `.app-container`: 标准居中容器。
- `.card`, `.card-header`, `.card-body`: 标准卡片样式。
- `.btn`, `.btn-primary`, `.btn-accent`: 按钮样式。
- `.form-control`, `.form-select`: 输入框样式。

### 2.2 共享 Python 工具库 (`fa_platform`)
在可能的情况下，务必使用这些助手函数，而不是标准库函数。

1.  **路径解析 (`fa_platform.paths`)**:
    *   `get_base_dir()`: 返回应用根目录 (处理打包后的 exe 与脚本环境的差异)。
    *   `get_resource_path(relative_path)`: 解析资源路径 (例如 `web/index.html`)。
    *   `ensure_dir(path)`: 安全创建目录。

2.  **Web 助手 (`fa_platform.webx`)**:
    *   `sanitize_json(obj)`: **必须使用**。在返回 JSON 前调用，用于处理 `NaN`, `Infinity`, `datetime` 以及 Pandas 类型，防止 `json.dumps` 报错。
    *   `sse_encode(event, data)`: 用于 SSE (Server-Sent Events) 日志流推送。

3.  **核心配置 (`financial_analyzer_core.py`)**:
    *   `AppConfig`: 基础配置数据类 (Dataclass)。
    *   `AnalysisResult`: 工具执行的标准返回类型。

---

## 3. 开发新工具

要添加一个新工具 (例如 `my_new_tool`)，请遵循以下步骤：

### 步骤 1: 创建目录
创建目录 `tools/my_new_tool/`。

### 步骤 2: 配置文件 (`config.json`)
定义默认设置。
```json
{
  "tool_id": "my_new_tool",
  "input_dir": "",
  "output_dir": "output"
}
```

### 步骤 3: 工具逻辑 (`core.py`)
实现匹配 `ToolRunFn` 签名的运行函数。
```python
from financial_analyzer_core import AppConfig, AnalysisResult

def run(config: AppConfig, logger=None, progress_cb=None, cancel_event=None) -> AnalysisResult:
    # 你的处理逻辑
    if cancel_event and cancel_event.is_set():
        return AnalysisResult(cancelled=True)
        
    logger.info("Starting task...")
    # ... 业务逻辑 ...
    return AnalysisResult(cleaned_rows=100)
```

### 步骤 4: 注册工具 (`__init__.py`)
向核心系统注册该工具。
```python
from financial_analyzer_core import ToolSpec, register_tool
from .core import run

# 定义工具元数据
TOOL_SPEC = ToolSpec(
    id="my_new_tool",
    name="My New Tool",
    run=run
)

# 暴露给发现机制
def register():
    register_tool(TOOL_SPEC)

# 导入时自动注册
register()
```

### 步骤 5: Web 界面 (`web/`)
这是用户交互的核心。
1.  **`manifest.json`**: 定义工具的基本信息和**Tabs**。
    ```json
    {
      "entry": "index.html",
      "title": "我的新工具",
      "icon": "bi-calculator",
      "tabs": [
        {"id": "config", "name": "配置", "icon": "bi-gear"},
        {"id": "results", "name": "结果", "icon": "bi-file-earmark"}
      ]
    }
    ```
2.  **`index.html`**: 工具的 UI 界面，运行在 iframe 中。

---

## 4. Web 开发特别注意事项

### 4.1 全局架构与 Iframe
工具的 UI 运行在独立的 iframe 中，但通过全局 Web Shell (`web/index.html`) 进行管理。
*   **宿主 (Host)**: 负责显示顶栏、工具切换、**日志面板**和**开始/停止按钮**。
*   **工具 (Tool)**: 负责显示配置表单、数据筛选器和结果预览。

### 4.2 默认功能 (无需开发)
你不需要在工具的 HTML 中实现以下功能，它们由平台提供：
*   **日志面板**: 全局有一个“日志” Tab。你在 Python `logger.info(...)` 输出的内容会自动流式传输到这里。
*   **开始/停止按钮**: 位于全局顶栏。点击“开始”会自动调用你工具的 `run()` 函数。

### 4.3 Tab 导航与路由
在 `manifest.json` 中定义的 `tabs` 会显示在全局顶栏。
当用户点击 Tab 时，宿主会更新 iframe 的 URL Hash。
**工具必须响应 Hash 变更**:
```javascript
function handleHash() {
  const hash = window.location.hash; // e.g., "#tab=config"
  const tab = hash.replace('#tab=', '') || 'config';
  
  // 隐藏所有视图
  document.querySelectorAll('.view-pane').forEach(el => el.classList.add('d-none'));
  // 显示当前视图
  const active = document.getElementById('view-' + tab);
  if (active) active.classList.remove('d-none');
}

window.addEventListener('hashchange', handleHash);
handleHash(); // 初始化
```

### 4.4 前后端通信
*   **API 调用**: 工具与后端同源，直接使用相对路径调用 API。
    *   获取配置: `fetch('/api/config?tool_id=my_new_tool')`
    *   保存配置: `fetch('/api/config', { method: 'POST', body: ... })`
*   **事件监听**: 宿主通过 `postMessage` 转发后端事件 (如进度、完成状态)。
    ```javascript
    window.addEventListener('message', (event) => {
      const { type, data } = event.data; // type: 'fa.event'
      if (data.event === 'progress') {
        console.log('进度:', data.data);
      }
      if (data.event === 'done') {
        // 任务完成，自动刷新结果视图
      }
    });
    ```

---

## 5. Web API 参考

宿主环境提供了工具可以使用的 API：

*   **配置 (Config)**: `GET /api/config?tool_id=...`, `POST /api/config`
*   **执行 (Execution)**: `POST /api/run/start`, `POST /api/run/stop`
*   **状态 (Status)**: `GET /api/run/status` (返回进度和日志)
*   **资源 (Assets)**: 工具资源通过 `/tools/{tool_id}/web/...` 提供服务。

---

## 6. 数据输出与存储规范

### 6.1 输出目录结构
工具在运行时应避免直接覆盖用户之前的结果。推荐的输出目录结构如下：

```text
output/
└── {tool_id}/               # 工具专属目录
    └── {YYYYMMDD_HHMMSS}/   # 每次运行的时间戳目录
        ├── result.xlsx      # 结果文件
        └── report.html      # (可选) 报告文件
```

**实现建议**:
在 `core.py` 中生成带有时间戳的输出路径：
```python
import os
import datetime

def get_output_path(config):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tool_dir = os.path.join(config.output_dir, config.tool_id, timestamp)
    os.makedirs(tool_dir, exist_ok=True)
    return os.path.join(tool_dir, config.output_basename)
```

### 6.2 SQLite 中间存储
如果工具产生大量结构化数据，建议将清洗后的数据存入 SQLite 数据库，以便前端进行快速筛选和查询。

*   **存储位置**: `data/` 目录 (通过 `fa_platform.paths.default_data_root()`).
*   **命名规范**: `{basename}_{timestamp}.sqlite`.
*   **关联方式**: 在 `AnalysisResult` 返回对象中，设置 `cleaned_sqlite_path` 字段。

---

## 7. 日志与调试

### 7.1 日志记录
不要使用 `print()`。平台集成了 Web 日志流系统。
请使用传递给 `run()` 函数的 `logger` 对象：

```python
def run(config, logger=None, ...):
    if logger:
        logger.info("正在处理文件 A...")
        logger.warning("发现异常数据")
```
这些日志会实时推送到 Web 界面的“日志”面板中。

### 7.2 进度反馈
使用 `progress_cb` 回调函数更新前端进度条：

```python
if progress_cb:
    progress_cb("正在清洗", current=10, total=100, detail="处理 Row 10...")
```

---

## 8. 打包与发布

项目使用 `PyInstaller` 进行打包，配置文件为 `build_exe.py`。

### 8.1 添加新依赖
如果你的工具引入了新的 Python 库（例如 `openpyxl`, `numpy`）：
1.  **开发环境**: 确保在当前环境中已安装 (`pip install ...`)。
2.  **打包**: `PyInstaller` 通常能自动分析依赖。但由于工具是动态加载的，某些库可能无法被检测到。
    *   **解决**: 如果打包后运行报错 `ModuleNotFoundError`，请在 `build_exe.py` 的 `hiddenimports` 参数中添加该库名（如果脚本中支持），或者确保该库在 `financial_analyzer.py` 或 `core.py` 静态导入链中可达。

### 8.2 资源文件
`build_exe.py` 默认会打包 `tools/` 目录下的所有内容。确保你的 HTML/JS/CSS 文件都在工具的 `web/` 子目录下，这样它们会被自动包含在最终的 `.exe` 文件中。

---

## 9. 常见问题与陷阱

1.  **路径问题**:
    *   *问题*: 使用 `os.path.join("web", "index.html")` 在开发环境中正常，但在打包后的 `.exe` 中失败。
    *   *解决*: 始终使用 `fa_platform.paths.get_resource_path()`。

2.  **JSON 序列化**:
    *   *问题*: `TypeError: Object of type int64 is not JSON serializable` (Pandas 常见问题)。
    *   *解决*: 始终使用 `fa_platform.jsonx.sanitize_json()` 包装响应数据。

3.  **阻塞服务器**:
    *   *问题*: 在主线程中运行耗时任务会冻结 Web UI。
    *   *解决*: `run()` 函数由 `financial_analyzer_web.py` 在工作线程中执行。确保你的代码检查 `cancel_event` 以支持优雅停止。

4.  **样式隔离**:
    *   *注意*: 工具位于 iframe 中，不会自动继承全局样式。必须在工具的 `index.html` 中显式引入 `<link rel="stylesheet" href="/static/styles.css">`。
