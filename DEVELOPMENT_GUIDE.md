# Financial Data Analysis Platform Development Guide

This guide details the project structure, shared resources, and best practices for developing new tools within the platform.

## 1. Project Architecture Overview

The project is a **Python + Web** hybrid application.
- **Backend**: Python (FastAPI) handles core logic, file processing (Pandas/SQLite), and tool management.
- **Frontend**: A unified Web Shell (`web/index.html`) that loads individual tools into an `iframe`.
- **Modularity**: Capabilities are extended via "Tools" plugins located in the `tools/` directory.

### Directory Structure
```text
root/
├── fa_platform/             # [Shared] Core Platform Utilities
│   ├── paths.py             # Path handling & resource resolution
│   ├── webx.py              # Web server helpers & SSE
│   └── jsonx.py             # JSON sanitization (handles NaN/Date)
├── tools/                   # [Modules] Tool Plugins
│   ├── audit_report_cleaner/# Example Tool
│   │   ├── web/             # Tool-specific Frontend assets
│   │   ├── core.py          # Tool logic
│   │   ├── config.json      # Tool configuration
│   │   └── __init__.py      # Registration entry point
├── web/                     # [Shared] Global Web Assets
│   ├── styles.css           # Global CSS variables & styles
│   └── index.html           # Main Application Shell
├── financial_analyzer_core.py # App Core (Config, Registry)
├── financial_analyzer_web.py  # Web Server (FastAPI)
└── financial_analyzer.py      # Entry Point
```

---

## 2. Shared Resources (Crucial for Development)

### 2.1 Shared CSS (`web/styles.css`)
All tools should utilize the shared CSS variables to maintain UI consistency.
**Key Variables:**
- **Colors**: `--primary` (Dark Blue), `--accent` (Blue), `--success` (Green), `--danger` (Red), `--bg-body`, `--bg-surface`.
- **Layout**: `--radius-sm`, `--radius-md`, `--shadow-sm`.
- **Fonts**: `--font-sans` (Inter/System UI), `--font-mono` (JetBrains Mono).

**Common Classes:**
- `.app-container`: Standard centered container.
- `.card`, `.card-header`, `.card-body`: Standard card styling.
- `.btn`, `.btn-primary`, `.btn-accent`: Button styles.
- `.form-control`, `.form-select`: Input styles.

### 2.2 Shared Python Utilities (`fa_platform`)
Always use these helpers instead of standard library functions where applicable.

1.  **Path Resolution (`fa_platform.paths`)**:
    *   `get_base_dir()`: Returns the application root (handles frozen exe vs script).
    *   `get_resource_path(relative_path)`: Resolves path to a resource (e.g., `web/index.html`).
    *   `ensure_dir(path)`: Safely creates directories.

2.  **Web Helpers (`fa_platform.webx`)**:
    *   `sanitize_json(obj)`: **MUST USE** when returning JSON. Handles `NaN`, `Infinity`, `datetime`, and Pandas types that standard `json.dumps` fails on.
    *   `sse_encode(event, data)`: For Server-Sent Events (Log streaming).

3.  **Core Config (`financial_analyzer_core.py`)**:
    *   `AppConfig`: Base configuration dataclass.
    *   `AnalysisResult`: Standard return type for tool execution.

---

## 3. Developing a New Tool

To add a new tool (e.g., `my_new_tool`), follow this structure:

### Step 1: Create Directory
Create `tools/my_new_tool/`.

### Step 2: Configuration (`config.json`)
Define default settings.
```json
{
  "tool_id": "my_new_tool",
  "input_dir": "",
  "output_dir": "output"
}
```

### Step 3: Tool Logic (`core.py`)
Implement the run function matching the `ToolRunFn` signature.
```python
from financial_analyzer_core import AppConfig, AnalysisResult

def run(config: AppConfig, logger=None, progress_cb=None, cancel_event=None) -> AnalysisResult:
    # Your processing logic here
    if cancel_event and cancel_event.is_set():
        return AnalysisResult(cancelled=True)
        
    logger.info("Starting task...")
    # ... logic ...
    return AnalysisResult(cleaned_rows=100)
```

### Step 4: Registration (`__init__.py`)
Register the tool with the core system.
```python
from financial_analyzer_core import ToolSpec, register_tool
from .core import run

# Define tool metadata
TOOL_SPEC = ToolSpec(
    id="my_new_tool",
    name="My New Tool",
    run=run
)

# Expose for discovery
def register():
    register_tool(TOOL_SPEC)

# Auto-register on import
register()
```

### Step 5: Web Interface (`web/`)
This is the core of user interaction.
1.  **`manifest.json`**: Define tool metadata and **Tabs**.
    ```json
    {
      "entry": "index.html",
      "title": "My New Tool",
      "icon": "bi-calculator",
      "tabs": [
        {"id": "config", "name": "Config", "icon": "bi-gear"},
        {"id": "results", "name": "Results", "icon": "bi-file-earmark"}
      ]
    }
    ```
2.  **`index.html`**: The UI of the tool, running inside an iframe.

---

## 4. Web Development Special Notes

### 4.1 Global Architecture & Iframe
The tool's UI runs in an isolated iframe but is managed by the global Web Shell (`web/index.html`).
*   **Host**: Responsible for the top bar, tool switching, **Log Panel**, and **Start/Stop Buttons**.
*   **Tool**: Responsible for config forms, data filters, and result previews.

### 4.2 Default Features (No Dev Needed)
You do not need to implement these in your tool's HTML; they are provided by the platform:
*   **Log Panel**: There is a global "Logs" tab. Content logged via Python `logger.info(...)` is streamed here automatically.
*   **Start/Stop Buttons**: Located in the global top bar. Clicking "Start" calls your tool's `run()` function.

### 4.3 Tab Navigation & Routing
Tabs defined in `manifest.json` appear in the global top bar.
When a user clicks a tab, the host updates the iframe's URL Hash.
**The Tool MUST respond to Hash changes**:
```javascript
function handleHash() {
  const hash = window.location.hash; // e.g., "#tab=config"
  const tab = hash.replace('#tab=', '') || 'config';
  
  // Hide all views
  document.querySelectorAll('.view-pane').forEach(el => el.classList.add('d-none'));
  // Show current view
  const active = document.getElementById('view-' + tab);
  if (active) active.classList.remove('d-none');
}

window.addEventListener('hashchange', handleHash);
handleHash(); // Init
```

### 4.4 Frontend-Backend Communication
*   **API Calls**: The tool is same-origin with the backend. Use relative paths.
    *   Get Config: `fetch('/api/config?tool_id=my_new_tool')`
    *   Save Config: `fetch('/api/config', { method: 'POST', body: ... })`
*   **Event Listening**: The host forwards backend events (progress, status) via `postMessage`.
    ```javascript
    window.addEventListener('message', (event) => {
      const { type, data } = event.data; // type: 'fa.event'
      if (data.event === 'progress') {
        console.log('Progress:', data.data);
      }
      if (data.event === 'done') {
        // Task done, maybe refresh results
      }
    });
    ```

---

## 5. Web API Reference

The host provides APIs that tools can use:

*   **Config**: `GET /api/config?tool_id=...`, `POST /api/config`
*   **Execution**: `POST /api/run/start`, `POST /api/run/stop`
*   **Status**: `GET /api/run/status` (Returns progress and logs)
*   **Assets**: Tool assets are served at `/tools/{tool_id}/web/...`.

---

## 6. Data Output & Storage Standards

### 6.1 Output Directory Structure
Tools should avoid overwriting user's previous results. The recommended structure is:

```text
output/
└── {tool_id}/               # Tool-specific directory
    └── {YYYYMMDD_HHMMSS}/   # Timestamp directory for each run
        ├── result.xlsx      # Result file
        └── report.html      # (Optional) Report file
```

**Implementation Suggestion**:
Generate timestamped paths in `core.py`:
```python
import os
import datetime

def get_output_path(config):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tool_dir = os.path.join(config.output_dir, config.tool_id, timestamp)
    os.makedirs(tool_dir, exist_ok=True)
    return os.path.join(tool_dir, config.output_basename)
```

### 6.2 SQLite Intermediate Storage
If the tool generates large structured data, it is recommended to store cleaned data in SQLite for fast frontend querying.

*   **Location**: `data/` directory (via `fa_platform.paths.default_data_root()`).
*   **Naming Convention**: `{basename}_{timestamp}.sqlite`.
*   **Linkage**: In the `AnalysisResult` return object, set the `cleaned_sqlite_path` field.

---

## 7. Logging & Debugging

### 7.1 Logging
Do NOT use `print()`. The platform integrates a Web Log Streaming system.
Please use the `logger` object passed to the `run()` function:

```python
def run(config, logger=None, ...):
    if logger:
        logger.info("Processing file A...")
        logger.warning("Anomaly detected")
```
These logs will be pushed to the "Logs" panel in the Web UI in real-time.

### 7.2 Progress Feedback
Use the `progress_cb` callback to update the frontend progress bar:

```python
if progress_cb:
    progress_cb("Cleaning", current=10, total=100, detail="Row 10...")
```

---

## 8. Build & Distribution

The project uses `PyInstaller` for packaging, with configuration in `build_exe.py`.

### 8.1 Adding New Dependencies
If your tool introduces a new Python library (e.g., `openpyxl`, `numpy`):
1.  **Dev Environment**: Ensure it is installed (`pip install ...`).
2.  **Packaging**: `PyInstaller` usually detects dependencies automatically. However, since tools are loaded dynamically, some libraries might be missed.
    *   **Fix**: If you get `ModuleNotFoundError` in the built exe, add the library name to `hiddenimports` in `build_exe.py`, or ensure it is statically imported in `financial_analyzer.py` or `core.py`.

### 8.2 Static Assets
`build_exe.py` is configured to bundle everything under `tools/`. Ensure your HTML/JS/CSS files are within the `web/` subdirectory of your tool, so they are automatically included in the final `.exe`.

---

## 9. Common Pitfalls & Issues

1.  **Path Issues**:
    *   *Problem*: Using `os.path.join("web", "index.html")` works in dev but fails in the built `.exe`.
    *   *Solution*: Always use `fa_platform.paths.get_resource_path()`.

2.  **JSON Serialization**:
    *   *Problem*: `TypeError: Object of type int64 is not JSON serializable` (common with Pandas).
    *   *Solution*: Always wrap responses in `fa_platform.jsonx.sanitize_json()`.

3.  **Blocking the Server**:
    *   *Problem*: Long-running tasks in the main thread will freeze the Web UI.
    *   *Solution*: The `run()` function is executed in a worker thread by `financial_analyzer_web.py`. Ensure your code respects `cancel_event` to allow graceful stopping.

4.  **Style Isolation**:
    *   *Note*: Tools are in iframes, so they don't inherit global styles automatically. You must explicitly `<link rel="stylesheet" href="/static/styles.css">` in your tool's `index.html`.
