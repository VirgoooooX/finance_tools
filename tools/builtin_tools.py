from financial_analyzer_core import ToolSpec, register_tool
try:
    from tools.monthly_report_cleaner.core import run_analysis
    register_tool(ToolSpec(id="monthly_report_cleaner", name="月度报表清洗工具", run=run_analysis))
except ImportError as e:
    # 打印错误但不崩溃，避免 Web 服务起不来
    print(f"Failed to register builtin tool monthly_report_cleaner: {e}")

try:
    import tools.audit_report_cleaner.core as audit_core
    register_tool(ToolSpec(
        id="audit_report_cleaner",
        name="审定报表清洗工具",
        run=audit_core.run_analysis
    ))
except ImportError as e:
    print(f"Warning: Failed to import audit_report_cleaner: {e}")
