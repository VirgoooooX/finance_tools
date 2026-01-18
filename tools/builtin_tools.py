from financial_analyzer_core import ToolSpec, register_tool
try:
    from tools.report_ingestor.core import run_analysis
    register_tool(ToolSpec(id="report_ingestor", name="报表清洗与落库工具", run=run_analysis))
except ImportError as e:
    print(f"Failed to register builtin tool report_ingestor: {e}")

try:
    from tools.validation_report.core import run_analysis as run_validation_report
    register_tool(ToolSpec(id="validation_report", name="生成验证报告", run=run_validation_report))
except ImportError as e:
    print(f"Failed to register builtin tool validation_report: {e}")

try:
    from tools.financial_metrics.core import run_analysis as run_financial_metrics
    register_tool(ToolSpec(id="financial_metrics", name="生成财务指标", run=run_financial_metrics))
except ImportError as e:
    print(f"Failed to register builtin tool financial_metrics: {e}")
