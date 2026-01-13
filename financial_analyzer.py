import argparse
from typing import Optional, List


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="财务数据清洗、验证与指标计算")
    parser.add_argument("--ui", action="store_true", help="启动图形界面")
    parser.add_argument("--web", action="store_true", help="启动本机Web界面")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径(JSON)")
    parser.add_argument("--input-dir", type=str, default=None, help="输入目录")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录")
    parser.add_argument("--glob", type=str, default=None, help="文件匹配模式，如 *.xlsx")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Web监听地址")
    parser.add_argument("--port", type=int, default=8765, help="Web端口")
    parser.add_argument("--no-browser", action="store_true", help="启动Web时不自动打开浏览器")
    args = parser.parse_args(argv)

    # 如果没有提供任何模式参数（--ui 或 --web 或 --config 等 CLI 参数），默认启动 web
    import sys
    is_interactive = len(sys.argv) == 1
    
    if args.web or is_interactive:
        from financial_analyzer_core import DEFAULT_CONFIG_PATH
        from financial_analyzer_web import run_web

        return run_web(
            args.config or DEFAULT_CONFIG_PATH,
            host=args.host,
            port=int(args.port),
            open_browser=not bool(args.no_browser),
        )

    if args.ui:
        from financial_analyzer_core import DEFAULT_CONFIG_PATH
        from financial_analyzer_desktop import FinancialAnalyzerUI

        app = FinancialAnalyzerUI(config_path=args.config or DEFAULT_CONFIG_PATH)
        app.run()
        return 0

    from financial_analyzer_core import DEFAULT_CONFIG_PATH, main as cli_main

    cli_argv: List[str] = ["--config", (args.config or DEFAULT_CONFIG_PATH)]
    if args.input_dir:
        cli_argv.extend(["--input-dir", args.input_dir])
    if args.output_dir:
        cli_argv.extend(["--output-dir", args.output_dir])
    if args.glob:
        cli_argv.extend(["--glob", args.glob])

    return int(cli_main(cli_argv))


if __name__ == "__main__":
    raise SystemExit(main())

