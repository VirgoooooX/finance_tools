import argparse
from typing import Optional, List

from financial_analyzer_core import DEFAULT_CONFIG_PATH, load_config, _get_logger, analyze_directory


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="财务数据清洗、验证与指标计算（CLI）")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="配置文件路径(JSON)")
    parser.add_argument("--input-dir", type=str, default=None, help="输入目录")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录")
    parser.add_argument("--glob", type=str, default=None, help="文件匹配模式，如 *.xlsx")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.input_dir:
        cfg.input_dir = args.input_dir
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.glob:
        cfg.file_glob = args.glob

    logger = _get_logger()
    res = analyze_directory(cfg, logger=logger)
    if res.cancelled:
        return 2
    if res.errors:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

