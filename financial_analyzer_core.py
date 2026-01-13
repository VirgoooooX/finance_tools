import pandas as pd
import datetime
import re
import glob
import os
import json
import logging
import threading
from dataclasses import dataclass, asdict, field
from typing import Callable, Optional, Any, Dict, List, Tuple


OUTPUT_PATH = "清洗后的AI标准财务表.xlsx"

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "financial_analyzer_config.json")


@dataclass
class AppConfig:
    input_dir: str = field(default_factory=lambda: os.getcwd())
    file_glob: str = "*.xlsx"
    output_dir: str = field(default_factory=lambda: os.getcwd())
    output_basename: str = OUTPUT_PATH
    generate_validation: bool = True
    generate_metrics: bool = True
    exclude_output_files: bool = True
    sheet_keyword_bs: str = "BS"
    sheet_keyword_pl: str = "PL"
    sheet_keyword_cf: str = "CF"
    header_keyword_bs: str = "期末余额"
    header_keyword_pl: str = "本年累计"
    header_keyword_cf: str = "本期金额"
    date_cells_bs: List[List[int]] = field(default_factory=lambda: [[2, 3], [2, 2]])
    date_cells_pl: List[List[int]] = field(default_factory=lambda: [[2, 2], [2, 1]])
    date_cells_cf: List[List[int]] = field(default_factory=lambda: [[2, 4], [2, 0]])
    validation_tolerance: float = 0.01


@dataclass
class AnalysisResult:
    cancelled: bool = False
    errors: List[str] = field(default_factory=list)
    found_files: List[str] = field(default_factory=list)
    processed_files: int = 0
    cleaned_rows: int = 0
    metrics_groups: int = 0
    validation_groups: int = 0
    unbalanced_count: int = 0
    cleaned_path: Optional[str] = None
    validation_path: Optional[str] = None
    metrics_path: Optional[str] = None
    unbalanced_preview: List[Dict[str, Any]] = field(default_factory=list)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_int_pair(pair: Any) -> Optional[Tuple[int, int]]:
    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
        return None
    try:
        return int(pair[0]), int(pair[1])
    except Exception:
        return None


def load_config(path: str) -> AppConfig:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cfg = AppConfig()
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg
    except FileNotFoundError:
        return AppConfig()
    except Exception:
        return AppConfig()


def save_config(path: str, cfg: AppConfig) -> None:
    _ensure_dir(os.path.dirname(path) or os.getcwd())
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)


def _get_logger(name: str = "financial_analyzer", handler: Optional[logging.Handler] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
    if handler is None:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
    logger.addHandler(handler)
    return logger


def _is_date_like(date_val: Any) -> bool:
    if date_val is None:
        return False
    try:
        if pd.isna(date_val):
            return False
    except Exception:
        pass

    if isinstance(date_val, (pd.Timestamp, datetime.datetime, datetime.date)):
        return True

    if isinstance(date_val, (int, float)) and not isinstance(date_val, bool):
        try:
            n = float(date_val)
        except Exception:
            return False
        if pd.isna(n):
            return False
        if not (20000 <= n <= 80000):
            return False
        try:
            dt = datetime.datetime(1899, 12, 30) + datetime.timedelta(days=int(n))
            return 1990 <= dt.year <= 2100
        except Exception:
            return False

    text = str(date_val).strip()
    if not text:
        return False

    digits = re.findall(r"\d+", text)
    if len(digits) >= 2 and len(digits[0]) == 4:
        try:
            month = int(digits[1])
            if 1 <= month <= 12:
                if len(digits) >= 3:
                    day = int(digits[2])
                    if 1 <= day <= 31:
                        return True
                    return False
                return True
        except Exception:
            pass

    try:
        ts = pd.to_datetime(text, errors="coerce", infer_datetime_format=True)
        return not pd.isna(ts)
    except Exception:
        return False


def _read_date_nearby(df: pd.DataFrame, r: int, c: int, max_radius: int = 10) -> Any:
    rows, cols = int(df.shape[0]), int(df.shape[1])

    def in_bounds(rr: int, cc: int) -> bool:
        return 0 <= rr < rows and 0 <= cc < cols

    if in_bounds(r, c):
        val0 = df.iat[r, c]
        if _is_date_like(val0):
            return val0

    for radius in range(1, max_radius + 1):
        for dr in range(-radius, radius + 1):
            rr_top = r + dr
            cc_left = c - radius
            cc_right = c + radius
            if in_bounds(rr_top, cc_left):
                val = df.iat[rr_top, cc_left]
                if _is_date_like(val):
                    return val
            if in_bounds(rr_top, cc_right):
                val = df.iat[rr_top, cc_right]
                if _is_date_like(val):
                    return val

        for dc in range(-radius + 1, radius):
            cc = c + dc
            rr_top = r - radius
            rr_bottom = r + radius
            if in_bounds(rr_top, cc):
                val = df.iat[rr_top, cc]
                if _is_date_like(val):
                    return val
            if in_bounds(rr_bottom, cc):
                val = df.iat[rr_bottom, cc]
                if _is_date_like(val):
                    return val

    return None


def _read_date_from_cells(df: pd.DataFrame, cells: List[List[int]]) -> Any:
    for cell in cells:
        rc = _safe_int_pair(cell)
        if rc is None:
            continue
        r, c = rc
        if 0 <= r < df.shape[0] and 0 <= c < df.shape[1]:
            val = _read_date_nearby(df, r, c)
            if val is not None:
                return val
    return None


def _find_header_row(df: pd.DataFrame, keyword: str) -> Optional[int]:
    try:
        matches = df[df.apply(lambda x: x.astype(str).str.contains(keyword).any(), axis=1)].index
        if len(matches) == 0:
            return None
        return int(matches[0])
    except Exception:
        return None


def clean_date_str(date_val):
    """
    清洗日期：支持 Excel数字、'2025年11月'、'2025-11-30' 等格式
    """
    if pd.isna(date_val) or date_val == "":
        return "未知日期"

    if isinstance(date_val, (int, float)):
        try:
            return (datetime.datetime(1899, 12, 30) + datetime.timedelta(days=int(date_val))).strftime("%Y-%m-%d")
        except Exception:
            return str(date_val)

    text = str(date_val)
    digits = re.findall(r"\d+", text)
    if len(digits) >= 2:
        year = digits[0]
        month = digits[1].zfill(2)
        day = digits[2].zfill(2) if len(digits) > 2 else "01"
        return f"{year}-{month}-{day}"

    return text.split(" ")[0]


def clean_bs(file_path, sheet_name, cfg: AppConfig, logger: Optional[logging.Logger] = None):
    if logger:
        logger.info(f"正在处理: {sheet_name} ...")
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        date_val = _read_date_from_cells(df, cfg.date_cells_bs)
        report_date = clean_date_str(date_val)
        header_row = _find_header_row(df, cfg.header_keyword_bs)
        if header_row is None:
            raise ValueError(f"未找到表头关键字: {cfg.header_keyword_bs}")
        df_left = df.iloc[header_row + 1 :, [0, 1, 2]].copy()
        df_left.columns = ["科目", "年初余额", "期末余额"]
        df_left["大类"] = "资产"
        df_parts = [df_left]
        if df.shape[1] >= 6:
            df_right = df.iloc[header_row + 1 :, [3, 4, 5]].copy()
            df_right.columns = ["科目", "年初余额", "期末余额"]
            df_right["大类"] = "负债及权益"
            df_parts.append(df_right)
        df_clean = pd.concat(df_parts, ignore_index=True)
        df_clean = df_clean.dropna(subset=["科目"])
        df_clean = df_clean[df_clean["科目"].astype(str).str.strip() != ""]
        df_final = df_clean.melt(
            id_vars=["大类", "科目"], value_vars=["年初余额", "期末余额"], var_name="时间属性", value_name="金额"
        )
        df_final["报表类型"] = "资产负债表"
        df_final["日期"] = report_date
        df_final["来源Sheet"] = sheet_name
        return df_final
    except Exception as e:
        if logger:
            logger.error(f"{sheet_name} 处理失败: {e}")
        return pd.DataFrame()


def clean_pl(file_path, sheet_name, cfg: AppConfig, logger: Optional[logging.Logger] = None):
    if logger:
        logger.info(f"正在处理: {sheet_name} ...")
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        date_val = _read_date_from_cells(df, cfg.date_cells_pl)
        report_date = clean_date_str(date_val)
        header_row = _find_header_row(df, cfg.header_keyword_pl)
        if header_row is None:
            raise ValueError(f"未找到表头关键字: {cfg.header_keyword_pl}")
        df_clean = df.iloc[header_row + 1 :, [0, 2, 3]].copy()
        df_clean.columns = ["科目", "本期金额", "本年累计金额"]
        df_clean = df_clean.dropna(subset=["科目"])
        df_final = df_clean.melt(
            id_vars=["科目"], value_vars=["本期金额", "本年累计金额"], var_name="时间属性", value_name="金额"
        )
        df_final["大类"] = "损益"
        df_final["报表类型"] = "利润表"
        df_final["日期"] = report_date
        df_final["来源Sheet"] = sheet_name
        return df_final
    except Exception as e:
        if logger:
            logger.error(f"{sheet_name} 处理失败: {e}")
        return pd.DataFrame()


def clean_cf(file_path, sheet_name, cfg: AppConfig, logger: Optional[logging.Logger] = None):
    if logger:
        logger.info(f"正在处理: {sheet_name} ...")
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        date_val = _read_date_from_cells(df, cfg.date_cells_cf)
        report_date = clean_date_str(date_val)
        header_row = _find_header_row(df, cfg.header_keyword_cf)
        if header_row is None:
            raise ValueError(f"未找到表头关键字: {cfg.header_keyword_cf}")
        df_left = df.iloc[header_row + 1 :, [0, 2, 3]].copy()
        df_left.columns = ["科目", "本期金额", "本年累计金额"]
        if df.shape[1] >= 8:
            df_right = df.iloc[header_row + 1 :, [4, 6, 7]].copy()
            df_right.columns = ["科目", "本期金额", "本年累计金额"]
            df_combined = pd.concat([df_left, df_right], ignore_index=True)
        else:
            df_combined = df_left
        df_combined = df_combined.dropna(subset=["科目"])
        df_combined = df_combined[df_combined["科目"].astype(str).str.strip() != ""]
        df_final = df_combined.melt(
            id_vars=["科目"], value_vars=["本期金额", "本年累计金额"], var_name="时间属性", value_name="金额"
        )
        df_final["大类"] = "现金流"
        df_final["报表类型"] = "现金流量表"
        df_final["日期"] = report_date
        df_final["来源Sheet"] = sheet_name
        return df_final
    except Exception as e:
        if logger:
            logger.error(f"{sheet_name} 处理失败: {e}")
        return pd.DataFrame()


def extract_amount(df, keywords, sheet_type=None, time_attr=None, category=None):
    filtered_df = df.copy()
    if sheet_type:
        filtered_df = filtered_df[filtered_df["报表类型"] == sheet_type]
    if time_attr:
        filtered_df = filtered_df[filtered_df["时间属性"] == time_attr]
    if category:
        filtered_df = filtered_df[filtered_df["大类"] == category]
    for keyword in keywords:
        matched = filtered_df[filtered_df["科目"].str.strip().str.lower() == keyword.lower()]
        if not matched.empty:
            return matched.iloc[0]["金额"]
    return 0


def validate_balance_sheet(
    df_group,
    tolerance: float = 0.01,
    assets_keywords: Optional[List[str]] = None,
    liabilities_keywords: Optional[List[str]] = None,
    equity_keywords: Optional[List[str]] = None,
):
    assets = extract_amount(
        df_group,
        assets_keywords or ["资产总计", "资产总额", "资产合计"],
        sheet_type="资产负债表",
        category="资产",
    )
    liabilities = extract_amount(
        df_group,
        liabilities_keywords or ["负债合计", "负债总计", "负债总额"],
        sheet_type="资产负债表",
        category="负债及权益",
    )
    equity = extract_amount(
        df_group,
        equity_keywords or ["所有者权益合计", "股东权益合计", "所有者权益总计", "权益合计"],
        sheet_type="资产负债表",
        category="负债及权益",
    )
    diff = abs(assets - (liabilities + equity))
    is_balanced = diff <= tolerance
    return {
        "资产总计": assets,
        "负债合计": liabilities,
        "所有者权益合计": equity,
        "差额": diff,
        "是否平衡": "是" if is_balanced else "否",
        "验证结果": "通过" if is_balanced else f"不平衡(差额:{diff:.2f})",
    }


def calculate_financial_metrics(df_group):
    metrics = {}
    assets_total = extract_amount(df_group, ["资产总计", "资产总额"], sheet_type="资产负债表")
    current_assets = extract_amount(df_group, ["流动资产合计", "流动资产总计"], sheet_type="资产负债表")
    cash = extract_amount(df_group, ["货币资金", "现金及现金等价物"], sheet_type="资产负债表")
    inventory = extract_amount(df_group, ["存货"], sheet_type="资产负债表")
    liabilities_total = extract_amount(df_group, ["负债合计", "负债总计"], sheet_type="资产负债表")
    current_liabilities = extract_amount(df_group, ["流动负债合计", "流动负债总计"], sheet_type="资产负债表")
    equity_total = extract_amount(df_group, ["所有者权益合计", "股东权益合计", "权益合计"], sheet_type="资产负债表")
    revenue = extract_amount(df_group, ["营业收入", "主营业务收入"], sheet_type="利润表")
    cost = extract_amount(df_group, ["营业成本", "主营业务成本"], sheet_type="利润表")
    operating_profit = extract_amount(df_group, ["营业利润"], sheet_type="利润表")
    net_profit = extract_amount(df_group, ["净利润"], sheet_type="利润表")
    operating_cf = extract_amount(df_group, ["经营活动产生的现金流量净额", "经营活动现金流量净额"], sheet_type="现金流量表")
    investing_cf = extract_amount(df_group, ["投资活动产生的现金流量净额", "投资活动现金流量净额"], sheet_type="现金流量表")
    financing_cf = extract_amount(df_group, ["筹资活动产生的现金流量净额", "筹资活动现金流量净额"], sheet_type="现金流量表")
    metrics["流动比率"] = current_assets / current_liabilities if current_liabilities != 0 else None
    metrics["速动比率"] = (current_assets - inventory) / current_liabilities if current_liabilities != 0 else None
    metrics["现金比率"] = cash / current_liabilities if current_liabilities != 0 else None
    metrics["资产负债率"] = liabilities_total / assets_total if assets_total != 0 else None
    metrics["产权比率"] = liabilities_total / equity_total if equity_total != 0 else None
    metrics["权益乘数"] = assets_total / equity_total if equity_total != 0 else None
    metrics["毛利率"] = (revenue - cost) / revenue if revenue != 0 else None
    metrics["营业利润率"] = operating_profit / revenue if revenue != 0 else None
    metrics["净利率"] = net_profit / revenue if revenue != 0 else None
    metrics["ROE(净资产收益率)"] = net_profit / equity_total if equity_total != 0 else None
    metrics["ROA(总资产收益率)"] = net_profit / assets_total if assets_total != 0 else None
    metrics["经营活动现金流净额"] = operating_cf
    metrics["投资活动现金流净额"] = investing_cf
    metrics["筹资活动现金流净额"] = financing_cf
    metrics["现金流量比率"] = operating_cf / current_liabilities if current_liabilities != 0 else None
    return metrics


def _list_excel_files(cfg: AppConfig) -> List[str]:
    pattern = os.path.join(cfg.input_dir, cfg.file_glob)
    files = glob.glob(pattern)
    files = [os.path.abspath(p) for p in files if os.path.isfile(p)]
    if cfg.exclude_output_files:
        exclude = {
            os.path.abspath(os.path.join(cfg.output_dir, cfg.output_basename)),
            os.path.abspath(os.path.join(cfg.output_dir, cfg.output_basename.replace(".xlsx", "_验证报告.xlsx"))),
            os.path.abspath(os.path.join(cfg.output_dir, cfg.output_basename.replace(".xlsx", "_财务指标.xlsx"))),
        }
        files = [p for p in files if os.path.abspath(p) not in exclude]
    files.sort(key=lambda p: p.lower())
    return files


ProgressCallback = Callable[[str, int, int, str], None]


def analyze_directory(
    cfg: AppConfig,
    logger: Optional[logging.Logger] = None,
    progress_cb: Optional[ProgressCallback] = None,
    cancel_event: Optional[threading.Event] = None,
) -> AnalysisResult:
    result = AnalysisResult()
    if logger is None:
        logger = _get_logger()

    _ensure_dir(cfg.output_dir)
    files = _list_excel_files(cfg)
    result.found_files = files

    if not files:
        msg = "未找到任何匹配的 .xlsx 文件"
        logger.warning(msg)
        result.errors.append(msg)
        return result

    logger.info(f"找到 {len(files)} 个Excel文件")
    all_files_data = []

    for idx, file_path in enumerate(files, start=1):
        if cancel_event and cancel_event.is_set():
            result.cancelled = True
            logger.warning("已取消运行")
            return result

        if progress_cb:
            progress_cb("file", idx, len(files), os.path.basename(file_path))

        logger.info(f"正在处理文件: {os.path.basename(file_path)}")
        try:
            excel_file = pd.ExcelFile(file_path)
            all_sheets = excel_file.sheet_names
            bs_sheets = [s for s in all_sheets if cfg.sheet_keyword_bs.upper() in s.upper()]
            pl_sheets = [s for s in all_sheets if cfg.sheet_keyword_pl.upper() in s.upper()]
            cf_sheets = [s for s in all_sheets if cfg.sheet_keyword_cf.upper() in s.upper()]

            logger.info(f"发现 {len(all_sheets)} 个Sheet")
            logger.info(f"BS: {bs_sheets if bs_sheets else '无'} | PL: {pl_sheets if pl_sheets else '无'} | CF: {cf_sheets if cf_sheets else '无'}")

            file_sheets_data = []
            for sheet in bs_sheets:
                if cancel_event and cancel_event.is_set():
                    result.cancelled = True
                    logger.warning("已取消运行")
                    return result
                df = clean_bs(file_path, sheet, cfg, logger)
                if not df.empty:
                    file_sheets_data.append(df)

            for sheet in pl_sheets:
                if cancel_event and cancel_event.is_set():
                    result.cancelled = True
                    logger.warning("已取消运行")
                    return result
                df = clean_pl(file_path, sheet, cfg, logger)
                if not df.empty:
                    file_sheets_data.append(df)

            for sheet in cf_sheets:
                if cancel_event and cancel_event.is_set():
                    result.cancelled = True
                    logger.warning("已取消运行")
                    return result
                df = clean_cf(file_path, sheet, cfg, logger)
                if not df.empty:
                    file_sheets_data.append(df)

            if file_sheets_data:
                file_data = pd.concat(file_sheets_data, ignore_index=True)
                file_data["源文件"] = os.path.basename(file_path)
                all_files_data.append(file_data)
                result.processed_files += 1
                logger.info(f"完成: 提取 {len(file_data)} 行")
            else:
                logger.warning("未提取到任何数据，可能缺少包含BS/PL/CF的Sheet")
        except Exception as e:
            logger.error(f"读取失败: {e}")
            result.errors.append(f"{os.path.basename(file_path)}: {e}")

    if cancel_event and cancel_event.is_set():
        result.cancelled = True
        logger.warning("已取消运行")
        return result

    if not all_files_data:
        msg = "所有文件均未提取到有效数据"
        logger.warning(msg)
        result.errors.append(msg)
        return result

    all_data = pd.concat(all_files_data, ignore_index=True)
    all_data["金额"] = all_data["金额"].astype(str).str.replace("—", "0").str.replace(",", "")
    all_data["金额"] = pd.to_numeric(all_data["金额"], errors="coerce").fillna(0)
    all_data["科目"] = all_data["科目"].astype(str).str.strip()

    cols = ["源文件", "来源Sheet", "日期", "报表类型", "大类", "科目", "时间属性", "金额"]
    final_cols = [c for c in cols if c in all_data.columns]
    all_data = all_data[final_cols]

    cleaned_path = os.path.abspath(os.path.join(cfg.output_dir, cfg.output_basename))
    all_data.to_excel(cleaned_path, index=False)
    result.cleaned_path = cleaned_path
    result.cleaned_rows = int(len(all_data))
    logger.info(f"原始数据已保存: {cleaned_path}")

    group_cols = ["源文件", "来源Sheet", "日期", "时间属性"]
    existing_group_cols = [col for col in group_cols if col in all_data.columns]

    validation_results = []
    metrics_results = []
    if existing_group_cols:
        grouped = all_data.groupby(existing_group_cols, dropna=False)
        for group_keys, df_group in grouped:
            if cancel_event and cancel_event.is_set():
                result.cancelled = True
                logger.warning("已取消运行")
                return result

            group_info = dict(zip(existing_group_cols, group_keys if isinstance(group_keys, tuple) else [group_keys]))
            if cfg.generate_validation and "资产负债表" in df_group["报表类型"].values:
                validation = validate_balance_sheet(df_group, tolerance=float(cfg.validation_tolerance))
                validation.update(group_info)
                validation_results.append(validation)

            if cfg.generate_metrics:
                metrics = calculate_financial_metrics(df_group)
                metrics.update(group_info)
                metrics_results.append(metrics)

    if cfg.generate_validation and validation_results:
        df_validation = pd.DataFrame(validation_results)
        validation_path = cleaned_path.replace(".xlsx", "_验证报告.xlsx")
        df_validation.to_excel(validation_path, index=False)
        result.validation_path = validation_path
        result.validation_groups = int(len(df_validation))
        unbalanced = df_validation[df_validation["是否平衡"] == "否"] if "是否平衡" in df_validation.columns else pd.DataFrame()
        result.unbalanced_count = int(len(unbalanced))
        if not unbalanced.empty:
            preview_cols = [c for c in ["源文件", "来源Sheet", "日期", "时间属性", "差额", "验证结果"] if c in unbalanced.columns]
            result.unbalanced_preview = unbalanced[preview_cols].head(200).to_dict(orient="records")
        logger.info(f"验证报告已保存: {validation_path}")

    if cfg.generate_metrics and metrics_results:
        df_metrics = pd.DataFrame(metrics_results)
        metrics_path = cleaned_path.replace(".xlsx", "_财务指标.xlsx")
        df_metrics.to_excel(metrics_path, index=False)
        result.metrics_path = metrics_path
        result.metrics_groups = int(len(df_metrics))
        logger.info(f"财务指标已保存: {metrics_path}")

    return result

