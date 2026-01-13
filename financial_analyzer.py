import pandas as pd
import datetime
import re
import glob
import os
import json
import logging
import queue
import threading
import argparse
from dataclasses import dataclass, asdict, field
from typing import Callable, Optional, Any, Dict, List, Tuple
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, ttk

# ================= 配置区域 =================
# 脚本将自动处理当前目录下所有的 .xlsx 文件
OUTPUT_PATH = '清洗后的AI标准财务表.xlsx'
# ===========================================

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
    header_keyword_pl: str = "本期金额"
    header_keyword_cf: str = "本期金额"
    date_cells_bs: List[List[int]] = field(default_factory=lambda: [[2, 3], [2, 2]])
    date_cells_pl: List[List[int]] = field(default_factory=lambda: [[2, 0], [2, 2]])
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
    if pd.isna(date_val) or date_val == '':
        return "未知日期"
    
    # 1. Excel 数字格式 (例如 45991)
    if isinstance(date_val, (int, float)):
        try:
            return (datetime.datetime(1899, 12, 30) + datetime.timedelta(days=int(date_val))).strftime('%Y-%m-%d')
        except:
            return str(date_val)
            
    # 2. 字符串格式处理
    text = str(date_val)
    # 提取所有数字，简单拼接 (处理 "2025年11月")
    digits = re.findall(r'\d+', text)
    if len(digits) >= 2:
        year = digits[0]
        month = digits[1].zfill(2)
        day = digits[2].zfill(2) if len(digits) > 2 else "01" # 如果没有日，默认为01号
        return f"{year}-{month}-{day}"
        
    return text.split(' ')[0]

def clean_bs(file_path, sheet_name, cfg: AppConfig, logger: Optional[logging.Logger] = None):
    """处理资产负债表 (包含BS的sheet) - 图1格式"""
    if logger:
        logger.info(f"正在处理: {sheet_name} ...")
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        
        # 1. 抓取日期 (图1显示在第3行左右)
        date_val = _read_date_from_cells(df, cfg.date_cells_bs)
        report_date = clean_date_str(date_val)
        
        # 2. 定位表头 (包含 '期末余额')
        header_row = _find_header_row(df, cfg.header_keyword_bs)
        if header_row is None:
            raise ValueError(f"未找到表头关键字: {cfg.header_keyword_bs}")
        
        # 3. 拆解左右分栏
        # 左边资产: [科目, 年初, 期末] -> A, B, C (Index 0,1,2)
        df_left = df.iloc[header_row + 1:, [0, 1, 2]].copy()
        df_left.columns = ['科目', '年初余额', '期末余额']
        df_left['大类'] = '资产'
        
        # 右边负债: [科目, 年初, 期末] -> D, E, F (Index 3,4,5)
        df_parts = [df_left]
        if df.shape[1] >= 6:
            df_right = df.iloc[header_row + 1:, [3, 4, 5]].copy()
            df_right.columns = ['科目', '年初余额', '期末余额']
            df_right['大类'] = '负债及权益'
            df_parts.append(df_right)
        
        # 4. 合并与清洗
        df_clean = pd.concat(df_parts, ignore_index=True)
        df_clean = df_clean.dropna(subset=['科目']) # 删除空行
        df_clean = df_clean[df_clean['科目'].astype(str).str.strip() != '']
        
        # 5. 逆透视
        df_final = df_clean.melt(id_vars=['大类', '科目'], 
                                 value_vars=['年初余额', '期末余额'],
                                 var_name='时间属性', value_name='金额')
        
        df_final['报表类型'] = '资产负债表'
        df_final['日期'] = report_date
        df_final['来源Sheet'] = sheet_name
        return df_final
    except Exception as e:
        if logger:
            logger.error(f"{sheet_name} 处理失败: {e}")
        return pd.DataFrame()

def clean_pl(file_path, sheet_name, cfg: AppConfig, logger: Optional[logging.Logger] = None):
    """处理利润表 (包含PL的sheet) - 图2格式"""
    if logger:
        logger.info(f"正在处理: {sheet_name} ...")
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        
        # 1. 抓取日期 (图2显示在第3行左右)
        date_val = _read_date_from_cells(df, cfg.date_cells_pl)
        report_date = clean_date_str(date_val)
        
        # 2. 定位表头 (包含 '本期金额')
        header_row = _find_header_row(df, cfg.header_keyword_pl)
        if header_row is None:
            raise ValueError(f"未找到表头关键字: {cfg.header_keyword_pl}")
        
        # 3. 提取数据
        # 结构: [科目(A), 行次(B), 本期(C), 累计(D)] -> 取 Index 0, 2, 3
        df_clean = df.iloc[header_row + 1:, [0, 2, 3]].copy()
        df_clean.columns = ['科目', '本期金额', '本年累计金额']
        
        df_clean = df_clean.dropna(subset=['科目'])
        
        # 4. 逆透视
        df_final = df_clean.melt(id_vars=['科目'], 
                                 value_vars=['本期金额', '本年累计金额'],
                                 var_name='时间属性', value_name='金额')
        
        df_final['大类'] = '损益'
        df_final['报表类型'] = '利润表'
        df_final['日期'] = report_date
        df_final['来源Sheet'] = sheet_name
        return df_final
    except Exception as e:
        if logger:
            logger.error(f"{sheet_name} 处理失败: {e}")
        return pd.DataFrame()

def clean_cf(file_path, sheet_name, cfg: AppConfig, logger: Optional[logging.Logger] = None):
    """处理现金流量表 (包含CF的sheet) - 图3格式"""
    if logger:
        logger.info(f"正在处理: {sheet_name} ...")
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        
        # 1. 抓取日期
        date_val = _read_date_from_cells(df, cfg.date_cells_cf)
        report_date = clean_date_str(date_val)
        
        # 2. 定位表头 (包含 '本期金额')
        header_row = _find_header_row(df, cfg.header_keyword_cf)
        if header_row is None:
            raise ValueError(f"未找到表头关键字: {cfg.header_keyword_cf}")
        
        # 3. 拆解左右分栏 (特别注意：中间夹着行次列)
        
        # 左边: [科目(A), 行次(B), 本期(C), 累计(D)] -> 取 Index 0, 2, 3
        df_left = df.iloc[header_row+1:, [0, 2, 3]].copy()
        df_left.columns = ['科目', '本期金额', '本年累计金额']
        
        # 右边: [科目(E), 行次(F), 本期(G), 累计(H)] -> 取 Index 4, 6, 7
        # 先检查是否有足够的列，防止报错
        if df.shape[1] >= 8:
            df_right = df.iloc[header_row+1:, [4, 6, 7]].copy()
            df_right.columns = ['科目', '本期金额', '本年累计金额']
            df_combined = pd.concat([df_left, df_right], ignore_index=True)
        else:
            df_combined = df_left
            
        # 4. 清洗
        df_combined = df_combined.dropna(subset=['科目'])
        df_combined = df_combined[df_combined['科目'].astype(str).str.strip() != '']
        
        # 5. 逆透视
        df_final = df_combined.melt(id_vars=['科目'], 
                                    value_vars=['本期金额', '本年累计金额'],
                                    var_name='时间属性', value_name='金额')
        
        df_final['大类'] = '现金流'
        df_final['报表类型'] = '现金流量表'
        df_final['日期'] = report_date
        df_final['来源Sheet'] = sheet_name
        return df_final
    except Exception as e:
        if logger:
            logger.error(f"{sheet_name} 处理失败: {e}")
        return pd.DataFrame()

# ================= 数据验证与财务指标计算 =================

def extract_amount(df, keywords, sheet_type=None, time_attr=None, category=None):
    """
    从DataFrame中提取符合条件的科目金额
    
    参数:
        df: 数据DataFrame
        keywords: 科目关键字列表，匹配任意一个即可
        sheet_type: 报表类型筛选（资产负债表、利润表、现金流量表）
        time_attr: 时间属性筛选（期末余额、年初余额等）
        category: 大类筛选（资产、负债及权益、损益、现金流）
    
    返回: 匹配到的第一个金额值，未找到返回0
    """
    filtered_df = df.copy()
    
    # 筛选条件
    if sheet_type:
        filtered_df = filtered_df[filtered_df['报表类型'] == sheet_type]
    if time_attr:
        filtered_df = filtered_df[filtered_df['时间属性'] == time_attr]
    if category:
        filtered_df = filtered_df[filtered_df['大类'] == category]
    
    # 科目名称匹配（精确匹配，忽略前后空格，不区分大小写）
    for keyword in keywords:
        matched = filtered_df[filtered_df['科目'].str.strip().str.lower() == keyword.lower()]
        if not matched.empty:
            return matched.iloc[0]['金额']
    
    return 0

def validate_balance_sheet(
    df_group,
    tolerance: float = 0.01,
    assets_keywords: Optional[List[str]] = None,
    liabilities_keywords: Optional[List[str]] = None,
    equity_keywords: Optional[List[str]] = None,
):
    """
    验证资产负债表的会计恒等式：资产 = 负债 + 所有者权益
    
    参数:
        df_group: 单个分组的数据（同一文件、Sheet、日期、时间点）
    
    返回: dict包含验证结果
    """
    # 提取关键科目
    assets = extract_amount(df_group, assets_keywords or ['资产总计', '资产总额', '资产合计'], 
                           sheet_type='资产负债表', category='资产')
    liabilities = extract_amount(df_group, liabilities_keywords or ['负债合计', '负债总计', '负债总额'], 
                                 sheet_type='资产负债表', category='负债及权益')
    equity = extract_amount(df_group, equity_keywords or ['所有者权益合计', '股东权益合计', '所有者权益总计', '权益合计'], 
                           sheet_type='资产负债表', category='负债及权益')
    
    # 计算差额
    diff = abs(assets - (liabilities + equity))
    is_balanced = diff <= tolerance
    
    return {
        '资产总计': assets,
        '负债合计': liabilities,
        '所有者权益合计': equity,
        '差额': diff,
        '是否平衡': '是' if is_balanced else '否',
        '验证结果': '通过' if is_balanced else f'不平衡(差额:{diff:.2f})'
    }

def calculate_financial_metrics(df_group):
    """
    计算财务指标
    
    参数:
        df_group: 单个分组的数据（同一文件、Sheet、日期、时间点）
    
    返回: dict包含各类财务指标
    """
    metrics = {}
    
    # ===== 提取基础科目金额 =====
    # 资产负债表科目
    assets_total = extract_amount(df_group, ['资产总计', '资产总额'], sheet_type='资产负债表')
    current_assets = extract_amount(df_group, ['流动资产合计', '流动资产总计'], sheet_type='资产负债表')
    cash = extract_amount(df_group, ['货币资金', '现金及现金等价物'], sheet_type='资产负债表')
    inventory = extract_amount(df_group, ['存货'], sheet_type='资产负债表')
    
    liabilities_total = extract_amount(df_group, ['负债合计', '负债总计'], sheet_type='资产负债表')
    current_liabilities = extract_amount(df_group, ['流动负债合计', '流动负债总计'], sheet_type='资产负债表')
    equity_total = extract_amount(df_group, ['所有者权益合计', '股东权益合计', '权益合计'], sheet_type='资产负债表')
    
    # 利润表科目
    revenue = extract_amount(df_group, ['营业收入', '主营业务收入'], sheet_type='利润表')
    cost = extract_amount(df_group, ['营业成本', '主营业务成本'], sheet_type='利润表')
    operating_profit = extract_amount(df_group, ['营业利润'], sheet_type='利润表')
    net_profit = extract_amount(df_group, ['净利润'], sheet_type='利润表')
    
    # 现金流量表科目
    operating_cf = extract_amount(df_group, ['经营活动产生的现金流量净额', '经营活动现金流量净额'], sheet_type='现金流量表')
    investing_cf = extract_amount(df_group, ['投资活动产生的现金流量净额', '投资活动现金流量净额'], sheet_type='现金流量表')
    financing_cf = extract_amount(df_group, ['筹资活动产生的现金流量净额', '筹资活动现金流量净额'], sheet_type='现金流量表')
    
    # ===== 计算流动性指标 =====
    metrics['流动比率'] = current_assets / current_liabilities if current_liabilities != 0 else None
    metrics['速动比率'] = (current_assets - inventory) / current_liabilities if current_liabilities != 0 else None
    metrics['现金比率'] = cash / current_liabilities if current_liabilities != 0 else None
    
    # ===== 计算偿债能力指标 =====
    metrics['资产负债率'] = liabilities_total / assets_total if assets_total != 0 else None
    metrics['产权比率'] = liabilities_total / equity_total if equity_total != 0 else None
    metrics['权益乘数'] = assets_total / equity_total if equity_total != 0 else None
    
    # ===== 计算盈利能力指标 =====
    metrics['毛利率'] = (revenue - cost) / revenue if revenue != 0 else None
    metrics['营业利润率'] = operating_profit / revenue if revenue != 0 else None
    metrics['净利率'] = net_profit / revenue if revenue != 0 else None
    metrics['ROE(净资产收益率)'] = net_profit / equity_total if equity_total != 0 else None
    metrics['ROA(总资产收益率)'] = net_profit / assets_total if assets_total != 0 else None
    
    # ===== 现金流指标 =====
    metrics['经营活动现金流净额'] = operating_cf
    metrics['投资活动现金流净额'] = investing_cf
    metrics['筹资活动现金流净额'] = financing_cf
    metrics['现金流量比率'] = operating_cf / current_liabilities if current_liabilities != 0 else None
    
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
                file_data['源文件'] = os.path.basename(file_path)
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
    all_data['金额'] = all_data['金额'].astype(str).str.replace('—', '0').str.replace(',', '')
    all_data['金额'] = pd.to_numeric(all_data['金额'], errors='coerce').fillna(0)
    all_data['科目'] = all_data['科目'].astype(str).str.strip()

    cols = ['源文件', '来源Sheet', '日期', '报表类型', '大类', '科目', '时间属性', '金额']
    final_cols = [c for c in cols if c in all_data.columns]
    all_data = all_data[final_cols]

    cleaned_path = os.path.abspath(os.path.join(cfg.output_dir, cfg.output_basename))
    all_data.to_excel(cleaned_path, index=False)
    result.cleaned_path = cleaned_path
    result.cleaned_rows = int(len(all_data))
    logger.info(f"原始数据已保存: {cleaned_path}")

    group_cols = ['源文件', '来源Sheet', '日期', '时间属性']
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
            if cfg.generate_validation and '资产负债表' in df_group['报表类型'].values:
                validation = validate_balance_sheet(df_group, tolerance=float(cfg.validation_tolerance))
                validation.update(group_info)
                validation_results.append(validation)

            if cfg.generate_metrics:
                metrics = calculate_financial_metrics(df_group)
                metrics.update(group_info)
                metrics_results.append(metrics)

    if cfg.generate_validation and validation_results:
        df_validation = pd.DataFrame(validation_results)
        validation_path = cleaned_path.replace('.xlsx', '_验证报告.xlsx')
        df_validation.to_excel(validation_path, index=False)
        result.validation_path = validation_path
        result.validation_groups = int(len(df_validation))
        unbalanced = df_validation[df_validation['是否平衡'] == '否'] if '是否平衡' in df_validation.columns else pd.DataFrame()
        result.unbalanced_count = int(len(unbalanced))
        if not unbalanced.empty:
            preview_cols = [c for c in ['源文件', '来源Sheet', '日期', '时间属性', '差额', '验证结果'] if c in unbalanced.columns]
            result.unbalanced_preview = unbalanced[preview_cols].head(200).to_dict(orient="records")
        logger.info(f"验证报告已保存: {validation_path}")

    if cfg.generate_metrics and metrics_results:
        df_metrics = pd.DataFrame(metrics_results)
        metrics_path = cleaned_path.replace('.xlsx', '_财务指标.xlsx')
        df_metrics.to_excel(metrics_path, index=False)
        result.metrics_path = metrics_path
        result.metrics_groups = int(len(df_metrics))
        logger.info(f"财务指标已保存: {metrics_path}")

    return result


class _QueueLogHandler(logging.Handler):
    def __init__(self, q: "queue.Queue[Dict[str, Any]]"):
        super().__init__()
        self.q = q
        self.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self.q.put({"type": "log", "level": record.levelname, "message": msg})
        except Exception:
            pass


class FinancialAnalyzerUI:
    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH):
        try:
            import customtkinter as ctk
        except Exception:
            messagebox.showerror("缺少依赖", "未安装 customtkinter。请先安装：pip install customtkinter")
            raise

        self.ctk = ctk
        self.config_path = config_path
        self.cfg = load_config(config_path)

        self.ui_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self.cancel_event = threading.Event()
        self.worker_thread: Optional[threading.Thread] = None
        self.last_result: Optional[AnalysisResult] = None

        ctk.set_appearance_mode("Light")
        ctk.set_default_color_theme("blue")
        ctk.set_widget_scaling(1.15)

        try:
            tkfont.nametofont("TkDefaultFont").configure(size=14)
            tkfont.nametofont("TkTextFont").configure(size=14)
            tkfont.nametofont("TkFixedFont").configure(size=13)
            tkfont.nametofont("TkMenuFont").configure(size=13)
        except Exception:
            pass

        self.root = ctk.CTk()
        self.root.title("财务数据分析")
        self.root.geometry("1100x720")

        self.tabview = ctk.CTkTabview(self.root)
        self.tabview.pack(fill="both", expand=True, padx=14, pady=14)

        self.tab_run = self.tabview.add("运行")
        self.tab_config = self.tabview.add("配置")
        self.tab_logs = self.tabview.add("日志")
        self.tab_results = self.tabview.add("结果")

        self._build_run_tab()
        self._build_config_tab()
        self._build_logs_tab()
        self._build_results_tab()

        try:
            style = ttk.Style()
            style.configure("Treeview", font=("Segoe UI", 13), rowheight=30)
            style.configure("Treeview.Heading", font=("Segoe UI", 13, "bold"))
        except Exception:
            pass

        self._refresh_config_to_ui()
        self._poll_queue()

    def run(self) -> None:
        self.root.mainloop()

    def _build_run_tab(self) -> None:
        ctk = self.ctk
        frame = ctk.CTkFrame(self.tab_run)
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        grid = ctk.CTkFrame(frame)
        grid.pack(fill="x", padx=12, pady=12)
        grid.columnconfigure(1, weight=1)
        grid.columnconfigure(3, weight=1)

        ctk.CTkLabel(grid, text="数据文件夹").grid(row=0, column=0, sticky="w", padx=10, pady=8)
        self.var_input_dir = tk.StringVar()
        self.entry_input_dir = ctk.CTkEntry(grid, textvariable=self.var_input_dir)
        self.entry_input_dir.grid(row=0, column=1, sticky="ew", padx=10, pady=8)
        ctk.CTkButton(grid, text="选择", width=80, command=self._pick_input_dir).grid(row=0, column=2, padx=10, pady=8)

        ctk.CTkLabel(grid, text="文件匹配").grid(row=0, column=3, sticky="w", padx=10, pady=8)
        self.var_file_glob = tk.StringVar()
        ctk.CTkEntry(grid, textvariable=self.var_file_glob).grid(row=0, column=4, sticky="ew", padx=10, pady=8)

        ctk.CTkLabel(grid, text="输出文件夹").grid(row=1, column=0, sticky="w", padx=10, pady=8)
        self.var_output_dir = tk.StringVar()
        self.entry_output_dir = ctk.CTkEntry(grid, textvariable=self.var_output_dir)
        self.entry_output_dir.grid(row=1, column=1, sticky="ew", padx=10, pady=8)
        ctk.CTkButton(grid, text="选择", width=80, command=self._pick_output_dir).grid(row=1, column=2, padx=10, pady=8)

        ctk.CTkLabel(grid, text="输出文件名").grid(row=1, column=3, sticky="w", padx=10, pady=8)
        self.var_output_basename = tk.StringVar()
        ctk.CTkEntry(grid, textvariable=self.var_output_basename).grid(row=1, column=4, sticky="ew", padx=10, pady=8)

        opt = ctk.CTkFrame(frame)
        opt.pack(fill="x", padx=12, pady=(0, 12))

        self.var_gen_validation = tk.BooleanVar()
        self.var_gen_metrics = tk.BooleanVar()
        self.var_exclude_outputs = tk.BooleanVar()

        ctk.CTkCheckBox(opt, text="生成验证报告", variable=self.var_gen_validation).pack(side="left", padx=10, pady=10)
        ctk.CTkCheckBox(opt, text="生成财务指标", variable=self.var_gen_metrics).pack(side="left", padx=10, pady=10)
        ctk.CTkCheckBox(opt, text="排除输出文件", variable=self.var_exclude_outputs).pack(side="left", padx=10, pady=10)

        btns = ctk.CTkFrame(frame)
        btns.pack(fill="x", padx=12, pady=(0, 12))
        self.btn_scan = ctk.CTkButton(btns, text="扫描文件", command=self._scan_files)
        self.btn_scan.pack(side="left", padx=10, pady=10)
        self.btn_start = ctk.CTkButton(btns, text="开始运行", command=self._start_run)
        self.btn_start.pack(side="left", padx=10, pady=10)
        self.btn_stop = ctk.CTkButton(btns, text="停止", fg_color="#8B0000", hover_color="#A40000", command=self._stop_run, state="disabled")
        self.btn_stop.pack(side="left", padx=10, pady=10)

        self.progress = ctk.CTkProgressBar(frame)
        self.progress.pack(fill="x", padx=12, pady=(0, 10))
        self.progress.set(0)

        self.var_status = tk.StringVar(value="就绪")
        ctk.CTkLabel(frame, textvariable=self.var_status).pack(anchor="w", padx=14, pady=(0, 12))

        list_frame = ctk.CTkFrame(frame)
        list_frame.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        ctk.CTkLabel(list_frame, text="文件预览").pack(anchor="w", padx=10, pady=(10, 6))
        self.files_box = ctk.CTkTextbox(list_frame)
        self.files_box.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.files_box.configure(state="disabled")

    def _build_config_tab(self) -> None:
        ctk = self.ctk
        frame = ctk.CTkFrame(self.tab_config)
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        top = ctk.CTkFrame(frame)
        top.pack(fill="x", padx=12, pady=12)

        self.btn_load_cfg = ctk.CTkButton(top, text="从文件加载", command=self._load_config_file)
        self.btn_load_cfg.pack(side="left", padx=10, pady=10)
        self.btn_save_cfg = ctk.CTkButton(top, text="保存到文件", command=self._save_config_file)
        self.btn_save_cfg.pack(side="left", padx=10, pady=10)
        self.btn_reset_cfg = ctk.CTkButton(top, text="恢复默认", command=self._reset_config)
        self.btn_reset_cfg.pack(side="left", padx=10, pady=10)

        cfg_tabs = ctk.CTkTabview(frame)
        cfg_tabs.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        tab_form = cfg_tabs.add("表单")
        tab_json = cfg_tabs.add("JSON")

        form_host = ctk.CTkScrollableFrame(tab_form)
        form_host.pack(fill="both", expand=True, padx=10, pady=10)
        form_host.columnconfigure(1, weight=1)
        form_host.columnconfigure(3, weight=1)

        self.cfg_form_vars: Dict[str, Any] = {}

        help_texts: Dict[str, str] = {
            "input_dir": "要扫描的 Excel 所在文件夹。",
            "file_glob": "glob 匹配模式，例如 *.xlsx 或 *合并*.xlsx。",
            "output_dir": "清洗结果输出的文件夹。",
            "output_basename": "输出文件名（.xlsx）。会自动生成 _验证报告.xlsx / _财务指标.xlsx。",
            "generate_validation": "勾选后输出 资产=负债+权益 的验证报告。",
            "generate_metrics": "勾选后输出常用财务指标汇总。",
            "exclude_output_files": "避免把已生成的输出文件再次扫描。",
            "sheet_keyword_bs": "用于从 Sheet 名称识别资产负债表（包含即可，不区分大小写）。",
            "sheet_keyword_pl": "用于从 Sheet 名称识别利润表（包含即可，不区分大小写）。",
            "sheet_keyword_cf": "用于从 Sheet 名称识别现金流量表（包含即可，不区分大小写）。",
            "header_keyword_bs": "用于定位资产负债表表头行（整行包含关键字即可）。",
            "header_keyword_pl": "用于定位利润表表头行（整行包含关键字即可）。",
            "header_keyword_cf": "用于定位现金流量表表头行（整行包含关键字即可）。",
            "date_cells_bs": "候选单元格格式：行,列;行,列（从0开始）。非日期会自动在附近搜索。",
            "date_cells_pl": "候选单元格格式：行,列;行,列（从0开始）。非日期会自动在附近搜索。",
            "date_cells_cf": "候选单元格格式：行,列;行,列（从0开始）。非日期会自动在附近搜索。",
            "validation_tolerance": "验证容差阈值（数值），例如 0.01。",
        }

        def add_row(row: int, label: str, key: str, kind: str = "entry", width: int = 0, col: int = 0):
            r_main = row * 2
            r_help = r_main + 1
            ctk.CTkLabel(form_host, text=label).grid(row=r_main, column=col, sticky="w", padx=10, pady=8)
            if kind == "bool":
                var = tk.BooleanVar()
                self.cfg_form_vars[key] = var
                ctk.CTkSwitch(form_host, text="", variable=var).grid(row=r_main, column=col + 1, sticky="w", padx=10, pady=8)
                help_text = help_texts.get(key, "")
                if help_text:
                    ctk.CTkLabel(
                        form_host,
                        text=help_text,
                        justify="left",
                        text_color="#555555",
                        font=("Segoe UI", 12),
                        wraplength=470,
                    ).grid(row=r_help, column=col + 1, columnspan=2, sticky="w", padx=10, pady=(0, 10))
                return
            var = tk.StringVar()
            self.cfg_form_vars[key] = var
            if width:
                entry = ctk.CTkEntry(form_host, textvariable=var, width=width)
            else:
                entry = ctk.CTkEntry(form_host, textvariable=var)
            entry.grid(row=r_main, column=col + 1, sticky="ew", padx=10, pady=8)

            help_text = help_texts.get(key, "")
            if help_text:
                ctk.CTkLabel(
                    form_host,
                    text=help_text,
                    justify="left",
                    text_color="#555555",
                    font=("Segoe UI", 12),
                    wraplength=470,
                ).grid(row=r_help, column=col + 1, columnspan=2, sticky="w", padx=10, pady=(0, 10))

        add_row(0, "输入目录", "input_dir", col=0)
        ctk.CTkButton(form_host, text="选择", width=90, command=self._pick_input_dir_from_config).grid(row=0, column=2, padx=10, pady=8, sticky="e")
        add_row(0, "文件匹配", "file_glob", col=3)

        add_row(1, "输出目录", "output_dir", col=0)
        ctk.CTkButton(form_host, text="选择", width=90, command=self._pick_output_dir_from_config).grid(row=2, column=2, padx=10, pady=8, sticky="e")
        add_row(1, "输出文件名", "output_basename", col=3)

        add_row(2, "生成验证报告", "generate_validation", kind="bool", col=0)
        add_row(2, "生成财务指标", "generate_metrics", kind="bool", col=3)
        add_row(3, "排除输出文件", "exclude_output_files", kind="bool", col=0)

        add_row(4, "BS Sheet关键字", "sheet_keyword_bs", col=0)
        add_row(4, "PL Sheet关键字", "sheet_keyword_pl", col=3)
        add_row(5, "CF Sheet关键字", "sheet_keyword_cf", col=0)

        add_row(6, "BS 表头关键字", "header_keyword_bs", col=0)
        add_row(6, "PL 表头关键字", "header_keyword_pl", col=3)
        add_row(7, "CF 表头关键字", "header_keyword_cf", col=0)

        add_row(8, "BS 日期单元格", "date_cells_bs", col=0)
        add_row(8, "PL 日期单元格", "date_cells_pl", col=3)
        add_row(9, "CF 日期单元格", "date_cells_cf", col=0)
        add_row(9, "验证容差", "validation_tolerance", col=3)

        form_btns = ctk.CTkFrame(tab_form)
        form_btns.pack(fill="x", padx=10, pady=(0, 10))
        ctk.CTkButton(form_btns, text="应用表单到配置", command=self._apply_config_form).pack(side="left", padx=10, pady=10)
        ctk.CTkButton(form_btns, text="从配置刷新表单", command=self._refresh_config_to_ui).pack(side="left", padx=10, pady=10)

        json_host = ctk.CTkFrame(tab_json)
        json_host.pack(fill="both", expand=True, padx=10, pady=10)
        json_btns = ctk.CTkFrame(json_host)
        json_btns.pack(fill="x", padx=10, pady=10)
        self.btn_apply_cfg = ctk.CTkButton(json_btns, text="应用JSON到配置", command=self._apply_config_json)
        self.btn_apply_cfg.pack(side="left", padx=10, pady=10)
        ctk.CTkButton(json_btns, text="从配置刷新JSON", command=self._refresh_config_to_ui).pack(side="left", padx=10, pady=10)

        self.cfg_box = ctk.CTkTextbox(json_host)
        self.cfg_box.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def _build_logs_tab(self) -> None:
        ctk = self.ctk
        frame = ctk.CTkFrame(self.tab_logs)
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        top = ctk.CTkFrame(frame)
        top.pack(fill="x", padx=12, pady=12)
        ctk.CTkButton(top, text="清空", command=self._clear_logs).pack(side="left", padx=10, pady=10)
        ctk.CTkButton(top, text="导出", command=self._export_logs).pack(side="left", padx=10, pady=10)
        ctk.CTkButton(top, text="复制全部", command=self._copy_logs).pack(side="left", padx=10, pady=10)

        self.logs_box = ctk.CTkTextbox(frame)
        self.logs_box.pack(fill="both", expand=True, padx=12, pady=(0, 12))

    def _build_results_tab(self) -> None:
        ctk = self.ctk
        frame = ctk.CTkFrame(self.tab_results)
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        top = ctk.CTkFrame(frame)
        top.pack(fill="x", padx=12, pady=12)
        top.columnconfigure(1, weight=1)

        self.var_out_cleaned = tk.StringVar(value="")
        self.var_out_validation = tk.StringVar(value="")
        self.var_out_metrics = tk.StringVar(value="")

        ctk.CTkLabel(top, text="清洗数据").grid(row=0, column=0, sticky="w", padx=10, pady=8)
        ctk.CTkEntry(top, textvariable=self.var_out_cleaned).grid(row=0, column=1, sticky="ew", padx=10, pady=8)
        ctk.CTkButton(top, text="打开", width=80, command=lambda: self._open_path(self.var_out_cleaned.get())).grid(row=0, column=2, padx=10, pady=8)

        ctk.CTkLabel(top, text="验证报告").grid(row=1, column=0, sticky="w", padx=10, pady=8)
        ctk.CTkEntry(top, textvariable=self.var_out_validation).grid(row=1, column=1, sticky="ew", padx=10, pady=8)
        ctk.CTkButton(top, text="打开", width=80, command=lambda: self._open_path(self.var_out_validation.get())).grid(row=1, column=2, padx=10, pady=8)

        ctk.CTkLabel(top, text="财务指标").grid(row=2, column=0, sticky="w", padx=10, pady=8)
        ctk.CTkEntry(top, textvariable=self.var_out_metrics).grid(row=2, column=1, sticky="ew", padx=10, pady=8)
        ctk.CTkButton(top, text="打开", width=80, command=lambda: self._open_path(self.var_out_metrics.get())).grid(row=2, column=2, padx=10, pady=8)

        mid = ctk.CTkFrame(frame)
        mid.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        ctk.CTkLabel(mid, text="日志检查：不平衡记录预览（最多200条）").pack(anchor="w", padx=10, pady=(10, 6))

        table_host = tk.Frame(mid, background="#FFFFFF")
        table_host.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.tree = ttk.Treeview(table_host, columns=("源文件", "来源Sheet", "日期", "时间属性", "差额", "验证结果"), show="headings", height=12)
        for col in self.tree["columns"]:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=140, anchor="w")
        self.tree.pack(fill="both", expand=True, side="left")
        sb = ttk.Scrollbar(table_host, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        sb.pack(fill="y", side="right")

    def _refresh_config_to_ui(self) -> None:
        self.var_input_dir.set(self.cfg.input_dir)
        self.var_file_glob.set(self.cfg.file_glob)
        self.var_output_dir.set(self.cfg.output_dir)
        self.var_output_basename.set(self.cfg.output_basename)
        self.var_gen_validation.set(bool(self.cfg.generate_validation))
        self.var_gen_metrics.set(bool(self.cfg.generate_metrics))
        self.var_exclude_outputs.set(bool(self.cfg.exclude_output_files))
        if hasattr(self, "cfg_box"):
            self.cfg_box.delete("1.0", "end")
            self.cfg_box.insert("1.0", json.dumps(asdict(self.cfg), ensure_ascii=False, indent=2))

        if hasattr(self, "cfg_form_vars") and self.cfg_form_vars:
            def set_str(key: str, value: Any) -> None:
                var = self.cfg_form_vars.get(key)
                if isinstance(var, tk.StringVar):
                    var.set("" if value is None else str(value))

            def set_bool(key: str, value: Any) -> None:
                var = self.cfg_form_vars.get(key)
                if isinstance(var, tk.BooleanVar):
                    var.set(bool(value))

            set_str("input_dir", self.cfg.input_dir)
            set_str("file_glob", self.cfg.file_glob)
            set_str("output_dir", self.cfg.output_dir)
            set_str("output_basename", self.cfg.output_basename)
            set_bool("generate_validation", self.cfg.generate_validation)
            set_bool("generate_metrics", self.cfg.generate_metrics)
            set_bool("exclude_output_files", self.cfg.exclude_output_files)
            set_str("sheet_keyword_bs", self.cfg.sheet_keyword_bs)
            set_str("sheet_keyword_pl", self.cfg.sheet_keyword_pl)
            set_str("sheet_keyword_cf", self.cfg.sheet_keyword_cf)
            set_str("header_keyword_bs", self.cfg.header_keyword_bs)
            set_str("header_keyword_pl", self.cfg.header_keyword_pl)
            set_str("header_keyword_cf", self.cfg.header_keyword_cf)
            set_str("date_cells_bs", self._cells_to_text(self.cfg.date_cells_bs))
            set_str("date_cells_pl", self._cells_to_text(self.cfg.date_cells_pl))
            set_str("date_cells_cf", self._cells_to_text(self.cfg.date_cells_cf))
            set_str("validation_tolerance", str(self.cfg.validation_tolerance))

    def _cells_to_text(self, cells: Any) -> str:
        pairs = []
        if isinstance(cells, list):
            for item in cells:
                rc = _safe_int_pair(item)
                if rc is not None:
                    pairs.append(f"{rc[0]},{rc[1]}")
        return ";".join(pairs)

    def _text_to_cells(self, text: str) -> List[List[int]]:
        text = (text or "").strip()
        if not text:
            return []
        parts = []
        for chunk in re.split(r"[;\n|]+", text):
            chunk = chunk.strip()
            if not chunk:
                continue
            items = re.split(r"[, \t]+", chunk)
            if len(items) < 2:
                continue
            try:
                r = int(items[0])
                c = int(items[1])
                parts.append([r, c])
            except Exception:
                continue
        return parts

    def _apply_config_form(self) -> None:
        if not getattr(self, "cfg_form_vars", None):
            return
        get_str = lambda k: str(self.cfg_form_vars.get(k).get()).strip() if self.cfg_form_vars.get(k) is not None else ""
        get_bool = lambda k: bool(self.cfg_form_vars.get(k).get()) if self.cfg_form_vars.get(k) is not None else False

        self.cfg.input_dir = get_str("input_dir") or os.getcwd()
        self.cfg.file_glob = get_str("file_glob") or "*.xlsx"
        self.cfg.output_dir = get_str("output_dir") or os.getcwd()
        self.cfg.output_basename = get_str("output_basename") or OUTPUT_PATH
        self.cfg.generate_validation = get_bool("generate_validation")
        self.cfg.generate_metrics = get_bool("generate_metrics")
        self.cfg.exclude_output_files = get_bool("exclude_output_files")
        self.cfg.sheet_keyword_bs = get_str("sheet_keyword_bs") or "BS"
        self.cfg.sheet_keyword_pl = get_str("sheet_keyword_pl") or "PL"
        self.cfg.sheet_keyword_cf = get_str("sheet_keyword_cf") or "CF"
        self.cfg.header_keyword_bs = get_str("header_keyword_bs") or "期末余额"
        self.cfg.header_keyword_pl = get_str("header_keyword_pl") or "本期金额"
        self.cfg.header_keyword_cf = get_str("header_keyword_cf") or "本期金额"
        self.cfg.date_cells_bs = self._text_to_cells(get_str("date_cells_bs")) or AppConfig().date_cells_bs
        self.cfg.date_cells_pl = self._text_to_cells(get_str("date_cells_pl")) or AppConfig().date_cells_pl
        self.cfg.date_cells_cf = self._text_to_cells(get_str("date_cells_cf")) or AppConfig().date_cells_cf
        try:
            self.cfg.validation_tolerance = float(get_str("validation_tolerance"))
        except Exception:
            self.cfg.validation_tolerance = AppConfig().validation_tolerance

        self._refresh_config_to_ui()
        self.var_status.set("已应用表单配置")

    def _pick_input_dir_from_config(self) -> None:
        initial = ""
        if getattr(self, "cfg_form_vars", None) and isinstance(self.cfg_form_vars.get("input_dir"), tk.StringVar):
            initial = self.cfg_form_vars["input_dir"].get()
        path = filedialog.askdirectory(title="选择输入目录", initialdir=initial or os.getcwd())
        if path and getattr(self, "cfg_form_vars", None) and isinstance(self.cfg_form_vars.get("input_dir"), tk.StringVar):
            self.cfg_form_vars["input_dir"].set(path)

    def _pick_output_dir_from_config(self) -> None:
        initial = ""
        if getattr(self, "cfg_form_vars", None) and isinstance(self.cfg_form_vars.get("output_dir"), tk.StringVar):
            initial = self.cfg_form_vars["output_dir"].get()
        path = filedialog.askdirectory(title="选择输出目录", initialdir=initial or os.getcwd())
        if path and getattr(self, "cfg_form_vars", None) and isinstance(self.cfg_form_vars.get("output_dir"), tk.StringVar):
            self.cfg_form_vars["output_dir"].set(path)

    def _sync_ui_to_config(self) -> None:
        self.cfg.input_dir = self.var_input_dir.get().strip() or os.getcwd()
        self.cfg.file_glob = self.var_file_glob.get().strip() or "*.xlsx"
        self.cfg.output_dir = self.var_output_dir.get().strip() or os.getcwd()
        self.cfg.output_basename = self.var_output_basename.get().strip() or OUTPUT_PATH
        self.cfg.generate_validation = bool(self.var_gen_validation.get())
        self.cfg.generate_metrics = bool(self.var_gen_metrics.get())
        self.cfg.exclude_output_files = bool(self.var_exclude_outputs.get())

    def _pick_input_dir(self) -> None:
        path = filedialog.askdirectory(title="选择数据文件夹", initialdir=self.var_input_dir.get() or os.getcwd())
        if path:
            self.var_input_dir.set(path)

    def _pick_output_dir(self) -> None:
        path = filedialog.askdirectory(title="选择输出文件夹", initialdir=self.var_output_dir.get() or os.getcwd())
        if path:
            self.var_output_dir.set(path)

    def _scan_files(self) -> None:
        self._sync_ui_to_config()
        files = _list_excel_files(self.cfg)
        self.files_box.configure(state="normal")
        self.files_box.delete("1.0", "end")
        for p in files[:500]:
            self.files_box.insert("end", os.path.basename(p) + "\n")
        if len(files) > 500:
            self.files_box.insert("end", f"... 还有 {len(files) - 500} 个文件未显示\n")
        self.files_box.configure(state="disabled")
        self.var_status.set(f"扫描到 {len(files)} 个文件")

    def _clear_logs(self) -> None:
        self.logs_box.delete("1.0", "end")

    def _copy_logs(self) -> None:
        text = self.logs_box.get("1.0", "end")
        self.root.clipboard_clear()
        self.root.clipboard_append(text)

    def _export_logs(self) -> None:
        path = filedialog.asksaveasfilename(title="导出日志", defaultextension=".txt", filetypes=[("Text", "*.txt")])
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.logs_box.get("1.0", "end"))
        messagebox.showinfo("完成", f"已导出：{path}")

    def _open_path(self, path: str) -> None:
        path = (path or "").strip()
        if not path:
            return
        if not os.path.exists(path):
            messagebox.showwarning("不存在", f"路径不存在：{path}")
            return
        try:
            os.startfile(path)
        except Exception as e:
            messagebox.showerror("打开失败", str(e))

    def _load_config_file(self) -> None:
        path = filedialog.askopenfilename(title="选择配置文件", filetypes=[("JSON", "*.json")], initialdir=os.path.dirname(self.config_path))
        if not path:
            return
        self.cfg = load_config(path)
        self.config_path = path
        self._refresh_config_to_ui()
        self.var_status.set("已加载配置")

    def _save_config_file(self) -> None:
        self._apply_config_json(silent=True)
        path = filedialog.asksaveasfilename(title="保存配置文件", defaultextension=".json", filetypes=[("JSON", "*.json")], initialdir=os.path.dirname(self.config_path))
        if not path:
            return
        save_config(path, self.cfg)
        self.config_path = path
        self.var_status.set("已保存配置")

    def _reset_config(self) -> None:
        self.cfg = AppConfig()
        self._refresh_config_to_ui()
        self.var_status.set("已恢复默认配置")

    def _apply_config_json(self, silent: bool = False) -> None:
        try:
            text = self.cfg_box.get("1.0", "end").strip()
            data = json.loads(text) if text else {}
            cfg = AppConfig()
            for k, v in data.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
            self.cfg = cfg
            self._refresh_config_to_ui()
            if not silent:
                self.var_status.set("已应用配置JSON")
        except Exception as e:
            if not silent:
                messagebox.showerror("配置解析失败", str(e))

    def _start_run(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            return
        self._apply_config_json(silent=True)
        self._sync_ui_to_config()
        self.progress.set(0)
        self.var_status.set("运行中...")
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.tabview.set("日志")
        self._clear_logs()
        self._clear_results()
        self.cancel_event.clear()

        handler = _QueueLogHandler(self.ui_queue)
        logger = _get_logger(handler=handler)

        def progress_cb(stage: str, current: int, total: int, detail: str) -> None:
            self.ui_queue.put({"type": "progress", "stage": stage, "current": current, "total": total, "detail": detail})

        def worker() -> None:
            try:
                res = analyze_directory(self.cfg, logger=logger, progress_cb=progress_cb, cancel_event=self.cancel_event)
                self.ui_queue.put({"type": "done", "result": res})
            except Exception as e:
                self.ui_queue.put({"type": "done", "result": AnalysisResult(errors=[str(e)])})

        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()

    def _stop_run(self) -> None:
        self.cancel_event.set()
        self.var_status.set("正在停止...")

    def _clear_results(self) -> None:
        self.var_out_cleaned.set("")
        self.var_out_validation.set("")
        self.var_out_metrics.set("")
        for item in self.tree.get_children():
            self.tree.delete(item)

    def _apply_result(self, res: AnalysisResult) -> None:
        self.last_result = res
        self.var_out_cleaned.set(res.cleaned_path or "")
        self.var_out_validation.set(res.validation_path or "")
        self.var_out_metrics.set(res.metrics_path or "")

        for item in self.tree.get_children():
            self.tree.delete(item)
        for row in res.unbalanced_preview:
            self.tree.insert("", "end", values=(
                row.get("源文件", ""),
                row.get("来源Sheet", ""),
                row.get("日期", ""),
                row.get("时间属性", ""),
                row.get("差额", ""),
                row.get("验证结果", ""),
            ))

    def _poll_queue(self) -> None:
        try:
            while True:
                item = self.ui_queue.get_nowait()
                if item.get("type") == "log":
                    self.logs_box.insert("end", item.get("message", "") + "\n")
                    self.logs_box.see("end")
                elif item.get("type") == "progress":
                    current = int(item.get("current", 0))
                    total = int(item.get("total", 1)) or 1
                    detail = str(item.get("detail", ""))
                    self.progress.set(min(1.0, max(0.0, current / total)))
                    self.var_status.set(f"{current}/{total} | {detail}")
                elif item.get("type") == "done":
                    res = item.get("result")
                    if isinstance(res, AnalysisResult):
                        self._apply_result(res)
                        if res.cancelled:
                            self.var_status.set("已取消")
                        elif res.errors:
                            self.var_status.set("完成（有错误）")
                        else:
                            self.var_status.set("完成")
                        self.tabview.set("结果")
                    self.btn_start.configure(state="normal")
                    self.btn_stop.configure(state="disabled")
        except queue.Empty:
            pass
        self.root.after(120, self._poll_queue)


class _WebEventHub:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subs: List["queue.Queue[Dict[str, Any]]"] = []

    def subscribe(self) -> "queue.Queue[Dict[str, Any]]":
        q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=2000)
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: "queue.Queue[Dict[str, Any]]") -> None:
        with self._lock:
            try:
                self._subs.remove(q)
            except ValueError:
                pass

    def publish(self, item: Dict[str, Any]) -> None:
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(item)
            except Exception:
                pass


class _WebQueueLogHandler(logging.Handler):
    def __init__(self, hub: _WebEventHub):
        super().__init__()
        self.hub = hub
        self.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self.hub.publish({"type": "log", "level": record.levelname, "message": msg})
        except Exception:
            pass


class _WebRunner:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.cfg = load_config(config_path)
        self.hub = _WebEventHub()
        self.cancel_event = threading.Event()
        self.worker_thread: Optional[threading.Thread] = None
        self.last_result: Optional[AnalysisResult] = None
        self.running = False
        self.current_progress: Dict[str, Any] = {"current": 0, "total": 1, "detail": ""}
        self._lock = threading.Lock()

    def set_config(self, data: Dict[str, Any]) -> AppConfig:
        cfg = AppConfig()
        for k, v in (data or {}).items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        if not getattr(cfg, "input_dir", None):
            cfg.input_dir = os.getcwd()
        if not getattr(cfg, "file_glob", None):
            cfg.file_glob = "*.xlsx"
        if not getattr(cfg, "output_dir", None):
            cfg.output_dir = os.getcwd()
        if not getattr(cfg, "output_basename", None):
            cfg.output_basename = OUTPUT_PATH
        try:
            cfg.validation_tolerance = float(getattr(cfg, "validation_tolerance", AppConfig().validation_tolerance))
        except Exception:
            cfg.validation_tolerance = AppConfig().validation_tolerance

        with self._lock:
            self.cfg = cfg
        save_config(self.config_path, cfg)
        self.hub.publish({"type": "status", "message": "已保存配置"})
        return cfg

    def get_config(self) -> AppConfig:
        with self._lock:
            return self.cfg

    def scan(self) -> List[str]:
        cfg = self.get_config()
        files = _list_excel_files(cfg)
        return files

    def start(self) -> bool:
        with self._lock:
            if self.worker_thread and self.worker_thread.is_alive():
                return False
            self.cancel_event.clear()
            self.running = True
            self.last_result = None
            self.current_progress = {"current": 0, "total": 1, "detail": ""}

        handler = _WebQueueLogHandler(self.hub)
        logger = _get_logger(handler=handler)

        def progress_cb(stage: str, current: int, total: int, detail: str) -> None:
            with self._lock:
                self.current_progress = {"stage": stage, "current": int(current), "total": int(total) or 1, "detail": str(detail)}
            self.hub.publish({"type": "progress", "stage": stage, "current": int(current), "total": int(total) or 1, "detail": str(detail)})

        def worker() -> None:
            try:
                cfg = self.get_config()
                res = analyze_directory(cfg, logger=logger, progress_cb=progress_cb, cancel_event=self.cancel_event)
            except Exception as e:
                res = AnalysisResult(errors=[str(e)])
            with self._lock:
                self.last_result = res
                self.running = False
            self.hub.publish({"type": "done", "result": asdict(res)})

        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()
        self.hub.publish({"type": "status", "message": "已开始运行"})
        return True

    def stop(self) -> None:
        self.cancel_event.set()
        self.hub.publish({"type": "status", "message": "正在停止..."})

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            res = asdict(self.last_result) if self.last_result is not None else None
            return {
                "running": bool(self.running),
                "progress": dict(self.current_progress),
                "result": res,
            }


def _web_index_html() -> str:
    try:
        from pathlib import Path

        p = Path(__file__).resolve().parent / "web" / "index.html"
        return p.read_text(encoding="utf-8")
    except Exception:
        return "<!doctype html><html><head><meta charset='utf-8'><title>Web</title></head><body>缺少 web/index.html</body></html>"


def run_web(config_path: str, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> int:
    try:
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
        from fastapi.staticfiles import StaticFiles
    except Exception:
        print("缺少依赖：fastapi。请先安装：py -m pip install fastapi uvicorn")
        return 1

    runner = _WebRunner(config_path=config_path)
    app = FastAPI()
    try:
        from pathlib import Path

        web_dir = Path(__file__).resolve().parent / "web"
        app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")
    except Exception:
        pass

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _web_index_html()

    @app.get("/api/config")
    def api_get_config() -> Dict[str, Any]:
        return asdict(runner.get_config())

    @app.post("/api/config")
    def api_set_config(data: Dict[str, Any]) -> Dict[str, Any]:
        cfg = runner.set_config(data)
        return asdict(cfg)

    @app.post("/api/scan")
    def api_scan() -> Dict[str, Any]:
        files = runner.scan()
        return {"files": files}

    @app.post("/api/run/start")
    def api_start() -> Dict[str, Any]:
        ok = runner.start()
        return {"ok": ok, "message": "已启动" if ok else "任务正在运行中"}

    @app.post("/api/run/stop")
    def api_stop() -> Dict[str, Any]:
        runner.stop()
        return {"ok": True}

    @app.get("/api/run/status")
    def api_status() -> Dict[str, Any]:
        return runner.get_status()

    @app.post("/api/open")
    def api_open(payload: Dict[str, Any]) -> Dict[str, Any]:
        path = str((payload or {}).get("path", "")).strip()
        if not path or not os.path.exists(path):
            return {"ok": False, "message": "路径不存在"}
        try:
            os.startfile(path)
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "message": str(e)}

    @app.get("/api/stream")
    def api_stream():
        import json as _json
        import time as _time

        q = runner.hub.subscribe()

        def encode(event: str, data_obj: Any) -> str:
            s = _json.dumps(data_obj, ensure_ascii=False)
            return f"event: {event}\ndata: {s}\n\n"

        def gen():
            try:
                yield encode("status", {"message": "已连接"})
                while True:
                    try:
                        item = q.get(timeout=0.8)
                    except queue.Empty:
                        st = runner.get_status()
                        yield encode("progress", st.get("progress", {}))
                        continue

                    t = str(item.get("type", "message"))
                    if t == "log":
                        yield encode("log", {"message": item.get("message", "")})
                    elif t == "progress":
                        yield encode("progress", item)
                    elif t == "status":
                        yield encode("status", {"message": item.get("message", "")})
                    elif t == "done":
                        yield encode("done", {"result": item.get("result", {})})
                    else:
                        yield encode("status", {"message": str(item)})
            finally:
                runner.hub.unsubscribe(q)

        return StreamingResponse(gen(), media_type="text/event-stream")

    if open_browser:
        try:
            import webbrowser
            webbrowser.open(f"http://{host}:{port}/")
        except Exception:
            pass

    try:
        import uvicorn
    except Exception:
        print("缺少依赖：uvicorn。请先安装：py -m pip install uvicorn")
        return 1

    uvicorn.run(app, host=host, port=int(port), log_level="info")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="财务数据清洗、验证与指标计算")
    parser.add_argument("--ui", action="store_true", help="启动图形界面")
    parser.add_argument("--web", action="store_true", help="启动本机Web界面")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="配置文件路径(JSON)")
    parser.add_argument("--input-dir", type=str, default=None, help="输入目录")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录")
    parser.add_argument("--glob", type=str, default=None, help="文件匹配模式，如 *.xlsx")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Web监听地址")
    parser.add_argument("--port", type=int, default=8765, help="Web端口")
    parser.add_argument("--no-browser", action="store_true", help="启动Web时不自动打开浏览器")
    args = parser.parse_args(argv)

    if args.ui:
        app = FinancialAnalyzerUI(config_path=args.config)
        app.run()
        return 0

    if args.web:
        return run_web(args.config, host=args.host, port=int(args.port), open_browser=not bool(args.no_browser))

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
