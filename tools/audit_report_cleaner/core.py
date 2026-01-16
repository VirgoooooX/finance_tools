import os
import re
import logging
import pandas as pd
import threading
import sqlite3
from typing import Optional, Dict, List, Any, Callable
from dataclasses import asdict

from financial_analyzer_core import AppConfig, AnalysisResult, ProgressCallback
from financial_analyzer_core import _cleaned_sqlite_path_for as _cleaned_sqlite_path_for_common
from fa_platform.paths import (
    ensure_dir as _ensure_dir_common,
    default_output_root as _default_output_root_common,
    default_data_root as _default_data_root_common,
    resolve_under_base as _resolve_under_base_common,
    get_base_dir as _get_base_dir_common,
)
from fa_platform.jsonx import sanitize_json
from fa_platform.pipeline import build_artifacts as _build_artifacts_common, build_run_dir as _build_run_dir_common, write_sqlite_tables as _write_sqlite_tables_common

# --- Helpers ---

_TOOL_ID = os.path.basename(os.path.dirname(__file__))

def _ensure_dir(path: str) -> None:
    _ensure_dir_common(path)

def _default_output_root() -> str:
    return _default_output_root_common()

def _default_data_root() -> str:
    return _default_data_root_common()

def _resolve_under_base(path: str) -> str:
    return _resolve_under_base_common(path)

def _get_base_dir():
    return _get_base_dir_common()

def _get_logger(name: str = "audit_report_cleaner") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
    return logger

def _is_timestamp_folder(name: str) -> bool:
    import re
    return bool(re.match(r"^\d{8}_\d{6}$", str(name or "").strip()))

def _run_timestamp() -> str:
    import datetime
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def _cleaned_sqlite_path_for(cleaned_path: str) -> str:
    return _cleaned_sqlite_path_for_common(cleaned_path)

def _write_cleaned_sqlite(df: pd.DataFrame, sqlite_path: str) -> None:
    _write_sqlite_tables_common(sqlite_path, df)

def _list_excel_files(cfg: AppConfig) -> List[str]:
    import glob
    pattern = os.path.join(cfg.input_dir, cfg.file_glob)
    files = [os.path.abspath(p) for p in glob.glob(pattern) if os.path.isfile(p)]
    if cfg.exclude_output_files:
        exclude = {
            os.path.abspath(os.path.join(cfg.output_dir, cfg.output_basename)),
        }
        # Also exclude temporary files
        files = [p for p in files if os.path.abspath(p) not in exclude and not os.path.basename(p).startswith("~$")]
    files.sort(key=lambda p: p.lower())
    return files

def _tool_params(cfg: AppConfig) -> Dict[str, Any]:
    tp = getattr(cfg, "tool_params", None)
    if not isinstance(tp, dict):
        return {}
    bucket = tp.get(_TOOL_ID)
    return bucket if isinstance(bucket, dict) else {}

def _get_param(cfg: AppConfig, key: str, default: Any) -> Any:
    params = _tool_params(cfg)
    if key in params:
        return params.get(key)
    if hasattr(cfg, key):
        v = getattr(cfg, key)
        return default if v is None else v
    return default

# --- Audit Cleaning Logic ---

def get_file_date(filename):
    match = re.search(r'(20\d{2})', filename)
    if match:
        return f"{match.group(1)}-12-31"
    return "未知日期"

def get_report_type(sheet_name, cfg: AppConfig):
    name = str(sheet_name or "").upper()
    
    def _match(p):
        if isinstance(p, list):
            return any(str(x).upper() in name for x in p if x)
        return str(p or "").upper() in name if p else False

    try:
        if _match(_get_param(cfg, "sheet_keyword_bs", ["BS", "资产", "01"])): return "资产负债表"
        if _match(_get_param(cfg, "sheet_keyword_pl", ["PL", "利润", "损益", "02"])): return "利润表"
        if _match(_get_param(cfg, "sheet_keyword_cf", ["CF", "现金", "03"])): return "现金流量表"
    except Exception:
        pass

    return "其他报表"

def clean_amount(val):
    try:
        s = str(val).replace(',', '').replace('，', '').replace(' ', '')
        if s in ['-', '', 'nan', 'None']: return 0.0
        return float(s)
    except:
        return None

def _find_header_row(df: pd.DataFrame, keyword: str) -> Optional[int]:
    def norm(s: Any) -> str:
        t = str(s or "").strip().lower().replace("（", "(").replace("）", ")").replace("\u3000", " ")
        return re.sub(r"\s+", "", t)
    try:
        kw = norm(keyword)
        if not kw: return None
        max_rows = min(int(df.shape[0]), 120)
        sub = df.iloc[:max_rows]
        values = sub.to_numpy()
        for i in range(values.shape[0]):
            row = values[i]
            for j in range(row.shape[0]):
                if kw in norm(row[j]): return int(sub.index[i])
        return None
    except Exception: return None

def find_col(headers, keywords, start=0, end=999):
    limit = min(end, len(headers))
    for i in range(start, limit):
        h = str(headers[i])
        for kw in keywords:
            if kw in h: return i, h
    return None, None

def process_sheet(filename, file_date, sheet_name, df, cfg: AppConfig, logger=None) -> List[Dict[str, Any]]:
    rpt_type = get_report_type(sheet_name, cfg)
    if rpt_type == "其他报表": return []
    
    # Get column keywords from config or use defaults
    kw_subject = _get_param(cfg, "col_keyword_subject", ["资产", "项目", "科目", "摘要"]) or ["资产", "项目", "科目", "摘要"]
    kw_curr = _get_param(cfg, "col_keyword_period_end", ["期末", "本期", "本年", "金额"]) or ["期末", "本期", "本年", "金额"]
    kw_prev = _get_param(cfg, "col_keyword_period_start", ["上年", "上期", "年初"]) or ["上年", "上期", "年初"]
    if isinstance(kw_subject, str): kw_subject = [x.strip() for x in kw_subject.split(",")]
    if isinstance(kw_curr, str): kw_curr = [x.strip() for x in kw_curr.split(",")]
    if isinstance(kw_prev, str): kw_prev = [x.strip() for x in kw_prev.split(",")]
    
    header_idx = None
    try:
        if rpt_type == "资产负债表":
            header_idx = _find_header_row(df, str(_get_param(cfg, "header_keyword_bs", "") or ""))
        elif rpt_type == "利润表":
            header_idx = _find_header_row(df, str(_get_param(cfg, "header_keyword_pl", "") or ""))
        elif rpt_type == "现金流量表":
            header_idx = _find_header_row(df, str(_get_param(cfg, "header_keyword_cf", "") or ""))
    except Exception:
        header_idx = None
    if header_idx is None:
        tmp_idx = -1
        for idx, row in df.iterrows():
            if idx > 20: break
            txt = "".join([str(x) for x in row.values])
            if "余额" in txt or "金额" in txt:
                tmp_idx = idx
                break
        if tmp_idx == -1: return []
        header_idx = tmp_idx

    # 整理表头
    raw_headers = df.iloc[header_idx].fillna("").astype(str).tolist()
    headers = [h.replace("\n", "").replace(" ", "") for h in raw_headers]
    data_rows = df.iloc[header_idx + 1:]

    result_list = []
    tasks = []
    
    # === 规则 A: 资产负债表 (切分为左右两块) ===
    if rpt_type == "资产负债表":
        subj_L_idx, _ = find_col(headers, kw_subject)
        subj_R_idx, _ = find_col(headers, ["负债"], start=(subj_L_idx+1 if subj_L_idx is not None else 0))
        
        if subj_R_idx is None: subj_R_idx = 9999

        if subj_L_idx is not None:
            range_end = min(subj_R_idx, len(headers))
            idx_curr, n_curr = find_col(headers, kw_curr, subj_L_idx+1, range_end)
            if idx_curr: tasks.append({"大类": "资产", "科目列": subj_L_idx, "金额列": idx_curr, "属性": n_curr})
            idx_prev, n_prev = find_col(headers, kw_prev, subj_L_idx+1, range_end)
            if idx_prev: tasks.append({"大类": "资产", "科目列": subj_L_idx, "金额列": idx_prev, "属性": n_prev})

        if subj_R_idx != 9999:
            range_start = subj_R_idx + 1
            range_end = len(headers)
            idx_curr, n_curr = find_col(headers, kw_curr, range_start, range_end)
            if idx_curr: tasks.append({"大类": "负债和权益", "科目列": subj_R_idx, "金额列": idx_curr, "属性": n_curr})
            idx_prev, n_prev = find_col(headers, kw_prev, range_start, range_end)
            if idx_prev: tasks.append({"大类": "负债和权益", "科目列": subj_R_idx, "金额列": idx_prev, "属性": n_prev})

    # === 规则 B: 利润表 ===
    elif rpt_type == "利润表":
        subj_idx, _ = find_col(headers, kw_subject)
        if subj_idx is None: subj_idx = 0
        
        idx_curr, n_curr = find_col(headers, kw_curr, subj_idx+1)
        if idx_curr: tasks.append({"大类": "权益", "科目列": subj_idx, "金额列": idx_curr, "属性": n_curr})
        
        idx_prev, n_prev = find_col(headers, kw_prev, subj_idx+1)
        if idx_prev: tasks.append({"大类": "权益", "科目列": subj_idx, "金额列": idx_prev, "属性": n_prev})

    # === 规则 C: 现金流量表 ===
    elif rpt_type == "现金流量表":
        subj_idx, _ = find_col(headers, kw_subject)
        if subj_idx is None: subj_idx = 0
        
        idx_curr, n_curr = find_col(headers, kw_curr, subj_idx+1)
        if idx_curr: tasks.append({"大类": "现金流", "科目列": subj_idx, "金额列": idx_curr, "属性": n_curr})
        
        idx_prev, n_prev = find_col(headers, kw_prev, subj_idx+1)
        if idx_prev: tasks.append({"大类": "现金流", "科目列": subj_idx, "金额列": idx_prev, "属性": n_prev})

    for task in tasks:
        for _, row in data_rows.iterrows():
            try:
                s_val = row[task['科目列']]
                if pd.isna(s_val): continue
                subject = str(s_val).strip().replace(" ", "")
                if subject in ["", "nan", "None"]: continue
            except: continue

            try:
                amt = clean_amount(row[task['金额列']])
                if amt is None: continue
            except: continue

            result_list.append({
                "源文件": filename,
                "来源Sheet": sheet_name,
                "日期": file_date,
                "报表类型": rpt_type,
                "大类": task['大类'],
                "科目": subject,
                "时间属性": task['属性'],
                "金额": amt
            })
            
    if logger and result_list:
        logger.info(f"{rpt_type} 处理完成: {sheet_name}, 提取 {len(result_list)} 行")
        
    return result_list

def run_analysis(
    cfg: AppConfig,
    logger: Optional[logging.Logger] = None,
    progress_cb: Optional[ProgressCallback] = None,
    cancel_event: Optional[threading.Event] = None,
) -> AnalysisResult:
    result = AnalysisResult()
    if logger is None: logger = _get_logger()
    
    base_dir = os.path.abspath(_get_base_dir())
    output_root_raw = str(cfg.output_dir or "").strip() or _default_output_root()
    output_root = _resolve_under_base(output_root_raw)
    if output_root == base_dir: output_root = os.path.abspath(os.path.join(output_root, "output"))
    
    stamp = _run_timestamp()
    
    # Force tool_id to be derived from the module path to ensure correct output folder
    # even if config has a wrong default value.
    derived_id = os.path.basename(os.path.dirname(__file__))
    tool_id = str(getattr(cfg, "tool_id", "")).strip()
    if not tool_id or tool_id == "monthly_report_cleaner":
        tool_id = derived_id
        
    _, run_dir = _build_run_dir_common(output_root_raw, tool_id, stamp=stamp)
    
    _ensure_dir(run_dir)
    _ensure_dir(_default_data_root())
    
    files = _list_excel_files(cfg)
    result.found_files = files
    
    if not files:
        logger.warning("未找到任何匹配的 .xlsx 文件")
        return result
        
    logger.info(f"找到 {len(files)} 个Excel文件")
    all_data_list = []
    
    for idx, file_path in enumerate(files, start=1):
        if cancel_event and cancel_event.is_set():
            result.cancelled = True; logger.warning("已取消运行"); return result
        if progress_cb: progress_cb("file", idx, len(files), os.path.basename(file_path))
        
        filename = os.path.basename(file_path)
        logger.info(f"正在处理: {filename}")
        file_date = get_file_date(filename)
        
        try:
            excel_file = pd.ExcelFile(file_path)
            try:
                for sheet_name in excel_file.sheet_names:
                    if cancel_event and cancel_event.is_set(): result.cancelled = True; return result
                    df = excel_file.parse(sheet_name=sheet_name, header=None)
                    sheet_data = process_sheet(filename, file_date, sheet_name, df, cfg, logger)
                    all_data_list.extend(sheet_data)
                result.processed_files += 1
            finally:
                excel_file.close()
        except Exception as e:
            logger.error(f"读取失败: {e}")
            result.errors.append(f"{filename}: {e}")

    if cancel_event and cancel_event.is_set(): result.cancelled = True; return result
    if not all_data_list: return result
    
    all_data = pd.DataFrame(all_data_list)
    cols = ["源文件", "来源Sheet", "日期", "报表类型", "大类", "科目", "时间属性", "金额"]
    # Ensure all cols exist
    for c in cols:
        if c not in all_data.columns:
            all_data[c] = None
    all_data = all_data[cols]
    
    cleaned_path = os.path.abspath(os.path.join(run_dir, cfg.output_basename))
    all_data.to_excel(cleaned_path, index=False)
    result.cleaned_path = cleaned_path
    result.cleaned_rows = int(len(all_data))
    
    cleaned_sqlite_path = _cleaned_sqlite_path_for(cleaned_path)
    result.cleaned_sqlite_path = cleaned_sqlite_path
    _write_cleaned_sqlite(all_data, cleaned_sqlite_path)
    
    result.artifacts = _build_artifacts_common(
        cleaned_path=str(result.cleaned_path or ""),
        cleaned_sqlite_path=str(result.cleaned_sqlite_path or ""),
        validation_path=str(result.validation_path or ""),
        metrics_path=str(result.metrics_path or ""),
    )
    return result
