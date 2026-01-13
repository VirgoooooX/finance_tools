import os
import json
import re
import logging
import queue
import threading
from dataclasses import asdict
from typing import Optional, Any, Dict, List
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, ttk

from financial_analyzer_core import (
    OUTPUT_PATH,
    DEFAULT_CONFIG_PATH,
    AppConfig,
    AnalysisResult,
    load_config,
    save_config,
    analyze_directory,
    _get_logger,
    _list_excel_files,
    _safe_int_pair,
)


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


def main() -> int:
    app = FinancialAnalyzerUI(config_path=DEFAULT_CONFIG_PATH)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

