"""收款进度监控工具核心逻辑"""
import os
import glob
import logging
import sqlite3
from typing import Optional, Callable, Any
from threading import Event
from datetime import datetime

from financial_analyzer_core import AnalysisResult, AppConfig
from fa_platform.pipeline import build_run_dir, build_artifacts
from fa_platform.run_index import upsert_run_from_result

from .database import create_database_schema, insert_contract, insert_invoice, insert_receipt, insert_payment_match
from .ocr_engine import OCREngine
from .extractor import InfoExtractor
from .matching import MatchingEngine
from .monitor import ProgressMonitor


def run_analysis(
    cfg: AppConfig,
    logger: Optional[logging.Logger] = None,
    progress_cb: Optional[Callable[[str, int, int, str], None]] = None,
    cancel_event: Optional[Event] = None
) -> AnalysisResult:
    """
    收款进度监控主函数（骨架版本）
    
    参数:
        cfg: 应用配置
        logger: 日志记录器
        progress_cb: 进度回调函数 (message, current, total, stage)
        cancel_event: 取消事件
    
    返回:
        AnalysisResult: 分析结果
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if progress_cb is None:
        progress_cb = lambda msg, cur, tot, stage: None
    
    if cancel_event is None:
        cancel_event = Event()
    
    logger.info("收款进度监控工具启动")
    progress_cb("初始化...", 0, 100, "init")
    
    errors = []
    
    try:
        # 创建输出目录
        ts, run_dir = build_run_dir(cfg.output_dir, "payment_monitor")
        run_id = ts
        
        # 创建数据库
        db_path = os.path.join("data", "warehouse.sqlite")
        create_database_schema(db_path)
        logger.info(f"数据库初始化完成: {db_path}")
        
        progress_cb("数据库初始化完成", 5, 100, "init")
        
        # 获取工具参数
        tool_params = cfg.tool_params.get("payment_monitor", {})
        ocr_engine_type = tool_params.get("ocr_engine", "paddleocr")  # paddleocr 或 lmstudio
        use_gpu = tool_params.get("ocr_use_gpu", False)
        lang = tool_params.get("ocr_lang", "ch")
        use_lightweight = tool_params.get("ocr_use_lightweight", True)
        max_workers = tool_params.get("ocr_max_workers", 4)
        tolerance = tool_params.get("tolerance", 0.01)
        name_threshold = tool_params.get("name_similarity_threshold", 0.8)
        max_file_size = tool_params.get("max_file_size_mb", 50) * 1024 * 1024
        
        # 扫描文件
        if not cfg.input_dir or not os.path.exists(cfg.input_dir):
            raise ValueError(f"输入目录不存在: {cfg.input_dir}")
        
        pattern = os.path.join(cfg.input_dir, "**", "*")
        all_files = glob.glob(pattern, recursive=True)
        
        # 过滤文件类型和大小
        valid_files = []
        for file_path in all_files:
            if not os.path.isfile(file_path):
                continue
            
            ext = os.path.splitext(file_path)[1].lower()
            if ext not in ['.pdf', '.jpg', '.jpeg', '.png']:
                continue
            
            file_size = os.path.getsize(file_path)
            if file_size > max_file_size:
                errors.append(f"文件过大，已跳过: {file_path} ({file_size / 1024 / 1024:.1f} MB)")
                continue
            
            valid_files.append(file_path)
        
        logger.info(f"找到 {len(valid_files)} 个有效文件")
        progress_cb(f"找到 {len(valid_files)} 个文件", 10, 100, "scan")
        
        if not valid_files:
            raise ValueError("未找到有效的文件")
        
        # 初始化 OCR 引擎
        if ocr_engine_type == "lmstudio":
            from .lmstudio_ocr_engine import LMStudioOCREngine
            
            lmstudio_api_url = tool_params.get("lmstudio_api_url", "http://localhost:11225/v1/chat/completions")
            lmstudio_model = tool_params.get("lmstudio_model", "qwen/qwen3.5-9b")
            lmstudio_temperature = tool_params.get("lmstudio_temperature", 0.1)
            lmstudio_max_tokens = tool_params.get("lmstudio_max_tokens", 2000)
            lmstudio_enable_thinking = tool_params.get("lmstudio_enable_thinking", False)
            lmstudio_batch_size = tool_params.get("lmstudio_batch_size", 1)
            
            ocr_engine = LMStudioOCREngine(
                api_url=lmstudio_api_url,
                model=lmstudio_model,
                temperature=lmstudio_temperature,
                max_tokens=lmstudio_max_tokens,
                enable_thinking=lmstudio_enable_thinking
            )
            logger.info(f"使用 LM Studio OCR 引擎: {lmstudio_model} (thinking={lmstudio_enable_thinking}, batch_size={lmstudio_batch_size})")
        else:
            # 默认使用 PaddleOCR
            lmstudio_batch_size = 1  # PaddleOCR 不支持批量
            ocr_engine = OCREngine(use_gpu=use_gpu, lang=lang, use_lightweight=use_lightweight)
            logger.info(f"使用 PaddleOCR 引擎: GPU={use_gpu}")
        
        if not ocr_engine.initialized:
            raise RuntimeError(f"OCR 引擎初始化失败: {ocr_engine.init_error}")
        
        progress_cb("OCR 引擎初始化完成", 15, 100, "init")
        
        # 批量识别（使用多线程）
        conn = sqlite3.connect(db_path)
        try:
            ocr_results = ocr_engine.recognize_batch(
                valid_files,
                progress_cb=lambda msg, cur, tot, stage: progress_cb(msg, 15 + int(cur / tot * 40), 100, "ocr"),
                cancel_event=cancel_event,
                db_conn=conn,
                max_workers=max_workers,
                batch_size=lmstudio_batch_size if ocr_engine_type == "lmstudio" else 1
            )
            
            logger.info(f"OCR 识别完成，共 {len(ocr_results)} 个结果")
            progress_cb("OCR 识别完成", 55, 100, "ocr")
            
            # 提取信息
            extractor = InfoExtractor()
            contracts = []
            invoices = []
            receipts = []
            
            for idx, ocr_result in enumerate(ocr_results):
                if cancel_event.is_set():
                    break
                
                if ocr_result.error:
                    errors.append(f"OCR 识别失败: {ocr_result.file_path} - {ocr_result.error}")
                    continue
                
                # 根据文件名判断文档类型
                filename = os.path.basename(ocr_result.file_path).lower()
                
                try:
                    if '合同' in filename or 'contract' in filename:
                        contract = extractor.extract_contract(ocr_result)
                        contracts.append(contract)
                        insert_contract(conn, contract)
                    elif '发票' in filename or 'invoice' in filename:
                        invoice = extractor.extract_invoice(ocr_result)
                        invoices.append(invoice)
                        insert_invoice(conn, invoice)
                    elif '回执' in filename or 'receipt' in filename or '银行' in filename:
                        receipt = extractor.extract_receipt(ocr_result)
                        receipts.append(receipt)
                        insert_receipt(conn, receipt)
                    else:
                        errors.append(f"无法判断文档类型: {ocr_result.file_path}")
                
                except Exception as e:
                    errors.append(f"信息提取失败: {ocr_result.file_path} - {str(e)}")
                
                progress_cb(f"信息提取中 ({idx + 1}/{len(ocr_results)})", 55 + int((idx + 1) / len(ocr_results) * 20), 100, "extract")
            
            conn.commit()
            logger.info(f"信息提取完成: {len(contracts)} 个合同, {len(invoices)} 个发票, {len(receipts)} 个回执")
            progress_cb("信息提取完成", 75, 100, "extract")
            
            # 执行三方匹配
            matching_engine = MatchingEngine(tolerance=tolerance, name_similarity_threshold=name_threshold)
            matches = matching_engine.match_three_way(contracts, invoices, receipts)
            
            # 写入匹配结果
            for match in matches:
                insert_payment_match(conn, match)
            
            conn.commit()
            logger.info(f"匹配完成，共 {len(matches)} 条匹配记录")
            progress_cb("匹配完成", 90, 100, "match")
            
            # 生成报告
            report_path = os.path.join(run_dir, cfg.output_basename)
            monitor = ProgressMonitor(db_path)
            monitor.export_report(report_path)
            
            logger.info(f"报告生成完成: {report_path}")
            progress_cb("报告生成完成", 100, 100, "complete")
            
            # 构建产物列表
            artifacts = build_artifacts(
                cleaned_path=report_path,
                cleaned_sqlite_path=db_path,
                validation_path=None,
                metrics_path=None
            )
            
            # 构建结果
            result = AnalysisResult(
                run_id=run_id,
                cleaned_path=report_path,
                cleaned_sqlite_path=db_path,
                validation_path=None,
                metrics_path=None,
                artifacts=artifacts,
                cleaned_rows=len(matches),
                processed_files=len(valid_files),
                errors=errors
            )
            
            # 记录到 Run Index
            upsert_run_from_result("payment_monitor", result, cfg)
            
            logger.info("收款进度监控工具完成")
            return result
        
        finally:
            conn.close()
        
    except Exception as e:
        logger.error(f"处理失败: {e}", exc_info=True)
        return AnalysisResult(
            run_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
            cleaned_path=None,
            cleaned_sqlite_path=None,
            validation_path=None,
            metrics_path=None,
            artifacts=[],
            cleaned_rows=0,
            processed_files=0,
            errors=[str(e)]
        )
