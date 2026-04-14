"""OCR 识别引擎"""
import os
import io
import time
import hashlib
from typing import List, Optional, Any
from threading import Event

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False

from .models import OCRResult, TextBlock


class OCREngine:
    """OCR 识别引擎（优化版）"""
    
    def __init__(self, use_gpu: bool = False, lang: str = 'ch', use_lightweight: bool = True):
        """
        初始化 OCR 引擎
        
        参数:
            use_gpu: 是否使用 GPU
            lang: 语言（'ch' 中文, 'en' 英文）
            use_lightweight: 是否使用轻量级模型（更快但精度略低）
        """
        if not PIL_AVAILABLE:
            self.ocr = None
            self.initialized = False
            self.init_error = "PIL (Pillow) 未安装"
            return
        
        if not FITZ_AVAILABLE:
            self.ocr = None
            self.initialized = False
            self.init_error = "PyMuPDF (fitz) 未安装"
            return
        
        try:
            from paddleocr import PaddleOCR
            # 使用新版 PaddleOCR API，优化性能
            ocr_params = {
                'use_textline_orientation': False,  # 禁用文本方向检测以提速
                'lang': lang,
                'enable_mkldnn': False  # 禁用 oneDNN 以避免兼容性问题
            }
            
            # 使用轻量级模型以提升速度
            if use_lightweight:
                ocr_params['det_model_dir'] = None  # 使用默认轻量级检测模型
                ocr_params['rec_model_dir'] = None  # 使用默认轻量级识别模型
            
            # 尝试添加 GPU 参数（如果支持）
            if use_gpu:
                try:
                    self.ocr = PaddleOCR(**ocr_params, use_gpu=True)
                except TypeError:
                    # 如果不支持 use_gpu 参数，使用默认配置
                    self.ocr = PaddleOCR(**ocr_params)
            else:
                self.ocr = PaddleOCR(**ocr_params)
            
            self.initialized = True
            self.init_error = None
            
            # 初始化缓存
            self._cache = {}
            self._cache_enabled = True
            
        except Exception as e:
            self.ocr = None
            self.initialized = False
            self.init_error = str(e)
    
    def convert_pdf_to_images(self, pdf_path: str, dpi: int = 150) -> List:
        """
        将 PDF 转换为图片列表（优化版：降低 DPI 以提速）
        
        参数:
            pdf_path: PDF 文件路径
            dpi: 分辨率（默认 150，降低以提速）
        
        返回:
            图片列表
        """
        if not PIL_AVAILABLE or not FITZ_AVAILABLE:
            return []
        
        images = []
        doc = fitz.open(pdf_path)
        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                mat = fitz.Matrix(dpi / 72, dpi / 72)
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                images.append(img)
        finally:
            doc.close()
        return images
    
    def preprocess_image(self, image) -> Any:
        """
        图片预处理（优化版：更激进的尺寸压缩）
        
        参数:
            image: 原始图片
        
        返回:
            预处理后的图片
        """
        if not PIL_AVAILABLE:
            return image
        
        # 转换为 RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 更激进的尺寸压缩以提速（降低到 1500）
        max_size = 1500
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            # 使用更快的 BILINEAR 插值
            image = image.resize(new_size, Image.Resampling.BILINEAR)
        
        return image
    
    def recognize_file(self, file_path: str, use_cache: bool = True) -> OCRResult:
        """
        识别单个文件（支持缓存）
        
        参数:
            file_path: 文件路径
            use_cache: 是否使用缓存
        
        返回:
            OCRResult: 识别结果
        """
        # 检查缓存
        if use_cache and self._cache_enabled:
            file_hash = self._calculate_file_hash(file_path)
            if file_hash in self._cache:
                cached_result = self._cache[file_hash]
                # 返回缓存结果的副本
                return OCRResult(
                    file_path=file_path,
                    file_type=cached_result.file_type,
                    text_blocks=cached_result.text_blocks,
                    confidence=cached_result.confidence,
                    processing_time=0.0  # 缓存命中，处理时间为 0
                )
        
        start_time = time.time()
        
        if not self.initialized:
            return OCRResult(
                file_path=file_path,
                file_type='unknown',
                error=f"OCR 引擎初始化失败: {self.init_error}"
            )
        
        try:
            # 判断文件类型
            ext = os.path.splitext(file_path)[1].lower()
            
            if ext == '.pdf':
                result = self._recognize_pdf(file_path, start_time)
            elif ext in ['.jpg', '.jpeg', '.png']:
                result = self._recognize_image(file_path, start_time)
            else:
                result = OCRResult(
                    file_path=file_path,
                    file_type='unknown',
                    error=f"不支持的文件类型: {ext}"
                )
            
            # 缓存成功的结果
            if use_cache and self._cache_enabled and not result.error:
                file_hash = self._calculate_file_hash(file_path)
                self._cache[file_hash] = result
            
            return result
        
        except Exception as e:
            return OCRResult(
                file_path=file_path,
                file_type=ext[1:] if ext else 'unknown',
                error=f"识别失败: {str(e)}",
                processing_time=time.time() - start_time
            )
    
    def _recognize_pdf(self, pdf_path: str, start_time: float) -> OCRResult:
        """识别 PDF 文件"""
        if not PIL_AVAILABLE or not FITZ_AVAILABLE:
            return OCRResult(
                file_path=pdf_path,
                file_type='pdf',
                error="PIL 或 PyMuPDF 未安装"
            )
        
        text_blocks = []
        total_confidence = 0.0
        block_count = 0
        
        try:
            # 转换 PDF 为图片
            images = self.convert_pdf_to_images(pdf_path)
            
            # 识别每一页
            for img in images:
                img = self.preprocess_image(img)
                
                # 转换为 numpy array
                import numpy as np
                img_array = np.array(img)
                
                # OCR 识别 - 使用新版 API
                result = self.ocr.predict(img_array)
                
                if result and len(result) > 0:
                    ocr_result = result[0]
                    
                    # 提取文本、分数和边界框
                    rec_texts = ocr_result.get('rec_texts', [])
                    rec_scores = ocr_result.get('rec_scores', [])
                    dt_polys = ocr_result.get('dt_polys', [])
                    
                    for text, score, poly in zip(rec_texts, rec_scores, dt_polys):
                        try:
                            # 转换 bbox 格式
                            x_coords = [p[0] for p in poly]
                            y_coords = [p[1] for p in poly]
                            bbox_tuple = (
                                int(min(x_coords)),
                                int(min(y_coords)),
                                int(max(x_coords)),
                                int(max(y_coords))
                            )
                            
                            text_blocks.append(TextBlock(
                                text=text,
                                bbox=bbox_tuple,
                                confidence=float(score)
                            ))
                            
                            total_confidence += float(score)
                            block_count += 1
                        except (IndexError, TypeError, KeyError, ValueError) as e:
                            # 跳过格式不正确的行
                            continue
            
            avg_confidence = total_confidence / block_count if block_count > 0 else 0.0
            
            return OCRResult(
                file_path=pdf_path,
                file_type='pdf',
                text_blocks=text_blocks,
                confidence=avg_confidence,
                processing_time=time.time() - start_time
            )
        except Exception as e:
            return OCRResult(
                file_path=pdf_path,
                file_type='pdf',
                error=f"识别失败: {str(e)}",
                processing_time=time.time() - start_time
            )
    
    def _recognize_image(self, image_path: str, start_time: float) -> OCRResult:
        """识别图片文件"""
        if not PIL_AVAILABLE:
            return OCRResult(
                file_path=image_path,
                file_type='unknown',
                error="PIL (Pillow) 未安装"
            )
        
        text_blocks = []
        
        try:
            # 加载图片
            img = Image.open(image_path)
            img = self.preprocess_image(img)
            
            # 转换为 numpy array
            import numpy as np
            img_array = np.array(img)
            
            # OCR 识别 - 使用新版 API
            result = self.ocr.predict(img_array)
            
            total_confidence = 0.0
            block_count = 0
            
            if result and len(result) > 0:
                ocr_result = result[0]
                
                # 提取文本、分数和边界框
                rec_texts = ocr_result.get('rec_texts', [])
                rec_scores = ocr_result.get('rec_scores', [])
                dt_polys = ocr_result.get('dt_polys', [])
                
                for text, score, poly in zip(rec_texts, rec_scores, dt_polys):
                    try:
                        # 转换 bbox 格式
                        x_coords = [p[0] for p in poly]
                        y_coords = [p[1] for p in poly]
                        bbox_tuple = (
                            int(min(x_coords)),
                            int(min(y_coords)),
                            int(max(x_coords)),
                            int(max(y_coords))
                        )
                        
                        text_blocks.append(TextBlock(
                            text=text,
                            bbox=bbox_tuple,
                            confidence=float(score)
                        ))
                        
                        total_confidence += float(score)
                        block_count += 1
                    except (IndexError, TypeError, KeyError, ValueError) as e:
                        # 跳过格式不正确的行
                        continue
            
            avg_confidence = total_confidence / block_count if block_count > 0 else 0.0
            ext = os.path.splitext(image_path)[1].lower()[1:]
            
            return OCRResult(
                file_path=image_path,
                file_type=ext,
                text_blocks=text_blocks,
                confidence=avg_confidence,
                processing_time=time.time() - start_time
            )
        except Exception as e:
            return OCRResult(
                file_path=image_path,
                file_type=os.path.splitext(image_path)[1].lower()[1:],
                error=f"识别失败: {str(e)}",
                processing_time=time.time() - start_time
            )

    def recognize_batch(
        self,
        file_paths: List[str],
        progress_cb: Optional[callable] = None,
        cancel_event: Optional[Event] = None,
        db_conn = None,
        max_workers: int = 4
    ) -> List[OCRResult]:
        """
        批量识别文件（支持多线程并行）
        
        参数:
            file_paths: 文件路径列表
            progress_cb: 进度回调函数 (message, current, total, stage)
            cancel_event: 取消事件
            db_conn: 数据库连接（用于去重检查）
            max_workers: 最大并行线程数（默认 4）
        
        返回:
            识别结果列表
        """
        from threading import Event as ThreadEvent, Lock
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        if progress_cb is None:
            progress_cb = lambda msg, cur, tot, stage: None
        
        if cancel_event is None:
            cancel_event = ThreadEvent()
        
        results = []
        total = len(file_paths)
        completed = 0
        lock = Lock()
        
        def process_file(file_path: str):
            """处理单个文件"""
            nonlocal completed
            
            # 检查取消事件
            if cancel_event.is_set():
                return None
            
            # 计算文件哈希
            file_hash = self._calculate_file_hash(file_path)
            
            # 检查是否已处理过（去重）
            if db_conn:
                with lock:
                    if self._is_file_processed(db_conn, file_hash):
                        with lock:
                            completed += 1
                            progress_cb(f"Skip duplicate: {os.path.basename(file_path)}", completed, total, "ocr")
                        return None
            
            # 识别文件
            with lock:
                completed += 1
                progress_cb(f"Processing: {os.path.basename(file_path)}", completed, total, "ocr")
            
            result = self.recognize_file(file_path, use_cache=True)
            
            # 记录文件哈希
            if db_conn and not result.error:
                with lock:
                    self._mark_file_processed(db_conn, file_hash, file_path)
            
            return result
        
        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_path = {executor.submit(process_file, fp): fp for fp in file_paths}
            
            # 收集结果
            for future in as_completed(future_to_path):
                if cancel_event.is_set():
                    progress_cb(f"Cancelled, processed {completed}/{total} files", completed, total, "cancelled")
                    break
                
                result = future.result()
                if result is not None:
                    results.append(result)
        
        return results
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """计算文件 SHA256 哈希"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def _is_file_processed(self, conn, file_hash: str) -> bool:
        """检查文件是否已处理过"""
        cur = conn.execute(
            "SELECT 1 FROM file_hashes WHERE file_hash = ?",
            (file_hash,)
        )
        return cur.fetchone() is not None
    
    def _mark_file_processed(self, conn, file_hash: str, file_path: str) -> None:
        """标记文件已处理"""
        from datetime import datetime
        conn.execute(
            "INSERT OR REPLACE INTO file_hashes (file_hash, file_path, processed_at) VALUES (?, ?, ?)",
            (file_hash, file_path, datetime.now().isoformat())
        )
        conn.commit()
