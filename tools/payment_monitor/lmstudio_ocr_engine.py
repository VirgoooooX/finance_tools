"""LM Studio 多模态 OCR 引擎"""
import os
import io
import time
import base64
import hashlib
import requests
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


class LMStudioOCREngine:
    """LM Studio 多模态 OCR 引擎"""
    
    def __init__(
        self,
        api_url: str = "http://localhost:11225/v1/chat/completions",
        model: str = "qwen/qwen3.5-9b",
        temperature: float = 0.1,
        max_tokens: int = 2000,
        enable_thinking: bool = False
    ):
        """
        初始化 LM Studio OCR 引擎
        
        参数:
            api_url: LM Studio API 地址
            model: 使用的模型名称
            temperature: 生成温度（越低越准确）
            max_tokens: 最大生成 token 数
            enable_thinking: 是否启用思考模式（OCR 建议关闭）
        """
        self.api_url = api_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        
        # 检查依赖
        if not PIL_AVAILABLE:
            self.initialized = False
            self.init_error = "PIL (Pillow) 未安装"
            return
        
        if not FITZ_AVAILABLE:
            self.initialized = False
            self.init_error = "PyMuPDF (fitz) 未安装"
            return
        
        # 测试连接
        try:
            response = requests.get(
                api_url.replace("/v1/chat/completions", "/v1/models"),
                timeout=5
            )
            if response.status_code == 200:
                self.initialized = True
                self.init_error = None
            else:
                self.initialized = False
                self.init_error = f"LM Studio 连接失败: HTTP {response.status_code}"
        except Exception as e:
            self.initialized = False
            self.init_error = f"LM Studio 连接失败: {str(e)}"
        
        # 初始化缓存
        self._cache = {}
        self._cache_enabled = True
    
    def _encode_image_to_base64(self, image) -> str:
        """将 PIL Image 编码为 base64"""
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def _call_lmstudio_api(self, image_base64: str) -> Optional[str]:
        """调用 LM Studio API 进行单张图片 OCR 识别"""
        return self._call_lmstudio_api_batch([image_base64])
    
    def _call_lmstudio_api_batch(self, image_base64_list: List[str]) -> Optional[str]:
        """调用 LM Studio API 进行批量图片 OCR 识别"""
        
        # 构建 prompt
        if len(image_base64_list) == 1:
            system_prompt = """你是一个专业的 OCR 文字识别助手。请识别图片中的所有文字内容。

要求：
1. 按照从上到下、从左到右的顺序识别
2. 保持原文的格式和结构
3. 准确识别所有文字，包括数字、标点符号
4. 不要添加任何解释、说明或额外内容
5. 只输出识别的文字内容

请开始识别："""
        else:
            system_prompt = f"""你是一个专业的 OCR 文字识别助手。我将提供 {len(image_base64_list)} 张图片，请依次识别每张图片中的所有文字内容。

要求：
1. 按照从上到下、从左到右的顺序识别
2. 保持原文的格式和结构
3. 准确识别所有文字，包括数字、标点符号
4. 不要添加任何解释、说明或额外内容
5. 每张图片的识别结果之间用 "---PAGE---" 分隔
6. 只输出识别的文字内容

请开始识别："""
        
        # 构建 content 列表
        content = [{"type": "text", "text": system_prompt}]
        
        # 添加所有图片
        for image_base64 in image_base64_list:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_base64}"
                }
            })
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens * len(image_base64_list),  # 根据图片数量调整 token 限制
            "stream": False,
            "thinking": {
                "type": "enabled" if self.enable_thinking else "disabled",
                "budget_tokens": 1000 if self.enable_thinking else 0
            }
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=120 * len(image_base64_list)  # 根据图片数量调整超时时间
            )
            
            if response.status_code == 200:
                result = response.json()
                text = result['choices'][0]['message']['content']
                return text.strip()
            else:
                return None
                
        except Exception as e:
            return None
    
    def convert_pdf_to_images(self, pdf_path: str, dpi: int = 150) -> List:
        """将 PDF 转换为图片列表"""
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
        """图片预处理"""
        if not PIL_AVAILABLE:
            return image
        
        # 转换为 RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 限制图片尺寸（避免 base64 过大）
        max_size = 2048
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        return image
    
    def recognize_file(self, file_path: str, use_cache: bool = True) -> OCRResult:
        """
        识别单个文件
        
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
                return OCRResult(
                    file_path=file_path,
                    file_type=cached_result.file_type,
                    text_blocks=cached_result.text_blocks,
                    confidence=cached_result.confidence,
                    processing_time=0.0
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
        
        try:
            # 转换 PDF 为图片
            images = self.convert_pdf_to_images(pdf_path)
            
            all_text = []
            
            # 识别每一页
            for img in images:
                img = self.preprocess_image(img)
                
                # 编码图片
                image_base64 = self._encode_image_to_base64(img)
                
                # 调用 LM Studio API
                text = self._call_lmstudio_api(image_base64)
                
                if text:
                    all_text.append(text)
            
            # 合并所有文本
            full_text = "\n\n".join(all_text)
            
            # 创建单个 TextBlock（LM Studio 返回的是完整文本，没有边界框）
            text_blocks = [TextBlock(
                text=full_text,
                bbox=(0, 0, 0, 0),  # 没有边界框信息
                confidence=0.9  # 假设置信度
            )]
            
            return OCRResult(
                file_path=pdf_path,
                file_type='pdf',
                text_blocks=text_blocks,
                confidence=0.9,
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
        
        try:
            # 加载图片
            img = Image.open(image_path)
            img = self.preprocess_image(img)
            
            # 编码图片
            image_base64 = self._encode_image_to_base64(img)
            
            # 调用 LM Studio API
            text = self._call_lmstudio_api(image_base64)
            
            if text:
                # 创建单个 TextBlock
                text_blocks = [TextBlock(
                    text=text,
                    bbox=(0, 0, 0, 0),
                    confidence=0.9
                )]
                
                ext = os.path.splitext(image_path)[1].lower()[1:]
                
                return OCRResult(
                    file_path=image_path,
                    file_type=ext,
                    text_blocks=text_blocks,
                    confidence=0.9,
                    processing_time=time.time() - start_time
                )
            else:
                return OCRResult(
                    file_path=image_path,
                    file_type=os.path.splitext(image_path)[1].lower()[1:],
                    error="LM Studio API 调用失败",
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
        max_workers: int = 1,  # LM Studio 通常不支持并发
        batch_size: int = 1  # 每次 API 调用处理的图片数量
    ) -> List[OCRResult]:
        """
        批量识别文件
        
        参数:
            file_paths: 文件路径列表
            progress_cb: 进度回调函数
            cancel_event: 取消事件
            db_conn: 数据库连接
            max_workers: 最大并行数（LM Studio 建议为 1）
            batch_size: 每次 API 调用处理的图片数量（建议 1-5）
        
        返回:
            识别结果列表
        """
        from threading import Event as ThreadEvent
        
        if progress_cb is None:
            progress_cb = lambda msg, cur, tot, stage: None
        
        if cancel_event is None:
            cancel_event = ThreadEvent()
        
        results = []
        total = len(file_paths)
        
        # 如果 batch_size > 1，使用批量识别
        if batch_size > 1:
            return self._recognize_batch_optimized(
                file_paths, progress_cb, cancel_event, db_conn, batch_size
            )
        
        # 否则使用原来的逐个识别
        for idx, file_path in enumerate(file_paths):
            # 检查取消事件
            if cancel_event.is_set():
                progress_cb(f"已取消，已处理 {idx}/{total} 个文件", idx, total, "cancelled")
                break
            
            # 计算文件哈希
            file_hash = self._calculate_file_hash(file_path)
            
            # 检查是否已处理过
            if db_conn and self._is_file_processed(db_conn, file_hash):
                progress_cb(f"跳过重复: {os.path.basename(file_path)}", idx + 1, total, "ocr")
                continue
            
            # 识别文件
            progress_cb(f"处理中: {os.path.basename(file_path)}", idx + 1, total, "ocr")
            result = self.recognize_file(file_path, use_cache=True)
            
            # 记录文件哈希
            if db_conn and not result.error:
                self._mark_file_processed(db_conn, file_hash, file_path)
            
            results.append(result)
        
        return results
    
    def _recognize_batch_optimized(
        self,
        file_paths: List[str],
        progress_cb: callable,
        cancel_event: Event,
        db_conn,
        batch_size: int
    ) -> List[OCRResult]:
        """
        优化的批量识别（一次 API 调用处理多张图片）
        
        参数:
            file_paths: 文件路径列表
            progress_cb: 进度回调函数
            cancel_event: 取消事件
            db_conn: 数据库连接
            batch_size: 每次 API 调用处理的图片数量
        
        返回:
            识别结果列表
        """
        results = []
        total = len(file_paths)
        
        # 按 batch_size 分组
        for batch_start in range(0, total, batch_size):
            # 检查取消事件
            if cancel_event.is_set():
                progress_cb(f"已取消，已处理 {batch_start}/{total} 个文件", batch_start, total, "cancelled")
                break
            
            batch_end = min(batch_start + batch_size, total)
            batch_paths = file_paths[batch_start:batch_end]
            
            progress_cb(
                f"批量处理中: {batch_start + 1}-{batch_end}/{total}",
                batch_end, total, "ocr"
            )
            
            # 批量识别
            batch_results = self._recognize_files_batch(batch_paths, db_conn)
            results.extend(batch_results)
        
        return results
    
    def _recognize_files_batch(self, file_paths: List[str], db_conn) -> List[OCRResult]:
        """
        一次 API 调用识别多个文件
        
        参数:
            file_paths: 文件路径列表
            db_conn: 数据库连接
        
        返回:
            识别结果列表
        """
        if not self.initialized:
            return [
                OCRResult(
                    file_path=fp,
                    file_type='unknown',
                    error=f"OCR 引擎初始化失败: {self.init_error}"
                )
                for fp in file_paths
            ]
        
        start_time = time.time()
        
        try:
            # 加载所有图片
            images = []
            valid_paths = []
            
            for file_path in file_paths:
                # 检查缓存
                file_hash = self._calculate_file_hash(file_path)
                if self._cache_enabled and file_hash in self._cache:
                    # 跳过已缓存的文件
                    continue
                
                # 检查是否已处理过
                if db_conn and self._is_file_processed(db_conn, file_hash):
                    continue
                
                ext = os.path.splitext(file_path)[1].lower()
                
                if ext == '.pdf':
                    # PDF 转图片
                    pdf_images = self.convert_pdf_to_images(file_path)
                    for img in pdf_images:
                        img = self.preprocess_image(img)
                        images.append(img)
                        valid_paths.append(file_path)
                elif ext in ['.jpg', '.jpeg', '.png']:
                    # 直接加载图片
                    img = Image.open(file_path)
                    img = self.preprocess_image(img)
                    images.append(img)
                    valid_paths.append(file_path)
            
            if not images:
                # 所有文件都已缓存或已处理
                return []
            
            # 编码所有图片
            image_base64_list = [self._encode_image_to_base64(img) for img in images]
            
            # 批量调用 API
            text = self._call_lmstudio_api_batch(image_base64_list)
            
            if not text:
                return [
                    OCRResult(
                        file_path=fp,
                        file_type=os.path.splitext(fp)[1].lower()[1:],
                        error="LM Studio API 调用失败",
                        processing_time=time.time() - start_time
                    )
                    for fp in valid_paths
                ]
            
            # 分割结果
            if len(images) > 1:
                # 多张图片，按分隔符分割
                texts = text.split("---PAGE---")
                # 确保结果数量匹配
                if len(texts) != len(images):
                    # 如果分割失败，尝试平均分配
                    texts = [text]
            else:
                texts = [text]
            
            # 创建结果
            results = []
            elapsed = time.time() - start_time
            
            for idx, (file_path, result_text) in enumerate(zip(valid_paths, texts)):
                text_blocks = [TextBlock(
                    text=result_text.strip(),
                    bbox=(0, 0, 0, 0),
                    confidence=0.9
                )]
                
                ext = os.path.splitext(file_path)[1].lower()[1:]
                
                result = OCRResult(
                    file_path=file_path,
                    file_type=ext,
                    text_blocks=text_blocks,
                    confidence=0.9,
                    processing_time=elapsed / len(valid_paths)  # 平均分配时间
                )
                
                # 缓存结果
                if self._cache_enabled:
                    file_hash = self._calculate_file_hash(file_path)
                    self._cache[file_hash] = result
                
                # 记录文件哈希
                if db_conn:
                    self._mark_file_processed(db_conn, file_hash, file_path)
                
                results.append(result)
            
            return results
            
        except Exception as e:
            return [
                OCRResult(
                    file_path=fp,
                    file_type=os.path.splitext(fp)[1].lower()[1:],
                    error=f"批量识别失败: {str(e)}",
                    processing_time=time.time() - start_time
                )
                for fp in file_paths
            ]
    
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
