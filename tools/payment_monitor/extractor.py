"""信息提取器"""
import re
from typing import List, Optional
from decimal import Decimal
from datetime import datetime, date
from uuid import uuid4

from .models import OCRResult, InvoiceInfo, ContractInfo, ReceiptInfo


def extract_field_by_patterns(text: str, patterns: List[str]) -> Optional[str]:
    """
    使用正则模式列表提取字段
    
    参数:
        text: 文本内容
        patterns: 正则模式列表
    
    返回:
        提取的字段值，如果未找到返回 None
    """
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip() if match.groups() else match.group(0).strip()
    return None


def parse_date(date_str: str) -> Optional[date]:
    """
    解析日期字符串，支持多种格式
    
    参数:
        date_str: 日期字符串
    
    返回:
        date 对象，解析失败返回 None
    """
    date_formats = [
        '%Y-%m-%d',
        '%Y/%m/%d',
        '%Y年%m月%d日',
        '%Y.%m.%d',
        '%Y%m%d'
    ]
    
    for fmt in date_formats:
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    
    return None


def parse_amount(amount_str: str) -> Optional[Decimal]:
    """
    解析金额字符串，支持千分位和中文大写
    
    参数:
        amount_str: 金额字符串
    
    返回:
        Decimal 对象，解析失败返回 None
    """
    # 移除常见符号
    amount_str = amount_str.replace('¥', '').replace('￥', '').replace(',', '').replace('，', '').strip()
    
    # 尝试直接解析数字
    try:
        return Decimal(amount_str)
    except:
        pass
    
    # TODO: 支持中文大写金额（壹贰叁肆伍陆柒捌玖拾佰仟万亿元角分整）
    # 这里先返回 None，后续可以扩展
    
    return None


def normalize_name(name: str) -> str:
    """
    标准化客户名称
    
    参数:
        name: 原始名称
    
    返回:
        标准化后的名称
    """
    # 移除空白字符
    name = name.strip()
    
    # 移除常见后缀
    suffixes = ['有限公司', '有限责任公司', '股份有限公司', '集团', '公司', 'Co.', 'Ltd.', 'Inc.']
    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[:-len(suffix)].strip()
    
    # 统一大小写
    name = name.upper()
    
    return name


class InfoExtractor:
    """信息提取器"""
    
    def __init__(self):
        """初始化提取器"""
        # 发票号模式
        self.invoice_patterns = [
            r'发票号码[：:]\s*([A-Z0-9]+)',
            r'发票代码[：:]\s*([0-9]+)',
            r'NO[.：:]\s*([A-Z0-9]+)',
        ]
        
        # 合同编号模式
        self.contract_patterns = [
            r'合同编号[：:]\s*([A-Z0-9\-]+)',
            r'合同号[：:]\s*([A-Z0-9\-]+)',
            r'CONTRACT\s+NO[.：:]\s*([A-Z0-9\-]+)',
        ]
        
        # 回执号模式
        self.receipt_patterns = [
            r'回执号[：:]\s*([A-Z0-9]+)',
            r'交易流水号[：:]\s*([0-9]+)',
            r'凭证号[：:]\s*([0-9]+)',
        ]
        
        # 金额模式
        self.amount_patterns = [
            r'金额[：:]\s*[¥￥]?\s*([\d,，]+\.?\d*)',
            r'合计[：:]\s*[¥￥]?\s*([\d,，]+\.?\d*)',
            r'AMOUNT[：:]\s*[¥￥]?\s*([\d,，]+\.?\d*)',
        ]
        
        # 日期模式
        self.date_patterns = [
            r'日期[：:]\s*(\d{4}[-/年]\d{1,2}[-/月]\d{1,2}日?)',
            r'开票日期[：:]\s*(\d{4}[-/年]\d{1,2}[-/月]\d{1,2}日?)',
            r'签订日期[：:]\s*(\d{4}[-/年]\d{1,2}[-/月]\d{1,2}日?)',
            r'到账日期[：:]\s*(\d{4}[-/年]\d{1,2}[-/月]\d{1,2}日?)',
        ]
        
        # 客户名称模式
        self.customer_patterns = [
            r'购买方[：:]\s*([^\n]+)',
            r'客户名称[：:]\s*([^\n]+)',
            r'甲方[：:]\s*([^\n]+)',
            r'付款方[：:]\s*([^\n]+)',
        ]
    
    def extract_invoice(self, ocr_result: OCRResult) -> InvoiceInfo:
        """
        从 OCR 结果提取发票信息
        
        参数:
            ocr_result: OCR 识别结果
        
        返回:
            InvoiceInfo: 发票信息
        
        异常:
            ValueError: 缺少必填字段
        """
        # 合并所有文本
        full_text = '\n'.join([block.text for block in ocr_result.text_blocks])
        
        # 提取字段
        invoice_number = extract_field_by_patterns(full_text, self.invoice_patterns)
        amount_str = extract_field_by_patterns(full_text, self.amount_patterns)
        date_str = extract_field_by_patterns(full_text, self.date_patterns)
        customer_name = extract_field_by_patterns(full_text, self.customer_patterns)
        
        # 验证必填字段
        if not invoice_number:
            raise ValueError("缺少发票号码")
        if not amount_str:
            raise ValueError("缺少金额")
        if not date_str:
            raise ValueError("缺少日期")
        if not customer_name:
            raise ValueError("缺少客户名称")
        
        # 解析金额和日期
        amount = parse_amount(amount_str)
        if amount is None:
            raise ValueError(f"无法解析金额: {amount_str}")
        
        issue_date = parse_date(date_str)
        if issue_date is None:
            raise ValueError(f"无法解析日期: {date_str}")
        
        # 标准化客户名称
        customer_name = normalize_name(customer_name)
        
        return InvoiceInfo(
            invoice_id=str(uuid4()),
            invoice_number=invoice_number,
            amount=amount,
            issue_date=issue_date,
            customer_name=customer_name,
            source_file=ocr_result.file_path,
            extracted_at=datetime.now()
        )
    
    def extract_contract(self, ocr_result: OCRResult) -> ContractInfo:
        """
        从 OCR 结果提取合同信息
        
        参数:
            ocr_result: OCR 识别结果
        
        返回:
            ContractInfo: 合同信息
        
        异常:
            ValueError: 缺少必填字段
        """
        # 合并所有文本
        full_text = '\n'.join([block.text for block in ocr_result.text_blocks])
        
        # 提取字段
        contract_number = extract_field_by_patterns(full_text, self.contract_patterns)
        amount_str = extract_field_by_patterns(full_text, self.amount_patterns)
        date_str = extract_field_by_patterns(full_text, self.date_patterns)
        customer_name = extract_field_by_patterns(full_text, self.customer_patterns)
        
        # 验证必填字段
        if not contract_number:
            raise ValueError("缺少合同编号")
        if not amount_str:
            raise ValueError("缺少合同金额")
        if not date_str:
            raise ValueError("缺少签订日期")
        if not customer_name:
            raise ValueError("缺少客户名称")
        
        # 解析金额和日期
        contract_amount = parse_amount(amount_str)
        if contract_amount is None:
            raise ValueError(f"无法解析金额: {amount_str}")
        
        sign_date = parse_date(date_str)
        if sign_date is None:
            raise ValueError(f"无法解析日期: {date_str}")
        
        # 标准化客户名称
        customer_name = normalize_name(customer_name)
        
        return ContractInfo(
            contract_id=str(uuid4()),
            contract_number=contract_number,
            contract_amount=contract_amount,
            sign_date=sign_date,
            customer_name=customer_name,
            source_file=ocr_result.file_path,
            extracted_at=datetime.now()
        )
    
    def extract_receipt(self, ocr_result: OCRResult) -> ReceiptInfo:
        """
        从 OCR 结果提取回执信息
        
        参数:
            ocr_result: OCR 识别结果
        
        返回:
            ReceiptInfo: 回执信息
        
        异常:
            ValueError: 缺少必填字段
        """
        # 合并所有文本
        full_text = '\n'.join([block.text for block in ocr_result.text_blocks])
        
        # 提取字段
        receipt_number = extract_field_by_patterns(full_text, self.receipt_patterns)
        amount_str = extract_field_by_patterns(full_text, self.amount_patterns)
        date_str = extract_field_by_patterns(full_text, self.date_patterns)
        payer_name = extract_field_by_patterns(full_text, self.customer_patterns)
        
        # 验证必填字段
        if not receipt_number:
            raise ValueError("缺少回执号")
        if not amount_str:
            raise ValueError("缺少到账金额")
        if not date_str:
            raise ValueError("缺少到账日期")
        if not payer_name:
            raise ValueError("缺少付款方名称")
        
        # 解析金额和日期
        received_amount = parse_amount(amount_str)
        if received_amount is None:
            raise ValueError(f"无法解析金额: {amount_str}")
        
        received_date = parse_date(date_str)
        if received_date is None:
            raise ValueError(f"无法解析日期: {date_str}")
        
        # 标准化付款方名称
        payer_name = normalize_name(payer_name)
        
        return ReceiptInfo(
            receipt_id=str(uuid4()),
            receipt_number=receipt_number,
            received_amount=received_amount,
            received_date=received_date,
            payer_name=payer_name,
            source_file=ocr_result.file_path,
            extracted_at=datetime.now()
        )
