"""数据模型定义"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any
from decimal import Decimal
from datetime import date, datetime


@dataclass
class TextBlock:
    """OCR 识别的文本块"""
    text: str
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float


@dataclass
class OCRResult:
    """OCR 识别结果"""
    file_path: str
    file_type: str  # 'pdf', 'jpg', 'png'
    text_blocks: List[TextBlock] = field(default_factory=list)
    confidence: float = 0.0
    processing_time: float = 0.0
    error: Optional[str] = None


@dataclass
class InvoiceInfo:
    """发票信息"""
    invoice_id: str
    invoice_number: str
    amount: Decimal
    issue_date: date
    customer_name: str
    tax_number: Optional[str] = None
    source_file: str = ""
    extracted_at: datetime = field(default_factory=datetime.now)


@dataclass
class ContractInfo:
    """合同信息"""
    contract_id: str
    contract_number: str
    contract_amount: Decimal
    sign_date: date
    customer_name: str
    payment_terms: Optional[str] = None
    source_file: str = ""
    extracted_at: datetime = field(default_factory=datetime.now)


@dataclass
class ReceiptInfo:
    """银行回执信息"""
    receipt_id: str
    receipt_number: str
    received_amount: Decimal
    received_date: date
    payer_name: str
    bank_account: Optional[str] = None
    source_file: str = ""
    extracted_at: datetime = field(default_factory=datetime.now)


@dataclass
class PaymentMatch:
    """收款匹配记录"""
    match_id: str
    contract_id: str
    invoice_ids: List[str] = field(default_factory=list)
    receipt_ids: List[str] = field(default_factory=list)
    contract_amount: Decimal = Decimal('0')
    invoiced_amount: Decimal = Decimal('0')
    received_amount: Decimal = Decimal('0')
    status: str = 'unpaid'  # 'unpaid', 'partial', 'paid'
    payment_ratio: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class PaymentSummary:
    """收款统计摘要"""
    total_contracts: int = 0
    unpaid_count: int = 0
    partial_count: int = 0
    paid_count: int = 0
    total_contract_amount: Decimal = Decimal('0')
    total_received_amount: Decimal = Decimal('0')
    overall_payment_ratio: float = 0.0
    generated_at: datetime = field(default_factory=datetime.now)
