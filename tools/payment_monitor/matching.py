"""智能匹配引擎"""
from typing import List, Dict, Optional, Tuple
from decimal import Decimal
from datetime import datetime
from uuid import uuid4

from .models import InvoiceInfo, ContractInfo, ReceiptInfo, PaymentMatch


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    计算两个字符串的编辑距离
    
    参数:
        s1: 字符串 1
        s2: 字符串 2
    
    返回:
        编辑距离
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def calculate_similarity(name1: str, name2: str) -> float:
    """
    计算两个名称的相似度
    
    参数:
        name1: 名称 1
        name2: 名称 2
    
    返回:
        相似度 [0.0, 1.0]
    """
    if not name1 or not name2:
        return 0.0
    
    distance = levenshtein_distance(name1, name2)
    max_len = max(len(name1), len(name2))
    
    if max_len == 0:
        return 1.0
    
    return 1.0 - (distance / max_len)


def is_name_similar(name1: str, name2: str, threshold: float = 0.8) -> bool:
    """
    判断两个名称是否相似
    
    参数:
        name1: 名称 1
        name2: 名称 2
        threshold: 相似度阈值
    
    返回:
        是否相似
    """
    similarity = calculate_similarity(name1, name2)
    return similarity >= threshold


def find_best_match(target_name: str, candidates: List[str], threshold: float = 0.8) -> Optional[Tuple[str, float]]:
    """
    从候选列表中找到最佳匹配
    
    参数:
        target_name: 目标名称
        candidates: 候选名称列表
        threshold: 相似度阈值
    
    返回:
        (最佳匹配名称, 相似度) 或 None
    """
    best_match = None
    best_similarity = 0.0
    
    for candidate in candidates:
        similarity = calculate_similarity(target_name, candidate)
        if similarity >= threshold and similarity > best_similarity:
            best_match = candidate
            best_similarity = similarity
    
    return (best_match, best_similarity) if best_match else None


def is_amount_within_tolerance(amount1: Decimal, amount2: Decimal, tolerance: float = 0.01) -> bool:
    """
    检查两个金额是否在容差范围内
    
    参数:
        amount1: 金额 1
        amount2: 金额 2
        tolerance: 容差比例（默认 1%）
    
    返回:
        是否在容差范围内
    """
    if amount1 == 0 and amount2 == 0:
        return True
    
    max_amount = max(amount1, amount2)
    if max_amount == 0:
        return False
    
    diff_ratio = abs(amount1 - amount2) / max_amount
    return diff_ratio <= tolerance


class MatchingEngine:
    """智能匹配引擎"""
    
    def __init__(self, tolerance: float = 0.01, name_similarity_threshold: float = 0.8):
        """
        初始化匹配引擎
        
        参数:
            tolerance: 金额容差比例
            name_similarity_threshold: 名称相似度阈值
        """
        self.tolerance = tolerance
        self.name_similarity_threshold = name_similarity_threshold
    
    def match_invoice_contract(
        self,
        invoice: InvoiceInfo,
        contracts: List[ContractInfo]
    ) -> Optional[ContractInfo]:
        """
        匹配发票与合同
        
        参数:
            invoice: 发票信息
            contracts: 合同列表
        
        返回:
            匹配的合同，如果未找到返回 None
        """
        # 先按名称筛选候选合同
        candidates = []
        for contract in contracts:
            if is_name_similar(invoice.customer_name, contract.customer_name, self.name_similarity_threshold):
                candidates.append(contract)
        
        if not candidates:
            return None
        
        # 在候选合同中找金额匹配的
        for contract in candidates:
            if is_amount_within_tolerance(invoice.amount, contract.contract_amount, self.tolerance):
                return contract
        
        # 如果没有金额完全匹配的，返回名称最相似的
        best_match = None
        best_similarity = 0.0
        for contract in candidates:
            similarity = calculate_similarity(invoice.customer_name, contract.customer_name)
            if similarity > best_similarity:
                best_match = contract
                best_similarity = similarity
        
        return best_match
    
    def match_receipt_invoice(
        self,
        receipt: ReceiptInfo,
        invoices: List[InvoiceInfo]
    ) -> Optional[InvoiceInfo]:
        """
        匹配回执与发票
        
        参数:
            receipt: 回执信息
            invoices: 发票列表
        
        返回:
            匹配的发票，如果未找到返回 None
        """
        # 先按名称筛选候选发票
        candidates = []
        for invoice in invoices:
            if is_name_similar(receipt.payer_name, invoice.customer_name, self.name_similarity_threshold):
                candidates.append(invoice)
        
        if not candidates:
            return None
        
        # 在候选发票中找金额匹配的
        for invoice in candidates:
            if is_amount_within_tolerance(receipt.received_amount, invoice.amount, self.tolerance):
                return invoice
        
        # 如果没有金额完全匹配的，返回名称最相似的
        best_match = None
        best_similarity = 0.0
        for invoice in candidates:
            similarity = calculate_similarity(receipt.payer_name, invoice.customer_name)
            if similarity > best_similarity:
                best_match = invoice
                best_similarity = similarity
        
        return best_match
    
    def match_three_way(
        self,
        contracts: List[ContractInfo],
        invoices: List[InvoiceInfo],
        receipts: List[ReceiptInfo]
    ) -> List[PaymentMatch]:
        """
        执行三方匹配（合同-发票-回执）
        
        参数:
            contracts: 合同列表
            invoices: 发票列表
            receipts: 回执列表
        
        返回:
            匹配结果列表
        """
        matches = []
        matched_invoices = set()
        matched_receipts = set()
        
        # 为每个合同创建匹配记录
        for contract in contracts:
            # 找到所有匹配的发票
            contract_invoices = []
            for invoice in invoices:
                if invoice.invoice_id in matched_invoices:
                    continue
                
                if is_name_similar(contract.customer_name, invoice.customer_name, self.name_similarity_threshold):
                    if is_amount_within_tolerance(contract.contract_amount, invoice.amount, self.tolerance):
                        contract_invoices.append(invoice)
                        matched_invoices.add(invoice.invoice_id)
            
            # 找到所有匹配的回执
            contract_receipts = []
            for receipt in receipts:
                if receipt.receipt_id in matched_receipts:
                    continue
                
                if is_name_similar(contract.customer_name, receipt.payer_name, self.name_similarity_threshold):
                    # 回执可以匹配合同或发票的金额
                    amount_match = is_amount_within_tolerance(contract.contract_amount, receipt.received_amount, self.tolerance)
                    
                    # 或者匹配任何一张发票的金额
                    for invoice in contract_invoices:
                        if is_amount_within_tolerance(invoice.amount, receipt.received_amount, self.tolerance):
                            amount_match = True
                            break
                    
                    if amount_match:
                        contract_receipts.append(receipt)
                        matched_receipts.add(receipt.receipt_id)
            
            # 计算收款状态
            invoiced_amount = sum(inv.amount for inv in contract_invoices)
            received_amount = sum(rec.received_amount for rec in contract_receipts)
            
            payment_ratio = min(float(received_amount / contract.contract_amount), 1.0) if contract.contract_amount > 0 else 0.0
            
            if payment_ratio == 0:
                status = 'unpaid'
            elif payment_ratio >= 1.0:
                status = 'paid'
            else:
                status = 'partial'
            
            match = PaymentMatch(
                match_id=str(uuid4()),
                contract_id=contract.contract_id,
                invoice_ids=[inv.invoice_id for inv in contract_invoices],
                receipt_ids=[rec.receipt_id for rec in contract_receipts],
                contract_amount=contract.contract_amount,
                invoiced_amount=invoiced_amount,
                received_amount=received_amount,
                status=status,
                payment_ratio=payment_ratio,
                last_updated=datetime.now()
            )
            
            matches.append(match)
        
        return matches
    
    def calculate_payment_status(
        self,
        contract_amount: Decimal,
        received_amount: Decimal
    ) -> Tuple[str, float]:
        """
        计算收款状态
        
        参数:
            contract_amount: 合同金额
            received_amount: 已收款金额
        
        返回:
            (状态, 收款比例)
        """
        if contract_amount == 0:
            return ('unpaid', 0.0)
        
        payment_ratio = min(float(received_amount / contract_amount), 1.0)
        
        if payment_ratio == 0:
            status = 'unpaid'
        elif payment_ratio >= 1.0:
            status = 'paid'
        else:
            status = 'partial'
        
        return (status, payment_ratio)
