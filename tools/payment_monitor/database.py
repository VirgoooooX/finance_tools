"""数据库操作"""
import sqlite3
import json
from typing import Optional, List, Dict, Any
from decimal import Decimal
from datetime import datetime

from .models import InvoiceInfo, ContractInfo, ReceiptInfo, PaymentMatch, PaymentSummary


def create_database_schema(db_path: str) -> None:
    """创建数据库表结构"""
    conn = sqlite3.connect(db_path)
    try:
        # 合同表
        conn.execute("""
            CREATE TABLE IF NOT EXISTS contracts (
                contract_id TEXT PRIMARY KEY,
                contract_number TEXT NOT NULL UNIQUE,
                contract_amount REAL NOT NULL,
                sign_date TEXT NOT NULL,
                customer_name TEXT NOT NULL,
                payment_terms TEXT,
                source_file TEXT,
                extracted_at TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_contracts_customer ON contracts(customer_name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_contracts_date ON contracts(sign_date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_contracts_amount ON contracts(contract_amount)")
        
        # 发票表
        conn.execute("""
            CREATE TABLE IF NOT EXISTS invoices (
                invoice_id TEXT PRIMARY KEY,
                invoice_number TEXT NOT NULL,
                amount REAL NOT NULL,
                issue_date TEXT NOT NULL,
                customer_name TEXT NOT NULL,
                tax_number TEXT,
                source_file TEXT,
                extracted_at TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_invoices_customer ON invoices(customer_name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_invoices_date ON invoices(issue_date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_invoices_amount ON invoices(amount)")
        
        # 回执表
        conn.execute("""
            CREATE TABLE IF NOT EXISTS receipts (
                receipt_id TEXT PRIMARY KEY,
                receipt_number TEXT NOT NULL,
                received_amount REAL NOT NULL,
                received_date TEXT NOT NULL,
                payer_name TEXT NOT NULL,
                bank_account TEXT,
                source_file TEXT,
                extracted_at TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_receipts_payer ON receipts(payer_name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_receipts_date ON receipts(received_date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_receipts_amount ON receipts(received_amount)")
        
        # 匹配表
        conn.execute("""
            CREATE TABLE IF NOT EXISTS payment_matches (
                match_id TEXT PRIMARY KEY,
                contract_id TEXT NOT NULL,
                invoice_ids TEXT NOT NULL,
                receipt_ids TEXT NOT NULL,
                contract_amount REAL NOT NULL,
                invoiced_amount REAL NOT NULL,
                received_amount REAL NOT NULL,
                status TEXT NOT NULL,
                payment_ratio REAL NOT NULL,
                last_updated TEXT NOT NULL,
                FOREIGN KEY (contract_id) REFERENCES contracts(contract_id)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_matches_contract ON payment_matches(contract_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_matches_status ON payment_matches(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_matches_updated ON payment_matches(last_updated)")
        
        # 文件哈希去重表
        conn.execute("""
            CREATE TABLE IF NOT EXISTS file_hashes (
                file_hash TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                processed_at TEXT NOT NULL
            )
        """)
        
        conn.commit()
    finally:
        conn.close()


def insert_contract(conn: sqlite3.Connection, contract: ContractInfo) -> None:
    """插入合同记录"""
    conn.execute("""
        INSERT OR REPLACE INTO contracts 
        (contract_id, contract_number, contract_amount, sign_date, customer_name, 
         payment_terms, source_file, extracted_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        contract.contract_id,
        contract.contract_number,
        float(contract.contract_amount),
        contract.sign_date.isoformat(),
        contract.customer_name,
        contract.payment_terms,
        contract.source_file,
        contract.extracted_at.isoformat()
    ))


def insert_invoice(conn: sqlite3.Connection, invoice: InvoiceInfo) -> None:
    """插入发票记录"""
    conn.execute("""
        INSERT OR REPLACE INTO invoices 
        (invoice_id, invoice_number, amount, issue_date, customer_name, 
         tax_number, source_file, extracted_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        invoice.invoice_id,
        invoice.invoice_number,
        float(invoice.amount),
        invoice.issue_date.isoformat(),
        invoice.customer_name,
        invoice.tax_number,
        invoice.source_file,
        invoice.extracted_at.isoformat()
    ))


def insert_receipt(conn: sqlite3.Connection, receipt: ReceiptInfo) -> None:
    """插入回执记录"""
    conn.execute("""
        INSERT OR REPLACE INTO receipts 
        (receipt_id, receipt_number, received_amount, received_date, payer_name, 
         bank_account, source_file, extracted_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        receipt.receipt_id,
        receipt.receipt_number,
        float(receipt.received_amount),
        receipt.received_date.isoformat(),
        receipt.payer_name,
        receipt.bank_account,
        receipt.source_file,
        receipt.extracted_at.isoformat()
    ))


def insert_payment_match(conn: sqlite3.Connection, match: PaymentMatch) -> None:
    """插入匹配记录"""
    conn.execute("""
        INSERT OR REPLACE INTO payment_matches 
        (match_id, contract_id, invoice_ids, receipt_ids, contract_amount, 
         invoiced_amount, received_amount, status, payment_ratio, last_updated)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        match.match_id,
        match.contract_id,
        json.dumps(match.invoice_ids),
        json.dumps(match.receipt_ids),
        float(match.contract_amount),
        float(match.invoiced_amount),
        float(match.received_amount),
        match.status,
        match.payment_ratio,
        match.last_updated.isoformat()
    ))


def query_payment_summary(conn: sqlite3.Connection, filters: Optional[Dict] = None) -> PaymentSummary:
    """查询收款统计摘要"""
    where_clauses = []
    params = []
    
    if filters:
        if filters.get('customer_name'):
            where_clauses.append("customer_name LIKE ?")
            params.append(f"%{filters['customer_name']}%")
    
    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    
    # 统计各状态数量
    cur = conn.execute(f"""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN status = 'unpaid' THEN 1 ELSE 0 END) as unpaid,
            SUM(CASE WHEN status = 'partial' THEN 1 ELSE 0 END) as partial,
            SUM(CASE WHEN status = 'paid' THEN 1 ELSE 0 END) as paid,
            SUM(contract_amount) as total_contract,
            SUM(received_amount) as total_received
        FROM payment_matches
        {where_sql}
    """, params)
    
    row = cur.fetchone()
    if not row:
        return PaymentSummary()
    
    total_contract = Decimal(str(row[4] or 0))
    total_received = Decimal(str(row[5] or 0))
    overall_ratio = float(total_received / total_contract) if total_contract > 0 else 0.0
    
    return PaymentSummary(
        total_contracts=row[0] or 0,
        unpaid_count=row[1] or 0,
        partial_count=row[2] or 0,
        paid_count=row[3] or 0,
        total_contract_amount=total_contract,
        total_received_amount=total_received,
        overall_payment_ratio=overall_ratio,
        generated_at=datetime.now()
    )


def query_payment_details(conn: sqlite3.Connection, status: str, 
                         filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
    """查询收款明细"""
    where_clauses = ["status = ?"]
    params = [status]
    
    if filters:
        if filters.get('customer_name'):
            where_clauses.append("customer_name LIKE ?")
            params.append(f"%{filters['customer_name']}%")
    
    where_sql = " AND ".join(where_clauses)
    
    cur = conn.execute(f"""
        SELECT 
            pm.match_id,
            c.contract_number,
            c.customer_name,
            pm.contract_amount,
            pm.invoiced_amount,
            pm.received_amount,
            pm.status,
            pm.payment_ratio,
            pm.last_updated
        FROM payment_matches pm
        LEFT JOIN contracts c ON pm.contract_id = c.contract_id
        WHERE {where_sql}
        ORDER BY pm.last_updated DESC
    """, params)
    
    rows = cur.fetchall()
    return [
        {
            'match_id': r[0],
            'contract_number': r[1],
            'customer_name': r[2],
            'contract_amount': r[3],
            'invoiced_amount': r[4],
            'received_amount': r[5],
            'status': r[6],
            'payment_ratio': r[7],
            'last_updated': r[8]
        }
        for r in rows
    ]
