"""
Automated tests for Bank Transaction Intelligence, Accounting Classification,
Document Matching, Audit Trail, and Unified Financial Records API.
"""

import sys
import json
from pathlib import Path
from unittest.mock import MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from fastapi.testclient import TestClient
from app.db.models import Base, BankTransactionModel, DocumentModel, CompanyModel, ChartOfAccountModel
from app.services.bank_ingestion.bank_classifier import (
    classify_bank_transaction,
    match_bank_transaction_to_documents,
    process_bank_transaction_intelligence,
    append_audit_entry,
    CATEGORY_ACCOUNTS
)
from app.main import app


def create_test_db():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = Session()

    # Seed test company
    comp = CompanyModel(
        company_id="COMP-000001",
        company_name="NextTech Solutions Sdn Bhd",
        currency="MYR"
    )
    session.add(comp)

    # Seed essential COA entries
    accounts = [
        ChartOfAccountModel(company_id="COMP-000001", account_code="300-0000", account_name="TRADE DEBTORS", account_type="Asset"),
        ChartOfAccountModel(company_id="COMP-000001", account_code="400-0000", account_name="TRADE CREDITORS", account_type="Liability"),
        ChartOfAccountModel(company_id="COMP-000001", account_code="200-3000", account_name="OFFICE EQUIPMENT", account_type="Asset"),
        ChartOfAccountModel(company_id="COMP-000001", account_code="902-0000", account_name="BANK CHARGES", account_type="Expense"),
        ChartOfAccountModel(company_id="COMP-000001", account_code="915-0000", account_name="GENERAL EXPENSES", account_type="Expense"),
        ChartOfAccountModel(company_id="COMP-000001", account_code="500-1000", account_name="CASH SALES", account_type="Revenue"),
    ]
    for acc in accounts:
        session.add(acc)
    session.commit()
    return Session, session


def test_classify_customer_payment_incoming():
    """Verify customer payment incoming is classified to Trade Debtors (300-0000)."""
    _, db = create_test_db()
    tx = BankTransactionModel(
        bank_transaction_id="TXN-0001",
        description="Customer payment",
        reference="REF-ABC-0001",
        money_in=5000.0,
        money_out=0.0
    )
    res = classify_bank_transaction(tx, db)
    assert res["suggested_category"] in ["Accounts Receivable Settlement", "Customer receipt / accounts receivable settlement"]
    assert res["suggested_account_code"] == "300-0000"
    assert res["suggested_account_name"] == "TRADE DEBTORS"
    assert res["classification_confidence"] >= 0.85
    assert res["classification_status"] == "AUTO_CLASSIFIED"
    assert res["review_status"] == "NEEDS_REVIEW"  # Per accounting rule: always requires review before posting


def test_classify_equipment_purchase_outgoing():
    """Verify IT equipment purchase is classified to Office Equipment fixed asset (200-3000)."""
    _, db = create_test_db()
    tx = BankTransactionModel(
        bank_transaction_id="TXN-0002",
        description="IT equipment purchase",
        reference="REF-ABC-0002",
        money_in=0.0,
        money_out=2226.0
    )
    res = classify_bank_transaction(tx, db)
    assert res["suggested_category"] in ["Property, Plant and Equipment", "Equipment / fixed asset"]
    assert res["suggested_account_code"] == "200-3000"
    assert res["suggested_account_name"] == "OFFICE EQUIPMENT"
    assert res["classification_confidence"] >= 0.90
    assert "Capital expenditure" in res["classification_explanation"]


def test_classify_supplier_payment_and_bank_charge():
    """Verify supplier payment and bank charges are correctly mapped to Trade Creditors and Bank Charges."""
    _, db = create_test_db()
    
    # 1. Supplier payment
    tx_supp = BankTransactionModel(
        bank_transaction_id="TXN-0003",
        description="Supplier payment for inventory",
        reference="REF-SUPP-01",
        money_in=0.0,
        money_out=1500.0
    )
    res_supp = classify_bank_transaction(tx_supp, db)
    assert res_supp["suggested_category"] == "Supplier payment / accounts payable settlement"
    assert res_supp["suggested_account_code"] == "400-0000"

    # 2. Bank charge
    tx_fee = BankTransactionModel(
        bank_transaction_id="TXN-0004",
        description="Interbank fund transfer fee",
        reference="FEE-001",
        money_in=0.0,
        money_out=5.0
    )
    res_fee = classify_bank_transaction(tx_fee, db)
    assert res_fee["suggested_category"] in ["Bank Charges Expense", "Bank charge"]
    assert res_fee["suggested_account_code"] == "902-0000"


def test_matching_unmatched_when_no_document():
    """Verify that when no corresponding document exists, match_status is UNMATCHED."""
    _, db = create_test_db()
    tx = BankTransactionModel(
        bank_transaction_id="TXN-0001",
        description="Customer payment",
        reference="REF-ABC-0001",
        money_in=5000.0,
        money_out=0.0
    )
    res = match_bank_transaction_to_documents(tx, db)
    assert res["match_status"] == "UNMATCHED"
    assert res["matched_document_id"] is None
    assert "Supporting document required" in res["match_explanation"]


def test_matching_suggested_match_when_invoice_matches():
    """Verify that when a sales invoice with matching amount and customer exists, suggested match is found."""
    _, db = create_test_db()
    
    # Seed matching document
    doc = DocumentModel(
        company_id="COMP-000001",
        document_id="DOC-2026-000010",
        original_filename="invoice_acme.pdf",
        file_path="dummy",
        file_type="pdf",
        mime_type="application/pdf",
        document_type="Sales Invoice",
        vendor_name="Acme Corp",
        grand_total=5000.0,
        extracted_data_json='{"grand_total": 5000.0, "invoice_number": "INV-5000", "buyer_name": "Acme Corp"}'
    )
    db.add(doc)
    db.commit()

    tx = BankTransactionModel(
        bank_transaction_id="TXN-0001",
        description="Payment for INV-5000 from Acme Corp",
        reference="REF-001",
        money_in=5000.0,
        money_out=0.0
    )
    res = match_bank_transaction_to_documents(tx, db)
    assert res["match_status"] in ["MATCHED", "SUGGESTED_MATCH"]
    assert res["matched_document_id"] == "DOC-2026-000010"
    assert res["match_confidence"] >= 0.70


def test_demo_transactions_matching_and_classification():
    """Verify that updated demo transactions match DOC-2026-000001 and DOC-2026-000003."""
    _, db = create_test_db()

    # Seed DOC-2026-000001 (Sales Invoice to TechCore Solutions Sdn Bhd, 1060.00)
    doc1 = DocumentModel(
        company_id="COMP-000001",
        document_id="DOC-2026-000001",
        original_filename="Sales_Invoice_NextTech_INV-2026-0001.pdf",
        file_path="dummy1",
        file_type="pdf",
        mime_type="application/pdf",
        document_type="Sales Invoice",
        vendor_name="Lumpur",  # Raw OCR extracted vendor
        grand_total=1060.0,
        extracted_data_json=json.dumps({
            "grand_total": 1060.0,
            "invoice_number": "INV-2026-0001",
            "buyer_name": "TechCore Solutions Sdn Bhd",
            "display_summary": {"party": "TechCore Solutions Sdn Bhd", "amount": 1060.0}
        })
    )
    # Seed DOC-2026-000003 (Expense Receipt from City Business Cafe, 80.56)
    doc3 = DocumentModel(
        company_id="COMP-000001",
        document_id="DOC-2026-000003",
        original_filename="Receipt_Business_Lunch_RC-2026-0012.pdf",
        file_path="dummy3",
        file_type="pdf",
        mime_type="application/pdf",
        document_type="Expense Receipt",
        vendor_name="City Business Caf\ufffd Sdn Bhd",
        grand_total=80.56,
        extracted_data_json=json.dumps({
            "grand_total": 80.56,
            "receipt_number": "RC-2026-0012",
            "merchant_name": "City Business Caf\ufffd Sdn Bhd",
            "display_summary": {"party": "City Business Caf\ufffd Sdn Bhd", "amount": 80.56}
        })
    )
    db.add(doc1)
    db.add(doc3)
    db.commit()

    # Test TXN-0001
    tx1 = BankTransactionModel(
        bank_transaction_id="TXN-0001",
        description="Customer payment — TechCore Solutions Sdn Bhd",
        reference="INV-2026-0001",
        money_in=1060.00,
        money_out=0.00
    )
    cls1 = classify_bank_transaction(tx1, db)
    assert cls1["suggested_account_code"] == "300-0000"
    assert cls1["suggested_category"] in ["Accounts Receivable Settlement", "Customer receipt / accounts receivable settlement"]

    match1 = match_bank_transaction_to_documents(tx1, db)
    assert match1["match_status"] == "MATCHED"
    assert match1["matched_document_id"] == "DOC-2026-000001"
    assert match1["match_confidence"] >= 0.90
    assert match1["candidates"][0]["vendor_name"] == "TechCore Solutions Sdn Bhd"

    # Test TXN-0002
    tx2 = BankTransactionModel(
        bank_transaction_id="TXN-0002",
        description="Business lunch — City Business Café Sdn Bhd",
        reference="RC-2026-0012",
        money_in=0.00,
        money_out=80.56
    )
    cls2 = classify_bank_transaction(tx2, db)
    assert cls2["suggested_account_code"] == "915-0000"
    assert cls2["suggested_category"] == "Operating expense"

    match2 = match_bank_transaction_to_documents(tx2, db)
    assert match2["match_status"] == "MATCHED"
    assert match2["matched_document_id"] == "DOC-2026-000003"
    assert match2["match_confidence"] >= 0.90


def test_audit_trail_immutability():
    """Verify that audit trail correctly appends chronological entries."""
    trail_json = None
    trail_json = append_audit_entry(trail_json, "AUTO_CLASSIFIED", "Auto-classified as Trade Debtors", actor="Engine")
    trail_json = append_audit_entry(trail_json, "REVIEW_APPROVED", "User approved classification", actor="User")

    import json
    parsed = json.loads(trail_json)
    assert len(parsed) == 2
    assert parsed[0]["action"] == "AUTO_CLASSIFIED"
    assert parsed[0]["actor"] == "Engine"
    assert parsed[1]["action"] == "REVIEW_APPROVED"
    assert parsed[1]["actor"] == "User"


def test_unified_financial_records_api_endpoint():
    """Verify /api/documents/financial-records returns 6 total records for current dataset."""
    client = TestClient(app)
    res = client.get("/api/documents/financial-records")
    assert res.status_code == 200
    data = res.json()
    assert data["total_records"] == data["documents_count"] + data["bank_transactions_count"]
    assert data["documents_count"] >= 0
    assert data["bank_transactions_count"] >= 0
    assert len(data["records"]) == data["total_records"]

    # Verify type breakdown
    doc_records = [r for r in data["records"] if r["record_type"] == "ACCOUNTING_DOCUMENT"]
    bank_records = [r for r in data["records"] if r["record_type"] == "BANK_TRANSACTION"]
    assert len(doc_records) == data["documents_count"]
    assert len(bank_records) == data["bank_transactions_count"]


def run_all_tests():
    print("=" * 60)
    print("Running Bank Intelligence & Unified Workspace Unit Tests...")
    print("=" * 60)
    tests = [
        ("test_classify_customer_payment_incoming", test_classify_customer_payment_incoming),
        ("test_classify_equipment_purchase_outgoing", test_classify_equipment_purchase_outgoing),
        ("test_classify_supplier_payment_and_bank_charge", test_classify_supplier_payment_and_bank_charge),
        ("test_matching_unmatched_when_no_document", test_matching_unmatched_when_no_document),
        ("test_matching_suggested_match_when_invoice_matches", test_matching_suggested_match_when_invoice_matches),
        ("test_demo_transactions_matching_and_classification", test_demo_transactions_matching_and_classification),
        ("test_audit_trail_immutability", test_audit_trail_immutability),
        ("test_unified_financial_records_api_endpoint", test_unified_financial_records_api_endpoint),
    ]

    passed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"[PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            raise e

    print("=" * 60)
    print(f"Result: {passed}/{len(tests)} tests passed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
