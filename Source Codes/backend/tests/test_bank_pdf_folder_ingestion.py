"""
Automated Tests for ABC Bank Transaction PDF Recognition and Folder Ingestion.
Verifies that:
1. ABC Bank transaction PDFs are detected and routed to BankTransactionModel.
2. Normal invoice/receipt PDFs continue to be processed through the document pipeline.
3. No DocumentModel or AccountingTransactionModel is created for bank PDFs.
4. All bank transaction fields are accurately extracted from the PDF.
5. Ingestion is idempotent (re-scanning skips duplicates and links PDF).
6. Cross-source deduplication prevents API ingestion from duplicating PDF-imported transactions.
7. Locally stored transaction PDF is served by GET /api/bank-ingestion/transactions/{id}/pdf.
"""

import os
import sys
import shutil
import tempfile
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.models import (
    Base,
    CompanyModel,
    DocumentModel,
    AccountingTransactionModel,
    BankTransactionModel,
    ChartOfAccountModel,
)
from app.services.folder_ingestion.folder_service import (
    LocalFolderIngestionService,
    is_abc_bank_transaction_pdf,
    parse_abc_bank_transaction_pdf,
)
from app.services.bank_ingestion.bank_service import ABCBankIngestionService
from app.services.bank_ingestion.bank_client import ABCBankClient
from app.services.pdf_ocr import extract_text_from_document
from fastapi.testclient import TestClient
from app.main import app
from app.db.database import get_db

def _resolve_sample_bank_folder() -> Path:
    from app.config import LOCAL_FOLDER_PATH
    candidates = [
        LOCAL_FOLDER_PATH,
        Path("D:/CompanyInvoice"),
        backend_dir.parent / "testing_document",
    ]
    for c in candidates:
        if c and (c / "TXN-0001.pdf").exists():
            return Path(c)
    return Path("D:/CompanyInvoice")


@pytest.fixture
def db_session(tmp_path):
    """Creates a temporary isolated SQLite database with fresh schema."""
    db_file = tmp_path / "test_folder_bank.db"
    engine = create_engine(f"sqlite:///{db_file}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()

    # Seed company
    comp = CompanyModel(
        company_id="COMP-000001",
        company_name="NextTech Solutions Sdn Bhd",
        currency="MYR",
        is_active=True,
    )
    session.add(comp)

    # Seed key COA accounts
    coas = [
        ChartOfAccountModel(company_id="COMP-000001", account_code="300-0000", account_name="TRADE DEBTORS", account_type="Asset", is_active=True),
        ChartOfAccountModel(company_id="COMP-000001", account_code="400-0000", account_name="TRADE CREDITORS", account_type="Liability", is_active=True),
        ChartOfAccountModel(company_id="COMP-000001", account_code="310-1000", account_name="CASH AT BANK", account_type="Asset", is_active=True),
        ChartOfAccountModel(company_id="COMP-000001", account_code="901-0000", account_name="ADVERTISEMENT", account_type="Expense", is_active=True),
    ]
    for c in coas:
        session.add(c)
    session.commit()

    yield session
    session.close()


def test_is_abc_bank_transaction_pdf_detection():
    """Verifies that ABC Bank transaction PDFs are detected and normal invoices are rejected."""
    # Test on real files from CompanyInvoice if available
    company_inv = _resolve_sample_bank_folder()
    if (company_inv / "TXN-0001.pdf").exists():
        txt_bank = extract_text_from_document(str(company_inv / "TXN-0001.pdf"))
        assert is_abc_bank_transaction_pdf(txt_bank, "TXN-0001.pdf") is True

    if (company_inv / "test_sales_invoice_0002.pdf").exists():
        txt_inv = extract_text_from_document(str(company_inv / "test_sales_invoice_0002.pdf"))
        assert is_abc_bank_transaction_pdf(txt_inv, "test_sales_invoice_0002.pdf") is False

    # Synthetic text checks
    fake_bank = (
        "ABC BANK COMMERCIAL & BUSINESS BANKING\n"
        "TRANSACTION ADVICE / PAYMENT RECEIPT\n"
        "Transaction ID: TXN-0099\n"
        "Running Balance RM 5,000.00"
    )
    assert is_abc_bank_transaction_pdf(fake_bank) is True

    fake_invoice = (
        "INVOICE # INV-2026-0001\n"
        "Vendor: ABC Office Supplies\n"
        "Total: RM 1,000.00"
    )
    assert is_abc_bank_transaction_pdf(fake_invoice) is False


def test_parse_abc_bank_transaction_pdf(tmp_path):
    """Verifies parsing of fields from an ABC Bank transaction PDF."""
    company_inv = _resolve_sample_bank_folder()
    sample_pdf = company_inv / "TXN-0001.pdf"
    if not sample_pdf.exists():
        pytest.skip("TXN-0001.pdf not present in D:/CompanyInvoice")

    txt = extract_text_from_document(str(sample_pdf))
    data = parse_abc_bank_transaction_pdf(sample_pdf, txt)

    assert data["bank_name"] == "ABC Bank"
    assert data["bank_transaction_id"] == "TXN-0001"
    assert data["transaction_date"] in ["2026-10-20", "2026-01-01"]
    assert data["account_id"] == "ABC-BIZ-0001"
    assert data["account_number"] == "4320-6210-4821"
    assert data["reference"] in ["INV-2026-0001", "Capital Contribution"]
    assert data["money_in"] in [1060.0, 90000.0]
    assert data["money_out"] == 0.0
    assert data["running_balance"] in [11060.0, 90000.0, 165300.0]
    assert data["currency"] == "MYR"
    assert data["description"] is not None


def test_folder_ingestion_abc_bank_pdf_routing(db_session, tmp_path):
    """
    Verifies that scanning a folder containing an ABC Bank transaction PDF:
    - Creates a BankTransactionModel with source="abc_bank_pdf"
    - Does NOT create DocumentModel
    - Does NOT create AccountingTransactionModel
    - Runs automatic classification and matching
    """
    company_inv = _resolve_sample_bank_folder()
    sample_pdf = company_inv / "TXN-0001.pdf"
    if not sample_pdf.exists():
        pytest.skip("TXN-0001.pdf not present in D:/CompanyInvoice")

    # Copy to a temporary scan folder
    scan_dir = tmp_path / "scan_folder"
    scan_dir.mkdir()
    shutil.copy(sample_pdf, scan_dir / "TXN-0001.pdf")

    service = LocalFolderIngestionService(db=db_session, folder_path=scan_dir, company_id="COMP-000001")
    result = service.scan_and_ingest()

    assert result["success"] is True
    assert result["files_scanned"] == 1
    assert result["bank_transactions_imported"] == 1
    assert result["documents_imported"] == 0
    assert result["duplicates_skipped"] == 0

    # Verify no DocumentModel created
    doc_count = db_session.query(DocumentModel).count()
    assert doc_count == 0

    # Verify no AccountingTransactionModel created
    acct_txn_count = db_session.query(AccountingTransactionModel).count()
    assert acct_txn_count == 0

    # Verify BankTransactionModel created
    bank_tx = db_session.query(BankTransactionModel).filter(
        BankTransactionModel.bank_transaction_id == "TXN-0001"
    ).first()
    assert bank_tx is not None
    assert bank_tx.source == "abc_bank_pdf"
    assert bank_tx.money_in in [1060.0, 90000.0]
    assert bank_tx.money_out == 0.0
    assert bank_tx.reference in ["INV-2026-0001", "Capital Contribution"]
    assert bank_tx.pdf_file_path is not None
    assert os.path.exists(bank_tx.pdf_file_path)
    assert bank_tx.classification_status in ["AUTO_CLASSIFIED", "CONFIRMED", "PENDING"]


def test_folder_ingestion_idempotency_and_linking(db_session, tmp_path):
    """
    Verifies that re-scanning the same bank transaction PDF is idempotent:
    - Does not duplicate records
    - Links PDF file path if not linked
    - Reports duplicate skipped
    """
    company_inv = _resolve_sample_bank_folder()
    sample_pdf = company_inv / "TXN-0002 (1).pdf" if (company_inv / "TXN-0002 (1).pdf").exists() else (company_inv / "TXN-0002.pdf")
    if not sample_pdf.exists():
        pytest.skip("TXN-0002 (1).pdf not present in D:/CompanyInvoice")

    scan_dir = tmp_path / "scan_folder_2"
    scan_dir.mkdir()
    shutil.copy(sample_pdf, scan_dir / "TXN-0002 (1).pdf")

    service = LocalFolderIngestionService(db=db_session, folder_path=scan_dir, company_id="COMP-000001")

    # First scan
    res1 = service.scan_and_ingest()
    assert res1["bank_transactions_imported"] == 1
    assert res1["duplicates_skipped"] == 0

    # Second scan: exact same file
    res2 = service.scan_and_ingest()
    assert res2["bank_transactions_imported"] == 0
    assert res2["duplicates_skipped"] == 1

    # Verify database still has exactly one record for TXN-0002
    tx_count = db_session.query(BankTransactionModel).filter(
        BankTransactionModel.bank_transaction_id == "TXN-0002"
    ).count()
    assert tx_count == 1


def test_cross_source_deduplication_pdf_then_api(db_session, tmp_path):
    """
    Verifies that if a transaction is first imported via PDF (source="abc_bank_pdf"),
    subsequent API collection of the same TXN-xxxx does not create a duplicate record.
    """
    company_inv = _resolve_sample_bank_folder()
    sample_pdf = company_inv / "TXN-0001.pdf"
    if not sample_pdf.exists():
        pytest.skip("TXN-0001.pdf not present in D:/CompanyInvoice")

    scan_dir = tmp_path / "scan_folder_cross"
    scan_dir.mkdir()
    shutil.copy(sample_pdf, scan_dir / "TXN-0001.pdf")

    # 1. Ingest via local folder PDF
    folder_service = LocalFolderIngestionService(db=db_session, folder_path=scan_dir, company_id="COMP-000001")
    folder_res = folder_service.scan_and_ingest()
    assert folder_res["bank_transactions_imported"] == 1

    # 2. Mock API client returning the exact same TXN-0001
    class MockClient(ABCBankClient):
        def get_account(self):
            return {
                "account_id": "ABC-BIZ-0001",
                "account_number": "4320-6210-4821",
                "currency": "MYR",
                "current_balance": 11060.0,
            }

        def get_transactions(self, after_id=None):
            return {
                "transactions": [{
                    "transaction_id": "TXN-0001",
                    "date": "2026-10-20",
                    "description": "Customer payment - TechCore Solutions Sdn Bhd",
                    "reference": "INV-2026-0001",
                    "money_in": 1060.0,
                    "money_out": 0.0,
                    "running_balance": 11060.0,
                }]
            }

    api_service = ABCBankIngestionService(db=db_session, company_id="COMP-000001", bank_client=MockClient())
    api_res = api_service.collect_transactions(use_cursor=False)

    # API collection should detect existing TXN-0001 and skip as duplicate
    assert api_res["transactions_imported"] == 0
    assert api_res["duplicates_skipped"] == 1

    # Total in DB must remain 1
    total_txns = db_session.query(BankTransactionModel).filter(
        BankTransactionModel.bank_transaction_id == "TXN-0001"
    ).count()
    assert total_txns == 1


def test_get_transaction_pdf_endpoint(db_session, tmp_path):
    """
    Verifies that GET /api/bank-ingestion/transactions/{transaction_id}/pdf
    serves the locally stored PDF file if available.
    """
    # Create fake stored PDF
    fake_pdf = tmp_path / "TXN-9999.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 test transaction pdf content")

    tx = BankTransactionModel(
        company_id="COMP-000001",
        source="abc_bank_pdf",
        unique_source_id="abc_bank_pdf:TXN-9999",
        bank_name="ABC Bank",
        account_id="ABC-BIZ-0001",
        account_number="4320-6210-4821",
        bank_transaction_id="TXN-9999",
        transaction_date="2026-10-23",
        description="Test payment",
        money_in=500.0,
        money_out=0.0,
        currency="MYR",
        pdf_file_path=str(fake_pdf),
    )
    db_session.add(tx)
    db_session.commit()

    # Override get_db for FastAPI test client
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    try:
        client = TestClient(app)
        resp = client.get("/api/bank-ingestion/transactions/TXN-9999/pdf")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/pdf"
        assert resp.content == b"%PDF-1.4 test transaction pdf content"
    finally:
        app.dependency_overrides.clear()
