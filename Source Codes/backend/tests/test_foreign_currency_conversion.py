import os
import sys
import json
import pytest
from decimal import Decimal
from unittest.mock import patch, MagicMock

# Add backend directory to sys.path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from fastapi.testclient import TestClient
from app.main import app
from app.db.database import get_db, SessionLocal, ensure_db_schema
from app.db.models import DocumentModel, AccountingTransactionModel, BankTransactionModel
from app.services.currency_service import (
    CurrencyExchangeService,
    CurrencyExchangeError,
    ExchangeRateResult
)
from app.services.document_agent import DocumentProcessingAgent
from app.agents.accounting_agent import AccountingAgent
from app.services.bank_ingestion.bank_classifier import match_bank_transaction_to_documents

client = TestClient(app)


@pytest.fixture(autouse=True)
def init_db():
    ensure_db_schema()
    yield


def test_or_000002_live_bnm_conversion():
    """
    Test Case 1 & Prompt Specific Case:
    Official Receipt OR-000002
    Amount: USD 1,400.00
    Date: 2026-01-25 (Sunday)
    BNM Lookback to Friday (2026-01-23 Session 1700) returns rate 4.005.
    Expected Converted MYR: RM 5,607.00.
    """
    conv = CurrencyExchangeService.convert_document_amounts(
        currency="USD",
        transaction_date="2026-01-25",
        grand_total=1400.0,
        subtotal=1400.0,
        discount=0.0,
        shipping=0.0,
        tax_amount=0.0,
        line_items=[{
            "item_number": "1",
            "description": "Professional Consulting Services",
            "quantity": 1.0,
            "unit_price": 1400.0,
            "line_total": 1400.0
        }]
    )

    assert conv["currency"] == "USD"
    assert conv["original_currency"] == "USD"
    assert conv["original_amount"] == 1400.0
    assert conv["base_currency"] == "MYR"
    assert conv["exchange_rate"] == 4.005
    assert conv["exchange_rate_date"] == "2026-01-23"
    assert conv["converted_amount_myr"] == 5607.00
    assert "Bank Negara Malaysia" in conv["exchange_rate_provider"]
    assert conv["conversion_status"] == "CONVERTED"

    # Line item converted values
    assert len(conv["line_items"]) == 1
    item = conv["line_items"][0]
    assert item["original_unit_price"] == 1400.0
    assert item["converted_unit_price_myr"] == 5607.00
    assert item["original_total_amount"] == 1400.0
    assert item["converted_total_amount_myr"] == 5607.00


def test_myr_base_currency_no_conversion_needed():
    """
    Test Case 2: Documents in MYR must not be altered, exchange_rate is 1.0,
    and conversion_status is NOT_REQUIRED.
    """
    conv = CurrencyExchangeService.convert_document_amounts(
        currency="MYR",
        transaction_date="2026-01-15",
        grand_total=2500.0,
        subtotal=2500.0
    )

    assert conv["currency"] == "MYR"
    assert conv["exchange_rate"] == 1.0
    assert conv["converted_amount_myr"] == 2500.0
    assert conv["conversion_status"] == "NOT_REQUIRED"
    assert conv["is_foreign"] is False


def test_unit_normalization_jpy():
    """
    Test Case 3: Unit normalization for currencies quoted per 100 units like JPY.
    """
    res = CurrencyExchangeService.get_exchange_rate("JPY", "2026-10-22", session="1700")
    assert res.currency == "JPY"
    assert res.unit == 100
    assert res.rate < 0.1  # 2.5999 / 100 = ~0.025999
    assert abs(res.rate - (res.raw_rate / res.unit)) < 0.000001


def test_document_agent_foreign_currency_processing():
    """
    Test Case 4: DocumentProcessingAgent automatically applies currency conversion
    to foreign documents and flags for review if conversion fails.
    """
    agent = DocumentProcessingAgent(company_context={"company_id": "COMP-000001", "company_name": "NextTech Solutions Sdn Bhd"})
    
    raw_text = """
    OFFICIAL RECEIPT
    Receipt No: OR-000002
    Date: 25/01/2026
    Received From: ACME Global Corp
    For: Consultation Services
    Amount: USD 1,400.00
    Payment Method: Wire Transfer
    """

    doc_data, passed, messages = agent.process_document("TEST-DOC-USD", raw_text, "OR-000002.pdf")

    assert doc_data.currency == "USD"
    assert doc_data.grand_total == 1400.0
    assert doc_data.exchange_rate == 4.005
    assert doc_data.converted_amount_myr == 5607.00
    assert doc_data.conversion_status == "CONVERTED"


def test_strict_safeguard_failure_does_not_crash_extraction():
    """
    Test Case 5: When exchange rate retrieval fails, original values are preserved,
    conversion_status is FAILED, converted_amount_myr is None, needs_review is True,
    and extraction does NOT throw 422 or crash.
    """
    agent = DocumentProcessingAgent(company_context={"company_id": "COMP-000001", "company_name": "NextTech Solutions Sdn Bhd"})
    
    raw_text = """
    OFFICIAL RECEIPT
    Receipt No: OR-999999
    Date: 15/06/2026
    Received From: Global Partner
    Amount: EUR 3,000.00
    """

    with patch.object(CurrencyExchangeService, "_fetch_bnm_rate", return_value=None):
        with patch.object(CurrencyExchangeService, "_fetch_frankfurter_rate", return_value=None):
            doc_data, passed, messages = agent.process_document("TEST-FAIL-EUR", raw_text, "OR-999999.pdf")

            assert doc_data.currency == "EUR"
            assert doc_data.grand_total == 3000.0
            assert doc_data.converted_amount_myr is None
            assert doc_data.conversion_status == "FAILED"
            assert doc_data.needs_review is True
            assert any("Currency Conversion Warning" in m for m in messages)


def test_accounting_agent_uses_converted_amount_for_journal():
    """
    Test Case 6: AccountingAgent generates balanced Journal Entry in MYR using
    converted_amount_myr, NOT the foreign currency amount.
    """
    agent = AccountingAgent(company_id="COMP-000001")

    doc_dict = {
        "company_id": "COMP-000001",
        "document_id": "DOC-USD-002",
        "document_type": "Official Receipt",
        "receipt_number": "OR-000002",
        "transaction_date": "2026-01-25",
        "customer_name": "ACME Global Corp",
        "currency": "USD",
        "grand_total": 1400.0,
        "converted_amount_myr": 5607.00,
        "exchange_rate": 4.005,
        "exchange_rate_provider": "Bank Negara Malaysia (BNM OpenAPI)",
        "conversion_status": "CONVERTED"
    }

    analysis = agent.analyze(doc_dict)
    assert analysis.needs_review is False
    assert analysis.journal_entry is not None
    je = analysis.journal_entry
    assert je.is_balanced is True
    assert je.total_debit == 5607.00
    assert je.total_credit == 5607.00


def test_bank_matching_with_converted_myr_amount():
    """
    Test Case 7: Bank matching engine matches an MYR Bank Transaction of RM 5,607.00
    with a foreign USD document whose converted_amount_myr is RM 5,607.00.
    """
    db = SessionLocal()
    try:
        db.query(DocumentModel).filter(DocumentModel.document_id == "DOC-MATCH-USD-01").delete()
        db.commit()

        # Create test document
        test_doc = DocumentModel(
            company_id="COMP-000001",
            document_id="DOC-MATCH-USD-01",
            original_filename="OR-000002.pdf",
            file_path="/tmp/OR-000002.pdf",
            file_type="pdf",
            mime_type="application/pdf",
            document_type="Official Receipt",
            status="EXTRACTED",
            vendor_name="ACME Global Corp",
            currency="USD",
            grand_total=1400.0,
            converted_amount_myr=5607.00,
            exchange_rate=4.005,
            exchange_rate_date="2026-01-23",
            exchange_rate_provider="Bank Negara Malaysia (BNM OpenAPI)",
            conversion_status="CONVERTED",
            extracted_data_json=json.dumps({
                "receipt_number": "OR-000002",
                "customer_name": "ACME Global Corp",
                "buyer_name": "ACME Global Corp",
                "currency": "USD",
                "grand_total": 1400.0,
                "converted_amount_myr": 5607.00
            })
        )
        db.add(test_doc)
        db.commit()

        # Bank transaction in MYR
        bank_tx = BankTransactionModel(
            company_id="COMP-000001",
            bank_transaction_id="TX-BNK-991",
            transaction_date="2026-01-26",
            description="Payment received ACME Global Corp",
            reference="OR-000002",
            money_in=5607.00,
            money_out=0.0
        )

        match_res = match_bank_transaction_to_documents(bank_tx, db, "COMP-000001")
        assert match_res["match_status"] in ["MATCHED", "CONFIRMED"]
        assert match_res["matched_document_id"] == "DOC-MATCH-USD-01"
        assert match_res["match_confidence"] >= 0.8
    finally:
        db.close()


def test_retry_currency_conversion_endpoint():
    """
    Test Case 8: Verify POST /api/documents/{document_id}/retry-currency-conversion endpoint
    retries conversion, sets CONVERTED status, and updates converted_amount_myr.
    """
    db = SessionLocal()
    try:
        db.query(DocumentModel).filter(DocumentModel.document_id == "DOC-RETRY-TEST-01").delete()
        db.commit()

        # Create a document in FAILED state
        doc = DocumentModel(
            company_id="COMP-000001",
            document_id="DOC-RETRY-TEST-01",
            original_filename="Invoice-USD.pdf",
            file_path="/tmp/Invoice-USD.pdf",
            file_type="pdf",
            mime_type="application/pdf",
            document_type="Sales Invoice",
            status="NEEDS_REVIEW",
            vendor_name="Global Handbag Supplies",
            currency="USD",
            grand_total=1400.0,
            converted_amount_myr=None,
            conversion_status="FAILED",
            conversion_error="Network timeout on rate fetch",
            extracted_data_json=json.dumps({
                "invoice_number": "INV-USD-1400",
                "transaction_date": "2026-01-25",
                "currency": "USD",
                "grand_total": 1400.0,
                "subtotal": 1400.0
            })
        )
        db.add(doc)
        db.commit()

        # Call retry endpoint
        response = client.post("/api/documents/DOC-RETRY-TEST-01/retry-currency-conversion")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["document"]["converted_amount_myr"] == 5607.00
        assert data["document"]["exchange_rate"] == 4.005
        assert data["document"]["conversion_status"] == "CONVERTED"
    finally:
        db.close()
