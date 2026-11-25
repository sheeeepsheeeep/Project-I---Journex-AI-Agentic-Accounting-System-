import os
import sys
from pathlib import Path

# Add backend directory to sys.path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import pytest
from decimal import Decimal
from unittest.mock import patch, MagicMock

from app.services.currency_service import (
    CurrencyExchangeService,
    CurrencyExchangeError,
    ExchangeRateResult
)
from app.agents.accounting_agent import AccountingAgent
from app.schemas.document import ExtractedDocumentData


def test_bnm_session_1700_sgd_rate():
    """
    Test 1: Verify Bank Negara Malaysia (BNM OpenAPI) Session 1700 official rate
    retrieval for SGD on 2026-10-22.
    """
    res = CurrencyExchangeService.get_exchange_rate("SGD", "2026-10-22", session="1700")
    assert res.currency == "SGD"
    assert res.rate == 3.1967
    assert res.unit == 1
    assert res.rate_date == "2026-10-22"
    assert res.session == "1700"
    assert "Bank Negara Malaysia" in res.provider
    assert res.is_fallback is False


def test_currency_unit_handling_jpy():
    """
    Test 2: Verify unit normalization (rate / unit), especially for currencies quoted per 100 units like JPY.
    """
    res = CurrencyExchangeService.get_exchange_rate("JPY", "2026-10-22", session="1700")
    assert res.currency == "JPY"
    assert res.unit == 100
    # JPY raw rate is ~2.5999 per 100 units, effective rate must be divided by 100
    assert abs(res.rate - 0.025999) < 0.0001
    assert res.rate == float(Decimal(str(res.raw_rate)) / Decimal(str(res.unit)))


def test_weekend_holiday_lookback():
    """
    Test 3: Verify lookback behavior on weekends / public holidays.
    2026-10-25 is a Sunday. The service must look back and find the Friday rate (2026-10-23).
    """
    res = CurrencyExchangeService.get_exchange_rate("SGD", "2026-10-25", session="1700")
    assert res.currency == "SGD"
    assert res.rate_date == "2026-10-23"
    assert res.rate == 3.1947
    assert "Bank Negara Malaysia" in res.provider


def test_frankfurter_fallback_when_bnm_unavailable():
    """
    Test 4: Verify fallback to Frankfurter API when BNM is unavailable.
    """
    with patch.object(CurrencyExchangeService, "_fetch_bnm_rate", return_value=None):
        res = CurrencyExchangeService.get_exchange_rate("SGD", "2026-10-22")
        assert res.currency == "SGD"
        assert res.is_fallback is True
        assert "Frankfurter" in res.provider
        assert res.rate > 0.0


def test_strict_safeguard_both_fail():
    """
    Test 5: Verify strict safeguard: if both BNM and Frankfurter fail to return a usable rate,
    the system must stop foreign-currency posting and raise CurrencyExchangeError (never silently treat as MYR).
    """
    with patch.object(CurrencyExchangeService, "_fetch_bnm_rate", return_value=None):
        with patch.object(CurrencyExchangeService, "_fetch_frankfurter_rate", return_value=None):
            with pytest.raises(CurrencyExchangeError) as exc_info:
                CurrencyExchangeService.get_exchange_rate("SGD", "2026-10-22")
            assert "Foreign currency posting halted" in str(exc_info.value)


def test_convert_document_amounts_with_rounding_reconciliation():
    """
    Test 6: Verify dual-amount conversion with rounding reconciliation:
    Subtotal: SGD 500.00 -> MYR 1598.35
    Discount: SGD 20.00  -> MYR 63.93
    Shipping: SGD 10.00  -> MYR 31.97
    Grand Total: SGD 490.00 -> Converted MYR 1566.39
    """
    conv = CurrencyExchangeService.convert_document_amounts(
        currency="SGD",
        transaction_date="2026-10-22",
        grand_total=490.0,
        subtotal=500.0,
        discount=20.0,
        shipping=10.0,
        tax_amount=0.0
    )

    assert conv["currency"] == "SGD"
    assert conv["exchange_rate"] == 3.1967
    assert conv["subtotal_myr"] == 1598.35
    assert conv["discount_myr"] == 63.93
    assert conv["shipping_myr"] == 31.97
    assert conv["tax_amount_myr"] == 0.0
    assert conv["converted_amount_myr"] == 1566.39
    assert conv["is_foreign"] is True


def test_accounting_agent_journal_balancing_foreign_currency():
    """
    Test 7: Verify AccountingAgent creates a perfectly balanced Journal Entry
    with Carriage Inwards (615-0000), Discount Received (540-0000), Purchases (610-0000),
    and Trade Creditors (400-0000), reconciling any 1-cent rounding discrepancy.
    """
    agent = AccountingAgent(company_id="COMP-000001")

    doc_data = {
        "company_id": "COMP-000001",
        "document_id": "TEST-SGD-001",
        "document_type": "Purchase Invoice",
        "vendor_name": "Singapore Digital Solutions Pte. Ltd.",
        "currency": "SGD",
        "transaction_date": "2026-10-22",
        "invoice_number": "INV-SGD-001",
        "subtotal": 500.0,
        "discount": 20.0,
        "shipping": 10.0,
        "tax_amount": 0.0,
        "grand_total": 490.0,
        "subtotal_myr": 1598.35,
        "discount_myr": 63.93,
        "shipping_myr": 31.97,
        "tax_amount_myr": 0.0,
        "converted_amount_myr": 1566.39,
        "exchange_rate": 3.1967,
        "exchange_rate_provider": "Bank Negara Malaysia (BNM OpenAPI)",
        "line_items": [{"item_description": "Website Development Service", "quantity": 1, "unit_price": 500.0, "total_amount": 500.0}]
    }

    analysis = agent.analyze(doc_data)
    assert analysis.needs_review is False
    assert analysis.journal_entry is not None

    je = analysis.journal_entry
    assert je.is_balanced is True
    assert je.total_debit == je.total_credit
    assert je.total_debit == 1630.32

    line_map = {l.account_code: (l.debit, l.credit) for l in je.lines}
    assert line_map["610-0000"] == (1598.35, 0.0)  # Purchases
    assert line_map["615-0000"] == (31.97, 0.0)    # Carriage Inwards
    assert line_map["540-0000"] == (0.0, 63.93)    # Discount Received
    assert line_map["400-0000"] == (0.0, 1566.39)  # Trade Creditors


def test_domestic_myr_regression():
    """
    Test 8: Verify domestic MYR invoices continue working as before without unnecessary API calls.
    """
    conv = CurrencyExchangeService.convert_document_amounts(
        currency="MYR",
        transaction_date="2026-10-22",
        grand_total=1000.0,
        subtotal=1000.0,
        discount=50.0,
        shipping=20.0,
        tax_amount=60.0
    )
    assert conv["currency"] == "MYR"
    assert conv["exchange_rate"] == 1.0
    assert conv["converted_amount_myr"] == 1000.0
    assert conv["subtotal_myr"] == 1000.0
    assert conv["discount_myr"] == 50.0
    assert conv["shipping_myr"] == 20.0
    assert conv["tax_amount_myr"] == 60.0
    assert conv["is_foreign"] is False

    agent = AccountingAgent(company_id="COMP-000001")
    myr_doc = {
        "company_id": "COMP-000001",
        "document_id": "TEST-MYR-001",
        "document_type": "Purchase Invoice",
        "vendor_name": "ABC Office Supplies Sdn Bhd",
        "currency": "MYR",
        "transaction_date": "2026-10-22",
        "invoice_number": "INV-MYR-001",
        "subtotal": 1000.0,
        "discount": 50.0,
        "shipping": 20.0,
        "tax_amount": 60.0,
        "grand_total": 1030.0,
        "line_items": [{"item_description": "Office paper and supplies", "quantity": 10, "unit_price": 100.0, "total_amount": 1000.0}]
    }
    analysis = agent.analyze(myr_doc)
    assert analysis.needs_review is False
    assert analysis.journal_entry is not None
    je = analysis.journal_entry
    assert je.is_balanced is True
    assert je.total_debit == je.total_credit == 1080.0
