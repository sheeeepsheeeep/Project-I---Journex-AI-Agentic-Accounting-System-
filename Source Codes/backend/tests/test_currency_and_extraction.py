import os
import sys

backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import pytest
from pathlib import Path
from decimal import Decimal

from app.services.document_agent import (
    DocumentProcessingAgent,
    is_address_line,
    is_valid_company_entity,
    detect_currency,
    generate_duplicate_key
)
from app.services.calculation_agent import CalculationTools, CalculationAgent
from app.agents.accounting_agent import AccountingAgent
from app.db.database import SessionLocal


@pytest.fixture
def doc_agent():
    return DocumentProcessingAgent(
        company_context={"company_id": "COMP-000001", "company_name": "NextTech Solutions Sdn Bhd"}
    )


@pytest.fixture
def calc_agent():
    return CalculationAgent()


def test_address_detection_and_company_validation():
    # International & Singapore addresses must be detected as address lines
    assert is_address_line("10 Anson Road, Singapore 079903") is True
    assert is_valid_company_entity("10 Anson Road, Singapore 079903") is False

    assert is_address_line("Level 12, Menara Tech, 50400 Kuala Lumpur") is True
    assert is_valid_company_entity("Level 12, Menara Tech, 50400 Kuala Lumpur") is False

    assert is_address_line("Jalan Tun Razak") is True
    assert is_valid_company_entity("Jalan Tun Razak") is False

    # Valid company names must NOT be flagged as addresses
    assert is_address_line("Singapore Digital Solutions Pte. Ltd.") is False
    assert is_valid_company_entity("Singapore Digital Solutions Pte. Ltd.") is True

    assert is_address_line("NextTech Solutions Sdn Bhd") is False
    assert is_valid_company_entity("NextTech Solutions Sdn Bhd") is True

    assert is_address_line("TechCore Solutions Sdn Bhd") is False
    assert is_valid_company_entity("TechCore Solutions Sdn Bhd") is True


def test_currency_detection():
    # Primary ISO currency codes
    assert detect_currency("Amount: SGD 500.00") == "SGD"
    assert detect_currency("Total: USD 1,200.00") == "USD"
    assert detect_currency("Balance: EUR 450.00") == "EUR"
    assert detect_currency("Net: GBP 300.00") == "GBP"
    assert detect_currency("Price: AUD 750.00") == "AUD"
    assert detect_currency("Charge: JPY 50,000") == "JPY"
    assert detect_currency("Subtotal: MYR 1,000.00") == "MYR"

    # Symbols
    assert detect_currency("Price: S$ 490.00") == "SGD"
    assert detect_currency("Price: RM 500.00") == "MYR"
    assert detect_currency("Total: €150.00") == "EUR"
    assert detect_currency("Total: £85.00") == "GBP"
    assert detect_currency("Total: ¥10,000") == "JPY"
    assert detect_currency("Total: $250.00") == "USD"

    # Default fallback
    assert detect_currency("Total Amount Due: 100.00") == "MYR"


def test_invoice_number_formats(doc_agent):
    formats = [
        ("INV-2026-0001", "Invoice # INV-2026-0001\nVendor: Global Tech\nTotal: RM 100.00"),
        ("INV-SGD-001", "# INV-SGD-001\nVendor: SG Dev Pte Ltd\nTotal: SGD 500.00"),
        ("INV-ABC123", "Tax Invoice\nINV-ABC123\nDate: 2026-10-01\nTotal: RM 200.00"),
        ("SI-00123", "Supplier Invoice\nSI-00123\nDate: 2026-10-01\nTotal: RM 300.00"),
    ]

    for expected_no, text in formats:
        extracted = doc_agent._extract_raw_fields("TEST-DOC", text, "Purchase Invoice", {})
        assert extracted.get("invoice_number") == expected_no, f"Failed for {expected_no}"


def test_decimal_calculation_with_discount_and_shipping():
    # 500.00 - 20.00 + 10.00 + 0.00 = 490.00
    calc_total = CalculationTools.calculate_invoice_total(
        subtotal=500.00,
        tax_amount=0.00,
        discount=20.00,
        shipping=10.00
    )
    assert calc_total == 490.00

    # With tax: 1000.00 - 50.00 + 20.00 + 60.00 = 1030.00
    calc_tax_total = CalculationTools.calculate_invoice_total(
        subtotal=1000.00,
        tax_amount=60.00,
        discount=50.00,
        shipping=20.00
    )
    assert calc_tax_total == 1030.00


def test_calculation_agent_verification(calc_agent):
    doc_data = {
        "document_type": "Purchase Invoice",
        "currency": "SGD",
        "subtotal": 500.00,
        "discount": 20.00,
        "shipping": 10.00,
        "tax_amount": 0.00,
        "grand_total": 490.00,
        "line_items": [
            {
                "item_description": "Website Development Service",
                "quantity": 1.0,
                "unit_price": 500.00,
                "total_amount": 500.00
            }
        ]
    }

    verification = calc_agent.process_and_verify(doc_data)
    assert verification.overall_status == "PASSED"
    assert verification.math_verified is True
    assert verification.calculated_total == 490.00
    assert verification.document_total == 490.00
    assert verification.difference == 0.0


def test_calculation_agent_flags_discrepancy(calc_agent):
    doc_data = {
        "document_type": "Purchase Invoice",
        "currency": "SGD",
        "subtotal": 500.00,
        "discount": 20.00,
        "shipping": 10.00,
        "tax_amount": 0.00,
        "grand_total": 520.00,  # Intentional discrepancy: calculated is 490.00
        "line_items": [
            {
                "item_description": "Website Development Service",
                "quantity": 1.0,
                "unit_price": 500.00,
                "total_amount": 500.00
            }
        ]
    }

    verification = calc_agent.process_and_verify(doc_data)
    assert verification.overall_status == "FAILED"
    assert verification.math_verified is False
    assert verification.grand_total_check is False
    assert len(verification.details) > 0
    assert "Grand Total Discrepancy" in verification.details[0]


def test_sgd_invoice_extraction_full(doc_agent):
    invoice_text = """Singapore Digital Solutions Pte. Ltd. INVOICE
10 Anson Road, Singapore 079903
# INV-SGD-001
Date: Oct 22, 2026
Bill To:
Payment Terms: Payment within 30
NextTech Solutions Sdn Bhd
days.
Level 12, Menara Tech
Jalan Tun Razak
Balance Due: SGD 490.00
50400 Kuala Lumpur
Malaysia
Item Quantity Rate Amount
Website Development Service 1 SGD 500.00 SGD 500.00
Subtotal: SGD 500.00
Discount: SGD 20.00
Shipping: SGD 10.00
Total: SGD 490.00"""

    structured_data, passed, val_msgs = doc_agent.process_document("TEST-DOC-SGD", invoice_text, "Invoice INV-SGD-001.pdf")

    assert structured_data.document_type == "Purchase Invoice"
    assert structured_data.vendor_name == "Singapore Digital Solutions Pte. Ltd."
    assert structured_data.vendor_address == "10 Anson Road, Singapore 079903"
    assert structured_data.buyer_name == "NextTech Solutions Sdn Bhd"
    assert structured_data.invoice_number == "INV-SGD-001"
    assert structured_data.currency == "SGD"
    assert structured_data.subtotal == 500.00
    assert structured_data.discount == 20.00
    assert structured_data.shipping == 10.00
    assert structured_data.grand_total == 490.00
    assert len(structured_data.line_items) == 1
    assert structured_data.line_items[0].item_description == "Website Development Service"
    assert structured_data.line_items[0].unit_price == 500.00
    assert structured_data.line_items[0].total_amount == 500.00
    assert structured_data.display_summary.party == "Singapore Digital Solutions Pte. Ltd."
    assert structured_data.display_summary.amount == 490.00
    assert passed is True
    assert structured_data.calculation_verification.overall_status == "PASSED"


def test_duplicate_key_alphanumeric_invoice():
    extracted = {
        "invoice_number": "INV-SGD-001",
        "vendor_name": "Singapore Digital Solutions Pte. Ltd.",
        "business_roles": {
            "vendor": "Singapore Digital Solutions Pte. Ltd."
        }
    }
    key, label, val = generate_duplicate_key("COMP-000001", "Purchase Invoice", extracted)
    assert key == "COMP-000001|PURCHASE_INVOICE|SINGAPORE-DIGITAL-SOLUTIONS-PTE.-LTD.|INV-SGD-001"
    assert label == "Invoice Number"
    assert val == "INV-SGD-001"


def test_accounting_classification_purchase_invoice():
    db = SessionLocal()
    try:
        acct = AccountingAgent(db=db, company_id="COMP-000001")
        input_data = {
            "document_type": "Purchase Invoice",
            "invoice_number": "INV-SGD-001",
            "transaction_date": "Oct 22, 2026",
            "vendor_name": "Singapore Digital Solutions Pte. Ltd.",
            "currency": "SGD",
            "subtotal": 500.00,
            "discount": 20.00,
            "shipping": 10.00,
            "tax_amount": 0.00,
            "grand_total": 490.00,
            "business_roles": {
                "our_company": "NextTech Solutions Sdn Bhd",
                "vendor": "Singapore Digital Solutions Pte. Ltd."
            }
        }
        analysis = acct.analyze(input_data)
        assert analysis.accounting_event == "Credit purchase of goods/services"
        assert analysis.book_of_prime_entry == "Purchases Day Book"
        assert analysis.confidence_score >= 0.90
        codes = [acc.account_code for acc in analysis.selected_accounts]
        assert "610-0000" in codes  # Purchases
        assert "400-0000" in codes  # Trade Creditors
    finally:
        db.close()
