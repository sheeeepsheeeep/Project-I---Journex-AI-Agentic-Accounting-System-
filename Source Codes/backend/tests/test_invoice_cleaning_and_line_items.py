import pytest
from app.services.document_agent import DocumentProcessingAgent, clean_invoice_number
from app.services.pdf_ocr import extract_document_data
from pathlib import Path


def test_clean_invoice_number_variants():
    """Verify various dirty invoice label formats are cleaned accurately."""
    assert clean_invoice_number(". : I-000004") == "I-000004"
    assert clean_invoice_number("No. : I-000004") == "I-000004"
    assert clean_invoice_number("No.       : I-000004") == "I-000004"
    assert clean_invoice_number("Invoice No. : I-000004") == "I-000004"
    assert clean_invoice_number("Invoice Number : INV-2026-0001") == "INV-2026-0001"
    assert clean_invoice_number(": PI-000005") == "PI-000005"
    assert clean_invoice_number("# INV-SGD-001") == "INV-SGD-001"
    
    # Must reject unwanted references
    assert clean_invoice_number("Your Ref. : PO-001") is None
    assert clean_invoice_number("Our D/O No. : DO-001") is None
    assert clean_invoice_number("Terms : Net 30 days") is None


def test_angela_boutique_invoice_extraction():
    """Verify end-to-end extraction on Angela's Boutique PDF."""
    candidates = [
        Path("Sample Scenario - Ms Wani/Upload/I-000004 (2101).pdf"),
        Path("../Sample Scenario - Ms Wani/Upload/I-000004 (2101).pdf"),
        Path("Agentic-AI-Accounting-System/Sample Scenario - Ms Wani/Upload/I-000004 (2101).pdf"),
    ]
    pdf_path = next((p for p in candidates if p.exists()), None)
    if not pdf_path:
        pytest.skip("Sample PDF not found")
        
    data = extract_document_data(str(pdf_path))
    agent = DocumentProcessingAgent(company_context={
        "company_name": "0126 Handbag Solutions Sdn. Bhd.",
        "company_id": "COMP-000002"
    })
    doc, passed, msgs = agent.process_document("DOC-TEST-ANGELA", data["raw_text"], "I-000004 (2101).pdf")

    assert doc.document_type == "Sales Invoice"
    assert doc.invoice_number == "I-000004"
    assert doc.invoice_date == "2026-01-21"
    assert doc.payment_terms == "Net 30 days"
    assert doc.currency == "USD"
    assert doc.business_roles.customer == "Angela's Boutique"
    assert doc.grand_total == 1400.0
    assert doc.subtotal == 1400.0

    # Line items
    assert len(doc.line_items) == 2
    item1 = doc.line_items[0]
    assert item1.item_number == "1"
    assert item1.description == "Travel Bag"
    assert item1.quantity == 1.0
    assert item1.uom == "UNIT"
    assert item1.unit_price == 600.0
    assert item1.discount is None
    assert item1.line_total == 600.0

    item2 = doc.line_items[1]
    assert item2.item_number == "2"
    assert item2.description == "Sling Bag"
    assert item2.quantity == 2.0
    assert item2.uom == "UNIT"
    assert item2.unit_price == 400.0
    assert item2.discount is None
    assert item2.line_total == 800.0

    # Calculation verification
    assert doc.math_verified is True
    assert doc.calculation_verification.overall_status == "PASSED"
    assert len(msgs) == 0


def test_discounted_line_item_extraction():
    """Verify that line items with discounts (e.g. SO-000001) are handled properly."""
    candidates = [
        Path("Sample Scenario - Ms Wani/Upload/SO-000001 (0601).pdf"),
        Path("../Sample Scenario - Ms Wani/Upload/SO-000001 (0601).pdf"),
        Path("Agentic-AI-Accounting-System/Sample Scenario - Ms Wani/Upload/SO-000001 (0601).pdf"),
    ]
    pdf_path = next((p for p in candidates if p.exists()), None)
    if not pdf_path:
        pytest.skip("Sample PDF not found")
        
    data = extract_document_data(str(pdf_path))
    agent = DocumentProcessingAgent(company_context={
        "company_name": "0126 Handbag Solutions Sdn. Bhd.",
        "company_id": "COMP-000002"
    })
    doc, passed, msgs = agent.process_document("DOC-TEST-SO", data["raw_text"], "SO-000001 (0601).pdf")

    assert len(doc.line_items) == 2
    # Item 2 has discount 4.0: 2 * 2800 - 4 = 5596
    it2 = doc.line_items[1]
    assert it2.description == "Mini Handbag"
    assert it2.quantity == 2.0
    assert it2.unit_price == 2800.0
    assert it2.discount == 4.0
    assert it2.line_total == 5596.0
