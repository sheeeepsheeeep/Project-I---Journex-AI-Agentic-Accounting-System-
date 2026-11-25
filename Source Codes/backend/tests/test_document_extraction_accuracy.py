import os
import json
import pytest
from app.services.pdf_ocr import extract_document_data
from app.services.document_agent import DocumentProcessingAgent
from app.services.calculation_agent import CalculationAgent
from app.agents.accounting_agent import AccountingAgent
from app.api.documents import review_or_update_document, ReviewDocumentRequest
from app.db.database import SessionLocal
from app.db.models import DocumentModel, CompanyModel

SCENARIO_FOLDER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "Sample Scenario - Ms Wani",
    "Upload"
)

ACTIVE_COMPANY = {
    "company_id": "COMP-000002",
    "company_name": "0126 HANDBAG SOLUTIONS SDN. BHD.",
    "registration_number": "202601000126",
    "country": "Malaysia",
    "base_currency": "MYR"
}


def test_stock_take_extraction_and_math():
    """Verify Stock Take (SK-000001) extracts columns, handles missing physical qty, and verifies signed net adjustment."""
    pdf_path = os.path.join(SCENARIO_FOLDER, "SK-000001 (3101).pdf")
    assert os.path.exists(pdf_path), f"File not found: {pdf_path}"

    ocr_res = extract_document_data(pdf_path)
    raw_text = ocr_res["raw_text"]
    agent = DocumentProcessingAgent(company_context=ACTIVE_COMPANY)
    ext, needs_review, msgs = agent.process_document(
        document_id="DOC-TEST-SK",
        raw_text=raw_text,
        filename="SK-000001 (3101).pdf"
    )

    # 1. Document classification
    assert ext.document_type == "Stock Take"
    assert ext.document_number == "SK-000001"
    assert ext.company_match_status == "Matched"

    # 2. Line items extracted
    assert len(ext.line_items) == 8

    # 3. Clutch Premium missing physical count
    clutch = next(item for item in ext.line_items if "Clutch (Premium)" in item.description)
    assert clutch.balance_quantity == 4.0
    assert clutch.physical_quantity is None
    assert clutch.difference_quantity is None
    assert clutch.missing_physical is True
    assert clutch.amount == 0.0

    # 4. Calculation verification & signed net adjustment
    calc_agent = CalculationAgent()
    calc_res = calc_agent.process_and_verify(ext.model_dump())
    assert calc_res.calculation_required is True
    assert calc_res.math_verified is True
    assert round(calc_res.document_total, 2) == -177043.23
    assert round(calc_res.calculated_total, 2) == -177043.23

    # 5. Accounting agent classification
    db = SessionLocal()
    try:
        acct_agent = AccountingAgent(db, ACTIVE_COMPANY["company_id"])
        acct_res = acct_agent.analyze(ext)
        assert "stock" in acct_res.accounting_event.lower() or "inventory" in acct_res.accounting_event.lower()
        assert acct_res.book_of_prime_entry in ["Inventory / Stock Take Register", "Stock Take Register", "Inventory Module"]
    finally:
        db.close()


def test_official_receipt_company_identification():
    """Verify Official Receipt (OR-000003) Malaysian signature block, roles, and accounting category."""
    pdf_path = os.path.join(SCENARIO_FOLDER, "OR-000003 (2601).pdf")
    assert os.path.exists(pdf_path), f"File not found: {pdf_path}"

    ocr_res = extract_document_data(pdf_path)
    raw_text = ocr_res["raw_text"]
    agent = DocumentProcessingAgent(company_context=ACTIVE_COMPANY)
    ext, needs_review, msgs = agent.process_document(
        document_id="DOC-TEST-OR",
        raw_text=raw_text,
        filename="OR-000003 (2601).pdf"
    )

    assert ext.document_type == "Official Receipt"
    assert ext.receipt_number == "OR-000003"
    assert ext.company_match_status == "Matched"
    assert ext.business_roles.customer == "Fussy Fashion House Sdn. Bhd."
    assert ext.business_roles.our_company == "0126 HANDBAG SOLUTIONS SDN. BHD."
    assert ext.account_code == "300-9000"
    assert ext.payment_description == "Paid full"
    assert ext.amount_received == 3500.0

    db = SessionLocal()
    try:
        acct_agent = AccountingAgent(db, ACTIVE_COMPANY["company_id"])
        acct_res = acct_agent.analyze(ext)
        assert "receipt" in acct_res.accounting_event.lower() or "payment" in acct_res.accounting_event.lower()
        assert acct_res.book_of_prime_entry == "Cash Book"
    finally:
        db.close()


def test_payment_voucher_extraction():
    """Verify Payment Voucher (PV-000001) extraction, Cash Book prime entry, and matching."""
    pdf_path = os.path.join(SCENARIO_FOLDER, "PV-000001 (2801).pdf")
    assert os.path.exists(pdf_path), f"File not found: {pdf_path}"

    ocr_res = extract_document_data(pdf_path)
    raw_text = ocr_res["raw_text"]
    agent = DocumentProcessingAgent(company_context=ACTIVE_COMPANY)
    ext, needs_review, msgs = agent.process_document(
        document_id="DOC-TEST-PV",
        raw_text=raw_text,
        filename="PV-000001 (2801).pdf"
    )

    assert ext.document_type == "Payment Voucher"
    assert ext.voucher_number == "PV-000001"
    assert ext.amount_paid == 570.0
    assert ext.company_match_status == "Matched"

    db = SessionLocal()
    try:
        acct_agent = AccountingAgent(db, ACTIVE_COMPANY["company_id"])
        acct_res = acct_agent.analyze(ext)
        assert acct_res.book_of_prime_entry == "Cash Book"
    finally:
        db.close()


def test_delivery_order_and_purchase_orders():
    """Verify Delivery Order (DO-000001) and Purchase Order (PO-000001)."""
    # DO-000001
    pdf_path_do = os.path.join(SCENARIO_FOLDER, "DO-000001 (2601).pdf")
    ocr_do = extract_document_data(pdf_path_do)
    agent = DocumentProcessingAgent(company_context=ACTIVE_COMPANY)
    ext_do, _, _ = agent.process_document(document_id="DOC-TEST-DO", raw_text=ocr_do["raw_text"], filename="DO-000001 (2601).pdf")
    assert ext_do.document_type == "Delivery Order"
    assert ext_do.delivery_order_number == "DO-000001"
    assert ext_do.business_roles.customer == "Fussy Fashion House Sdn. Bhd."

    # PO-000001
    pdf_path_po = os.path.join(SCENARIO_FOLDER, "PO-000001 (0301).pdf")
    ocr_po = extract_document_data(pdf_path_po)
    ext_po, _, _ = agent.process_document(document_id="DOC-TEST-PO", raw_text=ocr_po["raw_text"], filename="PO-000001 (0301).pdf")
    assert ext_po.document_type == "Purchase Order"
    assert ext_po.purchase_order_number == "PO-000001"
    assert ext_po.grand_total == 110500.0


def test_all_scenario_documents_batch():
    """Verify all 13 scenario documents extract cleanly with valid document types and statuses."""
    expected = {
        "DO-000001 (2601).pdf": ("Delivery Order", 3.0),
        "I-000004 (2101).pdf": ("Sales Invoice", 1400.0),
        "I-000006 (2602).pdf": ("Sales Invoice", 7000.0),
        "OR-000002 (2501).pdf": ("Official Receipt", 1400.0),
        "OR-000003 (2601).pdf": ("Official Receipt", 3500.0),
        "PI-000005 (1001).pdf": ("Purchase Invoice", 42800.0),
        "PO-000001 (0301).pdf": ("Purchase Order", 110500.0),
        "PO-000003 (0501).pdf": ("Purchase Order", 11800.0),
        "PR-000001 (1001).pdf": ("Purchase Return", 800.0),
        "PV-000001 (2801).pdf": ("Payment Voucher", 570.0),
        "SK-000001 (3101).pdf": ("Stock Take", -177043.23),
        "SO-000001 (0601).pdf": ("Sales Order", 7596.0),
        "SO-000002 (1101).pdf": ("Sales Order", 6200.0),
    }

    agent = DocumentProcessingAgent(company_context=ACTIVE_COMPANY)
    calc_agent = CalculationAgent()

    for filename, (expected_type, expected_amount) in expected.items():
        pdf_path = os.path.join(SCENARIO_FOLDER, filename)
        assert os.path.exists(pdf_path), f"Missing scenario file: {filename}"

        ocr = extract_document_data(pdf_path)
        ext, needs_rev, msgs = agent.process_document(f"DOC-{filename}", ocr["raw_text"], filename)
        assert ext.document_type == expected_type, f"Mismatch in {filename}: got {ext.document_type}, expected {expected_type}"

        # Calculation check
        calc = calc_agent.process_and_verify(ext.model_dump())
        assert calc.math_verified is True, f"Math verification failed for {filename}"

        # Status check
        assert ext.validation_status in ["Passed", "Review Required"]
        assert ext.company_match_status in ["Matched", "Unable to Determine"]


def test_review_endpoint_stock_take_recalculation():
    """Verify that submitting a review for Stock Take recalculates difference, amount, and updates review status."""
    db = SessionLocal()
    try:
        # Create a test document in db
        doc_id = "DOC-TEST-REVIEW-SK"
        db.query(DocumentModel).filter(DocumentModel.document_id == doc_id).delete()
        db.commit()

        initial_items = [
            {"item_number": "1", "description": "Tote Bag", "balance_quantity": 10.0, "physical_quantity": 8.0, "unit_cost": 100.0, "amount": -200.0, "difference_quantity": -2.0},
            {"item_number": "2", "description": "Clutch (Premium)", "balance_quantity": 4.0, "physical_quantity": None, "unit_cost": 500.0, "amount": 0.0, "difference_quantity": None, "missing_physical": True}
        ]

        doc = DocumentModel(
            document_id=doc_id,
            company_id=ACTIVE_COMPANY["company_id"],
            document_type="Stock Take",
            original_filename="SK-TEST.pdf",
            file_path="uploads/SK-TEST.pdf",
            file_type="pdf",
            mime_type="application/pdf",
            grand_total=-200.0,
            status="NEEDS_REVIEW",
            extracted_data_json=json.dumps({
                "document_type": "Stock Take",
                "document_number": "SK-TEST",
                "line_items": initial_items,
                "grand_total": -200.0,
                "subtotal": -200.0
            })
        )
        db.add(doc)
        db.commit()

        # Review: user enters physical quantity 5.0 for Clutch (Premium)
        reviewed_items = [
            {"item_number": "1", "description": "Tote Bag", "balance_quantity": 10.0, "physical_quantity": 8.0, "unit_cost": 100.0, "difference_quantity": -2.0, "amount": -200.0},
            {"item_number": "2", "description": "Clutch (Premium)", "balance_quantity": 4.0, "physical_quantity": 5.0, "unit_cost": 500.0, "difference_quantity": 1.0, "amount": 500.0, "missing_physical": False}
        ]

        req = ReviewDocumentRequest(
            document_number="SK-TEST",
            line_items=reviewed_items,
            review_status="Verified"
        )

        res = review_or_update_document(doc_id, req, db)
        
        # Verify grand_total updated (-200 + 500 = +300.0)
        assert res.grand_total == 300.0
        assert res.review_status == "Verified"
        assert res.status == "COMPLETED"

        # Clean up
        db.query(DocumentModel).filter(DocumentModel.document_id == doc_id).delete()
        db.commit()
    finally:
        db.close()
