"""
Integration & Unit Tests for GmailAttachmentIngestionService (Stage 5).
Verifies attachment ingestion into the existing document pipeline:
- PDF generation via ReportLab (synthetic valid invoice)
- Pipeline execution: Text Extraction -> DocumentProcessingAgent -> AccountingAgent -> DB & Transaction Sync
- Duplicate prevention (Message ID and SHA-256 hash across emails)
- Unrelated / Uncertain email filtering
- Multiple attachments
- Error handling (empty, invalid signature, unsupported types)
- Source traceability
"""

import sys
import io
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

# ReportLab to generate valid synthetic PDFs
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Ensure backend folder is in path
backend_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_path))

from app.db.models import (
    Base,
    CompanyModel,
    DocumentModel,
    EmailSourceModel,
    EmailAttachmentModel,
    AccountingTransactionModel,
)
from app.services.email_ingestion.ingestion_service import GmailIngestionService
from app.services.email_ingestion.gmail_client import GmailClient


def generate_synthetic_invoice_pdf(
    invoice_no: str = "INV-2026-9001",
    vendor_name: str = "Apex Office Supplies Sdn Bhd",
    buyer_name: str = "NextTech Solutions Sdn Bhd",
    grand_total: float = 212.0
) -> bytes:
    """Generates a valid binary PDF with recognizable invoice structure and text."""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 750, "TAX INVOICE")

    c.setFont("Helvetica", 10)
    c.drawString(100, 720, f"Invoice Number: {invoice_no}")
    c.drawString(100, 705, "Invoice Date: 20 October 2026")
    c.drawString(100, 685, f"Vendor: {vendor_name}")
    c.drawString(100, 670, f"Bill To: {buyer_name}")

    c.drawString(100, 630, "Description                      Qty    Unit Price    Total")
    c.drawString(100, 615, f"Ergonomic Office Chair             2        100.00   200.00")

    c.drawString(100, 570, "Subtotal: MYR 200.00")
    c.drawString(100, 555, "Tax (SST 6%): MYR 12.00")
    c.drawString(100, 540, f"Grand Total: MYR {grand_total:.2f}")

    c.showPage()
    c.save()
    return buffer.getvalue()


def create_test_db():
    """Creates an isolated in-memory/temp SQLite database session for testing."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Seed active company
    comp = CompanyModel(
        company_id="COMP-000001",
        company_name="NextTech Solutions Sdn Bhd",
        currency="MYR",
        financial_year=2026
    )
    session.add(comp)
    session.commit()
    return session


def test_1_eligible_invoice_enters_pipeline():
    """1. An eligible invoice attachment enters the existing document processing pipeline."""
    db = create_test_db()
    mock_gmail = MagicMock(spec=GmailClient)

    invoice_bytes = generate_synthetic_invoice_pdf(invoice_no="INV-2026-9001")
    service = GmailIngestionService(db=db, gmail_client=mock_gmail)

    email_detail = {
        "message_id": "msg_invoice_001",
        "thread_id": "th_001",
        "sender_email": "billing@apexsupplies.com",
        "sender_name": "Apex Supplies Billing",
        "subject": "Tax Invoice INV-2026-9001 from Apex Office Supplies",
        "snippet": "Dear Customer, please find attached your invoice for office chairs.",
        "received_at": "2026-10-20T10:00:00Z",
        "attachments": [
            {
                "attachment_id": "att_inv_1",
                "filename": "INV-2026-9001.pdf",
                "mime_type": "application/pdf",
                "size_bytes": len(invoice_bytes),
                "raw_bytes": invoice_bytes
            }
        ]
    }

    result = service.process_email(email_detail)

    assert result["status"] == "PROCESSED"
    assert result["attachments_processed"] == 1
    assert result["attachments_skipped"] == 0
    assert len(result["documents_created"]) == 1

    created_doc_id = result["documents_created"][0]["document_id"]
    assert created_doc_id.startswith("DOC-2026-")

    # Verify DocumentModel record
    doc = db.query(DocumentModel).filter(DocumentModel.document_id == created_doc_id).first()
    assert doc is not None
    assert doc.original_filename == "INV-2026-9001.pdf"
    assert doc.status in ["EXTRACTED", "COMPLETED"]
    assert doc.source_email_id is not None

    # Verify EmailSourceModel record
    source = db.query(EmailSourceModel).filter(EmailSourceModel.gmail_message_id == "msg_invoice_001").first()
    assert source is not None
    assert source.classification == "ACCOUNTING_RELATED"
    assert source.ingestion_status == "PROCESSED"
    assert doc.source_email_id == source.id

    # Verify EmailAttachmentModel record
    email_att = db.query(EmailAttachmentModel).filter(EmailAttachmentModel.gmail_message_id == "msg_invoice_001").first()
    assert email_att is not None
    assert email_att.document_id == created_doc_id
    assert email_att.status == "PROCESSED"

    # Verify transaction was synchronized
    txn = db.query(AccountingTransactionModel).filter(AccountingTransactionModel.document_id == created_doc_id).first()
    assert txn is not None


def test_2_unrelated_email_skipped():
    """2. An unrelated meeting email is skipped without creating document records."""
    db = create_test_db()
    service = GmailIngestionService(db=db)

    meeting_pdf = generate_synthetic_invoice_pdf()
    email_detail = {
        "message_id": "msg_meeting_999",
        "sender_email": "pm@consulting.com",
        "subject": "Sprint planning agenda and meeting schedule",
        "snippet": "Here is the agenda for our upcoming sprint review.",
        "attachments": [
            {
                "attachment_id": "att_agenda",
                "filename": "agenda.pdf",
                "mime_type": "application/pdf",
                "raw_bytes": meeting_pdf
            }
        ]
    }

    result = service.process_email(email_detail)
    assert result["status"] == "SKIPPED_UNRELATED"
    assert len(result["documents_created"]) == 0

    source = db.query(EmailSourceModel).filter(EmailSourceModel.gmail_message_id == "msg_meeting_999").first()
    assert source.ingestion_status == "SKIPPED_UNRELATED"
    assert db.query(DocumentModel).count() == 0


def test_3_uncertain_email_deferred():
    """3. An uncertain email is deferred safely."""
    db = create_test_db()
    service = GmailIngestionService(db=db)

    email_detail = {
        "message_id": "msg_vague_444",
        "sender_email": "partner@unknown.com",
        "subject": "Files for your reference",
        "snippet": "Please check attached files.",
        "attachments": [
            {
                "attachment_id": "att_scan",
                "filename": "scan_document.pdf",
                "mime_type": "application/pdf",
                "raw_bytes": generate_synthetic_invoice_pdf()
            }
        ]
    }

    result = service.process_email(email_detail)
    assert result["status"] == "DEFERRED_UNCERTAIN"
    assert len(result["documents_created"]) == 0

    source = db.query(EmailSourceModel).filter(EmailSourceModel.gmail_message_id == "msg_vague_444").first()
    assert source.ingestion_status == "DEFERRED_UNCERTAIN"
    assert db.query(DocumentModel).count() == 0


def test_4_multiple_attachments_in_one_email():
    """4. Multiple distinct invoice attachments in one email are both processed independently."""
    db = create_test_db()
    service = GmailIngestionService(db=db)

    pdf1 = generate_synthetic_invoice_pdf(invoice_no="INV-2026-0001", grand_total=100.0)
    pdf2 = generate_synthetic_invoice_pdf(invoice_no="INV-2026-0002", grand_total=250.0)

    email_detail = {
        "message_id": "msg_multi_inv_77",
        "sender_email": "accounts@vendor.com",
        "subject": "Invoices for September: INV-0001 and INV-0002",
        "snippet": "Attached are your monthly invoices.",
        "attachments": [
            {"attachment_id": "att_1", "filename": "INV-0001.pdf", "mime_type": "application/pdf", "raw_bytes": pdf1},
            {"attachment_id": "att_2", "filename": "INV-0002.pdf", "mime_type": "application/pdf", "raw_bytes": pdf2}
        ]
    }

    result = service.process_email(email_detail)
    assert result["status"] == "PROCESSED"
    assert result["attachments_processed"] == 2
    assert len(result["documents_created"]) == 2

    doc_ids = [d["document_id"] for d in result["documents_created"]]
    assert len(set(doc_ids)) == 2  # Unique system document IDs!
    assert db.query(DocumentModel).count() == 2


def test_5_duplicate_message_id():
    """5. Re-fetching an already processed Gmail message ID is safely skipped."""
    db = create_test_db()
    service = GmailIngestionService(db=db)

    pdf = generate_synthetic_invoice_pdf(invoice_no="INV-2026-0010")
    email_detail = {
        "message_id": "msg_duplicate_check_1",
        "sender_email": "vendor@abc.com",
        "subject": "Tax Invoice INV-2026-0010",
        "snippet": "Here is invoice 0010.",
        "attachments": [
            {"attachment_id": "att_1", "filename": "INV-0010.pdf", "mime_type": "application/pdf", "raw_bytes": pdf}
        ]
    }

    # First fetch
    res1 = service.process_email(email_detail)
    assert res1["status"] == "PROCESSED"

    # Second fetch of same message
    res2 = service.process_email(email_detail)
    assert res2["status"] == "DUPLICATE_MESSAGE_SKIPPED"
    assert db.query(DocumentModel).count() == 1


def test_6_duplicate_attachment_across_different_emails():
    """6. Same attachment arriving in a second email is detected as duplicate while preserving source metadata."""
    db = create_test_db()
    service = GmailIngestionService(db=db)

    identical_pdf = generate_synthetic_invoice_pdf(invoice_no="INV-2026-SHARED")

    # Email 1
    email_1 = {
        "message_id": "msg_email_A",
        "sender_email": "supplier@corp.com",
        "subject": "Tax Invoice INV-2026-SHARED",
        "attachments": [
            {"attachment_id": "att_A", "filename": "invoice_A.pdf", "mime_type": "application/pdf", "raw_bytes": identical_pdf}
        ]
    }
    res1 = service.process_email(email_1)
    assert res1["status"] == "PROCESSED"
    first_doc_id = res1["documents_created"][0]["document_id"]

    # Email 2 with the same attachment bytes but different filename and message ID
    email_2 = {
        "message_id": "msg_email_B",
        "sender_email": "forwarding@corp.com",
        "subject": "Forwarded Tax Invoice INV-2026-SHARED",
        "attachments": [
            {"attachment_id": "att_B", "filename": "invoice_forwarded.pdf", "mime_type": "application/pdf", "raw_bytes": identical_pdf}
        ]
    }
    res2 = service.process_email(email_2)
    assert res2["status"] == "SKIPPED"
    assert res2["attachments_skipped"] == 1
    assert len(res2["documents_created"]) == 0

    # Ensure still only 1 document in DocumentModel
    assert db.query(DocumentModel).count() == 1

    # Ensure source metadata for Email B is preserved in source_emails and email_attachments
    source_b = db.query(EmailSourceModel).filter(EmailSourceModel.gmail_message_id == "msg_email_B").first()
    assert source_b is not None
    assert source_b.ingestion_status == "SKIPPED"

    att_b = db.query(EmailAttachmentModel).filter(EmailAttachmentModel.gmail_message_id == "msg_email_B").first()
    assert att_b is not None
    assert att_b.status == "DUPLICATE_SKIPPED"
    assert att_b.document_id == first_doc_id  # Linked to the original document!


def test_7_unsupported_file_types():
    """7. Unsupported file types (.zip, .exe) are skipped and recorded appropriately."""
    db = create_test_db()
    service = GmailIngestionService(db=db)

    email_detail = {
        "message_id": "msg_unsupported_archive",
        "sender_email": "billing@tech.com",
        "subject": "Supplier Invoice bundle archive",
        "attachments": [
            {"attachment_id": "att_zip", "filename": "invoices.zip", "mime_type": "application/zip", "raw_bytes": b"PK\x03\x04 fake zip"}
        ]
    }

    result = service.process_email(email_detail)
    assert result["attachments_skipped"] == 1
    assert len(result["documents_created"]) == 0

    att_record = db.query(EmailAttachmentModel).filter(EmailAttachmentModel.gmail_message_id == "msg_unsupported_archive").first()
    assert att_record.status == "UNSUPPORTED_SKIPPED"
    assert "Unsupported" in att_record.error_message


def test_8_invalid_or_empty_attachment_content():
    """8. Empty content or fake PDF headers are rejected."""
    db = create_test_db()
    service = GmailIngestionService(db=db)

    email_detail = {
        "message_id": "msg_corrupt_1",
        "sender_email": "vendor@test.com",
        "subject": "Tax Invoice document",
        "attachments": [
            {"attachment_id": "att_empty", "filename": "empty.pdf", "mime_type": "application/pdf", "raw_bytes": b""},
            {"attachment_id": "att_fake", "filename": "fake.pdf", "mime_type": "application/pdf", "raw_bytes": b"not a pdf content"}
        ]
    }

    result = service.process_email(email_detail)
    assert result["attachments_skipped"] == 2
    assert len(result["documents_created"]) == 0

    records = db.query(EmailAttachmentModel).filter(EmailAttachmentModel.gmail_message_id == "msg_corrupt_1").all()
    assert len(records) == 2
    statuses = {r.status for r in records}
    assert "EMPTY_SKIPPED" in statuses
    assert "INVALID_CONTENT_SKIPPED" in statuses


def test_9_source_traceability():
    """9. Complete traceability: Document -> EmailSource and EmailSource -> EmailAttachments."""
    db = create_test_db()
    service = GmailIngestionService(db=db)

    pdf = generate_synthetic_invoice_pdf(invoice_no="INV-2026-TRACE")
    email_detail = {
        "message_id": "msg_trace_001",
        "sender_email": "vendor@traceable.com",
        "subject": "Tax Invoice for Traceability",
        "attachments": [
            {"attachment_id": "att_t1", "filename": "trace.pdf", "mime_type": "application/pdf", "raw_bytes": pdf}
        ]
    }

    result = service.process_email(email_detail)
    doc_id = result["documents_created"][0]["document_id"]

    # Trace from Document to Source Email
    doc = db.query(DocumentModel).filter(DocumentModel.document_id == doc_id).first()
    source_email = db.query(EmailSourceModel).filter(EmailSourceModel.id == doc.source_email_id).first()
    assert source_email.gmail_message_id == "msg_trace_001"
    assert source_email.sender_email == "vendor@traceable.com"

    # Trace from Source Email to Attachments
    attachments = db.query(EmailAttachmentModel).filter(EmailAttachmentModel.email_source_id == source_email.id).all()
    assert len(attachments) == 1
    assert attachments[0].document_id == doc_id
    assert attachments[0].status == "PROCESSED"


def run_all_tests():
    print("=" * 65)
    print("Running GmailAttachmentIngestionService Integration Tests (Stage 5)...")
    print("=" * 65)
    tests = [
        ("test_1_eligible_invoice_enters_pipeline", test_1_eligible_invoice_enters_pipeline),
        ("test_2_unrelated_email_skipped", test_2_unrelated_email_skipped),
        ("test_3_uncertain_email_deferred", test_3_uncertain_email_deferred),
        ("test_4_multiple_attachments_in_one_email", test_4_multiple_attachments_in_one_email),
        ("test_5_duplicate_message_id", test_5_duplicate_message_id),
        ("test_6_duplicate_attachment_across_different_emails", test_6_duplicate_attachment_across_different_emails),
        ("test_7_unsupported_file_types", test_7_unsupported_file_types),
        ("test_8_invalid_or_empty_attachment_content", test_8_invalid_or_empty_attachment_content),
        ("test_9_source_traceability", test_9_source_traceability),
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

    print("=" * 65)
    print(f"Result: {passed}/{len(tests)} integration tests passed successfully!")
    print("=" * 65)


if __name__ == "__main__":
    run_all_tests()
