"""
Unit Tests for Rule-Based Email Classifier (Stage 4).
Verifies multi-signal classification into ACCOUNTING_RELATED, UNRELATED, or UNCERTAIN.
"""

import sys
from pathlib import Path

# Ensure backend folder is in path
backend_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_path))

from app.services.email_ingestion.email_classifier import (
    EmailClassifier,
    EmailClassification
)


def test_1_invoice_email_with_pdf():
    """1. Invoice email with an invoice PDF attachment -> ACCOUNTING_RELATED."""
    classifier = EmailClassifier()
    res = classifier.classify(
        subject="Tax Invoice INV-2026-0042 from TechCore Solutions",
        snippet="Dear team, please find attached our tax invoice for services rendered in September.",
        sender_email="billing@techcore.com",
        sender_name="TechCore Billing",
        attachments=[
            {"filename": "INV-2026-0042.pdf", "extension": ".pdf", "mime_type": "application/pdf"}
        ]
    )
    assert res.classification == EmailClassification.ACCOUNTING_RELATED
    assert res.confidence >= 0.85
    assert any("invoice" in r.lower() for r in res.reasons)


def test_2_receipt_email_with_image():
    """2. Receipt email with a receipt image attachment -> ACCOUNTING_RELATED."""
    classifier = EmailClassifier()
    res = classifier.classify(
        subject="Payment Receipt - Stationary Supplies",
        snippet="Attached is the official payment receipt for today's office supplies purchase.",
        sender_email="store@popular.com.my",
        attachments=[
            {"filename": "receipt_scan.png", "extension": ".png", "mime_type": "image/png"}
        ]
    )
    assert res.classification == EmailClassification.ACCOUNTING_RELATED
    assert res.confidence >= 0.85


def test_3_bank_statement_with_pdf():
    """3. Bank statement email with a PDF attachment -> ACCOUNTING_RELATED."""
    classifier = EmailClassifier()
    res = classifier.classify(
        subject="Monthly Bank Statement - Account ending 4821",
        snippet="Your electronic bank statement for the month of September 2026 is ready.",
        sender_email="estatements@maybank.com",
        attachments=[
            {"filename": "Statement_SEP2026.pdf", "extension": ".pdf", "mime_type": "application/pdf"}
        ]
    )
    assert res.classification == EmailClassification.ACCOUNTING_RELATED
    assert res.confidence >= 0.85


def test_4_unrelated_meeting_with_pdf():
    """4. Unrelated meeting email with a PDF attachment -> UNRELATED."""
    classifier = EmailClassifier()
    res = classifier.classify(
        subject="Project Status Meeting & Sprint Review Agenda",
        snippet="Hi everyone, please review the attached agenda before our sync tomorrow at 10 AM.",
        sender_email="project.manager@techcorp.com",
        attachments=[
            {"filename": "sprint_agenda.pdf", "extension": ".pdf", "mime_type": "application/pdf"}
        ]
    )
    assert res.classification == EmailClassification.UNRELATED
    assert res.confidence >= 0.8


def test_5_accounting_wording_no_attachment():
    """5. Accounting-related wording with no attachment -> UNCERTAIN."""
    classifier = EmailClassifier()
    res = classifier.classify(
        subject="Quick question regarding outstanding invoice INV-9021",
        snippet="Could you clarify when the billing statement will be approved for payment?",
        sender_email="vendor@supplier.com",
        attachments=[]  # No attachment
    )
    assert res.classification == EmailClassification.UNCERTAIN
    assert any("no attachment" in r.lower() for r in res.reasons)


def test_6_vague_subject_with_ambiguous_attachment():
    """6. Vague subject with an ambiguous attachment -> UNCERTAIN."""
    classifier = EmailClassifier()
    res = classifier.classify(
        subject="Documents for your review",
        snippet="Please see attached file.",
        sender_email="contact@partner.com",
        attachments=[
            {"filename": "scan_001.pdf", "extension": ".pdf", "mime_type": "application/pdf"}
        ]
    )
    assert res.classification == EmailClassification.UNCERTAIN
    assert any("lack" in r.lower() or "generic" in r.lower() or "attachment" in r.lower() for r in res.reasons)


def test_7_mixed_conflicting_signals():
    """7. Mixed or conflicting email signals -> UNCERTAIN."""
    classifier = EmailClassifier()
    res = classifier.classify(
        subject="Team lunch invoice and meeting minutes for quarterly review",
        snippet="Here are the minutes of meeting along with the receipt from lunch.",
        sender_email="colleague@office.com",
        attachments=[
            {"filename": "notes.pdf", "extension": ".pdf", "mime_type": "application/pdf"}
        ]
    )
    assert res.classification == EmailClassification.UNCERTAIN
    assert any("conflicting" in r.lower() for r in res.reasons)


def test_8_case_insensitive_matching():
    """8. Case-insensitive keyword matching -> ACCOUNTING_RELATED."""
    classifier = EmailClassifier()
    res = classifier.classify(
        subject="COMMERCIAL INVOICE AND REMITTANCE ADVICE",
        snippet="PLEASE FIND THE OFFICIAL BILLING STATEMENT ATTACHED.",
        sender_email="ACCOUNTS@GLOBALTRADING.COM",
        attachments=[
            {"filename": "INVOICE_OCTOBER.PDF", "extension": ".pdf", "mime_type": "application/pdf"}
        ]
    )
    assert res.classification == EmailClassification.ACCOUNTING_RELATED
    assert res.confidence >= 0.85


def run_all_tests():
    print("=" * 60)
    print("Running EmailClassifier Unit Tests...")
    print("=" * 60)
    tests = [
        ("test_1_invoice_email_with_pdf", test_1_invoice_email_with_pdf),
        ("test_2_receipt_email_with_image", test_2_receipt_email_with_image),
        ("test_3_bank_statement_with_pdf", test_3_bank_statement_with_pdf),
        ("test_4_unrelated_meeting_with_pdf", test_4_unrelated_meeting_with_pdf),
        ("test_5_accounting_wording_no_attachment", test_5_accounting_wording_no_attachment),
        ("test_6_vague_subject_with_ambiguous_attachment", test_6_vague_subject_with_ambiguous_attachment),
        ("test_7_mixed_conflicting_signals", test_7_mixed_conflicting_signals),
        ("test_8_case_insensitive_matching", test_8_case_insensitive_matching),
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
    print(f"Result: {passed}/{len(tests)} classification tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
