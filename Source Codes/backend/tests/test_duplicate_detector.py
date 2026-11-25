"""
Unit Tests for DuplicateDetector (Stage 4).
Verifies:
1. Gmail message ID duplicate prevention.
2. Attachment SHA-256 content hash duplicate detection.
3. Filename independence (different filenames with same bytes = duplicate).
4. Content independence (same filename with different bytes = NOT duplicate).
5. Cross-email duplicate detection with source tracking.
6. Empty content handling.
7. Multiple attachments per email.
"""

import sys
import hashlib
from pathlib import Path

# Ensure backend folder is in path
backend_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_path))

from app.services.email_ingestion.duplicate_detector import (
    DuplicateDetector,
    compute_content_hash
)


def test_1_previously_unseen_message_id():
    """1. A previously unseen Gmail message ID is reported as not duplicate."""
    detector = DuplicateDetector()
    res = detector.check_message("msg_new_101")
    assert res.is_duplicate is False
    assert res.message_id == "msg_new_101"


def test_2_already_processed_message_id():
    """2. A Gmail message ID that was already processed is detected as duplicate."""
    detector = DuplicateDetector()
    detector.register_message("msg_already_processed_001")

    res = detector.check_message("msg_already_processed_001")
    assert res.is_duplicate is True
    assert "already been processed" in res.reason


def test_3_previously_unseen_attachment_hash():
    """3. An attachment with a previously unseen hash is marked as unique."""
    detector = DuplicateDetector()
    file_bytes = b"%PDF-1.4 unique content for sales invoice 001"
    res = detector.check_attachment(file_bytes, filename="invoice_001.pdf", current_message_id="msg_001")

    assert res.is_duplicate is False
    assert res.is_empty is False
    assert res.file_hash == hashlib.sha256(file_bytes).hexdigest()
    assert "unique" in res.reason.lower()


def test_4_identical_bytes_different_filenames():
    """4. Identical attachment bytes with different filenames are detected as content duplicates."""
    detector = DuplicateDetector()
    file_bytes = b"%PDF-1.4 exact same document content"
    expected_hash = hashlib.sha256(file_bytes).hexdigest()

    # Register first file
    detector.register_attachment_hash(
        file_hash=expected_hash,
        document_id="DOC-2026-000001",
        message_id="msg_first_email",
        filename="original_invoice.pdf"
    )

    # Check same bytes with completely different filename
    res = detector.check_attachment(
        content_bytes=file_bytes,
        filename="renamed_copy_by_user.pdf",
        current_message_id="msg_second_email"
    )

    assert res.is_duplicate is True
    assert res.file_hash == expected_hash
    assert res.existing_document_id == "DOC-2026-000001"
    assert res.original_message_id == "msg_first_email"
    assert "duplicate" in res.reason.lower()


def test_5_same_attachment_in_two_different_emails():
    """5. The same attachment received in two different emails preserves source tracking."""
    detector = DuplicateDetector()
    supplier_invoice_bytes = b"%PDF-1.4 supplier invoice data 9999"
    invoice_hash = hashlib.sha256(supplier_invoice_bytes).hexdigest()

    # Message A arrives and registers
    detector.register_message("msg_A")
    detector.register_attachment_hash(
        file_hash=invoice_hash,
        document_id="DOC-2026-000005",
        message_id="msg_A",
        filename="invoice_may.pdf"
    )

    # Message B arrives later with the same attachment
    msg_b_check = detector.check_message("msg_B")
    assert msg_b_check.is_duplicate is False  # Message B is a new message

    att_b_check = detector.check_attachment(
        content_bytes=supplier_invoice_bytes,
        filename="invoice_may_duplicate.pdf",
        current_message_id="msg_B"
    )
    assert att_b_check.is_duplicate is True
    assert att_b_check.original_message_id == "msg_A"
    assert att_b_check.existing_document_id == "DOC-2026-000005"
    # Note: source info of msg_B is retained for the caller


def test_6_different_bytes_same_filename():
    """6. Different attachment bytes with the same filename are NOT marked as duplicate."""
    detector = DuplicateDetector()
    file_bytes_1 = b"%PDF-1.4 version 1 content"
    file_bytes_2 = b"%PDF-1.4 version 2 different content"

    hash_1 = hashlib.sha256(file_bytes_1).hexdigest()
    detector.register_attachment_hash(
        file_hash=hash_1,
        document_id="DOC-2026-000010",
        message_id="msg_001",
        filename="invoice.pdf"
    )

    # Second file with same name "invoice.pdf" but different bytes
    res = detector.check_attachment(
        content_bytes=file_bytes_2,
        filename="invoice.pdf",
        current_message_id="msg_002"
    )
    assert res.is_duplicate is False
    assert res.file_hash == hashlib.sha256(file_bytes_2).hexdigest()
    assert res.file_hash != hash_1


def test_7_empty_attachment_content():
    """7. Empty attachment content (0 bytes) is flagged properly and not marked as duplicate."""
    detector = DuplicateDetector()
    res = detector.check_attachment(b"", filename="empty.pdf", current_message_id="msg_001")
    assert res.is_duplicate is False
    assert res.is_empty is True
    assert "empty" in res.reason.lower()


def test_8_multiple_attachments_within_one_email():
    """8. Multiple distinct attachments within a single email are checked independently."""
    detector = DuplicateDetector()
    email_id = "msg_bundle_99"

    att1_bytes = b"%PDF-1.4 invoice part 1"
    att2_bytes = b"%PDF-1.4 invoice part 2"
    att3_bytes = b"%PDF-1.4 invoice part 1"  # Identical to att1 within the same email

    res1 = detector.check_attachment(att1_bytes, "part1.pdf", current_message_id=email_id)
    assert res1.is_duplicate is False
    detector.register_attachment_hash(res1.file_hash, document_id="DOC-2026-000021", message_id=email_id, filename="part1.pdf")

    res2 = detector.check_attachment(att2_bytes, "part2.pdf", current_message_id=email_id)
    assert res2.is_duplicate is False
    detector.register_attachment_hash(res2.file_hash, document_id="DOC-2026-000022", message_id=email_id, filename="part2.pdf")

    # att3 should be detected as duplicate within the same email
    res3 = detector.check_attachment(att3_bytes, "part1_copy.pdf", current_message_id=email_id)
    assert res3.is_duplicate is True
    assert "within the same email" in res3.reason


def test_9_hash_lookup_callback_interface():
    """9. Tests database callback interface for external lookup."""
    # Simulate DB lookup function
    mock_db = {
        "fakehash1234567890abcdef": {
            "document_id": "DOC-2026-000099",
            "message_id": "msg_db_legacy",
            "filename": "db_existing.pdf"
        }
    }

    def db_lookup(h: str):
        return mock_db.get(h)

    detector = DuplicateDetector(hash_lookup_fn=db_lookup)
    
    # Check matching hash
    mock_bytes = b"mock bytes content"
    mock_hash = compute_content_hash(mock_bytes)
    mock_db[mock_hash] = {
        "document_id": "DOC-2026-000099",
        "message_id": "msg_db_legacy",
        "filename": "db_existing.pdf"
    }

    res = detector.check_attachment(mock_bytes, "db_existing.pdf", "msg_new")
    assert res.is_duplicate is True
    assert res.existing_document_id == "DOC-2026-000099"
    assert "existing records" in res.reason or "database" in res.reason


def run_all_tests():
    print("=" * 60)
    print("Running DuplicateDetector Unit Tests...")
    print("=" * 60)
    tests = [
        ("test_1_previously_unseen_message_id", test_1_previously_unseen_message_id),
        ("test_2_already_processed_message_id", test_2_already_processed_message_id),
        ("test_3_previously_unseen_attachment_hash", test_3_previously_unseen_attachment_hash),
        ("test_4_identical_bytes_different_filenames", test_4_identical_bytes_different_filenames),
        ("test_5_same_attachment_in_two_different_emails", test_5_same_attachment_in_two_different_emails),
        ("test_6_different_bytes_same_filename", test_6_different_bytes_same_filename),
        ("test_7_empty_attachment_content", test_7_empty_attachment_content),
        ("test_8_multiple_attachments_within_one_email", test_8_multiple_attachments_within_one_email),
        ("test_9_hash_lookup_callback_interface", test_9_hash_lookup_callback_interface),
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
    print(f"Result: {passed}/{len(tests)} duplicate detector tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
