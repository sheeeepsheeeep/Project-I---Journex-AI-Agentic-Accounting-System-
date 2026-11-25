import sys
import json
from pathlib import Path
import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.db.models import (
    Base,
    DocumentModel,
    AccountingTransactionModel,
    JournalEntryModel,
    BankTransactionModel,
    EmailSourceModel,
    EmailAttachmentModel,
    SystemSettingModel
)
from app.api.documents import delete_document, batch_delete_documents, BatchDeleteDocumentsRequest


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_delete_nonexistent_document(db_session):
    with pytest.raises(HTTPException) as exc_info:
        delete_document(document_id="DOC-9999-999999", db=db_session)
    assert exc_info.value.status_code == 404
    assert "not found" in exc_info.value.detail.lower()


def test_delete_document_blocked_by_posted_journal(db_session):
    doc = DocumentModel(
        document_id="DOC-2026-000010",
        original_filename="invoice_posted.pdf",
        file_path="data/test.pdf",
        file_type=".pdf",
        mime_type="application/pdf",
        status="COMPLETED"
    )
    journal = JournalEntryModel(
        journal_id="JRN-0010",
        transaction_id="TXN-0010",
        document_id="DOC-2026-000010",
        status="POSTED"
    )
    db_session.add(doc)
    db_session.add(journal)
    db_session.commit()

    with pytest.raises(HTTPException) as exc_info:
        delete_document(document_id="DOC-2026-000010", db=db_session)
    assert exc_info.value.status_code == 400
    assert "posted general ledger" in exc_info.value.detail.lower()


def test_delete_document_unlinks_matched_bank_transaction(db_session):
    doc = DocumentModel(
        document_id="DOC-2026-000011",
        original_filename="invoice_bank.pdf",
        file_path="data/test.pdf",
        file_type=".pdf",
        mime_type="application/pdf",
        status="EXTRACTED"
    )
    bank_txn = BankTransactionModel(
        bank_transaction_id="TXN-0001",
        unique_source_id="abc_bank:TXN-0001",
        transaction_date="2026-09-24",
        description="Payment from customer",
        matched_document_id="DOC-2026-000011",
        match_status="MATCHED",
        reconciliation_status="RECONCILED",
        audit_trail_json=json.dumps([{"action": "AUTO_MATCH", "doc": "DOC-2026-000011"}])
    )
    db_session.add(doc)
    db_session.add(bank_txn)
    db_session.commit()

    res = delete_document(document_id="DOC-2026-000011", db=db_session)
    assert res["success"] is True
    assert "TXN-0001" in res["unlinked_bank_transactions"]

    # Verify bank transaction remains in DB, unlinked and reset
    bank_after = db_session.query(BankTransactionModel).filter(BankTransactionModel.bank_transaction_id == "TXN-0001").first()
    assert bank_after is not None
    assert bank_after.matched_document_id is None
    assert bank_after.match_status == "UNMATCHED"
    assert bank_after.match_confidence == 0.0
    assert bank_after.match_explanation is None
    assert bank_after.reconciliation_status == "UNRECONCILED"
    assert "DOCUMENT_DELETED_UNLINK" in bank_after.audit_trail_json


def test_delete_document_success(db_session):
    doc = DocumentModel(
        document_id="DOC-2026-000012",
        original_filename="draft_invoice.pdf",
        file_path="data/test.pdf",
        file_type=".pdf",
        mime_type="application/pdf",
        status="EXTRACTED"
    )
    txn = AccountingTransactionModel(
        transaction_id="TXN-0012",
        document_id="DOC-2026-000012",
        status="Completed"
    )
    draft_journal = JournalEntryModel(
        journal_id="JRN-0012",
        transaction_id="TXN-0012",
        document_id="DOC-2026-000012",
        status="DRAFT"
    )
    email_src = EmailSourceModel(
        gmail_message_id="msg_12345",
        subject="Invoice email",
        ingestion_status="PROCESSED"
    )
    db_session.add(doc)
    db_session.add(txn)
    db_session.add(draft_journal)
    db_session.add(email_src)
    db_session.commit()

    att = EmailAttachmentModel(
        email_source_id=email_src.id,
        gmail_message_id="msg_12345",
        filename="draft_invoice.pdf",
        file_type=".pdf",
        mime_type="application/pdf",
        file_hash="hash_12345",
        document_id="DOC-2026-000012"
    )
    db_session.add(att)
    db_session.commit()

    res = delete_document(document_id="DOC-2026-000012", db=db_session)
    assert res["success"] is True

    # Verify DB state
    assert db_session.query(DocumentModel).filter(DocumentModel.document_id == "DOC-2026-000012").first() is None
    assert db_session.query(AccountingTransactionModel).filter(AccountingTransactionModel.document_id == "DOC-2026-000012").first() is None
    assert db_session.query(JournalEntryModel).filter(JournalEntryModel.document_id == "DOC-2026-000012").first() is None

    # EmailSource is reset for reprocessing since its attachment was deleted
    src_after = db_session.query(EmailSourceModel).filter(EmailSourceModel.gmail_message_id == "msg_12345").first()
    assert src_after is not None
    assert src_after.ingestion_status == "RESET_FOR_REPROCESSING"

    # Attachment record is cleaned up
    assert db_session.query(EmailAttachmentModel).filter(EmailAttachmentModel.document_id == "DOC-2026-000012").first() is None


def test_batch_delete_documents_success(db_session):
    doc1 = DocumentModel(
        document_id="DOC-2026-000021",
        original_filename="doc1.pdf",
        file_path="data/doc1.pdf",
        file_type=".pdf",
        mime_type="application/pdf",
        status="EXTRACTED"
    )
    doc2 = DocumentModel(
        document_id="DOC-2026-000022",
        original_filename="doc2.pdf",
        file_path="data/doc2.pdf",
        file_type=".pdf",
        mime_type="application/pdf",
        status="EXTRACTED"
    )
    bank_txn = BankTransactionModel(
        bank_transaction_id="TXN-0002",
        unique_source_id="abc_bank:TXN-0002",
        transaction_date="2026-09-25",
        description="Payment 2",
        matched_document_id="DOC-2026-000021",
        match_status="MATCHED",
        reconciliation_status="RECONCILED"
    )
    db_session.add_all([doc1, doc2, bank_txn])
    db_session.commit()

    payload = BatchDeleteDocumentsRequest(document_ids=["DOC-2026-000021", "DOC-2026-000022"])
    res = batch_delete_documents(payload=payload, db=db_session)

    assert res["success"] is True
    assert res["deleted_count"] == 2
    assert "DOC-2026-000021" in res["deleted_document_ids"]
    assert "DOC-2026-000022" in res["deleted_document_ids"]
    assert "TXN-0002" in res["unlinked_bank_transactions"]

    # Verify documents are gone
    assert db_session.query(DocumentModel).filter(DocumentModel.document_id.in_(["DOC-2026-000021", "DOC-2026-000022"])).count() == 0

    # Verify bank transaction unlinked
    b_after = db_session.query(BankTransactionModel).filter(BankTransactionModel.bank_transaction_id == "TXN-0002").first()
    assert b_after.matched_document_id is None
    assert b_after.match_status == "UNMATCHED"


def test_batch_delete_blocked_by_posted_journal(db_session):
    doc1 = DocumentModel(
        document_id="DOC-2026-000031",
        original_filename="doc31.pdf",
        file_path="data/doc31.pdf",
        file_type=".pdf",
        mime_type="application/pdf",
        status="EXTRACTED"
    )
    doc2 = DocumentModel(
        document_id="DOC-2026-000032",
        original_filename="doc32.pdf",
        file_path="data/doc32.pdf",
        file_type=".pdf",
        mime_type="application/pdf",
        status="COMPLETED"
    )
    posted_jrn = JournalEntryModel(
        journal_id="JRN-0032",
        transaction_id="TXN-0032",
        document_id="DOC-2026-000032",
        status="POSTED"
    )
    db_session.add_all([doc1, doc2, posted_jrn])
    db_session.commit()

    payload = BatchDeleteDocumentsRequest(document_ids=["DOC-2026-000031", "DOC-2026-000032"])
    with pytest.raises(HTTPException) as exc_info:
        batch_delete_documents(payload=payload, db=db_session)
    assert exc_info.value.status_code == 400
    assert "posted general ledger" in exc_info.value.detail.lower()

    # Verify neither document was deleted due to transaction integrity
    assert db_session.query(DocumentModel).filter(DocumentModel.document_id == "DOC-2026-000031").first() is not None
    assert db_session.query(DocumentModel).filter(DocumentModel.document_id == "DOC-2026-000032").first() is not None


def test_batch_delete_nonexistent_document(db_session):
    payload = BatchDeleteDocumentsRequest(document_ids=["DOC-MISSING-001"])
    with pytest.raises(HTTPException) as exc_info:
        batch_delete_documents(payload=payload, db=db_session)
    assert exc_info.value.status_code == 404



def test_delete_only_bank_transactions(db_session):
    # Setup doc that is matched to TXN-0001
    doc = DocumentModel(
        document_id="DOC-2026-000041",
        original_filename="doc41.pdf",
        file_path="data/doc41.pdf",
        file_type=".pdf",
        mime_type="application/pdf",
        status="EXTRACTED"
    )
    b1 = BankTransactionModel(
        bank_transaction_id="TXN-0001",
        unique_source_id="abc_bank:TXN-0001",
        transaction_date="2026-09-24",
        description="Payment 1",
        matched_document_id="DOC-2026-000041",
        match_status="MATCHED",
        reconciliation_status="RECONCILED"
    )
    b2 = BankTransactionModel(
        bank_transaction_id="TXN-0002",
        unique_source_id="abc_bank:TXN-0002",
        transaction_date="2026-09-25",
        description="Payment 2",
        match_status="UNMATCHED",
        reconciliation_status="UNRECONCILED"
    )
    cursor_setting = SystemSettingModel(
        setting_key="abc_bank_last_cursor",
        setting_value="TXN-0002"
    )
    db_session.add_all([doc, b1, b2, cursor_setting])
    db_session.commit()

    # Delete only TXN-0002
    from app.api.documents import execute_unified_financial_records_deletion
    res = execute_unified_financial_records_deletion(db_session, bank_transaction_ids=["TXN-0002"])

    assert res["success"] is True
    assert res["deleted_bank_transactions_count"] == 1
    assert "TXN-0002" in res["deleted_bank_transaction_ids"]

    # Verify TXN-0002 deleted, TXN-0001 preserved
    assert db_session.query(BankTransactionModel).filter(BankTransactionModel.bank_transaction_id == "TXN-0002").first() is None
    assert db_session.query(BankTransactionModel).filter(BankTransactionModel.bank_transaction_id == "TXN-0001").first() is not None

    # Verify surviving document DOC-2026-000041 remains untouched
    surviving_doc = db_session.query(DocumentModel).filter(DocumentModel.document_id == "DOC-2026-000041").first()
    assert surviving_doc is not None

    # Verify cursor rewound to TXN-0001
    updated_cursor = db_session.query(SystemSettingModel).filter(SystemSettingModel.setting_key == "abc_bank_last_cursor").first()
    assert updated_cursor.setting_value == "TXN-0001"


def test_delete_both_document_and_bank_transaction(db_session):
    doc = DocumentModel(
        document_id="DOC-2026-000051",
        original_filename="doc51.pdf",
        file_path="data/doc51.pdf",
        file_type=".pdf",
        mime_type="application/pdf",
        status="EXTRACTED"
    )
    b_tx = BankTransactionModel(
        bank_transaction_id="TXN-0005",
        unique_source_id="abc_bank:TXN-0005",
        transaction_date="2026-09-26",
        description="Payment 5",
        matched_document_id="DOC-2026-000051",
        match_status="MATCHED",
        reconciliation_status="RECONCILED"
    )
    db_session.add_all([doc, b_tx])
    db_session.commit()

    from app.api.documents import execute_unified_financial_records_deletion
    res = execute_unified_financial_records_deletion(
        db_session,
        document_ids=["DOC-2026-000051"],
        bank_transaction_ids=["TXN-0005"]
    )

    assert res["success"] is True
    assert res["total_deleted"] == 2
    assert "DOC-2026-000051" in res["deleted_document_ids"]
    assert "TXN-0005" in res["deleted_bank_transaction_ids"]

    # Verify both are deleted
    assert db_session.query(DocumentModel).filter(DocumentModel.document_id == "DOC-2026-000051").first() is None
    assert db_session.query(BankTransactionModel).filter(BankTransactionModel.bank_transaction_id == "TXN-0005").first() is None


def test_delete_bank_transaction_blocked_by_posted_status(db_session):
    b_tx = BankTransactionModel(
        bank_transaction_id="TXN-0009",
        unique_source_id="abc_bank:TXN-0009",
        transaction_date="2026-09-27",
        description="Posted payment",
        reconciliation_status="POSTED"
    )
    db_session.add(b_tx)
    db_session.commit()

    from app.api.documents import execute_unified_financial_records_deletion
    with pytest.raises(HTTPException) as exc_info:
        execute_unified_financial_records_deletion(db_session, bank_transaction_ids=["TXN-0009"])
    assert exc_info.value.status_code == 400
    assert "posted" in exc_info.value.detail.lower()


def test_delete_financial_records_endpoint_supports_mixed(db_session):
    doc = DocumentModel(
        document_id="DOC-2026-000061",
        original_filename="doc61.pdf",
        file_path="data/doc61.pdf",
        file_type=".pdf",
        mime_type="application/pdf",
        status="EXTRACTED"
    )
    b_tx = BankTransactionModel(
        bank_transaction_id="TXN-0006",
        unique_source_id="abc_bank:TXN-0006",
        transaction_date="2026-09-28",
        description="Payment 6"
    )
    db_session.add_all([doc, b_tx])
    db_session.commit()

    from app.api.documents import delete_financial_records_endpoint, DeleteFinancialRecordsRequest, DeleteRecordItem
    payload = DeleteFinancialRecordsRequest(
        records=[
            DeleteRecordItem(id="DOC-2026-000061", type="ACCOUNTING_DOCUMENT"),
            DeleteRecordItem(id="TXN-0006", type="BANK_TRANSACTION")
        ]
    )
    res = delete_financial_records_endpoint(payload=payload, db=db_session)
    assert res["success"] is True
    assert res["total_deleted"] == 2


def test_testing_reset_deletes_posted_journal_and_gl_dependencies(db_session):
    """
    Verifies that in Testing Reset mode (allow_posted_reset=True), documents with
    POSTED journal entries and General Ledger dependencies are cleanly purged.
    """
    doc = DocumentModel(
        document_id="DOC-2026-000071",
        original_filename="sales_posted.pdf",
        file_path="data/sales_posted.pdf",
        file_type=".pdf",
        mime_type="application/pdf",
        status="EXTRACTED",
        grand_total=1060.00
    )
    txn = AccountingTransactionModel(
        transaction_id="TXN-2026-000071",
        document_id="DOC-2026-000071",
        status="Completed",
        book_of_prime_entry="Sales Journal"
    )
    journal = JournalEntryModel(
        journal_id="JRN-2026-000071",
        transaction_id="TXN-2026-000071",
        document_id="DOC-2026-000071",
        status="POSTED",
        total_debit=1060.00,
        total_credit=1060.00,
        is_balanced=True,
        lines_json=json.dumps([
            {"account_code": "300-0000", "account_name": "TRADE DEBTORS", "debit": 1060.00, "credit": 0.0},
            {"account_code": "500-0000", "account_name": "SALES", "debit": 0.0, "credit": 1000.00},
            {"account_code": "430-0000", "account_name": "SALES TAX", "debit": 0.0, "credit": 60.00}
        ])
    )
    db_session.add_all([doc, txn, journal])
    db_session.commit()

    from app.api.documents import execute_unified_financial_records_deletion

    # Without allow_posted_reset, it must be blocked
    with pytest.raises(HTTPException) as exc_info:
        execute_unified_financial_records_deletion(
            db_session,
            document_ids=["DOC-2026-000071"],
            allow_posted_reset=False
        )
    assert exc_info.value.status_code == 400
    assert "posted" in exc_info.value.detail.lower()

    # With allow_posted_reset=True, it executes full cleanup
    res = execute_unified_financial_records_deletion(
        db_session,
        document_ids=["DOC-2026-000071"],
        allow_posted_reset=True
    )
    assert res["success"] is True
    assert res["mode"] == "FULL_TESTING_RESET"
    assert res["deleted_posted_journals_count"] == 1
    assert res["deleted_accounting_transactions_count"] == 1

    # Verify DB records are completely gone
    assert db_session.query(DocumentModel).filter(DocumentModel.document_id == "DOC-2026-000071").first() is None
    assert db_session.query(AccountingTransactionModel).filter(AccountingTransactionModel.document_id == "DOC-2026-000071").first() is None
    assert db_session.query(JournalEntryModel).filter(JournalEntryModel.document_id == "DOC-2026-000071").first() is None


def test_testing_reset_blocked_in_production_environment(db_session, monkeypatch):
    """
    Verifies that destructive testing reset with posted journal deletion is blocked
    in production environment unless overridden explicitly.
    """
    monkeypatch.setenv("APP_ENV", "production")

    doc = DocumentModel(
        document_id="DOC-2026-000081",
        original_filename="prod_doc.pdf",
        file_path="data/prod.pdf",
        file_type=".pdf",
        mime_type="application/pdf",
        status="EXTRACTED"
    )
    journal = JournalEntryModel(
        journal_id="JRN-2026-000081",
        transaction_id="TXN-2026-000081",
        document_id="DOC-2026-000081",
        status="POSTED"
    )
    db_session.add_all([doc, journal])
    db_session.commit()

    from app.api.documents import execute_unified_financial_records_deletion

    # Must be forbidden in production
    with pytest.raises(HTTPException) as exc_info:
        execute_unified_financial_records_deletion(
            db_session,
            document_ids=["DOC-2026-000081"],
            allow_posted_reset=True
        )
    assert exc_info.value.status_code == 403
    assert "production" in exc_info.value.detail.lower()

    # Allowed with explicit administrative override
    res = execute_unified_financial_records_deletion(
        db_session,
        document_ids=["DOC-2026-000081"],
        allow_posted_reset=True,
        environment_override="ALLOW_PROD_RESET"
    )
    assert res["success"] is True


def test_testing_reset_shared_journal_detection_blocks_unsafe_deletion(db_session):
    """
    Verifies Requirement 3: If a journal contains line items or references associated
    with both selected and unselected records, the operation is blocked to preserve
    unrelated accounting records.
    """
    doc_a = DocumentModel(
        document_id="DOC-2026-000091",
        original_filename="doc_a.pdf",
        file_path="data/doc_a.pdf",
        file_type=".pdf",
        mime_type="application/pdf",
        status="EXTRACTED"
    )
    doc_b = DocumentModel(
        document_id="DOC-2026-000092",
        original_filename="doc_b.pdf",
        file_path="data/doc_b.pdf",
        file_type=".pdf",
        mime_type="application/pdf",
        status="EXTRACTED"
    )
    # Journal attached to DOC-A has a line item associated with DOC-B
    journal = JournalEntryModel(
        journal_id="JRN-2026-000091",
        transaction_id="TXN-2026-000091",
        document_id="DOC-2026-000091",
        status="POSTED",
        lines_json=json.dumps([
            {"account_code": "300-0000", "debit": 100.0, "credit": 0.0, "document_id": "DOC-2026-000091"},
            {"account_code": "500-0000", "debit": 0.0, "credit": 100.0, "document_id": "DOC-2026-000092"}
        ])
    )
    db_session.add_all([doc_a, doc_b, journal])
    db_session.commit()

    from app.api.documents import execute_unified_financial_records_deletion

    # Attempting to delete only DOC-A while DOC-B is unselected must be blocked
    with pytest.raises(HTTPException) as exc_info:
        execute_unified_financial_records_deletion(
            db_session,
            document_ids=["DOC-2026-000091"],
            allow_posted_reset=True
        )
    assert exc_info.value.status_code == 400
    assert "DOC-2026-000092" in exc_info.value.detail or "unselected" in exc_info.value.detail.lower()
