"""
Unit & Integration Tests for Email Ingestion API and Background Worker (Stage 6).
Calls API handlers and worker methods directly without requiring external HTTP client packages.
"""

import sys
import io
import asyncio
from pathlib import Path
from unittest.mock import MagicMock
from fastapi import HTTPException

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Ensure backend folder is in path
backend_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_path))

from app.db.models import Base, CompanyModel, DocumentModel
from app.services.email_ingestion.gmail_client import GmailClient
from app.services.email_ingestion.background_worker import GmailBackgroundWorker, worker_singleton
from app.api.email_ingestion import (
    get_ingestion_status,
    trigger_manual_fetch,
    start_ingestion_worker,
    stop_ingestion_worker,
    authorize_gmail,
    StartWorkerRequest,
    FetchEmailRequest,
    AuthorizeRequest,
)


def generate_invoice_pdf(invoice_no="INV-2026-API-01"):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.drawString(100, 750, "TAX INVOICE")
    c.drawString(100, 720, f"Invoice Number: {invoice_no}")
    c.drawString(100, 700, "Vendor: FastTech Cloud Sdn Bhd")
    c.drawString(100, 680, "Buyer: NextTech Solutions Sdn Bhd")
    c.drawString(100, 650, "Item 1    1    150.00    150.00")
    c.drawString(100, 600, "Grand Total: MYR 150.00")
    c.showPage()
    c.save()
    return buffer.getvalue()


def create_test_db():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    comp = CompanyModel(
        company_id="COMP-000001",
        company_name="NextTech Solutions Sdn Bhd",
        currency="MYR",
        financial_year=2026
    )
    session.add(comp)
    session.commit()
    return session, Session


def test_1_default_stopped_state():
    """1. Default worker state must be STOPPED upon initialization."""
    worker = GmailBackgroundWorker()
    status = worker.get_status()
    assert status["status"] == "STOPPED"
    assert status["is_running"] is False
    assert status["poll_interval_seconds"] == 300
    assert status["company_inbox"] == "shriley.ling@gmail.com"
    assert status["last_check_time"] == "Not started"


def test_2_status_endpoint():
    """2. GET /api/email-ingestion/status handler returns complete metadata."""
    data = get_ingestion_status()
    assert "status" in data
    assert "is_running" in data
    assert "company_inbox" in data
    assert "last_result" in data
    assert "oauth_configured" in data


def test_3_manual_fetch_when_unauthorized():
    """3. Manual fetch handler when Gmail is unauthorized raises 401 HTTPException."""
    mock_gmail = MagicMock(spec=GmailClient)
    mock_gmail.get_status.return_value = {
        "configured": False,
        "authorized": False,
        "status_message": "OAuth credentials not found."
    }
    worker_singleton.set_gmail_client(mock_gmail)
    db, _ = create_test_db()

    try:
        asyncio.run(trigger_manual_fetch(req=None, db=db))
        assert False, "Should have raised 401 HTTPException"
    except HTTPException as e:
        assert e.status_code == 401
        assert "not authorized" in str(e.detail).lower()


def test_4_manual_fetch_when_authorized():
    """4. Manual fetch when authorized processes emails and worker remains STOPPED."""
    invoice_bytes = generate_invoice_pdf("INV-2026-FETCH-99")
    mock_gmail = MagicMock(spec=GmailClient)
    mock_gmail.get_status.return_value = {
        "configured": True,
        "authorized": True,
        "company_inbox": "shriley.ling@gmail.com",
        "status_message": "Authorized"
    }
    mock_gmail.list_message_ids.return_value = ["msg_api_test_01"]
    mock_gmail.get_message_detail.return_value = {
        "message_id": "msg_api_test_01",
        "thread_id": "th_api_01",
        "sender_email": "billing@fasttech.com",
        "sender_name": "FastTech Billing",
        "subject": "Tax Invoice INV-2026-FETCH-99 from FastTech",
        "snippet": "Here is your invoice.",
        "received_at": "2026-10-20T11:00:00Z",
        "attachments": [
            {
                "attachment_id": "att_api_1",
                "filename": "INV-2026-FETCH-99.pdf",
                "mime_type": "application/pdf",
                "size_bytes": len(invoice_bytes),
                "raw_bytes": invoice_bytes
            }
        ]
    }
    worker_singleton.set_gmail_client(mock_gmail)
    db, session_factory = create_test_db()

    result = asyncio.run(worker_singleton.run_fetch_cycle(db_session_factory=session_factory))
    assert result["success"] is True
    assert result["emails_checked"] == 1
    assert result["documents_created"] >= 1

    # Worker must still be STOPPED (manual fetch does not silently enable continuous polling)
    st = worker_singleton.get_status()
    assert st["status"] == "STOPPED"
    assert st["is_running"] is False


def test_5_start_and_stop_worker():
    """5. Starting and stopping worker updates running state cleanly."""
    mock_gmail = MagicMock(spec=GmailClient)
    mock_gmail.get_status.return_value = {"authorized": True, "configured": True}
    mock_gmail.list_message_ids.return_value = []
    worker_singleton.set_gmail_client(mock_gmail)

    async def _test():
        # Start
        start_res = await start_ingestion_worker(StartWorkerRequest(poll_interval_seconds=120))
        assert start_res["started"] is True
        assert worker_singleton.is_running is True
        assert worker_singleton.poll_interval == 120

        # Repeated Start request must NOT create another worker
        dup_start = await start_ingestion_worker(None)
        assert dup_start["started"] is False
        assert "already running" in dup_start["message"]

        # Stop
        stop_res = await stop_ingestion_worker()
        assert stop_res["stopped"] is True
        assert worker_singleton.is_running is False

        # Repeated Stop request
        dup_stop = await stop_ingestion_worker()
        assert dup_stop["stopped"] is False

    asyncio.run(_test())


def test_6_concurrency_lock_protection():
    """6. Concurrent fetch attempts return 409 Conflict when worker lock is held."""
    test_worker = GmailBackgroundWorker()

    async def run_test():
        await test_worker.lock.acquire()
        try:
            res = await test_worker.run_fetch_cycle(db_session_factory=lambda: None)
            assert res["status"] == "CONCURRENT_BUSY"
        finally:
            test_worker.lock.release()

    asyncio.run(run_test())


def test_7_authorize_endpoint_success():
    """7. POST /api/email-ingestion/authorize initiates reauthorization flow."""
    mock_gmail = MagicMock(spec=GmailClient)
    mock_gmail.authorize.return_value = (True, "Gmail successfully authorized for shriley.ling@gmail.com.")
    worker_singleton.set_gmail_client(mock_gmail)

    async def _run():
        res = await authorize_gmail(AuthorizeRequest(open_browser=False, timeout_seconds=10))
        assert res["success"] is True
        assert "successfully authorized" in res["message"]

    asyncio.run(_run())


def test_8_authorize_endpoint_failure():
    """8. POST /api/email-ingestion/authorize raises 400 on authorization error."""
    mock_gmail = MagicMock(spec=GmailClient)
    mock_gmail.authorize.return_value = (False, "Consent denied by user.")
    worker_singleton.set_gmail_client(mock_gmail)

    async def _run():
        try:
            await authorize_gmail(AuthorizeRequest(open_browser=False, timeout_seconds=10))
            assert False, "Should have raised 400 HTTPException"
        except HTTPException as exc:
            assert exc.status_code == 400
            assert "Consent denied" in str(exc.detail)

    asyncio.run(_run())


def run_all_tests():
    print("=" * 65)
    print("Running Email Ingestion API & Background Worker Tests (Stage 6)...")
    print("=" * 65)
    tests = [
        ("test_1_default_stopped_state", test_1_default_stopped_state),
        ("test_2_status_endpoint", test_2_status_endpoint),
        ("test_3_manual_fetch_when_unauthorized", test_3_manual_fetch_when_unauthorized),
        ("test_4_manual_fetch_when_authorized", test_4_manual_fetch_when_authorized),
        ("test_5_start_and_stop_worker", test_5_start_and_stop_worker),
        ("test_6_concurrency_lock_protection", test_6_concurrency_lock_protection),
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
    print(f"Result: {passed}/{len(tests)} API/worker tests passed successfully!")
    print("=" * 65)


if __name__ == "__main__":
    run_all_tests()
