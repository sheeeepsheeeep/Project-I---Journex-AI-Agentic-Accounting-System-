"""
Tests for GmailClient OAuth status, read-only email retrieval,
and attachment parsing using mocked Gmail API responses.
Can be executed with either pytest or standard python test_gmail_client.py.
"""

import sys
import os
import base64
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
from googleapiclient.errors import HttpError
import httplib2

# Ensure backend folder is in path
backend_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_path))

from app.services.email_ingestion.gmail_client import GmailClient, GmailAuthError


def test_unconfigured_oauth_status(tmp_path: Path):
    """Test OAuth status when credentials and token files do not exist."""
    client = GmailClient(
        credentials_path=str(tmp_path / "nonexistent_creds.json"),
        token_path=str(tmp_path / "nonexistent_token.json"),
        company_inbox="shriley.ling@gmail.com"
    )
    status = client.get_status()
    assert status["configured"] is False, "Should be unconfigured"
    assert status["authorized"] is False, "Should be unauthorized"
    assert status["token_exists"] is False, "Token should not exist"
    assert status["company_inbox"] == "shriley.ling@gmail.com"
    assert "https://www.googleapis.com/auth/gmail.readonly" in status["scopes"]
    assert client.is_authorized() is False, "is_authorized() should return False"


def test_credentials_exist_but_unauthorized(tmp_path: Path):
    """Test status when credentials.json exists but token.json has not been created."""
    creds_file = tmp_path / "credentials.json"
    creds_file.write_text('{"installed": {"client_id": "test"}}')

    client = GmailClient(
        credentials_path=str(creds_file),
        token_path=str(tmp_path / "nonexistent_token.json"),
        company_inbox="shriley.ling@gmail.com"
    )
    status = client.get_status()
    assert status["configured"] is True, "Should be configured"
    assert status["authorized"] is False, "Should be unauthorized"
    assert status["token_exists"] is False, "Token should not exist"
    assert client.is_authorized() is False


def test_list_messages_with_mocked_service():
    """Test listing messages using a mocked Gmail API service."""
    mock_service = MagicMock()
    mock_list = MagicMock()
    mock_list.execute.return_value = {
        "messages": [
            {"id": "msg_001", "threadId": "th_001"},
            {"id": "msg_002", "threadId": "th_002"}
        ]
    }
    mock_service.users().messages().list.return_value = mock_list

    client = GmailClient(service=mock_service)
    message_ids = client.list_message_ids(query="has:attachment", max_results=10)

    assert message_ids == ["msg_001", "msg_002"]
    mock_service.users().messages().list.assert_called_once_with(
        userId="me",
        q="has:attachment",
        maxResults=10
    )


def test_get_message_detail_with_attachments():
    """Test parsing message metadata, sender, subject, date, and multiple attachments."""
    mock_service = MagicMock()
    mock_get = MagicMock()
    mock_get.execute.return_value = {
        "id": "msg_invoice_123",
        "threadId": "th_123",
        "snippet": "Please find attached our latest supplier invoice for September.",
        "payload": {
            "headers": [
                {"name": "From", "value": "TechCore Solutions <billing@techcore.com>"},
                {"name": "To", "value": "shriley.ling@gmail.com"},
                {"name": "Subject", "value": "Invoice INV-2026-0042 from TechCore"},
                {"name": "Date", "value": "Tue, 20 Oct 2026 14:30:00 +0800"}
            ],
            "parts": [
                {
                    "partId": "0",
                    "mimeType": "text/plain",
                    "filename": "",
                    "body": {"size": 65}
                },
                {
                    "partId": "1",
                    "mimeType": "application/pdf",
                    "filename": "supplier_invoice_0042.pdf",
                    "body": {"attachmentId": "att_pdf_999", "size": 104200}
                },
                {
                    "partId": "2",
                    "mimeType": "image/png",
                    "filename": "payment_receipt.png",
                    "body": {"attachmentId": "att_png_888", "size": 52100}
                },
                {
                    "partId": "3",
                    "mimeType": "application/zip",
                    "filename": "archive.zip",
                    "body": {"attachmentId": "att_zip_777", "size": 204800}
                }
            ]
        }
    }
    mock_service.users().messages().get.return_value = mock_get

    client = GmailClient(service=mock_service)
    detail = client.get_message_detail("msg_invoice_123")

    assert detail["message_id"] == "msg_invoice_123"
    assert detail["sender_name"] == "TechCore Solutions"
    assert detail["sender_email"] == "billing@techcore.com"
    assert detail["subject"] == "Invoice INV-2026-0042 from TechCore"
    assert "2026-10-20" in detail["received_at"]
    assert detail["attachment_count"] == 3

    # Check attachments
    attachments = detail["attachments"]
    pdf_att = next(a for a in attachments if a["filename"] == "supplier_invoice_0042.pdf")
    assert pdf_att["is_supported"] is True
    assert pdf_att["extension"] == ".pdf"
    assert pdf_att["attachment_id"] == "att_pdf_999"

    png_att = next(a for a in attachments if a["filename"] == "payment_receipt.png")
    assert png_att["is_supported"] is True
    assert png_att["extension"] == ".png"

    zip_att = next(a for a in attachments if a["filename"] == "archive.zip")
    assert zip_att["is_supported"] is False  # Unsupported format


def test_get_message_detail_without_attachments():
    """Test parsing an email that contains no attachments."""
    mock_service = MagicMock()
    mock_get = MagicMock()
    mock_get.execute.return_value = {
        "id": "msg_plain_001",
        "threadId": "th_001",
        "snippet": "Just a casual inquiry regarding meeting schedule.",
        "payload": {
            "headers": [
                {"name": "From", "value": "john.doe@example.com"},
                {"name": "Subject", "value": "Quick question"}
            ],
            "parts": [
                {
                    "partId": "0",
                    "mimeType": "text/plain",
                    "filename": "",
                    "body": {"size": 45}
                }
            ]
        }
    }
    mock_service.users().messages().get.return_value = mock_get

    client = GmailClient(service=mock_service)
    detail = client.get_message_detail("msg_plain_001")

    assert detail["message_id"] == "msg_plain_001"
    assert detail["sender_email"] == "john.doe@example.com"
    assert detail["attachment_count"] == 0
    assert len(detail["attachments"]) == 0


def test_download_attachment_bytes():
    """Test downloading and base64-decoding attachment content."""
    mock_service = MagicMock()
    mock_attachment_req = MagicMock()

    sample_content = b"%PDF-1.4 sample pdf document content"
    b64_encoded = base64.urlsafe_b64encode(sample_content).decode("ascii")

    mock_attachment_req.execute.return_value = {
        "size": len(sample_content),
        "data": b64_encoded
    }
    mock_service.users().messages().attachments().get.return_value = mock_attachment_req

    client = GmailClient(service=mock_service)
    downloaded_bytes = client.download_attachment_bytes("msg_123", "att_999")

    assert downloaded_bytes == sample_content
    mock_service.users().messages().attachments().get.assert_called_once_with(
        userId="me",
        messageId="msg_123",
        id="att_999"
    )


def test_http_error_handling():
    """Test handling of Gmail API HttpError gracefully."""
    mock_service = MagicMock()
    mock_get = MagicMock()

    resp = httplib2.Response({"status": 404, "reason": "Not Found"})
    mock_get.execute.side_effect = HttpError(resp=resp, content=b"Message not found")
    mock_service.users().messages().get.return_value = mock_get

    client = GmailClient(service=mock_service)
    try:
        client.get_message_detail("nonexistent_id")
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "Gmail API error" in str(e)


def test_auto_refresh_on_expired_token(tmp_path: Path):
    """Test that an expired token with refresh_token is automatically refreshed on get_status()."""
    token_file = tmp_path / "token.json"
    token_file.write_text('{"token": "dummy"}')

    with patch("app.services.email_ingestion.gmail_client.Credentials") as mock_creds_cls:
        mock_creds = MagicMock()
        mock_creds.valid = False
        mock_creds.expired = True
        mock_creds.refresh_token = "dummy_ref"
        mock_creds.to_json.return_value = '{"token": "new_refreshed_tok"}'

        def simulate_refresh(req):
            mock_creds.valid = True
            mock_creds.expired = False

        mock_creds.refresh.side_effect = simulate_refresh
        mock_creds_cls.from_authorized_user_file.return_value = mock_creds

        client = GmailClient(
            credentials_path=str(tmp_path / "creds.json"),
            token_path=str(token_file),
            company_inbox="shriley.ling@gmail.com"
        )
        status = client.get_status()

        mock_creds.refresh.assert_called_once()
        assert status["authorized"] is True
        assert status["token_expired"] is False
        assert "Authorized for shriley.ling@gmail.com" in status["status_message"]


def test_invalid_grant_revoked_token_status(tmp_path: Path):
    """Test that invalid_grant / revoked token produces clean error and flags re-authorization required."""
    token_file = tmp_path / "token.json"
    token_file.write_text('{"token": "dummy_revoked"}')

    with patch("app.services.email_ingestion.gmail_client.Credentials") as mock_creds_cls:
        mock_creds = MagicMock()
        mock_creds.valid = False
        mock_creds.expired = True
        mock_creds.refresh_token = "dummy_ref"
        mock_creds.refresh.side_effect = Exception("invalid_grant: Token has been expired or revoked.")
        mock_creds_cls.from_authorized_user_file.return_value = mock_creds

        client = GmailClient(
            credentials_path=str(tmp_path / "creds.json"),
            token_path=str(token_file),
            company_inbox="shriley.ling@gmail.com"
        )
        status = client.get_status()

        assert status["authorized"] is False
        assert status["needs_reauthorization"] is True
        assert "expired or been revoked" in status["status_message"].lower()


def test_authorize_flow_success(tmp_path: Path):
    """Test interactive OAuth authorize() saves token.json and sets offline access."""
    creds_file = tmp_path / "credentials.json"
    creds_file.write_text('{"installed": {"client_id": "test_cid"}}')
    token_file = tmp_path / "token.json"

    client = GmailClient(
        credentials_path=str(creds_file),
        token_path=str(token_file),
        company_inbox="shriley.ling@gmail.com"
    )

    with patch("app.services.email_ingestion.gmail_client.InstalledAppFlow") as mock_flow_cls:
        mock_flow = MagicMock()
        mock_creds = MagicMock()
        mock_creds.to_json.return_value = '{"token": "fresh_authorized_token", "refresh_token": "fresh_ref"}'
        mock_flow.run_local_server.return_value = mock_creds
        mock_flow_cls.from_client_secrets_file.return_value = mock_flow

        success, msg = client.authorize(open_browser=False, timeout_seconds=60)

        assert success is True
        assert "successfully authorized" in msg.lower()
        mock_flow.run_local_server.assert_called_once()
        _, kwargs = mock_flow.run_local_server.call_args
        assert kwargs.get("access_type") == "offline"
        assert kwargs.get("prompt") == "consent"
        assert token_file.exists()
        assert "fresh_authorized_token" in token_file.read_text()


def run_all_tests():
    """Runs all test cases and prints clean summary."""
    print("=" * 60)
    print("Running GmailClient Unit Tests (Mocked API Responses)...")
    print("=" * 60)
    
    temp_dir = Path(tempfile.mkdtemp(prefix="gmail_test_"))
    try:
        tests = [
            ("test_unconfigured_oauth_status", lambda: test_unconfigured_oauth_status(temp_dir)),
            ("test_credentials_exist_but_unauthorized", lambda: test_credentials_exist_but_unauthorized(temp_dir)),
            ("test_auto_refresh_on_expired_token", lambda: test_auto_refresh_on_expired_token(temp_dir)),
            ("test_list_messages_with_mocked_service", test_list_messages_with_mocked_service),
            ("test_get_message_detail_with_attachments", test_get_message_detail_with_attachments),
            ("test_get_message_detail_without_attachments", test_get_message_detail_without_attachments),
            ("test_download_attachment_bytes", test_download_attachment_bytes),
            ("test_http_error_handling", test_http_error_handling)
        ]
        
        passed = 0
        for name, test_fn in tests:
            try:
                test_fn()
                print(f"[PASS] {name}")
                passed += 1
            except Exception as e:
                print(f"[FAIL] {name}: {e}")
                raise e
                
        print("=" * 60)
        print(f"Result: {passed}/{len(tests)} tests passed successfully!")
        print("=" * 60)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    run_all_tests()
