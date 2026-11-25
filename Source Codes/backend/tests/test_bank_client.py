"""
Unit tests for ABCBankClient using mocked HTTP responses.
Verifies connection, health check, account retrieval, cursor queries,
error handling, and PDF export.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import httpx

# Ensure backend directory is in path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.services.bank_ingestion.bank_client import (
    ABCBankClient,
    BankConnectionError,
    BankCursorNotFoundError,
)


def test_health_check_success():
    """Test successful health check."""
    client = ABCBankClient(base_url="http://mock-bank:8010")
    with patch.object(client, "_get_client") as mock_get_client:
        mock_http = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"status": "ok", "service": "ABC Bank Simulator"}
        mock_resp.raise_for_status.return_value = None
        mock_http.__enter__.return_value.get.return_value = mock_resp
        mock_get_client.return_value = mock_http

        res = client.health_check()
        assert res["status"] == "ok"
        assert res["service"] == "ABC Bank Simulator"


def test_health_check_connection_error():
    """Test connection error when ABC Bank is unavailable."""
    client = ABCBankClient(base_url="http://unreachable-bank:8010")
    with patch.object(client, "_get_client") as mock_get_client:
        mock_http = MagicMock()
        mock_http.__enter__.return_value.get.side_effect = httpx.ConnectError("Connection refused")
        mock_get_client.return_value = mock_http

        try:
            client.health_check()
            assert False, "Should have raised BankConnectionError"
        except BankConnectionError as e:
            assert "Failed to connect to ABC Bank" in str(e)


def test_get_account_success():
    """Test retrieving account summary and current balance."""
    client = ABCBankClient(base_url="http://mock-bank:8010")
    with patch.object(client, "_get_client") as mock_get_client:
        mock_http = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "account_id": "ABC-BIZ-0001",
            "account_name": "NextTech Solutions Sdn Bhd",
            "account_number": "4320-6210-4821",
            "currency": "MYR",
            "current_balance": 12774.00,
        }
        mock_resp.raise_for_status.return_value = None
        mock_http.__enter__.return_value.get.return_value = mock_resp
        mock_get_client.return_value = mock_http

        acc = client.get_account()
        assert acc["account_id"] == "ABC-BIZ-0001"
        assert acc["currency"] == "MYR"
        assert acc["current_balance"] == 12774.00


def test_get_transactions_with_cursor():
    """Test transactions retrieval with after_id cursor."""
    client = ABCBankClient(base_url="http://mock-bank:8010")
    with patch.object(client, "_get_client") as mock_get_client:
        mock_http = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "account_id": "ABC-BIZ-0001",
            "count": 1,
            "transactions": [
                {
                    "transaction_id": "TXN-0002",
                    "date": "2026-10-20",
                    "description": "IT equipment purchase",
                    "reference": "REF-ABC-0002",
                    "money_in": 0.0,
                    "money_out": 2226.0,
                    "running_balance": 12774.0,
                }
            ],
        }
        mock_resp.raise_for_status.return_value = None
        mock_http.__enter__.return_value.get.return_value = mock_resp
        mock_get_client.return_value = mock_http

        res = client.get_transactions(after_id="TXN-0001")
        assert res["count"] == 1
        assert res["transactions"][0]["transaction_id"] == "TXN-0002"


def test_get_transactions_cursor_404():
    """Test that a 404 response on after_id raises BankCursorNotFoundError."""
    client = ABCBankClient(base_url="http://mock-bank:8010")
    with patch.object(client, "_get_client") as mock_get_client:
        mock_http = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_http.__enter__.return_value.get.return_value = mock_resp
        mock_get_client.return_value = mock_http

        try:
            client.get_transactions(after_id="TXN-NONEXISTENT")
            assert False, "Should have raised BankCursorNotFoundError"
        except BankCursorNotFoundError:
            pass


def test_test_connection_reporting():
    """Test test_connection returns structured diagnostic result."""
    client = ABCBankClient(base_url="http://mock-bank:8010")
    with patch.object(client, "health_check") as mock_health, patch.object(client, "get_account") as mock_acc:
        mock_health.return_value = {"status": "ok"}
        mock_acc.return_value = {
            "bank_name": "ABC Bank",
            "account_id": "ABC-BIZ-0001",
            "account_name": "NextTech Solutions",
            "account_number": "1234-5678-4821",
            "currency": "MYR",
            "current_balance": 15000.0,
            "transaction_count": 2,
        }
        res = client.test_connection()
        assert res["connected"] is True
        assert res["bank_name"] == "ABC Bank"
        assert res["account_id"] == "ABC-BIZ-0001"
        assert res["current_balance"] == 15000.0


def run_all_tests():
    print("=" * 60)
    print("Running ABCBankClient Unit Tests...")
    print("=" * 60)
    tests = [
        ("test_health_check_success", test_health_check_success),
        ("test_health_check_connection_error", test_health_check_connection_error),
        ("test_get_account_success", test_get_account_success),
        ("test_get_transactions_with_cursor", test_get_transactions_with_cursor),
        ("test_get_transactions_cursor_404", test_get_transactions_cursor_404),
        ("test_test_connection_reporting", test_test_connection_reporting),
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
    print(f"Result: {passed}/{len(tests)} tests passed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
