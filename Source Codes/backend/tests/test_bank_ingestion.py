"""
Automated tests for ABC Bank Transaction Ingestion Service,
validation, duplicate prevention, and background worker concurrency.
Uses isolated in-memory SQLite fixtures to protect development data.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.db.models import Base, BankTransactionModel, SystemSettingModel, CompanyModel
from app.services.bank_ingestion.bank_service import (
    ABCBankIngestionService,
    parse_txn_number,
    to_decimal,
)
from app.services.bank_ingestion.bank_client import ABCBankClient, BankCursorNotFoundError
from app.services.bank_ingestion.background_worker import ABCBankBackgroundWorker


def create_test_db():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = Session()

    # Seed test company
    comp = CompanyModel(
        company_id="COMP-000001",
        company_name="NextTech Solutions Sdn Bhd",
        currency="MYR"
    )
    session.add(comp)
    session.commit()
    return Session, session


def test_natural_txn_id_ordering():
    """Verify TXN-0010 is recognized as newer than TXN-0009."""
    assert parse_txn_number("TXN-0009") == 9
    assert parse_txn_number("TXN-0010") == 10
    assert parse_txn_number("TXN-0010") > parse_txn_number("TXN-0009")
    assert parse_txn_number("TXN-0100") > parse_txn_number("TXN-0099")


def test_transaction_validation_rules():
    """Test data validation catches malformed or invalid financial entries."""
    _, db = create_test_db()
    service = ABCBankIngestionService(db=db)

    # 1. Missing transaction_id
    valid, err = service.validate_transaction({"date": "2026-10-20", "money_in": 100.0, "money_out": 0.0})
    assert not valid
    assert "Missing transaction ID" in err

    # 2. Missing date
    valid, err = service.validate_transaction({"transaction_id": "TXN-0001", "money_in": 100.0, "money_out": 0.0})
    assert not valid
    assert "Missing transaction date" in err

    # 3. Negative amount
    valid, err = service.validate_transaction({"transaction_id": "TXN-0001", "date": "2026-10-20", "money_in": -50.0, "money_out": 0.0})
    assert not valid
    assert "cannot be negative" in err

    # 4. Both money_in and money_out positive
    valid, err = service.validate_transaction({"transaction_id": "TXN-0001", "date": "2026-10-20", "money_in": 500.0, "money_out": 200.0})
    assert not valid
    assert "cannot have both positive" in err

    # 5. Both zero
    valid, err = service.validate_transaction({"transaction_id": "TXN-0001", "date": "2026-10-20", "money_in": 0.0, "money_out": 0.0})
    assert not valid
    assert "amount cannot be zero" in err

    # 6. Valid incoming
    valid, err = service.validate_transaction({"transaction_id": "TXN-0001", "date": "2026-10-20", "money_in": 5000.0, "money_out": 0.0})
    assert valid
    assert err is None


def test_collection_and_deduplication():
    """Test importing transactions and preventing duplicate records."""
    _, db = create_test_db()
    mock_client = MagicMock(spec=ABCBankClient)
    mock_client.get_account.return_value = {
        "account_id": "ABC-BIZ-0001",
        "account_number": "4320-6210-4821",
        "currency": "MYR",
        "current_balance": 15000.0
    }
    mock_client.get_transactions.return_value = {
        "account_id": "ABC-BIZ-0001",
        "count": 2,
        "transactions": [
            {
                "transaction_id": "TXN-0001",
                "date": "2026-10-20",
                "description": "Customer payment",
                "reference": "REF-ABC-0001",
                "money_in": 5000.0,
                "money_out": 0.0,
                "running_balance": 15000.0
            },
            {
                "transaction_id": "TXN-0002",
                "date": "2026-10-20",
                "description": "IT equipment purchase",
                "reference": "REF-ABC-0002",
                "money_in": 0.0,
                "money_out": 2226.0,
                "running_balance": 12774.0
            }
        ]
    }

    service = ABCBankIngestionService(db=db, bank_client=mock_client)
    res = service.collect_transactions(use_cursor=False)

    assert res["success"] is True
    assert res["transactions_imported"] == 2
    assert res["duplicates_skipped"] == 0

    # Verify DB records
    records = db.query(BankTransactionModel).all()
    assert len(records) == 2
    assert records[0].unique_source_id == "abc_bank:TXN-0001"
    assert records[0].currency == "MYR"
    assert records[0].money_in == 5000.0
    assert records[1].unique_source_id == "abc_bank:TXN-0002"

    # Run collection second time: all should be skipped as duplicates
    res2 = service.collect_transactions(use_cursor=False)
    assert res2["transactions_imported"] == 0
    assert res2["duplicates_skipped"] == 2
    assert len(db.query(BankTransactionModel).all()) == 2


def test_distinct_transactions_with_identical_amounts():
    """Verify two distinct transactions with identical amounts are both preserved."""
    _, db = create_test_db()
    mock_client = MagicMock(spec=ABCBankClient)
    mock_client.get_account.return_value = {
        "account_id": "ABC-BIZ-0001",
        "account_number": "4320-6210-4821",
        "currency": "MYR"
    }
    mock_client.get_transactions.return_value = {
        "account_id": "ABC-BIZ-0001",
        "count": 2,
        "transactions": [
            {
                "transaction_id": "TXN-0001",
                "date": "2026-10-20",
                "description": "Client A subscription",
                "reference": "REF-101",
                "money_in": 1500.0,
                "money_out": 0.0
            },
            {
                "transaction_id": "TXN-0002",
                "date": "2026-10-20",
                "description": "Client B subscription",
                "reference": "REF-102",
                "money_in": 1500.0,  # Exact same amount
                "money_out": 0.0
            }
        ]
    }

    service = ABCBankIngestionService(db=db, bank_client=mock_client)
    res = service.collect_transactions(use_cursor=False)

    assert res["transactions_imported"] == 2
    assert res["duplicates_skipped"] == 0
    records = db.query(BankTransactionModel).all()
    assert len(records) == 2
    assert records[0].bank_transaction_id == "TXN-0001"
    assert records[1].bank_transaction_id == "TXN-0002"


def test_cursor_fallback_on_404():
    """Verify that when cursor after_id returns 404, service falls back to full query and updates cursor."""
    _, db = create_test_db()
    mock_client = MagicMock(spec=ABCBankClient)
    mock_client.get_account.return_value = {"account_id": "ABC-BIZ-0001", "currency": "MYR"}

    def side_effect(after_id=None):
        if after_id == "TXN-OLD-CURSOR":
            raise BankCursorNotFoundError("Cursor not found on bank server")
        return {
            "transactions": [
                {"transaction_id": "TXN-0001", "date": "2026-10-20", "money_in": 500.0, "money_out": 0.0}
            ]
        }

    mock_client.get_transactions.side_effect = side_effect

    service = ABCBankIngestionService(db=db, bank_client=mock_client)
    service.set_last_cursor("TXN-OLD-CURSOR")

    res = service.collect_transactions(use_cursor=True)
    assert res["success"] is True
    assert res["transactions_imported"] == 1
    assert res["last_cursor"] == "TXN-0001"
    assert service.get_last_cursor() == "TXN-0001"


def test_first_fetch_and_repeated_fetch_metrics():
    """Verify first fetch imports TXN-0001 and TXN-0002, and repeated incremental fetch imports no duplicates."""
    _, db = create_test_db()
    mock_client = MagicMock(spec=ABCBankClient)
    mock_client.get_account.return_value = {
        "account_id": "ABC-BIZ-0001",
        "account_number": "4320-6210-4821",
        "currency": "MYR",
        "current_balance": 12774.0
    }

    # Bank data initially has 2 transactions
    all_bank_txs = [
        {"transaction_id": "TXN-0001", "date": "2026-10-20", "description": "Customer payment", "money_in": 5000.0, "money_out": 0.0},
        {"transaction_id": "TXN-0002", "date": "2026-10-20", "description": "IT equipment", "money_in": 0.0, "money_out": 2226.0},
    ]

    def mock_get_txs(after_id=None, **kwargs):
        if not after_id:
            return {"count": len(all_bank_txs), "transactions": all_bank_txs}
        # filter after_id
        after_num = parse_txn_number(after_id)
        filtered = [t for t in all_bank_txs if parse_txn_number(t["transaction_id"]) > after_num]
        return {"count": len(filtered), "transactions": filtered}

    mock_client.get_transactions.side_effect = mock_get_txs
    service = ABCBankIngestionService(db=db, bank_client=mock_client)

    # 1. First fetch: imports TXN-0001 and TXN-0002
    res1 = service.collect_transactions(use_cursor=True)
    assert res1["success"] is True
    assert res1["transactions_checked"] == 2
    assert res1["transactions_imported"] == 2
    assert res1["duplicates_skipped"] == 0
    assert res1["errors_encountered"] == 0
    assert res1["last_cursor"] == "TXN-0002"
    assert res1["total_imported_transactions"] == 2
    assert service.get_last_cursor() == "TXN-0002"

    # 2. Repeated incremental fetch: after_id='TXN-0002' returns empty list
    res2 = service.collect_transactions(use_cursor=True)
    assert res2["success"] is True
    assert res2["transactions_checked"] == 0
    assert res2["transactions_imported"] == 0
    assert res2["duplicates_skipped"] == 0
    assert res2["errors_encountered"] == 0
    assert res2["last_cursor"] == "TXN-0002"
    assert res2["total_imported_transactions"] == 2
    # Cursor must remain unchanged
    assert service.get_last_cursor() == "TXN-0002"


def test_empty_incremental_fetch_does_not_corrupt_cursor():
    """Verify that empty incremental response does not wipe, nullify, or corrupt the cursor."""
    _, db = create_test_db()
    mock_client = MagicMock(spec=ABCBankClient)
    mock_client.get_account.return_value = {"account_id": "ABC-BIZ-0001", "currency": "MYR"}
    mock_client.get_transactions.return_value = {"count": 0, "transactions": []}

    service = ABCBankIngestionService(db=db, bank_client=mock_client)
    service.set_last_cursor("TXN-0009")

    res = service.collect_transactions(use_cursor=True)
    assert res["transactions_checked"] == 0
    assert res["transactions_imported"] == 0
    assert res["last_cursor"] == "TXN-0009"
    # Verify cursor remains TXN-0009
    assert service.get_last_cursor() == "TXN-0009"


def test_new_transaction_after_cursor_imported():
    """Verify that a new transaction arriving at the bank after cursor is imported and advances cursor."""
    _, db = create_test_db()
    mock_client = MagicMock(spec=ABCBankClient)
    mock_client.get_account.return_value = {"account_id": "ABC-BIZ-0001", "currency": "MYR"}

    service = ABCBankIngestionService(db=db, bank_client=mock_client)
    # Existing cursor at TXN-0002
    service.set_last_cursor("TXN-0002")

    # Bank returns TXN-0003 for query after_id=TXN-0002
    mock_client.get_transactions.return_value = {
        "count": 1,
        "transactions": [
            {"transaction_id": "TXN-0003", "date": "2026-10-21", "description": "Consulting fee", "money_in": 3500.0, "money_out": 0.0}
        ]
    }

    res = service.collect_transactions(use_cursor=True)
    assert res["success"] is True
    assert res["transactions_checked"] == 1
    assert res["transactions_imported"] == 1
    assert res["duplicates_skipped"] == 0
    assert res["last_cursor"] == "TXN-0003"
    assert service.get_last_cursor() == "TXN-0003"

    # Verify newly saved record
    record = db.query(BankTransactionModel).filter(BankTransactionModel.bank_transaction_id == "TXN-0003").first()
    assert record is not None
    assert record.money_in == 3500.0


def test_simulator_reset_recovers_cursor_and_avoids_duplicates():
    """Verify that if simulator resets (returns 404 on stale cursor), fallback fetches all and resets cursor properly."""
    _, db = create_test_db()
    mock_client = MagicMock(spec=ABCBankClient)
    mock_client.get_account.return_value = {"account_id": "ABC-BIZ-0001", "currency": "MYR"}

    service = ABCBankIngestionService(db=db, bank_client=mock_client)

    # Pre-populate DB with TXN-0001 and TXN-0002
    now_tx1 = BankTransactionModel(
        company_id="COMP-000001",
        source="abc_bank",
        unique_source_id="abc_bank:TXN-0001",
        bank_name="ABC Bank",
        account_id="ABC-BIZ-0001",
        account_number="4320-6210-4821",
        bank_transaction_id="TXN-0001",
        transaction_date="2026-10-20",
        description="Payment 1",
        money_in=5000.0,
        money_out=0.0,
        currency="MYR",
        collection_status="IMPORTED"
    )
    db.add(now_tx1)
    db.commit()

    # Suppose database cursor was at TXN-0005, but bank simulator was reset back to TXN-0001 and TXN-0002
    service.set_last_cursor("TXN-0005")

    def mock_get_txs_reset(after_id=None, **kwargs):
        if after_id == "TXN-0005":
            raise BankCursorNotFoundError("Cursor TXN-0005 not found on bank server.")
        return {
            "count": 2,
            "transactions": [
                {"transaction_id": "TXN-0001", "date": "2026-10-20", "description": "Payment 1", "money_in": 5000.0, "money_out": 0.0},
                {"transaction_id": "TXN-0002", "date": "2026-10-20", "description": "Payment 2", "money_in": 1000.0, "money_out": 0.0},
            ]
        }

    mock_client.get_transactions.side_effect = mock_get_txs_reset

    res = service.collect_transactions(use_cursor=True)
    assert res["success"] is True
    assert res["transactions_checked"] == 2
    assert res["transactions_imported"] == 1  # TXN-0002 is imported
    assert res["duplicates_skipped"] == 1    # TXN-0001 was already in DB
    # Cursor must be resynchronized to TXN-0002 (the highest ID on the reset bank)
    assert res["last_cursor"] == "TXN-0002"
    assert service.get_last_cursor() == "TXN-0002"


def test_backend_restart_preserves_records_and_cursor():
    """Verify that re-instantiating services (simulating backend restart) retains all records and cursor."""
    SessionFactory, db = create_test_db()
    mock_client = MagicMock(spec=ABCBankClient)
    mock_client.test_connection.return_value = {
        "connected": True,
        "bank_name": "ABC Bank",
        "account_id": "ABC-BIZ-0001",
        "account_number": "4320-6210-4821",
        "current_balance": 12774.0
    }

    # Add 2 transactions and cursor
    tx1 = BankTransactionModel(
        company_id="COMP-000001",
        source="abc_bank",
        unique_source_id="abc_bank:TXN-0001",
        bank_name="ABC Bank",
        account_id="ABC-BIZ-0001",
        account_number="4320-6210-4821",
        bank_transaction_id="TXN-0001",
        transaction_date="2026-10-20",
        description="Payment",
        money_in=5000.0,
        money_out=0.0,
        currency="MYR",
        collection_status="IMPORTED"
    )
    tx2 = BankTransactionModel(
        company_id="COMP-000001",
        source="abc_bank",
        unique_source_id="abc_bank:TXN-0002",
        bank_name="ABC Bank",
        account_id="ABC-BIZ-0001",
        account_number="4320-6210-4821",
        bank_transaction_id="TXN-0002",
        transaction_date="2026-10-20",
        description="IT equipment",
        money_in=0.0,
        money_out=2226.0,
        currency="MYR",
        collection_status="IMPORTED"
    )
    setting = SystemSettingModel(
        setting_key="abc_bank_last_cursor",
        setting_value="TXN-0002"
    )
    db.add(tx1)
    db.add(tx2)
    db.add(setting)
    db.commit()

    # Simulate fresh backend start: new DB session and new worker singleton
    db.close()
    fresh_session = SessionFactory()

    new_worker = ABCBankBackgroundWorker()
    new_worker.set_client(mock_client)

    status = new_worker.get_status(db=fresh_session)
    assert status["status"] == "STOPPED"  # Default STOPPED on restart
    assert status["is_running"] is False
    assert status["total_imported_transactions"] == 2  # Records preserved
    assert status["last_cursor"] == "TXN-0002"         # Cursor preserved

    new_service = ABCBankIngestionService(db=fresh_session, bank_client=mock_client)
    assert new_service.get_last_cursor() == "TXN-0002"
    fresh_session.close()


def test_collection_metrics_distinguish_latest_fetch_from_total():
    """Verify latest fetch counters (checked, imported, duplicates) are distinct from total database count."""
    _, db = create_test_db()
    mock_client = MagicMock(spec=ABCBankClient)
    mock_client.get_account.return_value = {"account_id": "ABC-BIZ-0001", "currency": "MYR"}

    service = ABCBankIngestionService(db=db, bank_client=mock_client)

    # 1. First fetch imports 2 transactions
    mock_client.get_transactions.return_value = {
        "count": 2,
        "transactions": [
            {"transaction_id": "TXN-0001", "date": "2026-10-20", "money_in": 100.0, "money_out": 0.0},
            {"transaction_id": "TXN-0002", "date": "2026-10-20", "money_in": 200.0, "money_out": 0.0},
        ]
    }
    res1 = service.collect_transactions(use_cursor=True)
    assert res1["transactions_checked"] == 2
    assert res1["transactions_imported"] == 2
    assert res1["total_imported_transactions"] == 2

    # 2. Incremental fetch: 0 new items
    mock_client.get_transactions.return_value = {"count": 0, "transactions": []}
    res2 = service.collect_transactions(use_cursor=True)
    # Latest fetch is 0 checked, 0 imported
    assert res2["transactions_checked"] == 0
    assert res2["transactions_imported"] == 0
    # But total in database is still 2!
    assert res2["total_imported_transactions"] == 2


def test_worker_lifecycle_and_single_instance():
    """Verify starting worker twice does not duplicate background workers, and stop works cleanly."""
    SessionFactory, _ = create_test_db()
    worker = ABCBankBackgroundWorker()

    mock_client = MagicMock(spec=ABCBankClient)
    mock_client.test_connection.return_value = {"connected": True}
    mock_client.get_account.return_value = {"account_id": "ABC-BIZ-0001", "currency": "MYR"}
    mock_client.get_transactions.return_value = {"transactions": []}
    worker.set_client(mock_client)

    # 1. Initial state is STOPPED
    assert worker.is_running is False
    assert worker.get_status()["status"] == "STOPPED"

    # 2. Start worker
    res1 = worker.start(SessionFactory, poll_interval=10)
    assert res1["started"] is True
    assert worker.is_running is True
    assert worker.get_status()["status"] == "RUNNING"

    # 3. Attempt starting again -> rejected as already running
    res2 = worker.start(SessionFactory)
    assert res2["started"] is False
    assert "already running" in res2["message"]

    # 4. Stop worker
    res_stop = worker.stop()
    assert res_stop["stopped"] is True
    assert worker.is_running is False
    assert worker.get_status()["status"] == "STOPPED"


def run_all_tests():
    print("=" * 60)
    print("Running ABCBankIngestionService Unit & Integration Tests...")
    print("=" * 60)
    tests = [
        ("test_natural_txn_id_ordering", test_natural_txn_id_ordering),
        ("test_transaction_validation_rules", test_transaction_validation_rules),
        ("test_collection_and_deduplication", test_collection_and_deduplication),
        ("test_distinct_transactions_with_identical_amounts", test_distinct_transactions_with_identical_amounts),
        ("test_cursor_fallback_on_404", test_cursor_fallback_on_404),
        ("test_first_fetch_and_repeated_fetch_metrics", test_first_fetch_and_repeated_fetch_metrics),
        ("test_empty_incremental_fetch_does_not_corrupt_cursor", test_empty_incremental_fetch_does_not_corrupt_cursor),
        ("test_new_transaction_after_cursor_imported", test_new_transaction_after_cursor_imported),
        ("test_simulator_reset_recovers_cursor_and_avoids_duplicates", test_simulator_reset_recovers_cursor_and_avoids_duplicates),
        ("test_backend_restart_preserves_records_and_cursor", test_backend_restart_preserves_records_and_cursor),
        ("test_collection_metrics_distinguish_latest_fetch_from_total", test_collection_metrics_distinguish_latest_fetch_from_total),
        ("test_worker_lifecycle_and_single_instance", test_worker_lifecycle_and_single_instance),
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
