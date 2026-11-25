import os
import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

import pytest
import datetime
from decimal import Decimal
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.db.database import Base, get_db, seed_default_chart_of_accounts
from app.db.models import (
    ChartOfAccountModel,
    AccrualPrepaymentModel,
    AccrualAdjustmentRecordModel,
    JournalEntryModel
)
from app.services.accrual_prepayment_service import AccrualPrepaymentService, _to_decimal, _round_float
from app.schemas.accrual_prepayment import (
    AccrualPrepaymentCreate,
    AccrualAdjustmentRequest,
    AccrualSettlementRequest,
    AccrualReversalRequest,
    AccrualCancellationRequest
)


# Setup in-memory SQLite with StaticPool for isolated test execution across requests
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="function")
def db_session():
    Base.metadata.create_all(bind=engine)
    session = TestingSessionLocal()
    seed_default_chart_of_accounts(session, "COMP-000001", force_reset=True)
    yield session
    session.close()
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def client(db_session):
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    test_client = TestClient(app)
    yield test_client
    app.dependency_overrides.clear()


# =========================================================================
# 1. CHART OF ACCOUNTS & CLASSIFICATION TESTS
# =========================================================================

def test_chart_of_accounts_accruals_prepayments_exist(db_session):
    # Accrued Expenses: 410-0000 ACCRUALS (Liability)
    acc = db_session.query(ChartOfAccountModel).filter(ChartOfAccountModel.account_code == "410-0000").first()
    assert acc is not None
    assert acc.account_type == "Liability"
    assert acc.is_active is True

    # Current Prepayments: 340-1000 PREPAID EXPENSES (Asset)
    prp = db_session.query(ChartOfAccountModel).filter(ChartOfAccountModel.account_code == "340-1000").first()
    assert prp is not None
    assert prp.account_type == "Asset"

    # Accrued Income: 305-1000 ACCRUED INCOME (Asset)
    ai = db_session.query(ChartOfAccountModel).filter(ChartOfAccountModel.account_code == "305-1000").first()
    assert ai is not None
    assert ai.account_type == "Asset"

    # Deferred Income: 440-1000 DEFERRED INCOME (Liability)
    di = db_session.query(ChartOfAccountModel).filter(ChartOfAccountModel.account_code == "440-1000").first()
    assert di is not None
    assert di.account_type == "Liability"

    # Non-current accounts (> 12 months)
    nc_prp = db_session.query(ChartOfAccountModel).filter(ChartOfAccountModel.account_code == "220-0000").first()
    assert nc_prp is not None
    assert nc_prp.account_type == "Asset"

    nc_def = db_session.query(ChartOfAccountModel).filter(ChartOfAccountModel.account_code == "450-0000").first()
    assert nc_def is not None
    assert nc_def.account_type == "Liability"


# =========================================================================
# 2. SITUATION A: ACCRUED EXPENSES (RECOGNITION, SETTLEMENT, REVERSAL)
# =========================================================================

def test_accrued_expense_full_flow(db_session):
    service = AccrualPrepaymentService(db_session)

    # 1. December Electricity: RM 500
    create_req = AccrualPrepaymentCreate(
        transaction_type="ACCRUED_EXPENSE",
        title="December Electricity Accrual",
        vendor_customer_name="Tenaga Nasional Berhad",
        original_amount=500.0,
        currency="MYR",
        transaction_date="2026-12-31",
        start_date="2026-12-01",
        end_date="2026-12-31",
        target_account_code="907-0000",  # WATER & ELECTRICITY (Expense)
        post_initial_journal=True
    )
    record = service.create_record(create_req)
    assert record.id is not None
    assert record.remaining_balance == 500.0
    assert record.balance_sheet_account_code == "410-0000"
    assert record.initial_journal_id is not None

    # Check Initial Journal: Dr 907-0000 RM 500, Cr 410-0000 RM 500
    j_init = db_session.query(JournalEntryModel).filter(JournalEntryModel.journal_id == record.initial_journal_id).first()
    assert j_init.is_balanced is True
    assert j_init.total_debit == 500.0
    assert j_init.total_credit == 500.0

    # 2. January Payment Settlement: RM 500
    # Debit 410-0000 RM 500, Credit 310-1000 RM 500 (Expense is NOT duplicated)
    settle_req = AccrualSettlementRequest(
        settlement_date="2027-01-15",
        amount=500.0,
        bank_account_code="310-1000"
    )
    adj = service.post_settlement(record.id, settle_req)
    assert record.remaining_balance == 0.0
    assert record.status == "FULLY_ADJUSTED"
    assert adj.journal_id is not None

    j_settle = db_session.query(JournalEntryModel).filter(JournalEntryModel.journal_id == adj.journal_id).first()
    assert j_settle.is_balanced is True
    assert j_settle.total_debit == 500.0
    assert j_settle.total_credit == 500.0


def test_accrued_expense_partial_settlement(db_session):
    service = AccrualPrepaymentService(db_session)
    record = service.create_record(AccrualPrepaymentCreate(
        transaction_type="ACCRUED_EXPENSE",
        title="Legal Fee Accrual",
        original_amount=1000.0,
        transaction_date="2026-10-31",
        start_date="2026-10-01",
        end_date="2026-10-31",
        target_account_code="915-0000"  # GENERAL EXPENSES
    ))

    # Partial payment of RM 400
    service.post_settlement(record.id, AccrualSettlementRequest(
        settlement_date="2026-11-10",
        amount=400.0
    ))
    assert record.remaining_balance == 600.0
    assert record.status == "PARTIALLY_ADJUSTED"

    # Settle remaining RM 600
    service.post_settlement(record.id, AccrualSettlementRequest(
        settlement_date="2026-11-25",
        amount=600.0
    ))
    assert record.remaining_balance == 0.0
    assert record.status == "FULLY_ADJUSTED"


def test_accrued_expense_reversal_and_duplicate_reversal_prevention(db_session):
    service = AccrualPrepaymentService(db_session)
    record = service.create_record(AccrualPrepaymentCreate(
        transaction_type="ACCRUED_EXPENSE",
        title="Audit Fee Accrual",
        original_amount=2500.0,
        transaction_date="2026-12-31",
        start_date="2026-12-01",
        end_date="2026-12-31",
        target_account_code="915-0000"
    ))

    # Reversal on 2027-01-01: Dr 410-0000 RM 2500, Cr 915-0000 RM 2500
    rev_req = AccrualReversalRequest(reversal_date="2027-01-01")
    service.post_reversal(record.id, rev_req)
    assert record.is_reversed is True
    assert record.reversal_journal_id is not None
    assert record.remaining_balance == 0.0
    assert record.status == "FULLY_ADJUSTED"

    j_rev = db_session.query(JournalEntryModel).filter(JournalEntryModel.journal_id == record.reversal_journal_id).first()
    assert j_rev.is_balanced is True
    assert j_rev.total_debit == 2500.0
    assert j_rev.total_credit == 2500.0

    # Duplicate reversal MUST fail
    with pytest.raises(ValueError, match="already been reversed"):
        service.post_reversal(record.id, rev_req)


# =========================================================================
# 3. SITUATION B: PREPAID EXPENSES (PAYMENT, MONTHLY ALLOCATION, REFUND)
# =========================================================================

def test_prepaid_expense_monthly_amortization(db_session):
    service = AccrualPrepaymentService(db_session)

    # 12-month Insurance: RM 1,200 (Jan 1 to Dec 31, 2026)
    record = service.create_record(AccrualPrepaymentCreate(
        transaction_type="PREPAID_EXPENSE",
        title="Annual Fire Insurance",
        vendor_customer_name="Allianz Malaysia",
        original_amount=1200.0,
        currency="MYR",
        transaction_date="2026-01-01",
        start_date="2026-01-01",
        end_date="2026-12-31",
        target_account_code="915-0000",
        bank_account_code="310-1000",
        post_initial_journal=True
    ))

    # Initial Journal: Dr 340-1000 RM 1200, Cr 310-1000 RM 1200
    j_init = db_session.query(JournalEntryModel).filter(JournalEntryModel.journal_id == record.initial_journal_id).first()
    assert j_init.is_balanced is True
    assert j_init.total_debit == 1200.0

    # Schedule calculation: exactly 12 periods, total allocated = RM 1200.00
    schedule = service.calculate_amortization_schedule(1200.0, 1.0, "2026-01-01", "2026-12-31")
    assert len(schedule) == 12
    tot_sched = sum(s.allocated_amount for s in schedule)
    assert abs(tot_sched - 1200.0) < 0.01

    # January Amortization: Dr 915-0000, Cr 340-1000 (RM 101.92 based on 31 days)
    jan_amt = schedule[0].allocated_amount
    service.post_amortization_adjustment(record.id, AccrualAdjustmentRequest(
        adjustment_date="2026-01-31",
        amount=jan_amt,
        period_name="January 2026"
    ))
    assert abs(record.remaining_balance - (1200.0 - jan_amt)) < 0.01
    assert record.status == "PARTIALLY_ADJUSTED"


def test_prepaid_expense_early_cancellation_and_refund(db_session):
    service = AccrualPrepaymentService(db_session)
    record = service.create_record(AccrualPrepaymentCreate(
        transaction_type="PREPAID_EXPENSE",
        title="Vehicle Insurance Policy",
        original_amount=1000.0,
        transaction_date="2026-01-01",
        start_date="2026-01-01",
        end_date="2026-12-31",
        target_account_code="906-0000"  # UPKEEP OF MOTOR VEHICLE
    ))

    # Amortize 200 after 2 months
    service.post_amortization_adjustment(record.id, AccrualAdjustmentRequest(
        adjustment_date="2026-02-28",
        amount=200.0
    ))
    assert record.remaining_balance == 800.0

    # Early cancellation and refund of remaining 800 in cash: Dr 310-1000, Cr 340-1000
    cncl_req = AccrualCancellationRequest(
        cancellation_date="2026-03-01",
        refund_amount=800.0,
        bank_account_code="310-1000"
    )
    adj = service.post_cancellation_and_refund(record.id, cncl_req)
    assert record.remaining_balance == 0.0
    assert record.status == "CANCELLED"

    j_rf = db_session.query(JournalEntryModel).filter(JournalEntryModel.journal_id == adj.journal_id).first()
    assert j_rf.is_balanced is True
    assert j_rf.total_debit == 800.0
    assert j_rf.total_credit == 800.0


# =========================================================================
# 4. SITUATION C: ACCRUED INCOME (RECOGNITION, COLLECTION, REVERSAL)
# =========================================================================

def test_accrued_income_full_flow(db_session):
    service = AccrualPrepaymentService(db_session)

    # Completed IT Consultancy in Dec: RM 2,000 unbilled
    record = service.create_record(AccrualPrepaymentCreate(
        transaction_type="ACCRUED_INCOME",
        title="December IT Consulting Services",
        vendor_customer_name="Apex Global Bhd",
        original_amount=2000.0,
        transaction_date="2026-12-31",
        start_date="2026-12-01",
        end_date="2026-12-31",
        target_account_code="500-0000"  # SALES (Revenue)
    ))

    # Initial Journal: Dr 305-1000 (Accrued Income Asset) RM 2000, Cr 500-0000 (Sales) RM 2000
    j_init = db_session.query(JournalEntryModel).filter(JournalEntryModel.journal_id == record.initial_journal_id).first()
    assert j_init.is_balanced is True
    assert j_init.total_debit == 2000.0
    assert record.balance_sheet_account_code == "305-1000"

    # January Collection from customer: Dr 310-1000 RM 2000, Cr 305-1000 RM 2000 (Revenue NOT duplicated)
    service.post_settlement(record.id, AccrualSettlementRequest(
        settlement_date="2027-01-20",
        amount=2000.0,
        bank_account_code="310-1000"
    ))
    assert record.remaining_balance == 0.0
    assert record.status == "FULLY_ADJUSTED"


def test_accrued_income_reversal(db_session):
    service = AccrualPrepaymentService(db_session)
    record = service.create_record(AccrualPrepaymentCreate(
        transaction_type="ACCRUED_INCOME",
        title="Unbilled Project Milestone",
        original_amount=3000.0,
        transaction_date="2026-12-31",
        start_date="2026-12-01",
        end_date="2026-12-31",
        target_account_code="500-0000"
    ))

    # Reversal on 2027-01-01: Dr 500-0000 RM 3000, Cr 305-1000 RM 3000
    service.post_reversal(record.id, AccrualReversalRequest(reversal_date="2027-01-01"))
    assert record.is_reversed is True
    assert record.status == "FULLY_ADJUSTED"

    j_rev = db_session.query(JournalEntryModel).filter(JournalEntryModel.journal_id == record.reversal_journal_id).first()
    assert j_rev.is_balanced is True
    assert j_rev.total_debit == 3000.0


# =========================================================================
# 5. SITUATION D: DEFERRED INCOME (COLLECTION, AMORTIZATION, REFUND)
# =========================================================================

def test_deferred_income_full_flow(db_session):
    service = AccrualPrepaymentService(db_session)

    # 3-Month Customer Maintenance Retainer: RM 3,000 (RM 1,000/month)
    record = service.create_record(AccrualPrepaymentCreate(
        transaction_type="DEFERRED_INCOME",
        title="Q4 IT Maintenance Retainer",
        vendor_customer_name="CloudTech Enterprise",
        original_amount=3000.0,
        transaction_date="2026-10-01",
        start_date="2026-10-01",
        end_date="2026-12-31",
        target_account_code="500-0000",
        bank_account_code="310-1000"
    ))

    # Initial Journal: Dr 310-1000 RM 3000, Cr 440-1000 (Deferred Income Liability) RM 3000
    j_init = db_session.query(JournalEntryModel).filter(JournalEntryModel.journal_id == record.initial_journal_id).first()
    assert j_init.is_balanced is True
    assert j_init.total_debit == 3000.0
    assert record.balance_sheet_account_code == "440-1000"

    # October Amortization: Dr 440-1000 RM 1000, Cr 500-0000 RM 1000
    service.post_amortization_adjustment(record.id, AccrualAdjustmentRequest(
        adjustment_date="2026-10-31",
        amount=1000.0,
        period_name="October 2026"
    ))
    assert record.remaining_balance == 2000.0
    assert record.status == "PARTIALLY_ADJUSTED"

    # November & December Amortization (RM 2,000)
    service.post_amortization_adjustment(record.id, AccrualAdjustmentRequest(
        adjustment_date="2026-12-31",
        amount=2000.0,
        period_name="Nov-Dec 2026"
    ))
    assert record.remaining_balance == 0.0
    assert record.status == "FULLY_ADJUSTED"


def test_deferred_income_refund(db_session):
    service = AccrualPrepaymentService(db_session)
    record = service.create_record(AccrualPrepaymentCreate(
        transaction_type="DEFERRED_INCOME",
        title="Website Hosting Advance",
        original_amount=1200.0,
        transaction_date="2026-01-01",
        start_date="2026-01-01",
        end_date="2026-12-31",
        target_account_code="500-0000"
    ))

    # Amortize RM 100 for January
    service.post_amortization_adjustment(record.id, AccrualAdjustmentRequest(
        adjustment_date="2026-01-31",
        amount=100.0
    ))
    assert record.remaining_balance == 1100.0

    # Customer cancels and unearned RM 1,100 is refunded: Dr 440-1000, Cr 310-1000
    service.post_cancellation_and_refund(record.id, AccrualCancellationRequest(
        cancellation_date="2026-02-01",
        refund_amount=1100.0
    ))
    assert record.remaining_balance == 0.0
    assert record.status == "CANCELLED"


# =========================================================================
# 6. CALENDAR, LEAP YEAR & DATE EDGE CASES
# =========================================================================

def test_leap_year_february_schedule(db_session):
    service = AccrualPrepaymentService(db_session)
    # 2024 is a leap year (Feb has 29 days!)
    # Jan 15 to Mar 15, 2024
    schedule = service.calculate_amortization_schedule(900.0, 1.0, "2024-01-15", "2024-03-15")
    feb_period = next(s for s in schedule if "February" in s.period_name)
    assert feb_period.days_in_period == 29
    tot = sum(s.allocated_amount for s in schedule)
    assert abs(tot - 900.0) < 0.01


def test_crossing_financial_years(db_session):
    service = AccrualPrepaymentService(db_session)
    # Oct 1, 2026 to Mar 31, 2027 (Crosses fiscal year end Dec 31)
    schedule = service.calculate_amortization_schedule(6000.0, 1.0, "2026-10-01", "2027-03-31")
    assert len(schedule) == 6
    months = [s.period_name for s in schedule]
    assert "December 2026" in months
    assert "January 2027" in months
    assert abs(sum(s.allocated_amount for s in schedule) - 6000.0) < 0.01


# =========================================================================
# 7. MULTICURRENCY SUPPORT
# =========================================================================

def test_multicurrency_foreign_prepayment(db_session):
    service = AccrualPrepaymentService(db_session)
    # SGD 1,000 with exchange rate 3.1967 -> MYR 3,196.70
    record = service.create_record(AccrualPrepaymentCreate(
        transaction_type="PREPAID_EXPENSE",
        title="Singapore Cloud Hosting",
        original_amount=1000.0,
        currency="SGD",
        exchange_rate=3.1967,
        transaction_date="2026-10-22",
        start_date="2026-10-22",
        end_date="2027-10-21",
        target_account_code="915-0000"
    ))
    assert record.original_amount == 1000.0
    assert record.currency == "SGD"
    assert record.amount_myr == 3196.70
    assert record.remaining_balance == 1000.0

    j_init = db_session.query(JournalEntryModel).filter(JournalEntryModel.journal_id == record.initial_journal_id).first()
    assert j_init.total_debit == 3196.70
    assert j_init.total_credit == 3196.70


# =========================================================================
# 8. NON-CURRENT VS CURRENT CLASSIFICATION (> 12 MONTHS)
# =========================================================================

def test_long_term_prepayment_non_current_classification(db_session):
    service = AccrualPrepaymentService(db_session)
    # 2-Year Contract: Jan 1, 2026 to Dec 31, 2027 (730 days > 366 days)
    record = service.create_record(AccrualPrepaymentCreate(
        transaction_type="PREPAID_EXPENSE",
        title="2-Year Enterprise License",
        original_amount=24000.0,
        transaction_date="2026-01-01",
        start_date="2026-01-01",
        end_date="2027-12-31",
        target_account_code="915-0000"
    ))
    # Automatically classified as Non-Current Prepayment: 220-0000
    assert record.balance_sheet_account_code == "220-0000"


# =========================================================================
# 9. GENERAL LEDGER, TRIAL BALANCE, & FINANCIAL STATEMENTS INTEGRATION
# =========================================================================

def test_trial_balance_and_financial_statements_integration(client, db_session):
    service = AccrualPrepaymentService(db_session)

    # 1. Create Accrued Expense: RM 1,500
    service.create_record(AccrualPrepaymentCreate(
        transaction_type="ACCRUED_EXPENSE",
        title="Electricity Accrual",
        original_amount=1500.0,
        transaction_date="2026-10-31",
        start_date="2026-10-01",
        end_date="2026-10-31",
        target_account_code="907-0000"
    ))

    # 2. Check Trial Balance via API: Must be perfectly balanced
    res_gl = client.get("/api/ledger/")
    assert res_gl.status_code == 200
    gl_data = res_gl.json()
    assert gl_data["trial_balance"]["is_balanced"] is True
    assert abs(gl_data["trial_balance"]["total_debit"] - gl_data["trial_balance"]["total_credit"]) < 0.01

    # 3. Check Profit & Loss via API: Accrual must increase expense
    res_pl = client.get("/api/ledger/profit_loss")
    assert res_pl.status_code == 200
    pl_data = res_pl.json()
    op_exp = pl_data["mfrs101"]["operating_expenses_by_function"]["administrative_and_other"]
    elec_item = next((i for i in op_exp if i["account_code"] == "907-0000"), None)
    assert elec_item is not None
    assert elec_item["balance"] == 1500.0

    # 4. Check Balance Sheet via API: Accrual must appear in payables
    res_bs = client.get("/api/ledger/balance_sheet")
    assert res_bs.status_code == 200
    bs_data = res_bs.json()
    payables = bs_data["mfrs101"]["current_liabilities"]["trade_and_other_payables"]["items"]
    accrual_item = next((i for i in payables if i["account_code"] == "410-0000"), None)
    assert accrual_item is not None
    assert accrual_item["balance"] == 1500.0


# =========================================================================
# 10. FASTAPI ENDPOINTS WORKFLOW
# =========================================================================

def test_api_full_accruals_workflow(client, db_session):
    # 1. Create record via API
    res_create = client.post("/api/accruals-prepayments/", json={
        "transaction_type": "PREPAID_EXPENSE",
        "title": "Office High-Speed Internet",
        "original_amount": 1200.0,
        "transaction_date": "2026-01-01",
        "start_date": "2026-01-01",
        "end_date": "2026-12-31",
        "target_account_code": "909-0000",  # TELEPHONE CHARGES
        "bank_account_code": "310-1000",
        "post_initial_journal": True
    })
    assert res_create.status_code == 200
    rec_data = res_create.json()
    rec_id = rec_data["id"]

    # 2. Schedule preview via API
    res_sched = client.get(f"/api/accruals-prepayments/{rec_id}/schedule-preview")
    assert res_sched.status_code == 200
    assert len(res_sched.json()) == 12

    # 3. Post amortization via API
    res_adj = client.post(f"/api/accruals-prepayments/{rec_id}/adjust", json={
        "adjustment_date": "2026-01-31",
        "amount": 100.0,
        "period_name": "January 2026"
    })
    assert res_adj.status_code == 200

    # 4. Check summary endpoint
    res_sum = client.get("/api/accruals-prepayments/summary")
    assert res_sum.status_code == 200
    sum_data = res_sum.json()
    assert sum_data["total_prepaid_expenses"] == 1100.0
