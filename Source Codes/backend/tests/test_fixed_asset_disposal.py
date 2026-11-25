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
from app.db.database import Base, get_db, seed_default_chart_of_accounts, DEFAULT_CHART_OF_ACCOUNTS
from app.db.models import ChartOfAccountModel, FixedAssetModel, JournalEntryModel
from app.services.fixed_asset_service import FixedAssetService, _to_decimal, _round_float
from app.schemas.fixed_asset import FixedAssetCreate, FixedAssetDisposalRequest
from app.api.ledger import _determine_normal_balance


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
# 1. CHART OF ACCOUNTS & NORMAL BALANCE VERIFICATION
# =========================================================================

def test_chart_of_accounts_disposal_accounts_exist(db_session):
    """
    Verify all required disposal accounts are seeded, correctly classified,
    and associated with proper group headers.
    """
    gain_acc = db_session.query(ChartOfAccountModel).filter(
        ChartOfAccountModel.account_code == "550-0000",
        ChartOfAccountModel.company_id == "COMP-000001"
    ).first()
    assert gain_acc is not None
    assert gain_acc.account_name == "GAIN ON DISPOSAL OF FIXED ASSETS"
    assert gain_acc.account_type == "Revenue"
    assert gain_acc.is_active is True

    loss_acc = db_session.query(ChartOfAccountModel).filter(
        ChartOfAccountModel.account_code == "916-0000",
        ChartOfAccountModel.company_id == "COMP-000001"
    ).first()
    assert loss_acc is not None
    assert loss_acc.account_name == "LOSS ON DISPOSAL OF FIXED ASSETS"
    assert loss_acc.account_type == "Expense"
    assert loss_acc.is_active is True

    disp_exp_acc = db_session.query(ChartOfAccountModel).filter(
        ChartOfAccountModel.account_code == "917-0000",
        ChartOfAccountModel.company_id == "COMP-000001"
    ).first()
    assert disp_exp_acc is not None
    assert disp_exp_acc.account_name == "FIXED ASSET DISPOSAL EXPENSES"
    assert disp_exp_acc.account_type == "Expense"
    assert disp_exp_acc.is_active is True


def test_normal_balance_directions(db_session):
    """
    Verify double-entry normal balance rules:
    - Fixed Assets (200-2000, 200-3000, 200-4000): DEBIT
    - Contra-Asset Accum. Depr (200-2005, 200-3005, 200-4005): CREDIT
    - Cash at Bank (310-1000): DEBIT
    - Gain on Disposal (550-0000): CREDIT
    - Loss on Disposal (916-0000): DEBIT
    - Disposal Expenses (917-0000): DEBIT
    """
    assert _determine_normal_balance("200-2000", "Asset") == "DEBIT"
    assert _determine_normal_balance("200-3000", "Asset") == "DEBIT"
    assert _determine_normal_balance("200-4000", "Asset") == "DEBIT"

    assert _determine_normal_balance("200-2005", "Asset") == "CREDIT"
    assert _determine_normal_balance("200-3005", "Asset") == "CREDIT"
    assert _determine_normal_balance("200-4005", "Asset") == "CREDIT"

    assert _determine_normal_balance("310-1000", "Asset") == "DEBIT"
    assert _determine_normal_balance("550-0000", "Revenue") == "CREDIT"
    assert _determine_normal_balance("916-0000", "Expense") == "DEBIT"
    assert _determine_normal_balance("917-0000", "Expense") == "DEBIT"


# =========================================================================
# 2. FIXED ASSET CREATION & VALIDATION
# =========================================================================

def test_create_fixed_asset_success(db_session):
    service = FixedAssetService(db_session)
    asset_req = FixedAssetCreate(
        company_id="COMP-000001",
        asset_code="FA-2026-001",
        asset_name="Dell Precision Workstation",
        asset_category="OFFICE_EQUIPMENT",
        cost_account_code="200-3000",
        accum_depr_account_code="200-3005",
        acquisition_date="2024-01-01",
        purchase_cost=10000.0,
        salvage_value=1000.0,
        useful_life_years=5.0,
        accumulated_depreciation=3600.0
    )
    asset = service.create_fixed_asset(asset_req)
    assert asset.id is not None
    assert asset.asset_code == "FA-2026-001"
    assert asset.current_carrying_amount == 6400.0  # 10,000 - 3,600
    assert asset.status == "ACTIVE"


def test_create_fixed_asset_invalid_cost_and_salvage(db_session):
    service = FixedAssetService(db_session)
    # Purchase cost <= 0
    with pytest.raises(ValueError, match="Purchase cost must be greater than zero"):
        service.create_fixed_asset(FixedAssetCreate(
            asset_code="FA-INV-1",
            asset_name="Bad Cost Asset",
            asset_category="OFFICE_EQUIPMENT",
            cost_account_code="200-3000",
            accum_depr_account_code="200-3005",
            acquisition_date="2024-01-01",
            purchase_cost=0.0
        ))

    # Salvage >= Cost
    with pytest.raises(ValueError, match="Salvage value must be strictly less than purchase cost"):
        service.create_fixed_asset(FixedAssetCreate(
            asset_code="FA-INV-2",
            asset_name="Bad Salvage Asset",
            asset_category="OFFICE_EQUIPMENT",
            cost_account_code="200-3000",
            accum_depr_account_code="200-3005",
            acquisition_date="2024-01-01",
            purchase_cost=1000.0,
            salvage_value=1000.0
        ))


def test_create_fixed_asset_invalid_account_codes(db_session):
    service = FixedAssetService(db_session)
    with pytest.raises(ValueError, match="Cost account '999-9999' does not exist"):
        service.create_fixed_asset(FixedAssetCreate(
            asset_code="FA-INV-3",
            asset_name="Invalid COA Asset",
            asset_category="OFFICE_EQUIPMENT",
            cost_account_code="999-9999",
            accum_depr_account_code="200-3005",
            acquisition_date="2024-01-01",
            purchase_cost=5000.0
        ))


# =========================================================================
# 3. DEPRECIATION CALCULATION TO DISPOSAL DATE
# =========================================================================

def test_depreciation_calculation_to_date(db_session):
    service = FixedAssetService(db_session)
    asset = service.create_fixed_asset(FixedAssetCreate(
        asset_code="FA-DEP-001",
        asset_name="Office Server",
        asset_category="OFFICE_EQUIPMENT",
        cost_account_code="200-3000",
        accum_depr_account_code="200-3005",
        acquisition_date="2023-01-01",
        purchase_cost=7300.0,
        salvage_value=0.0,
        useful_life_years=2.0,  # 7300 / 2 = 3650/year = 10/day
        accumulated_depreciation=0.0
    ))

    # 365 days later (2024-01-01) -> exactly 3650.0
    depr_1yr = service.calculate_depreciation_to_date(asset, "2024-01-01")
    assert abs(depr_1yr - 3650.0) < 1.0


# =========================================================================
# 4. DISPOSAL SCENARIOS & BALANCED JOURNAL ENTRIES
# =========================================================================

def test_disposal_scenario_gain_on_disposal(db_session):
    """
    Scenario 1: Asset sold for more than carrying amount (NBV).
    Cost: RM 10,000. Accum Depr: RM 6,000 -> NBV: RM 4,000.
    Disposal Proceeds: RM 5,000. Disposal Expenses: RM 200. Net Proceeds: RM 4,800.
    Derecognition Gain before expense: RM 1,000 (Credit 550-0000).
    Incidental Disposal Expense: RM 200 (Debit 917-0000).
    Net cash received: RM 4,800 (Debit 310-1000).
    Accum Depr removed: RM 6,000 (Debit 200-3005).
    Asset Cost removed: RM 10,000 (Credit 200-3000).

    Debits: 4,800 + 200 + 6,000 = RM 11,000.00
    Credits: 10,000 + 1,000 = RM 11,000.00
    Net P&L effect: Gain 1,000 - Expense 200 = Net Gain RM 800.00.
    """
    service = FixedAssetService(db_session)
    asset = service.create_fixed_asset(FixedAssetCreate(
        asset_code="FA-GAIN-001",
        asset_name="Company Van",
        asset_category="MOTOR_VEHICLE",
        cost_account_code="200-4000",
        accum_depr_account_code="200-4005",
        acquisition_date="2022-01-01",
        purchase_cost=10000.0,
        accumulated_depreciation=6000.0
    ))

    disp_req = FixedAssetDisposalRequest(
        disposal_date="2026-10-25",
        disposal_proceeds=5000.0,
        disposal_expenses=200.0,
        proceeds_account_code="310-1000",
        additional_depreciation=0.0
    )

    preview = service.preview_disposal(asset.id, disp_req)
    assert preview.is_balanced is True
    assert preview.total_debit == 11000.0
    assert preview.total_credit == 11000.0
    assert preview.carrying_amount_nbv == 4000.0
    assert preview.gain_loss_type == "GAIN"
    assert preview.gain_loss_amount == 800.0  # Net gain (Net proceeds 4,800 - NBV 4,000)

    # Execute disposal
    disposed_asset, journal = service.execute_disposal(asset.id, disp_req)
    assert disposed_asset.status == "DISPOSED"
    assert disposed_asset.current_carrying_amount == 0.0
    assert journal.is_balanced is True
    assert journal.status == "POSTED"
    assert journal.total_debit == 11000.0
    assert journal.total_credit == 11000.0


def test_disposal_scenario_loss_on_disposal(db_session):
    """
    Scenario 2: Asset sold below carrying amount (NBV).
    Cost: RM 8,000. Accum Depr: RM 4,000 -> NBV: RM 4,000.
    Disposal Proceeds: RM 2,500. Disposal Expenses: RM 100. Net Proceeds: RM 2,400.
    Derecognition Loss: RM 1,500 (Debit 916-0000).
    Incidental Disposal Expense: RM 100 (Debit 917-0000).
    Net cash received: RM 2,400 (Debit 310-1000).
    Accum Depr removed: RM 4,000 (Debit 200-3005).
    Asset Cost removed: RM 8,000 (Credit 200-3000).

    Debits: 2,400 + 100 + 4,000 + 1,500 = RM 8,000.00
    Credits: 8,000.00 = RM 8,000.00
    Net P&L effect: Loss 1,500 + Expense 100 = Total Loss RM 1,600.00.
    """
    service = FixedAssetService(db_session)
    asset = service.create_fixed_asset(FixedAssetCreate(
        asset_code="FA-LOSS-001",
        asset_name="Office Conference Table",
        asset_category="FURNITURE_FITTINGS",
        cost_account_code="200-2000",
        accum_depr_account_code="200-2005",
        acquisition_date="2023-01-01",
        purchase_cost=8000.0,
        accumulated_depreciation=4000.0
    ))

    disp_req = FixedAssetDisposalRequest(
        disposal_date="2026-10-25",
        disposal_proceeds=2500.0,
        disposal_expenses=100.0,
        proceeds_account_code="310-1000"
    )

    preview = service.preview_disposal(asset.id, disp_req)
    assert preview.is_balanced is True
    assert preview.total_debit == 8000.0
    assert preview.total_credit == 8000.0
    assert preview.carrying_amount_nbv == 4000.0
    assert preview.gain_loss_type == "LOSS"
    assert preview.gain_loss_amount == -1600.0

    disposed_asset, journal = service.execute_disposal(asset.id, disp_req)
    assert disposed_asset.status == "DISPOSED"
    assert journal.is_balanced is True
    assert journal.total_debit == 8000.0
    assert journal.total_credit == 8000.0


def test_disposal_scenario_scrap_full_write_off(db_session):
    """
    Scenario 3: Asset scrapped or written off for zero proceeds.
    Cost: RM 5,000. Accum Depr: RM 3,500 -> NBV: RM 1,500.
    Proceeds: RM 0. Expenses: RM 0.
    Derecognition Loss: RM 1,500 (Debit 916-0000).
    Accum Depr removed: RM 3,500 (Debit 200-3005).
    Asset Cost removed: RM 5,000 (Credit 200-3000).

    Debits: 3,500 + 1,500 = RM 5,000.00
    Credits: RM 5,000.00
    """
    service = FixedAssetService(db_session)
    asset = service.create_fixed_asset(FixedAssetCreate(
        asset_code="FA-SCRAP-001",
        asset_name="Damaged Laptop",
        asset_category="OFFICE_EQUIPMENT",
        cost_account_code="200-3000",
        accum_depr_account_code="200-3005",
        acquisition_date="2023-01-01",
        purchase_cost=5000.0,
        accumulated_depreciation=3500.0
    ))

    disp_req = FixedAssetDisposalRequest(
        disposal_date="2026-10-25",
        disposal_proceeds=0.0,
        disposal_expenses=0.0,
        proceeds_account_code="310-1000"
    )

    preview = service.preview_disposal(asset.id, disp_req)
    assert preview.is_balanced is True
    assert preview.total_debit == 5000.0
    assert preview.total_credit == 5000.0
    assert preview.gain_loss_type == "LOSS"
    assert preview.gain_loss_amount == -1500.0

    disposed_asset, journal = service.execute_disposal(asset.id, disp_req)
    assert disposed_asset.status == "DISPOSED"
    assert journal.is_balanced is True
    assert journal.total_debit == 5000.0
    assert journal.total_credit == 5000.0


def test_prevent_duplicate_disposal(db_session):
    """
    Attempting to dispose an already disposed asset must be strictly prevented.
    """
    service = FixedAssetService(db_session)
    asset = service.create_fixed_asset(FixedAssetCreate(
        asset_code="FA-DUP-001",
        asset_name="Office Chair",
        asset_category="FURNITURE_FITTINGS",
        cost_account_code="200-2000",
        accum_depr_account_code="200-2005",
        acquisition_date="2023-01-01",
        purchase_cost=1000.0,
        accumulated_depreciation=500.0
    ))

    disp_req = FixedAssetDisposalRequest(
        disposal_date="2026-10-25",
        disposal_proceeds=600.0
    )

    # First disposal succeeds
    service.execute_disposal(asset.id, disp_req)

    # Second disposal must raise ValueError
    with pytest.raises(ValueError, match="already DISPOSED"):
        service.preview_disposal(asset.id, disp_req)

    with pytest.raises(ValueError, match="already DISPOSED"):
        service.execute_disposal(asset.id, disp_req)


# =========================================================================
# 5. FINANCIAL STATEMENT PRESENTATION (MFRS 101)
# =========================================================================

def test_financial_statement_integration_gain(client, db_session):
    """
    Verify that upon disposal with Gain:
    - P&L: Gain on Disposal appears under other_operating_income_section (not Sales).
    - P&L: Disposal expenses appear under operating_expenses.
    - Balance Sheet: Net Book Value of disposed asset is removed.
    """
    import json

    # 1. Post initial acquisition & depreciation journals on the General Ledger
    acq_lines = [
        {"line_id": 1, "account_code": "200-2000", "account_name": "FURNITURES & FITTINGS", "account_type": "Asset", "debit": 4000.0, "credit": 0.0, "description": "Acquisition"},
        {"line_id": 2, "account_code": "310-1000", "account_name": "CASH AT BANK", "account_type": "Asset", "debit": 0.0, "credit": 4000.0, "description": "Bank payment"},
        {"line_id": 3, "account_code": "903-0000", "account_name": "DEPRECIATION OF FIXED ASSETS", "account_type": "Expense", "debit": 2500.0, "credit": 0.0, "description": "Depreciation"},
        {"line_id": 4, "account_code": "200-2005", "account_name": "ACCUM. DEPRN. - FURNITURES & FITTINGS", "account_type": "Asset", "debit": 0.0, "credit": 2500.0, "description": "Accum Depr"}
    ]
    acq_journal = JournalEntryModel(
        journal_id="JRN-ACQ-TEST-001",
        transaction_id="TXN-ACQ-001",
        company_id="COMP-000001",
        document_id="DOC-ACQ-001",
        journal_date="2022-01-01",
        reference="Asset Acquisition",
        description="Initial acquisition & depr",
        total_debit=6500.0,
        total_credit=6500.0,
        is_balanced=True,
        lines_json=json.dumps(acq_lines),
        status="POSTED"
    )
    db_session.add(acq_journal)
    db_session.commit()

    service = FixedAssetService(db_session)
    asset = service.create_fixed_asset(FixedAssetCreate(
        asset_code="FA-MFRS-GAIN",
        asset_name="Executive Desk",
        asset_category="FURNITURE_FITTINGS",
        cost_account_code="200-2000",
        accum_depr_account_code="200-2005",
        acquisition_date="2022-01-01",
        purchase_cost=4000.0,
        accumulated_depreciation=2500.0
    ))

    disp_req = FixedAssetDisposalRequest(
        disposal_date="2026-10-25",
        disposal_proceeds=2000.0,
        disposal_expenses=100.0,
        proceeds_account_code="310-1000"
    )
    service.execute_disposal(asset.id, disp_req)

    # Call P&L API
    res_pl = client.get("/api/ledger/profit_loss")
    assert res_pl.status_code == 200
    pl_data = res_pl.json()

    # Gain on disposal must appear under other_operating_income_section
    other_inc = pl_data["mfrs101"]["other_operating_income_section"]["items"]
    gain_codes = [i["account_code"] for i in other_inc]
    assert "550-0000" in gain_codes

    # Disposal expenses must appear under operating expenses
    op_exp = pl_data["mfrs101"]["operating_expenses_by_function"]["administrative_and_other"]
    exp_codes = [i["account_code"] for i in op_exp]
    assert "917-0000" in exp_codes

    # Call Balance Sheet API
    res_bs = client.get("/api/ledger/balance_sheet")
    assert res_bs.status_code == 200
    bs_data = res_bs.json()

    # Total fixed asset cost & accum depr for 200-2000 and 200-2005 must be 0
    ppe_cost = bs_data["mfrs101"]["non_current_assets"]["property_plant_equipment"]["cost_items"]
    desk_cost = next((i for i in ppe_cost if i["account_code"] == "200-2000"), None)
    if desk_cost:
        assert desk_cost["balance"] == 0.0

    ppe_accum = bs_data["mfrs101"]["non_current_assets"]["property_plant_equipment"]["accum_depr_items"]
    desk_accum = next((i for i in ppe_accum if i["account_code"] == "200-2005"), None)
    if desk_accum:
        assert desk_accum["balance"] == 0.0


def test_financial_statement_integration_loss(client, db_session):
    """
    Verify that upon disposal with Loss:
    - P&L: Loss on Disposal appears under operating expenses (never in COGS).
    """
    service = FixedAssetService(db_session)
    asset = service.create_fixed_asset(FixedAssetCreate(
        asset_code="FA-MFRS-LOSS",
        asset_name="Old Printer",
        asset_category="OFFICE_EQUIPMENT",
        cost_account_code="200-3000",
        accum_depr_account_code="200-3005",
        acquisition_date="2023-01-01",
        purchase_cost=3000.0,
        accumulated_depreciation=1000.0
    ))

    disp_req = FixedAssetDisposalRequest(
        disposal_date="2026-10-25",
        disposal_proceeds=1200.0,
        disposal_expenses=0.0,
        proceeds_account_code="310-1000"
    )
    service.execute_disposal(asset.id, disp_req)

    res_pl = client.get("/api/ledger/profit_loss")
    assert res_pl.status_code == 200
    pl_data = res_pl.json()

    # Loss on disposal must appear under operating expenses
    op_exp = pl_data["mfrs101"]["operating_expenses_by_function"]["administrative_and_other"]
    loss_item = next((i for i in op_exp if i["account_code"] == "916-0000"), None)
    assert loss_item is not None
    assert loss_item["balance"] == 800.0  # NBV 2000 - Proceeds 1200


# =========================================================================
# 6. FASTAPI ENDPOINT INTEGRATION TESTS
# =========================================================================

def test_api_full_disposal_workflow(client, db_session):
    # 1. Create asset via API
    res_create = client.post("/api/fixed-assets/", json={
        "asset_code": "FA-API-001",
        "asset_name": "Delivery Scooter",
        "asset_category": "MOTOR_VEHICLE",
        "cost_account_code": "200-4000",
        "accum_depr_account_code": "200-4005",
        "acquisition_date": "2024-01-01",
        "purchase_cost": 6000.0,
        "salvage_value": 500.0,
        "useful_life_years": 5.0,
        "accumulated_depreciation": 2200.0
    })
    assert res_create.status_code == 200
    asset_data = res_create.json()
    asset_id = asset_data["id"]

    # 2. Preview disposal via API
    res_preview = client.post(f"/api/fixed-assets/{asset_id}/disposal-preview", json={
        "disposal_date": "2026-10-25",
        "disposal_proceeds": 4200.0,
        "disposal_expenses": 150.0,
        "proceeds_account_code": "310-1000",
        "additional_depreciation": 0.0
    })
    assert res_preview.status_code == 200
    preview_data = res_preview.json()
    assert preview_data["is_balanced"] is True
    assert preview_data["gain_loss_type"] == "GAIN"

    # 3. Execute disposal via API
    res_disp = client.post(f"/api/fixed-assets/{asset_id}/dispose", json={
        "disposal_date": "2026-10-25",
        "disposal_proceeds": 4200.0,
        "disposal_expenses": 150.0,
        "proceeds_account_code": "310-1000",
        "additional_depreciation": 0.0
    })
    assert res_disp.status_code == 200
    disp_result = res_disp.json()
    assert disp_result["is_balanced"] is True
    assert disp_result["status"] == "POSTED"

    # 4. Attempt duplicate disposal via API -> 400 Bad Request
    res_dup = client.post(f"/api/fixed-assets/{asset_id}/dispose", json={
        "disposal_date": "2026-10-25",
        "disposal_proceeds": 4200.0
    })
    assert res_dup.status_code == 400
    assert "already DISPOSED" in res_dup.json()["detail"] or "Asset cannot be disposed" in res_dup.json()["detail"]


def test_disposal_scenario_gain_with_unpaid_expenses(db_session):
    """
    Scenario: Asset sold at a gain with unpaid disposal expenses.
    - Gross Proceeds: RM 5,000
    - Incidental Expenses: RM 200 (Unpaid -> Credited to 405-0000 OTHER CREDITORS)
    - Net Cash Received: RM 5,000 (Not deducted because expenses are unpaid!)
    - Gain on Derecognition: Gross Proceeds (5000) - NBV (4000) = RM 1,000 (550-0000)
    - Disposal Expenses: RM 200 (917-0000)
    
    Debits:
    - Dr Cash at Bank (310-1000): RM 5,000.00
    - Dr Accum Depr (200-4005): RM 6,000.00
    - Dr Disposal Expenses (917-0000): RM 200.00
    Total Debits = RM 11,200.00

    Credits:
    - Cr Asset Cost (200-4000): RM 10,000.00
    - Cr Gain on Disposal (550-0000): RM 1,000.00
    - Cr Other Creditors (405-0000): RM 200.00
    Total Credits = RM 11,200.00
    """
    service = FixedAssetService(db_session)
    asset = service.create_fixed_asset(FixedAssetCreate(
        asset_code="FA-UNPAID-001",
        asset_name="Delivery Truck",
        asset_category="MOTOR_VEHICLE",
        cost_account_code="200-4000",
        accum_depr_account_code="200-4005",
        acquisition_date="2022-01-01",
        purchase_cost=10000.0,
        accumulated_depreciation=6000.0
    ))

    disp_req = FixedAssetDisposalRequest(
        disposal_date="2026-10-25",
        disposal_proceeds=5000.0,
        disposal_expenses=200.0,
        expenses_paid=False,  # UNPAID EXPENSES
        expense_liability_account_code="405-0000",
        proceeds_account_code="310-1000"
    )

    preview = service.preview_disposal(asset.id, disp_req)
    assert preview.is_balanced is True
    assert preview.expenses_paid is False
    assert preview.net_cash_received == 5000.0  # Not reduced!
    assert preview.total_debit == 11200.0
    assert preview.total_credit == 11200.0

    disposed_asset, journal = service.execute_disposal(asset.id, disp_req)
    assert disposed_asset.status == "DISPOSED"
    assert disposed_asset.expenses_paid is False
    assert disposed_asset.expense_liability_account_code == "405-0000"
    assert journal.is_balanced is True
    assert journal.total_debit == 11200.0
    assert journal.total_credit == 11200.0


def test_disposal_scenario_no_gain_no_loss(db_session):
    """
    Scenario: Asset sold exactly at carrying amount (Proceeds == NBV, Zero Expenses).
    Cost: RM 6,000. Accum Depr: RM 4,000 -> NBV: RM 2,000.
    Proceeds: RM 2,000. Expenses: 0.
    Gain / Loss = 0.00.
    Debits: 2,000 + 4,000 = RM 6,000.00
    Credits: RM 6,000.00
    """
    service = FixedAssetService(db_session)
    asset = service.create_fixed_asset(FixedAssetCreate(
        asset_code="FA-NOGAIN-001",
        asset_name="Office Server",
        asset_category="OFFICE_EQUIPMENT",
        cost_account_code="200-3000",
        accum_depr_account_code="200-3005",
        acquisition_date="2023-01-01",
        purchase_cost=6000.0,
        accumulated_depreciation=4000.0
    ))

    disp_req = FixedAssetDisposalRequest(
        disposal_date="2026-10-25",
        disposal_proceeds=2000.0,
        disposal_expenses=0.0,
        proceeds_account_code="310-1000"
    )

    preview = service.preview_disposal(asset.id, disp_req)
    assert preview.is_balanced is True
    assert preview.gain_loss_type == "ZERO"
    assert preview.total_debit == 6000.0
    assert preview.total_credit == 6000.0

    disposed_asset, journal = service.execute_disposal(asset.id, disp_req)
    assert disposed_asset.status == "DISPOSED"
    assert journal.is_balanced is True


def test_trial_balance_exact_balancing(client, db_session):
    """
    Verify that after disposal transactions, the Trial Balance is perfectly balanced.
    """
    service = FixedAssetService(db_session)
    asset = service.create_fixed_asset(FixedAssetCreate(
        asset_code="FA-TB-001",
        asset_name="Office Furniture",
        asset_category="FURNITURE_FITTINGS",
        cost_account_code="200-2000",
        accum_depr_account_code="200-2005",
        acquisition_date="2023-01-01",
        purchase_cost=5000.0,
        accumulated_depreciation=2000.0
    ))

    service.execute_disposal(asset.id, FixedAssetDisposalRequest(
        disposal_date="2026-10-25",
        disposal_proceeds=3500.0,
        disposal_expenses=150.0,
        proceeds_account_code="310-1000"
    ))

    res_gl = client.get("/api/ledger/")
    assert res_gl.status_code == 200
    gl_data = res_gl.json()

    # Total debits must equal total credits in trial balance
    assert gl_data["trial_balance"]["is_balanced"] is True
    assert abs(gl_data["trial_balance"]["total_debit"] - gl_data["trial_balance"]["total_credit"]) < 0.01
