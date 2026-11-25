import os
import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_mfrs101_profit_and_loss_endpoint():
    res = client.get("/api/ledger/profit_loss")
    assert res.status_code == 200
    data = res.json()

    # Legacy backward-compatible top-level keys
    for key in ["revenue", "total_revenue", "cogs", "total_cogs", "gross_profit", "expenses", "total_expenses", "net_profit"]:
        assert key in data

    # MFRS 101 structured section
    assert "mfrs101" in data
    mfrs = data["mfrs101"]
    assert "MFRS 101" in mfrs["standard"]
    for key in [
        "revenue_section",
        "cost_of_sales_section",
        "gross_profit",
        "other_operating_income_section",
        "operating_expenses_by_function",
        "operating_expenses_by_nature",
        "operating_profit",
        "finance_costs_section",
        "profit_before_tax",
        "taxation_section",
        "profit_for_period",
    ]:
        assert key in mfrs

    # Verify Function vs Nature operating expenses reconcile to the same total
    assert round(mfrs["operating_expenses_by_function"]["total_operating_expenses"], 2) == round(
        mfrs["operating_expenses_by_nature"]["total_operating_expenses"], 2
    )


def test_mfrs101_balance_sheet_endpoint():
    res = client.get("/api/ledger/balance_sheet")
    assert res.status_code == 200
    data = res.json()

    # Legacy backward-compatible top-level keys
    for key in ["assets", "total_assets", "liabilities", "total_liabilities", "equity", "total_equity", "is_balanced"]:
        assert key in data

    # MFRS 101 structured section
    assert "mfrs101" in data
    mfrs = data["mfrs101"]
    assert "MFRS 101" in mfrs["standard"]
    for key in [
        "non_current_assets",
        "current_assets",
        "total_assets",
        "equity_section",
        "non_current_liabilities",
        "current_liabilities",
        "total_liabilities",
        "total_equity_and_liabilities",
        "is_balanced",
    ]:
        assert key in mfrs

    assert mfrs["is_balanced"] is True
    assert round(mfrs["total_assets"], 2) == round(mfrs["total_equity_and_liabilities"], 2)


def test_mfrs101_unified_reports_package():
    res = client.get("/api/ledger/mfrs_reports")
    assert res.status_code == 200
    data = res.json()

    for key in [
        "company",
        "reporting_metadata",
        "statement_of_profit_or_loss",
        "statement_of_financial_position",
        "trial_balance",
        "financial_ratios",
    ]:
        assert key in data

    assert data["reporting_metadata"]["standard"] == "MFRS 101"
    assert data["reporting_metadata"]["currency"] in ["MYR", "RM"]
    assert data["statement_of_financial_position"]["mfrs101"]["is_balanced"] is True
    assert data["trial_balance"]["is_balanced"] is True
