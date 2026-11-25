import os
import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

import json
import pytest
from decimal import Decimal
from fastapi.testclient import TestClient

from app.main import app
from app.db.database import get_db, SessionLocal
from app.db.models import (
    UserModel,
    CompanyModel,
    OpeningBalanceBatchModel,
    OpeningBalanceDraftRecordModel,
    CustomerModel,
    SupplierModel,
    InventoryItemModel,
    JournalEntryModel,
    OpeningBalanceDocumentModel
)
from app.services.onboarding_service import OnboardingService

client = TestClient(app)


def test_user_registration_and_login():
    email = f"test_user_{os.urandom(4).hex()}@handbag.com"
    reg_resp = client.post("/api/auth/register", json={
        "email": email,
        "password": "Password123!",
        "full_name": "Wani Account Manager",
        "role": "OWNER"
    })
    assert reg_resp.status_code == 200, reg_resp.text
    reg_data = reg_resp.json()
    assert "access_token" in reg_data
    assert reg_data["user"]["email"] == email
    assert reg_data["user"]["full_name"] == "Wani Account Manager"

    # Test login
    login_resp = client.post("/api/auth/login", json={
        "email": email,
        "password": "Password123!"
    })
    assert login_resp.status_code == 200
    token = login_resp.json()["access_token"]

    # Test /api/auth/me
    me_resp = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_resp.status_code == 200
    assert me_resp.json()["email"] == email


def test_pdf_text_parsers_direct():
    """Test the parser logic directly on the exact raw text formats from the scenario."""
    db = SessionLocal()
    try:
        service = OnboardingService(db)
        comp = service.create_company_onboarding(
            type('Req', (), {
                'company_name': 'Parser Test Handbag Sdn Bhd',
                'registration_number': None,
                'address': None,
                'email': None,
                'phone': None,
                'currency': 'MYR',
                'financial_year': 2026,
                'financial_year_start': '2026-01-01',
                'financial_year_end': '2026-12-31'
            })()
        )
        batch = service.get_or_create_batch(comp.company_id)

        doc = OpeningBalanceDocumentModel(
            company_id=comp.company_id,
            doc_category="TRIAL_BALANCE",
            original_filename="Opening_TB_2025.pdf",
            file_path="dummy.pdf"
        )
        db.add(doc)
        db.commit()

        # 1. Trial Balance Text
        tb_sample_text = """
        XXX Handbag Solutions Sdn Bhd
        Trial Balance as at 31 December 2025
        Account Debit (RM) Credit (RM)
        Capital 145,000
        Bank Loan 140,000
        Other Creditors 55,400
        Office Equipment 120,500
        Motor Vehicles 165,000
        Accumulated Depreciation – Motor Vehicles 18,500
        Accumulated Depreciation – Office Equipment 30,600
        Inventory 59,000
        Trade Receivables 50,200
        Trade Payables 19,800
        Cash at Bank – Metra Bank 75,300
        Cash in Hand 6,500
        Prepaid Rent 4,000
        Retained Earnings 71,200
        Total 480,500 480,500
        """
        tb_records = service._parse_trial_balance(tb_sample_text, doc, batch)
        assert len(tb_records) == 14
        tot_dr = sum(r.debit for r in tb_records)
        tot_cr = sum(r.credit for r in tb_records)
        assert tot_dr == 480500.0
        assert tot_cr == 480500.0

        # 2. Debtors Text
        doc.doc_category = "DEBTORS"
        debtor_text = """
        Debtor List as at 31 Dec 2025
        Urban Style Sdn. Bhd. – RM9,000
        Luxe Boutique Sdn. Bhd. – RM11,200
        Orion Fashion Sdn. Bhd. – RM30,000
        """
        debtor_records = service._parse_debtors(debtor_text, doc, batch)
        assert len(debtor_records) == 3
        assert sum(d.amount for d in debtor_records) == 50200.0

        # 3. Trade Payables Text
        doc.doc_category = "CREDITORS"
        cred_text = """
        Trade Payables List
        Design Leather Sdn. Bhd. – RM14,040
        Bag Wholesalers Sdn. Bhd. – RM1,200
        Fashion Tech Ltd – RM4,560
        """
        cred_records = service._parse_creditors(cred_text, doc, batch)
        assert len(cred_records) == 3
        assert sum(c.amount for c in cred_records) == 19800.0

        # 4. Other Creditors Text
        doc.doc_category = "OTHER_CREDITORS"
        other_cred_text = """
        Other Creditors
        Mega Bag Factory – RM22,000
        Leath Supplies Sdn. Bhd. – RM3,500
        Tax Services Sdn. Bhd. – RM29,900
        """
        other_records = service._parse_other_creditors(other_cred_text, doc, batch)
        assert len(other_records) == 3
        assert sum(o.amount for o in other_records) == 55400.0

        # 5. Stock Text
        doc.doc_category = "STOCK"
        stock_text = """
        Opening Stock as at 31 December 2025
        Leather Handbag (LHB): 10 units × RM3,500 = RM35,000
        Tote Bag (TTB): 26 units × RM200 = RM5,200
        Sling Bag (SLB): 8 units × RM800 = RM6,400
        Wallet (WAL): 4 units × RM300 = RM1,200
        Travel Bag (TRB): 2 units × RM2,000 = RM4,000
        Clutch (Premium) (CLP): 6 units × RM1,200 = RM7,200
        Total opening inventory: RM59,000.
        """
        stock_records = service._parse_stock(stock_text, doc, batch)
        assert len(stock_records) == 6
        assert sum(s.amount for s in stock_records) == 59000.0
        assert sum(s.quantity for s in stock_records) == 56.0
    finally:
        db.close()


def test_company_onboarding_lifecycle_and_test_scenario():
    """
    Full automated verification of the user's initial test scenario:
    Company Name: XXX Handbag Solutions Sdn Bhd
    Address: Unit No: Lot 88, Ace Plaza, Jalan Batu, 86000 Johor Bahru
    Trial Balance: Total RM 480,500.00
    Debtors: RM 50,200.00
    Trade Payables: RM 19,800.00
    Other Creditors: RM 55,400.00
    Stock: RM 59,000.00 (6 items)
    """
    # 1. Register User
    email = f"admin_{os.urandom(4).hex()}@xxxhandbag.com"
    auth_resp = client.post("/api/auth/register", json={
        "email": email,
        "password": "SecurePassword888!",
        "full_name": "Director Tan",
        "role": "OWNER"
    })
    assert auth_resp.status_code == 200
    token = auth_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 2. Register Company
    comp_payload = {
        "company_name": "XXX Handbag Solutions Sdn Bhd",
        "registration_number": "202501008888 (1600000-X)",
        "address": "Unit No: Lot 88, Ace Plaza, Jalan Batu, 86000 Johor Bahru",
        "email": email,
        "phone": "+607-3388888",
        "currency": "MYR",
        "financial_year": 2026,
        "financial_year_start": "2026-01-01",
        "financial_year_end": "2026-12-31"
    }
    comp_resp = client.post("/api/onboarding/company", json=comp_payload, headers=headers)
    assert comp_resp.status_code == 200, comp_resp.text
    comp_data = comp_resp.json()
    comp_id = comp_data["company_id"]
    assert comp_data["company_name"] == "XXX Handbag Solutions Sdn Bhd"
    assert comp_data["onboarding_status"] in ["COMPANY_SETUP", "DOCUMENT_UPLOAD"]
    assert comp_data["is_onboarded"] is False

    # Check status endpoint
    status_resp = client.get(f"/api/onboarding/status?company_id={comp_id}", headers=headers)
    assert status_resp.status_code == 200
    assert status_resp.json()["onboarding_status"] in ["COMPANY_SETUP", "DOCUMENT_UPLOAD"]

    # 3. Simulate Trial Balance Draft Entries (RM 480,500.00 Balanced)
    tb_entries = [
        {"account_code": "100-0000", "account_name": "Capital", "debit": 0.0, "credit": 145000.0},
        {"account_code": "425-0000", "account_name": "Bank Loan", "debit": 0.0, "credit": 140000.0},
        {"account_code": "405-0000", "account_name": "Other Creditors", "debit": 0.0, "credit": 55400.0},
        {"account_code": "200-3000", "account_name": "Office Equipment", "debit": 120500.0, "credit": 0.0},
        {"account_code": "200-4000", "account_name": "Motor Vehicles", "debit": 165000.0, "credit": 0.0},
        {"account_code": "200-4005", "account_name": "Accumulated Depreciation – Motor Vehicles", "debit": 0.0, "credit": 18500.0},
        {"account_code": "200-3005", "account_name": "Accumulated Depreciation – Office Equipment", "debit": 0.0, "credit": 30600.0},
        {"account_code": "330-0000", "account_name": "Inventory", "debit": 59000.0, "credit": 0.0},
        {"account_code": "300-0000", "account_name": "Trade Receivables", "debit": 50200.0, "credit": 0.0},
        {"account_code": "400-0000", "account_name": "Trade Payables", "debit": 0.0, "credit": 19800.0},
        {"account_code": "310-1000", "account_name": "Cash at Bank – Metra Bank", "debit": 75300.0, "credit": 0.0},
        {"account_code": "320-0000", "account_name": "Cash in Hand", "debit": 6500.0, "credit": 0.0},
        {"account_code": "340-2000", "account_name": "Prepaid Rent", "debit": 4000.0, "credit": 0.0},
        {"account_code": "150-0000", "account_name": "Retained Earnings", "debit": 0.0, "credit": 71200.0},
    ]
    for item in tb_entries:
        r = client.post(f"/api/onboarding/drafts?company_id={comp_id}", json={
            "category": "TRIAL_BALANCE",
            "account_code": item["account_code"],
            "account_name": item["account_name"],
            "debit": item["debit"],
            "credit": item["credit"],
            "amount": item["debit"] if item["debit"] > 0 else item["credit"]
        }, headers=headers)
        assert r.status_code == 200

    # 4. Populate Debtors Subledger (Total RM 50,200.00)
    debtor_entries = [
        {"entity_name": "Urban Style Sdn. Bhd.", "amount": 9000.0},
        {"entity_name": "Luxe Boutique Sdn. Bhd.", "amount": 11200.0},
        {"entity_name": "Orion Fashion Sdn. Bhd.", "amount": 30000.0},
    ]
    for d in debtor_entries:
        r = client.post(f"/api/onboarding/drafts?company_id={comp_id}", json={
            "category": "DEBTOR",
            "entity_name": d["entity_name"],
            "account_code": "300-0000",
            "account_name": "TRADE DEBTORS",
            "debit": d["amount"],
            "amount": d["amount"],
            "credit_term": "30 Days"
        }, headers=headers)
        assert r.status_code == 200

    # 5. Populate Trade Payables Subledger (Total RM 19,800.00)
    payable_entries = [
        {"entity_name": "Design Leather Sdn. Bhd.", "amount": 14040.0},
        {"entity_name": "Bag Wholesalers Sdn. Bhd.", "amount": 1200.0},
        {"entity_name": "Fashion Tech Ltd", "amount": 4560.0},
    ]
    for p in payable_entries:
        r = client.post(f"/api/onboarding/drafts?company_id={comp_id}", json={
            "category": "CREDITOR",
            "entity_name": p["entity_name"],
            "account_code": "400-0000",
            "account_name": "TRADE CREDITORS",
            "credit": p["amount"],
            "amount": p["amount"],
            "credit_term": "30 Days"
        }, headers=headers)
        assert r.status_code == 200

    # 6. Populate Other Creditors Subledger (Total RM 55,400.00)
    other_cred_entries = [
        {"entity_name": "Mega Bag Factory", "amount": 22000.0},
        {"entity_name": "Leath Supplies Sdn. Bhd.", "amount": 3500.0},
        {"entity_name": "Tax Services Sdn. Bhd.", "amount": 29900.0},
    ]
    for o in other_cred_entries:
        r = client.post(f"/api/onboarding/drafts?company_id={comp_id}", json={
            "category": "OTHER_CREDITOR",
            "entity_name": o["entity_name"],
            "account_code": "405-0000",
            "account_name": "OTHER CREDITORS",
            "credit": o["amount"],
            "amount": o["amount"],
            "credit_term": "30 Days"
        }, headers=headers)
        assert r.status_code == 200

    # 7. Populate Stock Subledger (Total RM 59,000.00)
    stock_entries = [
        {"item_code": "LHB", "entity_name": "Leather Handbag", "quantity": 10.0, "unit_cost": 3500.0, "amount": 35000.0},
        {"item_code": "TTB", "entity_name": "Tote Bag", "quantity": 26.0, "unit_cost": 200.0, "amount": 5200.0},
        {"item_code": "SLB", "entity_name": "Sling Bag", "quantity": 8.0, "unit_cost": 800.0, "amount": 6400.0},
        {"item_code": "WAL", "entity_name": "Wallet", "quantity": 4.0, "unit_cost": 300.0, "amount": 1200.0},
        {"item_code": "TRB", "entity_name": "Travel Bag", "quantity": 2.0, "unit_cost": 2000.0, "amount": 4000.0},
        {"item_code": "CLP", "entity_name": "Clutch (Premium)", "quantity": 6.0, "unit_cost": 1200.0, "amount": 7200.0},
    ]
    for s in stock_entries:
        r = client.post(f"/api/onboarding/drafts?company_id={comp_id}", json={
            "category": "STOCK",
            "item_code": s["item_code"],
            "entity_name": s["entity_name"],
            "quantity": s["quantity"],
            "unit_cost": s["unit_cost"],
            "amount": s["amount"],
            "account_code": "330-0000",
            "account_name": "STOCK"
        }, headers=headers)
        assert r.status_code == 200

    # Test editing a draft record
    drafts_list = client.get(f"/api/onboarding/drafts?company_id={comp_id}&category=STOCK", headers=headers).json()
    first_stock = drafts_list[0]
    edit_resp = client.put(f"/api/onboarding/drafts/{first_stock['id']}", json={
        "notes": "Verified verified unit cost from supplier batch"
    }, headers=headers)
    assert edit_resp.status_code == 200
    assert edit_resp.json()["notes"] == "Verified verified unit cost from supplier batch"

    # 8. Run Validation
    val_resp = client.get(f"/api/onboarding/validate?company_id={comp_id}", headers=headers)
    assert val_resp.status_code == 200
    val_data = val_resp.json()

    assert val_data["is_valid"] is True, f"Validation failed: {val_data['errors']}"
    assert val_data["total_debit"] == 480500.0
    assert val_data["total_credit"] == 480500.0
    assert val_data["tb_balanced"] is True
    assert val_data["tb_diff"] == 0.0

    # Check all 4 subledger tie-outs
    tie_outs = {c["category"]: c for c in val_data["subledger_checks"]}
    assert tie_outs["Trade Receivables / Debtors"]["subledger_total"] == 50200.0
    assert tie_outs["Trade Receivables / Debtors"]["is_reconciled"] is True

    assert tie_outs["Trade Payables / Creditors"]["subledger_total"] == 19800.0
    assert tie_outs["Trade Payables / Creditors"]["is_reconciled"] is True

    assert tie_outs["Other Creditors"]["subledger_total"] == 55400.0
    assert tie_outs["Other Creditors"]["is_reconciled"] is True

    assert tie_outs["Opening Inventory / Stock"]["subledger_total"] == 59000.0
    assert tie_outs["Opening Inventory / Stock"]["is_reconciled"] is True

    # 9. Post Opening Balances
    post_resp = client.post(
        f"/api/onboarding/post?company_id={comp_id}",
        json={"as_at_date": "2025-12-31", "confirmation_notes": "Director sign-off for XXX Handbag Solutions"},
        headers=headers
    )
    assert post_resp.status_code == 200, post_resp.text
    post_data = post_resp.json()
    assert post_data["success"] is True
    assert post_data["onboarding_status"] == "ACTIVE"
    assert post_data["customers_created"] == 3
    assert post_data["suppliers_created"] == 6
    assert post_data["stock_items_created"] == 6

    # 10. Verify Company Status is now ACTIVE
    final_status = client.get(f"/api/onboarding/status?company_id={comp_id}", headers=headers).json()
    assert final_status["onboarding_status"] == "ACTIVE"
    assert final_status["is_onboarded"] is True

    # 11. Verify Duplicate Posting is Prevented
    dup_resp = client.post(
        f"/api/onboarding/post?company_id={comp_id}",
        json={"as_at_date": "2025-12-31"},
        headers=headers
    )
    assert dup_resp.status_code == 400
    assert "already been posted" in dup_resp.json()["detail"].lower()


def test_validation_blocks_unbalanced_trial_balance():
    # Setup company with unbalanced TB
    email = f"unbal_{os.urandom(4).hex()}@test.com"
    auth_resp = client.post("/api/auth/register", json={
        "email": email,
        "password": "SecurePassword888!",
        "full_name": "Test User",
        "role": "OWNER"
    })
    headers = {"Authorization": f"Bearer {auth_resp.json()['access_token']}"}

    comp_resp = client.post("/api/onboarding/company", json={
        "company_name": "Unbalanced Test Co Sdn Bhd",
        "currency": "MYR"
    }, headers=headers)
    comp_id = comp_resp.json()["company_id"]

    # Add single debit
    client.post(f"/api/onboarding/drafts?company_id={comp_id}", json={
        "category": "TRIAL_BALANCE",
        "account_code": "310-1000",
        "account_name": "CASH AT BANK",
        "debit": 10000.0,
        "credit": 0.0
    }, headers=headers)

    val = client.get(f"/api/onboarding/validate?company_id={comp_id}", headers=headers).json()
    assert val["is_valid"] is False
    assert val["tb_balanced"] is False
    assert len(val["errors"]) > 0

    # Posting must fail
    post_resp = client.post(f"/api/onboarding/post?company_id={comp_id}", json={}, headers=headers)
    assert post_resp.status_code == 400
    assert "validation errors exist" in post_resp.json()["detail"].lower()
