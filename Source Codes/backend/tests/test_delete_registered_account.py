import os
import sys
import shutil
import pytest
import datetime
from pathlib import Path
from decimal import Decimal
from fastapi.testclient import TestClient

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.main import app
from app.db.database import get_db, SessionLocal
from app.config import DOCUMENTS_DIR
from app.db.models import (
    UserModel,
    CompanyModel,
    UserCompanyModel,
    DocumentModel,
    AccountingTransactionModel,
    JournalEntryModel,
    CustomerModel,
    SupplierModel,
    InventoryItemModel,
    BankTransactionModel,
    OpeningBalanceBatchModel,
    OpeningBalanceDraftRecordModel,
    OpeningBalanceDocumentModel,
    RegistrationSessionModel,
    ChartOfAccountModel,
    get_local_now,
)
from app.services.auth_service import hash_password

client = TestClient(app)


def test_delete_account_validation_errors():
    """Verify delete-account requires exact 'DELETE' string and valid password."""
    db = SessionLocal()
    email = f"test_val_{os.urandom(4).hex()}@handbag.com"
    pwd = "ValidPassword123!"
    user = UserModel(
        email=email,
        password_hash=hash_password(pwd),
        full_name="Validation Tester",
        role="USER",
        is_active=True
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    db.close()

    # Login to obtain token
    login_resp = client.post("/api/auth/login", json={"email": email, "password": pwd})
    assert login_resp.status_code == 200
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 1. Invalid confirmation text
    res = client.post("/api/auth/delete-account", headers=headers, json={
        "password": pwd,
        "confirmation_text": "delete"  # lowercase
    })
    assert res.status_code == 400
    assert "DELETE" in res.json()["detail"]

    # 2. Invalid password
    res = client.post("/api/auth/delete-account", headers=headers, json={
        "password": "WrongPassword!",
        "confirmation_text": "DELETE"
    })
    assert res.status_code == 401
    assert "Incorrect password" in res.json()["detail"]

    # Clean up test user
    db = SessionLocal()
    db.query(UserModel).filter(UserModel.email == email).delete()
    db.commit()
    db.close()


def test_delete_registered_account_with_sole_company_and_re_registration():
    """
    Test complete registered account deletion:
    - User owns a company exclusively with documents, transactions, customers, suppliers, inventory, bank txs, OBs.
    - Files on disk exist.
    - User confirms deletion with DELETE + password.
    - User, company, all child records and files are wiped.
    - Same email can register again immediately with a fresh onboarding session.
    """
    db = SessionLocal()
    unique_suffix = os.urandom(4).hex()
    email = f"test_purge_{unique_suffix}@handbag.com"
    pwd = "PurgePassword123!"
    comp_id = f"COMP-DEL-{unique_suffix}"

    # 1. Create User
    user = UserModel(
        email=email,
        password_hash=hash_password(pwd),
        full_name="Purge Candidate",
        role="USER",
        is_active=True
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    # 2. Create Company & Membership
    comp = CompanyModel(
        company_id=comp_id,
        company_name=f"Purge Target {unique_suffix} Sdn Bhd",
        registration_number="202601009999",
        currency="MYR",
        financial_year=2026,
        is_active=False
    )
    db.add(comp)
    db.commit()

    membership = UserCompanyModel(
        user_id=user.id,
        company_id=comp_id,
        role="OWNER",
        is_default=True
    )
    db.add(membership)

    # 3. Create Child Records across various tables
    # Subledgers
    cust = CustomerModel(company_id=comp_id, customer_code="CUST-001", customer_name="Test Customer")
    supp = SupplierModel(company_id=comp_id, supplier_code="SUPP-001", supplier_name="Test Supplier")
    item = InventoryItemModel(company_id=comp_id, item_code="ITEM-001", item_name="Test Bag", unit_cost=50.0, selling_price=100.0)
    db.add_all([cust, supp, item])

    # Bank transaction
    bank_tx = BankTransactionModel(
        company_id=comp_id,
        source="MANUAL",
        unique_source_id=f"UNIQ-{unique_suffix}",
        bank_name="ABC Bank",
        account_number="12345678",
        bank_transaction_id=f"BTX-{unique_suffix}",
        transaction_date="2026-01-15",
        description="Deposit from Customer",
        money_in=500.0,
        money_out=0.0
    )
    db.add(bank_tx)

    # Opening balance
    batch = OpeningBalanceBatchModel(company_id=comp_id, as_at_date="2025-12-31", is_balanced=True)
    db.add(batch)
    db.commit()

    ob_draft = OpeningBalanceDraftRecordModel(
        company_id=comp_id,
        batch_id=batch.id,
        category="DEBTOR",
        account_code="300-0000",
        account_name="Trade Debtors",
        debit=1000.0,
        credit=0.0
    )
    db.add(ob_draft)

    # Document & Transactions
    doc_id = f"DOC-DEL-{unique_suffix}"
    dummy_doc_dir = (DOCUMENTS_DIR / doc_id).resolve()
    dummy_doc_dir.mkdir(parents=True, exist_ok=True)
    dummy_file = dummy_doc_dir / "invoice.pdf"
    dummy_file.write_text("Dummy invoice content", encoding="utf-8")

    doc = DocumentModel(
        company_id=comp_id,
        document_id=doc_id,
        original_filename="invoice.pdf",
        file_path=str(dummy_file),
        file_type="application/pdf",
        mime_type="application/pdf",
        document_type="Sales Invoice",
        status="EXTRACTED",
        grand_total=1000.0
    )
    db.add(doc)
    db.commit()

    tx = AccountingTransactionModel(
        company_id=comp_id,
        transaction_id=f"TX-{unique_suffix}",
        document_id=doc_id,
        accounting_event="SALES_INVOICE",
        original_amount=1000.0,
        original_currency="MYR",
        status="POSTED"
    )
    db.add(tx)
    db.commit()

    je = JournalEntryModel(
        company_id=comp_id,
        journal_id=f"JE-{unique_suffix}",
        transaction_id=tx.transaction_id,
        document_id=doc_id,
        journal_date="2026-01-15",
        total_debit=1000.0,
        total_credit=1000.0,
        is_balanced=True
    )
    db.add(je)

    # Registration Session
    reg_session = RegistrationSessionModel(
        session_id=f"sess_{unique_suffix}",
        email=email,
        full_name="Purge Candidate",
        password_hash=hash_password(pwd),
        step=3,
        status="IN_PROGRESS",
        expires_at=get_local_now() + datetime.timedelta(hours=24)
    )
    db.add(reg_session)
    db.commit()

    user_original_id = user.id
    db.close()

    assert dummy_file.exists(), "Dummy test file should exist before deletion"

    # Login to authenticate
    login_resp = client.post("/api/auth/login", json={"email": email, "password": pwd})
    assert login_resp.status_code == 200
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Execute Account Deletion
    del_resp = client.post("/api/auth/delete-account", headers=headers, json={
        "password": pwd,
        "confirmation_text": "DELETE"
    })
    assert del_resp.status_code == 200, del_resp.text
    del_data = del_resp.json()
    assert "permanently deleted" in del_data["message"]
    assert del_data["deleted_user_email"] == email

    # Verify Database state
    db = SessionLocal()
    # 1. UserModel is deleted
    assert db.query(UserModel).filter(UserModel.id == user_original_id).first() is None
    assert db.query(UserModel).filter(UserModel.email == email).first() is None

    # 2. CompanyModel and UserCompanyModel are deleted
    assert db.query(CompanyModel).filter(CompanyModel.company_id == comp_id).first() is None
    assert db.query(UserCompanyModel).filter(UserCompanyModel.company_id == comp_id).first() is None
    assert db.query(UserCompanyModel).filter(UserCompanyModel.user_id == user_original_id).first() is None

    # 3. Child tables are deleted
    assert db.query(CustomerModel).filter(CustomerModel.company_id == comp_id).count() == 0
    assert db.query(SupplierModel).filter(SupplierModel.company_id == comp_id).count() == 0
    assert db.query(InventoryItemModel).filter(InventoryItemModel.company_id == comp_id).count() == 0
    assert db.query(BankTransactionModel).filter(BankTransactionModel.company_id == comp_id).count() == 0
    assert db.query(OpeningBalanceBatchModel).filter(OpeningBalanceBatchModel.company_id == comp_id).count() == 0
    assert db.query(OpeningBalanceDraftRecordModel).filter(OpeningBalanceDraftRecordModel.company_id == comp_id).count() == 0
    assert db.query(DocumentModel).filter(DocumentModel.company_id == comp_id).count() == 0
    assert db.query(AccountingTransactionModel).filter(AccountingTransactionModel.company_id == comp_id).count() == 0
    assert db.query(JournalEntryModel).filter(JournalEntryModel.company_id == comp_id).count() == 0

    # 4. Registration sessions are deleted
    assert db.query(RegistrationSessionModel).filter(RegistrationSessionModel.email == email).count() == 0
    db.close()

    # 5. Disk files are cleaned up
    assert not dummy_file.exists(), "Dummy document file should have been deleted"
    assert not dummy_doc_dir.exists(), "Dummy document directory should have been deleted"

    # 6. Verify Re-Registration: Deleted email can register again immediately
    re_reg_resp = client.post("/api/auth/register", json={
        "email": email,
        "password": "NewPassword456!",
        "full_name": "Re-registered User",
        "role": "USER"
    })
    assert re_reg_resp.status_code == 200, re_reg_resp.text
    new_user_data = re_reg_resp.json()["user"]
    assert new_user_data["email"] == email
    assert new_user_data["id"] is not None

    # Verify new user has no old company records
    db = SessionLocal()
    new_user_comps = db.query(UserCompanyModel).filter(UserCompanyModel.user_id == new_user_data["id"]).all()
    assert all(c.company_id != comp_id for c in new_user_comps)
    # Clean up re-registered user
    db.query(UserModel).filter(UserModel.id == new_user_data["id"]).delete()
    db.commit()
    db.close()

    # 7. Also verify that starting a brand new 3-step Onboarding Session succeeds for this email
    onboarding_start_resp = client.post("/api/onboarding/session/start", json={
        "email": email,
        "password": "BrandNewPassword123!",
        "full_name": "Brand New Onboarding User"
    })
    assert onboarding_start_resp.status_code == 200, onboarding_start_resp.text
    session_data = onboarding_start_resp.json()
    assert "session_id" in session_data
    assert session_data["email"] == email
    assert session_data["step"] == 1

    # Clean up temporary registration session
    db = SessionLocal()
    db.query(RegistrationSessionModel).filter(RegistrationSessionModel.session_id == session_data["session_id"]).delete()
    db.commit()
    db.close()


def test_delete_account_shared_company_preservation():
    """Verify that if a user deletes their account, shared company records remain for other users."""
    db = SessionLocal()
    unique_suffix = os.urandom(4).hex()
    email_a = f"user_a_{unique_suffix}@handbag.com"
    email_b = f"user_b_{unique_suffix}@handbag.com"
    comp_id = f"COMP-SHARED-{unique_suffix}"

    user_a = UserModel(email=email_a, password_hash=hash_password("PassA123!"), full_name="User A", role="USER", is_active=True)
    user_b = UserModel(email=email_b, password_hash=hash_password("PassB123!"), full_name="User B", role="USER", is_active=True)
    db.add_all([user_a, user_b])
    db.commit()
    user_a_id = user_a.id
    user_b_id = user_b.id

    comp = CompanyModel(company_id=comp_id, company_name=f"Shared Co {unique_suffix}", registration_number="123456", currency="MYR")
    db.add(comp)
    db.commit()

    uc_a = UserCompanyModel(user_id=user_a_id, company_id=comp_id, role="OWNER", is_default=True)
    uc_b = UserCompanyModel(user_id=user_b_id, company_id=comp_id, role="ADMIN", is_default=True)
    db.add_all([uc_a, uc_b])
    db.commit()
    db.close()

    # User A logs in and deletes their account
    login_resp = client.post("/api/auth/login", json={"email": email_a, "password": "PassA123!"})
    token_a = login_resp.json()["access_token"]

    del_resp = client.post("/api/auth/delete-account", headers={"Authorization": f"Bearer {token_a}"}, json={
        "password": "PassA123!",
        "confirmation_text": "DELETE"
    })
    assert del_resp.status_code == 200

    db = SessionLocal()
    # User A is deleted
    assert db.query(UserModel).filter(UserModel.id == user_a_id).first() is None
    # Shared company still exists
    assert db.query(CompanyModel).filter(CompanyModel.company_id == comp_id).first() is not None
    # User B still has access, and has been promoted to OWNER
    uc_b_check = db.query(UserCompanyModel).filter(UserCompanyModel.user_id == user_b_id, UserCompanyModel.company_id == comp_id).first()
    assert uc_b_check is not None
    assert uc_b_check.role == "OWNER"

    # Clean up test records
    db.query(UserCompanyModel).filter(UserCompanyModel.company_id == comp_id).delete()
    db.query(CompanyModel).filter(CompanyModel.company_id == comp_id).delete()
    db.query(UserModel).filter(UserModel.id == user_b_id).delete()
    db.commit()
    db.close()
