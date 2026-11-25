import os
import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

import pytest
import json
from fastapi.testclient import TestClient
from app.main import app
from app.db.database import Base, engine, SessionLocal
from app.db.models import JournalEntryModel, ChartOfAccountModel

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_teardown_db():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    # Clean up test journals and accounts
    db.query(JournalEntryModel).filter(JournalEntryModel.journal_id.like("TEST_OVR_%")).delete()
    db.query(ChartOfAccountModel).filter(ChartOfAccountModel.account_code == "300-TEST").delete()
    db.commit()
    
    # Create test asset account with 0 starting balance
    test_acc = ChartOfAccountModel(
        account_code="300-TEST",
        account_name="Test Bank Account",
        account_type="Asset"
    )
    db.add(test_acc)
    db.commit()
    
    yield db
    
    # Cleanup
    db.query(JournalEntryModel).filter(JournalEntryModel.journal_id.like("TEST_OVR_%")).delete()
    db.query(ChartOfAccountModel).filter(ChartOfAccountModel.account_code == "300-TEST").delete()
    db.commit()
    db.close()

def test_override_post_and_apply_fix(setup_teardown_db):
    db = setup_teardown_db
    
    # Create a pending journal entry that would cause abnormal negative balance in 300-TEST (credit of RM 500 when balance is 0)
    lines = [
        {"account_code": "5000", "account_name": "Office Expense", "debit": 500.0, "credit": 0.0},
        {"account_code": "300-TEST", "account_name": "Test Bank Account", "debit": 0.0, "credit": 500.0}
    ]
    
    journal = JournalEntryModel(
        journal_id="TEST_OVR_001",
        transaction_id="TXN-OVR-001",
        document_id="DOC-OVR-001",
        journal_date="2026-10-29",
        description="Test payment causing abnormal negative balance",
        reference="REF-001",
        lines_json=json.dumps(lines),
        total_debit=500.0,
        total_credit=500.0,
        status="PENDING_AUDIT"
    )
    db.add(journal)
    db.commit()
    
    # 1. Standard POST without force should be blocked (422)
    resp = client.post("/api/journals/TEST_OVR_001/post")
    assert resp.status_code == 422
    assert "POSTING_BLOCKED_ABNORMAL_BALANCE" in str(resp.json())
    
    # 2. Standard apply-fix without force_override should be blocked (422)
    resp = client.post("/api/journals/TEST_OVR_001/apply-fix", json={
        "new_lines": lines,
        "total_debit": 500.0,
        "total_credit": 500.0,
        "reasoning": "Standard fix attempt",
        "force_override": False
    })
    assert resp.status_code == 422
    assert "POSTING_BLOCKED_ABNORMAL_BALANCE" in str(resp.json())
    
    # 3. Apply fix WITH force_override=True should succeed
    resp = client.post("/api/journals/TEST_OVR_001/apply-fix", json={
        "new_lines": lines,
        "total_debit": 500.0,
        "total_credit": 500.0,
        "reasoning": "User authorized override",
        "force_override": True
    })
    assert resp.status_code == 200
    db.refresh(journal)
    assert journal.status == "POSTED"
    reasoning_data = json.loads(journal.audit_reasoning)
    assert reasoning_data.get("user_override") is True
    assert "HUMAN OVERRIDE" in reasoning_data.get("thought_process", {}).get("decision_rationale", "")
    
    # 4. Reset to PENDING_AUDIT and test /post?force=true
    journal.status = "PENDING_AUDIT"
    db.commit()
    
    resp_force = client.post("/api/journals/TEST_OVR_001/post?force=true")
    assert resp_force.status_code == 200
    db.refresh(journal)
    assert journal.status == "POSTED"
