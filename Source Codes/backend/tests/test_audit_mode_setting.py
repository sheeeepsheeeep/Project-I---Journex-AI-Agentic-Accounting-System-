import os
import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_audit_mode_default_is_manual():
    res = client.get("/api/settings/audit-mode")
    assert res.status_code == 200
    data = res.json()
    assert data.get("audit_mode") in ["MANUAL", "AUTOMATIC"]

def test_switch_audit_mode_flow():
    # Set to AUTOMATIC
    res = client.post("/api/settings/audit-mode", json={"audit_mode": "AUTOMATIC"})
    assert res.status_code == 200
    assert res.json().get("audit_mode") == "AUTOMATIC"

    # Verify GET returns AUTOMATIC
    res = client.get("/api/settings/audit-mode")
    assert res.status_code == 200
    assert res.json().get("audit_mode") == "AUTOMATIC"

    # Set back to MANUAL (default)
    res = client.post("/api/settings/audit-mode", json={"audit_mode": "MANUAL"})
    assert res.status_code == 200
    assert res.json().get("audit_mode") == "MANUAL"

    # Invalid mode rejection
    res = client.post("/api/settings/audit-mode", json={"audit_mode": "INVALID_MODE"})
    assert res.status_code == 400
