"""Route-level tests for POST /api/qbo/test-sales-receipt diagnostic endpoint.

Skipped automatically if app fails to import (e.g. Pinecone not configured).
"""
import importlib
import sys
from pathlib import Path

import pytest


def _app(monkeypatch, tmp_db):
    """Import app with DATA_DIR pre-set to tmp_db's parent, so LOG_DB defaults safely."""
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    # Set DATA_DIR BEFORE importing app, so LOG_DB computes to tmp_db's parent.
    monkeypatch.setenv("DATA_DIR", str(Path(tmp_db).parent))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app module not importable: {e}")


def _setup(monkeypatch, tmp_db):
    """Patch the app onto a fresh tmp DB with workspace schema + a known console key."""
    app = _app(monkeypatch, tmp_db)
    monkeypatch.setattr(app, "CONSOLE_SECRET", "secret-xyz")
    app._init_workspace_schema()
    return app


def test_route_rejects_without_console_key(monkeypatch, tmp_db):
    # With CONSOLE_SECRET set and no key sent, _qbo_auth_ok() → 401 (mirrors qbo_test_invoice).
    app = _setup(monkeypatch, tmp_db)
    client = app.app.test_client()
    r = client.post("/api/qbo/test-sales-receipt")
    assert r.status_code == 401


def test_route_books_receipt_when_authorized(monkeypatch, tmp_db):
    app = _setup(monkeypatch, tmp_db)
    from dashboard import qbo_billing
    monkeypatch.setattr(qbo_billing, "find_or_create_customer",
                        lambda *a, **k: {"Id": "C1"})
    monkeypatch.setattr(qbo_billing, "create_sales_receipt",
                        lambda *a, **k: {"Id": "SR9", "DocNumber": "42", "TotalAmt": 1.0})
    client = app.app.test_client()
    r = client.post("/api/qbo/test-sales-receipt", headers={"X-Console-Key": "secret-xyz"})
    assert r.status_code == 200
    assert r.get_json()["id"] == "SR9"
