"""Tests for portal_data's pricing_config/pricing_ceilings + POST /api/practitioner/pricing.

Patterns borrowed from sibling tests:
  - app-import missing-env guard: tests/test_phase3_paylink.py
  - fake Supabase cursor for the practitioners row: tests/test_practitioner_portal.py
  - pointing both app.LOG_DB and practitioner_portal._LOG_DB at the same tmp sqlite
    file (they are two independent module-level constants): tests/test_phase3_paylink.py
"""

import importlib
import sys
from pathlib import Path

import pytest

PID = "1"

PRACTITIONER_ROW = {
    "id": PID,
    "name": "Dr Test",
    "practice_name": "Test Clinic",
    "email": "dr@test.com",
    "portal_role": "licensed",
    "modules_completed": 0,
    "wallet_balance_cents": 0,
    "wholesale_unlocked_at": None,
    "application_status": "approved",
    "application_submitted_at": None,
    "approval_notes": None,
    "resale_license_number": None,
    "credentials": "",
}


class _FakeCur:
    """Minimal cursor stand-in for the single SELECT portal_data issues."""
    def __init__(self, row):
        self._row = row

    def execute(self, sql, params=()):
        pass

    def fetchone(self):
        return self._row


class _FakeCtx:
    def __init__(self, cur):
        self.cur = cur

    def __enter__(self):
        return self.cur

    def __exit__(self, *a):
        return False


def _app():
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")


@pytest.fixture
def wired(monkeypatch, tmp_path):
    """Wires up: fake Supabase practitioners row, a shared tmp sqlite file for
    both app.LOG_DB and practitioner_portal._LOG_DB, and a signed-in session."""
    app_module = _app()
    db_path = tmp_path / "chat_log.db"

    import db_supabase
    monkeypatch.setattr(db_supabase, "supabase_cursor",
                        lambda: _FakeCtx(_FakeCur(dict(PRACTITIONER_ROW))))

    from dashboard import practitioner_portal as pp
    monkeypatch.setattr(pp, "_LOG_DB", db_path)
    monkeypatch.setattr(app_module, "LOG_DB", db_path)
    monkeypatch.setattr(app_module, "_practitioner_session_pid", lambda: PID)

    return app_module, pp, db_path


def test_portal_data_carries_config_and_ceilings(wired):
    _app_module, pp, _db_path = wired
    data = pp.portal_data(PID)
    assert "pricing_config" in data
    assert data["pricing_config"] == {}
    assert "pricing_ceilings" in data
    assert data["pricing_ceilings"]["open_total"] == 29.0


def test_post_pricing_rejects_bad_dial(wired):
    app_module, _pp, _db_path = wired
    client = app_module.app.test_client()
    r = client.post("/api/practitioner/pricing",
                    json={"config": {"standard": {"same_sku": {"enabled": True, "dial": 2}}}})
    assert r.status_code == 400
    assert not r.get_json()["ok"]


def test_post_pricing_saves_valid(wired):
    app_module, pp, _db_path = wired
    client = app_module.app.test_client()
    cfg = {"standard": {"open_total": {"enabled": True, "dial": 0.5}}}
    r = client.post("/api/practitioner/pricing", json={"config": cfg})
    assert r.status_code == 200
    assert r.get_json()["ok"] is True
    assert pp.portal_data(PID)["pricing_config"] == cfg


def test_post_pricing_requires_auth(monkeypatch, wired):
    app_module, _pp, _db_path = wired
    monkeypatch.setattr(app_module, "_practitioner_session_pid", lambda: None)
    client = app_module.app.test_client()
    r = client.post("/api/practitioner/pricing", json={"config": {}})
    assert r.status_code == 401
    assert not r.get_json()["ok"]
