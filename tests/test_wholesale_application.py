"""Tier-2 wholesale application → approval gate.

Covers the data layer (validate / submit / decide / list against a fake Supabase
cursor) and the route surface (apply ToS gate, console-key guard on approve/reject).
The apply path sits ALONGSIDE the existing licensed/coach auto-unlock paths and
converges on the same wholesale_unlocked_at flag.
"""

import importlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest


def _load_app():
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app module not importable in this env: {e}")


# ── Fake Supabase cursor ──────────────────────────────────────────────────────

class _FakeCursor:
    def __init__(self, fetchone_queue=None, fetchall_result=None):
        self.executed = []
        self._one = list(fetchone_queue or [])
        self._all = fetchall_result if fetchall_result is not None else []

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchone(self):
        return self._one.pop(0) if self._one else None

    def fetchall(self):
        return self._all


class _FakeCtx:
    def __init__(self, cur):
        self.cur = cur

    def __enter__(self):
        return self.cur

    def __exit__(self, *a):
        return False


def _patch_cursor(monkeypatch, cur):
    import db_supabase
    monkeypatch.setattr(db_supabase, "supabase_cursor", lambda: _FakeCtx(cur))


# ── validate (pure) ───────────────────────────────────────────────────────────

def test_validate_ok():
    from dashboard.practitioner_portal import validate_wholesale_application
    clean, err = validate_wholesale_application({
        "name": "Jane Smith", "email": "JANE@Shop.com",
        "resale_license_number": "GE-1", "tos": True})
    assert err is None
    assert clean["email"] == "jane@shop.com"
    assert clean["resale_license_number"] == "GE-1"


def test_validate_requires_resale_and_tos():
    from dashboard.practitioner_portal import validate_wholesale_application
    assert validate_wholesale_application(
        {"name": "J", "email": "a@b.com", "tos": True})[0] is None      # no resale
    assert validate_wholesale_application(
        {"name": "J", "email": "a@b.com", "resale_license_number": "X"})[1]  # no tos
    assert validate_wholesale_application(
        {"name": "J", "email": "bad", "resale_license_number": "X", "tos": True})[0] is None


# ── submit ────────────────────────────────────────────────────────────────────

CLEAN = {"email": "a@b.com", "name": "Jane Smith", "resale_license_number": "GE-1",
         "license_state": "HI", "practice_name": None, "credentials": None,
         "phone": None, "website": None}


def test_submit_new_creates_pending(monkeypatch):
    from dashboard import practitioner_portal as pp
    cur = _FakeCursor(fetchone_queue=[None, {"id": "new-id"}])
    _patch_cursor(monkeypatch, cur)
    pid, already = pp.submit_wholesale_application(CLEAN)
    assert pid == "new-id" and already is False
    insert_sql = cur.executed[1][0]
    assert "INSERT INTO practitioners" in insert_sql
    assert "'pending'" in insert_sql
    assert "wholesale_unlocked_at" not in insert_sql   # stays NULL → not unlocked


def test_submit_existing_unlocked_is_noop(monkeypatch):
    from dashboard import practitioner_portal as pp
    cur = _FakeCursor(fetchone_queue=[{"id": "x", "wholesale_unlocked_at": datetime.now(timezone.utc)}])
    _patch_cursor(monkeypatch, cur)
    pid, already = pp.submit_wholesale_application(CLEAN)
    assert pid == "x" and already is True
    assert len(cur.executed) == 1   # only the SELECT; no write


def test_submit_existing_locked_sets_pending(monkeypatch):
    from dashboard import practitioner_portal as pp
    cur = _FakeCursor(fetchone_queue=[{"id": "x", "wholesale_unlocked_at": None}])
    _patch_cursor(monkeypatch, cur)
    pid, already = pp.submit_wholesale_application(CLEAN)
    assert pid == "x" and already is False
    assert "application_status='pending'" in cur.executed[1][0]


# ── decide ────────────────────────────────────────────────────────────────────

def test_decide_approve_unlocks(monkeypatch):
    from dashboard import practitioner_portal as pp
    cur = _FakeCursor(fetchone_queue=[{"email": "a@b.com", "name": "Jane"}])
    _patch_cursor(monkeypatch, cur)
    who = pp.decide_application("x", approve=True, notes="ok")
    assert who == {"email": "a@b.com", "name": "Jane"}
    sql = cur.executed[0][0]
    assert "application_status='approved'" in sql
    assert "wholesale_unlocked_at=COALESCE" in sql


def test_decide_reject_leaves_locked(monkeypatch):
    from dashboard import practitioner_portal as pp
    cur = _FakeCursor(fetchone_queue=[{"email": "a@b.com", "name": "Jane"}])
    _patch_cursor(monkeypatch, cur)
    who = pp.decide_application("x", approve=False, notes="bad cert")
    assert who["email"] == "a@b.com"
    sql = cur.executed[0][0]
    assert "application_status='rejected'" in sql
    assert "wholesale_unlocked_at" not in sql


def test_decide_unknown_id_returns_none(monkeypatch):
    from dashboard import practitioner_portal as pp
    cur = _FakeCursor(fetchone_queue=[None])
    _patch_cursor(monkeypatch, cur)
    assert pp.decide_application("nope", approve=True) is None


def test_list_pending(monkeypatch):
    from dashboard import practitioner_portal as pp
    cur = _FakeCursor(fetchall_result=[{
        "id": "x", "name": "Jane", "email": "a@b.com", "practice_name": "Shop",
        "portal_role": "reseller", "resale_license_number": "GE-1",
        "license_state": "HI", "application_submitted_at": datetime(2026, 6, 11, tzinfo=timezone.utc)}])
    _patch_cursor(monkeypatch, cur)
    out = pp.list_pending_applications()
    assert out[0]["id"] == "x"
    assert out[0]["application_submitted_at"].startswith("2026-06-11")


# ── routes ────────────────────────────────────────────────────────────────────

def test_apply_requires_tos(monkeypatch, tmp_path):
    app_module = _load_app()
    monkeypatch.setattr(app_module, "LOG_DB", str(tmp_path / "chat_log.db"))
    client = app_module.app.test_client()
    r = client.post("/api/wholesale/apply",
                    json={"name": "J", "email": "a@b.com", "resale_license_number": "X"})
    assert r.status_code == 400
    assert "Terms" in (r.get_json() or {}).get("error", "")


def test_apply_happy_path(monkeypatch, tmp_path):
    app_module = _load_app()
    monkeypatch.setattr(app_module, "LOG_DB", str(tmp_path / "chat_log.db"))
    monkeypatch.setattr(app_module._pp, "submit_wholesale_application",
                        lambda clean: ("pid-1", False))
    monkeypatch.setattr(app_module._pp, "create_magic_link_token",
                        lambda pid, email: "tok")
    monkeypatch.setattr(app_module, "_send_practitioner_magic_link",
                        lambda *a, **k: ("console", None))
    monkeypatch.setattr(app_module, "_send_full_report_email",
                        lambda *a, **k: ("console", None))
    client = app_module.app.test_client()
    r = client.post("/api/wholesale/apply", json={
        "name": "Jane Smith", "email": "a@b.com",
        "resale_license_number": "GE-1", "tos": True})
    assert r.status_code == 200
    assert r.get_json().get("ok") is True


def test_admin_approve_requires_console_key(monkeypatch):
    app_module = _load_app()
    import dashboard
    monkeypatch.setattr(dashboard, "CONSOLE_SECRET", "s3cret")
    monkeypatch.setattr(app_module._pp, "decide_application",
                        lambda pid, approve, notes="": {"email": "a@b.com", "name": "Jane"})
    monkeypatch.setattr(app_module, "_send_wholesale_decision_email", lambda *a, **k: None)
    client = app_module.app.test_client()
    # no key → 401
    r = client.post("/admin/wholesale/approve", json={"id": "x"})
    assert r.status_code == 401
    # with key → 200
    r = client.post("/admin/wholesale/approve?key=s3cret", json={"id": "x"})
    assert r.status_code == 200
    assert r.get_json().get("data", {}).get("approved") == "x" or r.get_json().get("approved") == "x"
