"""Funnel ToS consistency - gate affiliate/reorder/referral/concierge."""
import importlib, sqlite3, sys
from pathlib import Path
import pytest


def _load_app():
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")


def _fresh(app_module, monkeypatch, tmp_path):
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    import begin_funnel
    with sqlite3.connect(db) as cx:
        begin_funnel.init_journey_tables(cx)
    app_module._init_referral_tables()
    monkeypatch.setattr(app_module, "ghl_onboard_contact", lambda *a, **k: {"contact_id": "x"})
    monkeypatch.setattr(app_module, "_capture_concierge_referral", lambda *a, **k: None)
    return db


def _make_member(app_module, db, email, session="m1"):
    import begin_funnel
    with sqlite3.connect(db) as cx:
        begin_funnel.record_unlock(cx, session_id=session, trigger="tos", email=email, tos=True)


def test_affiliate_apply_blocked_for_non_member(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    client = app_module.app.test_client()
    r = client.post("/affiliate/apply", json={"name": "Ann B", "email": "ann@x.com"})
    assert r.status_code == 403 and r.get_json().get("need_optin") is True
    # not created
    with sqlite3.connect(db) as cx:
        n = cx.execute("SELECT COUNT(*) FROM affiliate_signups WHERE LOWER(email)='ann@x.com'").fetchone()[0]
    assert n == 0


def test_affiliate_apply_with_tos_creates_and_sets_membership(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    client = app_module.app.test_client()
    r = client.post("/affiliate/apply", json={"name": "Ann B", "email": "ann@x.com", "tos": True})
    assert r.status_code == 200
    assert app_module.is_member(email="ann@x.com") is True
    with sqlite3.connect(db) as cx:
        n = cx.execute("SELECT COUNT(*) FROM affiliate_signups WHERE LOWER(email)='ann@x.com'").fetchone()[0]
    assert n == 1


def test_affiliate_apply_member_passes_through(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    _make_member(app_module, db, "lee@x.com")
    client = app_module.app.test_client()
    r = client.post("/affiliate/apply", json={"name": "Lee X", "email": "lee@x.com"})
    assert r.status_code == 200


def test_affiliate_form_without_tos_redirects_to_error(monkeypatch, tmp_path):
    app_module = _load_app(); _fresh(app_module, monkeypatch, tmp_path)
    client = app_module.app.test_client()
    r = client.post("/affiliate/apply-form", data={"name": "Ann B", "email": "ann@x.com"})
    assert r.status_code in (302, 303)
    assert "error=" in r.headers.get("Location", "")


# ---------------------------------------------------------------------------
# Task 2: Reorder gate
# ---------------------------------------------------------------------------

def _reorder_client(app_module, email):
    c = app_module.app.test_client()
    c.set_cookie("rm_reorder_email", email)
    return c


def test_reorder_checkout_blocked_for_non_member(monkeypatch, tmp_path):
    app_module = _load_app(); _fresh(app_module, monkeypatch, tmp_path)
    c = _reorder_client(app_module, "ann@x.com")
    r = c.post("/reorder/checkout", json={"items": []})
    assert r.status_code == 403 and r.get_json().get("need_optin") is True


def test_reorder_items_blocked_for_non_member(monkeypatch, tmp_path):
    app_module = _load_app(); _fresh(app_module, monkeypatch, tmp_path)
    c = _reorder_client(app_module, "ann@x.com")
    r = c.get("/api/reorder/items")
    assert r.status_code == 403 and r.get_json().get("need_optin") is True


def test_reorder_subscribe_blocked_for_non_member(monkeypatch, tmp_path):
    app_module = _load_app(); _fresh(app_module, monkeypatch, tmp_path)
    c = _reorder_client(app_module, "ann@x.com")
    r = c.post("/reorder/subscribe", json={"items": []})
    assert r.status_code == 403 and r.get_json().get("need_optin") is True


def test_reorder_items_member_not_gated(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    _make_member(app_module, db, "lee@x.com")
    c = _reorder_client(app_module, "lee@x.com")
    r = c.get("/api/reorder/items")
    assert r.status_code != 403  # passes the gate (200 or its normal non-gate response)


# ---------------------------------------------------------------------------
# Task 3: Referral my-code gate
# ---------------------------------------------------------------------------

def test_referral_mycode_blocked_for_non_member(monkeypatch, tmp_path):
    app_module = _load_app(); _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "_REFERRALS", True)
    c = app_module.app.test_client()
    c.set_cookie("rm_reorder_email", "ann@x.com")
    r = c.get("/api/referral/my-code")
    assert r.status_code == 403 and r.get_json().get("need_optin") is True


def test_referral_mycode_member_not_gated(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "_REFERRALS", True)
    _make_member(app_module, db, "lee@x.com")
    c = app_module.app.test_client()
    c.set_cookie("rm_reorder_email", "lee@x.com")
    r = c.get("/api/referral/my-code")
    assert r.status_code != 403


# ---------------------------------------------------------------------------
# Task 4: Concierge gate
# ---------------------------------------------------------------------------

def test_concierge_add_blocked_for_non_member(monkeypatch, tmp_path):
    app_module = _load_app(); _fresh(app_module, monkeypatch, tmp_path)
    c = app_module.app.test_client()
    r = c.post("/begin/concierge/add", json={"slug": "some-slug", "invoice_id": "inv123"})
    assert r.status_code == 403
    data = r.get_json()
    assert data.get("need_optin") is True


def test_concierge_add_member_not_gated(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    _make_member(app_module, db, "member@test.com", session="s-concierge")
    c = app_module.app.test_client()
    c.set_cookie("amg_session", "s-concierge")
    r = c.post("/begin/concierge/add", json={"slug": "nonexistent", "invoice_id": "inv123"})
    # must NOT be 403 (may be 400 for unknown slug, but not ToS gate)
    assert r.status_code != 403
