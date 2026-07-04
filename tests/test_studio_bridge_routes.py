# tests/test_studio_bridge_routes.py
"""Routes for the Studio.com bridge Flow B free-month claim.

A customer who joined studio.com/drglen claims a free first month of live group
coaching, vaults a card (mode=setup, $0), then auto-continues at $99/mo. Ships
dark behind STUDIO_BRIDGE_ENABLED (default off).
"""
import sqlite3
import datetime

import app as appmod
from dashboard import studio_bridge as sb, subscriptions as subs


def _open(db):
    cx = sqlite3.connect(db)
    cx.row_factory = sqlite3.Row
    return cx


def _setup(monkeypatch, tmp_path, *, member=True):
    """Stub LOG_DB, consent, and the Stripe setup-vault calls."""
    db = str(tmp_path / "log.db")
    monkeypatch.setattr(appmod, "LOG_DB", db)
    monkeypatch.setattr(appmod, "is_member", lambda sid, email: member)

    cap = {}

    def fake_setup_session(*, customer_email, metadata, success_url, cancel_url):
        cap["setup_email"] = customer_email
        cap["setup_metadata"] = metadata
        cap["setup_success"] = success_url
        cap["setup_cancel"] = cancel_url
        return {"id": "cs_1", "url": "https://stripe/setup"}

    monkeypatch.setattr(appmod.stripe_pay, "create_setup_session",
                        fake_setup_session, raising=False)
    monkeypatch.setattr(
        appmod.stripe_pay, "get_session",
        lambda sid: {"setup_intent": "si_1", "metadata": {"email": "p@x.com"}},
        raising=False)
    monkeypatch.setattr(
        appmod.stripe_pay, "get_setup_intent",
        lambda si: {"customer": "cus_1", "payment_method": "pm_1"},
        raising=False)
    return cap, db


# ── claim API: consent gate ──────────────────────────────────────────────────

def test_claim_blocked_without_consent(monkeypatch, tmp_path):
    cap, db = _setup(monkeypatch, tmp_path, member=False)
    monkeypatch.setenv("STUDIO_BRIDGE_ENABLED", "1")
    c = appmod.app.test_client()
    r = c.post("/api/studio/claim", json={"email": "p@x.com", "name": "P"})
    assert r.status_code == 403, r.get_data(as_text=True)
    assert r.get_json().get("need_optin") is True
    # no setup session created, no claim recorded
    assert "setup_email" not in cap
    cx = _open(db)
    sb.init_table(cx)
    assert sb.get(cx, "p@x.com") is None
    cx.close()


def test_claim_records_pending_and_returns_stripe_url(monkeypatch, tmp_path):
    cap, db = _setup(monkeypatch, tmp_path, member=True)
    monkeypatch.setenv("STUDIO_BRIDGE_ENABLED", "1")
    c = appmod.app.test_client()
    r = c.post("/api/studio/claim", json={"email": "p@x.com", "name": "P"})
    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_json()
    assert body["ok"] is True
    assert body["stripe_url"] == "https://stripe/setup"
    # metadata carries the kind + email so the return can reconnect
    assert cap["setup_metadata"]["kind"] == "studio_bridge"
    assert cap["setup_metadata"]["email"] == "p@x.com"
    # pending claim recorded
    cx = _open(db)
    sb.init_table(cx)
    rec = sb.get(cx, "p@x.com")
    assert rec is not None
    assert rec["status"] == "pending"
    cx.close()


# ── claim-return: grant the membership ───────────────────────────────────────

def test_claim_return_creates_membership_and_is_idempotent(monkeypatch, tmp_path):
    cap, db = _setup(monkeypatch, tmp_path, member=True)
    monkeypatch.setenv("STUDIO_BRIDGE_ENABLED", "1")
    c = appmod.app.test_client()

    # First the claim records pending
    c.post("/api/studio/claim", json={"email": "p@x.com", "name": "P"})

    r = c.get("/studio/claim-return?session_id=cs_1")
    assert r.status_code in (302, 303), r.get_data(as_text=True)

    cx = _open(db)
    subs.init_subscriptions_table(cx)
    subs.migrate_add_membership_columns(cx)
    subs.migrate_add_term_cap_column(cx)
    subs.migrate_add_attribution_column(cx)
    rows = subs.active_memberships_by_email(cx, "p@x.com")
    assert len(rows) == 1, rows
    m = rows[0]
    assert m["kind"] == "membership"
    assert m["amount_cents"] == 9900
    assert m["cadence_months"] == 1
    today = datetime.date.today().isoformat()
    assert m["next_charge_date"] == subs.add_months(today, 1)
    # marked granted
    sb.init_table(cx)
    assert sb.already_granted(cx, "p@x.com") is True
    cx.close()

    # Idempotent: a second identical return creates NO second membership
    r2 = c.get("/studio/claim-return?session_id=cs_1")
    assert r2.status_code in (302, 303)
    cx = _open(db)
    rows = subs.active_memberships_by_email(cx, "p@x.com")
    assert len(rows) == 1, rows
    cx.close()


# ── flag off ─────────────────────────────────────────────────────────────────

def test_claim_disabled_when_flag_off(monkeypatch, tmp_path):
    cap, db = _setup(monkeypatch, tmp_path, member=True)
    monkeypatch.delenv("STUDIO_BRIDGE_ENABLED", raising=False)
    c = appmod.app.test_client()
    r = c.post("/api/studio/claim", json={"email": "p@x.com", "name": "P"})
    assert r.status_code in (403, 404), r.get_data(as_text=True)
    assert "setup_email" not in cap


# ── claim page ───────────────────────────────────────────────────────────────

def test_claim_page_served(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, member=True)
    monkeypatch.setenv("STUDIO_BRIDGE_ENABLED", "1")
    c = appmod.app.test_client()
    r = c.get("/studio/claim")
    assert r.status_code == 200
    assert "Studio" in r.get_data(as_text=True)
