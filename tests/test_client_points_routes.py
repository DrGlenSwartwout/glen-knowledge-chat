"""Tests for /begin/checkout-return — kind='client' patient points settlement.

Flag-gated by CLIENT_POINTS_ENABLED. When a paid Stripe session carries
kind='client' plus the patient-points metadata (patient_email, subtotal_cents,
points_redeemed_cents), the checkout-return handler must:

  - earn path  (redeemed=0): credit round(subtotal * points_earn_pct), idempotent
  - redeem path (redeemed>0): redeem min(redeemed, balance), idempotent
  - flag off: do nothing

Drives the same HTTP harness as test_client_margin_credit.py.
"""

import sqlite3

import app as appmod
from dashboard import stripe_pay as _stripe_pay_mod
from dashboard import points as _points


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_session(*, practitioner_id="42", margin_cents="1340",
                  invoice_id="INVP1", patient_email="p@x.com",
                  subtotal_cents="8000", points_redeemed_cents="0",
                  slug="brain-boost"):
    return {
        "payment_status": "paid",
        "payment_intent": "pi_test",
        "amount_total": 8000,
        "metadata": {
            "kind": "client",
            "practitioner_id": str(practitioner_id),
            "margin_cents": str(margin_cents),
            "invoice_id": invoice_id,
            "patient_email": patient_email,
            "subtotal_cents": subtotal_cents,
            "points_redeemed_cents": points_redeemed_cents,
            "slug": slug,
        },
    }


def _isolate(monkeypatch, tmp_path):
    """Point LOG_DB at a temp db and silence every side effect except points."""
    db = str(tmp_path / "points-test.db")
    monkeypatch.setattr(appmod, "LOG_DB", db)

    # Silence wallet margin credit + dispensary order recording.
    import dashboard.wallet as _wallet_mod
    monkeypatch.setattr(_wallet_mod, "earn_dropship_margin",
                        lambda *a, **k: 0)
    monkeypatch.setattr(appmod._pp, "record_dispensary_order",
                        lambda *a, **k: None)

    # Silence unrelated side effects (QBO, BOS lookups, retail points, referral).
    try:
        from dashboard import qbo_billing as _qb
        monkeypatch.setattr(_qb, "record_payment", lambda *a, **k: None)
    except Exception:
        pass
    monkeypatch.setattr(appmod._bos_orders, "find_order_by_external_ref",
                        lambda *a, **k: None)
    monkeypatch.setattr(appmod._bos_orders, "set_order_stripe_pi",
                        lambda *a, **k: None)
    monkeypatch.setattr(appmod, "_settle_order_points", lambda *a, **k: None)
    monkeypatch.setattr(appmod, "_settle_referral", lambda *a, **k: None)
    return db


def _run_return(session_id="cs_x"):
    appmod.app.config["TESTING"] = True
    client = appmod.app.test_client()
    resp = client.get(f"/begin/checkout-return?session_id={session_id}")
    assert resp.status_code in (301, 302, 303, 307, 308)
    return resp


# ── earn path ────────────────────────────────────────────────────────────────

def test_earn_credits_points_and_is_idempotent(monkeypatch, tmp_path):
    db = _isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("CLIENT_POINTS_ENABLED", "1")
    monkeypatch.setattr(
        _stripe_pay_mod, "get_session",
        lambda sid: _make_session(points_redeemed_cents="0",
                                  subtotal_cents="8000"),
    )

    _run_return("cs_earn1")

    with sqlite3.connect(db) as cx:
        cx.row_factory = sqlite3.Row
        _points.init_points_table(cx)
        assert _points.balance(cx, "p@x.com", scope="dispensary:42") == 400

    # Re-run with the same invoice_id → no double credit.
    _run_return("cs_earn1b")
    with sqlite3.connect(db) as cx:
        cx.row_factory = sqlite3.Row
        assert _points.balance(cx, "p@x.com", scope="dispensary:42") == 400


# ── redeem path ──────────────────────────────────────────────────────────────

def test_redeem_debits_points_and_is_idempotent(monkeypatch, tmp_path):
    db = _isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("CLIENT_POINTS_ENABLED", "1")

    # Seed a 500-cent balance.
    with sqlite3.connect(db) as cx:
        cx.row_factory = sqlite3.Row
        _points.init_points_table(cx)
        _points.credit(cx, "p@x.com", value_cents=500,
                       reason="earn:dispensary", order_ref="SEED",
                       scope="dispensary:42")

    monkeypatch.setattr(
        _stripe_pay_mod, "get_session",
        lambda sid: _make_session(invoice_id="INVP2",
                                  points_redeemed_cents="300",
                                  subtotal_cents="8000"),
    )

    _run_return("cs_redeem1")
    with sqlite3.connect(db) as cx:
        cx.row_factory = sqlite3.Row
        assert _points.balance(cx, "p@x.com", scope="dispensary:42") == 200

    # Re-run same invoice_id → no double redeem.
    _run_return("cs_redeem1b")
    with sqlite3.connect(db) as cx:
        cx.row_factory = sqlite3.Row
        assert _points.balance(cx, "p@x.com", scope="dispensary:42") == 200


# ── flag off ─────────────────────────────────────────────────────────────────

def test_flag_off_no_points(monkeypatch, tmp_path):
    db = _isolate(monkeypatch, tmp_path)
    monkeypatch.delenv("CLIENT_POINTS_ENABLED", raising=False)
    monkeypatch.setattr(
        _stripe_pay_mod, "get_session",
        lambda sid: _make_session(points_redeemed_cents="0",
                                  subtotal_cents="8000"),
    )

    _run_return("cs_off1")
    with sqlite3.connect(db) as cx:
        cx.row_factory = sqlite3.Row
        _points.init_points_table(cx)
        assert _points.balance(cx, "p@x.com", scope="dispensary:42") == 0


# ── POST /api/client/<code>/points — balance endpoint ─────────────────────────

def _points_client():
    appmod.app.config["TESTING"] = True
    appmod.app.config["SECRET_KEY"] = "test"
    return appmod.app.test_client()


def test_points_unknown_code_returns_404(monkeypatch):
    """Unknown dispensary code → 404."""
    monkeypatch.setattr(appmod._pp, "practitioner_id_by_dispensary_code",
                        lambda code: None)
    resp = _points_client().post("/api/client/BADCODE/points",
                                 json={"email": "p@x.com"},
                                 content_type="application/json")
    assert resp.status_code == 404
    assert resp.get_json()["ok"] is False


def test_points_non_member_returns_403(monkeypatch):
    """Patient who isn't a member (no consent) → 403 with need_optin."""
    monkeypatch.setattr(appmod._pp, "practitioner_id_by_dispensary_code",
                        lambda code: "42")
    monkeypatch.setattr(appmod, "is_member", lambda session_id, email: False)
    resp = _points_client().post("/api/client/ABC123/points",
                                 json={"email": "p@x.com"},
                                 content_type="application/json")
    assert resp.status_code == 403
    data = resp.get_json()
    assert data["ok"] is False
    assert data.get("need_optin") is True


def test_points_member_returns_scoped_balance(monkeypatch, tmp_path):
    """Member, flag on, seeded scoped balance → 200 with balance_cents + flag."""
    db = str(tmp_path / "points-route.db")
    monkeypatch.setattr(appmod, "LOG_DB", db)
    monkeypatch.setenv("CLIENT_POINTS_ENABLED", "1")
    monkeypatch.setattr(appmod._pp, "practitioner_id_by_dispensary_code",
                        lambda code: "42")
    monkeypatch.setattr(appmod, "is_member", lambda session_id, email: True)

    with sqlite3.connect(db) as cx:
        cx.row_factory = sqlite3.Row
        _points.init_points_table(cx)
        _points.credit(cx, "p@x.com", value_cents=500,
                       reason="earn:dispensary", order_ref="S1",
                       scope="dispensary:42")

    resp = _points_client().post("/api/client/ABC123/points",
                                 json={"email": "p@x.com"},
                                 content_type="application/json")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["ok"] is True
    assert data["balance_cents"] == 500
    assert data["client_points_enabled"] is True


# ── GET /api/client/<code>/catalog — exposes the flag ─────────────────────────

def test_catalog_exposes_client_points_flag(monkeypatch):
    """Catalog response carries client_points_enabled (True when flag is set)."""
    monkeypatch.setenv("CLIENT_POINTS_ENABLED", "1")
    monkeypatch.setattr(appmod._pp, "practitioner_id_by_dispensary_code",
                        lambda code: "42")
    monkeypatch.setattr(appmod._pp, "portal_data",
                        lambda pid, **kw: {"practice_name": "Test Clinic"})
    resp = _points_client().get("/api/client/ABC123/catalog")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["ok"] is True
    assert data["client_points_enabled"] is True
