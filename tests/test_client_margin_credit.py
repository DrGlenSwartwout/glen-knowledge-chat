"""Tests for /begin/checkout-return — kind='client' margin credit.

When a paid Stripe session carries kind='client' in its metadata, the
checkout-return handler must call wallet.earn_dropship_margin with
(practitioner_id, margin_cents, qbo_invoice_id=invoice_id).

Three scenarios:
  1. Paid client order → earn_dropship_margin called once with correct args.
  2. Non-client kind (e.g. 'retail') → earn_dropship_margin NOT called.
  3. earn_dropship_margin raises → redirect still happens (never breaks).
"""

import app as appmod
from dashboard import stripe_pay as _stripe_pay_mod
import dashboard.wallet as _wallet_mod


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_session(kind, payment_status="paid", practitioner_id="p1",
                  margin_cents="1340", invoice_id="INV", slug="brain-boost"):
    """Build a minimal fake Stripe session dict."""
    return {
        "payment_status": payment_status,
        "payment_intent": "pi_test",
        "amount_total": 7000,
        "metadata": {
            "kind": kind,
            "practitioner_id": str(practitioner_id),
            "margin_cents": str(margin_cents),
            "invoice_id": invoice_id,
            "slug": slug,
        },
    }


def _stub_out_side_effects(monkeypatch):
    """Silence all side effects that are NOT the wallet credit:
    QBO payment recording, SQLite order lookups, points, referral."""
    # Silence QBO recording
    try:
        from dashboard import qbo_billing as _qb
        monkeypatch.setattr(_qb, "record_payment", lambda *a, **k: None)
    except Exception:
        pass

    # Silence SQLite / BOS order lookups via the module-level alias in app.py
    monkeypatch.setattr(appmod._bos_orders, "find_order_by_external_ref",
                        lambda *a, **k: None)
    monkeypatch.setattr(appmod._bos_orders, "set_order_stripe_pi",
                        lambda *a, **k: None)

    # Silence points + referral settlers
    monkeypatch.setattr(appmod, "_settle_order_points", lambda *a, **k: None)
    monkeypatch.setattr(appmod, "_settle_referral", lambda *a, **k: None)

    # Silence _pp.record_dispensary_order
    monkeypatch.setattr(appmod._pp, "record_dispensary_order", lambda *a, **k: None)


# ── Test 1: paid client kind → earn_dropship_margin called correctly ──────────

def test_paid_client_kind_credits_margin(monkeypatch):
    """Paid session with kind='client': earn_dropship_margin called once with
    (practitioner_id='p1', margin_cents=1340, qbo_invoice_id='INV')."""

    _stub_out_side_effects(monkeypatch)

    # Patch stripe_pay.get_session at the module level so the local import
    # inside begin_checkout_return picks it up.
    monkeypatch.setattr(
        _stripe_pay_mod, "get_session",
        lambda sid: _make_session("client"),
    )

    calls = []
    monkeypatch.setattr(
        _wallet_mod, "earn_dropship_margin",
        lambda pid, margin_cents, *, qbo_invoice_id, ref=None:
            calls.append({"pid": pid, "margin_cents": margin_cents,
                          "qbo_invoice_id": qbo_invoice_id}) or 1340,
    )

    appmod.app.config["TESTING"] = True
    client = appmod.app.test_client()
    resp = client.get("/begin/checkout-return?session_id=cs_test123")

    # Redirect must still happen
    assert resp.status_code in (301, 302, 303, 307, 308)

    # earn_dropship_margin was called exactly once
    assert len(calls) == 1, f"expected 1 call, got {calls}"
    assert calls[0]["pid"] == "p1"
    assert calls[0]["margin_cents"] == 1340   # int, not string
    assert calls[0]["qbo_invoice_id"] == "INV"


# ── Test 2: non-client kind → earn_dropship_margin NOT called ────────────────

def test_non_client_kind_does_not_credit_margin(monkeypatch):
    """Paid session with kind='retail': earn_dropship_margin must NOT be called."""

    _stub_out_side_effects(monkeypatch)

    monkeypatch.setattr(
        _stripe_pay_mod, "get_session",
        lambda sid: _make_session("retail"),
    )

    calls = []
    monkeypatch.setattr(
        _wallet_mod, "earn_dropship_margin",
        lambda pid, margin_cents, *, qbo_invoice_id, ref=None:
            calls.append(True) or 0,
    )

    appmod.app.config["TESTING"] = True
    client = appmod.app.test_client()
    resp = client.get("/begin/checkout-return?session_id=cs_test456")

    assert resp.status_code in (301, 302, 303, 307, 308)
    assert calls == [], "earn_dropship_margin must NOT be called for non-client kind"


# ── Test 3: earn_dropship_margin raises → redirect still happens ──────────────

def test_earn_dropship_margin_raise_does_not_break_redirect(monkeypatch):
    """If earn_dropship_margin raises, the handler must still redirect (never 500)."""

    _stub_out_side_effects(monkeypatch)

    monkeypatch.setattr(
        _stripe_pay_mod, "get_session",
        lambda sid: _make_session("client"),
    )

    def _boom(pid, margin_cents, *, qbo_invoice_id, ref=None):
        raise RuntimeError("db exploded")

    monkeypatch.setattr(_wallet_mod, "earn_dropship_margin", _boom)

    appmod.app.config["TESTING"] = True
    client = appmod.app.test_client()
    resp = client.get("/begin/checkout-return?session_id=cs_test789")

    # Must redirect — never a 500
    assert resp.status_code in (301, 302, 303, 307, 308)
