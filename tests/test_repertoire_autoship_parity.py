"""Member autoship / founding ship-charge <-> portal reorder pricing parity.

Two `_price_cart(...)` callers (cron_charge_subscriptions, and
_ship_founding_reservation) price recurring/ship charges but historically
didn't pass `email=`, so a member's autoship billed the regular/volume rate
while their manual portal reorder billed the repertoire ($50) rate. Both
call sites now pass `email=<the sub's email>` so REPERTOIRE_ENABLED +
_is_paid_member(email) resolves the same repertoire pricing on autoship as
on a manual reorder (dashboard/repertoire.py, wired into _price_cart per
tests/test_repertoire_wiring.py).
"""
import sqlite3
from datetime import datetime, timedelta

import app as appmod
from dashboard import subscriptions as subs
from dashboard import repertoire as rep


def _seed_active_membership(db, email, *, source="founding"):
    """Insert an active (future-expiring) memberships grant row so
    _is_paid_member(email) is True (non-biofield_trial source -> not
    classified as 'trial')."""
    expires = (datetime.utcnow() + timedelta(days=30)).isoformat() + "Z"
    with sqlite3.connect(db) as cx:
        cx.execute(
            "INSERT INTO memberships (id, email, granted_at, expires_at, granted_by, source) "
            "VALUES (?,?,?,?,?,?)",
            ("mem_parity_1", email, datetime.utcnow().isoformat() + "Z", expires,
             "test", source),
        )
        cx.commit()


# ── _ship_founding_reservation: real _price_cart, real repertoire lookup ──────

def _run_ship_and_capture_pricing(monkeypatch, tmp_path, *, repertoire_enabled, email):
    """Shared harness: seed a founding reservation for a member with a
    repertoire SKU, run the real (unmocked) _price_cart via
    _ship_founding_reservation, and capture the pc dict it computed (the
    invoice-facing discount_cents lives there — qbo_lines always carries
    LIST price by design; cart-level discounts ride discount_cents, see
    _price_cart's docstring)."""
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(appmod, "LOG_DB", db)
    monkeypatch.setattr(appmod, "REPERTOIRE_ENABLED", repertoire_enabled)

    cx = sqlite3.connect(db)
    cx.row_factory = sqlite3.Row
    subs.init_subscriptions_table(cx)
    subs.migrate_add_founding_columns(cx)
    appmod.init_membership_tables(cx)
    rep.init_repertoire_table(cx)
    cx.commit()

    _seed_active_membership(db, email)
    rep.add_skus(cx, email, ["neuro-magnesium"])

    sid = subs.create_founding_reservation(
        cx, email=email, stripe_customer_id="cus_1",
        stripe_payment_method_id="pm_1",
        items=[{"slug": "neuro-magnesium", "qty": 1}],
        ship_address={"state": "HI"},
        founding_slug="neuro-magnesium",
    )
    sub = subs.get(cx, sid)

    real_price_cart = appmod._price_cart
    captured = []

    def _spy(cart, **k):
        pc = real_price_cart(cart, **k)
        captured.append(pc)
        return pc

    monkeypatch.setattr(appmod, "_price_cart", _spy)
    monkeypatch.setattr(appmod.stripe_pay, "charge_off_session",
                        lambda *a, **k: {"status": "succeeded", "id": "pi_1"})
    monkeypatch.setattr(appmod.qb, "find_or_create_customer", lambda *a, **k: {"Id": "C1"})
    monkeypatch.setattr(appmod.qb, "create_invoice",
                        lambda cust, lines, **kw: {"Id": "INV1", "TotalAmt": 50.0})
    monkeypatch.setattr(appmod, "_ingest_order", lambda **kw: None)

    res = appmod._ship_founding_reservation(cx, sub)
    return res, captured[0]


def test_ship_founding_reservation_applies_repertoire_price(monkeypatch, tmp_path):
    """End-to-end (real _price_cart, not mocked): a paid member with a
    repertoire SKU gets the repertoire rate on their founding ship-charge,
    reflected as a non-zero cart discount_cents (the value forwarded to
    qb.create_invoice's discount_cents=), same as a manual reorder through
    _price_cart(email=...) (test_repertoire_wiring.py)."""
    res, pc = _run_ship_and_capture_pricing(
        monkeypatch, tmp_path, repertoire_enabled=True,
        email="foundingrepertoire@x.com")

    assert res["charged"] is True
    assert pc["discount_cents"] > 0, (
        f"expected a repertoire discount on the founding ship-charge, got pc={pc}")


def test_ship_founding_reservation_flag_off_no_discount(monkeypatch, tmp_path):
    """Control: same setup with REPERTOIRE_ENABLED off must bill list price
    (discount_cents == 0) - confirms the parity fix is flag-gated, not a
    blanket discount."""
    res, pc = _run_ship_and_capture_pricing(
        monkeypatch, tmp_path, repertoire_enabled=False,
        email="foundingflagoff@x.com")

    assert res["charged"] is True
    assert pc["discount_cents"] == 0


# ── cron_charge_subscriptions: focused - the _price_cart call now carries email ──
# Full end-to-end (real pricing engine) coverage for this branch would need to
# also thread through tier_pct/points/discount-cap paths already covered by
# tests/test_subscriptions_cron.py, which mocks _price_cart wholesale (its
# seeded slug "x" isn't a real product). Rather than duplicate that harness,
# this asserts directly on the call the cron makes: it now forwards
# email=sub["email"], the same parameter test_repertoire_wiring.py proves
# resolves repertoire pricing inside _price_cart.

def test_cron_charge_subscriptions_price_cart_call_carries_email(monkeypatch):
    """The regular product-subscription branch of cron_charge_subscriptions
    now passes email=sub['email'] to _price_cart (parity with portal
    reorder), so REPERTOIRE_ENABLED + _is_paid_member(email) can resolve the
    member's repertoire SKU set for their autoship."""
    import os

    def _cron_secret():
        return os.environ.get("CRON_SECRET") or os.environ.get("CONSOLE_SECRET", "test-secret")

    monkeypatch.setenv("SUBSCRIPTIONS_ENABLED", "true")
    if not (os.environ.get("CRON_SECRET") or os.environ.get("CONSOLE_SECRET")):
        monkeypatch.setenv("CONSOLE_SECRET", _cron_secret())

    calls = []

    def _spy_price_cart(cart, **k):
        calls.append(k)
        return {
            "priced": {"subtotal_cents": 5000, "total_cents": 5000, "get_cents": 0},
            "qbo_lines": [{"name": "X", "amount": 50.0, "qty": 1}],
            "items_rec": [{"name": "X", "qty": 1, "desc": "X"}],
            "discount_cents": 0,
            "points_redeemed_cents": 0,
            "shipping_cents": 1265,
        }

    monkeypatch.setattr(appmod, "_price_cart", _spy_price_cart)
    monkeypatch.setattr(appmod.qb, "find_or_create_customer", lambda *a, **k: {"Id": "C1"})
    monkeypatch.setattr(appmod.qb, "create_invoice",
                        lambda *a, **k: {"Id": "INV1", "TotalAmt": 50.0})
    monkeypatch.setattr(appmod.stripe_pay, "charge_off_session",
                        lambda *a, **k: {"status": "succeeded", "id": "pi_x"})
    monkeypatch.setattr(appmod, "_send_subscription_email", lambda *a, **k: ("smtp", None))
    monkeypatch.setattr(appmod, "_ingest_order", lambda **kw: None)

    email = "autoship-parity@example.com"
    cx = sqlite3.connect(appmod.LOG_DB)
    cx.row_factory = sqlite3.Row
    subs.init_subscriptions_table(cx)
    subs.migrate_add_failed_count(cx)
    cx.execute("DELETE FROM subscriptions WHERE email=?", (email,))
    cx.commit()
    sid = subs.create(
        cx, email=email, stripe_customer_id="cus_parity",
        stripe_payment_method_id="pm_parity",
        items=[{"slug": "x", "qty": 1}], cadence_months=1,
        ship_address={"state": "CA"}, next_charge_date="2000-01-01",
    )
    cx.commit()

    try:
        c = appmod.app.test_client()
        r = c.post("/api/cron/charge-subscriptions",
                   headers={"X-Cron-Secret": _cron_secret()})
        assert r.status_code == 200, r.data

        matching = [k for k in calls if k.get("email") == email]
        assert matching, f"no _price_cart call carried email={email!r}; calls={calls}"
    finally:
        cx.execute("DELETE FROM subscriptions WHERE email=?", (email,))
        cx.commit()
        cx.close()
