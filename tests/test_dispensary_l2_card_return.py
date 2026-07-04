"""Integration guard: a CARD dispensary sale credits the upline L2 exactly once
through the REAL /begin/checkout-return handler (not by calling _settle_order_points
directly). This locks in that dispensary card orders reach the settler via the
kind-agnostic `if inv and cid:` block, and guards against a regression that drops
customer_id from the metadata or reorders the settlement blocks."""
import importlib
import sqlite3

import dashboard.stripe_pay as sp_mod
import dashboard.qbo_billing as qbo_mod
import dashboard.wallet as wallet_mod
import dashboard.practitioner_portal as pp_mod
from dashboard import referrals as rf, points


def _reload(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("REFERRALS", "true")
    monkeypatch.setenv("REFERRAL_TIER2_ENABLED", "true")
    monkeypatch.setenv("REFERRER_REWARD_PCT", "20")
    import app as appmod
    importlib.reload(appmod)
    appmod.app.config["TESTING"] = True
    return appmod


def test_card_dispensary_return_credits_upline_l2_once(monkeypatch, tmp_path):
    appmod = _reload(monkeypatch, tmp_path)

    # Seed: the paid dispensary order (source='dispensary', with practitioner_id) and
    # the doctor's own referral row (upline@x.com referred doc@x.com).
    with sqlite3.connect(appmod.LOG_DB) as cx:
        from dashboard import orders as o
        o.init_orders_table(cx)
        rf.init_tables(cx)
        points.init_points_table(cx)
        o.upsert_order(cx, source="dispensary", external_ref="INV1", email="pat@x.com",
                       total_cents=7000, shipping_cents=1300, practitioner_id="prac-1")
        rf.record_redemption(cx, "UP", "upline@x.com", "doc@x.com", "INV-DOC")
        cx.commit()

    # Mock the external surfaces the return handler touches.
    monkeypatch.setattr(sp_mod, "get_session", lambda sid: {
        "payment_status": "paid", "payment_intent": "pi_1",
        "amount_total": 7000,
        "metadata": {"kind": "client", "invoice_id": "INV1", "customer_id": "cus_1",
                     "practitioner_id": "prac-1", "margin_cents": "0",
                     "patient_email": "pat@x.com"}})
    monkeypatch.setattr(qbo_mod, "record_payment", lambda *a, **k: None)
    monkeypatch.setattr(wallet_mod, "earn_dropship_margin", lambda *a, **k: None)
    monkeypatch.setattr(pp_mod, "record_dispensary_order", lambda *a, **k: None)
    monkeypatch.setattr(pp_mod, "practitioner_email_by_id", lambda pid: "doc@x.com")

    c = appmod.app.test_client()
    r = c.get("/begin/checkout-return?session_id=sess_1")
    assert r.status_code in (200, 302)   # handler completes, then redirects to the buy page

    with sqlite3.connect(appmod.LOG_DB) as cx:
        # Assert on the L2 credit SPECIFICALLY (reason='referral_reward_l2'), so the
        # patient's orthogonal 5% buyer-earn (reason='earn', same 'rm' balance) doesn't
        # confound it. product 5700; L2 = 5700 * 20 // 200 = 570, to the doc's upline.
        assert points.earned_by_reason(cx, "upline@x.com", "referral_reward_l2") == 570
        assert points.earned_by_reason(cx, "doc@x.com", "referral_reward_l2") == 0    # practitioner (L1) never L2
        assert points.earned_by_reason(cx, "pat@x.com", "referral_reward_l2") == 0    # patient never L2

    # Idempotency: a second return (Stripe redirect refresh) must not double-credit L2.
    r2 = c.get("/begin/checkout-return?session_id=sess_1")
    assert r2.status_code in (200, 302)
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert points.earned_by_reason(cx, "upline@x.com", "referral_reward_l2") == 570
