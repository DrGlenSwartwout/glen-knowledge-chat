# tests/test_membership_products_fulfill.py
"""Fulfillment tests for the membership-product checkout (month / year_monthly /
year_prepay tiers from dashboard.membership_products): _fulfill_membership_product.

Mirrors tests/test_continuous_care_monthly.py fixture style: tmp-file sqlite via
monkeypatched LOG_DB (no shared-LOG_DB pollution), Stripe faked by monkeypatching
dashboard.stripe_pay (app.py's module-level `stripe_pay` alias is the same module
object, so appmod.stripe_pay.* and dashboard.stripe_pay.* patches are equivalent).
QBO faked by monkeypatching dashboard.qbo_billing so the one-time-tier booking
never hits the network.
"""
import importlib, sys, os, sqlite3, datetime
import pytest


def _load_app():
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app import failed: {e}")


@pytest.fixture
def appmod(monkeypatch, tmp_path):
    app = _load_app()
    db = str(tmp_path / "t.db")
    monkeypatch.setattr(app, "LOG_DB", db, raising=False)
    monkeypatch.setattr(app, "PUBLIC_BASE_URL", "https://illtowell.com", raising=False)
    from dashboard import subscriptions as subs
    cx = sqlite3.connect(db)
    subs.init_subscriptions_table(cx)
    for name in dir(subs):
        if name.startswith("migrate_add"):
            try: getattr(subs, name)(cx)
            except Exception: pass
    app.init_membership_tables(cx)
    cx.close()
    # no-op the side effects
    for fn in ("_ingest_order", "_member_join_welcome"):
        if hasattr(app, fn): monkeypatch.setattr(app, fn, lambda *a, **k: None, raising=False)
    # stub QBO so one-time booking does not hit the network
    import dashboard.qbo_billing as qb
    monkeypatch.setattr(qb, "find_or_create_customer", lambda *a, **k: {"Id": "1"})
    monkeypatch.setattr(qb, "create_invoice", lambda *a, **k: {"Id": "inv1"})
    monkeypatch.setattr(qb, "record_payment", lambda *a, **k: {"Id": "pay1"})
    return app


def _mock_stripe(appmod, monkeypatch, *, tier, with_card=True):
    monkeypatch.setattr(appmod.stripe_pay, "get_session", lambda sid: {
        "id": sid, "payment_status": "paid",
        "metadata": {"kind": "membership_product", "tier": tier, "email": "a@x.com"},
        "payment_intent": "pi_1", "customer": ("cus_1" if with_card else None),
    })
    monkeypatch.setattr(appmod.stripe_pay, "get_payment_intent", lambda pi: {
        "status": "succeeded",
        "customer": ("cus_1" if with_card else None),
        "payment_method": ("pm_1" if with_card else None)})


def test_one_time_month_grants_paid_member(appmod, monkeypatch):
    _mock_stripe(appmod, monkeypatch, tier="month")
    assert appmod._fulfill_membership_product("cs_1") == "ok"
    assert appmod._is_paid_member("a@x.com") is True
    from dashboard import membership_products as mp
    cx = sqlite3.connect(appmod.LOG_DB); cx.row_factory = sqlite3.Row
    assert mp.owns_group(cx, "a@x.com") is True


def test_year_monthly_creates_capped_sub(appmod, monkeypatch):
    _mock_stripe(appmod, monkeypatch, tier="year_monthly")
    assert appmod._fulfill_membership_product("cs_2") == "ok"
    from dashboard import subscriptions as subs
    cx = sqlite3.connect(appmod.LOG_DB); cx.row_factory = sqlite3.Row
    rows = subs.active_memberships_by_email(cx, "a@x.com")
    assert rows and rows[0]["term_charges_total"] == 12
    assert rows[0]["order_count"] == 1  # month 1 charged at checkout
    assert appmod.membership_category("a@x.com") == "full"


def test_year_monthly_requires_card(appmod, monkeypatch):
    _mock_stripe(appmod, monkeypatch, tier="year_monthly", with_card=False)
    assert appmod._fulfill_membership_product("cs_3") == "no_card"
    cx = sqlite3.connect(appmod.LOG_DB)
    n = cx.execute(
        "SELECT COUNT(*) FROM membership_product_grants WHERE session_id=?",
        ("cs_3",)).fetchone()[0]
    assert n == 0  # claim row unwound, so a retry is not permanently blocked


def test_idempotent_replay(appmod, monkeypatch):
    _mock_stripe(appmod, monkeypatch, tier="month")
    appmod._fulfill_membership_product("cs_4")
    appmod._fulfill_membership_product("cs_4")  # replay
    cx = sqlite3.connect(appmod.LOG_DB)
    n = cx.execute("SELECT COUNT(*) FROM memberships WHERE email='a@x.com'").fetchone()[0]
    assert n == 1


def test_duplicate_member_no_second_sub(appmod, monkeypatch):
    _mock_stripe(appmod, monkeypatch, tier="year_monthly")
    appmod._fulfill_membership_product("cs_5")
    monkeypatch.setattr(appmod.stripe_pay, "get_session", lambda sid: {
        "id": sid, "payment_status": "paid",
        "metadata": {"kind": "membership_product", "tier": "year_monthly", "email": "a@x.com"},
        "payment_intent": "pi_2", "customer": "cus_1"})
    assert appmod._fulfill_membership_product("cs_6") == "duplicate_member"
    cx = sqlite3.connect(appmod.LOG_DB)
    n = cx.execute(
        "SELECT COUNT(*) FROM subscriptions WHERE email=?", ("a@x.com",)).fetchone()[0]
    assert n == 1  # no double sub — the duplicate-member guard held
