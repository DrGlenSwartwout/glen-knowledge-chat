# tests/test_membership_products_e2e.py
"""End-to-end coverage across all three membership-product tiers (month /
year_monthly / year_prepay): checkout -> fulfill -> entitlement.

Fixture style mirrors tests/test_membership_products_fulfill.py (tmp-file
sqlite via monkeypatched LOG_DB, subscriptions + membership table init, QBO
and side-effect stubs) plus tests/test_membership_products_checkout.py
(flag + Stripe-active flags, checkout POST + metadata capture pattern).
"""
import importlib, sys, os, sqlite3
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
    monkeypatch.setattr(app, "_STRIPE_ACTIVE", True, raising=False)
    monkeypatch.setattr(app, "MEMBERSHIP_PRODUCTS_ENABLED", True, raising=False)
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


def _checkout(appmod, monkeypatch, *, email, tier):
    """POST /membership/checkout, capturing the metadata passed to the mocked
    stripe_pay.create_checkout_session. Returns the captured kwargs dict."""
    cap = {}
    def fake_sess(amount, **kw):
        cap["amount"] = amount
        cap["kw"] = kw
        return {"id": f"cs_{email}", "url": "https://checkout.stripe.com/x"}
    monkeypatch.setattr(appmod.stripe_pay, "create_checkout_session", fake_sess)
    r = appmod.app.test_client().post(
        "/membership/checkout", json={"email": email, "tier": tier})
    assert r.status_code == 200, r.get_data(as_text=True)
    assert r.get_json()["ok"] is True
    assert cap["kw"]["metadata"]["kind"] == "membership_product"
    assert cap["kw"]["metadata"]["tier"] == tier
    return cap


def _mock_fulfill_session(appmod, monkeypatch, *, tier, email, with_card=True):
    """Stub the Stripe session/payment-intent lookups that
    _fulfill_membership_product drives, mirroring the checkout metadata."""
    monkeypatch.setattr(appmod.stripe_pay, "get_session", lambda sid: {
        "id": sid, "payment_status": "paid",
        "metadata": {"kind": "membership_product", "tier": tier, "email": email},
        "payment_intent": "pi_1", "customer": ("cus_1" if with_card else None),
    })
    monkeypatch.setattr(appmod.stripe_pay, "get_payment_intent", lambda pi: {
        "status": "succeeded",
        "customer": ("cus_1" if with_card else None),
        "payment_method": ("pm_1" if with_card else None)})


@pytest.mark.parametrize("tier,email", [
    ("month", "e2e-month@x.com"),
    ("year_monthly", "e2e-year-monthly@x.com"),
    ("year_prepay", "e2e-year-prepay@x.com"),
])
def test_full_flow_checkout_to_entitlement(appmod, monkeypatch, tier, email):
    # 1. checkout -> capture + assert metadata
    _checkout(appmod, monkeypatch, email=email, tier=tier)

    # 2. fulfill -> success status
    _mock_fulfill_session(appmod, monkeypatch, tier=tier, email=email, with_card=True)
    session_id = f"cs_{email}"
    result = appmod._fulfill_membership_product(session_id)
    assert result == "ok", f"expected ok, got {result!r} for tier={tier}"

    # 3. entitlement: paid member + group ownership
    assert appmod._is_paid_member(email) is True
    from dashboard import membership_products as mp
    cx = sqlite3.connect(appmod.LOG_DB)
    cx.row_factory = sqlite3.Row
    assert mp.owns_group(cx, email) is True

    # 4. year_monthly: capped-term subscription row
    if tier == "year_monthly":
        from dashboard import subscriptions as subs
        rows = subs.active_memberships_by_email(cx, email)
        assert rows, "expected an active subscription row for year_monthly"
        assert rows[0]["term_charges_total"] == 12
        assert rows[0]["order_count"] == 1
    cx.close()
