# tests/test_membership_products_checkout.py — mirrors tests/test_continuous_care_monthly.py
import importlib, sys, os
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
    monkeypatch.setattr(app, "LOG_DB", str(tmp_path / "t.db"), raising=False)
    monkeypatch.setattr(app, "PUBLIC_BASE_URL", "https://illtowell.com", raising=False)
    monkeypatch.setattr(app, "_STRIPE_ACTIVE", True, raising=False)
    monkeypatch.setattr(app, "MEMBERSHIP_PRODUCTS_ENABLED", True, raising=False)
    return app

def test_month_tier_builds_one_time_session(appmod, monkeypatch):
    cap = {}
    def fake_sess(amount, **kw):
        cap["amount"] = amount; cap["kw"] = kw
        return {"id": "cs_test", "url": "https://checkout.stripe.com/x"}
    monkeypatch.setattr(appmod.stripe_pay, "create_checkout_session", fake_sess)
    r = appmod.app.test_client().post("/membership/checkout",
                                      json={"email": "a@x.com", "tier": "month"})
    assert r.status_code == 200 and r.get_json()["ok"] is True
    assert cap["amount"] == 9900
    assert cap["kw"]["save_card"] is False
    assert cap["kw"]["metadata"]["kind"] == "membership_product"
    assert cap["kw"]["metadata"]["tier"] == "month"

def test_year_monthly_tier_vaults_card(appmod, monkeypatch):
    cap = {}
    def fake_sess(amount, **kw):
        cap["amount"] = amount; cap["kw"] = kw
        return {"id": "cs_test", "url": "https://checkout.stripe.com/x"}
    monkeypatch.setattr(appmod.stripe_pay, "create_checkout_session", fake_sess)
    r = appmod.app.test_client().post("/membership/checkout",
                                      json={"email": "a@x.com", "tier": "year_monthly"})
    assert r.status_code == 200
    assert cap["amount"] == 9900
    assert cap["kw"]["save_card"] is True

def test_year_prepay_amount(appmod, monkeypatch):
    cap = {}
    monkeypatch.setattr(appmod.stripe_pay, "create_checkout_session",
                        lambda amount, **kw: (cap.update(amount=amount, kw=kw)
                                              or {"id": "cs", "url": "u"}))
    appmod.app.test_client().post("/membership/checkout",
                                  json={"email": "a@x.com", "tier": "year_prepay"})
    assert cap["amount"] == 99000

def test_unknown_tier_rejected(appmod):
    r = appmod.app.test_client().post("/membership/checkout",
                                      json={"email": "a@x.com", "tier": "nope"})
    assert r.status_code == 400

def test_flag_off_returns_404(appmod, monkeypatch):
    monkeypatch.setattr(appmod, "MEMBERSHIP_PRODUCTS_ENABLED", False, raising=False)
    r = appmod.app.test_client().post("/membership/checkout",
                                      json={"email": "a@x.com", "tier": "month"})
    assert r.status_code == 404
