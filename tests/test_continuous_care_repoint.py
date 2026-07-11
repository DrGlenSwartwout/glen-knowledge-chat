import sqlite3
import pytest


@pytest.fixture
def client(monkeypatch, tmp_path):
    import app as appmod
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    appmod._init_auth_tables()
    monkeypatch.setattr(appmod, "CONSOLE_SECRET", "test-secret")
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client(), appmod


def test_existing_continuous_care_checkout_unchanged(client, monkeypatch):
    c, appmod = client
    monkeypatch.setattr(appmod, "CONTINUOUS_CARE_MONTHLY_ENABLED", True)
    monkeypatch.setattr(appmod, "_STRIPE_ACTIVE", True)
    from dashboard import stripe_pay
    seen = {}
    def _fake(amount, **kw):
        seen["amount"] = amount
        seen["metadata"] = kw.get("metadata")
        return {"url": "https://stripe.test/existing"}
    monkeypatch.setattr(stripe_pay, "create_checkout_session", _fake)
    r = c.post("/continuous-care/checkout", json={"email": "a@x.com", "term_months": 12})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["url"] == "https://stripe.test/existing"
    assert seen["metadata"]["kind"] == "continuous_care_monthly"
    assert seen["metadata"]["term_months"] == "12"


def test_existing_continuous_care_checkout_invalid(client, monkeypatch):
    c, appmod = client
    monkeypatch.setattr(appmod, "CONTINUOUS_CARE_MONTHLY_ENABLED", True)
    monkeypatch.setattr(appmod, "_STRIPE_ACTIVE", True)
    r = c.post("/continuous-care/checkout", json={"email": "", "term_months": 12})
    assert r.get_json() == {"ok": False, "error": "invalid"}


def _seed_portal(appmod, email, name="Test"):
    from dashboard import client_portal as cp
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cp.init_client_portal_table(cx)
        token, _id = cp.upsert_portal(cx, email, name, {})
    return token


def test_cc_wrapper_404_when_flag_off(client, monkeypatch):
    c, appmod = client
    monkeypatch.delenv("PROGRAM_PAID_LIVE_ENABLED", raising=False)
    r = c.post("/portal/offer/continuous-care/checkout?token=whatever")
    assert r.status_code == 404


def test_cc_wrapper_returns_stripe_url(client, monkeypatch):
    c, appmod = client
    monkeypatch.setenv("PROGRAM_PAID_LIVE_ENABLED", "1")
    monkeypatch.setattr(appmod, "CONTINUOUS_CARE_MONTHLY_ENABLED", True)
    monkeypatch.setattr(appmod, "_STRIPE_ACTIVE", True)
    from dashboard import stripe_pay
    captured = {}
    def _fake(amount, **kw):
        captured["metadata"] = kw.get("metadata")
        return {"url": "https://stripe.test/cc"}
    monkeypatch.setattr(stripe_pay, "create_checkout_session", _fake)
    tok = _seed_portal(appmod, "cc@x.com")
    r = c.post(f"/portal/offer/continuous-care/checkout?token={tok}")
    assert r.status_code == 200
    assert r.get_json()["stripe_url"] == "https://stripe.test/cc"
    assert captured["metadata"]["term_months"] == "12"
    assert captured["metadata"]["email"] == "cc@x.com"


def test_cc_wrapper_404_for_bad_token(client, monkeypatch):
    c, appmod = client
    monkeypatch.setenv("PROGRAM_PAID_LIVE_ENABLED", "1")
    monkeypatch.setattr(appmod, "CONTINUOUS_CARE_MONTHLY_ENABLED", True)
    monkeypatch.setattr(appmod, "_STRIPE_ACTIVE", True)
    r = c.post("/portal/offer/continuous-care/checkout?token=nope")
    assert r.status_code == 404
