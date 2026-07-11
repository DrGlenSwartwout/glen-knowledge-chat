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
