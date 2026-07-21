import json
import pytest
import app as app_module


@pytest.fixture
def client():
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def test_client_360_returns_bundle_shape(client, monkeypatch):
    # Gate open: _bos_actor returns a truthy actor (independent of rbac internals).
    monkeypatch.setattr(app_module, "_bos_actor", lambda: object())
    fake = {"person": {"name": "Test"}, "clinical": {"active": [], "suggested": []},
            "tests": [], "invoices": {"total_paid_cents": 0, "open_balance_cents": 0,
                                      "orders": [], "fmp": []},
            "comms": [], "process": {"source": None, "order_id": None, "stages": []}}
    monkeypatch.setattr(app_module.client_360, "bundle", lambda cx, email, **k: fake)
    r = client.get("/api/console/client-360?email=a@b.com")
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    assert data["person"]["name"] == "Test"
    assert "invoices" in data and "process" in data


def test_client_360_requires_auth(client, monkeypatch):
    # Gate closed: _bos_actor returns None.
    monkeypatch.setattr(app_module, "_bos_actor", lambda: None)
    r = client.get("/api/console/client-360?email=a@b.com")
    assert r.status_code == 401
    assert r.get_json()["ok"] is False
