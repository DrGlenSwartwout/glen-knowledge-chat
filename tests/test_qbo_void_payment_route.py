"""POST /api/qbo/payment/<id>/void — owner-only, QBO-only, ledger untouched.

This endpoint DELETES a QBO Payment (QBO has no void-in-place for payments), so the
gate matters more than usual: unauthorized callers, non-numeric ids, and unconfirmed
requests must never reach qbo_billing.void_payment.

Skipped automatically if app fails to import (repo convention -- app.py needs live
Pinecone at import, so this runs locally under real secrets, not in secretless CI).
"""
import pytest


@pytest.fixture
def client(monkeypatch):
    try:
        import app as appmod
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"app not importable: {e}")
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client(), appmod


def _auth(appmod, monkeypatch, ok=True):
    monkeypatch.setattr(appmod, "_qbo_auth_ok", lambda: ok)


def test_unauthorized_is_rejected(client, monkeypatch):
    c, appmod = client
    _auth(appmod, monkeypatch, ok=False)
    called = []
    monkeypatch.setattr("dashboard.qbo_billing.void_payment", lambda t: called.append(t))
    r = c.post("/api/qbo/payment/24770/void", json={"confirmed": True})
    assert r.status_code == 401
    assert called == []  # never reached QBO


def test_unconfirmed_is_rejected(client, monkeypatch):
    c, appmod = client
    _auth(appmod, monkeypatch)
    called = []
    monkeypatch.setattr("dashboard.qbo_billing.void_payment", lambda t: called.append(t))
    r = c.post("/api/qbo/payment/24770/void", json={})
    assert r.status_code == 400
    assert "confirmed" in r.get_json()["error"]
    assert called == []  # the speed bump held


def test_non_numeric_id_is_rejected(client, monkeypatch):
    c, appmod = client
    _auth(appmod, monkeypatch)
    called = []
    monkeypatch.setattr("dashboard.qbo_billing.void_payment", lambda t: called.append(t))
    r = c.post("/api/qbo/payment/not-an-id/void", json={"confirmed": True})
    assert r.status_code == 400
    assert called == []


def test_confirmed_deletes_and_reports_ledger_untouched(client, monkeypatch):
    c, appmod = client
    _auth(appmod, monkeypatch)
    called = []
    monkeypatch.setattr("dashboard.qbo_billing.void_payment", lambda t: called.append(str(t)))
    r = c.post("/api/qbo/payment/24770/void", json={"confirmed": True})
    assert r.status_code == 200
    d = r.get_json()
    assert d["ok"] is True
    assert d["txn_id"] == "24770"
    assert d["action"] == "deleted"       # not "voided" -- QBO deletes payments
    assert d["ledger_touched"] is False   # the whole point: ledger figures must not move
    assert called == ["24770"]


def test_qbo_failure_surfaces_as_500_not_silent_success(client, monkeypatch):
    c, appmod = client
    _auth(appmod, monkeypatch)

    def _boom(_t):
        raise RuntimeError("payment 999 not found")

    monkeypatch.setattr("dashboard.qbo_billing.void_payment", _boom)
    r = c.post("/api/qbo/payment/999/void", json={"confirmed": True})
    assert r.status_code == 500
    assert r.get_json()["ok"] is False
    assert "not found" in r.get_json()["error"]
