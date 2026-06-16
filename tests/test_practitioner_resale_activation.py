"""Tests for the resale-license activation gate: a logged-in coach submits a
resale license to request reselling activation (POST /api/practitioner/resale-apply),
which marks their EXISTING record pending for the /admin/wholesale approve flow.

Mirrors the app-route test style in test_cert_student.py."""

import pytest


@pytest.fixture
def client(monkeypatch):
    import app as appmod
    appmod.app.config["TESTING"] = True
    # Mailer is best-effort; make it a no-op so tests never send.
    monkeypatch.setattr(appmod, "_send_full_report_email",
                        lambda *a, **kw: None)
    return appmod.app.test_client(), appmod


def test_route_not_signed_in(client, monkeypatch):
    c, appmod = client
    monkeypatch.setattr(appmod, "_practitioner_session_pid", lambda: None)
    r = c.post("/api/practitioner/resale-apply",
               json={"resale_license_number": "RL123", "license_state": "TX"})
    assert r.status_code == 401


def test_route_missing_license(client, monkeypatch):
    c, appmod = client
    monkeypatch.setattr(appmod, "_practitioner_session_pid", lambda: "pid-1")
    r = c.post("/api/practitioner/resale-apply", json={"license_state": "TX"})
    assert r.status_code == 400


def test_route_ok_marks_pending(client, monkeypatch):
    c, appmod = client
    monkeypatch.setattr(appmod, "_practitioner_session_pid", lambda: "pid-1")
    calls = []
    monkeypatch.setattr(appmod._pp, "submit_resale_for_pid",
                        lambda *a, **kw: calls.append((a, kw)))
    r = c.post("/api/practitioner/resale-apply",
               json={"resale_license_number": "RL123", "license_state": "TX"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["status"] == "pending"
    # helper called with the pid + license (+ state)
    assert len(calls) == 1
    args, _kw = calls[0]
    assert args[0] == "pid-1"
    assert args[1] == "RL123"
    assert "TX" in args[2:]
