"""Endpoint-level tests for POST /api/portal/<token>/health-profile.

Covers: flag OFF -> 404, a valid field_id/value write-back persisting under
PORTAL_HEALTH_PROFILE_ENABLED, the malformed non-dict `answers` payload
returning 400 (not a raw 500 from intake.save_self_edit), and identity being
resolved from the portal TOKEN rather than any email in the request body.

Follows the tests/test_cert_portal_routes.py fixture pattern: swap LOG_DB to
a tmp sqlite file so tests never touch the dev db, and use app.test_client().
"""
import os
os.environ.setdefault("OPENAI_API_KEY", "sk-dummy")

import sqlite3
import pytest


@pytest.fixture
def client(monkeypatch, tmp_path):
    import app as appmod
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client(), appmod


def _mint_portal(appmod, email):
    from dashboard import client_portal as _cp
    with sqlite3.connect(appmod.LOG_DB) as cx:
        _cp.init_client_portal_table(cx)
        token, _pid = _cp.upsert_portal(cx, email, "Test Client", {})
    return token


def test_flag_off_returns_404(client, monkeypatch):
    c, appmod = client
    monkeypatch.setattr(appmod, "_PORTAL_HEALTH_PROFILE_ENABLED", False)
    r = c.post("/api/portal/whatever-token/health-profile",
                json={"field_id": "sleep", "value": "ok"})
    assert r.status_code == 404


def test_valid_field_id_value_persists(client, monkeypatch):
    c, appmod = client
    monkeypatch.setattr(appmod, "_PORTAL_HEALTH_PROFILE_ENABLED", True)
    token = _mint_portal(appmod, "a@b.com")
    r = c.post(f"/api/portal/{token}/health-profile",
                json={"field_id": "sleep", "value": "improving"})
    assert r.status_code == 200
    assert r.get_json()["ok"] is True
    from dashboard import intake
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        row = intake.get_response(cx, "a@b.com")
    assert row["answers"]["sleep"] == "improving"


def test_non_dict_answers_returns_400_not_500(client, monkeypatch):
    c, appmod = client
    monkeypatch.setattr(appmod, "_PORTAL_HEALTH_PROFILE_ENABLED", True)
    token = _mint_portal(appmod, "a@b.com")
    r = c.post(f"/api/portal/{token}/health-profile",
                json={"answers": ["terrain", "sleep"]})
    assert r.status_code == 400
    assert r.get_json()["ok"] is False


def test_unknown_token_still_400_before_500_on_bad_payload(client, monkeypatch):
    """The 400 for a malformed payload must fire even for a token that doesn't
    resolve to any portal — the isinstance guard runs before the DB is opened."""
    c, appmod = client
    monkeypatch.setattr(appmod, "_PORTAL_HEALTH_PROFILE_ENABLED", True)
    r = c.post("/api/portal/not-a-real-token/health-profile",
                json={"answers": ["terrain", "sleep"]})
    assert r.status_code == 400
    assert r.get_json()["ok"] is False


def test_identity_from_token_not_body(client, monkeypatch):
    c, appmod = client
    monkeypatch.setattr(appmod, "_PORTAL_HEALTH_PROFILE_ENABLED", True)
    token = _mint_portal(appmod, "real@b.com")
    r = c.post(f"/api/portal/{token}/health-profile",
                json={"field_id": "sleep", "value": "great", "email": "attacker@evil.com"})
    assert r.status_code == 200
    from dashboard import intake
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        real = intake.get_response(cx, "real@b.com")
        attacker = intake.get_response(cx, "attacker@evil.com")
    assert real["answers"]["sleep"] == "great"
    assert attacker is None
