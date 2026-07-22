"""Endpoint-level tests for the portal health-suggestions review queue:
GET /api/portal/<token>/health-suggestions and
POST /api/portal/<token>/health-suggestions/<sid>/resolve.

Follows the tests/test_health_profile_endpoint.py fixture pattern: swap LOG_DB
to a tmp sqlite file so tests never touch the dev db, and use app.test_client().
"""
import os
# Dummy keys so `import app` (which constructs OpenAI + Pinecone clients at import)
# succeeds under a secretless CI without doppler. The clients are constructed but
# never called by these endpoints under test (DB-only write path).
os.environ.setdefault("OPENAI_API_KEY", "sk-dummy")
os.environ.setdefault("PINECONE_API_KEY", "pc-dummy")

import sqlite3
import pytest


@pytest.fixture
def client(monkeypatch, tmp_path):
    import app as appmod
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    monkeypatch.setattr(appmod, "_PORTAL_HEALTH_PROFILE_ENABLED", True)
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client(), appmod


def _mint_portal(appmod, email):
    from dashboard import client_portal as _cp
    with sqlite3.connect(appmod.LOG_DB) as cx:
        _cp.init_client_portal_table(cx)
        token, _pid = _cp.upsert_portal(cx, email, "Test Client", {})
    return token


def _seed_suggestion(appmod, email, field_id, value, source="chat"):
    from dashboard import health_suggestions as hs
    with sqlite3.connect(appmod.LOG_DB) as cx:
        sid = hs.add_pending(cx, email, field_id, value, "mentioned in chat", source=source)
    return sid


def test_flag_off_returns_404_for_list(client, monkeypatch):
    c, appmod = client
    monkeypatch.setattr(appmod, "_PORTAL_HEALTH_PROFILE_ENABLED", False)
    r = c.get("/api/portal/whatever-token/health-suggestions")
    assert r.status_code == 404


def test_flag_off_returns_404_for_resolve(client, monkeypatch):
    c, appmod = client
    monkeypatch.setattr(appmod, "_PORTAL_HEALTH_PROFILE_ENABLED", False)
    r = c.post("/api/portal/whatever-token/health-suggestions/1/resolve",
               json={"action": "dismiss"})
    assert r.status_code == 404


def test_list_pending_for_token_email(client):
    c, appmod = client
    token = _mint_portal(appmod, "a@b.com")
    _seed_suggestion(appmod, "a@b.com", "terrain", 3)
    r = c.get(f"/api/portal/{token}/health-suggestions")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert len(body["suggestions"]) == 1
    assert body["suggestions"][0]["field_id"] == "terrain"


def test_confirm_writes_value_and_clears_pending(client):
    c, appmod = client
    token = _mint_portal(appmod, "a@b.com")
    sid = _seed_suggestion(appmod, "a@b.com", "terrain", 3)
    r = c.post(f"/api/portal/{token}/health-suggestions/{sid}/resolve",
               json={"action": "confirm"})
    assert r.status_code == 200
    assert r.get_json()["ok"] is True

    from dashboard import intake, health_suggestions as hs
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        row = intake.get_response(cx, "a@b.com")
        pending = hs.list_pending(cx, "a@b.com")
    assert row["answers"]["terrain"] == 3
    assert pending == []


def test_edit_writes_supplied_value_not_suggested_value(client):
    c, appmod = client
    token = _mint_portal(appmod, "a@b.com")
    sid = _seed_suggestion(appmod, "a@b.com", "terrain", 3)
    r = c.post(f"/api/portal/{token}/health-suggestions/{sid}/resolve",
               json={"action": "edit", "value": 4})
    assert r.status_code == 200

    from dashboard import intake, health_suggestions as hs
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        row = intake.get_response(cx, "a@b.com")
        pending = hs.list_pending(cx, "a@b.com")
    assert row["answers"]["terrain"] == 4
    assert pending == []


def test_dismiss_clears_pending_without_writing_record(client):
    c, appmod = client
    token = _mint_portal(appmod, "a@b.com")
    sid = _seed_suggestion(appmod, "a@b.com", "sleep", "improving")
    r = c.post(f"/api/portal/{token}/health-suggestions/{sid}/resolve",
               json={"action": "dismiss"})
    assert r.status_code == 200

    from dashboard import intake, health_suggestions as hs
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        row = intake.get_response(cx, "a@b.com")
        pending = hs.list_pending(cx, "a@b.com")
    assert row is None  # never had an intake row -- dismiss never creates one
    assert pending == []


def test_unknown_action_returns_400(client):
    c, appmod = client
    token = _mint_portal(appmod, "a@b.com")
    sid = _seed_suggestion(appmod, "a@b.com", "terrain", 3)
    r = c.post(f"/api/portal/{token}/health-suggestions/{sid}/resolve",
               json={"action": "delete"})
    assert r.status_code == 400
    assert r.get_json()["ok"] is False

    from dashboard import health_suggestions as hs
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        pending = hs.list_pending(cx, "a@b.com")
    assert len(pending) == 1  # untouched


def test_sid_belonging_to_different_email_404_and_no_write(client):
    c, appmod = client
    token = _mint_portal(appmod, "victim@b.com")
    sid = _seed_suggestion(appmod, "other@b.com", "terrain", 3)
    r = c.post(f"/api/portal/{token}/health-suggestions/{sid}/resolve",
               json={"action": "confirm"})
    assert r.status_code == 404

    from dashboard import intake, health_suggestions as hs
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        victim_row = intake.get_response(cx, "victim@b.com")
        other_row = intake.get_response(cx, "other@b.com")
        other_pending = hs.list_pending(cx, "other@b.com")
    assert victim_row is None
    assert other_row is None  # never written -- ownership check blocked it
    assert len(other_pending) == 1  # still pending -- not resolved either


def test_identity_from_token_not_body(client):
    c, appmod = client
    token = _mint_portal(appmod, "real@b.com")
    sid = _seed_suggestion(appmod, "real@b.com", "terrain", 3)
    r = c.post(f"/api/portal/{token}/health-suggestions/{sid}/resolve",
               json={"action": "confirm", "email": "attacker@evil.com"})
    assert r.status_code == 200

    from dashboard import intake
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        real = intake.get_response(cx, "real@b.com")
        attacker = intake.get_response(cx, "attacker@evil.com")
    assert real["answers"]["terrain"] == 3
    assert attacker is None
