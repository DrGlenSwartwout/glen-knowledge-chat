"""Endpoint-level tests for POST /api/console/client/<email>/health-suggestion.

Covers: console-auth gate (missing/invalid key -> 401), flag OFF -> 404, a
non-editable field_id -> 400, and a valid clinician suggestion landing as a
source='clinician' PENDING row in health_suggestions -- never a direct write
into intake_responses (which stays untouched).

Follows the tests/test_health_profile_endpoint.py fixture pattern: swap
LOG_DB to a tmp sqlite file so tests never touch the dev db, and use
app.test_client().
"""
import os
# Dummy keys so `import app` (which constructs OpenAI + Pinecone clients at
# import) succeeds under a secretless CI without doppler. The clients are
# constructed but never called by this endpoint (DB-only write path).
os.environ.setdefault("OPENAI_API_KEY", "sk-dummy")
os.environ.setdefault("PINECONE_API_KEY", "pc-dummy")

import sqlite3
import pytest

CONSOLE_KEY = "test-console-secret"


@pytest.fixture
def client(monkeypatch, tmp_path):
    import app as appmod
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    monkeypatch.setattr(appmod, "CONSOLE_SECRET", CONSOLE_KEY)
    monkeypatch.setattr(appmod, "_PORTAL_HEALTH_PROFILE_ENABLED", True)
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client(), appmod


def test_clinician_suggestion_lands_pending_and_does_not_write_intake(client):
    c, appmod = client
    r = c.post(
        "/api/console/client/a@b.com/health-suggestion",
        json={"field_id": "sleep", "value": "improving",
              "rationale": "reported better sleep at visit"},
        headers={"X-Console-Key": CONSOLE_KEY},
    )
    assert r.status_code == 200
    assert r.get_json()["ok"] is True

    from dashboard import health_suggestions as _hs, intake
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        pending = _hs.list_pending(cx, "a@b.com")
        intake.init_intake_table(cx)
        row = intake.get_response(cx, "a@b.com")

    assert len(pending) == 1
    assert pending[0]["field_id"] == "sleep"
    assert pending[0]["source"] == "clinician"
    # list_pending only returns status='pending' rows, so its presence here
    # already proves status=='pending'; intake_responses stays untouched.
    assert row is None


def test_non_editable_field_id_returns_400(client):
    c, appmod = client
    r = c.post(
        "/api/console/client/a@b.com/health-suggestion",
        json={"field_id": "terms", "value": "yes", "rationale": "n/a"},
        headers={"X-Console-Key": CONSOLE_KEY},
    )
    assert r.status_code == 400
    assert r.get_json()["ok"] is False

    from dashboard import health_suggestions as _hs
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        pending = _hs.list_pending(cx, "a@b.com")
    assert pending == []


def test_missing_console_key_returns_401(client):
    c, appmod = client
    r = c.post(
        "/api/console/client/a@b.com/health-suggestion",
        json={"field_id": "sleep", "value": "improving", "rationale": "x"},
    )
    assert r.status_code == 401


def test_wrong_console_key_returns_401(client):
    c, appmod = client
    r = c.post(
        "/api/console/client/a@b.com/health-suggestion",
        json={"field_id": "sleep", "value": "improving", "rationale": "x"},
        headers={"X-Console-Key": "wrong-key"},
    )
    assert r.status_code == 401


def test_flag_off_returns_404(client, monkeypatch):
    c, appmod = client
    monkeypatch.setattr(appmod, "_PORTAL_HEALTH_PROFILE_ENABLED", False)
    r = c.post(
        "/api/console/client/a@b.com/health-suggestion",
        json={"field_id": "sleep", "value": "improving", "rationale": "x"},
        headers={"X-Console-Key": CONSOLE_KEY},
    )
    assert r.status_code == 404
