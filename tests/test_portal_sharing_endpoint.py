"""Tests for POST /api/portal/<token>/sharing (Task 4: token-scoped write endpoint
for the data-sharing opt-in feature). Modeled on
tests/test_portal_scan_history_payload.py's app/db idiom."""

import importlib
import sqlite3
import sys
from pathlib import Path

import pytest


def _app(monkeypatch, tmp_db):
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        app = importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")
    monkeypatch.setattr(app, "LOG_DB", str(tmp_db))
    return app


def _seed_portal(tmp_db, email):
    from dashboard import client_portal as cp
    with sqlite3.connect(str(tmp_db)) as cx:
        cp.init_client_portal_table(cx)
        token, _ = cp.upsert_portal(cx, email, "M", {})
    return token


def test_sharing_sets_consent_and_grants(monkeypatch, tmp_db):
    monkeypatch.setenv("DATA_SHARING_REWARD_ENABLED", "true")
    app = _app(monkeypatch, tmp_db)
    token = _seed_portal(tmp_db, "member@ex.com")

    app.app.config["TESTING"] = True
    r = app.app.test_client().post(
        f"/api/portal/{token}/sharing",
        json={
            "toggles": {"research_results": True},
            # attacker attempts to override identity -- must be ignored
            "email": "attacker@evil.com",
        },
    )
    assert r.status_code == 200
    body = r.get_json()
    assert body["consent"]["tier"] == 2
    assert body["rewards"].get("free_reveal_unlock") == "granted"


def test_identity_from_token_not_body(monkeypatch, tmp_db):
    # STRONG identity assertion: the consent must land under the TOKEN's email,
    # never the body's — even when the DB is inspected directly, not just the
    # response shape.
    monkeypatch.setenv("DATA_SHARING_REWARD_ENABLED", "true")
    app = _app(monkeypatch, tmp_db)
    token = _seed_portal(tmp_db, "member@ex.com")

    app.app.config["TESTING"] = True
    r = app.app.test_client().post(
        f"/api/portal/{token}/sharing",
        json={
            "toggles": {"research_results": True},
            "email": "attacker@evil.com",
        },
    )
    assert r.status_code == 200

    with sqlite3.connect(str(tmp_db)) as cx:
        member_row = cx.execute(
            "SELECT tier FROM member_data_sharing WHERE email=?", ("member@ex.com",)
        ).fetchone()
        attacker_row = cx.execute(
            "SELECT tier FROM member_data_sharing WHERE email=?", ("attacker@evil.com",)
        ).fetchone()
    assert member_row is not None and member_row[0] == 2
    assert attacker_row is None


def test_sharing_404_when_flag_off(monkeypatch, tmp_db):
    monkeypatch.delenv("DATA_SHARING_REWARD_ENABLED", raising=False)
    app = _app(monkeypatch, tmp_db)
    token = _seed_portal(tmp_db, "member@ex.com")

    app.app.config["TESTING"] = True
    r = app.app.test_client().post(f"/api/portal/{token}/sharing", json={"toggles": {}})
    assert r.status_code == 404
