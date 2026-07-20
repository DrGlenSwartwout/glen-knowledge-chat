import os
import sqlite3
import pytest

if not os.environ.get("PINECONE_API_KEY"):
    pytest.skip("needs doppler env for import app", allow_module_level=True)

import app as appmod
from dashboard import share_header as sh


@pytest.fixture
def client(monkeypatch, tmp_path):
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(appmod, "LOG_DB", db)
    monkeypatch.setenv("PUBLIC_SURFACE_ENABLED", "1")
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()


def test_console_approve_flips_status(client, monkeypatch):
    cx = sqlite3.connect(appmod.LOG_DB)
    cx.row_factory = sqlite3.Row
    sh.init_share_headers_table(cx)
    sh.upsert_header(cx, "a@b.com", "Ann", "Hello there.")
    cx.close()

    monkeypatch.setattr(appmod, "CONSOLE_SECRET", "test-secret")
    r = client.post("/api/console/share-header/a@b.com/approve",
                    headers={"X-Console-Key": "test-secret"})
    assert r.status_code == 200

    cx = sqlite3.connect(appmod.LOG_DB)
    cx.row_factory = sqlite3.Row
    assert sh.get_approved(cx, "a@b.com") is not None
    cx.close()


def test_console_reject_flips_status(client, monkeypatch):
    cx = sqlite3.connect(appmod.LOG_DB)
    cx.row_factory = sqlite3.Row
    sh.init_share_headers_table(cx)
    sh.upsert_header(cx, "r@b.com", "Rae", "Hello there.")
    cx.close()

    monkeypatch.setattr(appmod, "CONSOLE_SECRET", "test-secret")
    r = client.post("/api/console/share-header/r@b.com/reject",
                    headers={"X-Console-Key": "test-secret"})
    assert r.status_code == 200

    cx = sqlite3.connect(appmod.LOG_DB)
    cx.row_factory = sqlite3.Row
    assert sh.get_approved(cx, "r@b.com") is None
    row = cx.execute("SELECT status FROM share_headers WHERE email=?", ("r@b.com",)).fetchone()
    assert row["status"] == "rejected"
    cx.close()


def test_console_approve_requires_secret(client, monkeypatch):
    monkeypatch.setattr(appmod, "CONSOLE_SECRET", "test-secret")
    r = client.post("/api/console/share-header/a@b.com/approve")
    assert r.status_code == 401


def test_console_rejects_unknown_action(client, monkeypatch):
    monkeypatch.setattr(appmod, "CONSOLE_SECRET", "test-secret")
    r = client.post("/api/console/share-header/a@b.com/delete-everything",
                    headers={"X-Console-Key": "test-secret"})
    assert r.status_code == 400


def test_portal_write_lands_as_pending(client, monkeypatch):
    from dashboard import client_portal as _cp
    cx = sqlite3.connect(appmod.LOG_DB)
    cx.row_factory = sqlite3.Row
    _cp.init_client_portal_table(cx)
    token, _pid = _cp.upsert_portal(cx, "p@b.com", "Pam", {})
    cx.close()

    r = client.post(f"/api/portal/{token}/share-header",
                    json={"display_name": "Pam", "body": "Hello there."})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["status"] == "pending"

    cx = sqlite3.connect(appmod.LOG_DB)
    cx.row_factory = sqlite3.Row
    from dashboard import share_header as _sh
    assert _sh.get_approved(cx, "p@b.com") is None
    row = cx.execute("SELECT status FROM share_headers WHERE email=?", ("p@b.com",)).fetchone()
    assert row["status"] == "pending"
    cx.close()


def test_portal_write_unknown_token_404s(client):
    r = client.post("/api/portal/does-not-exist/share-header",
                    json={"display_name": "X", "body": "Hello there."})
    assert r.status_code == 404


def test_portal_write_empty_body_400s(client, monkeypatch):
    from dashboard import client_portal as _cp
    cx = sqlite3.connect(appmod.LOG_DB)
    cx.row_factory = sqlite3.Row
    _cp.init_client_portal_table(cx)
    token, _pid = _cp.upsert_portal(cx, "e@b.com", "Em", {})
    cx.close()

    r = client.post(f"/api/portal/{token}/share-header",
                    json={"display_name": "Em", "body": ""})
    assert r.status_code == 400
    assert "error" in r.get_json()


def test_portal_write_404s_when_flag_off(monkeypatch, tmp_path):
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    monkeypatch.setenv("PUBLIC_SURFACE_ENABLED", "")
    c = appmod.app.test_client()
    r = c.post("/api/portal/anything/share-header", json={"display_name": "X", "body": "hi"})
    assert r.status_code == 404
