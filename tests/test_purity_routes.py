"""Phase-2a purity routes: request (portal+console), screen, tier2, confirm.
Console writes are console-secret gated; client identity comes from the portal
token only."""
import sqlite3
import pytest
import app as app_mod
from dashboard import product_ratings as pr


@pytest.fixture
def client(monkeypatch, tmp_path):
    db = str(tmp_path / "c.db")
    cx = sqlite3.connect(db); cx.row_factory = sqlite3.Row
    pr.init_tables(cx); cx.commit(); cx.close()
    monkeypatch.setattr(app_mod, "LOG_DB", db)
    # portal token -> email; console ok on a secret header value only
    monkeypatch.setattr(app_mod, "_portal_record_for",
                        lambda cx, tok: {"email": "a@b.com"} if tok == "TOKA" else None)
    monkeypatch.setattr(app_mod, "_portal_console_ok", lambda: True)
    monkeypatch.setattr(app_mod, "membership_category", lambda email: "full", raising=False)
    app_mod.app.config["TESTING"] = True
    return app_mod.app.test_client()


def _get(db, key):
    cx = sqlite3.connect(db); cx.row_factory = sqlite3.Row
    r = cx.execute("SELECT * FROM product_ratings WHERE product_key=?", (key,)).fetchone()
    cx.close(); return dict(r) if r else None


def test_console_request_creates_row(client):
    r = client.post("/api/console/purity/request",
                    json={"product_key": "jarrow-mag", "brand": "Jarrow", "product_name": "Mag Taurate"})
    assert r.status_code == 200
    assert _get(app_mod.LOG_DB, "jarrow-mag")["status"] == "requested"


def test_portal_request_uses_token_email_not_a_field(client):
    r = client.post("/api/portal/TOKA/purity/request",
                    json={"product_key": "k", "brand": "B", "product_name": "N",
                          "email": "attacker@evil.com"})
    assert r.status_code == 200
    row = _get(app_mod.LOG_DB, "k")
    assert row["status"] == "requested"
    # requested_by came from the token (a@b.com), never the body's email field
    assert "attacker@evil.com" not in (row.get("requested_by") or "")


def test_portal_request_unknown_token_401(client):
    r = client.post("/api/portal/NOPE/purity/request",
                    json={"product_key": "k", "brand": "B", "product_name": "N"})
    assert r.status_code in (401, 404)
    assert _get(app_mod.LOG_DB, "k") is None


def test_portal_request_not_entitled_403(client, monkeypatch):
    # A non-paid client with no explicit grant is refused, and no row is created.
    monkeypatch.setattr(app_mod, "membership_category", lambda email: "none", raising=False)
    r = client.post("/api/portal/TOKA/purity/request",
                    json={"product_key": "blocked", "brand": "B", "product_name": "N"})
    assert r.status_code == 403
    assert _get(app_mod.LOG_DB, "blocked") is None
