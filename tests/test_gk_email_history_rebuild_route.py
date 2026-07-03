"""Tests for POST /api/console/gk-email-history-rebuild — mirrors
tests/test_fmp_history_rebuild_route.py's fixture pattern. Console-gated
(require_console_key + ok/fail). Gmail fetch is monkeypatched (never calls
Gmail in tests) via dashboard.gk_email_history.fetch_gk_order_emails.
"""
import sqlite3
import pytest


@pytest.fixture
def client(monkeypatch, tmp_path):
    import app as appmod
    import dashboard as _dashboard
    from dashboard import purchase_history as _ph

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    monkeypatch.setattr(appmod, "CONSOLE_SECRET", "test-secret")
    monkeypatch.setattr(_dashboard, "CONSOLE_SECRET", "test-secret")

    with sqlite3.connect(appmod.LOG_DB) as cx:
        _ph.init_purchase_history_table(cx)
        cx.commit()

    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()


def _fake_fetch():
    return [{
        "subject": "New order : #1010 - YOUHZZCDS",
        "body": (
            "A new order was placed on Remedy Match by the following customer:\n"
            "Pamela Kilmer (pkilmer108@gmail.com)\n\n"
            "ORDER: YOUHZZCDS Placed on 06-28-2026\n\n"
            '<a href="https://remedymatch.com/remedies/syntropy/265-ocuheal-eye-drops">OcuHeal Eye Drops - </a>\n'
        ),
    }]


def test_rebuild_requires_console_key(client):
    r = client.post("/api/console/gk-email-history-rebuild")
    assert r.status_code == 401


def test_rebuild_populates_purchase_history_and_returns_counts(client, monkeypatch):
    import app as appmod
    from dashboard import gk_email_history as _gh

    monkeypatch.setattr(_gh, "fetch_gk_order_emails", _fake_fetch)

    r = client.post("/api/console/gk-email-history-rebuild", headers={"X-Console-Key": "test-secret"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    data = body["data"]
    assert data["orders"] == 1
    assert data["rows"] == 1
    assert data["skipped_unmapped"] == 0
    assert data["skipped_noemail"] == 0

    with sqlite3.connect(appmod.LOG_DB) as cx:
        rows = cx.execute(
            "SELECT email, slug, purchased_at, source, source_ref FROM purchase_history"
        ).fetchall()
    assert rows == [("pkilmer108@gmail.com", "ocuheal-eye-drops", "2026-06-28", "groovekart", "YOUHZZCDS")]


def test_rebuild_fails_cleanly_on_missing_products_json(client, monkeypatch):
    """If data/products.json is missing/malformed, the route fails clean
    (400 + ok:false) instead of 500ing — an explicit admin action."""
    import builtins

    real_open = builtins.open

    def _raising_open(path, *a, **kw):
        if str(path).endswith("products.json"):
            raise FileNotFoundError(f"no such file: {path}")
        return real_open(path, *a, **kw)

    monkeypatch.setattr(builtins, "open", _raising_open)
    r = client.post("/api/console/gk-email-history-rebuild", headers={"X-Console-Key": "test-secret"})
    assert r.status_code == 400
    body = r.get_json()
    assert body["ok"] is False
    assert "products.json" in body["error"]
