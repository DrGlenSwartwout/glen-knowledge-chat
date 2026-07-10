"""Tests for the one-shot admin trigger POST /api/console/fmp-history-rebuild.

Reruns dashboard.fmp_history.rebuild_from_fmp against whatever fmp_* projection
tables are already on prod (loaded earlier via /api/console/fmp-orders-ingest),
without re-pushing a CSV. Console-gated like the other /api/console/... routes
(require_console_key + ok/fail — see /api/money/... ~app.py:23100).

Uses the real reviewed data/fmp_slug_map.json (id "1020" -> "terrain-restore")
rather than a fixture, since that file already ships in the repo.
"""
import sqlite3
import pytest


@pytest.fixture
def client(monkeypatch, tmp_path):
    import app as appmod
    import dashboard as _dashboard
    from dashboard import fmp_orders as _fo

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    monkeypatch.setattr(appmod, "CONSOLE_SECRET", "test-secret")
    monkeypatch.setattr(_dashboard, "CONSOLE_SECRET", "test-secret")

    with sqlite3.connect(appmod.LOG_DB) as cx:
        _fo.ensure_tables(cx)
        cx.execute(
            "INSERT INTO fmp_clients (id_pk,name_first,name_last,company,email,phone_res,phone_cell,phone_business) "
            "VALUES ('c1','Jo','Cud','','jo@x.com','','','')")
        cx.execute(
            "INSERT INTO fmp_invoices (id_pk,id_fk_client,invoice_date,status,subtotal,total,shipping,outstanding) "
            "VALUES ('i1','c1','2026-04-01','Closed','40.00','40.00','0.00','0.00')")
        # id_fk_product 1020 is resolved -> terrain-restore in the real
        # (reviewed) data/fmp_slug_map.json shipped in the repo.
        cx.execute(
            "INSERT INTO fmp_invoice_items (id_pk,id_fk_invoice,id_fk_product,description,qty,price,ext_price) "
            "VALUES ('it1','i1','1020','Terrain Restore','1','40.00','40.00')")
        cx.commit()

    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()


def test_rebuild_requires_console_key(client):
    r = client.post("/api/console/fmp-history-rebuild")
    assert r.status_code == 401


def test_rebuild_populates_purchase_history_and_returns_counts(client, monkeypatch):
    import app as appmod

    r = client.post("/api/console/fmp-history-rebuild", headers={"X-Console-Key": "test-secret"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    data = body["data"]
    assert data["rows"] == 1
    assert data["skipped_excluded"] == 0
    assert data["skipped_unmapped"] == 0
    assert data["skipped_noemail"] == 0

    with sqlite3.connect(appmod.LOG_DB) as cx:
        rows = cx.execute(
            "SELECT email, slug, purchased_at, source, source_ref FROM purchase_history"
        ).fetchall()
    assert rows == [("jo@x.com", "terrain-restore", "2026-04-01", "fmp", "it1")]


def test_rebuild_does_not_wipe_when_projection_tables_are_empty(monkeypatch, tmp_path):
    """An empty FMP extraction (projection tables not loaded) must NOT clear the existing
    slice. It returns rows:0 — a visible 'found nothing' — while leaving the slice intact,
    rather than the old behavior of deleting it with nothing to put back."""
    import app as appmod
    import dashboard as _dashboard
    from dashboard import fmp_orders as _fo

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    monkeypatch.setattr(appmod, "CONSOLE_SECRET", "test-secret")
    monkeypatch.setattr(_dashboard, "CONSOLE_SECRET", "test-secret")

    with sqlite3.connect(appmod.LOG_DB) as cx:
        _fo.ensure_tables(cx)  # empty fmp_* projection tables
        import dashboard.purchase_history as _ph
        _ph.init_purchase_history_table(cx)
        cx.execute("INSERT INTO purchase_history(email, slug, purchased_at, source, source_ref) "
                   "VALUES ('keep@x.com','terrain-restore','2026-01-01','fmp','it1')")
        cx.commit()

    appmod.app.config["TESTING"] = True
    c = appmod.app.test_client()
    r = c.post("/api/console/fmp-history-rebuild", headers={"X-Console-Key": "test-secret"})
    assert r.status_code == 200
    assert r.get_json()["data"]["rows"] == 0

    with sqlite3.connect(appmod.LOG_DB) as cx:
        kept = cx.execute("SELECT slug FROM purchase_history WHERE source='fmp'").fetchall()
    assert kept == [("terrain-restore",)], "the existing slice was wiped by an empty rebuild"


def test_rebuild_fails_cleanly_on_missing_slug_map(client, monkeypatch):
    """If data/fmp_slug_map.json is missing/malformed, the route fails clean
    (400 + ok:false) instead of 500ing or swallowing the error — this is an
    explicit admin action, unlike the auto-ingest hook which logs+skips."""
    import builtins

    real_open = builtins.open

    def _raising_open(path, *a, **kw):
        if str(path).endswith("fmp_slug_map.json"):
            raise FileNotFoundError(f"no such file: {path}")
        return real_open(path, *a, **kw)

    monkeypatch.setattr(builtins, "open", _raising_open)
    r = client.post("/api/console/fmp-history-rebuild", headers={"X-Console-Key": "test-secret"})
    assert r.status_code == 400
    body = r.get_json()
    assert body["ok"] is False
    assert "fmp_slug_map" in body["error"]
