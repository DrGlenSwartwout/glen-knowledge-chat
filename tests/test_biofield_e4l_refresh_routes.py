"""Routes for the client picker (/api/e4l/clients) and the on-demand live E4L
refresh (/author/<id>/e4l/refresh). A fake client_search + fetch_runner are injected
so the tests never touch the real e4l.db or the live portal."""
import sqlite3
import pytest

from biofield_local_app import create_app
from dashboard.biofield_e4l import scan_context


@pytest.fixture(autouse=True)
def _no_gate(monkeypatch):
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)


def _seed_e4l(path, scans):
    cx = sqlite3.connect(str(path))
    cx.executescript("""
        CREATE TABLE IF NOT EXISTS e4l_clients(client_id INTEGER PRIMARY KEY, name TEXT, email TEXT);
        CREATE TABLE IF NOT EXISTS e4l_scans(scan_id INTEGER PRIMARY KEY, client_id INTEGER, scan_date TEXT);
        CREATE TABLE IF NOT EXISTS e4l_scan_results(id INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_id INTEGER, item_code TEXT, priority_rank INTEGER);
        CREATE TABLE IF NOT EXISTS e4l_items(code TEXT PRIMARY KEY, name TEXT, full_name TEXT,
            category TEXT, e4l_description TEXT, clinical_notes TEXT);
    """)
    cx.execute("INSERT OR IGNORE INTO e4l_clients VALUES(52617,'Kauilani Perdomo','k@x.com')")
    for sid, sd in scans:
        cx.execute("INSERT OR REPLACE INTO e4l_scans VALUES(?,?,?)", (sid, 52617, sd))
    cx.commit(); cx.close()


def test_clients_api_returns_grouped_matches(tmp_path):
    db = str(tmp_path / "chat_log.db")
    app = create_app(db, client_search=lambda q: [{"name": "Kauilani Perdomo",
        "emails": [{"email": "k@x.com", "client_id": 52617, "last_scan_date": "2026-04-14"}]}]
        if "perd" in q.lower() else [])
    c = app.test_client()
    j = c.get("/api/e4l/clients?q=perd").get_json()
    assert j["clients"][0]["name"] == "Kauilani Perdomo"
    assert c.get("/api/e4l/clients?q=zzz").get_json()["clients"] == []


def test_refresh_pulls_newer_scan_and_returns_panel(tmp_path):
    e4l = tmp_path / "e4l.db"
    _seed_e4l(e4l, [(900, "2026-04-14")])               # stale scan only
    today = "2026-06-24"

    def lookup(email):
        return scan_context(email, today, db_path=str(e4l))

    def runner(client_id=None, name=None):               # "live fetch" inserts a fresh scan
        _seed_e4l(e4l, [(950, "2026-06-23")])
        return {"ok": True}

    db = str(tmp_path / "chat_log.db")
    c = create_app(db, scan_lookup=lookup, fetch_runner=runner).test_client()
    tid = c.post("/author/new").headers["Location"].rstrip("/").split("/")[-1]
    c.post(f"/author/{tid}/header", json={"email": "k@x.com"})
    # before refresh the panel is stale
    assert c.get(f"/author/{tid}/e4l").get_json()["e4l"]["status"] == "stale"
    r = c.post(f"/author/{tid}/e4l/refresh", json={"client_id": 52617}).get_json()
    assert r["ok"] is True and r["newer"] is True
    assert r["e4l"]["status"] == "fresh" and r["e4l"]["days_ago"] == 1
    assert "Recent E4L scan" in r["html"]


def test_refresh_reports_not_newer_when_nothing_changes(tmp_path):
    e4l = tmp_path / "e4l.db"
    _seed_e4l(e4l, [(950, "2026-06-23")])
    today = "2026-06-24"
    c = create_app(str(tmp_path / "chat_log.db"),
                   scan_lookup=lambda e: scan_context(e, today, db_path=str(e4l)),
                   fetch_runner=lambda client_id=None, name=None: {"ok": True}).test_client()
    tid = c.post("/author/new").headers["Location"].rstrip("/").split("/")[-1]
    c.post(f"/author/{tid}/header", json={"email": "k@x.com"})
    r = c.post(f"/author/{tid}/e4l/refresh", json={"client_id": 52617}).get_json()
    assert r["ok"] is True and r["newer"] is False
    assert r["e4l"]["status"] == "fresh"


def test_refresh_surfaces_fetch_error(tmp_path):
    e4l = tmp_path / "e4l.db"
    _seed_e4l(e4l, [(900, "2026-04-14")])
    def runner(client_id=None, name=None):
        raise RuntimeError("E4L login failed")
    c = create_app(str(tmp_path / "chat_log.db"),
                   scan_lookup=lambda em: scan_context(em, "2026-06-24", db_path=str(e4l)),
                   fetch_runner=runner).test_client()
    tid = c.post("/author/new").headers["Location"].rstrip("/").split("/")[-1]
    c.post(f"/author/{tid}/header", json={"email": "k@x.com"})
    r = c.post(f"/author/{tid}/e4l/refresh", json={"client_id": 52617}).get_json()
    assert r["ok"] is False and "login failed" in (r["error"] or "")
    # panel still rendered (falls back to whatever is local)
    assert "html" in r


def test_refresh_needs_a_client(tmp_path):
    c = create_app(str(tmp_path / "chat_log.db"),
                   fetch_runner=lambda **k: {"ok": True}).test_client()
    tid = c.post("/author/new").headers["Location"].rstrip("/").split("/")[-1]
    r = c.post(f"/author/{tid}/e4l/refresh", json={}).get_json()  # no email, no client_id
    assert r["ok"] is False
