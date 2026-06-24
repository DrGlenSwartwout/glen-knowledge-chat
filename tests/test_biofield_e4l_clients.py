"""Client name picker + on-demand live E4L fetch helpers.

search_clients() drives the name autocomplete (grouping a name's distinct emails so
the same-name/different-email case is pickable). fetch_live() shells out to the vault
scraper+parser via an INJECTED runner so tests never touch the live E4L portal."""
import sqlite3
import pytest

from dashboard.biofield_e4l import search_clients, fetch_live


def _seed(path):
    cx = sqlite3.connect(str(path))
    cx.executescript("""
        CREATE TABLE e4l_clients(client_id INTEGER PRIMARY KEY, name TEXT, email TEXT);
        CREATE TABLE e4l_scans(scan_id INTEGER PRIMARY KEY, client_id INTEGER, scan_date TEXT);
    """)
    cx.executemany("INSERT INTO e4l_clients VALUES(?,?,?)", [
        (19931, "Kauilani Perdomo", "caesarperdomo@yahoo.com"),
        (52617, "Kauilani Perdomo", "kauilaniperdomo@gmail.com"),
        (19681, "Kanehakai Perdomo", "kanehekaiperdomo@gmail.com"),
        (100, "Jane Doe", "jane@x.com")])
    cx.executemany("INSERT INTO e4l_scans VALUES(?,?,?)", [
        (900, 52617, "2026-04-14"), (901, 52617, "2026-02-11"),
        (902, 19931, "2019-08-02")])
    cx.commit(); cx.close()


def test_search_by_name_groups_distinct_emails(tmp_path):
    db = tmp_path / "e4l.db"; _seed(db)
    res = search_clients("perdomo", db_path=str(db))
    by_name = {c["name"]: c for c in res}
    # the duplicate name surfaces ONE entry with both emails
    k = by_name["Kauilani Perdomo"]
    emails = sorted(e["email"] for e in k["emails"])
    assert emails == ["caesarperdomo@yahoo.com", "kauilaniperdomo@gmail.com"]
    assert "Kanehakai Perdomo" in by_name


def test_search_returns_client_id_and_last_scan_per_email(tmp_path):
    db = tmp_path / "e4l.db"; _seed(db)
    k = {c["name"]: c for c in search_clients("kauilani", db_path=str(db))}["Kauilani Perdomo"]
    gmail = [e for e in k["emails"] if e["email"] == "kauilaniperdomo@gmail.com"][0]
    assert gmail["client_id"] == 52617
    assert gmail["last_scan_date"] == "2026-04-14"   # most recent of that client's scans


def test_search_matches_email_substring(tmp_path):
    db = tmp_path / "e4l.db"; _seed(db)
    names = {c["name"] for c in search_clients("caesar", db_path=str(db))}
    assert "Kauilani Perdomo" in names


def test_search_blank_query_returns_empty(tmp_path):
    db = tmp_path / "e4l.db"; _seed(db)
    assert search_clients("", db_path=str(db)) == []
    assert search_clients("  ", db_path=str(db)) == []


def test_search_missing_db_returns_empty(tmp_path):
    assert search_clients("perdomo", db_path=str(tmp_path / "nope.db")) == []


def test_search_limit_caps_name_groups(tmp_path):
    db = tmp_path / "e4l.db"
    cx = sqlite3.connect(str(db))
    cx.execute("CREATE TABLE e4l_clients(client_id INTEGER PRIMARY KEY, name TEXT, email TEXT)")
    cx.execute("CREATE TABLE e4l_scans(scan_id INTEGER PRIMARY KEY, client_id INTEGER, scan_date TEXT)")
    for i in range(10):
        cx.execute("INSERT INTO e4l_clients VALUES(?,?,?)", (i, f"Test Person{i}", f"p{i}@x.com"))
    cx.commit(); cx.close()
    assert len(search_clients("test person", db_path=str(db), limit=3)) == 3


def test_fetch_live_invokes_runner_with_client_id():
    seen = {}
    def runner(client_id=None, name=None):
        seen["client_id"] = client_id; seen["name"] = name
        return {"ok": True}
    out = fetch_live(client_id=52617, runner=runner)
    assert seen["client_id"] == 52617
    assert out["ok"] is True


def test_fetch_live_runner_error_is_captured_not_raised():
    def runner(client_id=None, name=None):
        raise RuntimeError("portal login failed")
    out = fetch_live(client_id=52617, runner=runner)
    assert out["ok"] is False and "portal login failed" in out["error"]


def test_fetch_live_requires_an_identifier():
    out = fetch_live(runner=lambda **k: {"ok": True})
    assert out["ok"] is False and "identifier" in out["error"].lower()
