import sqlite3

import pytest

from biofield_local_app import create_app
from dashboard import clinical_tags_console as ctc


def _seed_db(tmp_path):
    db = str(tmp_path / "e4l.db")
    cx = sqlite3.connect(db)
    cx.executescript("""
      CREATE TABLE e4l_clients(client_id INTEGER PRIMARY KEY, name TEXT, email TEXT, date_of_birth TEXT, ghl_contact_id TEXT, archived_at TIMESTAMP);
      CREATE TABLE client_clinical_tags(
        client_id INTEGER, axis TEXT, tag TEXT, confidence REAL, inferred INTEGER,
        status TEXT, source TEXT, evidence TEXT, first_seen TEXT, last_seen TEXT,
        retired_at TEXT, confirmed_by TEXT, UNIQUE(client_id,axis,tag));
    """)
    cx.execute("INSERT INTO e4l_clients(client_id,name) VALUES(1,'Steve Fox')")
    for axis, tag, status in [("status", "system:cardiovascular", "suggested"),
                              ("status", "focus:sleep", "suggested"),
                              ("status", "system:skeletal", "active")]:
        cx.execute("INSERT INTO client_clinical_tags(client_id,axis,tag,confidence,status,source) "
                   "VALUES(1,?,?,0.35,?,'pb-intake')", (axis, tag, status))
    cx.commit(); cx.close()
    return db


def test_review_queue_and_confirm_reject(tmp_path):
    db = _seed_db(tmp_path)
    cx = sqlite3.connect(db)
    q = ctc.review_queue(cx)
    assert q and q[0]["client_id"] == 1 and q[0]["n"] == 2
    assert ctc.confirm(cx, 1, ["system:cardiovascular"]) == 1
    assert ctc.confirm(cx, 1, ["system:skeletal"]) == 0   # already active → not re-confirmed
    assert ctc.reject(cx, 1, ["focus:sleep"]) == 1
    rows = dict(cx.execute("SELECT tag,status FROM client_clinical_tags WHERE client_id=1").fetchall())
    assert rows["system:cardiovascular"] == "active"
    assert rows["focus:sleep"] == "retired"
    cb = cx.execute("SELECT confirmed_by FROM client_clinical_tags WHERE tag='system:cardiovascular'").fetchone()[0]
    assert cb == "glen"


def test_routes_confirm_via_test_client(tmp_path, monkeypatch):
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)
    # dashboard/__init__.py captures CONSOLE_SECRET at import; reloading
    # app does not reset it, so clear the copy the guard actually reads.
    import dashboard as _d; monkeypatch.setattr(_d, "CONSOLE_SECRET", "", raising=False)
    db = _seed_db(tmp_path)
    # ledger lives in the e4l db, which the routes read via e4l_db (not the app's db_path)
    client = create_app(db, e4l_db=db).test_client()
    r = client.get("/clinical-tags")
    assert r.status_code == 200 and b"review queue" in r.data and b"Steve Fox" in r.data
    assert b"Business OS" in r.data and b"/console/orders" in r.data   # return link → BOS board, not Console home
    r = client.get("/clinical-tags/1")
    assert r.status_code == 200 and b"system:cardiovascular" in r.data
    r = client.post("/clinical-tags/1", data={"tags": ["system:cardiovascular", "focus:sleep"], "action": "confirm"})
    assert r.status_code in (302, 303)
    cx = sqlite3.connect(db)
    got = dict(cx.execute("SELECT tag,status FROM client_clinical_tags WHERE client_id=1").fetchall())
    assert got["system:cardiovascular"] == "active" and got["focus:sleep"] == "active"


def test_home_page_links_to_clinical_tags(tmp_path, monkeypatch):
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)
    # dashboard/__init__.py captures CONSOLE_SECRET at import; reloading
    # app does not reset it, so clear the copy the guard actually reads.
    import dashboard as _d; monkeypatch.setattr(_d, "CONSOLE_SECRET", "", raising=False)
    db = _seed_db(tmp_path)
    client = create_app(db).test_client()
    r = client.get("/")
    assert b"/clinical-tags" in r.data
