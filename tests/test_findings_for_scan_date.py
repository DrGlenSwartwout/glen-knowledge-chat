import sqlite3
from dashboard.biofield_e4l import findings_for_scan_date


def _mk(tmp_path):
    db = tmp_path / "e4l.db"
    cx = sqlite3.connect(str(db))
    cx.executescript("""
      CREATE TABLE e4l_clients(client_id INTEGER PRIMARY KEY, email TEXT);
      CREATE TABLE e4l_scans(scan_id INTEGER PRIMARY KEY, client_id INTEGER, scan_date TEXT);
      CREATE TABLE e4l_scan_results(id INTEGER PRIMARY KEY, scan_id INTEGER, item_code TEXT, priority_rank INTEGER);
      CREATE TABLE e4l_items(code TEXT PRIMARY KEY, category TEXT, name TEXT, full_name TEXT, e4l_description TEXT);
    """)
    cx.execute("INSERT INTO e4l_clients VALUES(1,'k@x.com')")
    cx.execute("INSERT INTO e4l_scans VALUES(100,1,'2026-06-20')")
    cx.execute("INSERT INTO e4l_scans VALUES(200,1,'2026-06-25')")
    cx.execute("INSERT INTO e4l_items VALUES('ED3','ED','Cell','Cell Driver','desc-cell')")
    cx.execute("INSERT INTO e4l_items VALUES('ET1','ET','Heart','Heart Driver','desc-heart')")
    cx.execute("INSERT INTO e4l_items VALUES('ER9','ER','Env','Env Load','')")
    cx.execute("INSERT INTO e4l_scan_results(scan_id,item_code,priority_rank) VALUES(100,'ED3',1)")
    cx.execute("INSERT INTO e4l_scan_results(scan_id,item_code,priority_rank) VALUES(200,'ET1',1)")
    cx.execute("INSERT INTO e4l_scan_results(scan_id,item_code,priority_rank) VALUES(200,'ER9',2)")
    cx.commit()
    cx.close()
    return str(db)


def test_returns_findings_for_the_specific_date(tmp_path):
    db = _mk(tmp_path)
    f20 = findings_for_scan_date("k@x.com", "2026-06-20", db_path=db)
    f25 = findings_for_scan_date("k@x.com", "2026-06-25", db_path=db)
    assert [x["code"] for x in f20] == ["ED3"]
    assert sorted(x["code"] for x in f25) == ["ER9", "ET1"]   # differs from the other date
    assert f20[0]["description"] == "desc-cell"


def test_unknown_date_or_email_returns_empty(tmp_path):
    db = _mk(tmp_path)
    assert findings_for_scan_date("k@x.com", "2026-01-01", db_path=db) == []
    assert findings_for_scan_date("nobody@x.com", "2026-06-20", db_path=db) == []
    assert findings_for_scan_date("", "2026-06-20", db_path=db) == []
