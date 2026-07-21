import sqlite3
from dashboard import client_360


def _make_e4l(path, *, with_tags=True):
    cx = sqlite3.connect(path)
    cx.execute("CREATE TABLE e4l_clients (client_id INTEGER PRIMARY KEY, name TEXT, email TEXT)")
    cx.execute("INSERT INTO e4l_clients VALUES (7, 'Jane Doe', 'jane@example.com')")
    if with_tags:
        cx.execute("CREATE TABLE client_clinical_tags (client_id INTEGER, axis TEXT, tag TEXT, "
                   "status TEXT, confidence REAL, source TEXT, evidence TEXT, confirmed_by TEXT)")
        cx.execute("INSERT INTO client_clinical_tags VALUES (7,'A','system:gut','active',0.9,'auto','x','glen')")
        cx.execute("INSERT INTO client_clinical_tags VALUES (7,'B','element:water','suggested',0.5,'infer','y',NULL)")
    cx.commit()
    cx.close()


def test_reads_active_and_suggested(tmp_path):
    p = str(tmp_path / "e4l.db")
    _make_e4l(p)
    out = client_360.client_tags_for_email("JANE@example.com", e4l_path=p)
    assert [t["tag"] for t in out["active"]] == ["system:gut"]
    assert [t["tag"] for t in out["suggested"]] == ["element:water"]


def test_missing_tags_table_degrades_empty(tmp_path):
    p = str(tmp_path / "e4l.db")
    _make_e4l(p, with_tags=False)
    out = client_360.client_tags_for_email("jane@example.com", e4l_path=p)
    assert out == {"active": [], "suggested": []}


def test_missing_db_file_degrades_empty(tmp_path):
    out = client_360.client_tags_for_email("jane@example.com", e4l_path=str(tmp_path / "nope.db"))
    assert out == {"active": [], "suggested": []}


def test_unknown_email_degrades_empty(tmp_path):
    p = str(tmp_path / "e4l.db")
    _make_e4l(p)
    out = client_360.client_tags_for_email("stranger@example.com", e4l_path=p)
    assert out == {"active": [], "suggested": []}
