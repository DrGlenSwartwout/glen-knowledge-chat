import sqlite3
from dashboard import scan_pull_requests as spr


def _cx():
    cx = sqlite3.connect(":memory:")
    spr.init_scan_pull_requests_table(cx)
    return cx


def test_create_pending_mark_get_roundtrip():
    cx = _cx()
    res = spr.create_request(cx, "luscombesean@gmail.com", "glen")
    assert res["created"] is True and res["status"] == "pending"
    rid = res["id"]
    assert spr.pending(cx) == [{"id": rid, "query": "luscombesean@gmail.com"}]
    spr.mark(cx, rid, "working")
    assert spr.pending(cx) == []  # working is not pending
    spr.mark(cx, rid, "done", scan_id="1037956", draft_id=52)
    row = spr.get(cx, rid)
    assert row["status"] == "done" and row["scan_id"] == "1037956" and row["draft_id"] == 52


def test_create_dedups_while_pending_or_working():
    cx = _cx()
    a = spr.create_request(cx, "Sean Luscombe")
    b = spr.create_request(cx, "sean luscombe")  # normalized dup
    assert b["created"] is False and b["id"] == a["id"] and b["status"] == "pending"
    spr.mark(cx, a["id"], "done")
    c = spr.create_request(cx, "Sean Luscombe")  # prior is done → new one allowed
    assert c["created"] is True and c["id"] != a["id"]


def test_blank_query_and_missing_get():
    cx = _cx()
    assert spr.create_request(cx, "   ")["created"] is False
    assert spr.get(cx, 9999) is None
