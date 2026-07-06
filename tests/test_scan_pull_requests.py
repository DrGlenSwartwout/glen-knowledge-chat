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


def test_stale_working_row_does_not_block_new_request():
    cx = _cx()
    a = spr.create_request(cx, "Sean Luscombe")
    spr.mark(cx, a["id"], "working")
    # a fresh working row DOES still dedup
    assert spr.create_request(cx, "Sean Luscombe")["id"] == a["id"]
    # backdate the working row past the staleness window → it no longer blocks
    cx.execute("UPDATE scan_pull_requests SET updated_at='2000-01-01 00:00:00' WHERE id=?", (a["id"],))
    cx.commit()
    c = spr.create_request(cx, "Sean Luscombe")
    assert c["created"] is True and c["id"] != a["id"]


import importlib, sys
from pathlib import Path
import pytest


def _app(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)   # auth open in test
    monkeypatch.setenv("SCAN_PULL_ENABLED", "1")
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        import app as appmod
        importlib.reload(appmod)
    except Exception as e:
        pytest.skip(f"app not importable: {e}")
    return appmod


def test_endpoints_enqueue_list_complete_get(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    c = appmod.app.test_client()
    r = c.post("/api/console/scan-pull-requests", json={"query": "luscombesean@gmail.com"})
    assert r.status_code == 200
    rid = r.get_json()["id"]
    assert rid
    lst = c.get("/api/console/scan-pull-requests?limit=50").get_json()
    assert any(x["id"] == rid and x["query"] == "luscombesean@gmail.com" for x in lst["requests"])
    done = c.post(f"/api/console/scan-pull-requests/{rid}/complete",
                  json={"status": "done", "scan_id": "1037956", "draft_id": 52})
    assert done.status_code == 200
    got = c.get(f"/api/console/scan-pull-requests/{rid}").get_json()["request"]
    assert got["status"] == "done" and got["draft_id"] == 52
    # completed → no longer pending
    assert c.get("/api/console/scan-pull-requests").get_json()["requests"] == []


def test_enqueue_requires_query_and_flag(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    c = appmod.app.test_client()
    assert c.post("/api/console/scan-pull-requests", json={"query": ""}).status_code == 400
    # flag off → inert (no row created)
    monkeypatch.setenv("SCAN_PULL_ENABLED", "0")
    importlib.reload(appmod)
    c2 = appmod.app.test_client()
    r = c2.post("/api/console/scan-pull-requests", json={"query": "x@y.com"})
    assert r.get_json().get("status") == "disabled"


def test_reveals_payload_exposes_flag(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    c = appmod.app.test_client()
    body = c.get("/api/console/biofield-reveals").get_json()
    assert body.get("scan_pull_enabled") is True
