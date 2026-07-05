import sqlite3
from dashboard import client_scans as cs


def _cx():
    cx = sqlite3.connect(":memory:"); cs.init_client_scans_table(cx); return cx


def test_upsert_and_list():
    cx = _cx()
    n = cs.upsert_scans(cx, "Karin@X.com", [{"scan_date": "2026-06-28", "scan_id": 1037676},
                                            {"scan_date": "2026-06-25", "scan_id": 1037001}])
    assert len(n) == 2                                                     # returns the newly-inserted rows
    got = cs.scans_for(cx, "karin@x.com")
    assert [g["scan_date"] for g in got] == ["2026-06-28", "2026-06-25"]   # most-recent first
    assert got[0]["scan_id"] == "1037676"                                  # stored as str


def test_upsert_idempotent():
    cx = _cx()
    cs.upsert_scans(cx, "k@x.com", [{"scan_date": "2026-06-28", "scan_id": 1}])
    cs.upsert_scans(cx, "k@x.com", [{"scan_date": "2026-06-28", "scan_id": 2}])   # same date, new id
    got = cs.scans_for(cx, "k@x.com")
    assert len(got) == 1 and got[0]["scan_id"] == "2"                     # no dup; scan_id updated


def test_upsert_returns_only_new_rows():
    # The new-scan email keys off this: a re-pushed manifest (rows already present)
    # returns [] so a flag-flip can't mass-email the historical backlog.
    cx = _cx()
    first = cs.upsert_scans(cx, "k@x.com", [{"scan_date": "2026-06-28"}, {"scan_date": "2026-06-25"}])
    assert {r["scan_date"] for r in first} == {"2026-06-28", "2026-06-25"}
    again = cs.upsert_scans(cx, "k@x.com", [{"scan_date": "2026-06-28"}, {"scan_date": "2026-06-25"}])
    assert again == []                                                    # nothing re-emails
    mixed = cs.upsert_scans(cx, "k@x.com", [{"scan_date": "2026-06-28"}, {"scan_date": "2026-07-02"}])
    assert [r["scan_date"] for r in mixed] == ["2026-07-02"]              # only the genuinely new one


def test_blank_email_and_date_skipped():
    cx = _cx()
    assert cs.upsert_scans(cx, "", [{"scan_date": "2026-06-28"}]) == []
    assert cs.upsert_scans(cx, "k@x.com", [{"scan_date": ""}]) == []
    assert cs.scans_for(cx, "k@x.com") == []


import importlib, sys
from pathlib import Path
import pytest


def _app(tmp_path, monkeypatch, *, flag="1"):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SCAN_LIST_ENABLED", flag)
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path: sys.path.insert(0, str(repo))
    try:
        import app as appmod; importlib.reload(appmod)
        import dashboard as _d; monkeypatch.setattr(_d, "CONSOLE_SECRET", "", raising=False)
    except Exception as e:
        pytest.skip(f"app not importable: {e}")
    return appmod


def test_sync_endpoint_upserts(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    c = appmod.app.test_client()
    r = c.post("/api/console/client-scans/sync",
               json={"email": "k@x.com", "scans": [{"scan_date": "2026-06-28", "scan_id": 5}]})
    assert r.status_code == 200 and r.get_json()["ok"] is True
    import sqlite3
    from dashboard import client_scans as cs
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert [s["scan_date"] for s in cs.scans_for(cx, "k@x.com")] == ["2026-06-28"]
    # batch form
    r2 = c.post("/api/console/client-scans/sync",
                json={"batch": [{"email": "a@x.com", "scans": [{"scan_date": "2026-07-01"}]}]})
    assert r2.status_code == 200


def test_sync_endpoint_malformed_batch_item_no_500(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    c = appmod.app.test_client()
    r = c.post("/api/console/client-scans/sync",
               json={"batch": [
                   {"email": "a@x.com", "scans": [{"scan_date": "2026-07-01"}]},
                   "not-a-dict",
                   {"email": "b@x.com", "scans": ["bad"]},
               ]})
    assert r.status_code == 200 and r.get_json()["ok"] is True
    import sqlite3
    from dashboard import client_scans as cs
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert cs.scans_for(cx, "a@x.com") != []
        assert cs.scans_for(cx, "b@x.com") == []


def test_available_scans_payload(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    from dashboard import client_portal as cp, client_scans as cs, portal_biofield_reports as pbr
    import sqlite3
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cp.init_client_portal_table(cx); cs.init_client_scans_table(cx); pbr.init_table(cx)
        cs.upsert_scans(cx, "k@x.com", [{"scan_date": "2026-06-28"}, {"scan_date": "2026-06-25"}])
        pbr.upsert_report(cx, "k@x.com", "2026-06-25", "s1", {"n": 1}, "confirmed")   # 06-25 processed
        tok = cp.upsert_portal(cx, "k@x.com", "K", {}); cx.commit()
    token = tok[0] if isinstance(tok, (tuple, list)) else tok
    if not token: pytest.skip("no mint helper")
    j = appmod.app.test_client().get(f"/api/portal/{token}").get_json()
    av = {s["scan_date"]: s["processed"] for s in j.get("available_scans", [])}
    assert av == {"2026-06-28": False, "2026-06-25": True}


def test_scan_list_flag_off(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch, flag="0")
    from dashboard import client_portal as cp
    import sqlite3
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cp.init_client_portal_table(cx); tok = cp.upsert_portal(cx, "k@x.com", "K", {}); cx.commit()
    token = tok[0] if isinstance(tok, (tuple, list)) else tok
    if not token: pytest.skip("no mint helper")
    j = appmod.app.test_client().get(f"/api/portal/{token}").get_json()
    assert "available_scans" not in j


def test_notified_flow():
    from dashboard import client_scans as cs
    import sqlite3
    cx = sqlite3.connect(":memory:"); cs.init_client_scans_table(cx)
    cs.upsert_scans(cx, "k@x.com", [{"scan_date": "2026-06-28"}, {"scan_date": "2026-06-25"}])
    un = cs.unnotified(cx, "k@x.com")
    assert {u["scan_date"] for u in un} == {"2026-06-28", "2026-06-25"}
    cs.mark_notified(cx, "k@x.com", "2026-06-28")
    assert [u["scan_date"] for u in cs.unnotified(cx, "k@x.com")] == ["2026-06-25"]
