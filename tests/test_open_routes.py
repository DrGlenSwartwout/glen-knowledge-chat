import importlib, sqlite3, sys
from pathlib import Path
import pytest


def _app(tmp_path, monkeypatch, *, rr="1"):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("READ_RECEIPTS_ENABLED", rr)
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path: sys.path.insert(0, str(repo))
    try:
        import app as appmod; importlib.reload(appmod)
        import dashboard as _d; monkeypatch.setattr(_d, "CONSOLE_SECRET", "", raising=False)
    except Exception as e:
        pytest.skip(f"app not importable: {e}")
    return appmod


def _seed_portal(appmod, email):
    from dashboard import client_portal as cp, portal_biofield_reports as pbr, opens
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cp.init_client_portal_table(cx); pbr.init_table(cx); opens.init_opens_table(cx)
        pbr.upsert_report(cx, email, "2026-06-25", "s1", {"n": 1}, "confirmed"); cx.commit()
        token = cp.upsert_portal(cx, email, "Client", {})   # confirm the real mint helper
        cx.commit()
    return token[0] if isinstance(token, (tuple, list)) else token


def test_report_open_records_and_payload(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    token = _seed_portal(appmod, "c@x.com")
    if not token: pytest.skip("no portal mint helper")
    c = appmod.app.test_client()
    # before open: payload opens map has no entry for the date
    j = c.get(f"/api/portal/{token}").get_json()
    assert (j.get("opens") or {}).get("2026-06-25") in (None,)
    # explicit open records it
    r = c.post(f"/api/portal/{token}/open", json={"scan_date": "2026-06-25"})
    assert r.status_code == 200 and r.get_json()["ok"] is True
    from dashboard import opens
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert opens.get_open(cx, "report", "c@x.com|2026-06-25")["open_count"] == 1
    # now the payload reflects the open
    j2 = c.get(f"/api/portal/{token}").get_json()
    assert (j2.get("opens") or {}).get("2026-06-25", {}).get("open_count") == 1


def test_owner_open_does_not_record(tmp_path, monkeypatch):
    # with CONSOLE_SECRET set, an owner-keyed call must NOT record
    monkeypatch.setenv("CONSOLE_SECRET", "sek")
    appmod = _app(tmp_path, monkeypatch)
    import dashboard as _d; monkeypatch.setattr(_d, "CONSOLE_SECRET", "sek", raising=False)
    token = _seed_portal(appmod, "c@x.com")
    if not token: pytest.skip("no portal mint helper")
    c = appmod.app.test_client()
    r = c.post(f"/api/portal/{token}/open?key=sek", json={"scan_date": "2026-06-25"})
    assert r.status_code == 200
    from dashboard import opens
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert opens.get_open(cx, "report", "c@x.com|2026-06-25") is None   # skipped


def test_flag_off_inert(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch, rr="0")
    token = _seed_portal(appmod, "c@x.com")
    if not token: pytest.skip("no portal mint helper")
    c = appmod.app.test_client()
    j = c.get(f"/api/portal/{token}").get_json()
    assert "opens" not in j                       # no payload key when flag off
    r = c.post(f"/api/portal/{token}/open", json={"scan_date": "2026-06-25"})
    from dashboard import opens
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert opens.get_open(cx, "report", "c@x.com|2026-06-25") is None   # inert


def test_console_opens_read(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    from dashboard import opens
    with sqlite3.connect(appmod.LOG_DB) as cx:
        opens.init_opens_table(cx)
        opens.record_open(cx, "invoice", "tokA", now="2026-07-04 10:00:00"); cx.commit()
    c = appmod.app.test_client()
    r = c.get("/api/console/opens?kind=invoice&keys=tokA,tokB")
    assert r.status_code == 200
    j = r.get_json()
    assert j["opens"]["tokA"]["open_count"] == 1 and "tokB" not in j["opens"]
