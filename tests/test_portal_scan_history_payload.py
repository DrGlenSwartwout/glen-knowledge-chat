import importlib, sqlite3, sys
from pathlib import Path
import pytest

def _app(monkeypatch, tmp_db):
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        app = importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")
    monkeypatch.setattr(app, "LOG_DB", str(tmp_db))
    app._init_workspace_schema()
    return app

def _seed(app, tmp_db):
    from dashboard import client_portal as cp, portal_biofield_reports as pbr
    with sqlite3.connect(str(tmp_db)) as cx:
        cp.init_client_portal_table(cx)
        pbr.init_table(cx)
        token, _ = cp.upsert_portal(cx, "a@x.com", "A", {"biofield_status": "confirmed"})
        pbr.upsert_report(cx, "a@x.com", "2026-07-02", "111", {"greeting": "old"}, "confirmed")
        pbr.upsert_report(cx, "a@x.com", "2026-07-09", "222", {"greeting": "new"}, "confirmed")
    return token

def test_flag_off_no_new_keys(monkeypatch, tmp_db):
    monkeypatch.delenv("PORTAL_SCAN_HISTORY_ENABLED", raising=False)
    app = _app(monkeypatch, tmp_db)
    token = _seed(app, tmp_db)
    j = app.app.test_client().get(f"/api/portal/{token}").get_json()
    assert "scan_history_enabled" not in j
    assert "auto_advance" not in j

def test_flag_on_exposes_prefs_and_current(monkeypatch, tmp_db):
    monkeypatch.setenv("PORTAL_SCAN_HISTORY_ENABLED", "1")
    app = _app(monkeypatch, tmp_db)
    token = _seed(app, tmp_db)
    j = app.app.test_client().get(f"/api/portal/{token}").get_json()
    assert j["scan_history_enabled"] is True
    assert j["auto_advance"] is True
    assert j["scan_date"] == "2026-07-09"          # newest when no pointer
    assert j["scan_dates"] == ["2026-07-09", "2026-07-02"]
    assert j["current_scan_date"] == "2026-07-09"   # persisted pointer: newest when no pointer set

def test_dangling_pointer_falls_to_newest(monkeypatch, tmp_db):
    monkeypatch.setenv("PORTAL_SCAN_HISTORY_ENABLED", "1")
    app = _app(monkeypatch, tmp_db)
    token = _seed(app, tmp_db)
    from dashboard import client_portal as cp
    with sqlite3.connect(str(tmp_db)) as cx:
        cp.set_current_scan(cx, "a@x.com", "2099-01-01")   # points nowhere
    j = app.app.test_client().get(f"/api/portal/{token}").get_json()
    assert j["scan_date"] == "2026-07-09"
    assert j["current_scan_date"] == "2026-07-09"   # persisted pointer: dangling falls to newest

def test_current_scan_date_is_persisted_pointer_not_displayed_scan(monkeypatch, tmp_db):
    # T7a regression: current_scan_date must track the PERSISTED pointer, not
    # whatever scan the caller is transiently viewing via ?scan_date=.
    monkeypatch.setenv("PORTAL_SCAN_HISTORY_ENABLED", "1")
    app = _app(monkeypatch, tmp_db)
    token = _seed(app, tmp_db)
    from dashboard import client_portal as cp
    with sqlite3.connect(str(tmp_db)) as cx:
        cp.set_current_scan(cx, "a@x.com", "2026-07-02")
    j = app.app.test_client().get(f"/api/portal/{token}?scan_date=2026-07-09").get_json()
    assert j["scan_date"] == "2026-07-09"           # displayed scan honors the query param
    assert j["current_scan_date"] == "2026-07-02"   # persisted pointer stays put
