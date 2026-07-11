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

def _seed(tmp_db):
    from dashboard import client_portal as cp
    with sqlite3.connect(str(tmp_db)) as cx:
        cp.init_client_portal_table(cx)
        token, _ = cp.upsert_portal(cx, "a@x.com", "A", {})
    return token

def test_flag_off_403(monkeypatch, tmp_db):
    monkeypatch.delenv("PORTAL_SCAN_HISTORY_ENABLED", raising=False)
    app = _app(monkeypatch, tmp_db)
    token = _seed(tmp_db)
    r = app.app.test_client().post(f"/api/portal/{token}/scan-prefs", json={"auto_advance": False})
    assert r.status_code == 403

def test_pin_sets_current_and_disables_autoadvance(monkeypatch, tmp_db):
    monkeypatch.setenv("PORTAL_SCAN_HISTORY_ENABLED", "1")
    app = _app(monkeypatch, tmp_db)
    token = _seed(tmp_db)
    r = app.app.test_client().post(f"/api/portal/{token}/scan-prefs", json={"pin_scan_date": "2026-07-02"})
    assert r.status_code == 200
    from dashboard import client_portal as cp
    with sqlite3.connect(str(tmp_db)) as cx:
        assert cp.get_current_scan(cx, "a@x.com") == "2026-07-02"
        assert cp.get_auto_advance(cx, "a@x.com") is False

def test_bad_token_404(monkeypatch, tmp_db):
    monkeypatch.setenv("PORTAL_SCAN_HISTORY_ENABLED", "1")
    app = _app(monkeypatch, tmp_db)
    r = app.app.test_client().post("/api/portal/BADTOKEN/scan-prefs", json={"auto_advance": False})
    assert r.status_code == 404
