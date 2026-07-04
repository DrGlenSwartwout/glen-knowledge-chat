import importlib, sys, threading
from pathlib import Path
import pytest


def _app(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CRON_SECRET", "s3cret")
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        import app as appmod
        importlib.reload(appmod)
    except Exception as e:
        pytest.skip(f"app not importable: {e}")
    return appmod


def test_sourcing_scan_requires_secret(monkeypatch, tmp_path):
    c = _app(monkeypatch, tmp_path).app.test_client()
    assert c.post("/api/cron/sourcing-scan").status_code == 401
    assert c.post("/api/cron/sourcing-scan", headers={"X-Cron-Secret": "wrong"}).status_code == 401


def test_dry_run_is_sync_bounded(monkeypatch, tmp_path):
    appmod = _app(monkeypatch, tmp_path)
    import scripts.scan_supplier_quotes as sq
    calls = {}

    def fake_scan(write=False, days=14, db_path=None, max_messages=None, **k):
        calls.update(write=write, days=days, db_path=db_path, max_messages=max_messages)
        return {"scanned": 5, "staged": 2, "mode": "write" if write else "dry_run"}

    monkeypatch.setattr(sq, "scan", fake_scan)
    c = appmod.app.test_client()
    r = c.post("/api/cron/sourcing-scan?dry=1", headers={"X-Cron-Secret": "s3cret"})
    assert r.status_code == 200
    j = r.get_json()
    assert j["ok"] is True and j["mode"] == "dry_run" and j["staged"] == 2
    assert calls["write"] is False and calls["max_messages"] == 30
    assert str(calls["db_path"]).endswith("chat_log.db")


def test_live_run_is_async(monkeypatch, tmp_path):
    appmod = _app(monkeypatch, tmp_path)
    import scripts.scan_supplier_quotes as sq
    called = threading.Event()
    seen = {}

    def fake_scan(write=False, days=14, db_path=None, max_messages=None, **k):
        seen.update(write=write, days=days, max_messages=max_messages)
        called.set()
        return {"scanned": 1, "staged": 1, "mode": "write"}

    monkeypatch.setattr(sq, "scan", fake_scan)
    c = appmod.app.test_client()
    r = c.post("/api/cron/sourcing-scan?days=3", headers={"X-Cron-Secret": "s3cret"})
    # returns immediately without waiting for the scan
    assert r.status_code == 200
    j = r.get_json()
    assert j["started"] is True and j["mode"] == "write_async" and j["days"] == 3
    # the background thread runs the real (mocked) scan shortly after
    assert called.wait(timeout=3.0), "background scan was not invoked"
    assert seen["write"] is True and seen["days"] == 3 and seen["max_messages"] == 400
