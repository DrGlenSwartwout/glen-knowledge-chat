import importlib, sys
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


def test_sourcing_scan_runs_write_and_dry(monkeypatch, tmp_path):
    appmod = _app(monkeypatch, tmp_path)
    import scripts.scan_supplier_quotes as sq
    calls = {}

    def fake_scan(write=False, days=14, db_path=None, **k):
        calls.update(write=write, days=days, db_path=db_path)
        return {"scanned": 5, "staged": 2, "mode": "write" if write else "dry_run"}

    monkeypatch.setattr(sq, "scan", fake_scan)
    c = appmod.app.test_client()

    # default = write against the live LOG_DB
    r = c.post("/api/cron/sourcing-scan", headers={"X-Cron-Secret": "s3cret"})
    assert r.status_code == 200
    j = r.get_json()
    assert j["ok"] is True and j["staged"] == 2 and j["mode"] == "write"
    assert calls["write"] is True and str(calls["db_path"]).endswith("chat_log.db")

    # ?dry=1 = no-write dry run
    r2 = c.post("/api/cron/sourcing-scan?dry=1", headers={"X-Cron-Secret": "s3cret"})
    assert r2.get_json()["mode"] == "dry_run" and calls["write"] is False
