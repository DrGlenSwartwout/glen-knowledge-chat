import importlib, sqlite3, sys
from pathlib import Path

import pytest

from dashboard import scan_analysis as sa


def test_upsert_get_idempotent():
    cx = sqlite3.connect(":memory:"); sa.init_table(cx)
    art = {"email": "a@x.com", "scan_count": 3, "date_range": ["2026-01-01", "2026-03-01"],
           "top_patterns": [], "narrative": "x"}
    sa.upsert(cx, "a@x.com", art); sa.upsert(cx, "a@x.com", art)
    got = sa.get(cx, "A@X.com")
    assert got["scan_count"] == 3 and got["analysis"]["narrative"] == "x"
    assert cx.execute("SELECT COUNT(*) FROM scan_analyses").fetchone()[0] == 1


# ---------------------------------------------------------------------------
# Endpoint tests (require DATA_DIR env + doppler)
# ---------------------------------------------------------------------------

def _load(mod):
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        return importlib.import_module(mod)
    except Exception as e:
        pytest.skip(f"{mod} not importable: {e}")


def _app_db(monkeypatch, tmp_path):
    app_module = _load("app")
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    with sqlite3.connect(db) as cx:
        sa.init_table(cx)
        cx.commit()
    return app_module, db


def test_endpoint_unauthorized(monkeypatch, tmp_path):
    app_module, db = _app_db(monkeypatch, tmp_path)
    c = app_module.app.test_client()
    resp = c.post("/api/e4l/scan-analysis", json={"email": "a@x.com"})
    assert resp.status_code == 401


def test_endpoint_missing_email(monkeypatch, tmp_path):
    app_module, db = _app_db(monkeypatch, tmp_path)
    key = app_module.os.environ.get("CRON_SECRET") or app_module.os.environ.get("CONSOLE_SECRET", "")
    if not key:
        pytest.skip("no secret configured")
    c = app_module.app.test_client()
    resp = c.post("/api/e4l/scan-analysis",
                  headers={"X-Cron-Secret": key},
                  json={"scan_count": 1})
    assert resp.status_code == 400


def test_endpoint_stores_artifact(monkeypatch, tmp_path):
    app_module, db = _app_db(monkeypatch, tmp_path)
    key = app_module.os.environ.get("CRON_SECRET") or app_module.os.environ.get("CONSOLE_SECRET", "")
    if not key:
        pytest.skip("no secret configured")
    c = app_module.app.test_client()
    art = {"email": "b@x.com", "scan_count": 5,
           "date_range": ["2026-01-01", "2026-06-01"],
           "top_patterns": [], "narrative": "test narrative"}
    resp = c.post("/api/e4l/scan-analysis",
                  headers={"X-Cron-Secret": key},
                  json=art)
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["ok"] is True and data["email"] == "b@x.com"
    # Verify stored
    with sqlite3.connect(db) as cx:
        got = sa.get(cx, "b@x.com")
    assert got["scan_count"] == 5 and got["analysis"]["narrative"] == "test narrative"
