"""Task 2: GET /api/console/life-stress-curation/<email> — console-key read of a
client's Life Stress curation, for the LOCAL biofield report renderer (which
fetches over X-Console-Key at render time). Read-only.

Patterns borrowed from tests/test_life_stress_curation_endpoint.py (app-import +
LOG_DB monkeypatch + flag env) and tests/test_console_points_ledger.py (console-key
auth via app.CONSOLE_SECRET + X-Console-Key header). Guard order mirrors the sibling
console route api_console_e4l_db_sync: flag off -> 404, bad/missing key -> 401
(via _portal_console_ok, not 403)."""
import importlib
import sqlite3
import sys
from pathlib import Path

import pytest

from dashboard import life_stress_curation

PID = "doc1"
EMAIL = "pat@x.com"
KEY = "test-console-secret"


def _app():
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")


@pytest.fixture
def wired(monkeypatch, tmp_path):
    """Wires up a real sqlite file at app.LOG_DB, a console key, and the flag ON."""
    app_module = _app()
    db_path = tmp_path / "chat_log.db"
    with sqlite3.connect(db_path) as cx:
        cx.row_factory = sqlite3.Row
        life_stress_curation.init_table(cx)

    monkeypatch.setattr(app_module, "LOG_DB", db_path)
    monkeypatch.setenv("LIFE_STRESS_ENABLED", "1")
    monkeypatch.setattr(app_module, "CONSOLE_SECRET", KEY)
    return app_module, db_path


def _get(client, email, key=None):
    headers = {"X-Console-Key": key} if key else {}
    return client.get(f"/api/console/life-stress-curation/{email}", headers=headers)


def test_flag_off_404s(monkeypatch, wired):
    app_module, _db = wired
    monkeypatch.delenv("LIFE_STRESS_ENABLED", raising=False)
    client = app_module.app.test_client()
    r = _get(client, EMAIL, KEY)
    assert r.status_code == 404


def test_missing_key_401(wired):
    app_module, _db = wired
    client = app_module.app.test_client()
    r = _get(client, EMAIL)
    assert r.status_code == 401


def test_wrong_key_401(wired):
    app_module, _db = wired
    client = app_module.app.test_client()
    r = _get(client, EMAIL, "wrong-key")
    assert r.status_code == 401


def test_valid_key_returns_curation(wired):
    app_module, db_path = wired
    with sqlite3.connect(db_path) as cx:
        cx.row_factory = sqlite3.Row
        life_stress_curation.set(cx, EMAIL, PID, ["grief-release"], "for the grief pattern")

    client = app_module.app.test_client()
    r = _get(client, EMAIL, KEY)
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["curation"]["slugs"] == ["grief-release"]
    assert body["curation"]["note"] == "for the grief pattern"
    assert body["curation"]["updated_at"] is not None


def test_no_curation_returns_null(wired):
    app_module, _db = wired
    client = app_module.app.test_client()
    r = _get(client, EMAIL, KEY)
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["curation"] is None
