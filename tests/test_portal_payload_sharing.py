import importlib, sqlite3, sys
from pathlib import Path
import pytest


def _app(monkeypatch, tmp_db):
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        import app
    except Exception as e:
        pytest.skip(f"app not importable: {e}")
    monkeypatch.setattr(app, "LOG_DB", str(tmp_db))
    return app


def _seed_portal(tmp_db, email="member@ex.com"):
    from dashboard import client_portal as cp
    with sqlite3.connect(str(tmp_db)) as cx:
        cp.init_client_portal_table(cx)
        token, _ = cp.upsert_portal(cx, email, "M", {})
    return token


def test_key_absent_when_flag_off(monkeypatch, tmp_db):
    monkeypatch.delenv("DATA_SHARING_REWARD_ENABLED", raising=False)
    app = _app(monkeypatch, tmp_db)
    token = _seed_portal(tmp_db)
    body = app.app.test_client().get(f"/api/portal/{token}").get_json()
    assert "data_sharing" not in body


def test_key_present_when_flag_on(monkeypatch, tmp_db):
    monkeypatch.setenv("DATA_SHARING_REWARD_ENABLED", "1")
    app = _app(monkeypatch, tmp_db)
    token = _seed_portal(tmp_db)
    body = app.app.test_client().get(f"/api/portal/{token}").get_json()
    assert "data_sharing" in body
    from dashboard import data_sharing
    assert set(body["data_sharing"]["toggles"].keys()) == set(data_sharing.TOGGLE_MAP.keys())
    assert body["data_sharing"]["tier"] == 0
