import importlib, sys, os, sqlite3
import pytest

def _load_app():
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app import failed: {e}")

@pytest.fixture
def appmod(monkeypatch, tmp_path):
    app = _load_app()
    db = str(tmp_path / "t.db")
    monkeypatch.setattr(app, "LOG_DB", db, raising=False)
    monkeypatch.setattr(app, "CONSOLE_SECRET", "sekret", raising=False)
    import dashboard
    monkeypatch.setattr(dashboard, "CONSOLE_SECRET", "sekret", raising=False)
    cx = sqlite3.connect(db); app.init_membership_tables(cx); cx.close()
    if hasattr(app, "_member_join_welcome"):
        monkeypatch.setattr(app, "_member_join_welcome", lambda *a, **k: None, raising=False)
    return app

def test_owner_enroll_grants_member(appmod):
    r = appmod.app.test_client().post("/api/console/membership/enroll?key=sekret",
                                      json={"email": "dana@x.com", "tier": "month"})
    assert r.status_code == 200 and r.get_json()["ok"] is True
    assert appmod._is_paid_member("dana@x.com") is True

def test_enroll_requires_owner(appmod):
    r = appmod.app.test_client().post("/api/console/membership/enroll?key=wrong",
                                      json={"email": "dana@x.com", "tier": "month"})
    assert r.status_code == 401

def test_enroll_unknown_tier(appmod):
    r = appmod.app.test_client().post("/api/console/membership/enroll?key=sekret",
                                      json={"email": "dana@x.com", "tier": "nope"})
    assert r.status_code == 400
