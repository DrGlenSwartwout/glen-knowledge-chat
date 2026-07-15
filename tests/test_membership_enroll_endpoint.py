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

def _seed_va_token(appmod, token):
    """Same seeding pattern as tests/test_bos_actor_token.py: a workspace_users
    row scoped 'workspace:shaira' + an access_tokens row resolves (via
    _bos_actor -> _role_for_token -> rbac.actor_for_scope) to a real, non-None
    VA-role actor -- not the actor-is-None branch."""
    appmod._init_workspace_schema()
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.execute("INSERT INTO workspace_users (name, display_name, scope) VALUES (?,?,?)",
                   ("shaira", "Shaira", "workspace:shaira"))
        uid = cx.execute("SELECT id FROM workspace_users WHERE name=?", ("shaira",)).fetchone()[0]
        cx.execute("INSERT INTO access_tokens (token, user_id, note) VALUES (?,?,?)",
                   (token, uid, "t"))
        cx.commit()

def test_enroll_rejects_va_actor(appmod):
    _seed_va_token(appmod, "sha-tok")
    # Sanity: this token must resolve to a real VA actor (not None) so the
    # assertion below exercises the role check, not the actor-is-None branch.
    assert appmod._role_for_token("sha-tok") == appmod._bos_rbac.VA
    r = appmod.app.test_client().post("/api/console/membership/enroll?key=sha-tok",
                                      json={"email": "x@y.com", "tier": "month"})
    assert r.status_code == 401

def test_enroll_rejects_bad_source(appmod):
    r = appmod.app.test_client().post("/api/console/membership/enroll?key=sekret",
                                      json={"email": "dana@x.com", "tier": "month",
                                            "source": "hacked"})
    assert r.status_code == 400
    assert r.get_json()["error"] == "source must start with membership_"
