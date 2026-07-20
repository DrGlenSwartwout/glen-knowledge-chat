# tests/test_practitioner_profile_routes.py
import os
import pytest
if not os.environ.get("PINECONE_API_KEY"):
    pytest.skip("needs doppler env for import app", allow_module_level=True)
import app as appmod


class _FakeCur:
    def __init__(self, row): self._row = row; self.calls = []
    def execute(self, sql, params=()): self.calls.append((" ".join(sql.split()), list(params)))
    def fetchone(self): return self._row
    def close(self): pass


class _FakeCtx:
    def __init__(self, cur): self.cur = cur
    def __enter__(self): return self.cur
    def __exit__(self, *a): return False


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(appmod, "_practitioner_session_pid", lambda: "pid-123")
    from dashboard import practitioner_settings as _ps
    monkeypatch.setattr(_ps, "init_settings_table", lambda cx: None)
    monkeypatch.setattr(_ps, "get_settings",
                        lambda cx, pid: {"branding": {}, "pricing": {}, "chat_enabled": False})
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()


def test_get_settings_includes_profile(client, monkeypatch):
    row = {"bio": "I heal", "photo_url": "https://x/p.jpg", "specialties": ["Acupuncture"],
           "city": "Hilo", "state": "HI", "accepting_new_patients": True,
           "profile_self_authored_at": "2026-07-20T00:00:00Z", "show_contact": False}
    import db_supabase
    monkeypatch.setattr(db_supabase, "supabase_cursor", lambda: _FakeCtx(_FakeCur(row)))
    r = client.get("/api/practitioner/settings")
    assert r.status_code == 200
    prof = r.get_json()["profile"]
    assert prof["bio"] == "I heal"
    assert prof["services"] == ["Acupuncture"]
    assert prof["self_authored"] is True


def test_get_settings_profile_omitted_when_supabase_down(client, monkeypatch):
    """Fail-soft: a Supabase error during the profile read must not 500 the
    settings page — the profile key is simply omitted, other keys remain."""
    import db_supabase
    def _boom():
        raise RuntimeError("supabase down")
    monkeypatch.setattr(db_supabase, "supabase_cursor", _boom)
    r = client.get("/api/practitioner/settings")
    assert r.status_code == 200
    body = r.get_json()
    assert "profile" not in body
    assert body["ok"] is True


def test_post_saves_profile_when_present(client, monkeypatch):
    saved = {}
    from dashboard import practitioner_profile as _pp
    def _fake_save(pid, profile):
        saved["pid"] = pid; saved["profile"] = profile
        return {"bio": "I heal", "photo_url": "", "services": [], "city": "Hilo",
                "state": "HI", "accepting_clients": True}
    monkeypatch.setattr(_pp, "save_profile", _fake_save)
    r = client.post("/api/practitioner/settings", json={
        "profile": {"bio": "I heal", "city": "Hilo", "state": "HI"}})
    assert r.status_code == 200
    assert saved["pid"] == "pid-123"
    assert r.get_json()["profile"]["bio"] == "I heal"


def test_post_without_profile_does_not_touch_it(client, monkeypatch):
    from dashboard import practitioner_profile as _pp
    called = {"n": 0}
    monkeypatch.setattr(_pp, "save_profile", lambda *a, **k: called.__setitem__("n", called["n"] + 1))
    r = client.post("/api/practitioner/settings", json={"branding": {}})
    assert r.status_code == 200
    assert called["n"] == 0
    assert "profile" not in r.get_json()


def test_post_profile_bad_bio_returns_400(client, monkeypatch):
    from dashboard import practitioner_profile as _pp
    def _boom(pid, profile): raise ValueError("bio exceeds 600 characters")
    monkeypatch.setattr(_pp, "save_profile", _boom)
    r = client.post("/api/practitioner/settings", json={"profile": {"bio": "x" * 601}})
    assert r.status_code == 400
