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
