"""Tests for the self-serve show_contact toggle on the practitioner settings
routes (GET/POST /api/practitioner/settings). The show_contact flag lives on the
Supabase practitioners row, not in the SQLite practitioner_settings table — so it
is read/written directly via db_supabase.supabase_cursor."""

import pytest


# ── fake Supabase cursor ──────────────────────────────────────────────────────

class _FakeCur:
    """Records every execute() and serves a configurable row for the
    SELECT show_contact read."""

    def __init__(self, show_contact_row):
        self._show_contact_row = show_contact_row
        self.executed = []          # list of (kind, sql, params)
        self._r = None

    def execute(self, sql, params=()):
        s = " ".join(sql.split())
        if s.startswith("SELECT show_contact FROM practitioners"):
            self._r = self._show_contact_row
            self.executed.append(("SELECT", s, list(params)))
        elif s.startswith("UPDATE practitioners SET show_contact"):
            self._r = None
            self.executed.append(("UPDATE", s, list(params)))
        else:
            self._r = None
            self.executed.append(("OTHER", s, list(params)))

    def fetchone(self):
        return self._r

    def kinds(self):
        return [k for k, _, _ in self.executed]

    def update_calls(self):
        return [(s, p) for k, s, p in self.executed if k == "UPDATE"]


class _FakeCtx:
    def __init__(self, cur): self.cur = cur
    def __enter__(self): return self.cur
    def __exit__(self, *a): return False


def _patch_cursor(monkeypatch, show_contact_row):
    cur = _FakeCur(show_contact_row)
    import db_supabase
    monkeypatch.setattr(db_supabase, "supabase_cursor", lambda: _FakeCtx(cur))
    return cur


# ── client fixture: signed-in practitioner + stubbed sqlite settings ──────────

@pytest.fixture
def client(monkeypatch):
    import app as appmod
    # Pretend a practitioner is signed in.
    monkeypatch.setattr(appmod, "_practitioner_session_pid", lambda: "pid-123")
    # Stub the SQLite-backed branding/pricing layer so the route doesn't touch
    # the real practitioner_settings table.
    from dashboard import practitioner_settings as _ps
    monkeypatch.setattr(_ps, "init_settings_table", lambda cx: None)
    monkeypatch.setattr(_ps, "set_branding", lambda *a, **kw: None)
    monkeypatch.setattr(_ps, "set_pricing", lambda *a, **kw: None)
    monkeypatch.setattr(_ps, "get_settings",
                        lambda cx, pid: {"branding": {}, "pricing": {}, "chat_enabled": False})
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client(), appmod


# ── GET returns show_contact ──────────────────────────────────────────────────

def test_get_returns_show_contact_true(client, monkeypatch):
    c, appmod = client
    _patch_cursor(monkeypatch, show_contact_row={"show_contact": True})
    r = c.get("/api/practitioner/settings")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["show_contact"] is True


def test_get_not_signed_in_401(client, monkeypatch):
    c, appmod = client
    monkeypatch.setattr(appmod, "_practitioner_session_pid", lambda: None)
    r = c.get("/api/practitioner/settings")
    assert r.status_code == 401


# ── POST with show_contact issues the UPDATE ──────────────────────────────────

def test_post_show_contact_true_issues_update(client, monkeypatch):
    c, appmod = client
    cur = _patch_cursor(monkeypatch, show_contact_row={"show_contact": True})
    r = c.post("/api/practitioner/settings", json={"show_contact": True})
    assert r.status_code == 200
    updates = cur.update_calls()
    assert len(updates) == 1, f"expected exactly one show_contact UPDATE, got {cur.kinds()}"
    sql, params = updates[0]
    assert "UPDATE practitioners SET show_contact" in sql
    assert True in params
    assert "pid-123" in params


# ── POST without show_contact does NOT update it ──────────────────────────────

def test_post_without_show_contact_no_update(client, monkeypatch):
    c, appmod = client
    cur = _patch_cursor(monkeypatch, show_contact_row={"show_contact": False})
    r = c.post("/api/practitioner/settings",
               json={"branding": {"practice_name": "X"}, "pricing": {}})
    assert r.status_code == 200
    assert "UPDATE" not in cur.kinds(), \
        f"show_contact must not be touched when key absent, got {cur.kinds()}"
