"""Tests for the cert show-contact admin action: console-gated POST
/api/cert/show-contact route. Mirrors test_cert_student.py's fake-cursor +
client fixture style."""

import pytest


# ── fake Supabase cursor ──────────────────────────────────────────────────────

class _FakeCur:
    """Records executed (kind, params) and reports a rowcount for UPDATEs."""

    def __init__(self):
        self.executed = []          # list of (kind, params)
        self.rowcount = 0

    def execute(self, sql, params=()):
        s = " ".join(sql.split())
        if s.startswith("UPDATE practitioners SET show_contact"):
            self.rowcount = 1
            self.executed.append(("UPDATE", list(params)))
        else:
            self.executed.append(("OTHER", list(params)))

    def kinds(self):
        return [k for k, _ in self.executed]


class _FakeCtx:
    def __init__(self, cur): self.cur = cur
    def __enter__(self): return self.cur
    def __exit__(self, *a): return False


def _patch_cursor(monkeypatch):
    cur = _FakeCur()
    import db_supabase
    monkeypatch.setattr(db_supabase, "supabase_cursor", lambda: _FakeCtx(cur))
    return cur


# ── route ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    import app as appmod
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client(), appmod


def test_route_unauthorized(client):
    c, appmod = client
    if not appmod.CONSOLE_SECRET:
        pytest.skip("CONSOLE_SECRET not set")
    r = c.post("/api/cert/show-contact", json={"email": "a@x.com", "show": True})
    assert r.status_code == 401


def test_route_missing_email(client):
    c, appmod = client
    key = appmod.CONSOLE_SECRET or ""
    r = c.post("/api/cert/show-contact?key=" + key, json={"show": True})
    assert r.status_code == 400


def test_route_ok_sets_show_contact_true(client, monkeypatch):
    c, appmod = client
    cur = _patch_cursor(monkeypatch)
    key = appmod.CONSOLE_SECRET or ""
    r = c.post("/api/cert/show-contact?key=" + key,
               json={"email": "od@x.com", "show": True})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["email"] == "od@x.com"
    assert body["show_contact"] is True
    assert body["updated"] == 1
    # An UPDATE ... show_contact ran with (True, email)
    assert "UPDATE" in cur.kinds()
    update_params = next(p for k, p in cur.executed if k == "UPDATE")
    assert update_params == [True, "od@x.com"]
