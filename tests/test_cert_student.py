"""Tests for the cert-student admin action: helper upsert_cert_student +
console-gated POST /api/cert/student route."""

import pytest


# ── fake Supabase cursor (modeled on test_practitioner_portal.py) ─────────────

class _FakeCur:
    """Queues fetchone() returns; branches on SQL text. The first SELECT id
    return is configurable (existing vs new); the INSERT ... RETURNING id then
    yields a fresh id."""

    def __init__(self, select_row):
        self._select_row = select_row
        self.executed = []          # list of (kind, params)
        self._r = None

    def execute(self, sql, params=()):
        s = " ".join(sql.split())
        if s.startswith("SELECT id FROM practitioners WHERE lower(email)"):
            self._r = self._select_row
            self.executed.append(("SELECT", list(params)))
        elif "INSERT INTO practitioners" in s and "RETURNING id" in s:
            self._r = {"id": "new-uuid"}
            self.executed.append(("INSERT", list(params)))
        elif s.startswith("UPDATE practitioners SET"):
            self._r = None
            self.executed.append(("UPDATE", list(params)))
        else:
            self._r = None

    def fetchone(self):
        return self._r

    def kinds(self):
        return [k for k, _ in self.executed]


class _FakeCtx:
    def __init__(self, cur): self.cur = cur
    def __enter__(self): return self.cur
    def __exit__(self, *a): return False


def _patch_cursor(monkeypatch, select_row):
    cur = _FakeCur(select_row)
    import db_supabase
    monkeypatch.setattr(db_supabase, "supabase_cursor", lambda: _FakeCtx(cur))
    return cur


# ── helper: insert path ───────────────────────────────────────────────────────

def test_helper_insert_path(monkeypatch):
    cur = _patch_cursor(monkeypatch, select_row=None)  # email not found -> insert
    from dashboard.practitioner_portal import upsert_cert_student
    pid, mc = upsert_cert_student("Mona@x.com", name="Mona", modules_completed=1)
    assert (pid, mc) == ("new-uuid", 1)
    assert "INSERT" in cur.kinds()
    assert "UPDATE" not in cur.kinds()


# ── helper: update path ───────────────────────────────────────────────────────

def test_helper_update_path(monkeypatch):
    cur = _patch_cursor(monkeypatch, select_row={"id": "existing"})
    from dashboard.practitioner_portal import upsert_cert_student
    pid, mc = upsert_cert_student("od@x.com", name="OD", modules_completed=3)
    assert (pid, mc) == ("existing", 3)
    assert "UPDATE" in cur.kinds()
    assert "INSERT" not in cur.kinds()


# ── helper: clamping ──────────────────────────────────────────────────────────

def test_helper_clamps_high(monkeypatch):
    _patch_cursor(monkeypatch, select_row={"id": "x"})
    from dashboard.practitioner_portal import upsert_cert_student
    _, mc = upsert_cert_student("a@x.com", modules_completed=99)
    assert mc == 12


def test_helper_clamps_negative(monkeypatch):
    _patch_cursor(monkeypatch, select_row={"id": "x"})
    from dashboard.practitioner_portal import upsert_cert_student
    _, mc = upsert_cert_student("a@x.com", modules_completed=-5)
    assert mc == 0


# ── route ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def client(monkeypatch):
    import app as appmod
    monkeypatch.setattr(appmod._pp, "upsert_cert_student",
                        lambda email, **kw: ("pid1", 1))
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client(), appmod


def test_route_unauthorized(client):
    c, appmod = client
    if not appmod.CONSOLE_SECRET:
        pytest.skip("CONSOLE_SECRET not set")
    r = c.post("/api/cert/student", json={"email": "a@x.com"})
    assert r.status_code == 401


def test_route_ok(client):
    c, appmod = client
    key = appmod.CONSOLE_SECRET or ""
    r = c.post("/api/cert/student?key=" + key,
               json={"email": "a@x.com", "name": "A", "modules_completed": 1})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["practitioner_id"] == "pid1"
    assert body["modules_completed"] == 1


def test_route_missing_email(client):
    c, appmod = client
    key = appmod.CONSOLE_SECRET or ""
    r = c.post("/api/cert/student?key=" + key, json={"name": "A"})
    assert r.status_code == 400
