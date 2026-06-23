"""Route tests for the console practitioner-admin page + API.

Supabase is unavailable in tests, so the DB-touching admin functions are
monkeypatched; we assert the routes gate on the console key, validate input,
and orchestrate create / geocode / invite / edit correctly.
"""
import pytest


@pytest.fixture
def client(monkeypatch, tmp_path):
    import app as appmod
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    appmod._init_auth_tables()
    monkeypatch.setattr(appmod, "send_magic_link_email", lambda *a, **k: ("test", None))
    monkeypatch.setattr(appmod, "_send_inquiry_email", lambda *a, **k: True)
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client(), appmod


def _key(appmod):
    return appmod.CONSOLE_SECRET or ""


def test_page_served(client):
    c, _ = client
    r = c.get("/console/practitioners")
    assert r.status_code == 200


def test_list_console_gated(client):
    c, appmod = client
    r = c.get("/api/console/practitioners")  # no key
    if appmod.CONSOLE_SECRET:
        assert r.status_code == 401
    else:
        assert r.status_code == 200


def test_list_returns_rows(client, monkeypatch):
    c, appmod = client
    from dashboard import practitioner_admin as pa
    monkeypatch.setattr(pa, "list_practitioners", lambda q=None: [
        {"id": "p1", "name": "Ashley King", "email": "a@b.com", "portal_role": "coach",
         "credentials": "Health Coach", "modules_completed": 0, "wallet_balance_cents": 0,
         "wholesale_unlocked_at": "2026-06-23T00:00:00", "application_status": None,
         "show_contact": True, "city": "Austin", "state": "TX"}])
    r = c.get("/api/console/practitioners?key=" + _key(appmod))
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["rows"][0]["id"] == "p1"
    assert body["rows"][0]["wholesale_access"] is True
    assert body["rows"][0]["section"] == "coach"


def test_create_validation_error_returns_400(client):
    c, appmod = client
    r = c.post("/api/console/practitioners?key=" + _key(appmod),
               json={"email": "bad", "name": "X", "role": "coach"})
    assert r.status_code == 400
    assert "email" in r.get_json()["error"].lower()


def test_create_calls_create_and_invite(client, monkeypatch):
    c, appmod = client
    from dashboard import practitioner_admin as pa
    captured = {}
    monkeypatch.setattr(pa, "create_or_update_practitioner",
                        lambda clean, **kw: captured.update({"clean": clean}) or "pid-1")
    geo = {}
    monkeypatch.setattr(pa, "geocode_and_set_location",
                        lambda pid, city, state: geo.update({"pid": pid, "city": city}))
    sent = {}
    monkeypatch.setattr(appmod, "_send_practitioner_magic_link",
                        lambda *a, **k: sent.update({"called": True, "args": a}))
    r = c.post("/api/console/practitioners?key=" + _key(appmod), json={
        "email": "aking@yahoo.com", "name": "Ashley King", "role": "coach",
        "credentials": "Health Coach", "wholesale_access": True, "level": 0,
        "list_in_finder": True, "city": "Austin", "state": "TX", "send_invite": True})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True and body["practitioner_id"] == "pid-1"
    assert body["invite_sent"] is True
    assert captured["clean"]["wholesale_access"] is True
    assert captured["clean"]["level"] == 0
    assert geo["city"] == "Austin"
    assert sent.get("called") is True


def test_create_skips_invite_when_not_requested(client, monkeypatch):
    c, appmod = client
    from dashboard import practitioner_admin as pa
    monkeypatch.setattr(pa, "create_or_update_practitioner", lambda clean, **kw: "pid-2")
    sent = {}
    monkeypatch.setattr(appmod, "_send_practitioner_magic_link",
                        lambda *a, **k: sent.update({"called": True}))
    r = c.post("/api/console/practitioners?key=" + _key(appmod), json={
        "email": "x@y.com", "name": "No Invite", "role": "licensed", "send_invite": False})
    assert r.status_code == 200
    assert r.get_json()["invite_sent"] is False
    assert sent.get("called") is None


def test_edit_level_access_dispatch(client, monkeypatch):
    c, appmod = client
    from dashboard import practitioner_admin as pa
    calls = {}
    monkeypatch.setattr(pa, "set_level_and_access",
                        lambda pid, level, wholesale_access: calls.update(
                            {"pid": pid, "level": level, "access": wholesale_access}))
    r = c.post("/api/console/practitioners/p9/edit?key=" + _key(appmod),
               json={"action": "level_access", "level": 3, "wholesale_access": False})
    assert r.status_code == 200
    assert calls == {"pid": "p9", "level": 3, "access": False}


def test_edit_finder_dispatch(client, monkeypatch):
    c, appmod = client
    from dashboard import practitioner_admin as pa
    calls = {}
    monkeypatch.setattr(pa, "set_finder_visibility",
                        lambda pid, show: calls.update({"pid": pid, "show": show}))
    r = c.post("/api/console/practitioners/p9/edit?key=" + _key(appmod),
               json={"action": "finder", "show": True})
    assert r.status_code == 200
    assert calls == {"pid": "p9", "show": True}


def test_edit_location_dispatch(client, monkeypatch):
    c, appmod = client
    from dashboard import practitioner_admin as pa
    calls = {}
    monkeypatch.setattr(pa, "geocode_and_set_location",
                        lambda pid, city, state: calls.update(
                            {"pid": pid, "city": city, "state": state}))
    r = c.post("/api/console/practitioners/p9/edit?key=" + _key(appmod),
               json={"action": "location", "city": "Denver", "state": "CO"})
    assert r.status_code == 200
    assert calls == {"pid": "p9", "city": "Denver", "state": "CO"}


def test_edit_resend_invite_dispatch(client, monkeypatch):
    c, appmod = client
    sent = {}
    monkeypatch.setattr(appmod, "_send_practitioner_magic_link",
                        lambda *a, **k: sent.update({"called": True, "args": a}))
    r = c.post("/api/console/practitioners/p9/edit?key=" + _key(appmod),
               json={"action": "resend_invite", "email": "a@b.com", "name": "Ashley"})
    assert r.status_code == 200
    assert r.get_json()["sent"] is True
    assert sent.get("called") is True


def test_edit_unknown_action_400(client, monkeypatch):
    c, appmod = client
    r = c.post("/api/console/practitioners/p9/edit?key=" + _key(appmod),
               json={"action": "explode"})
    assert r.status_code == 400


def test_edit_console_gated(client):
    c, appmod = client
    r = c.post("/api/console/practitioners/p9/edit", json={"action": "finder", "show": True})
    if appmod.CONSOLE_SECRET:
        assert r.status_code == 401
