# tests/test_two_door_choose.py
import sqlite3
import app as appmod
import begin_funnel


def test_care_fork_unlock_accepted_and_recorded():
    """The care_fork analytics beacon must NOT 400 (regression: trigger was unregistered),
    must persist the door choice in the event detail, and must not create a gate or path."""
    assert "care_fork" in begin_funnel.VALID_TRIGGERS
    assert "care_fork" not in begin_funnel.GATE_TRIGGERS
    c = appmod.app.test_client()
    c.set_cookie("amg_session", "sess-care-fork-test")
    r = c.post("/begin/unlock", json={"trigger": "care_fork", "detail": "care"})
    assert r.status_code == 200, r.data
    with sqlite3.connect(appmod.LOG_DB) as cx:
        state = begin_funnel.get_state(cx, "sess-care-fork-test")
    assert "care_fork" not in (state.get("unlocked_gates") or [])
    assert (state.get("path") or "none") == "none"


def test_choose_redirects_when_flag_off(monkeypatch):
    monkeypatch.setattr(appmod, "TWO_DOOR_ENABLED", False)
    c = appmod.app.test_client()
    r = c.get("/begin/choose")
    assert r.status_code == 302
    assert r.headers["Location"].endswith("/")


def test_choose_generic_when_no_token(monkeypatch):
    monkeypatch.setattr(appmod, "TWO_DOOR_ENABLED", True)
    c = appmod.app.test_client()
    r = c.get("/begin/choose")
    assert r.status_code == 200
    assert b"window.__CHOOSE__" in r.data
    assert b'"reveal_url": "/begin"' in r.data
    assert b'"token": null' in r.data
    assert "no-store" in r.headers.get("Cache-Control", "")


def test_choose_valid_token_links_back_to_reveal(monkeypatch):
    monkeypatch.setattr(appmod, "TWO_DOOR_ENABLED", True)
    monkeypatch.setattr(appmod, "_biofield_verify_token",
                        lambda th: (True, {"email": "a@x.com"}))
    c = appmod.app.test_client()
    r = c.get("/begin/choose?token=TOK123")
    assert r.status_code == 200
    assert b'"reveal_url": "/begin/biofield/TOK123"' in r.data
    assert b'"token": "TOK123"' in r.data


def test_choose_invalid_token_falls_back_generic(monkeypatch):
    monkeypatch.setattr(appmod, "TWO_DOOR_ENABLED", True)
    monkeypatch.setattr(appmod, "_biofield_verify_token", lambda th: (False, None))
    c = appmod.app.test_client()
    r = c.get("/begin/choose?token=BAD")
    assert r.status_code == 200
    assert b'"reveal_url": "/begin"' in r.data
    assert b'"token": null' in r.data


def test_choose_payload_carries_flag_state(monkeypatch):
    monkeypatch.setattr(appmod, "TWO_DOOR_ENABLED", True)
    monkeypatch.setattr(appmod, "PREPAY_LADDER_ENABLED", True)
    monkeypatch.setattr(appmod, "PROGRAM_CARE_TASTER_ENABLED", False)
    c = appmod.app.test_client()
    r = c.get("/begin/choose")
    assert r.status_code == 200
    assert b'"prepay_enabled": true' in r.data
    assert b'"program_enabled": false' in r.data


def test_reveal_includes_choose_enabled_paid(monkeypatch):
    monkeypatch.setattr(appmod, "TWO_DOOR_ENABLED", True)
    monkeypatch.setattr(appmod, "_biofield_verify_token",
                        lambda th: (True, {"email": "a@x.com", "interpretation": {},
                                           "remedies": [], "layers": [], "id": 1}))
    monkeypatch.setattr(appmod, "is_member", lambda email=None: True)
    monkeypatch.setattr(appmod, "_biofield_unlock_flags",
                        lambda row, email: {"first_approved": True, "top_unlocked": True,
                                            "free_available": False, "paid": True})
    monkeypatch.setattr(appmod, "_resolve_ship_address", lambda e, d: {})
    monkeypatch.setattr(appmod, "_record_entry_unlock", lambda *a, **k: None)
    c = appmod.app.test_client()
    r = c.get("/begin/biofield/TOK")
    assert r.status_code == 200
    assert b'"choose_enabled": true' in r.data


def test_reveal_choose_enabled_dark_when_flag_off(monkeypatch):
    monkeypatch.setattr(appmod, "TWO_DOOR_ENABLED", False)
    monkeypatch.setattr(appmod, "_biofield_verify_token",
                        lambda th: (True, {"email": "a@x.com", "interpretation": {},
                                           "remedies": [], "layers": [], "id": 1}))
    monkeypatch.setattr(appmod, "is_member", lambda email=None: True)
    monkeypatch.setattr(appmod, "_biofield_unlock_flags",
                        lambda row, email: {"first_approved": True, "top_unlocked": True,
                                            "free_available": False, "paid": True})
    monkeypatch.setattr(appmod, "_resolve_ship_address", lambda e, d: {})
    monkeypatch.setattr(appmod, "_record_entry_unlock", lambda *a, **k: None)
    c = appmod.app.test_client()
    r = c.get("/begin/biofield/TOK")
    assert r.status_code == 200
    assert b'"choose_enabled": false' in r.data
