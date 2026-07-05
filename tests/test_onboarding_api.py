import sqlite3
from unittest import mock
import app as appmod


def _client():
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()


def _seed_member(email):
    """Mint a REAL portal token so _evox_ident/resolve_identity resolves it.
    The client_portal token is sha256-hashed at rest, so faking the column will
    not resolve — use the real minting path and return the plaintext token."""
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        from dashboard import evox as _ev, client_portal as _cp
        _ev.init_evox_tables(cx)
        _cp.init_client_portal_table(cx)
        token = _ev.ensure_portal_token(cx, email, "")
        cx.commit()
    return token


def test_state_member_no_booking():
    c = _client()
    tok = _seed_member("m1@x.com")
    with mock.patch.object(appmod, "_is_paid_member", return_value=True):
        r = c.get(f"/api/onboarding/state?token={tok}")
    assert r.status_code == 200
    d = r.get_json()
    assert d["eligible"] is True
    assert d["booked"] is None


def test_state_non_member():
    c = _client()
    tok = _seed_member("m2@x.com")
    with mock.patch.object(appmod, "_is_paid_member", return_value=False):
        r = c.get(f"/api/onboarding/state?token={tok}")
    assert r.get_json()["eligible"] is False


def test_state_bad_token():
    c = _client()
    r = c.get("/api/onboarding/state?token=nope")
    assert r.status_code == 404


def test_availability_non_member_403():
    c = _client()
    tok = _seed_member("m3@x.com")
    with mock.patch.object(appmod, "_is_paid_member", return_value=False):
        r = c.get(f"/api/onboarding/availability?token={tok}")
    assert r.status_code == 403


def test_book_free_then_second_is_blocked():
    c = _client()
    tok = _seed_member("m4@x.com")
    with mock.patch.object(appmod, "_is_paid_member", return_value=True), \
         mock.patch.object(appmod, "send_evox_email") as send:
        avail = c.get(f"/api/onboarding/availability?token={tok}").get_json()["slots"]
        assert avail, "expected at least one Rae slot"
        slot = avail[0]
        r1 = c.post(f"/api/onboarding/book?token={tok}", json={"start_ts": slot})
        assert r1.status_code == 200 and r1.get_json()["ok"] is True
        assert send.called  # confirmation attempted
        # once-per-member: a second booking attempt is refused
        r2 = c.post(f"/api/onboarding/book?token={tok}", json={"start_ts": slot})
        assert r2.status_code == 409
        assert r2.get_json()["error"] == "already_booked"
        # and availability now returns no slots
        assert c.get(f"/api/onboarding/availability?token={tok}").get_json()["slots"] == []


def test_book_non_member_403():
    c = _client()
    tok = _seed_member("m5@x.com")
    with mock.patch.object(appmod, "_is_paid_member", return_value=False):
        r = c.post(f"/api/onboarding/book?token={tok}", json={"start_ts": "2026-07-10T09:00:00"})
    assert r.status_code == 403
