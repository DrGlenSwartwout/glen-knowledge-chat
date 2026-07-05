# tests/test_coach_request_api.py
import sqlite3
from unittest import mock
import pytest
import app as appmod
from dashboard import coach_directory as _cd, coach_connect as _cc


@pytest.fixture(autouse=True)
def _fresh_coach_tables():
    """LOG_DB is one fixed path for the whole pytest run (DATA_DIR is set once
    by the test harness), so coach volunteers/applications/waitlist/interest
    rows from one test would otherwise leak into the next. Reset just the
    coaching-arc tables before each test (other app tables are untouched)."""
    with sqlite3.connect(appmod.LOG_DB) as cx:
        _cd.init_coach_tables(cx)
        _cc.init_connect_tables(cx)
        cx.execute("DELETE FROM coach_volunteers")
        cx.execute("DELETE FROM coach_requests")
        cx.execute("DELETE FROM coach_waitlist")
        cx.execute("DELETE FROM coaching_interest")
        cx.commit()
    yield


def _client():
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()


def _seed_member(email, *, window=True, capacity=1):
    from datetime import datetime, timezone, timedelta
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        from dashboard import evox as _ev, client_portal as _cp, coaching as _co
        _ev.init_evox_tables(cx); _cp.init_client_portal_table(cx)
        _cd.init_coach_tables(cx); _cc.init_connect_tables(cx); _co.init_coaching_table(cx)
        _cd.upsert_volunteer(cx, email="coach1@x.com", name="Cora", focus="sleep",
                             intro_video_url="/portal-asset/c.mp4", capacity=capacity, cert_ok=1)
        if window:
            now = datetime.now(timezone.utc)
            ends = (now + timedelta(days=10)).isoformat()
            cx.execute("INSERT INTO coaching_windows (email,order_id,started_at,ends_at,"
                       "source,created_at) VALUES (?,?,?,?,?,?)",
                       (email, 1, now.isoformat(), ends, "test", now.isoformat()))
        token = _ev.ensure_portal_token(cx, email, "Mel")
        cx.commit()
    return token


def _coach_ref():
    return _cc.coach_ref("coach1@x.com")


def _add_coach(email, capacity=1):
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _cd.init_coach_tables(cx)
        _cd.upsert_volunteer(cx, email=email, name="C2", focus="adrenals",
                             intro_video_url="/portal-asset/c2.mp4", capacity=capacity, cert_ok=1)
        cx.commit()
    return _cc.coach_ref(email)


def test_coaches_lists_ref_no_email():
    c = _client(); tok = _seed_member("m@x.com")
    d = c.get(f"/api/community/coaches?token={tok}").get_json()
    assert d["eligible"] and d["coaches"][0]["ref"] == _coach_ref()
    assert "email" not in d["coaches"][0]
    assert d["all_full"] is False and d["applications"] == [] and d["matched"] is False


def test_apply_then_status_shows_in_applications():
    c = _client(); tok = _seed_member("m@x.com")
    r = c.post(f"/api/community/coach-request?token={tok}",
               json={"coach_ref": _coach_ref(), "note": "sleep trouble"})
    assert r.get_json()["status"] == "pending"
    d = c.get(f"/api/community/coaches?token={tok}").get_json()
    assert {"ref": _coach_ref(), "status": "pending"} in d["applications"] and d["matched"] is False


def test_apply_multiple_coaches_allowed_same_coach_409():
    c = _client(); tok = _seed_member("m@x.com")
    ref2 = _add_coach("coach2@x.com")
    a1 = c.post(f"/api/community/coach-request?token={tok}", json={"coach_ref": _coach_ref(), "note": "x"})
    a2 = c.post(f"/api/community/coach-request?token={tok}", json={"coach_ref": ref2, "note": "y"})
    assert a1.get_json()["status"] == "pending" and a2.get_json()["status"] == "pending"  # both OK
    dup = c.post(f"/api/community/coach-request?token={tok}", json={"coach_ref": _coach_ref(), "note": "z"})
    assert dup.status_code == 409                              # same coach again


def test_apply_with_video_is_stored():
    import io
    c = _client(); tok = _seed_member("m@x.com")
    data = {"coach_ref": _coach_ref(), "note": "n",
            "video": (io.BytesIO(b"\x00\x01vid"), "me.mp4")}
    r = c.post(f"/api/community/coach-request?token={tok}", data=data,
               content_type="multipart/form-data")
    assert r.get_json()["status"] == "pending"
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row; _cc.init_connect_tables(cx)
        pend = _cc.requests_for_coach(cx, "coach1@x.com")
        assert pend[0]["member_video_url"].startswith("/portal-asset/member-")


def test_full_coach_dropped_and_all_full():
    c = _client(); tok = _seed_member("m@x.com", capacity=1)
    # fill the only coach's single slot with an accepted request from another member
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _cc.init_connect_tables(cx)
        rid = _cc.create_request(cx, "coach1@x.com", "other@x.com", "Oth", "n")
        _cc.set_request_status(cx, rid, "accepted"); cx.commit()
    d = c.get(f"/api/community/coaches?token={tok}").get_json()
    assert d["coaches"] == [] and d["all_full"] is True


def test_apply_to_full_coach_404():
    c = _client(); tok = _seed_member("m@x.com", capacity=1)
    # fill the only coach's single slot with an accepted request from another member
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _cc.init_connect_tables(cx)
        rid = _cc.create_request(cx, "coach1@x.com", "other@x.com", "Oth", "n")
        _cc.set_request_status(cx, rid, "accepted"); cx.commit()
    r = c.post(f"/api/community/coach-request?token={tok}",
               json={"coach_ref": _coach_ref(), "note": "sleep trouble"})
    assert r.status_code == 404
    assert r.get_json()["error"] == "coach_unavailable"


def test_apply_no_auth_writes_no_file():
    import io
    c = _client()
    before = list(appmod._PORTAL_ASSETS_DIR.glob("member-*.mp4"))
    data = {"coach_ref": "whatever", "note": "n",
            "video": (io.BytesIO(b"\x00\x01vid"), "me.mp4")}
    r = c.post("/api/community/coach-request?token=badtoken", data=data,
               content_type="multipart/form-data")
    assert r.status_code == 404
    after = list(appmod._PORTAL_ASSETS_DIR.glob("member-*.mp4"))
    assert len(after) == len(before)


def test_waitlist_and_interest():
    c = _client(); tok = _seed_member("m@x.com")
    assert c.post(f"/api/community/coach-waitlist?token={tok}").get_json()["ok"] is True
    with mock.patch.object(appmod, "send_evox_email") as sent:
        r = c.post(f"/api/community/coaching-interest?token={tok}", json={"tier": "glen"})
    assert r.get_json()["ok"] is True and sent.called
    bad = c.post(f"/api/community/coaching-interest?token={tok}", json={"tier": "nope"})
    assert bad.status_code == 400
