# tests/test_coach_thread_coach_api.py
import json, sqlite3
from unittest import mock
import app as appmod
from dashboard import coach_connect as _cc, coach_directory as _cd, coach_threads as _ct


def _client():
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()


def _accepted(coach="c@x.com", member="m@x.com", name="Mel P"):
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        from dashboard import client_portal as _cp
        _cp.init_client_portal_table(cx); _cc.init_connect_tables(cx)
        _cd.init_coach_tables(cx); _ct.init_thread_tables(cx)
        _cp.ensure_token(cx, member, "Mel Palakiko")
        _cc.create_request(cx, coach, member, name, "help")
        rid = cx.execute("SELECT id FROM coach_requests WHERE member_email=?", (member,)).fetchone()["id"]
        _cc.set_request_status(cx, rid, "accepted")
        t = _ct.get_or_create_thread(cx, coach_email=coach, member_email=member); cx.commit()
    return t["id"]


def test_list_shows_first_name_not_email():
    c = _client(); _accepted()
    with mock.patch.object(appmod, "_coach_session_email", return_value="c@x.com"):
        d = c.get("/api/coach-thread/coach/list").get_json()
    assert d[0]["member_first_name"] == "Mel" and "m@x.com" not in json.dumps(d)


def test_unauthorized_401():
    c = _client()
    with mock.patch.object(appmod, "_coach_session_email", return_value=None):
        assert c.get("/api/coach-thread/coach/list").status_code == 401


def test_get_post_403_when_not_owner():
    c = _client(); tid = _accepted()
    with mock.patch.object(appmod, "_coach_session_email", return_value="other@x.com"):
        assert c.get(f"/api/coach-thread/coach/{tid}").status_code == 403
        assert c.post(f"/api/coach-thread/coach/{tid}/message",
                      json={"body": "hi"}).status_code == 403


def test_post_message_nudges_member():
    c = _client(); tid = _accepted()
    with mock.patch.object(appmod, "_coach_session_email", return_value="c@x.com"), \
         mock.patch.object(appmod, "send_evox_email") as sender:
        r = c.post(f"/api/coach-thread/coach/{tid}/message", json={"body": "welcome"})
    assert r.get_json()["ok"] is True and sender.called


def test_blocked_hides_history_for_coach():
    c = _client(); tid = _accepted()
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row; _ct.init_thread_tables(cx)
        _ct.post_message(cx, thread_id=tid, sender_role="member", body="hi")
        _ct.block_thread(cx, tid, "member")
    with mock.patch.object(appmod, "_coach_session_email", return_value="c@x.com"):
        d = c.get(f"/api/coach-thread/coach/{tid}").get_json()
    assert d["status"] == "blocked" and d["messages"] == []
