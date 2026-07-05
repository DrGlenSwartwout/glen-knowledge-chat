import sqlite3
from unittest import mock
import app as appmod
from dashboard import coach_connect as _cc, coach_threads as _ct


def _client():
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()


def _seed(coach="c@x.com", member="m@x.com"):
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _cc.init_connect_tables(cx); _ct.init_thread_tables(cx)
        _cc.create_request(cx, coach, member, "Mel P", "help")
        rid = cx.execute("SELECT id FROM coach_requests WHERE member_email=?", (member,)).fetchone()["id"]
        _cc.set_request_status(cx, rid, "accepted")
        t = _ct.get_or_create_thread(cx, coach_email=coach, member_email=member)
        _ct.post_message(cx, thread_id=t["id"], sender_role="member", body="secret")
        _ct.block_thread(cx, t["id"], "member"); cx.commit()
    return t["id"]


def test_all_owner_routes_401_without_key():
    c = _client()
    with mock.patch.object(appmod, "_portal_console_ok", return_value=False):
        assert c.get("/api/console/coach-threads").status_code == 401
        assert c.get("/api/console/coach-threads/1").status_code == 401
        assert c.post("/api/console/coach-threads/1/unmatch").status_code == 401


def test_transcript_shows_blocked_history():
    c = _client(); tid = _seed()
    with mock.patch.object(appmod, "_portal_console_ok", return_value=True):
        d = c.get(f"/api/console/coach-threads/{tid}").get_json()
    assert d["status"] == "blocked"
    assert any(m["body"] == "secret" for m in d["messages"])   # owner sees hidden history


def test_unmatch_blocks_and_ends_pairing():
    c = _client()
    # fresh active pair (not pre-blocked)
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _cc.init_connect_tables(cx); _ct.init_thread_tables(cx)
        _cc.create_request(cx, "c2@x.com", "m2@x.com", "Ana", "hi")
        rid = cx.execute("SELECT id FROM coach_requests WHERE member_email='m2@x.com'").fetchone()["id"]
        _cc.set_request_status(cx, rid, "accepted")
        t = _ct.get_or_create_thread(cx, coach_email="c2@x.com", member_email="m2@x.com"); cx.commit()
    with mock.patch.object(appmod, "_portal_console_ok", return_value=True), \
         mock.patch.object(appmod, "send_evox_email"):
        assert c.post(f"/api/console/coach-threads/{t['id']}/unmatch").get_json()["ok"] is True
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row; _cc.init_connect_tables(cx); _ct.init_thread_tables(cx)
        assert _ct.get_thread(cx, t["id"])["status"] == "blocked"
        assert _cc.accepted_count(cx, "c2@x.com") == 0


def test_resolve_report_clears_flag():
    c = _client()
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row; _cc.init_connect_tables(cx); _ct.init_thread_tables(cx)
        t = _ct.get_or_create_thread(cx, coach_email="c3@x.com", member_email="m3@x.com")
        _ct.report_thread(cx, thread_id=t["id"], reporter_role="member", reason="x"); cx.commit()
    with mock.patch.object(appmod, "_portal_console_ok", return_value=True):
        assert c.post(f"/api/console/coach-threads/{t['id']}/resolve-report").get_json()["ok"] is True
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row; _ct.init_thread_tables(cx)
        assert _ct.get_thread(cx, t["id"])["reported"] == 0
