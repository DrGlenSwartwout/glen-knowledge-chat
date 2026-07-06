"""Clean-slate re-pairing (coaching arc slice 3 fast-follow): when a previously
blocked pair is re-accepted, participants get a fresh thread (new epoch) while the
owner transcript retains the full prior history for moderation/audit."""
import sqlite3
from unittest import mock
import app as appmod
from dashboard import coach_connect as _cc, coach_directory as _cd, coach_threads as _ct


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    _ct.init_thread_tables(cx)
    _cc.init_connect_tables(cx)
    return cx


# ---- store-level ----

def test_reactivate_clean_slate_for_participants_full_for_owner():
    cx = _cx()
    t = _ct.get_or_create_thread(cx, coach_email="c@x.com", member_email="m@x.com")
    _ct.post_message(cx, thread_id=t["id"], sender_role="coach", body="old message")
    _ct.block_thread(cx, t["id"], "member")
    assert _ct.reactivate_thread(cx, "C@x.com", "M@x.com") is True   # case-insensitive
    t2 = _ct.get_thread(cx, t["id"])
    assert t2["status"] == "active" and t2["blocked_by"] is None and t2["active_epoch"] == 2
    assert _ct.messages(cx, t["id"], epoch=t2["active_epoch"]) == []          # participant clean slate
    assert any(m["body"] == "old message" for m in _ct.messages(cx, t["id"]))  # owner keeps history
    _ct.post_message(cx, thread_id=t["id"], sender_role="coach", body="fresh start")
    part = _ct.messages(cx, t["id"], epoch=t2["active_epoch"])
    assert [m["body"] for m in part] == ["fresh start"]


def test_reactivate_noop_when_no_thread():
    cx = _cx()
    assert _ct.reactivate_thread(cx, "c@x.com", "none@x.com") is False


def test_unread_counts_only_current_epoch():
    cx = _cx()
    t = _ct.get_or_create_thread(cx, coach_email="c@x.com", member_email="m@x.com")
    _ct.post_message(cx, thread_id=t["id"], sender_role="coach", body="old")
    _ct.reactivate_thread(cx, "c@x.com", "m@x.com")            # bump to epoch 2
    assert _ct.unread_count(cx, t["id"], "member") == 0         # old-epoch msg not counted
    _ct.post_message(cx, thread_id=t["id"], sender_role="coach", body="new")
    assert _ct.unread_count(cx, t["id"], "member") == 1


# ---- route-level ----

def _client():
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()


def _matched(member="m@x.com", coach="c@x.com"):
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        from dashboard import evox as _ev, client_portal as _cp
        _ev.init_evox_tables(cx); _cp.init_client_portal_table(cx)
        _cc.init_connect_tables(cx); _cd.init_coach_tables(cx); _ct.init_thread_tables(cx)
        _cd.upsert_volunteer(cx, email=coach, name="Coach Kai", focus="terrain",
                             intro_video_url="", capacity=5, cert_ok=1)
        _cc.create_request(cx, coach, member, "Mel P", "help")
        rid = cx.execute("SELECT id FROM coach_requests WHERE member_email=?", (member,)).fetchone()["id"]
        _cc.set_request_status(cx, rid, "accepted")
        tok = _ev.ensure_portal_token(cx, member, "Mel"); cx.commit()
    return tok


def test_reaccept_same_coach_clean_slate_owner_keeps_history():
    c = _client(); tok = _matched()
    c.post(f"/api/coach-thread/member/message?token={tok}", json={"body": "first round hello"})
    with mock.patch.object(appmod, "send_evox_email"):
        c.post(f"/api/coach-thread/member/block?token={tok}")     # ends pairing + blocks thread
    # member re-applies to the SAME coach; coach re-accepts
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row; _cc.init_connect_tables(cx)
        _cc.create_request(cx, "c@x.com", "m@x.com", "Mel P", "second chance")
        rid = cx.execute("SELECT id FROM coach_requests WHERE member_email='m@x.com' "
                         "AND status='pending'").fetchone()["id"]
        cx.commit()
    with mock.patch.object(appmod, "_coach_session_email", return_value="c@x.com"):
        r = c.post("/api/practitioner/coach-request/respond",
                   json={"request_id": rid, "accept": True})
    assert r.get_json().get("status") == "accepted"
    d = c.get(f"/api/coach-thread/member?token={tok}").get_json()
    assert d["status"] == "active" and d["can_post"] is True and d["messages"] == []
    r2 = c.post(f"/api/coach-thread/member/message?token={tok}", json={"body": "round two"})
    assert r2.get_json()["ok"] is True
    d2 = c.get(f"/api/coach-thread/member?token={tok}").get_json()
    assert [m["body"] for m in d2["messages"]] == ["round two"]      # only the fresh session
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row; _ct.init_thread_tables(cx)
        tid = _ct.thread_for_pair(cx, "c@x.com", "m@x.com")["id"]
    with mock.patch.object(appmod, "_portal_console_ok", return_value=True):
        td = c.get(f"/api/console/coach-threads/{tid}").get_json()
    bodies = [m["body"] for m in td["messages"]]
    assert "first round hello" in bodies and "round two" in bodies     # owner audit intact
