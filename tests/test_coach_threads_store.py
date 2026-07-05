import sqlite3
from dashboard import coach_threads as _ct
from dashboard import coach_connect as _cc


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    _ct.init_thread_tables(cx)
    _cc.init_connect_tables(cx)
    return cx


def test_get_or_create_idempotent():
    cx = _cx()
    t1 = _ct.get_or_create_thread(cx, coach_email="C@x.com", member_email="M@x.com")
    t2 = _ct.get_or_create_thread(cx, coach_email="c@x.com", member_email="m@x.com")
    assert t1["id"] == t2["id"] and t1["status"] == "active" and t1["source"] == "coaching"


def test_post_and_messages_chronological():
    cx = _cx()
    t = _ct.get_or_create_thread(cx, coach_email="c@x.com", member_email="m@x.com")
    _ct.post_message(cx, thread_id=t["id"], sender_role="coach", body="hello")
    _ct.post_message(cx, thread_id=t["id"], sender_role="member", body="hi back")
    ms = _ct.messages(cx, t["id"])
    assert [m["sender_role"] for m in ms] == ["coach", "member"]
    assert ms[0]["body"] == "hello"


def test_unread_counts_only_other_role_newer():
    cx = _cx()
    t = _ct.get_or_create_thread(cx, coach_email="c@x.com", member_email="m@x.com")
    _ct.post_message(cx, thread_id=t["id"], sender_role="coach", body="a")
    _ct.post_message(cx, thread_id=t["id"], sender_role="coach", body="b")
    assert _ct.unread_count(cx, t["id"], "member") == 2   # 2 from coach, unread by member
    assert _ct.unread_count(cx, t["id"], "coach") == 0     # own messages don't count
    _ct.mark_read(cx, t["id"], "member")
    assert _ct.unread_count(cx, t["id"], "member") == 0


def test_block_sets_status():
    cx = _cx()
    t = _ct.get_or_create_thread(cx, coach_email="c@x.com", member_email="m@x.com")
    _ct.block_thread(cx, t["id"], "member")
    b = _ct.get_thread(cx, t["id"])
    assert b["status"] == "blocked" and b["blocked_by"] == "member"


def test_report_sets_flag_and_row():
    cx = _cx()
    t = _ct.get_or_create_thread(cx, coach_email="c@x.com", member_email="m@x.com")
    _ct.report_thread(cx, thread_id=t["id"], reporter_role="member", reason="rude")
    assert _ct.get_thread(cx, t["id"])["reported"] == 1
    row = cx.execute("SELECT * FROM coach_thread_reports").fetchone()
    assert row["reporter_role"] == "member" and row["reason"] == "rude"


def test_list_all_sorts_flagged_first():
    cx = _cx()
    ok = _ct.get_or_create_thread(cx, coach_email="c1@x.com", member_email="m1@x.com")
    rep = _ct.get_or_create_thread(cx, coach_email="c2@x.com", member_email="m2@x.com")
    _ct.report_thread(cx, thread_id=rep["id"], reporter_role="coach", reason="x")
    ids = [t["id"] for t in _ct.list_all_threads(cx)]
    assert ids[0] == rep["id"]                              # reported first


def test_accepted_pair_and_members():
    cx = _cx()
    _cc.create_request(cx, "coach@x.com", "mem@x.com", "Mel P", "help me")
    rid = cx.execute("SELECT id FROM coach_requests").fetchone()["id"]
    _cc.set_request_status(cx, rid, "accepted")
    p = _cc.accepted_pair(cx, "mem@x.com")
    assert p["coach_email"] == "coach@x.com" and p["request_id"] == rid
    ms = _cc.accepted_members(cx, "coach@x.com")
    assert ms == [{"request_id": rid, "member_email": "mem@x.com", "member_name": "Mel P"}]
    _cc.set_request_status(cx, rid, "ended")
    assert _cc.accepted_pair(cx, "mem@x.com") is None       # ended is not accepted
