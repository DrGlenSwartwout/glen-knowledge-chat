import json, sqlite3
from unittest import mock
import app as appmod
from dashboard import coach_connect as _cc, coach_directory as _cd, coach_threads as _ct


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
        rid = cx.execute("SELECT id FROM coach_requests").fetchone()["id"]
        _cc.set_request_status(cx, rid, "accepted")
        t = _ev.ensure_portal_token(cx, member, "Mel"); cx.commit()
    return t


def test_get_materializes_shows_coach_name_not_email():
    c = _client(); tok = _matched()
    r = c.get(f"/api/coach-thread/member?token={tok}")
    d = r.get_json()
    assert d["coach_name"] == "Coach Kai" and d["status"] == "active"
    assert "c@x.com" not in json.dumps(d)          # never the coach email


def test_unmatched_member_404():
    c = _client()
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        from dashboard import evox as _ev, client_portal as _cp
        _ev.init_evox_tables(cx); _cp.init_client_portal_table(cx); _ct.init_thread_tables(cx)
        _cc.init_connect_tables(cx)
        tok = _ev.ensure_portal_token(cx, "solo@x.com", "Solo"); cx.commit()
    assert c.get(f"/api/coach-thread/member?token={tok}").status_code == 404


def test_post_message_rejects_empty_and_nudges():
    c = _client(); tok = _matched()
    assert c.post(f"/api/coach-thread/member/message?token={tok}",
                  json={"body": "  "}).status_code == 400
    with mock.patch.object(appmod, "send_evox_email") as sender:
        r = c.post(f"/api/coach-thread/member/message?token={tok}", json={"body": "hi coach"})
    assert r.get_json()["ok"] is True and sender.called


def test_block_ends_pairing_and_hides_history():
    c = _client(); tok = _matched()
    c.post(f"/api/coach-thread/member/message?token={tok}", json={"body": "hello"})
    with mock.patch.object(appmod, "send_evox_email"):
        assert c.post(f"/api/coach-thread/member/block?token={tok}").get_json()["ok"] is True
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row; _cc.init_connect_tables(cx)
        assert _cc.accepted_count(cx, "c@x.com") == 0          # capacity freed
        assert _cc.member_has_accepted(cx, "m@x.com") is False  # can re-apply
    d = c.get(f"/api/coach-thread/member?token={tok}").get_json()
    assert d["status"] == "blocked" and d["messages"] == [] and d["can_post"] is False
    assert c.post(f"/api/coach-thread/member/message?token={tok}",
                  json={"body": "more"}).status_code == 409     # blocked rejects posts


def test_report_flags_and_emails_owner():
    c = _client(); tok = _matched()
    with mock.patch.object(appmod, "send_evox_email") as sender:
        r = c.post(f"/api/coach-thread/member/report?token={tok}", json={"reason": "rude"})
    assert r.get_json()["ok"] is True and sender.called
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row; _ct.init_thread_tables(cx)
        assert _ct.thread_for_pair(cx, "c@x.com", "m@x.com")["reported"] == 1
