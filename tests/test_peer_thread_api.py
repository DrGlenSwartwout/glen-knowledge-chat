import json, sqlite3
from unittest import mock
import app as appmod
from dashboard import peer_connect as _pc, coach_threads as _ct


def _client():
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()


def _matched_pair(a="a@x.com", b="b@x.com"):
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        from dashboard import evox as _ev, client_portal as _cp
        _ev.init_evox_tables(cx); _cp.init_client_portal_table(cx)
        _pc.init_peer_tables(cx); _ct.init_thread_tables(cx)
        _cp.ensure_token(cx, a, "Ana Smith"); _cp.ensure_token(cx, b, "Ben Jones")
        lo, hi = sorted([a, b])
        t = _ct.get_or_create_thread(cx, coach_email=lo, member_email=hi, source="peer")
        _pc.create_match(cx, a, b, t["id"])
        ta = _ev.ensure_portal_token(cx, a, "Ana"); tb = _ev.ensure_portal_token(cx, b, "Ben")
        cx.commit()
    return t["id"], ta, tb


def test_participant_sees_other_first_name_not_email():
    c = _client(); tid, ta, tb = _matched_pair()
    d = c.get(f"/api/peer-thread/{tid}?token={ta}").get_json()
    assert d["other_first_name"] == "Ben" and "b@x.com" not in json.dumps(d)


def test_non_participant_403():
    c = _client(); tid, ta, tb = _matched_pair()
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        from dashboard import evox as _ev
        _ev.init_evox_tables(cx); tc = _ev.ensure_portal_token(cx, "c@x.com", "Cy"); cx.commit()
    assert c.get(f"/api/peer-thread/{tid}?token={tc}").status_code == 403


def test_message_then_block_ends_match():
    c = _client(); tid, ta, tb = _matched_pair()
    with mock.patch.object(appmod, "send_evox_email"):
        assert c.post(f"/api/peer-thread/{tid}/message?token={ta}",
                      json={"body": "hello peer"}).get_json()["ok"] is True
        assert c.post(f"/api/peer-thread/{tid}/block?token={ta}").get_json()["ok"] is True
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row; _pc.init_peer_tables(cx); _ct.init_thread_tables(cx)
        assert _pc.match_for_pair(cx, "a@x.com", "b@x.com")["status"] == "ended"
        assert _ct.get_thread(cx, tid)["status"] == "blocked"
    assert c.post(f"/api/peer-thread/{tid}/message?token={tb}",
                  json={"body": "still there?"}).status_code == 409


def test_owner_unmatch_peer_ends_match_not_coaching():
    c = _client(); tid, ta, tb = _matched_pair()
    with mock.patch.object(appmod, "_portal_console_ok", return_value=True), \
         mock.patch.object(appmod, "send_evox_email"):
        assert c.post(f"/api/console/coach-threads/{tid}/unmatch").get_json()["ok"] is True
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row; _pc.init_peer_tables(cx); _ct.init_thread_tables(cx)
        assert _pc.match_for_pair(cx, "a@x.com", "b@x.com")["status"] == "ended"
        assert _ct.get_thread(cx, tid)["status"] == "blocked"
