import json, sqlite3
from unittest import mock
import app as appmod
from dashboard import peer_connect as _pc, community_signals as _cs, coach_threads as _ct


def _client():
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()


def _member(email, *topics):
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        from dashboard import evox as _ev, client_portal as _cp
        _ev.init_evox_tables(cx); _cp.init_client_portal_table(cx)
        _pc.init_peer_tables(cx); _cs.init_signal_tables(cx); _ct.init_thread_tables(cx)
        _cp.ensure_token(cx, email, email.split("@")[0].title())
        for t in topics:
            _cs.set_signal(cx, email, "topic", t, "like")
        tok = _ev.ensure_portal_token(cx, email, email.split("@")[0]); cx.commit()
    return tok


def test_free_member_not_eligible():
    c = _client(); tok = _member("free@x.com", "liver")
    with mock.patch.object(appmod, "_is_paid_member", return_value=False):
        assert c.get(f"/api/peer/state?token={tok}").get_json()["eligible"] is False
        assert c.post(f"/api/peer/optin?token={tok}", json={"active": True}).status_code == 403


def test_proposal_is_anonymous():
    c = _client(); a = _member("a@x.com", "liver", "sleep"); b = _member("b@x.com", "liver", "sleep")
    with mock.patch.object(appmod, "_is_paid_member", return_value=True):
        c.post(f"/api/peer/optin?token={a}", json={"active": True})
        c.post(f"/api/peer/optin?token={b}", json={"active": True})
        d = c.get(f"/api/peer/proposal?token={a}").get_json()
    assert d["candidate"]["member_ref"] == _pc.member_ref("b@x.com")
    assert "b@x.com" not in json.dumps(d) and "Bob" not in json.dumps(d)   # anonymous


def test_mutual_connect_reveals_and_opens_thread():
    c = _client(); a = _member("a@x.com", "liver"); b = _member("b@x.com", "liver")
    with mock.patch.object(appmod, "_is_paid_member", return_value=True), \
         mock.patch.object(appmod, "send_evox_email"):
        c.post(f"/api/peer/optin?token={a}", json={"active": True})
        c.post(f"/api/peer/optin?token={b}", json={"active": True})
        ref_b = _pc.member_ref("b@x.com"); ref_a = _pc.member_ref("a@x.com")
        r1 = c.post(f"/api/peer/interest?token={a}", json={"member_ref": ref_b, "kind": "connect"})
        assert r1.get_json()["matched"] is False                # not yet mutual
        r2 = c.post(f"/api/peer/interest?token={b}", json={"member_ref": ref_a, "kind": "connect"})
        assert r2.get_json()["matched"] is True                 # mutual
    conns = c.get(f"/api/peer/connections?token={a}").get_json()
    assert conns[0]["first_name"] == "B" and "b@x.com" not in json.dumps(conns)
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row; _pc.init_peer_tables(cx)
        assert _pc.match_for_pair(cx, "a@x.com", "b@x.com")["thread_id"] is not None


def test_skip_removes_from_next_proposal():
    c = _client(); a = _member("a@x.com", "liver"); b = _member("b@x.com", "liver")
    with mock.patch.object(appmod, "_is_paid_member", return_value=True):
        c.post(f"/api/peer/optin?token={a}", json={"active": True})
        c.post(f"/api/peer/optin?token={b}", json={"active": True})
        c.post(f"/api/peer/interest?token={a}", json={"member_ref": _pc.member_ref("b@x.com"),
                                                       "kind": "skip"})
        assert c.get(f"/api/peer/proposal?token={a}").get_json()["candidate"] is None
