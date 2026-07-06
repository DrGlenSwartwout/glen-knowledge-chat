# tests/test_peer_skip_cooloff_api.py
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


def _optin(*emails):
    for e in emails:
        with sqlite3.connect(appmod.LOG_DB) as cx:
            cx.row_factory = sqlite3.Row; _pc.init_peer_tables(cx)
            _pc.set_optin(cx, e, True); cx.commit()


def _skip(frm, to, created_at):
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row; _pc.init_peer_tables(cx)
        _pc.record_interest(cx, frm, to, "skip")
        cx.execute("UPDATE peer_interest SET created_at=? WHERE from_email=? AND to_email=?",
                   (created_at, frm, to)); cx.commit()


def _proposal(tok, vecs):
    with mock.patch.object(appmod, "_is_paid_member", return_value=True), \
         mock.patch.object(appmod, "_member_interest_vec", side_effect=lambda cx, e, l: vecs.get(e, [])):
        return _client().get(f"/api/peer/proposal?token={tok}").get_json()


# Note: topics are namespaced per test group (sc1_liver, sc2_liver, ...) rather than
# the bare "liver" the brief's draft used -- LOG_DB persists for the whole pytest run
# (as in test_peer_blend_api.py / test_peer_semantic_api.py), so a shared topic name
# across tests would let an earlier test's still-opted-in member leak into a later
# test's candidate pool via the blend's shared-topic scoring.


def test_fresh_skip_not_proposed_even_when_empty():
    me = _member("sc1_me@x.com", "sc1_liver"); _member("sc1_sk@x.com", "sc1_liver")
    _optin("sc1_me@x.com", "sc1_sk@x.com")
    _skip("sc1_me@x.com", "sc1_sk@x.com", "2099-01-01T00:00:00+00:00")   # fresh (future)
    d = _proposal(me, {"sc1_me@x.com": [1.0, 0.0], "sc1_sk@x.com": [1.0, 0.0]})
    assert d["candidate"] is None                       # fresh skip stays excluded, card empty


def test_stale_skip_not_proposed_while_fresh_candidate_exists():
    me = _member("sc2_me@x.com", "sc2_liver")
    _member("sc2_stale@x.com", "sc2_liver"); _member("sc2_fresh@x.com", "sc2_liver")
    _optin("sc2_me@x.com", "sc2_stale@x.com", "sc2_fresh@x.com")
    _skip("sc2_me@x.com", "sc2_stale@x.com", "2020-01-01T00:00:00+00:00")   # stale
    d = _proposal(me, {"sc2_me@x.com": [1.0, 0.0], "sc2_stale@x.com": [1.0, 0.0],
                       "sc2_fresh@x.com": [1.0, 0.0]})
    assert d["candidate"]["member_ref"] == _pc.member_ref("sc2_fresh@x.com")   # fresh wins, not stale


def test_stale_skip_resurfaces_when_pool_dry():
    me = _member("sc3_me@x.com", "sc3_liver"); _member("sc3_stale@x.com", "sc3_liver")
    _optin("sc3_me@x.com", "sc3_stale@x.com")
    _skip("sc3_me@x.com", "sc3_stale@x.com", "2020-01-01T00:00:00+00:00")   # stale, only candidate
    d = _proposal(me, {"sc3_me@x.com": [1.0, 0.0], "sc3_stale@x.com": [1.0, 0.0]})
    assert d["candidate"]["member_ref"] == _pc.member_ref("sc3_stale@x.com")   # resurfaced
    assert "sc3_stale@x.com" not in json.dumps(d)       # still anonymous


def test_non_mutual_connect_never_resurfaces():
    me = _member("sc4_me@x.com", "sc4_liver"); _member("sc4_cn@x.com", "sc4_liver")
    _optin("sc4_me@x.com", "sc4_cn@x.com")
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row; _pc.init_peer_tables(cx)
        _pc.record_interest(cx, "sc4_me@x.com", "sc4_cn@x.com", "connect")
        cx.execute("UPDATE peer_interest SET created_at='2020-01-01T00:00:00+00:00'"); cx.commit()
    d = _proposal(me, {"sc4_me@x.com": [1.0, 0.0], "sc4_cn@x.com": [1.0, 0.0]})
    assert d["candidate"] is None                       # standing connect never re-proposed
