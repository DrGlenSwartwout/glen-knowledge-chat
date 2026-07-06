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
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row; _pc.init_peer_tables(cx)
        for email in emails:
            _pc.set_optin(cx, email, True)
        cx.commit()


# Note: emails are namespaced per test (zzz_sem1_*, zzz_sem2_*, zzz_sem3_*) rather than
# reused across tests as the brief's draft did — LOG_DB persists for the whole pytest
# run (as in test_peer_match_api.py), so a shared "me@x.com" across tests would let an
# earlier test's opted-in member leak into a later test's candidate pool.

# controlled vectors: me close to "close", far from "far"
def _vecs_for(prefix):
    return {f"{prefix}_me@x.com": [1.0, 0.0], f"{prefix}_close@x.com": [0.99, 0.14],
            f"{prefix}_far@x.com": [0.0, 1.0]}


def test_semantic_fills_when_no_exact_overlap():
    p = "zzz_sem1"
    c = _client()
    me = _member(f"{p}_me@x.com", f"{p}_liver")          # no one else likes this topic
    _member(f"{p}_close@x.com", f"{p}_gallbladder")
    _member(f"{p}_far@x.com", f"{p}_marathon")
    with mock.patch.object(appmod, "_is_paid_member", return_value=True):
        c.post(f"/api/peer/optin?token={me}", json={"active": True})
    _optin(f"{p}_close@x.com", f"{p}_far@x.com")     # opt the candidates in
    vecs = _vecs_for(p)
    def _fake_vec(cx, email, liked):
        return vecs.get(email, [])
    with mock.patch.object(appmod, "_is_paid_member", return_value=True), \
         mock.patch.object(appmod, "_member_interest_vec", side_effect=_fake_vec):
        d = c.get(f"/api/peer/proposal?token={me}").get_json()
    assert d["candidate"]["member_ref"] == _pc.member_ref(f"{p}_close@x.com")   # closest, above threshold
    assert d["candidate"]["shared_topics"] == [] and d["candidate"]["semantic"] is True
    assert f"{p}_close@x.com" not in json.dumps(d)                              # anonymous


def test_exact_topic_candidate_wins_in_blend():
    c = _client()
    me = _member("zzz_ex_me@x.com", "liver")
    _member("zzz_ex_share@x.com", "liver")
    for email in ("zzz_ex_me@x.com", "zzz_ex_share@x.com"):
        with sqlite3.connect(appmod.LOG_DB) as cx:
            cx.row_factory = sqlite3.Row; _pc.init_peer_tables(cx)
            _pc.set_optin(cx, email, True); cx.commit()
    with mock.patch.object(appmod, "_is_paid_member", return_value=True), \
         mock.patch.object(appmod, "_member_interest_vec",
                           side_effect=lambda cx, email, liked: [1.0, 0.0]):
        d = c.get(f"/api/peer/proposal?token={me}").get_json()
    assert d["candidate"]["shared_topics"] == ["liver"] and d["candidate"]["semantic"] is False


def test_below_threshold_returns_no_candidate():
    p = "zzz_sem3"
    c = _client()
    me = _member(f"{p}_me@x.com", f"{p}_liver"); _member(f"{p}_far@x.com", f"{p}_marathon")
    _optin(f"{p}_me@x.com", f"{p}_far@x.com")
    vecs = _vecs_for(p)
    def _fake_vec(cx, email, liked):
        return vecs.get(email, [])
    with mock.patch.object(appmod, "_is_paid_member", return_value=True), \
         mock.patch.object(appmod, "_member_interest_vec", side_effect=_fake_vec):
        d = c.get(f"/api/peer/proposal?token={me}").get_json()
    assert d["candidate"] is None                       # cosine(me,far)=0 < 0.80
