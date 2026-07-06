# tests/test_peer_blend_api.py
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


def _run(me_tok, vecs):
    with mock.patch.object(appmod, "_is_paid_member", return_value=True), \
         mock.patch.object(appmod, "_member_interest_vec",
                           side_effect=lambda cx, email, liked: vecs.get(email, [])):
        return _client().get(f"/api/peer/proposal?token={me_tok}").get_json()


def test_strong_semantic_beats_lone_shared_topic():
    # Topics are namespaced per test (blend1_*) rather than the brief's bare "liver" /
    # "marathon" -- LOG_DB persists for the whole pytest run (as noted in
    # test_peer_semantic_api.py), so a topic name reused across tests would let an
    # earlier test's still-opted-in member leak into a later test's candidate pool.
    # me shares 1 topic with "lone" (low vector match), 0 topics with "deep" (high match)
    me = _member("blend1_me@x.com", "blend1_liver")
    _member("blend1_lone@x.com", "blend1_liver")          # shares 1 topic
    _member("blend1_deep@x.com", "blend1_gallbladder")    # shares 0 topics
    _optin("blend1_me@x.com", "blend1_lone@x.com", "blend1_deep@x.com")
    vecs = {"blend1_me@x.com": [1.0, 0.0],
            "blend1_lone@x.com": [0.2, 0.98],       # cos ~0.20 -> lone score 1 + 1.75*0.20 = 1.35
            "blend1_deep@x.com": [0.99, 0.14]}      # cos ~0.99 -> deep score 0 + 1.75*0.99 = 1.73
    d = _run(me, vecs)
    assert d["candidate"]["member_ref"] == _pc.member_ref("blend1_deep@x.com")
    assert d["candidate"]["shared_topics"] == [] and d["candidate"]["semantic"] is True
    assert "blend1_deep@x.com" not in json.dumps(d)     # anonymous


def test_two_shared_topics_beats_any_semantic():
    me = _member("blend2_me@x.com", "blend2_liver", "blend2_sleep")
    _member("blend2_two@x.com", "blend2_liver", "blend2_sleep")   # shares 2 -> score >= 2.0
    _member("blend2_deep@x.com", "blend2_yoga")            # shares 0, near-perfect vector
    _optin("blend2_me@x.com", "blend2_two@x.com", "blend2_deep@x.com")
    vecs = {"blend2_me@x.com": [1.0, 0.0],
            "blend2_two@x.com": [0.0, 1.0],          # cos 0 -> score 2 + 0 = 2.0
            "blend2_deep@x.com": [1.0, 0.02]}        # cos ~1.0 -> score 1.75
    d = _run(me, vecs)
    assert d["candidate"]["member_ref"] == _pc.member_ref("blend2_two@x.com")
    assert d["candidate"]["shared_topics"] == ["blend2_liver", "blend2_sleep"] and d["candidate"]["semantic"] is False


def test_cosine_tiebreak_among_single_topic_matches():
    me = _member("blend3_me@x.com", "blend3_liver")
    _member("blend3_a@x.com", "blend3_liver")              # both share 1 topic
    _member("blend3_b@x.com", "blend3_liver")
    _optin("blend3_me@x.com", "blend3_a@x.com", "blend3_b@x.com")
    vecs = {"blend3_me@x.com": [1.0, 0.0],
            "blend3_a@x.com": [0.3, 0.95],           # cos ~0.30
            "blend3_b@x.com": [0.95, 0.31]}          # cos ~0.95 -> higher -> wins
    d = _run(me, vecs)
    assert d["candidate"]["member_ref"] == _pc.member_ref("blend3_b@x.com")
    assert d["candidate"]["shared_topics"] == ["blend3_liver"]


def test_zero_shared_below_floor_not_offered():
    me = _member("blend4_me@x.com", "blend4_liver")
    _member("blend4_far@x.com", "blend4_marathon")          # 0 shared, orthogonal vector
    _optin("blend4_me@x.com", "blend4_far@x.com")
    vecs = {"blend4_me@x.com": [1.0, 0.0], "blend4_far@x.com": [0.0, 1.0]}   # cos 0 < 0.80
    d = _run(me, vecs)
    assert d["candidate"] is None
