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


def _proposal(tok, vecs):
    with mock.patch.object(appmod, "_is_paid_member", return_value=True), \
         mock.patch.object(appmod, "_member_interest_vec", side_effect=lambda cx, e, l: vecs.get(e, [])):
        return _client().get(f"/api/peer/proposal?token={tok}").get_json()


def test_semantic_winner_anchors_on_asker_own_topic():
    # me likes rt1_zeta + rt1_alpha (disjoint from the winner's topic) -> semantic match
    me = _member("rt1_me@x.com", "rt1_zeta", "rt1_alpha")
    _member("rt1_win@x.com", "rt1_other")
    _optin("rt1_me@x.com", "rt1_win@x.com")
    d = _proposal(me, {"rt1_me@x.com": [1.0, 0.0], "rt1_win@x.com": [0.99, 0.14]})   # cos ~0.99
    cand = d["candidate"]
    assert cand["member_ref"] == _pc.member_ref("rt1_win@x.com") and cand["shared_topics"] == []
    assert cand["reason_topic"] == "rt1_alpha"                    # asker's FIRST sorted liked topic
    assert "rt1_other" not in json.dumps(d)                       # never the winner's topic
    assert "rt1_win@x.com" not in json.dumps(d)                   # anonymous


def test_exact_winner_has_no_reason_topic():
    me = _member("rt2_me@x.com", "rt2_liver"); _member("rt2_win@x.com", "rt2_liver")
    _optin("rt2_me@x.com", "rt2_win@x.com")
    d = _proposal(me, {"rt2_me@x.com": [1.0, 0.0], "rt2_win@x.com": [0.0, 1.0]})     # cos 0, shared topic
    cand = d["candidate"]
    assert cand["shared_topics"] == ["rt2_liver"] and "reason_topic" not in cand
