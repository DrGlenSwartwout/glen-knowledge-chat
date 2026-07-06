# tests/test_peer_eligible_pool.py
import sqlite3
from dashboard import peer_connect as _pc
from dashboard import community_signals as _cs


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    _pc.init_peer_tables(cx); _cs.init_signal_tables(cx)
    return cx


def _like(cx, email, *topics):
    for t in topics:
        _cs.set_signal(cx, email, "topic", t, "like")


def test_eligible_pool_applies_exclusions_without_topic_filter():
    cx = _cx()
    _pc.set_optin(cx, "me@x.com", True); _like(cx, "me@x.com", "liver")
    _pc.set_optin(cx, "noshare@x.com", True); _like(cx, "noshare@x.com", "sleep")   # 0 shared, still eligible
    _pc.set_optin(cx, "acted@x.com", True); _like(cx, "acted@x.com", "detox")
    _pc.record_interest(cx, "me@x.com", "acted@x.com", "skip")                        # I acted -> excluded
    _pc.set_optin(cx, "off@x.com", False); _like(cx, "off@x.com", "keto")            # not opted -> excluded
    pool = set(_pc.eligible_candidates(cx, "me@x.com"))
    assert pool == {"noshare@x.com"}                       # exact-less but eligible; excludes self/acted/non-opted


def test_eligible_pool_respects_is_paid_and_next_candidate_unchanged():
    cx = _cx()
    _pc.set_optin(cx, "me@x.com", True); _like(cx, "me@x.com", "liver", "sleep")
    _pc.set_optin(cx, "free@x.com", True); _like(cx, "free@x.com", "yoga")            # eligible-but-free
    _pc.set_optin(cx, "share@x.com", True); _like(cx, "share@x.com", "liver")         # shares a topic
    paid = lambda e: e != "free@x.com"
    assert set(_pc.eligible_candidates(cx, "me@x.com", is_paid=paid)) == {"share@x.com"}
    # next_candidate still returns the exact-shared one, unchanged
    c = _pc.next_candidate(cx, "me@x.com", is_paid=paid)
    assert c["member_ref"] == _pc.member_ref("share@x.com") and c["shared_topics"] == ["liver"]
