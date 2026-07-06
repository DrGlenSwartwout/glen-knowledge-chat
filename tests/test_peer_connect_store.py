# tests/test_peer_connect_store.py
import sqlite3
from dashboard import peer_connect as _pc
from dashboard import community_signals as _cs


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    _pc.init_peer_tables(cx)
    _cs.init_signal_tables(cx)
    return cx


def _like(cx, email, *topics):
    for t in topics:
        _cs.set_signal(cx, email, "topic", t, "like")


def test_member_ref_deterministic_and_case_insensitive():
    r = _pc.member_ref("A@x.com")
    assert r == _pc.member_ref("a@x.com")            # case-insensitive, deterministic
    assert len(r) == 16 and all(ch in "0123456789abcdef" for ch in r)


def test_member_ref_is_salted_when_secret_present(monkeypatch):
    import hashlib
    monkeypatch.setenv("PEER_REF_SALT", "s3cr3t-salt")
    r = _pc.member_ref("a@x.com")
    assert r != hashlib.sha256(b"a@x.com").hexdigest()[:16]   # HMAC path actually fires
    assert len(r) == 16


def test_optin_pool():
    cx = _cx()
    _pc.set_optin(cx, "A@x.com", True)
    assert _pc.is_opted_in(cx, "a@x.com") is True
    _pc.set_optin(cx, "a@x.com", False)
    assert _pc.is_opted_in(cx, "a@x.com") is False
    assert "a@x.com" not in _pc.opted_in_members(cx)


def test_liked_and_blocked_topics():
    cx = _cx()
    _like(cx, "m@x.com", "liver", "sleep")
    _cs.set_signal(cx, "m@x.com", "topic", "keto", "block")
    assert _pc.liked_topics(cx, "m@x.com") == {"liver", "sleep"}
    assert _pc.blocked_topics(cx, "m@x.com") == {"keto"}


def test_next_candidate_anonymous_shared_topics_ranked():
    cx = _cx()
    _pc.set_optin(cx, "me@x.com", True); _like(cx, "me@x.com", "liver", "sleep", "detox")
    _pc.set_optin(cx, "one@x.com", True); _like(cx, "one@x.com", "liver")           # 1 shared
    _pc.set_optin(cx, "two@x.com", True); _like(cx, "two@x.com", "liver", "sleep")  # 2 shared
    _pc.set_optin(cx, "off@x.com", False); _like(cx, "off@x.com", "liver", "sleep") # not opted in
    c = _pc.next_candidate(cx, "me@x.com")
    assert c["member_ref"] == _pc.member_ref("two@x.com")     # highest overlap
    assert c["shared_topics"] == ["liver", "sleep"]
    assert "two@x.com" not in str(c) and "email" not in c     # anonymous


def test_next_candidate_excludes_acted_skipped_matched():
    cx = _cx()
    _pc.set_optin(cx, "me@x.com", True); _like(cx, "me@x.com", "liver")
    _pc.set_optin(cx, "skipme@x.com", True); _like(cx, "skipme@x.com", "liver")
    _pc.record_interest(cx, "me@x.com", "skipme@x.com", "skip")   # I skipped them
    assert _pc.next_candidate(cx, "me@x.com") is None
    _pc.set_optin(cx, "theyskip@x.com", True); _like(cx, "theyskip@x.com", "liver")
    _pc.record_interest(cx, "theyskip@x.com", "me@x.com", "skip") # they skipped me
    assert _pc.next_candidate(cx, "me@x.com") is None


def test_next_candidate_keeps_someone_who_connected_to_me():
    cx = _cx()
    _pc.set_optin(cx, "me@x.com", True); _like(cx, "me@x.com", "liver")
    _pc.set_optin(cx, "keen@x.com", True); _like(cx, "keen@x.com", "liver")
    _pc.record_interest(cx, "keen@x.com", "me@x.com", "connect")  # they want to connect
    c = _pc.next_candidate(cx, "me@x.com")
    assert c["member_ref"] == _pc.member_ref("keen@x.com")       # still shown so I can reciprocate


def test_resolve_ref_and_match():
    cx = _cx()
    _pc.set_optin(cx, "a@x.com", True); _pc.set_optin(cx, "b@x.com", True)
    assert _pc.resolve_ref(cx, "a@x.com", _pc.member_ref("b@x.com")) == "b@x.com"
    assert _pc.resolve_ref(cx, "a@x.com", "deadbeefdeadbeef") is None
    _pc.create_match(cx, "b@x.com", "a@x.com", 42)               # normalized a<b
    m = _pc.match_for_pair(cx, "a@x.com", "b@x.com")
    assert m["thread_id"] == 42 and m["a_email"] == "a@x.com" and m["b_email"] == "b@x.com"
    assert [x["other_email"] for x in _pc.matches_for(cx, "a@x.com")] == ["b@x.com"]
    _pc.end_match(cx, 42)
    assert _pc.match_for_pair(cx, "a@x.com", "b@x.com")["status"] == "ended"
    # an ended match is never re-proposed
    _like(cx, "a@x.com", "liver"); _like(cx, "b@x.com", "liver")
    assert _pc.next_candidate(cx, "a@x.com") is None
