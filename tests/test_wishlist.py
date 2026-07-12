import sqlite3
from dashboard import wishlist as w

def _cx():
    cx = sqlite3.connect(":memory:")
    w.init_wishlist_table(cx)
    return cx

def test_resolve_owner_email_wins():
    assert w.resolve_owner("A@x.com ", "sess1") == "email:a@x.com"
    assert w.resolve_owner("", "sess1") == "sess:sess1"
    assert w.resolve_owner(None, None) is None

def test_toggle_round_trip():
    cx = _cx()
    assert w.toggle(cx, "sess:s1", "iop-syntropy") is True
    assert w.list_for(cx, "sess:s1") == ["iop-syntropy"]
    assert w.toggle(cx, "sess:s1", "iop-syntropy") is False
    assert w.list_for(cx, "sess:s1") == []

def test_owners_are_independent():
    cx = _cx()
    w.toggle(cx, "sess:s1", "a"); w.toggle(cx, "email:e@x.com", "b")
    assert w.list_for(cx, "sess:s1") == ["a"]
    assert w.list_for(cx, "email:e@x.com") == ["b"]

def test_list_for_is_newest_first():
    cx = _cx()
    for s in ["a", "b", "c"]:
        w.toggle(cx, "sess:s1", s)
    assert w.list_for(cx, "sess:s1") == ["c", "b", "a"]
