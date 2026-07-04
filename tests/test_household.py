import sqlite3
from dashboard import household as h


def _cx():
    cx = sqlite3.connect(":memory:")
    h.init_household_tables(cx)
    return cx


def test_add_members_and_list():
    cx = _cx()
    assert h.add_member(cx, "Karin@x.com", "mochi@x.com", "Mochi", "pet") is True
    h.add_member(cx, "karin@x.com", "kai@x.com", "Kai", "child")
    ms = h.members_for(cx, "karin@x.com")
    assert [m["email"] for m in ms] == ["mochi@x.com", "kai@x.com"]
    assert ms[0]["label"] == "Mochi" and ms[0]["relationship"] == "pet"


def test_add_member_rejects_self_and_blank():
    cx = _cx()
    assert h.add_member(cx, "a@x.com", "a@x.com") is False
    assert h.add_member(cx, "", "b@x.com") is False
    assert h.members_for(cx, "a@x.com") == []


def test_add_member_idempotent():
    cx = _cx()
    h.add_member(cx, "p@x.com", "m@x.com", "M", "pet")
    h.add_member(cx, "p@x.com", "m@x.com", "M2", "child")  # dup ignored
    assert len(h.members_for(cx, "p@x.com")) == 1


def test_remove_member():
    cx = _cx()
    h.add_member(cx, "p@x.com", "m@x.com")
    h.remove_member(cx, "P@x.com", "M@x.com")  # case-insensitive
    assert h.members_for(cx, "p@x.com") == []


def test_can_view():
    cx = _cx()
    h.add_member(cx, "p@x.com", "m@x.com")
    assert h.can_view(cx, "p@x.com", "p@x.com") is True     # self
    assert h.can_view(cx, "P@x.com", "M@x.com") is True     # linked (case-insensitive)
    assert h.can_view(cx, "p@x.com", "stranger@x.com") is False
    assert h.can_view(cx, "m@x.com", "p@x.com") is False    # reverse is NOT a view grant
    assert h.can_view(cx, "", "m@x.com") is False


def test_same_household():
    cx = _cx()
    h.add_member(cx, "p@x.com", "m1@x.com")
    h.add_member(cx, "p@x.com", "m2@x.com")
    assert h.same_household(cx, "p@x.com", "m1@x.com") is True   # primary↔member
    assert h.same_household(cx, "m1@x.com", "p@x.com") is True   # order-independent
    assert h.same_household(cx, "m1@x.com", "m2@x.com") is True  # siblings share primary
    assert h.same_household(cx, "m1@x.com", "stranger@x.com") is False
