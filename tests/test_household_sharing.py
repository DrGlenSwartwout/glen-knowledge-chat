# tests/test_household_sharing.py
import sqlite3
from dashboard import household as h


def _cx():
    cx = sqlite3.connect(":memory:"); h.init_household_tables(cx); return cx


def test_classification():
    assert h.default_cc_for("pet") == 1 and h.default_cc_for("child") == 1
    assert h.default_cc_for("caregiving-client") == 1 and h.default_cc_for("dependent") == 1
    assert h.default_cc_for("spouse") == 0 and h.default_cc_for("adult-child") == 0
    assert h.default_cc_for("") == 0 and h.default_cc_for("PET") == 1  # case-insensitive


def test_add_member_sets_cc_from_relationship_and_consent_default():
    cx = _cx()
    h.add_member(cx, "p@x.com", "sasha@x.com", "Sasha", "pet")
    h.add_member(cx, "p@x.com", "rob@x.com", "Rob", "spouse")
    ms = {m["email"]: m for m in h.members_for(cx, "p@x.com")}
    assert ms["sasha@x.com"]["cc_enabled"] == 1 and ms["sasha@x.com"]["share_consent"] == 1
    assert ms["rob@x.com"]["cc_enabled"] == 0 and ms["rob@x.com"]["share_consent"] == 1


def test_can_view_requires_consent():
    cx = _cx()
    h.add_member(cx, "p@x.com", "m@x.com", "M", "spouse")
    assert h.can_view(cx, "p@x.com", "m@x.com") is True       # default consented
    h.set_share_consent(cx, "p@x.com", "m@x.com", 0)
    assert h.can_view(cx, "p@x.com", "m@x.com") is False      # revoked → not viewable
    assert h.can_view(cx, "m@x.com", "m@x.com") is True       # self always


def test_viewable_members_excludes_revoked():
    cx = _cx()
    h.add_member(cx, "p@x.com", "a@x.com", "A", "child")
    h.add_member(cx, "p@x.com", "b@x.com", "B", "spouse")
    h.set_share_consent(cx, "p@x.com", "b@x.com", 0)
    assert [m["email"] for m in h.viewable_members_for(cx, "p@x.com")] == ["a@x.com"]
    # members_for (console) still lists both
    assert len(h.members_for(cx, "p@x.com")) == 2


def test_cc_recipients_two_switch_rule():
    cx = _cx()
    h.add_member(cx, "p@x.com", "m@x.com", "M", "pet")        # cc default 1, consent 1
    assert h.cc_recipients_for(cx, "m@x.com") == ["p@x.com"]
    h.set_cc_enabled(cx, "p@x.com", "m@x.com", 0)
    assert h.cc_recipients_for(cx, "m@x.com") == []            # cc off
    h.set_cc_enabled(cx, "p@x.com", "m@x.com", 1); h.set_share_consent(cx, "p@x.com", "m@x.com", 0)
    assert h.cc_recipients_for(cx, "m@x.com") == []            # consent off
    h.set_share_consent(cx, "p@x.com", "m@x.com", 1)
    assert h.cc_recipients_for(cx, "m@x.com") == ["p@x.com"]   # both on


def test_same_household_ignores_consent():
    cx = _cx()
    h.add_member(cx, "p@x.com", "a@x.com", "A", "child")
    h.add_member(cx, "p@x.com", "b@x.com", "B", "child")
    h.set_share_consent(cx, "p@x.com", "a@x.com", 0)          # revoked
    assert h.same_household(cx, "a@x.com", "b@x.com") is True  # reassignment still works


def test_migration_backfills_cc_for_existing_dependent_rows():
    # simulate a pre-feature table (no share_consent/cc_enabled columns)
    cx = sqlite3.connect(":memory:")
    cx.execute("CREATE TABLE household_members (id INTEGER PRIMARY KEY AUTOINCREMENT, "
               "primary_email TEXT, member_email TEXT, label TEXT, relationship TEXT, created_at TEXT, "
               "UNIQUE(primary_email, member_email))")
    cx.execute("INSERT INTO household_members (primary_email, member_email, label, relationship, created_at) "
               "VALUES ('p@x.com','pet@x.com','Sasha','pet','t'), ('p@x.com','sp@x.com','Sp','spouse','t')")
    cx.commit()
    h.init_household_tables(cx)   # ALTER + backfill
    ms = {m["email"]: m for m in h.members_for(cx, "p@x.com")}
    assert ms["pet@x.com"]["cc_enabled"] == 1 and ms["pet@x.com"]["share_consent"] == 1
    assert ms["sp@x.com"]["cc_enabled"] == 0
