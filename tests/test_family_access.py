# tests/test_family_access.py
import sqlite3
from dashboard import family_access as fa


def _cx(tmp_db):
    cx = sqlite3.connect(tmp_db)
    fa.init_tables(cx)
    return cx


def test_upsert_list_and_resolve_members(tmp_db):
    cx = _cx(tmp_db)
    fa.upsert_member(cx, "Karin@X.com ", "Karin@X.com", "Karin", "human", 0)
    fa.upsert_member(cx, "karin@x.com", "SASHA@fake.com", "Sasha (cat)", "pet", 1)
    members = fa.list_members(cx, "karin@x.com")
    assert [m["member_email"] for m in members] == ["karin@x.com", "sasha@fake.com"]
    assert members[1]["member_label"] == "Sasha (cat)"
    assert members[1]["member_type"] == "pet"
    assert fa.primary_for(cx, "sasha@fake.com") == "karin@x.com"
    assert fa.is_primary(cx, "karin@x.com") is True
    assert fa.is_primary(cx, "sasha@fake.com") is False


def test_upsert_is_idempotent_and_remove(tmp_db):
    cx = _cx(tmp_db)
    fa.upsert_member(cx, "p@x.com", "m@x.com", "M", "human", 0)
    fa.upsert_member(cx, "p@x.com", "m@x.com", "M2", "human", 5)  # update, not duplicate
    members = fa.list_members(cx, "p@x.com")
    assert len(members) == 1 and members[0]["member_label"] == "M2"
    fa.remove_member(cx, "p@x.com", "m@x.com")
    assert fa.list_members(cx, "p@x.com") == []
