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


def test_free_monthly_cap_is_per_member_and_permanent(tmp_db):
    cx = _cx(tmp_db)
    # first free unlock in July succeeds
    ok, reason = fa.grant_free_monthly(cx, "m@x.com", "s1", "2026-07-02", "2026-07-02T10:00:00Z")
    assert ok and reason == ""
    assert fa.has_unlock(cx, "m@x.com", "s1") is True
    # second free unlock same month -> capped
    ok, reason = fa.grant_free_monthly(cx, "m@x.com", "s2", "2026-07-05", "2026-07-05T10:00:00Z")
    assert ok is False and reason == "cap"
    assert fa.has_unlock(cx, "m@x.com", "s2") is False
    # next month -> allowed again; prior unlock still present (permanent)
    ok, reason = fa.grant_free_monthly(cx, "m@x.com", "s2", "2026-07-05", "2026-08-01T10:00:00Z")
    assert ok and reason == ""
    assert fa.has_unlock(cx, "m@x.com", "s1") is True
    # a different member is unaffected by m@x.com's usage
    ok, reason = fa.grant_free_monthly(cx, "other@x.com", "s9", "2026-07-02", "2026-07-02T10:00:00Z")
    assert ok and reason == ""


def test_grant_free_monthly_already_unlocked_is_noop(tmp_db):
    cx = _cx(tmp_db)
    fa.record_unlock(cx, "m@x.com", "s1", "2026-07-02", "paid", "2026-07-02T09:00:00Z")
    ok, reason = fa.grant_free_monthly(cx, "m@x.com", "s1", "2026-07-02", "2026-07-02T10:00:00Z")
    assert ok is True and reason == "already"
    # did not consume the monthly allowance
    ok, reason = fa.grant_free_monthly(cx, "m@x.com", "s2", "2026-07-03", "2026-07-03T10:00:00Z")
    assert ok and reason == ""


def test_family_is_paid_follows_primary(tmp_db):
    cx = _cx(tmp_db)
    fa.upsert_member(cx, "karin@x.com", "karin@x.com", "Karin", "human", 0)
    fa.upsert_member(cx, "karin@x.com", "sasha@fake.com", "Sasha", "pet", 1)
    assert fa.family_is_paid(cx, "sasha@fake.com") is False
    fa.set_family_membership(cx, "karin@x.com", True, "2026-07-02T10:00:00Z")
    assert fa.family_is_paid(cx, "sasha@fake.com") is True   # member inherits primary's plan
    assert fa.family_is_paid(cx, "karin@x.com") is True
    assert fa.family_is_paid(cx, "stranger@x.com") is False  # not in any family
