import sqlite3
from dashboard import cert_bonus as cb

def _cx():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    cb.init_tables(cx); return cx

def test_commitment_set_get_list():
    cx = _cx()
    cb.set_commitment(cx, "doc@x.com", kind="pif", started_at="2026-01-15")
    r = cb.get_commitment(cx, "doc@x.com")
    assert r["kind"] == "pif" and r["started_at"] == "2026-01-15" and r["active"]
    assert [c["email"] for c in cb.list_active(cx)] == ["doc@x.com"]
    cb.clear_commitment(cx, "doc@x.com")
    assert cb.get_commitment(cx, "doc@x.com")["active"] == 0
    assert cb.list_active(cx) == []

def test_due_bonuses_monthly_and_level():
    grants = cb.due_bonuses(started_at="2026-01-01", modules_completed=2,
                            granted=set(), today="2026-04-01")
    assert ("monthly", 1) in grants and ("monthly", 3) in grants
    assert ("monthly", 4) not in grants
    assert ("level", 1) in grants and ("level", 2) in grants
    assert ("level", 3) not in grants

def test_due_bonuses_excludes_granted_and_caps_12():
    grants = cb.due_bonuses(started_at="2024-01-01", modules_completed=12,
                            granted={("monthly", m) for m in range(1, 13)} | {("level", 1)},
                            today="2026-06-15")
    assert not any(k == "monthly" for k, _ in grants)
    assert ("level", 1) not in grants
    assert ("level", 12) in grants
    assert all(idx <= 12 for k, idx in grants if k == "monthly")

def test_due_bonuses_zero_when_nothing_elapsed_or_done():
    grants = cb.due_bonuses(started_at="2026-06-10", modules_completed=0,
                            granted=set(), today="2026-06-12")
    assert grants == []   # <1 month elapsed, no modules

def test_record_grant_idempotent_and_granted_pairs():
    cx = _cx()
    cb.record_grant(cx, "doc@x.com", kind="monthly", idx=1, todo_id=10)
    cb.record_grant(cx, "doc@x.com", kind="monthly", idx=1, todo_id=11)  # dup ignored
    assert cb.granted_pairs(cx, "doc@x.com") == {("monthly", 1)}
    n = cx.execute("SELECT COUNT(*) FROM cert_bonus_grants WHERE email=?", ("doc@x.com",)).fetchone()[0]
    assert n == 1
