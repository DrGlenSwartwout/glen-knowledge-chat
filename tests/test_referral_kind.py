import sqlite3
from dashboard import referrals as rf


def _cx():
    cx = sqlite3.connect(":memory:")
    rf.init_tables(cx)
    return cx


def test_kind_column_exists_and_defaults_to_referral():
    cx = _cx()
    cols = {r[1] for r in cx.execute("PRAGMA table_info(referral_redemptions)")}
    assert "kind" in cols
    rf.record_redemption(cx, "C1", "owner@x.com", "friend@x.com", "INV-1")
    row = rf.redemption_by_order_ref(cx, "INV-1")
    assert row["kind"] == "referral"   # default preserves the Ambassador flow


def test_record_redemption_writes_explicit_kind():
    cx = _cx()
    rf.record_redemption(cx, "C1", "doc@x.com", "patient@x.com", "INV-2",
                         kind="dispensary_portal")
    row = rf.redemption_by_order_ref(cx, "INV-2")
    assert row["kind"] == "dispensary_portal"
    assert row["owner_email"] == "doc@x.com"


def test_first_touch_preserves_original_owner_and_kind():
    cx = _cx()
    rf.record_redemption(cx, "C1", "ambassador@x.com", "patient@x.com", "INV-3")  # referral
    wrote = rf.record_redemption(cx, "C2", "doc@x.com", "patient@x.com", "INV-4",
                                 kind="dispensary_portal")  # same referee PK
    assert wrote is False   # INSERT OR IGNORE dropped the second
    row = rf.redemption_by_order_ref(cx, "INV-3")
    assert row["owner_email"] == "ambassador@x.com" and row["kind"] == "referral"


def test_init_tables_idempotent_adds_kind_once():
    cx = _cx()
    rf.init_tables(cx)   # second call must not raise
    cols = [r[1] for r in cx.execute("PRAGMA table_info(referral_redemptions)")]
    assert cols.count("kind") == 1
