"""
Spec 2b-2 — referrer reward columns + lookups.
Tests use sqlite3.connect(":memory:") — no app import needed.
"""
import sqlite3
from dashboard import referrals as rf


def _cx():
    return sqlite3.connect(":memory:")


# ---------------------------------------------------------------------------
# Schema: additive columns exist after init_tables
# ---------------------------------------------------------------------------

def test_init_tables_adds_reward_columns():
    """After init_tables the referral_redemptions table has rewarded_at and reward_cents."""
    cx = _cx()
    rf.init_tables(cx)
    cols = {row[1] for row in cx.execute("PRAGMA table_info(referral_redemptions)").fetchall()}
    assert "rewarded_at" in cols
    assert "reward_cents" in cols


def test_init_tables_idempotent():
    """Calling init_tables twice on a db that already has the columns must not raise."""
    cx = _cx()
    rf.init_tables(cx)
    rf.init_tables(cx)   # second call – idempotent


def test_init_tables_idempotent_on_existing_table():
    """init_tables must be safe even when table already exists without new cols (ALTER path)."""
    cx = _cx()
    # Create the table without the new columns to simulate a pre-existing schema
    cx.execute("CREATE TABLE IF NOT EXISTS referral_codes ("
               "email TEXT PRIMARY KEY, code TEXT UNIQUE, created_at TEXT)")
    cx.execute("CREATE TABLE IF NOT EXISTS referral_redemptions ("
               "referee_email TEXT PRIMARY KEY, code TEXT, owner_email TEXT, "
               "order_ref TEXT, created_at TEXT)")
    cx.commit()
    # Now run init_tables — must add the missing columns without error
    rf.init_tables(cx)
    cols = {row[1] for row in cx.execute("PRAGMA table_info(referral_redemptions)").fetchall()}
    assert "rewarded_at" in cols
    assert "reward_cents" in cols


# ---------------------------------------------------------------------------
# redemption_by_order_ref
# ---------------------------------------------------------------------------

def test_redemption_by_order_ref_returns_row():
    cx = _cx()
    code = rf.get_or_create_code(cx, "owner@x.com")
    rf.record_redemption(cx, code, "owner@x.com", "friend@x.com", "INV-42")
    row = rf.redemption_by_order_ref(cx, "INV-42")
    assert row is not None
    assert row["referee_email"] == "friend@x.com"
    assert row["order_ref"] == "INV-42"


def test_redemption_by_order_ref_returns_none_when_missing():
    cx = _cx()
    rf.init_tables(cx)
    assert rf.redemption_by_order_ref(cx, "NO-SUCH-ORDER") is None


def test_redemption_by_order_ref_none_when_empty_table():
    cx = _cx()
    rf.init_tables(cx)
    assert rf.redemption_by_order_ref(cx, "") is None


# ---------------------------------------------------------------------------
# mark_rewarded
# ---------------------------------------------------------------------------

def test_mark_rewarded_stamps_row():
    cx = _cx()
    code = rf.get_or_create_code(cx, "owner@x.com")
    rf.record_redemption(cx, code, "owner@x.com", "friend@x.com", "INV-1")
    rf.mark_rewarded(cx, "friend@x.com", reward_cents=500)
    row = cx.execute(
        "SELECT rewarded_at, reward_cents FROM referral_redemptions WHERE referee_email=?",
        ("friend@x.com",)
    ).fetchone()
    assert row is not None
    assert row[1] == 500
    assert row[0] is not None   # rewarded_at timestamp set


def test_mark_rewarded_normalizes_email():
    """Email passed to mark_rewarded is case-folded to match the stored lower-case key."""
    cx = _cx()
    code = rf.get_or_create_code(cx, "owner@x.com")
    rf.record_redemption(cx, code, "owner@x.com", "Friend@X.com", "INV-2")
    # The row is stored with normalized key "friend@x.com"
    rf.mark_rewarded(cx, "FRIEND@X.COM", reward_cents=250)
    row = cx.execute(
        "SELECT reward_cents FROM referral_redemptions WHERE referee_email=?",
        ("friend@x.com",)
    ).fetchone()
    assert row is not None and row[0] == 250


def test_mark_rewarded_overwrites_on_retry():
    """Calling mark_rewarded a second time updates the stamp (idempotent update)."""
    cx = _cx()
    code = rf.get_or_create_code(cx, "owner@x.com")
    rf.record_redemption(cx, code, "owner@x.com", "friend@x.com", "INV-3")
    rf.mark_rewarded(cx, "friend@x.com", reward_cents=100)
    rf.mark_rewarded(cx, "friend@x.com", reward_cents=200)
    row = cx.execute(
        "SELECT reward_cents FROM referral_redemptions WHERE referee_email=?",
        ("friend@x.com",)
    ).fetchone()
    assert row[0] == 200


# ---------------------------------------------------------------------------
# Task 2: _referrer_reward_pct + _settle_referrer_reward
# ---------------------------------------------------------------------------

import importlib


def _reload_reward_app(monkeypatch, tmp_path, pct="10", referrals="true"):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("REFERRALS", referrals)
    monkeypatch.setenv("REFERRER_REWARD_PCT", pct)
    import app as appmod
    importlib.reload(appmod)
    return appmod


def _seed_redemption(appmod, order_ref="INV-1", owner="owner@x.com", referee="friend@x.com"):
    import sqlite3
    from dashboard import referrals as rf
    with sqlite3.connect(appmod.LOG_DB) as cx:
        rf.record_redemption(cx, "CODE1", owner, referee, order_ref)


def _order(order_ref="INV-1", referee="friend@x.com", total=7000, shipping=1300, get=0):
    return {"email": referee, "total_cents": total, "shipping_cents": shipping, "get_cents": get}


def test_settle_referrer_reward_credits_pct(monkeypatch, tmp_path):
    appmod = _reload_reward_app(monkeypatch, tmp_path, pct="10")
    _seed_redemption(appmod)
    import sqlite3
    from dashboard import points, referrals as rf
    with sqlite3.connect(appmod.LOG_DB) as cx:
        credited = appmod._settle_referrer_reward(cx, _order(), "INV-1")
    # product spend = 7000 - 1300 - 0 = 5700; 10% = 570
    assert credited == 570
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert points.balance(cx, "owner@x.com") == 570
        assert rf.redemption_by_order_ref(cx, "INV-1")["rewarded_at"]
    # idempotent: a second settle credits nothing more
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert appmod._settle_referrer_reward(cx, _order(), "INV-1") == 0
        assert points.balance(cx, "owner@x.com") == 570


def test_no_reward_when_pct_zero(monkeypatch, tmp_path):
    appmod = _reload_reward_app(monkeypatch, tmp_path, pct="0")
    _seed_redemption(appmod)
    import sqlite3
    from dashboard import points
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert appmod._settle_referrer_reward(cx, _order(), "INV-1") == 0
        assert points.balance(cx, "owner@x.com") == 0


def test_no_reward_without_redemption(monkeypatch, tmp_path):
    appmod = _reload_reward_app(monkeypatch, tmp_path, pct="10")
    import sqlite3
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert appmod._settle_referrer_reward(cx, _order(order_ref="OTHER"), "OTHER") == 0


def test_zero_product_cents_stamps_no_credit(monkeypatch, tmp_path):
    appmod = _reload_reward_app(monkeypatch, tmp_path, pct="10")
    _seed_redemption(appmod)
    import sqlite3
    from dashboard import points, referrals as rf
    with sqlite3.connect(appmod.LOG_DB) as cx:
        # all shipping -> product_cents 0
        credited = appmod._settle_referrer_reward(cx, _order(total=1300, shipping=1300), "INV-1")
        assert credited == 0 and points.balance(cx, "owner@x.com") == 0
        assert rf.redemption_by_order_ref(cx, "INV-1")["rewarded_at"]   # stamped, won't retry
