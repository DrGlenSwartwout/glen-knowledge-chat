"""Characterization tests for _settle_referral plus direct tests for the
extracted _credit_referrer_by_slug helper. These must pass identically before
and after the refactor that extracts the crediting core out of _settle_referral
into _credit_referrer_by_slug -- no behavior change allowed.
"""
import json
import sqlite3

import app as appmod
from dashboard import points, rewards


def _db(monkeypatch, tmp_path):
    db = str(tmp_path / "t.db")
    monkeypatch.setattr(appmod, "LOG_DB", db)
    cx = sqlite3.connect(db)
    cx.row_factory = sqlite3.Row
    cx.execute("CREATE TABLE people (email TEXT UNIQUE, tags TEXT DEFAULT '[]')")
    cx.execute("CREATE TABLE affiliate_signups (slug TEXT UNIQUE, email TEXT, status TEXT)")
    cx.execute("""CREATE TABLE referral_events (received_at TEXT, email TEXT, utm_source TEXT)""")
    cx.execute("""CREATE TABLE orders (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  email TEXT, created_at TEXT, source TEXT, external_ref TEXT)""")
    cx.execute("""CREATE TABLE todos (id INTEGER PRIMARY KEY, created_at TEXT, owner TEXT,
                  category TEXT, title TEXT, body TEXT, priority TEXT, status TEXT DEFAULT 'open',
                  source TEXT, dedup_key TEXT UNIQUE)""")
    points.init_points_table(cx)
    rewards.init_affiliate_earnings_table(cx)
    monkeypatch.setenv("REWARDS_TIERS_ENABLED", "true")
    # Isolate from live Supabase: the cert-tiered referral pct calls
    # modules_completed_for_email, which otherwise hits the real practitioners
    # table (non-deterministic). Default to None (base pct).
    monkeypatch.setattr(appmod._pp, "modules_completed_for_email", lambda e: None)
    return cx


def _refer(cx, buyer, slug, ref_email, tags, status="approved"):
    cx.execute("INSERT INTO affiliate_signups VALUES (?,?,?)", (slug, ref_email, status))
    cx.execute("INSERT INTO people (email, tags) VALUES (?,?)", (ref_email, json.dumps(tags)))
    cx.execute("INSERT INTO referral_events VALUES ('2026-01-01', ?, ?)", (buyer, slug))
    cx.commit()


# ---------------------------------------------------------------------------
# CHARACTERIZATION: lock existing _settle_referral behavior (email-path).
# ---------------------------------------------------------------------------

def test_char_full_price_sale_credits_expected_points(monkeypatch, tmp_path):
    cx = _db(monkeypatch, tmp_path)
    _refer(cx, "buyer@x.com", "doc", "doc@x.com", ["type:practitioner"])
    order = {"email": "buyer@x.com", "total_cents": 6000, "shipping_cents": 0, "get_cents": 0,
             "discount_cents": 0, "points_redeemed_cents": 0}
    settings = rewards.load_settings(appmod._rewards_settings())
    pct = appmod._referral_pct_for_referrer("doc@x.com", settings)
    expected = round(6000 * pct)
    appmod._settle_referral(order, order_ref="CHAR-INV1")
    assert points.balance(cx, "doc@x.com") == expected
    # second call with same order_ref does not double-credit
    appmod._settle_referral(order, order_ref="CHAR-INV1")
    assert points.balance(cx, "doc@x.com") == expected


def test_char_self_referral_credits_nothing(monkeypatch, tmp_path):
    cx = _db(monkeypatch, tmp_path)
    _refer(cx, "self@x.com", "self", "self@x.com", ["type:client"])
    order = {"email": "self@x.com", "total_cents": 6000, "shipping_cents": 0, "get_cents": 0,
             "discount_cents": 0, "points_redeemed_cents": 0}
    appmod._settle_referral(order, order_ref="CHAR-INV2")
    assert points.balance(cx, "self@x.com") == 0


def test_char_rewards_disabled_credits_nothing(monkeypatch, tmp_path):
    cx = _db(monkeypatch, tmp_path)
    _refer(cx, "buyer@x.com", "doc", "doc@x.com", ["type:practitioner"])
    monkeypatch.delenv("REWARDS_TIERS_ENABLED", raising=False)
    order = {"email": "buyer@x.com", "total_cents": 6000, "shipping_cents": 0, "get_cents": 0,
             "discount_cents": 0, "points_redeemed_cents": 0}
    appmod._settle_referral(order, order_ref="CHAR-INV3")
    assert points.balance(cx, "doc@x.com") == 0


# ---------------------------------------------------------------------------
# HELPER: _credit_referrer_by_slug direct tests.
# ---------------------------------------------------------------------------

def test_helper_produces_same_ledger_entry_as_email_path(monkeypatch, tmp_path):
    cx = _db(monkeypatch, tmp_path)
    _refer(cx, "buyer@x.com", "doc", "doc@x.com", ["type:practitioner"])
    settings = rewards.load_settings(appmod._rewards_settings())
    pct = appmod._referral_pct_for_referrer("doc@x.com", settings)
    expected = round(6000 * pct)
    appmod._credit_referrer_by_slug(cx, "doc", "buyer@x.com", 6000, "HELPER-INV1")
    assert points.balance(cx, "doc@x.com") == expected


def test_helper_idempotent_per_order_ref(monkeypatch, tmp_path):
    cx = _db(monkeypatch, tmp_path)
    _refer(cx, "buyer@x.com", "doc", "doc@x.com", ["type:practitioner"])
    appmod._credit_referrer_by_slug(cx, "doc", "buyer@x.com", 6000, "HELPER-INV2")
    bal_first = points.balance(cx, "doc@x.com")
    appmod._credit_referrer_by_slug(cx, "doc", "buyer@x.com", 6000, "HELPER-INV2")
    assert points.balance(cx, "doc@x.com") == bal_first


def test_helper_unknown_slug_credits_nothing(monkeypatch, tmp_path):
    cx = _db(monkeypatch, tmp_path)
    appmod._credit_referrer_by_slug(cx, "ghost", "buyer@x.com", 6000, "HELPER-INV3")
    assert points.balance(cx, "ghost@x.com") == 0


def test_helper_unapproved_slug_credits_nothing(monkeypatch, tmp_path):
    cx = _db(monkeypatch, tmp_path)
    _refer(cx, "buyer@x.com", "pending-doc", "pending@x.com", ["type:practitioner"], status="pending")
    appmod._credit_referrer_by_slug(cx, "pending-doc", "buyer@x.com", 6000, "HELPER-INV4")
    assert points.balance(cx, "pending@x.com") == 0


def test_helper_self_referral_credits_nothing(monkeypatch, tmp_path):
    cx = _db(monkeypatch, tmp_path)
    _refer(cx, "self@x.com", "self", "self@x.com", ["type:client"])
    appmod._credit_referrer_by_slug(cx, "self", "self@x.com", 6000, "HELPER-INV5")
    assert points.balance(cx, "self@x.com") == 0
