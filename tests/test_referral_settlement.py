"""Tests for Task 2: referral crediting and buyer first-order suppression."""
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
    return cx


def _refer(cx, buyer, slug, ref_email, tags):
    cx.execute("INSERT INTO affiliate_signups VALUES (?,?,?)", (slug, ref_email, "approved"))
    cx.execute("INSERT INTO people (email, tags) VALUES (?,?)", (ref_email, json.dumps(tags)))
    cx.execute("INSERT INTO referral_events VALUES ('2026-01-01', ?, ?)", (buyer, slug))
    cx.commit()


def test_points_referrer_credited(monkeypatch, tmp_path):
    cx = _db(monkeypatch, tmp_path)
    _refer(cx, "buyer@x.com", "doc", "doc@x.com", ["type:practitioner"])
    order = {"email": "buyer@x.com", "total_cents": 6000, "shipping_cents": 0, "get_cents": 0,
             "discount_cents": 0, "points_redeemed_cents": 0}
    appmod._settle_referral(order, order_ref="INV1")
    assert points.balance(cx, "doc@x.com") == 300        # 5% of 6000 product
    # idempotent
    appmod._settle_referral(order, order_ref="INV1")
    assert points.balance(cx, "doc@x.com") == 300


def test_cash_referrer_accrues_not_points(monkeypatch, tmp_path):
    cx = _db(monkeypatch, tmp_path)
    _refer(cx, "buyer@x.com", "jane", "jane@x.com", ["tier:pro-influencer"])
    order = {"email": "buyer@x.com", "total_cents": 6000, "shipping_cents": 0, "get_cents": 0,
             "discount_cents": 0, "points_redeemed_cents": 0}
    appmod._settle_referral(order, order_ref="INV2")
    assert rewards.pending_cash_total(cx, "jane") == 300
    assert points.balance(cx, "jane@x.com") == 0


def test_buyer_first_order_affiliate_suppresses_buyer_points(monkeypatch, tmp_path):
    cx = _db(monkeypatch, tmp_path)
    _refer(cx, "buyer@x.com", "doc", "doc@x.com", ["type:practitioner"])
    # Insert the "current" order so list_orders_by_email returns exactly 1 row
    cx.execute("INSERT INTO orders (email, created_at) VALUES (?, '2026-01-01')", ("buyer@x.com",))
    cx.commit()
    # no prior orders for buyer -> first order; attributed to an affiliate -> suppress buyer earn
    order = {"email": "buyer@x.com", "total_cents": 6000, "shipping_cents": 0, "get_cents": 0,
             "discount_cents": 0, "points_redeemed_cents": 0}
    appmod._settle_order_points(order, order_ref="INV3")
    assert points.balance(cx, "buyer@x.com") == 0        # suppressed (affiliate-acquired first order)


def test_self_referral_excluded(monkeypatch, tmp_path):
    cx = _db(monkeypatch, tmp_path)
    # Same email is both buyer and referrer
    _refer(cx, "self@x.com", "self", "self@x.com", ["type:client"])
    order = {"email": "self@x.com", "total_cents": 6000, "shipping_cents": 0, "get_cents": 0,
             "discount_cents": 0, "points_redeemed_cents": 0}
    appmod._settle_referral(order, order_ref="INV4")
    # Self-referral: no points credited to referrer
    assert points.balance(cx, "self@x.com") == 0


def test_discounted_order_no_referral_credit(monkeypatch, tmp_path):
    cx = _db(monkeypatch, tmp_path)
    _refer(cx, "buyer@x.com", "doc", "doc@x.com", ["type:practitioner"])
    order = {"email": "buyer@x.com", "total_cents": 5000, "shipping_cents": 0, "get_cents": 0,
             "discount_cents": 1000, "points_redeemed_cents": 0}
    appmod._settle_referral(order, order_ref="INV5")
    assert points.balance(cx, "doc@x.com") == 0


def test_referral_cert_scaled_for_practitioner(monkeypatch, tmp_path):
    cx = _db(monkeypatch, tmp_path)
    _refer(cx, "buyer@x.com", "doc", "doc@x.com", ["type:practitioner"])
    monkeypatch.setattr(appmod._pp, "modules_completed_for_email", lambda e: 12)
    order = {"email": "buyer@x.com", "total_cents": 6000, "shipping_cents": 0, "get_cents": 0,
             "discount_cents": 0, "points_redeemed_cents": 0}
    appmod._settle_referral(order, order_ref="INVC1")
    assert points.balance(cx, "doc@x.com") == 900        # 15% of 6000


def test_referral_base_pct_for_non_practitioner(monkeypatch, tmp_path):
    cx = _db(monkeypatch, tmp_path)
    _refer(cx, "buyer@x.com", "doc", "doc@x.com", ["type:practitioner"])
    monkeypatch.setattr(appmod._pp, "modules_completed_for_email", lambda e: None)
    order = {"email": "buyer@x.com", "total_cents": 6000, "shipping_cents": 0, "get_cents": 0,
             "discount_cents": 0, "points_redeemed_cents": 0}
    appmod._settle_referral(order, order_ref="INVC2")
    assert points.balance(cx, "doc@x.com") == 300        # 5% of 6000


def test_referral_cert_lookup_failure_falls_back_to_base(monkeypatch, tmp_path):
    cx = _db(monkeypatch, tmp_path)
    _refer(cx, "buyer@x.com", "doc", "doc@x.com", ["type:practitioner"])
    def _boom(e): raise RuntimeError("supabase down")
    monkeypatch.setattr(appmod._pp, "modules_completed_for_email", _boom)
    order = {"email": "buyer@x.com", "total_cents": 6000, "shipping_cents": 0, "get_cents": 0,
             "discount_cents": 0, "points_redeemed_cents": 0}
    appmod._settle_referral(order, order_ref="INVC3")
    assert points.balance(cx, "doc@x.com") == 300        # base 5% on lookup failure
