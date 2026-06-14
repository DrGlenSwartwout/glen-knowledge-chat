"""Tests for Task 3: cash-out review todo creation at threshold."""
import sqlite3

import app as appmod
from dashboard import points, rewards


def _db(monkeypatch, tmp_path):
    db = str(tmp_path / "t.db")
    monkeypatch.setattr(appmod, "LOG_DB", db)
    cx = sqlite3.connect(db)
    cx.row_factory = sqlite3.Row
    cx.execute("""CREATE TABLE todos (id INTEGER PRIMARY KEY, created_at TEXT, owner TEXT,
                  category TEXT, title TEXT, body TEXT, priority TEXT, status TEXT DEFAULT 'open',
                  source TEXT, dedup_key TEXT UNIQUE)""")
    cx.execute("CREATE TABLE affiliate_signups (slug TEXT UNIQUE, email TEXT, status TEXT)")
    cx.execute("CREATE TABLE people (email TEXT UNIQUE, tags TEXT DEFAULT '[]')")
    points.init_points_table(cx)
    rewards.init_affiliate_earnings_table(cx)
    return cx


def test_cashout_review_raised_over_threshold_idempotent(monkeypatch, tmp_path):
    cx = _db(monkeypatch, tmp_path)
    rewards.accrue_cash(cx, slug="jane", email="jane@x.com", order_ref="INV1", amount_cents=12000)
    appmod._maybe_raise_cashout_review(cx, "jane", "cash")   # 12000 >= 10000 threshold
    n = cx.execute("SELECT COUNT(*) FROM todos WHERE source='affiliate-cashout'").fetchone()[0]
    assert n == 1
    appmod._maybe_raise_cashout_review(cx, "jane", "cash")   # idempotent (dedup_key)
    assert cx.execute("SELECT COUNT(*) FROM todos WHERE source='affiliate-cashout'").fetchone()[0] == 1


def test_no_review_under_threshold(monkeypatch, tmp_path):
    cx = _db(monkeypatch, tmp_path)
    rewards.accrue_cash(cx, slug="jane", email="jane@x.com", order_ref="INV1", amount_cents=500)
    appmod._maybe_raise_cashout_review(cx, "jane", "cash")
    assert cx.execute("SELECT COUNT(*) FROM todos WHERE source='affiliate-cashout'").fetchone()[0] == 0
