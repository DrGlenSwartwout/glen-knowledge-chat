import sqlite3
from datetime import datetime, timedelta
import app as appmod
from dashboard import coaching


def _seed():
    """Fresh tables in the real LOG_DB; return a Row-cx the helpers will reopen."""
    cx = sqlite3.connect(appmod.LOG_DB)
    cx.row_factory = sqlite3.Row
    coaching.init_coaching_table(cx)
    cx.execute("CREATE TABLE IF NOT EXISTS memberships (id TEXT PRIMARY KEY, email TEXT, "
               "granted_at TEXT, expires_at TEXT, granted_by TEXT, source TEXT, "
               "truly_vip_ref TEXT, notes TEXT, last_reminder_at TEXT)")
    cx.execute("CREATE TABLE IF NOT EXISTS orders (id INTEGER PRIMARY KEY AUTOINCREMENT, "
               "created_at TEXT, source TEXT, external_ref TEXT, email TEXT, status TEXT DEFAULT 'new')")
    for t in ("coaching_windows", "memberships", "orders"):
        cx.execute(f"DELETE FROM {t}")
    cx.commit()
    return cx


def _iso(days_off):
    return (datetime.utcnow() + timedelta(days=days_off)).isoformat() + "Z"


def test_membership_active_at_covers_order_in_paid_span():
    cx = _seed()
    cx.execute("INSERT INTO memberships (id,email,granted_at,expires_at,source) VALUES (?,?,?,?,?)",
               ("m1", "p@x.com", _iso(-40), _iso(20), "membership"))
    cx.commit()
    assert appmod._membership_active_at(cx, "p@x.com", _iso(-10)) is True   # inside span
    assert appmod._membership_active_at(cx, "p@x.com", _iso(-50)) is False  # before grant


def test_open_coaching_for_order_eligible_creates_window():
    cx = _seed()
    cx.execute("INSERT INTO memberships (id,email,granted_at,expires_at,source) VALUES (?,?,?,?,?)",
               ("m1", "p@x.com", _iso(-40), _iso(20), "membership"))
    cx.execute("INSERT INTO orders (created_at,source,external_ref,email) VALUES (?,?,?,?)",
               (_iso(-10), "reorder", "r1", "p@x.com"))
    cx.commit()
    oid = cx.execute("SELECT id FROM orders").fetchone()[0]
    res = appmod._open_coaching_for_order(cx, "p@x.com", oid, "reorder")
    assert res["ok"] is True and res["created"] is True
    assert coaching.active_window(cx, "p@x.com") is not None
    # premium access untouched: still exactly one membership row, expires unchanged
    assert cx.execute("SELECT COUNT(*) FROM memberships").fetchone()[0] == 1


def test_open_coaching_ineligible_when_order_outside_paid_month():
    cx = _seed()
    cx.execute("INSERT INTO memberships (id,email,granted_at,expires_at,source) VALUES (?,?,?,?,?)",
               ("m1", "p@x.com", _iso(-40), _iso(-20), "membership"))  # lapsed before order
    cx.execute("INSERT INTO orders (created_at,source,external_ref,email) VALUES (?,?,?,?)",
               (_iso(-10), "reorder", "r1", "p@x.com"))
    cx.commit()
    oid = cx.execute("SELECT id FROM orders").fetchone()[0]
    res = appmod._open_coaching_for_order(cx, "p@x.com", oid, "reorder")
    assert res["ok"] is False and res["reason"] == "ineligible" and res["offer_99"] is True
    assert coaching.active_window(cx, "p@x.com") is None


def test_non_qualifying_source_rejected():
    cx = _seed()
    cx.execute("INSERT INTO memberships (id,email,granted_at,expires_at,source) VALUES (?,?,?,?,?)",
               ("m1", "p@x.com", _iso(-40), _iso(20), "membership"))
    cx.execute("INSERT INTO orders (created_at,source,external_ref,email) VALUES (?,?,?,?)",
               (_iso(-10), "membership", "c1", "p@x.com"))
    cx.commit()
    oid = cx.execute("SELECT id FROM orders").fetchone()[0]
    res = appmod._open_coaching_for_order(cx, "p@x.com", oid, "membership")
    assert res["ok"] is False and res["reason"] == "not_qualifying"


def test_find_qualifying_order_picks_recent_eligible_unactivated():
    cx = _seed()
    cx.execute("INSERT INTO memberships (id,email,granted_at,expires_at,source) VALUES (?,?,?,?,?)",
               ("m1", "p@x.com", _iso(-60), _iso(20), "membership"))
    cx.execute("INSERT INTO orders (created_at,source,external_ref,email) VALUES (?,?,?,?)",
               (_iso(-30), "reorder", "old", "p@x.com"))
    cx.execute("INSERT INTO orders (created_at,source,external_ref,email) VALUES (?,?,?,?)",
               (_iso(-5), "biofield", "new", "p@x.com"))
    cx.commit()
    newest = cx.execute("SELECT id FROM orders WHERE external_ref='new'").fetchone()[0]
    assert appmod._find_qualifying_order_for_coaching(cx, "p@x.com") == newest


def test_membership_active_at_survives_mixed_timestamp_shapes():
    """memberships rows and the `when` argument can each carry any of the three
    stored shapes. Comparing a naive granted_at against an aware `when` used to
    raise inside a bare except, silently denying an active member their access.
    """
    from datetime import timezone
    cx = _seed()
    base = datetime.utcnow()
    aware = lambda d: d.replace(tzinfo=timezone.utc).isoformat()
    z_naive = lambda d: d.isoformat() + "Z"
    bare = lambda d: d.isoformat()

    # granted_at bare-naive, expires_at Z-suffixed: a real mix from the table
    cx.execute("INSERT INTO memberships (id,email,granted_at,expires_at,source) VALUES (?,?,?,?,?)",
               ("m1", "mix@x.com", bare(base - timedelta(days=40)),
                z_naive(base + timedelta(days=20)), "membership"))
    cx.commit()

    for label, fmt in (("aware", aware), ("z_naive", z_naive), ("bare", bare)):
        inside = fmt(base - timedelta(days=10))
        before = fmt(base - timedelta(days=50))
        assert appmod._membership_active_at(cx, "mix@x.com", inside) is True, label
        assert appmod._membership_active_at(cx, "mix@x.com", before) is False, label
