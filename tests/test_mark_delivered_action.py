import sqlite3
from datetime import datetime, timedelta
import app as appmod
from dashboard import coaching, dispatch as D, rbac as R


def _seed_order(email="md@x.com", src="reorder"):
    cx = sqlite3.connect(appmod.LOG_DB); cx.row_factory = sqlite3.Row
    coaching.init_coaching_table(cx)
    cx.execute("CREATE TABLE IF NOT EXISTS memberships (id TEXT PRIMARY KEY, email TEXT, granted_at TEXT, "
               "expires_at TEXT, granted_by TEXT, source TEXT, truly_vip_ref TEXT, notes TEXT, last_reminder_at TEXT)")
    cx.execute("CREATE TABLE IF NOT EXISTS orders (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, "
               "source TEXT, external_ref TEXT, email TEXT, status TEXT DEFAULT 'new', shipment_id INTEGER, updated_at TEXT)")
    cx.execute("DELETE FROM coaching_windows WHERE email=?", (email,))
    cx.execute("DELETE FROM orders WHERE email=?", (email,))
    cx.execute("DELETE FROM memberships WHERE email=?", (email,))
    iso = lambda d: (datetime.utcnow() + timedelta(days=d)).isoformat() + "Z"
    cx.execute("INSERT OR REPLACE INTO memberships (id,email,granted_at,expires_at,source) VALUES (?,?,?,?,?)",
               ("m-" + email, email, iso(-30), iso(20), "membership"))
    cx.execute("INSERT INTO orders (created_at,source,external_ref,email) VALUES (?,?,?,?)",
               (iso(-5), src, "MD-1", email))
    cx.commit()
    oid = cx.execute("SELECT id FROM orders WHERE email=?", (email,)).fetchone()[0]
    return cx, oid


def test_mark_delivered_opens_coaching():
    cx, oid = _seed_order(email="md1@x.com")
    actor = R.Actor(role=R.OWNER, name="glen")
    res = D.dispatch_action(cx, "orders.mark_delivered", {"order_id": oid}, actor,
                            source="panel", confirmed=True)
    assert res["status"] == "done"
    assert coaching.active_window(cx, "md1@x.com") is not None
    assert coaching.active_window(cx, "md1@x.com")["source"] == "delivery"


def test_mark_delivered_ineligible_order_no_window():
    cx, oid = _seed_order(email="md2@x.com", src="membership")  # non-qualifying source
    actor = R.Actor(role=R.OWNER, name="glen")
    res = D.dispatch_action(cx, "orders.mark_delivered", {"order_id": oid}, actor,
                            source="panel", confirmed=True)
    assert res["status"] == "done"            # action still succeeds
    assert coaching.active_window(cx, "md2@x.com") is None  # but no window
