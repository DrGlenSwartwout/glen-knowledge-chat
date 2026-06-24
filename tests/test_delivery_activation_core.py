import sqlite3
from datetime import datetime, timedelta
import app as appmod
from dashboard import coaching, tracking as T


def _seed():
    cx = sqlite3.connect(appmod.LOG_DB); cx.row_factory = sqlite3.Row
    coaching.init_coaching_table(cx)
    T.init_tracking_schema(cx); T.migrate_add_delivery_columns(cx)
    cx.execute("CREATE TABLE IF NOT EXISTS memberships (id TEXT PRIMARY KEY, email TEXT, "
               "granted_at TEXT, expires_at TEXT, granted_by TEXT, source TEXT, "
               "truly_vip_ref TEXT, notes TEXT, last_reminder_at TEXT)")
    cx.execute("CREATE TABLE IF NOT EXISTS orders (id INTEGER PRIMARY KEY AUTOINCREMENT, "
               "created_at TEXT, source TEXT, external_ref TEXT, email TEXT, status TEXT DEFAULT 'new', "
               "shipment_id INTEGER)")
    for t in ("coaching_windows", "shipments", "memberships", "orders"):
        cx.execute(f"DELETE FROM {t}")
    cx.commit()
    return cx


def _iso(d):
    return (datetime.utcnow() + timedelta(days=d)).isoformat() + "Z"


def test_window_source_threads_through():
    cx = _seed()
    cx.execute("INSERT INTO memberships (id,email,granted_at,expires_at,source) VALUES (?,?,?,?,?)",
               ("m", "d@x.com", _iso(-30), _iso(20), "membership"))
    cx.execute("INSERT INTO orders (created_at,source,external_ref,email) VALUES (?,?,?,?)",
               (_iso(-5), "reorder", "o1", "d@x.com"))
    cx.commit()
    oid = cx.execute("SELECT id FROM orders").fetchone()[0]
    res = appmod._open_coaching_for_order(cx, "d@x.com", oid, "reorder", window_source="delivery")
    assert res["ok"] is True
    assert coaching.active_window(cx, "d@x.com")["source"] == "delivery"


def test_default_window_source_is_self_serve():
    cx = _seed()
    cx.execute("INSERT INTO memberships (id,email,granted_at,expires_at,source) VALUES (?,?,?,?,?)",
               ("m", "d2@x.com", _iso(-30), _iso(20), "membership"))
    cx.execute("INSERT INTO orders (created_at,source,external_ref,email) VALUES (?,?,?,?)",
               (_iso(-5), "reorder", "o2", "d2@x.com"))
    cx.commit()
    oid = cx.execute("SELECT id FROM orders").fetchone()[0]
    appmod._open_coaching_for_order(cx, "d2@x.com", oid, "reorder")  # no window_source
    assert coaching.active_window(cx, "d2@x.com")["source"] == "self_serve"


def test_activate_for_shipment_opens_and_marks():
    cx = _seed()
    cx.execute("INSERT INTO memberships (id,email,granted_at,expires_at,source) VALUES (?,?,?,?,?)",
               ("m", "d3@x.com", _iso(-30), _iso(20), "membership"))
    cx.execute("INSERT INTO orders (created_at,source,external_ref,email) VALUES (?,?,?,?)",
               (_iso(-5), "biofield", "uuid-3", "d3@x.com"))
    sid = T.record_shipment(cx, tracking_number="TND3", order_uuid="uuid-3", status="sent")
    cx.commit()
    sh = T.shipment_by_tracking(cx, "TND3")
    res = appmod._activate_coaching_for_shipment(cx, sh, delivered_at=_iso(0))
    assert res["ok"] is True
    sh2 = T.shipment_by_tracking(cx, "TND3")
    assert sh2["delivered_at"] is not None and sh2["coaching_opened"] == 1
    # idempotent: re-activating the same shipment is a no-op
    res2 = appmod._activate_coaching_for_shipment(cx, sh2, delivered_at=_iso(0))
    assert res2.get("skipped") == "already_processed"


def test_activate_ineligible_marks_delivered_no_window():
    cx = _seed()
    cx.execute("INSERT INTO memberships (id,email,granted_at,expires_at,source) VALUES (?,?,?,?,?)",
               ("m", "d4@x.com", _iso(-30), _iso(-20), "membership"))  # lapsed before order
    cx.execute("INSERT INTO orders (created_at,source,external_ref,email) VALUES (?,?,?,?)",
               (_iso(-5), "reorder", "uuid-4", "d4@x.com"))
    sid = T.record_shipment(cx, tracking_number="TND4", order_uuid="uuid-4", status="sent")
    cx.commit()
    sh = T.shipment_by_tracking(cx, "TND4")
    res = appmod._activate_coaching_for_shipment(cx, sh, delivered_at=_iso(0))
    assert res["ok"] is False
    sh2 = T.shipment_by_tracking(cx, "TND4")
    assert sh2["delivered_at"] is not None and sh2["coaching_opened"] == 0
    assert coaching.active_window(cx, "d4@x.com") is None
