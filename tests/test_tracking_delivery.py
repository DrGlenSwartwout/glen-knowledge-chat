import sqlite3
from dashboard import tracking as T


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    T.init_tracking_schema(cx)
    T.migrate_add_delivery_columns(cx)
    return cx


def _seed(cx, tn="9400111", **kw):
    return T.record_shipment(cx, tracking_number=tn, order_uuid=kw.get("order_uuid", "u1"),
                             status="sent")


def test_migrate_adds_columns_idempotent():
    cx = _cx()
    T.migrate_add_delivery_columns(cx)  # second call must not raise
    cols = {r[1] for r in cx.execute("PRAGMA table_info(shipments)").fetchall()}
    assert {"delivered_at", "coaching_opened", "easypost_tracker_id"} <= cols


def test_shipment_by_tracking_roundtrip():
    cx = _cx()
    sid = _seed(cx, "TN-1")
    row = T.shipment_by_tracking(cx, "TN-1")
    assert row is not None and row["id"] == sid
    assert T.shipment_by_tracking(cx, "nope") is None


def test_mark_delivered_only_sets_once():
    cx = _cx()
    sid = _seed(cx, "TN-2")
    assert T.mark_shipment_delivered(cx, sid, "2026-06-23T00:00:00Z") is True
    # second call is a no-op (already set) -> returns False, value unchanged
    assert T.mark_shipment_delivered(cx, sid, "2026-07-01T00:00:00Z") is False
    assert T.shipment_by_tracking(cx, "TN-2")["delivered_at"] == "2026-06-23T00:00:00Z"


def test_set_tracker_id():
    cx = _cx()
    sid = _seed(cx, "TN-3")
    assert T.set_shipment_tracker(cx, sid, "trk_abc") is True
    assert T.shipment_by_tracking(cx, "TN-3")["easypost_tracker_id"] == "trk_abc"
