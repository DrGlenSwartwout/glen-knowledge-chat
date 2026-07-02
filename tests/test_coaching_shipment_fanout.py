"""Pure resolver behind the combined-shipment coaching fan-out: a delivered
household parcel (one tracking number, many member orders) must resolve ALL its
member orders, not just one, so every client gets their delivery coaching window."""
import sqlite3
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def _db():
    from dashboard import orders as O, coaching as Co
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    O.init_orders_table(cx)
    Co.init_coaching_table(cx)
    return O, Co, cx


def _order(O, cx, ref, *, email, source="funnel", shipment_id=None, status="new"):
    oid = O.upsert_order(cx, source=source, external_ref=ref, email=email,
                         items=[{"name": "X", "qty": 1}], status=status)
    if shipment_id is not None:
        cx.execute("UPDATE orders SET shipment_id=? WHERE id=?", (shipment_id, oid))
    cx.commit()
    return oid


def test_resolves_all_members_of_a_shared_shipment():
    O, Co, cx = _db()
    a = _order(O, cx, "A", email="Des@X.com", shipment_id=7)
    b = _order(O, cx, "B", email="jc@x.com", shipment_id=7)
    _order(O, cx, "C", email="other@x.com", shipment_id=9)  # different shipment
    members = Co.shipment_member_orders(cx, 7)
    assert sorted(m["id"] for m in members) == sorted([a, b])
    # email is lowercased/normalized for the coaching ledger
    assert {m["email"] for m in members} == {"des@x.com", "jc@x.com"}


def test_excludes_cancelled_members():
    O, Co, cx = _db()
    a = _order(O, cx, "A", email="a@x.com", shipment_id=7)
    _order(O, cx, "B", email="b@x.com", shipment_id=7, status="cancelled")
    members = Co.shipment_member_orders(cx, 7)
    assert [m["id"] for m in members] == [a]


def test_falls_back_to_order_uuid_when_no_shipment_link():
    O, Co, cx = _db()
    # legacy single-order shipment that never set orders.shipment_id
    a = _order(O, cx, "INV-123", email="a@x.com")
    members = Co.shipment_member_orders(cx, 555, order_uuid="INV-123")
    assert [m["id"] for m in members] == [a]


def test_shipment_link_wins_over_uuid_fallback():
    O, Co, cx = _db()
    linked = _order(O, cx, "A", email="a@x.com", shipment_id=7)
    _order(O, cx, "INV-123", email="b@x.com")  # would match uuid, but link exists
    members = Co.shipment_member_orders(cx, 7, order_uuid="INV-123")
    assert [m["id"] for m in members] == [linked]


def test_empty_when_nothing_matches():
    O, Co, cx = _db()
    _order(O, cx, "A", email="a@x.com", shipment_id=1)
    assert Co.shipment_member_orders(cx, 999) == []
    assert Co.shipment_member_orders(cx, None) == []


def test_fanout_opens_one_window_per_member_idempotently():
    """The mechanism app._activate_coaching_for_shipment now uses: resolve all
    members, open a window for each. Two clients sharing one household parcel each
    get their own window; re-running (a re-delivery signal) opens nothing new."""
    O, Co, cx = _db()
    a = _order(O, cx, "A", email="des@x.com", shipment_id=7)
    b = _order(O, cx, "B", email="jc@x.com", shipment_id=7)
    members = Co.shipment_member_orders(cx, 7)

    def _fan():
        for m in members:
            Co.open_window(cx, email=m["email"], order_id=m["id"], days=30, source="delivery")

    _fan()
    wins = Co.list_windows(cx)
    assert {w["email"] for w in wins} == {"des@x.com", "jc@x.com"}   # BOTH, not just one
    assert Co.window_for_order(cx, a) and Co.window_for_order(cx, b)
    n = len(wins)
    _fan()  # idempotent: no-stacking (per email) + one-per-order
    assert len(Co.list_windows(cx)) == n
