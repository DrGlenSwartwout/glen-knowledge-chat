import json
import os
import sqlite3
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def _db():
    from dashboard import orders as O
    from dashboard import combined_shipments as C
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    O.init_orders_table(cx)
    C.init_combined_shipments_table(cx)
    return O, C, cx


ADDR = {"street": "351 Wailuku Dr", "city": "Hilo", "state": "HI",
        "zip": "96720", "country": "US"}


def _order(O, cx, ref, *, name, email, items, channel="retail",
           address=None, total_cents=5000, status="new", paid=True):
    oid = O.upsert_order(cx, source="manual", external_ref=ref, name=name,
                         email=email, items=items, channel=channel,
                         address=(ADDR if address is None else address),
                         total_cents=total_cents)
    if paid:
        # -> status 'new', pay_status 'paid' (combinable requires paid-ready)
        O.set_order_payment(cx, oid, method="Check", amount_cents=total_cents)
    if status != "new":
        O.set_order_status(cx, oid, status)
    return oid


def _two(O, cx):
    a = _order(O, cx, "A", name="Desire'e Dalla Guardia", email="des@x.com",
               items=[{"name": "Mag", "qty": 2, "slug": "neuro-magnesium"}])
    b = _order(O, cx, "B", name="J.C. Davis", email="jc@x.com",
               items=[{"name": "Terrain", "qty": 1, "slug": "terrain-restore"}])
    return a, b


# ── create / grouping ────────────────────────────────────────────────────────

def test_combine_stamps_group_and_lists_members():
    O, C, cx = _db()
    a, b = _two(O, cx)
    sh = C.create_shipment(cx, [a, b], created_by="Rae")
    assert sh["status"] == "open"
    assert {m["id"] for m in sh["members"]} == {a, b}
    assert O.get_order(cx, a)["group_shipment_id"] == sh["id"]
    assert O.get_order(cx, b)["group_shipment_id"] == sh["id"]
    # ship-to defaulted to the first order's address
    assert sh["ship_to"]["zip"] == "96720"


def test_combine_allows_unpaid_orders_but_blocks_shipping_until_paid():
    # New policy: you may GROUP orders before they're paid (e.g. a proposed
    # in-house invoice + a cart order for the same household), so the shipment can
    # be set up early. Shipping (label/tracking/pack/ship) stays blocked until
    # every member is paid — you still can't ship an unpaid order.
    O, C, cx = _db()
    a = _order(O, cx, "A", name="J.C. Davis", email="jc@x.com",
               items=[{"name": "T", "qty": 1}], status="proposed", paid=False)
    b = _order(O, cx, "B", name="Desire'e", email="des@x.com",
               items=[{"name": "M", "qty": 1}], status="new", paid=False)  # cart
    sh = C.create_shipment(cx, [a, b])                       # grouping is allowed now
    assert sh["status"] == "open"
    assert {m["id"] for m in sh["members"]} == {a, b}
    # ...but a label can't be applied while a member is unpaid
    try:
        C.record_label(cx, sh["id"], tracking_number="9400111899")
        assert False, "expected ValueError — cannot ship unpaid members"
    except ValueError as e:
        assert "waiting on payment" in str(e).lower()
    # pay both, then the label goes through
    O.set_order_payment(cx, a, method="Check", amount_cents=5000)
    O.set_order_payment(cx, b, method="Zelle", amount_cents=5000)
    sh2 = C.record_label(cx, sh["id"], tracking_number="9400111899")
    assert sh2["status"] == "packed"


def test_split_shipping_proportional_by_own_shipping():
    from dashboard import combined_shipments as C
    # combined parcel $11.00; JC would ship $6 alone, Desiree $9 alone -> 6:9 of 1100
    shares = C.split_shipping_proportional(1100, [600, 900])
    assert sum(shares) == 1100
    assert shares == [440, 660]


def test_split_shipping_proportional_is_exact_with_rounding():
    from dashboard import combined_shipments as C
    shares = C.split_shipping_proportional(1000, [100, 200])  # 333.33 / 666.67
    assert sum(shares) == 1000
    assert shares == [333, 667]        # leftover cent to the larger fractional part


def test_split_shipping_zero_weights_falls_back_to_even():
    from dashboard import combined_shipments as C
    assert C.split_shipping_proportional(900, [0, 0]) == [450, 450]
    assert sum(C.split_shipping_proportional(1001, [0, 0, 0])) == 1001


def test_combinable_reason_allows_unpaid_pre_ship_order():
    O, C, cx = _db()
    a = _order(O, cx, "A", name="X", email="x@x.com", items=[{"name": "M", "qty": 1}],
               status="proposed", paid=False)
    assert C._combinable_reason(O.get_order(cx, a)) is None


def test_combine_needs_two_orders():
    O, C, cx = _db()
    a, _ = _two(O, cx)
    try:
        C.create_shipment(cx, [a])
        assert False, "expected ValueError"
    except ValueError as e:
        assert "at least 2" in str(e)


def test_combine_rejects_shipped_pickup_and_already_grouped():
    O, C, cx = _db()
    a, b = _two(O, cx)
    shipped = _order(O, cx, "S", name="Sam", email="s@x.com",
                     items=[{"name": "X", "qty": 1}], status="shipped")
    pickup = _order(O, cx, "P", name="Pat", email="p@x.com",
                    items=[{"name": "Y", "qty": 1}], channel="pickup")
    C.create_shipment(cx, [a, b])          # a,b now grouped
    try:
        C.create_shipment(cx, [a, shipped, pickup])
        assert False, "expected ValueError"
    except ValueError as e:
        msg = str(e)
        assert "already new" not in msg           # a is grouped, not "new"
        assert "already in shipment" in msg       # a
        assert "already shipped" in msg           # shipped
        assert "pickup" in msg                    # pickup


# ── merged view / label fan-out ──────────────────────────────────────────────

def test_merged_order_view_unions_items_one_address():
    O, C, cx = _db()
    a, b = _two(O, cx)
    sh = C.create_shipment(cx, [a, b])
    mv = C.merged_order_view(cx, sh)
    names = sorted(i["name"] for i in mv["items"])
    assert names == ["Mag", "Terrain"]
    assert mv["address"]["zip"] == "96720"


def test_record_label_fans_tracking_onto_all_members():
    O, C, cx = _db()
    a, b = _two(O, cx)
    sh = C.create_shipment(cx, [a, b])
    C.record_label(cx, sh["id"], tracking_number="9400111899",
                   label_url="http://lbl/1")
    assert O.get_order(cx, a)["tracking_number"] == "9400111899"
    assert O.get_order(cx, b)["tracking_number"] == "9400111899"
    assert O.get_order(cx, a)["label_url"] == "http://lbl/1"
    assert C.get_shipment(cx, sh["id"])["tracking_number"] == "9400111899"


# ── status lockstep / cancel ─────────────────────────────────────────────────

def test_set_status_moves_members_in_lockstep():
    O, C, cx = _db()
    a, b = _two(O, cx)
    sh = C.create_shipment(cx, [a, b])
    C.set_status(cx, sh["id"], "packed")
    assert O.get_order(cx, a)["status"] == "packed"
    assert O.get_order(cx, b)["status"] == "packed"
    assert C.get_shipment(cx, sh["id"])["status"] == "packed"


def test_cancel_ungroups_members():
    O, C, cx = _db()
    a, b = _two(O, cx)
    sh = C.create_shipment(cx, [a, b])
    C.cancel_shipment(cx, sh["id"])
    assert O.get_order(cx, a)["group_shipment_id"] is None
    assert O.get_order(cx, b)["group_shipment_id"] is None
    assert C.get_shipment(cx, sh["id"])["status"] == "cancelled"
    # order lifecycle status is untouched (still standalone 'new')
    assert O.get_order(cx, a)["status"] == "new"


def test_members_locked_after_leaving_open():
    O, C, cx = _db()
    a, b = _two(O, cx)
    c = _order(O, cx, "C", name="Cy", email="c@x.com", items=[{"name": "Z", "qty": 1}])
    sh = C.create_shipment(cx, [a, b])
    C.set_status(cx, sh["id"], "packed")
    try:
        C.add_order(cx, sh["id"], c)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "locked" in str(e)


def test_record_label_locks_membership():
    O, C, cx = _db()
    a, b = _two(O, cx)
    c = _order(O, cx, "C", name="Cy", email="c@x.com", items=[{"name": "Z", "qty": 1}])
    sh = C.create_shipment(cx, [a, b])
    C.record_label(cx, sh["id"], tracking_number="9400111899")   # open -> packed
    assert C.get_shipment(cx, sh["id"])["status"] == "packed"
    try:
        C.add_order(cx, sh["id"], c)
        assert False, "expected add to be locked after label"
    except ValueError as e:
        assert "locked" in str(e)


def test_mark_delivered():
    O, C, cx = _db()
    a, b = _two(O, cx)
    sh = C.create_shipment(cx, [a, b])
    C.set_status(cx, sh["id"], "delivered")
    assert C.get_shipment(cx, sh["id"])["status"] == "delivered"
    assert O.get_order(cx, a)["status"] == "delivered"


# ── add / remove while open ──────────────────────────────────────────────────

def test_add_and_remove_while_open():
    O, C, cx = _db()
    a, b = _two(O, cx)
    c = _order(O, cx, "C", name="Cy", email="c@x.com", items=[{"name": "Z", "qty": 1}])
    sh = C.create_shipment(cx, [a, b])
    sh = C.add_order(cx, sh["id"], c)
    assert {m["id"] for m in sh["members"]} == {a, b, c}
    sh = C.remove_order(cx, sh["id"], c)
    assert {m["id"] for m in sh["members"]} == {a, b}
    assert O.get_order(cx, c)["group_shipment_id"] is None


# ── suggestions ──────────────────────────────────────────────────────────────

def test_suggest_by_address_clusters_two_same_address():
    O, C, cx = _db()
    _two(O, cx)  # both at ADDR
    _order(O, cx, "F", name="Far", email="f@x.com", items=[{"name": "Q", "qty": 1}],
           address={"street": "9 Other Rd", "city": "Kona", "state": "HI", "zip": "96740"})
    sug = C.suggest_combinable(cx)
    assert len(sug) == 1
    assert sug[0]["key_type"] == "addr"
    assert len(sug[0]["orders"]) == 2


def test_suggest_ignores_grouped_terminal_pickup():
    O, C, cx = _db()
    a, b = _two(O, cx)
    C.create_shipment(cx, [a, b])  # now grouped -> excluded
    assert C.suggest_combinable(cx) == []


def test_suggest_by_household_hook():
    O, C, cx = _db()
    # two different addresses but same household via the injected hook
    a = _order(O, cx, "A", name="Des", email="d@x.com", items=[{"name": "M", "qty": 1}])
    b = _order(O, cx, "B", name="JC", email="j@x.com", items=[{"name": "T", "qty": 1}],
               address={"street": "2 Elsewhere", "city": "Hilo", "state": "HI", "zip": "96720"})
    O.set_order_group  # noqa
    cx.execute("UPDATE orders SET person_id=? WHERE id=?", (11, a))
    cx.execute("UPDATE orders SET person_id=? WHERE id=?", (12, b))
    cx.commit()
    sug = C.suggest_combinable(cx, household_of=lambda pid: "dalla-davis")
    assert len(sug) == 1
    assert sug[0]["key_type"] == "household"


# ── billing untouched (regression guard) ─────────────────────────────────────

def test_billing_never_changes_across_flow():
    O, C, cx = _db()
    a, b = _two(O, cx)
    before = {oid: (O.get_order(cx, oid)["total_cents"],
                    O.get_order(cx, oid)["pay_status"]) for oid in (a, b)}
    sh = C.create_shipment(cx, [a, b])
    C.record_label(cx, sh["id"], tracking_number="9400111899")
    C.set_status(cx, sh["id"], "shipped")
    C.cancel_shipment(cx, sh["id"])
    after = {oid: (O.get_order(cx, oid)["total_cents"],
                   O.get_order(cx, oid)["pay_status"]) for oid in (a, b)}
    assert before == after


# ── actions + flag gate ──────────────────────────────────────────────────────

class _Actor:
    role = "owner"
    name = "Rae"


def test_actions_gated_by_flag(monkeypatch):
    O, C, cx = _db()
    a, b = _two(O, cx)
    monkeypatch.delenv("HOUSEHOLD_SHIPMENTS_ENABLED", raising=False)
    res = C._combine_exec({"order_ids": [a, b]}, {"cx": cx, "actor": _Actor()})
    assert "turned off" in res["message"]
    # nothing grouped
    assert O.get_order(cx, a)["group_shipment_id"] is None


def test_combine_action_and_send_tracking(monkeypatch):
    O, C, cx = _db()
    a, b = _two(O, cx)
    monkeypatch.setenv("HOUSEHOLD_SHIPMENTS_ENABLED", "1")
    sent = []
    import dashboard.orders as _O
    monkeypatch.setattr(_O, "_gmail_send_tracking",
                        lambda to, subj, html: sent.append(to) or True)
    ctx = {"cx": cx, "actor": _Actor()}
    res = C._combine_exec({"order_ids": [a, b]}, ctx)
    sid = res["shipment_id"]
    C._set_tracking_exec({"shipment_id": sid, "tracking_number": "9400111899"}, ctx)
    out = C._send_tracking_exec({"shipment_id": sid}, ctx)
    # each member client got their own email
    assert set(sent) == {"des@x.com", "jc@x.com"}
    assert set(out["emailed"]) == {"des@x.com", "jc@x.com"}
    assert O.get_order(cx, a)["status"] == "shipped"
    assert O.get_order(cx, b)["status"] == "shipped"
    # every member is linked to the tracking `shipments` row (delivery/reporting join)
    from dashboard import tracking as T
    row = T.shipment_by_tracking(cx, "9400111899")
    assert row is not None
    assert O.get_order(cx, a)["shipment_id"] == row["id"]
    assert O.get_order(cx, b)["shipment_id"] == row["id"]


def test_send_tracking_resend_guard_no_double_email(monkeypatch):
    O, C, cx = _db()
    a, b = _two(O, cx)
    monkeypatch.setenv("HOUSEHOLD_SHIPMENTS_ENABLED", "1")
    sent = []
    import dashboard.orders as _O
    monkeypatch.setattr(_O, "_gmail_send_tracking",
                        lambda to, subj, html: sent.append(to) or True)
    ctx = {"cx": cx, "actor": _Actor()}
    sid = C._combine_exec({"order_ids": [a, b]}, ctx)["shipment_id"]
    C._set_tracking_exec({"shipment_id": sid, "tracking_number": "9400111899"}, ctx)
    C._send_tracking_exec({"shipment_id": sid}, ctx)
    assert len(sent) == 2
    # second click must NOT re-email
    out2 = C._send_tracking_exec({"shipment_id": sid}, ctx)
    assert len(sent) == 2
    assert out2["emailed"] == []
    assert "already sent" in out2["message"]


def test_send_tracking_survives_blank_name(monkeypatch):
    O, C, cx = _db()
    a = _order(O, cx, "A", name="   ", email="des@x.com", items=[{"name": "M", "qty": 1}])
    b = _order(O, cx, "B", name="J.C. Davis", email="jc@x.com", items=[{"name": "T", "qty": 1}])
    monkeypatch.setenv("HOUSEHOLD_SHIPMENTS_ENABLED", "1")
    sent = []
    import dashboard.orders as _O
    monkeypatch.setattr(_O, "_gmail_send_tracking",
                        lambda to, subj, html: sent.append(to) or True)
    ctx = {"cx": cx, "actor": _Actor()}
    sid = C._combine_exec({"order_ids": [a, b]}, ctx)["shipment_id"]
    C._set_tracking_exec({"shipment_id": sid, "tracking_number": "9400111899"}, ctx)
    out = C._send_tracking_exec({"shipment_id": sid}, ctx)
    # the blank-name member doesn't crash the send; both still get emailed
    assert set(sent) == {"des@x.com", "jc@x.com"}
    assert set(out["emailed"]) == {"des@x.com", "jc@x.com"}
