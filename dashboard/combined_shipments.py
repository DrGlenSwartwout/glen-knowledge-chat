"""Household / combined shipments — group several independent orders into ONE
physical parcel to one shared address, buy ONE label, and fan the tracking number
out onto every member order so each client still gets their own tracking email.

Deliberately a SHIPPING/FULFILLMENT layer only: each member order keeps its own
invoice, payment, and QBO mapping. This module never touches billing.

Mirrors dashboard/orders.py conventions: pure functions take a sqlite connection
(testable), and the board actions self-register on import.

Table name is `combined_shipments` — NOT `shipments`, which the tracking module
(dashboard/tracking.py) already owns keyed by tracking_number.
"""
import json
import os
from datetime import datetime, timezone

from dashboard import orders as _orders

# Statuses an order can be in and still be combinable (not yet out the door).
_TERMINAL = _orders._TERMINAL_STATUSES  # ("shipped","delivered","done","cancelled")
# Only paid-ready orders can join a shipment — mirrors the board's isPaidReady()
# so the combined path can't ship an order that hasn't been paid/claimed.
_PAID_OK = ("paid", "claimed")
# Combined-shipment lifecycle. 'open' = being assembled (members can be added /
# removed); the rest mirror the order lifecycle and are pushed onto every member.
SHIPMENT_STATUSES = ("open", "packed", "shipped", "delivered", "done", "cancelled")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _enabled():
    return str(os.environ.get("HOUSEHOLD_SHIPMENTS_ENABLED", "")).strip().lower() \
        in ("1", "true", "yes", "on")


def init_combined_shipments_table(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS combined_shipments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            created_by TEXT,
            household_id INTEGER,
            ship_to_json TEXT,
            status TEXT NOT NULL DEFAULT 'open',
            tracking_number TEXT,
            label_url TEXT,
            carrier_shipment_id TEXT,
            notes TEXT,
            updated_at TEXT
        )
    """)
    cx.execute("CREATE INDEX IF NOT EXISTS idx_combined_status "
               "ON combined_shipments(status)")
    cx.commit()


def _row_to_dict(row):
    if row is None:
        return None
    d = dict(row)
    d["ship_to"] = json.loads(d.pop("ship_to_json") or "{}")
    return d


def _combinable_reason(order):
    """Why an order can't join a combined shipment, or None if it can."""
    if order is None:
        return "not found"
    if order.get("status") in _TERMINAL:
        return f"already {order.get('status')}"
    if (order.get("channel") or "") == "pickup":
        return "pickup (no shipment)"
    if order.get("group_shipment_id") is not None:
        return f"already in shipment #{order.get('group_shipment_id')}"
    # NOTE: payment is NOT required to GROUP orders — a household can be set up
    # while orders are still proposed/cart. Shipping (label/tracking/pack/ship) is
    # gated on all-members-paid instead (see _require_all_paid).
    return None


def split_shipping_proportional(total_cents, weights):
    """Split a combined shipping charge across members in proportion to `weights`
    (each member's own standalone shipping — what their bottles would cost to ship
    alone). Returns a list of ints summing EXACTLY to total_cents (largest-remainder
    rounding). Falls back to an even split when every weight is zero."""
    total = max(0, int(total_cents or 0))
    n = len(weights)
    if n == 0:
        return []
    ws = [max(0, int(w or 0)) for w in weights]
    sw = sum(ws)
    if sw == 0:                      # no basis -> even split, remainder to the first members
        base = total // n
        rem = total - base * n
        return [base + (1 if i < rem else 0) for i in range(n)]
    raw = [total * w / sw for w in ws]
    floors = [int(r) for r in raw]
    leftover = total - sum(floors)
    # hand the leftover cents to the largest fractional remainders (deterministic)
    order = sorted(range(n), key=lambda i: (raw[i] - floors[i], -i), reverse=True)
    for k in range(leftover):
        floors[order[k]] += 1
    return floors


def paid_member_overpay_cents(paid_cents, total_cents, old_shipping_cents,
                              new_share_cents):
    """The shipping-overpayment credit owed to an ALREADY-PAID combined-shipment
    member after recalc lowered their fair one-parcel shipping share.

    The member paid `paid_cents` (fallback: their billed `total_cents`, for legacy
    paid rows recorded before paid_cents was captured) against a bill that included
    `old_shipping_cents`; their fair total is now
    `total_cents - old_shipping_cents + new_share_cents`. The credit is what they
    paid minus that fair total, never negative — combining only ever lowers a share,
    so a member is never owed a *charge*; if combining didn't lower their share the
    credit is 0. When paid == total (the normal case) this reduces to
    `old_shipping_cents - new_share_cents`, exactly the shipping saving."""
    paid = int(paid_cents or 0) or int(total_cents or 0)
    fair_total = max(0, int(total_cents or 0)
                     - int(old_shipping_cents or 0) + int(new_share_cents or 0))
    return max(0, paid - fair_total)


def _unpaid_members(cx, sid):
    """Member orders of a shipment that aren't paid/claimed yet."""
    return [m for m in _orders.orders_in_group(cx, sid)
            if (m.get("pay_status") or "unpaid") not in _PAID_OK]


def _require_all_paid(cx, sid, verb="ship"):
    """Block a shipping action while any member is still unpaid — you can group
    early, but you can't put a label on / ship an unpaid order."""
    unpaid = _unpaid_members(cx, sid)
    if unpaid:
        names = ", ".join((m.get("name") or m.get("email") or f"#{m['id']}") for m in unpaid)
        raise ValueError(f"cannot {verb} shipment #{sid} — waiting on payment from: {names}")


def create_shipment(cx, order_ids, *, ship_to=None, household_id=None,
                    created_by=None):
    """Group >=2 orders into a new combined shipment to one shared address.
    Validates every order is combinable (exists, not terminal, not pickup, not
    already grouped); raises ValueError listing all problems otherwise. If
    ship_to is omitted, defaults to the first order's address."""
    ids = [int(x) for x in (order_ids or [])]
    if len(ids) < 2:
        raise ValueError("combine needs at least 2 orders")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate order in selection")
    members = [_orders.get_order(cx, oid) for oid in ids]
    problems = []
    for oid, o in zip(ids, members):
        r = _combinable_reason(o)
        if r:
            problems.append(f"order #{oid}: {r}")
    if problems:
        raise ValueError("cannot combine — " + "; ".join(problems))
    if ship_to is None:
        ship_to = dict(members[0].get("address") or {})
        ship_to.setdefault("name", members[0].get("name") or "")
    cur = cx.execute(
        "INSERT INTO combined_shipments (created_at, created_by, household_id, "
        "ship_to_json, status, updated_at) VALUES (?,?,?,?,?,?)",
        (_now(), created_by,
         (int(household_id) if household_id is not None else None),
         json.dumps(ship_to), "open", _now()))
    sid = int(cur.lastrowid)
    for oid in ids:
        _orders.set_order_group(cx, oid, sid)
    cx.commit()
    return get_shipment(cx, sid)


def get_shipment(cx, sid):
    row = cx.execute("SELECT * FROM combined_shipments WHERE id=?", (sid,)).fetchone()
    d = _row_to_dict(row)
    if d is None:
        return None
    d["members"] = _orders.orders_in_group(cx, sid)
    return d


def list_open_shipments(cx):
    rows = cx.execute(
        "SELECT * FROM combined_shipments WHERE status!='cancelled' ORDER BY id DESC"
    ).fetchall()
    out = []
    for r in rows:
        d = _row_to_dict(r)
        d["members"] = _orders.orders_in_group(cx, d["id"])
        out.append(d)
    return out


def _require_open(cx, sid):
    sh = get_shipment(cx, sid)
    if sh is None:
        raise ValueError(f"shipment #{sid} not found")
    if sh["status"] != "open":
        raise ValueError(f"shipment #{sid} is '{sh['status']}' — members are locked "
                         "once a label is bought")
    return sh


def add_order(cx, sid, order_id):
    _require_open(cx, sid)
    oid = int(order_id)
    r = _combinable_reason(_orders.get_order(cx, oid))
    if r:
        raise ValueError(f"cannot add order #{oid}: {r}")
    _orders.set_order_group(cx, oid, sid)
    _touch(cx, sid)
    return get_shipment(cx, sid)


def remove_order(cx, sid, order_id):
    _require_open(cx, sid)
    _orders.set_order_group(cx, int(order_id), None)
    _touch(cx, sid)
    return get_shipment(cx, sid)


def _touch(cx, sid):
    cx.execute("UPDATE combined_shipments SET updated_at=? WHERE id=?", (_now(), sid))
    cx.commit()


def merged_order_view(cx, shipment):
    """Synthetic order-shaped dict feeding easypost.build_shipment/buy_label
    unchanged: union of all member line items, the one shared ship-to address."""
    members = shipment.get("members") or _orders.orders_in_group(cx, shipment["id"])
    ship_to = shipment.get("ship_to") or {}
    items = []
    for m in members:
        items.extend(m.get("items") or [])
    return {
        "name": ship_to.get("name") or (members[0].get("name") if members else "")
                or (members[0].get("email") if members else "") or "Customer",
        "email": (members[0].get("email") if members else "") or "",
        "address": ship_to,
        "items": items,
    }


def record_label(cx, sid, *, tracking_number, label_url="", carrier_shipment_id=None):
    """Store the bought label on the shipment AND fan the tracking number + label
    URL out onto every member order (so each order's own tracking flow works).
    Buying/recording a label locks membership: an 'open' shipment advances to
    'packed' so add/remove (which require 'open') can no longer change what's on
    the already-purchased parcel."""
    _require_all_paid(cx, sid, "label")
    row = cx.execute("SELECT status FROM combined_shipments WHERE id=?",
                     (sid,)).fetchone()
    new_status = "packed" if (row and row["status"] == "open") else None
    if new_status:
        cx.execute(
            "UPDATE combined_shipments SET tracking_number=?, label_url=?, "
            "carrier_shipment_id=?, status=?, updated_at=? WHERE id=?",
            (tracking_number, label_url, carrier_shipment_id, new_status, _now(), sid))
    else:
        cx.execute(
            "UPDATE combined_shipments SET tracking_number=?, label_url=?, "
            "carrier_shipment_id=?, updated_at=? WHERE id=?",
            (tracking_number, label_url, carrier_shipment_id, _now(), sid))
    cx.commit()
    for m in _orders.orders_in_group(cx, sid):
        _orders.set_order_label(cx, m["id"], label_url, tracking_number)
    return get_shipment(cx, sid)


def set_status(cx, sid, status):
    """Advance the shipment and every member order in lockstep. Not for 'cancel'
    (use cancel_shipment) or 'open'."""
    if status not in ("packed", "shipped", "delivered", "done"):
        raise ValueError(f"use set_status only for packed/shipped/delivered/done (got {status})")
    if get_shipment(cx, sid) is None:
        raise ValueError(f"shipment #{sid} not found")
    cx.execute("UPDATE combined_shipments SET status=?, updated_at=? WHERE id=?",
               (status, _now(), sid))
    cx.commit()
    for m in _orders.orders_in_group(cx, sid):
        _orders.set_order_status(cx, m["id"], status)
    return get_shipment(cx, sid)


def cancel_shipment(cx, sid):
    """Cancel the GROUPING: un-group every member back to a standalone order
    (their own status/invoice untouched) and mark the shipment cancelled."""
    if get_shipment(cx, sid) is None:
        raise ValueError(f"shipment #{sid} not found")
    for m in _orders.orders_in_group(cx, sid):
        _orders.set_order_group(cx, m["id"], None)
    cx.execute("UPDATE combined_shipments SET status='cancelled', updated_at=? WHERE id=?",
               (_now(), sid))
    cx.commit()
    return get_shipment(cx, sid)


def _addr_key(order):
    a = order.get("address") or {}
    street = str(a.get("street") or "").strip().lower()
    zc = str(a.get("zip") or "").strip().lower()
    if not street:
        return None
    return ("addr", street, zc)


def suggest_combinable(cx, household_of=None):
    """Read-only: cluster ungrouped, non-terminal, non-pickup orders that could
    ship together — by household (when a household_of(person_id)->id callable is
    provided) or by identical normalized ship-to (street+zip). Returns clusters
    of >=2 as [{key_type, key, orders:[{id,name,email,address,total_cents}]}]."""
    clusters = {}
    for o in _orders.list_orders(cx, limit=500):
        if _combinable_reason(o):  # skip terminal / pickup / already grouped
            continue
        key = None
        if household_of is not None and o.get("person_id") is not None:
            try:
                hh = household_of(o.get("person_id"))
            except Exception:
                hh = None
            if hh:
                key = ("household", str(hh))
        if key is None:
            key = _addr_key(o)
        if key is None:
            continue
        clusters.setdefault(key, []).append({
            "id": o.get("id"), "name": o.get("name") or "",
            "email": o.get("email") or "", "address": o.get("address") or {},
            "total_cents": o.get("total_cents") or 0,
        })
    out = []
    for key, orders in clusters.items():
        if len(orders) < 2:
            continue
        out.append({"key_type": key[0], "key": key[1:], "orders": orders})
    return out


# ── Board actions (self-register on import) ──────────────────────────────────
from dashboard.actions import action, LOW_WRITE
from dashboard.rbac import OWNER, OPS, VA


def _cx_of(params, ctx):
    cx = (ctx or {}).get("cx") or (params or {}).get("cx")
    if cx is None:
        raise ValueError("no db connection")
    return cx


def _actor_name(ctx):
    a = (ctx or {}).get("actor")
    return getattr(a, "name", None) or getattr(a, "role", None) if a else None


def _guard():
    if not _enabled():
        return {"message": "Household shipments are turned off "
                           "(HOUSEHOLD_SHIPMENTS_ENABLED)."}
    return None


def _combine_exec(params, ctx):
    off = _guard()
    if off:
        return off
    cx = _cx_of(params, ctx)
    order_ids = params.get("order_ids") or []
    sh = create_shipment(cx, order_ids, ship_to=params.get("ship_to"),
                         household_id=params.get("household_id"),
                         created_by=_actor_name(ctx))
    n = len(sh.get("members") or [])
    return {"shipment_id": sh["id"], "shipment": sh,
            "message": f"Combined {n} orders into shipment #{sh['id']}."}


def _add_exec(params, ctx):
    off = _guard()
    if off:
        return off
    cx = _cx_of(params, ctx)
    sh = add_order(cx, int(params["shipment_id"]), int(params["order_id"]))
    return {"shipment_id": sh["id"], "shipment": sh,
            "message": f"Added order #{params['order_id']} to shipment #{sh['id']}."}


def _remove_exec(params, ctx):
    off = _guard()
    if off:
        return off
    cx = _cx_of(params, ctx)
    sh = remove_order(cx, int(params["shipment_id"]), int(params["order_id"]))
    return {"shipment_id": sh["id"], "shipment": sh,
            "message": f"Removed order #{params['order_id']} from shipment #{sh['id']}."}


def _create_label_exec(params, ctx):
    off = _guard()
    if off:
        return off
    from dashboard import easypost as EP
    cx = _cx_of(params, ctx)
    sid = int(params["shipment_id"])
    sh = get_shipment(cx, sid)
    if sh is None:
        raise ValueError(f"shipment #{sid} not found")
    if sh.get("tracking_number"):
        return {"shipment_id": sid, "tracking_number": sh["tracking_number"],
                "label_url": sh.get("label_url", ""),
                "message": f"Shipment #{sid} already has a label "
                           f"(tracking {sh['tracking_number']})."}
    if not EP.is_configured():
        return {"shipment_id": sid, "handoff": EP.CLICKNSHIP_URL,
                "message": "No label API configured. Buy ONE label on USPS "
                           "Click-N-Ship for the combined parcel, then use "
                           "Set tracking to record the number."}
    from_addr = (ctx or {}).get("from_address") or {}
    out = EP.buy_label(merged_order_view(cx, sh), from_addr)
    record_label(cx, sid, tracking_number=out.get("tracking_number", ""),
                 label_url=out.get("label_url", ""))
    return {"shipment_id": sid, "tracking_number": out.get("tracking_number", ""),
            "label_url": out.get("label_url", ""),
            "message": f"Label bought for shipment #{sid} "
                       f"(tracking {out.get('tracking_number','')})."}


def _set_tracking_exec(params, ctx):
    """Record a manually-bought (Click-N-Ship) tracking number onto the combined
    shipment + every member order."""
    off = _guard()
    if off:
        return off
    cx = _cx_of(params, ctx)
    sid = int(params["shipment_id"])
    tn = str(params.get("tracking_number", "")).strip()
    if not tn:
        raise ValueError("tracking_number required")
    if get_shipment(cx, sid) is None:
        raise ValueError(f"shipment #{sid} not found")
    record_label(cx, sid, tracking_number=tn,
                 label_url=str(params.get("label_url", "") or ""))
    return {"shipment_id": sid, "tracking_number": tn,
            "message": f"Tracking {tn} recorded on shipment #{sid} and its orders."}


def _send_tracking_exec(params, ctx):
    """Email EACH member client their own tracking email for the shared number,
    then record the tracking `shipments` row once and mark members shipped. Does
    NOT reuse orders.send_tracking (its per-tracking-number dedup would skip the
    2nd client)."""
    off = _guard()
    if off:
        return off
    from dashboard import tracking as T
    from dashboard.orders import _gmail_send_tracking
    cx = _cx_of(params, ctx)
    sid = int(params["shipment_id"])
    sh = get_shipment(cx, sid)
    if sh is None:
        raise ValueError(f"shipment #{sid} not found")
    tn = sh.get("tracking_number") or ""
    if not tn:
        raise ValueError("shipment has no tracking number yet "
                         "(create a label or set tracking first)")
    try:
        T.init_tracking_schema(cx)
    except Exception:
        pass
    members = sh.get("members") or []
    # Re-send guard: if this tracking number was already recorded, the emails
    # already went out — don't re-email every client on a second button press.
    if T.shipment_exists(cx, tn):
        set_status(cx, sid, "shipped")
        return {"shipment_id": sid, "tracking_number": tn, "emailed": [],
                "message": f"Tracking {tn} was already sent for shipment #{sid}."}
    emailed = []
    for m in members:
        email = m.get("email") or ""
        if not email:
            continue
        # build_tracking_email splits the name -> guard blank-but-truthy names,
        # and never let one bad member abort the whole household's send.
        name = (m.get("name") or "").strip() or None
        try:
            em = T.build_tracking_email(tn, name)
            if _gmail_send_tracking(email, em.get("subject", "tracking number"),
                                    em.get("html", "")):
                emailed.append(email)
        except Exception as e:
            print(f"[combined_shipments] tracking email skipped for {email}: {e!r}",
                  flush=True)
    # Record the tracking `shipments` row once, then link EVERY member order to it
    # so delivery detection / reporting joins (orders.shipment_id) see all members,
    # not just the first.
    row_id = None
    try:
        first = members[0] if members else {}
        T.record_shipment(cx, tracking_number=tn, recipient_name=first.get("name"),
                          resolved_email=first.get("email"),
                          status=("sent" if emailed else "drafted"),
                          order_uuid=first.get("external_ref"))
        sh_row = T.shipment_by_tracking(cx, tn)
        if sh_row is not None:
            row_id = sh_row["id"]
    except Exception as e:
        print(f"[combined_shipments] shipment record: {e!r}", flush=True)
    if row_id is not None:
        for m in members:
            _orders.set_order_tracking(cx, m["id"], tn, shipment_id=row_id)
    set_status(cx, sid, "shipped")
    return {"shipment_id": sid, "tracking_number": tn, "emailed": emailed,
            "message": f"Tracking {tn} sent to {len(emailed)} client(s); "
                       f"shipment #{sid} + orders marked shipped."}


def _status_exec(new_status, verb):
    def _exec(params, ctx):
        off = _guard()
        if off:
            return off
        cx = _cx_of(params, ctx)
        sid = int(params["shipment_id"])
        if new_status in ("packed", "shipped"):
            _require_all_paid(cx, sid, "pack" if new_status == "packed" else "ship")
        set_status(cx, sid, new_status)
        return {"shipment_id": sid, "status": new_status,
                "message": f"Shipment #{sid} {verb}."}
    return _exec


def _cancel_exec(params, ctx):
    off = _guard()
    if off:
        return off
    cx = _cx_of(params, ctx)
    sid = int(params["shipment_id"])
    cancel_shipment(cx, sid)
    return {"shipment_id": sid, "status": "cancelled",
            "message": f"Shipment #{sid} cancelled — orders un-grouped."}


action(key="shipments.combine", module="orders", title="Combine into shipment",
       description="Group several orders into one household shipment to a shared address.",
       risk_tier=LOW_WRITE, permission=(OWNER, OPS, VA))(_combine_exec)
action(key="shipments.add", module="orders", title="Add order to shipment",
       description="Add an order to an open combined shipment.",
       risk_tier=LOW_WRITE, permission=(OWNER, OPS, VA))(_add_exec)
action(key="shipments.remove", module="orders", title="Remove order from shipment",
       description="Remove an order from an open combined shipment.",
       risk_tier=LOW_WRITE, permission=(OWNER, OPS, VA))(_remove_exec)
action(key="shipments.create_label", module="orders", title="Create combined label",
       description="Buy one USPS label for the combined parcel (or hand off to Click-N-Ship).",
       risk_tier=LOW_WRITE, permission=(OWNER, OPS, VA))(_create_label_exec)
action(key="shipments.set_tracking", module="orders", title="Set combined tracking",
       description="Record a Click-N-Ship tracking number on the shipment + its orders.",
       risk_tier=LOW_WRITE, permission=(OWNER, OPS, VA))(_set_tracking_exec)
action(key="shipments.send_tracking", module="orders", title="Send combined tracking",
       description="Email every member client the shared tracking number; mark shipped.",
       risk_tier=LOW_WRITE, permission=(OWNER, OPS, VA))(_send_tracking_exec)
action(key="shipments.mark_packed", module="orders", title="Mark shipment packed",
       description="Mark a combined shipment and its orders packed.",
       risk_tier=LOW_WRITE, permission=(OWNER, OPS, VA))(_status_exec("packed", "marked packed"))
action(key="shipments.mark_shipped", module="orders", title="Mark shipment shipped",
       description="Mark a combined shipment and its orders shipped.",
       risk_tier=LOW_WRITE, permission=(OWNER, OPS, VA))(_status_exec("shipped", "marked shipped"))
action(key="shipments.mark_delivered", module="orders", title="Mark shipment delivered",
       description="Mark a combined shipment and its orders delivered.",
       risk_tier=LOW_WRITE, permission=(OWNER, OPS, VA))(_status_exec("delivered", "marked delivered"))
action(key="shipments.mark_done", module="orders", title="Mark shipment done",
       description="Mark a combined shipment and its orders done.",
       risk_tier=LOW_WRITE, permission=(OWNER, OPS, VA))(_status_exec("done", "marked done"))
action(key="shipments.cancel", module="orders", title="Cancel combined shipment",
       description="Un-group a combined shipment back into standalone orders.",
       risk_tier=LOW_WRITE, permission=(OWNER, OPS, VA))(_cancel_exec)
