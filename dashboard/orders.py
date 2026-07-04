"""Business-OS Orders & Fulfillment model. One orders table unifies every order
flow (funnel/QBO retail+wholesale, GrooveKart, dispensary, manual) into a single
lifecycle: new -> packed -> shipped -> done (+ cancelled). Functions take a
sqlite connection for testability. Lifecycle actions + the Home signal register
on import (see Task 2)."""
import json
import os
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path

ORDER_STATUSES = ("proposed", "confirmed", "paid",
                  "new", "packed", "shipped", "done", "cancelled", "delivered")
_OPEN = ("new", "packed")  # unfulfilled
# Pre-fulfillment lead-in for in-house proposed invoices (before the kanban).
_PRE_FULFILL = ("proposed", "confirmed")
_TERMINAL_STATUSES = ("shipped", "delivered", "done", "cancelled")


def _now():
    return datetime.now(timezone.utc).isoformat()


def init_orders_table(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            source TEXT NOT NULL,
            external_ref TEXT NOT NULL,
            channel TEXT DEFAULT 'retail',
            email TEXT, name TEXT, phone TEXT,
            items_json TEXT, total_cents INTEGER DEFAULT 0,
            address_json TEXT,
            status TEXT NOT NULL DEFAULT 'new',
            tracking_number TEXT, shipment_id INTEGER,
            notes TEXT, updated_at TEXT,
            discount_cents INTEGER NOT NULL DEFAULT 0,
            points_redeemed_cents INTEGER NOT NULL DEFAULT 0,
            shipping_cents INTEGER NOT NULL DEFAULT 0,
            UNIQUE(source, external_ref)
        )
    """)
    cx.execute("CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status)")
    try:
        cx.execute("ALTER TABLE orders ADD COLUMN label_url TEXT")
    except Exception:
        pass  # already present
    try:
        cx.execute("ALTER TABLE orders ADD COLUMN stripe_payment_intent TEXT")
    except Exception:
        pass
    try:
        # Hawai'i GET owed on this order (absorbed — NOT charged to the customer;
        # recorded for remittance). 0 when out-of-state or tax tracking is off.
        cx.execute("ALTER TABLE orders ADD COLUMN get_cents INTEGER DEFAULT 0")
    except Exception:
        pass
    for col in ("discount_cents", "points_redeemed_cents", "shipping_cents"):
        try:
            cx.execute(f"ALTER TABLE orders ADD COLUMN {col} INTEGER NOT NULL DEFAULT 0")
        except Exception:
            pass  # already present (created by the CREATE TABLE above or prior migration)
    # In-house order-entry / proposed-invoice fields (Phase 1).
    for ddl in (
        "ALTER TABLE orders ADD COLUMN person_id INTEGER",
        "ALTER TABLE orders ADD COLUMN pay_method TEXT",
        "ALTER TABLE orders ADD COLUMN pay_status TEXT DEFAULT 'unpaid'",
        "ALTER TABLE orders ADD COLUMN paid_at TEXT",
        "ALTER TABLE orders ADD COLUMN paid_cents INTEGER DEFAULT 0",
        # Phase 3: customer pay-link invoice send timestamp.
        "ALTER TABLE orders ADD COLUMN invoice_sent_at TEXT",
        # Optional customer-facing note shown on the invoice (distinct from the
        # internal `notes` column, which never reaches the customer view).
        "ALTER TABLE orders ADD COLUMN invoice_note TEXT",
        # Household / combined-shipment grouping: when set, this order ships in a
        # shared parcel tracked by combined_shipments.id (see
        # dashboard/combined_shipments.py). NULL = ships on its own. Distinct from
        # `shipment_id`, which links to the tracking `shipments` table.
        "ALTER TABLE orders ADD COLUMN group_shipment_id INTEGER",
        # Manual invoice adjustment (SIGNED): negative = credit, positive =
        # debit/surcharge. Applied to the total on top of the rule-based discount;
        # shown as its own line on the invoice + QBO. 0 = none.
        "ALTER TABLE orders ADD COLUMN adjustment_cents INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE orders ADD COLUMN practitioner_id TEXT",
    ):
        try:
            cx.execute(ddl)
        except Exception:
            pass  # already present
    cx.commit()


def upsert_order(cx, *, source, external_ref, email="", name="", phone="",
                 items=None, total_cents=0, address=None, channel="retail",
                 status="new", get_cents=0, person_id=None,
                 discount_cents=0, points_redeemed_cents=0, shipping_cents=0,
                 invoice_note=None, adjustment_cents=0,
                 pay_method=None, practitioner_id=None):
    """Idempotent on (source, external_ref). Inserts a new order, or updates the
    soft fields of an existing one WITHOUT regressing its lifecycle status.
    items and address are only overwritten when explicitly provided (not None).
    get_cents = absorbed Hawai'i GET owed (recorded, not charged).
    discount_cents / points_redeemed_cents / shipping_cents = pricing breakdown."""
    ref = str(external_ref or "").strip()
    if not ref:
        raise ValueError("external_ref required")
    row = cx.execute("SELECT id FROM orders WHERE source=? AND external_ref=?",
                     (source, ref)).fetchone()
    if row:
        # Only overwrite items_json / address_json when caller provides them.
        sets = ["email=?", "name=?", "phone=?", "total_cents=?", "channel=?",
                "get_cents=?", "discount_cents=?", "points_redeemed_cents=?",
                "shipping_cents=?", "adjustment_cents=?", "updated_at=?"]
        vals = [email, name, phone, int(total_cents or 0), channel,
                int(get_cents or 0), int(discount_cents or 0),
                int(points_redeemed_cents or 0), int(shipping_cents or 0),
                int(adjustment_cents or 0), _now()]
        if items is not None:
            sets.insert(3, "items_json=?")
            vals.insert(3, json.dumps(items))
        if address is not None:
            sets.append("address_json=?")
            vals.append(json.dumps(address))
        if person_id is not None:
            sets.append("person_id=?")
            vals.append(int(person_id))
        if invoice_note is not None:
            sets.append("invoice_note=?")
            vals.append(str(invoice_note))
        if pay_method is not None:
            sets.append("pay_method=?")
            vals.append(str(pay_method))
        if practitioner_id is not None:
            sets.append("practitioner_id=?")
            vals.append(str(practitioner_id))
        vals.append(row[0])
        cx.execute(f"UPDATE orders SET {', '.join(sets)} WHERE id=?", vals)
        cx.commit()
        return row[0]
    cur = cx.execute(
        "INSERT INTO orders (created_at, source, external_ref, channel, email, name, "
        "phone, items_json, total_cents, address_json, status, get_cents, person_id, "
        "discount_cents, points_redeemed_cents, shipping_cents, invoice_note, adjustment_cents, "
        "pay_method, practitioner_id) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (_now(), source, ref, channel, email, name, phone,
         json.dumps(items or []), int(total_cents or 0), json.dumps(address or {}),
         status, int(get_cents or 0),
         (int(person_id) if person_id is not None else None),
         int(discount_cents or 0), int(points_redeemed_cents or 0), int(shipping_cents or 0),
         (str(invoice_note) if invoice_note is not None else None), int(adjustment_cents or 0),
         (str(pay_method) if pay_method is not None else None),
         (str(practitioner_id) if practitioner_id is not None else None)))
    cx.commit()
    return cur.lastrowid


def get_tax_report(cx, *, date_from, date_to):
    """Aggregate absorbed Hawai'i GET over a period, in the shape of a G-45 filing:
    HI retail, HI wholesale, out-of-state (export), and unknown-state (no ship-to)
    buckets — each with order count, gross receipts, and GET owed. GET accrues only
    on the HI buckets. Pure read; ship-state comes from each order's address."""
    df = str(date_from or "")
    dt = str(date_to or "")
    if dt and "T" not in dt:
        dt = dt + "T23:59:59.999999"
    rows = cx.execute(
        "SELECT channel, total_cents, get_cents, address_json, created_at FROM orders "
        "WHERE created_at >= ? AND created_at <= ?", (df, dt)).fetchall()

    def _bucket():
        return {"orders": 0, "gross_cents": 0, "get_cents": 0}

    out = {"hi_retail": _bucket(), "hi_wholesale": _bucket(),
           "out_of_state": _bucket(), "unknown_state": _bucket()}
    for r in rows:
        d = dict(r)
        try:
            state = (json.loads(d.get("address_json") or "{}").get("state") or "").strip().upper()
        except Exception:
            state = ""
        gross = int(d.get("total_cents") or 0)
        get_c = int(d.get("get_cents") or 0)
        if not state:
            key = "unknown_state"
        elif state != "HI":
            key = "out_of_state"
        else:
            key = "hi_wholesale" if d.get("channel") == "wholesale" else "hi_retail"
        b = out[key]
        b["orders"] += 1
        b["gross_cents"] += gross
        b["get_cents"] += get_c
    out["from"] = df
    out["to"] = dt
    out["total_get_cents"] = out["hi_retail"]["get_cents"] + out["hi_wholesale"]["get_cents"]
    return out


def _row_to_dict(row):
    if row is None:
        return None
    d = dict(row)
    d["items"] = json.loads(d.pop("items_json") or "[]")
    d["address"] = json.loads(d.pop("address_json") or "{}")
    return d


def get_order(cx, order_id):
    cur = cx.execute("SELECT * FROM orders WHERE id=?", (order_id,))
    return _row_to_dict(cur.fetchone())


def list_orders(cx, *, status=None, limit=200):
    if status:
        cur = cx.execute("SELECT * FROM orders WHERE status=? ORDER BY id DESC LIMIT ?",
                         (status, limit))
    else:
        cur = cx.execute("SELECT * FROM orders ORDER BY id DESC LIMIT ?", (limit,))
    return [_row_to_dict(r) for r in cur.fetchall()]


def list_orders_by_email(cx, email, limit=200):
    """A client's orders, most recent first (for the reorder cart). Caller sets
    cx.row_factory = sqlite3.Row."""
    cur = cx.execute(
        "SELECT * FROM orders WHERE lower(email)=? ORDER BY created_at DESC, id DESC LIMIT ?",
        ((email or "").strip().lower(), limit))
    return [_row_to_dict(r) for r in cur.fetchall()]


def attention_orders(limit=20):
    """Open orders needing attention (status not terminal), newest first, as a
    minimal subset for the briefing snapshot. Self-connects to chat_log.db
    (mirrors briefing_actions._DB). Best-effort: callers wrap in _safe."""
    db = Path(os.environ.get("DATA_DIR", str(Path(__file__).resolve().parent.parent))) / "chat_log.db"
    cx = sqlite3.connect(str(db), timeout=5)
    try:
        cx.row_factory = sqlite3.Row
        out = []
        for o in list_orders(cx, limit=200):
            if o.get("status") in _TERMINAL_STATUSES:
                continue
            out.append({
                "id": o.get("id"),
                "name": o.get("name") or "",
                "email": o.get("email") or "",
                "status": o.get("status"),
                "pay_status": o.get("pay_status") or "",
                "total_cents": o.get("total_cents") or 0,
                "created_at": o.get("created_at"),
                "backorder_units": order_backorder_units(cx, o.get("id")),
            })
            if len(out) >= limit:
                break
        return out
    finally:
        cx.close()


def set_order_status(cx, order_id, status):
    if status not in ORDER_STATUSES:
        raise ValueError(f"unknown status: {status}")
    cur = cx.execute("UPDATE orders SET status=?, updated_at=? WHERE id=?",
                     (status, _now(), order_id))
    cx.commit()
    if status == "cancelled" and cur.rowcount:
        try:
            _ungroup_cancelled_from_shipment(cx, order_id)
        except Exception as e:  # never let group cleanup block the cancel
            print(f"[orders] ungroup-on-cancel skipped for #{order_id}: {e!r}", flush=True)
    return cur.rowcount > 0


def _ungroup_cancelled_from_shipment(cx, order_id):
    """When an order is cancelled, drop it from any still-OPEN combined shipment so
    its stale group link can't block re-combining the remaining members. If that
    leaves the shipment with fewer than 2 members, dissolve it: un-group the lone
    remaining order back to standalone and mark the shipment cancelled — a
    one-order "combined" shipment isn't a combination. No-op once a label is bought
    (shipment status != 'open'), so a packed/shipped parcel is never touched."""
    row = cx.execute("SELECT group_shipment_id FROM orders WHERE id=?", (order_id,)).fetchone()
    sid = row[0] if row else None
    if not sid:
        return
    st = cx.execute("SELECT status FROM combined_shipments WHERE id=?", (sid,)).fetchone()
    if st is None or st[0] != "open":
        return
    cx.execute("UPDATE orders SET group_shipment_id=NULL, updated_at=? WHERE id=?",
               (_now(), order_id))
    remaining = cx.execute("SELECT id FROM orders WHERE group_shipment_id=?", (sid,)).fetchall()
    if len(remaining) < 2:
        for (rid,) in remaining:
            cx.execute("UPDATE orders SET group_shipment_id=NULL, updated_at=? WHERE id=?",
                       (_now(), rid))
        cx.execute("UPDATE combined_shipments SET status='cancelled', updated_at=? WHERE id=?",
                   (_now(), sid))
    cx.commit()


def set_order_tracking(cx, order_id, tracking_number, shipment_id=None):
    cur = cx.execute("UPDATE orders SET tracking_number=?, shipment_id=?, updated_at=? WHERE id=?",
                     (tracking_number, shipment_id, _now(), order_id))
    cx.commit()
    return cur.rowcount > 0


def set_order_shipping(cx, order_id, shipping_cents, total_cents):
    """Set an order's shipping charge and total together. Used when a combined
    shipment recomputes the one-parcel rate and splits it across members; the
    caller computes the new total (old total - old shipping + new shipping)."""
    cur = cx.execute(
        "UPDATE orders SET shipping_cents=?, total_cents=?, updated_at=? WHERE id=?",
        (int(shipping_cents or 0), int(total_cents or 0), _now(), order_id))
    cx.commit()
    return cur.rowcount > 0


def set_order_group(cx, order_id, group_shipment_id):
    """Link (or, with None, unlink) an order to a combined shipment. Returns True
    if the order existed."""
    cur = cx.execute(
        "UPDATE orders SET group_shipment_id=?, updated_at=? WHERE id=?",
        (group_shipment_id, _now(), order_id))
    cx.commit()
    return cur.rowcount > 0


def orders_in_group(cx, group_shipment_id):
    """Member orders of a combined shipment, oldest first."""
    cur = cx.execute("SELECT * FROM orders WHERE group_shipment_id=? ORDER BY id",
                     (group_shipment_id,))
    return [_row_to_dict(r) for r in cur.fetchall()]


def set_order_stripe_pi(cx, order_id, payment_intent):
    cur = cx.execute("UPDATE orders SET stripe_payment_intent=?, updated_at=? WHERE id=?",
                     (payment_intent, _now(), order_id))
    cx.commit()
    return cur.rowcount > 0


def find_order_by_external_ref(cx, external_ref):
    cur = cx.execute("SELECT * FROM orders WHERE external_ref=? ORDER BY id DESC LIMIT 1",
                     (str(external_ref),))
    return _row_to_dict(cur.fetchone())


def set_order_label(cx, order_id, label_url, tracking_number=None):
    if tracking_number:
        cur = cx.execute("UPDATE orders SET label_url=?, tracking_number=?, updated_at=? WHERE id=?",
                         (label_url, tracking_number, _now(), order_id))
    else:
        cur = cx.execute("UPDATE orders SET label_url=?, updated_at=? WHERE id=?",
                         (label_url, _now(), order_id))
    cx.commit()
    return cur.rowcount > 0


# --- Per-line partial fulfillment + backorder tracking (Phase 2) ---
# Each shipment of a line is one row in order_fulfillments. A line's backorder =
# its ordered qty minus the sum of its fulfillment events; cleared when 0.

def init_fulfillments_table(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS order_fulfillments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id INTEGER NOT NULL,
            line_index INTEGER NOT NULL,
            slug TEXT,
            qty INTEGER NOT NULL,
            fulfilled_at TEXT NOT NULL,
            note TEXT
        )
    """)
    cx.execute("CREATE INDEX IF NOT EXISTS idx_fulfill_order ON order_fulfillments(order_id)")
    cx.execute("CREATE INDEX IF NOT EXISTS idx_fulfill_slug ON order_fulfillments(slug)")
    cx.commit()


def fulfilled_qty(cx, order_id, line_index):
    row = cx.execute(
        "SELECT COALESCE(SUM(qty),0) FROM order_fulfillments WHERE order_id=? AND line_index=?",
        (order_id, int(line_index))).fetchone()
    return int(row[0] or 0)


def record_fulfillment(cx, order_id, line_index, slug, qty, at=None, note=None):
    """Record one partial-shipment event for a line. Clamps so cumulative
    fulfilled never exceeds the line's ordered qty. Returns the qty actually
    recorded after clamping (0 if the line is already fully fulfilled)."""
    qty = int(qty or 0)
    if qty <= 0:
        return 0
    order = get_order(cx, order_id)
    if not order:
        raise ValueError(f"order #{order_id} not found")
    li = int(line_index)
    items = order.get("items") or []
    ordered = int(items[li].get("qty") or 0) if 0 <= li < len(items) else 0
    qty = min(qty, max(0, ordered - fulfilled_qty(cx, order_id, li)))
    if qty <= 0:
        return 0
    if not slug and 0 <= li < len(items):
        slug = items[li].get("slug") or ""
    cx.execute(
        "INSERT INTO order_fulfillments (order_id, line_index, slug, qty, fulfilled_at, note) "
        "VALUES (?,?,?,?,?,?)",
        (order_id, li, slug or "", qty, at or _now(), note))
    cx.commit()
    return qty


def fulfillment_for_order(cx, order_id):
    """Per-line fulfillment state: [{index, slug, name, ordered, fulfilled,
    backordered, events:[{qty, fulfilled_at}]}]."""
    order = get_order(cx, order_id)
    if not order:
        return []
    by_line = {}
    for e in cx.execute(
            "SELECT line_index, qty, fulfilled_at FROM order_fulfillments "
            "WHERE order_id=? ORDER BY id", (order_id,)).fetchall():
        by_line.setdefault(int(e[0]), []).append({"qty": int(e[1]), "fulfilled_at": e[2]})
    out = []
    for i, it in enumerate(order.get("items") or []):
        evs = by_line.get(i, [])
        ordered = int(it.get("qty") or 0)
        filled = sum(x["qty"] for x in evs)
        out.append({"index": i, "slug": it.get("slug") or "", "name": it.get("name") or "",
                    "ordered": ordered, "fulfilled": filled,
                    "backordered": max(0, ordered - filled), "events": evs})
    return out


def order_backorder_units(cx, order_id):
    return sum(l["backordered"] for l in fulfillment_for_order(cx, order_id))


# Pre-payment invoices (proposed/confirmed) are not committed demand, so they are
# excluded from the reorder rollup along with done/cancelled.
_NOT_BACKORDERABLE = ("done", "cancelled", "proposed", "confirmed")


def backorder_rollup(cx):
    """Per-product backordered units across committed, open orders — the reorder
    worklist. Returns [{slug, name, units_backordered, order_count}] desc."""
    placeholders = ",".join("?" * len(_NOT_BACKORDERABLE))
    rows = cx.execute(
        f"SELECT id, items_json FROM orders WHERE status NOT IN ({placeholders})",
        _NOT_BACKORDERABLE).fetchall()
    filled = {}
    for e in cx.execute(
            "SELECT order_id, line_index, COALESCE(SUM(qty),0) FROM order_fulfillments "
            "GROUP BY order_id, line_index").fetchall():
        filled[(int(e[0]), int(e[1]))] = int(e[2])
    agg = {}
    for r in rows:
        oid = int(r[0])
        try:
            items = json.loads(r[1] or "[]")
        except Exception:
            items = []
        for i, it in enumerate(items):
            back = max(0, int(it.get("qty") or 0) - filled.get((oid, i), 0))
            if back <= 0:
                continue
            key = it.get("slug") or it.get("name") or f"line-{i}"
            a = agg.setdefault(key, {"slug": it.get("slug") or "",
                                     "name": it.get("name") or key,
                                     "units_backordered": 0, "orders": set()})
            a["units_backordered"] += back
            a["orders"].add(oid)
    out = [{"slug": a["slug"], "name": a["name"],
            "units_backordered": a["units_backordered"], "order_count": len(a["orders"])}
           for a in agg.values()]
    out.sort(key=lambda x: x["units_backordered"], reverse=True)
    return out


# --- Lifecycle actions + Home signal (register on import) ---
from dashboard.actions import action, LOW_WRITE
from dashboard.rbac import OWNER, OPS, VA
from dashboard.signals import signal as _signal, RED, AMBER, GREEN, GRAY


def _status_action(new_status, verb):
    def _exec(params, ctx):
        cx = (ctx or {}).get("cx") or (params or {}).get("cx")
        if cx is None:
            raise ValueError("no db connection")
        oid = int(params["order_id"])
        if not set_order_status(cx, oid, new_status):
            raise ValueError(f"order #{oid} not found")
        return {"order_id": oid, "status": new_status, "message": f"Order #{oid} {verb}."}
    return _exec


action(key="orders.mark_packed", module="orders", title="Mark packed",
       description="Mark an order packed.", risk_tier=LOW_WRITE,
       permission=(OWNER, OPS, VA))(_status_action("packed", "marked packed"))
action(key="orders.mark_shipped", module="orders", title="Mark shipped",
       description="Mark an order shipped.", risk_tier=LOW_WRITE,
       permission=(OWNER, OPS, VA))(_status_action("shipped", "marked shipped"))
action(key="orders.mark_done", module="orders", title="Mark done",
       description="Mark an order complete/delivered.", risk_tier=LOW_WRITE,
       permission=(OWNER, OPS, VA))(_status_action("done", "marked done"))
action(key="orders.cancel", module="orders", title="Cancel order",
       description="Cancel an order.", risk_tier=LOW_WRITE,
       permission=(OWNER, OPS, VA))(_status_action("cancelled", "cancelled"))


def set_order_payment(cx, order_id, *, method, amount_cents):
    """Record payment on an order: mark paid + capture method/amount/time, and
    drop it into the fulfillment board as 'new'. No money is moved (Phase 1)."""
    cur = cx.execute(
        "UPDATE orders SET status='new', pay_status='paid', pay_method=?, "
        "paid_cents=?, paid_at=?, updated_at=? WHERE id=?",
        (str(method or ""), int(amount_cents or 0), _now(), _now(), order_id))
    cx.commit()
    return cur.rowcount > 0


def mark_order_paid_keep_status(cx, order_id, *, method, amount_cents):
    """Record payment on an order WITHOUT moving it into the fulfillment board — for
    digital / no-ship charges (e.g. the $1 biofield trial) that are created 'done'
    and must stay 'done'. Sets the same payment fields as set_order_payment but
    leaves status untouched. paid_at is only stamped once (first mark wins)."""
    cur = cx.execute(
        "UPDATE orders SET pay_status='paid', pay_method=?, paid_cents=?, "
        "paid_at=COALESCE(paid_at,?), updated_at=? WHERE id=?",
        (str(method or ""), int(amount_cents or 0), _now(), _now(), order_id))
    cx.commit()
    return cur.rowcount > 0


def set_order_payment_claimed(cx, order_id, *, method):
    """Customer says they sent an off-platform payment (Zelle/Wise). Marks the
    invoice 'claimed' (pending OWNER confirmation) — does NOT mark it paid or
    move it into fulfillment; the OWNER confirms receipt via orders.record_payment."""
    cur = cx.execute(
        "UPDATE orders SET pay_status='claimed', pay_method=?, updated_at=? WHERE id=?",
        (str(method or ""), _now(), order_id))
    cx.commit()
    return cur.rowcount > 0


def mark_invoice_sent(cx, order_id):
    cur = cx.execute("UPDATE orders SET invoice_sent_at=?, updated_at=? WHERE id=?",
                     (_now(), _now(), order_id))
    cx.commit()
    return cur.rowcount > 0


def settle_order_points(cx, order, *, scope="rm", earn_pct=0.05):
    """On a PAID in-house order: redeem the points the customer applied and earn
    on full-price product spend. Idempotent per external_ref (guarded by
    points.has_entry), best-effort — never raises into the payment path. Email is
    lowercased to match the ledger (the funnel settler lowercases on write).

    Earn mirrors the funnel rule (full-price only: no discount AND no points used)
    but omits the affiliate-first-order suppression — in-house manual orders are
    not affiliate-acquired, so that gate is a no-op for them."""
    from dashboard import points as _points
    email = (order.get("email") or "").strip().lower()
    ref = (order.get("external_ref") or "").strip()
    if not email or not ref:
        return
    redeemed = int(order.get("points_redeemed_cents") or 0)
    discount = int(order.get("discount_cents") or 0)
    # Product spend nets out shipping + absorbed GET so points aren't earned on tax/shipping.
    product_cents = (int(order.get("total_cents") or 0)
                     - int(order.get("shipping_cents") or 0)
                     - int(order.get("get_cents") or 0))
    try:
        _points.init_points_table(cx)
        if redeemed > 0 and not _points.has_entry(cx, order_ref=ref, reason="redeem", scope=scope):
            try:
                _points.redeem(cx, email, value_cents=redeemed, order_ref=ref, scope=scope)
            except ValueError:
                pass  # balance already spent elsewhere; don't block the payment
        if discount == 0 and redeemed == 0 and product_cents > 0 \
                and not _points.has_entry(cx, order_ref=ref, reason="earn", scope=scope):
            _points.earn(cx, email, full_price_cents=product_cents, earn_pct=earn_pct,
                         order_ref=ref, scope=scope)
    except Exception as e:  # pragma: no cover - never crash the payment path
        print(f"[orders] points settle skipped for {ref}: {e!r}", flush=True)


def _invoice_paylink_enabled():
    return os.environ.get("INVOICE_PAYLINK_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")


# Statuses for which a customer invoice may be emailed: the pre-fulfillment
# invoice stages plus 'new' (covers both the unpaid Cart lane and the Paid lane,
# where a paid order sends as a receipt copy). Fulfilled/closed orders are excluded.
_SENDABLE_INVOICE = _PRE_FULFILL + ("new",)


def _send_invoice_exec(params, ctx):
    """OWNER: email the customer a tokenized invoice link. For an unpaid invoice
    it's a review-and-pay link; for a paid order it's a paid-invoice receipt.
    Flag-gated (INVOICE_PAYLINK_ENABLED) — real customer email."""
    cx = (ctx or {}).get("cx") or (params or {}).get("cx")
    if cx is None:
        raise ValueError("no db connection")
    if not _invoice_paylink_enabled():
        raise ValueError("invoice pay-link is disabled (INVOICE_PAYLINK_ENABLED off)")
    oid = int(params["order_id"])
    order = get_order(cx, oid)
    if not order:
        raise ValueError(f"order #{oid} not found")
    if order.get("status") not in _SENDABLE_INVOICE:
        raise ValueError(f"order #{oid} is '{order.get('status')}' — an invoice can only be "
                         "sent before it ships")
    email = (order.get("email") or "").strip()
    if not email:
        raise ValueError(f"order #{oid} has no customer email")
    from dashboard.practitioner_portal import create_order_invoice_token
    from dashboard import inbox as _inbox
    base = os.environ.get("PUBLIC_BASE_URL", "https://illtowell.com").rstrip("/")
    tok = create_order_invoice_token(oid)
    link = f"{base}/invoice/{tok}"
    name = order.get("name") or "there"
    ref = order.get("external_ref") or f"#{oid}"
    total = f"${(int(order.get('total_cents') or 0))/100:,.2f}"
    # Paid orders get a receipt; everything else (incl. claimed) gets the
    # review-and-pay link. Only the noun, lead sentence, and CTA button differ —
    # the greeting/signoff/link wrapper are shared so they can't drift.
    paid = (order.get("pay_status") == "paid")
    if paid:
        subject = f"Your Remedy Match receipt {ref}"
        lead_plain = (f"Here's your paid invoice {ref} ({total}) — paid in full, mahalo. "
                      f"View or download it here:")
        lead_html = (f"Here's your paid invoice <b>{ref}</b> (<b>{total}</b>) — paid in full, "
                     f"mahalo. You can view or download it here:")
        btn_color, btn_label = "#3a5a40", "View &amp; download your invoice"
    else:
        subject = f"Your Remedy Match invoice {ref}"
        lead_plain = (f"Your invoice {ref} is ready ({total}). View, adjust, ask "
                      f"questions, and pay here:")
        lead_html = (f"Your invoice <b>{ref}</b> is ready (<b>{total}</b>). You can review it, "
                     f"adjust quantities, ask questions, and pay securely here:")
        btn_color, btn_label = "#7c5cbf", "View &amp; pay your invoice"
    plain = f"Hi {name},\n\n{lead_plain}\n{link}\n\nMahalo,\nDr. Glen Swartwout"
    html = (f"<p>Hi {name},</p><p>{lead_html}</p>"
            f"<p><a href='{link}' style='background:{btn_color};color:#fff;padding:10px 18px;"
            f"border-radius:6px;text-decoration:none'>{btn_label}</a></p>"
            f"<p>Mahalo,<br>Dr. Glen Swartwout</p>")
    try:
        _inbox.send_email(email, subject, plain,
                          from_name="Dr. Glen Swartwout", html=html)
    except Exception as e:
        raise ValueError(f"could not send invoice email: {e}")
    mark_invoice_sent(cx, oid)
    return {"order_id": oid, "link": link,
            "message": f"{'Receipt' if paid else 'Invoice'} {ref} emailed to {email}."}


def _confirm_exec(params, ctx):
    cx = (ctx or {}).get("cx") or (params or {}).get("cx")
    if cx is None:
        raise ValueError("no db connection")
    oid = int(params["order_id"])
    order = get_order(cx, oid)
    if not order:
        raise ValueError(f"order #{oid} not found")
    if order.get("status") != "proposed":
        raise ValueError(f"order #{oid} is '{order.get('status')}', not a proposed invoice")
    set_order_status(cx, oid, "confirmed")
    return {"order_id": oid, "status": "confirmed",
            "message": f"Proposed invoice #{oid} confirmed — awaiting payment."}


def _record_payment_exec(params, ctx):
    cx = (ctx or {}).get("cx") or (params or {}).get("cx")
    if cx is None:
        raise ValueError("no db connection")
    oid = int(params["order_id"])
    order = get_order(cx, oid)
    if not order:
        raise ValueError(f"order #{oid} not found")
    # Payment can be recorded on a pre-fulfillment invoice (proposed/confirmed) OR on
    # an unpaid Cart order (status 'new', e.g. a portal-reorder the customer pays in
    # person by check/cash). An already-paid order is rejected (no double-record).
    status = order.get("status")
    if status not in _PRE_FULFILL and status != "new":
        raise ValueError(f"order #{oid} is '{status}'; payment is recorded before it ships")
    if order.get("pay_status") == "paid":
        raise ValueError(f"order #{oid} is already marked paid")
    method = str(params.get("method", "")).strip()
    amount_cents = int(params.get("amount_cents") or order.get("total_cents") or 0)
    set_order_payment(cx, oid, method=method, amount_cents=amount_cents)
    # Settle points now that it's paid: redeem applied points + earn (idempotent).
    settle_order_points(cx, get_order(cx, oid))
    _o = get_order(cx, oid)
    if _o and (_o.get("source") or "") == "dispensary":
        try:
            from dashboard.dispensary_rewards import settle_dispensary_l2
            settle_dispensary_l2(cx, _o, _o.get("external_ref"))
        except Exception as _de:
            print(f"[dispensary-l2] altpay settle skipped: {_de!r}", flush=True)
    return {"order_id": oid, "status": "new", "pay_status": "paid",
            "pay_method": method, "paid_cents": amount_cents,
            "message": f"Payment recorded for order #{oid}"
                       + (f" via {method}" if method else "") + " — now in fulfillment."}


action(key="orders.confirm", module="orders", title="Confirm proposed invoice",
       description="Mark a proposed invoice confirmed (customer agreed).",
       risk_tier=LOW_WRITE, permission=(OWNER,))(_confirm_exec)
action(key="orders.record_payment", module="orders", title="Record payment",
       description="Record payment on a proposed/confirmed invoice or an unpaid Cart "
                   "order (e.g. in-person check/cash) and move it into fulfillment.",
       risk_tier=LOW_WRITE, permission=(OWNER,))(_record_payment_exec)
action(key="orders.send_invoice", module="orders", title="Send invoice to customer",
       description="Email the customer a pay-link for a proposed/confirmed invoice.",
       risk_tier=LOW_WRITE, permission=(OWNER, OPS))(_send_invoice_exec)


def _set_tracking_exec(params, ctx):
    cx = (ctx or {}).get("cx") or (params or {}).get("cx")
    if cx is None:
        raise ValueError("no db connection")
    oid = int(params["order_id"])
    tn = str(params.get("tracking_number", "")).strip()
    if not set_order_tracking(cx, oid, tn):
        raise ValueError(f"order #{oid} not found")
    set_order_status(cx, oid, "shipped")
    return {"order_id": oid, "tracking_number": tn, "status": "shipped",
            "message": f"Order #{oid} shipped" + (f" (tracking {tn})." if tn else ".")}


action(key="orders.set_tracking", module="orders", title="Set tracking + ship",
       description="Record a tracking number and mark the order shipped.",
       risk_tier=LOW_WRITE, permission=(OWNER, OPS, VA))(_set_tracking_exec)


def _create_label_exec(params, ctx):
    from dashboard import easypost as EP
    cx = (ctx or {}).get("cx") or (params or {}).get("cx")
    if cx is None:
        raise ValueError("no db connection")
    oid = int(params["order_id"])
    order = get_order(cx, oid)
    if not order:
        raise ValueError(f"order #{oid} not found")
    if not EP.is_configured():
        return {"order_id": oid, "handoff": EP.CLICKNSHIP_URL,
                "message": "No label API configured. Buy the label on USPS Click-N-Ship, "
                           "then use Ship + tracking to record the number."}
    from_addr = (ctx or {}).get("from_address") or {}
    out = EP.buy_label(order, from_addr)
    set_order_label(cx, oid, out.get("label_url", ""), out.get("tracking_number"))
    return {"order_id": oid, "tracking_number": out.get("tracking_number", ""),
            "label_url": out.get("label_url", ""),
            "message": f"Label bought for order #{oid} (tracking {out.get('tracking_number','')})."}


action(key="orders.create_label", module="orders", title="Create shipping label",
       description="Buy a USPS label (EasyPost) or hand off to Click-N-Ship.",
       risk_tier=LOW_WRITE, permission=(OWNER, OPS, VA))(_create_label_exec)


def _fulfill_lines_exec(params, ctx):
    """Record a shipment: qty sent per line. Shortfall stays backordered; when
    every line is cleared the order is marked done, else it stays shipped."""
    cx = (ctx or {}).get("cx") or (params or {}).get("cx")
    if cx is None:
        raise ValueError("no db connection")
    oid = int(params["order_id"])
    order = get_order(cx, oid)
    if not order:
        raise ValueError(f"order #{oid} not found")
    if order.get("status") not in ("new", "packed", "shipped"):
        raise ValueError(
            f"order #{oid} is '{order.get('status')}' — only paid orders in "
            "fulfillment (new/packed/shipped) can be fulfilled")
    note = params.get("note")
    recorded = []
    for ln in (params.get("lines") or []):
        try:
            idx = int(ln.get("index"))
            q = int(ln.get("qty") or 0)
        except (TypeError, ValueError):
            continue
        got = record_fulfillment(cx, oid, idx, ln.get("slug"), q, note=note)
        if got > 0:
            recorded.append({"index": idx, "qty": got})
    if not recorded:
        raise ValueError("no quantities to record")
    back = order_backorder_units(cx, oid)
    new_status = "done" if back <= 0 else "shipped"
    set_order_status(cx, oid, new_status)
    sent = sum(r["qty"] for r in recorded)
    return {"order_id": oid, "fulfilled": recorded, "backorder_units": back,
            "status": new_status,
            "message": f"Order #{oid}: recorded {sent} unit(s) shipped"
                       + (f", {back} still backordered." if back else " — fully fulfilled.")}


action(key="orders.fulfill_lines", module="orders", title="Record shipment (fulfill lines)",
       description="Record qty shipped per line; backorder = ordered minus fulfilled.",
       risk_tier=LOW_WRITE, permission=(OWNER, OPS, VA))(_fulfill_lines_exec)


def _gmail_send_tracking(to, subject, html):
    """Best-effort customer email via the inbox Gmail sender. Production-only;
    returns False (not an error) when Gmail is unavailable."""
    try:
        from dashboard.inbox import send_email
        send_email(to, subject, html, from_name="Dr. Glen Swartwout")
        return True
    except Exception as e:
        print(f"[orders] tracking email skipped: {e!r}", flush=True)
        return False


def _send_tracking_exec(params, ctx):
    from dashboard import tracking as T
    cx = (ctx or {}).get("cx") or (params or {}).get("cx")
    if cx is None:
        raise ValueError("no db connection")
    oid = int(params["order_id"])
    order = get_order(cx, oid)
    if not order:
        raise ValueError(f"order #{oid} not found")
    tn = order.get("tracking_number") or ""
    if not tn:
        raise ValueError("order has no tracking number yet (create a label or set tracking first)")
    email = order.get("email") or ""
    # Re-send guard: if this tracking number was already notified, do not email
    # the customer again (avoids a duplicate email on a second button press).
    try:
        T.init_tracking_schema(cx)
    except Exception:
        pass
    existed = cx.execute("SELECT id FROM shipments WHERE tracking_number=?", (tn,)).fetchone()
    if existed:
        set_order_tracking(cx, oid, tn, shipment_id=existed[0])
        return {"order_id": oid, "tracking_number": tn, "emailed": False,
                "message": f"Tracking {tn} was already sent for order #{oid}."}
    em = T.build_tracking_email(tn, order.get("name"))
    sent = _gmail_send_tracking(email, em.get("subject", "tracking number"), em.get("html", "")) if email else False
    try:
        T.record_shipment(cx, tracking_number=tn, recipient_name=order.get("name"),
                          resolved_email=email, status=("sent" if sent else "drafted"),
                          order_uuid=order.get("external_ref"))
        sh = cx.execute("SELECT id FROM shipments WHERE tracking_number=?", (tn,)).fetchone()
        if sh:
            set_order_tracking(cx, oid, tn, shipment_id=sh[0])
    except Exception as e:
        print(f"[orders] shipment record: {e!r}", flush=True)
    verb = "sent to" if sent else "drafted for"
    return {"order_id": oid, "tracking_number": tn, "emailed": sent,
            "message": f"Tracking {tn} {verb} {email or 'customer'}."}


action(key="orders.send_tracking", module="orders", title="Send tracking email",
       description="Email the customer their tracking number and record the shipment.",
       risk_tier=LOW_WRITE, permission=(OWNER, OPS, VA))(_send_tracking_exec)


def open_fulfillment_orders(cx):
    """Open (new/packed) orders as [{created_at, source}] — the single source of
    the fulfillment-queue read. Shared by orders_signal and b2b_signal so the
    home board reads the queue once per load (request-cached during aggregation)."""
    from dashboard.signals import request_cached

    def _read():
        cur = cx.execute(
            "SELECT created_at, source FROM orders WHERE status IN ('new','packed')")
        return [{"created_at": row[0], "source": row[1]} for row in cur.fetchall()]

    return request_cached("orders:open_fulfillment", _read)


@_signal("orders")
def orders_signal(cx, actor=None, now=None):
    try:
        rows = open_fulfillment_orders(cx)
    except Exception:
        return {"level": GRAY, "summary": "Not yet wired", "top_actions": [], "count": 0}
    n = len(rows)
    if n == 0:
        return {"level": GREEN, "summary": "No orders to fulfill", "top_actions": [], "count": 0}
    cutoff = (now or datetime.now(timezone.utc)) - timedelta(hours=24)
    aging = 0
    for r in rows:
        try:
            ts = datetime.fromisoformat(r["created_at"])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts < cutoff:
                aging += 1
        except Exception:
            pass
    level = RED if aging else AMBER
    summary = f"{n} to fulfill" + (f", {aging} aging over 24h" if aging else "")
    return {"level": level, "summary": summary,
            "top_actions": [{"label": "Open orders board", "href": "/console/orders"}],
            "count": n}


def effective_shipping_cents(pickup, computed_cents):
    """Shipping for an order: 0 when it's a pickup (no shipping), else the
    computed amount. Single source of the pickup-shipping rule."""
    if pickup:
        return 0
    try:
        return int(computed_cents or 0)
    except (TypeError, ValueError):
        return 0
