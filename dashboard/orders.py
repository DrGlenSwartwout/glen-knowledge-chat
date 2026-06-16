"""Business-OS Orders & Fulfillment model. One orders table unifies every order
flow (funnel/QBO retail+wholesale, GrooveKart, dispensary, manual) into a single
lifecycle: new -> packed -> shipped -> done (+ cancelled). Functions take a
sqlite connection for testability. Lifecycle actions + the Home signal register
on import (see Task 2)."""
import json
import os
from datetime import datetime, timezone, timedelta

ORDER_STATUSES = ("proposed", "confirmed", "paid",
                  "new", "packed", "shipped", "done", "cancelled")
_OPEN = ("new", "packed")  # unfulfilled
# Pre-fulfillment lead-in for in-house proposed invoices (before the kanban).
_PRE_FULFILL = ("proposed", "confirmed")


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
    ):
        try:
            cx.execute(ddl)
        except Exception:
            pass  # already present
    cx.commit()


def upsert_order(cx, *, source, external_ref, email="", name="", phone="",
                 items=None, total_cents=0, address=None, channel="retail",
                 status="new", get_cents=0, person_id=None,
                 discount_cents=0, points_redeemed_cents=0, shipping_cents=0):
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
                "shipping_cents=?", "updated_at=?"]
        vals = [email, name, phone, int(total_cents or 0), channel,
                int(get_cents or 0), int(discount_cents or 0),
                int(points_redeemed_cents or 0), int(shipping_cents or 0), _now()]
        if items is not None:
            sets.insert(3, "items_json=?")
            vals.insert(3, json.dumps(items))
        if address is not None:
            sets.append("address_json=?")
            vals.append(json.dumps(address))
        if person_id is not None:
            sets.append("person_id=?")
            vals.append(int(person_id))
        vals.append(row[0])
        cx.execute(f"UPDATE orders SET {', '.join(sets)} WHERE id=?", vals)
        cx.commit()
        return row[0]
    cur = cx.execute(
        "INSERT INTO orders (created_at, source, external_ref, channel, email, name, "
        "phone, items_json, total_cents, address_json, status, get_cents, person_id, "
        "discount_cents, points_redeemed_cents, shipping_cents) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (_now(), source, ref, channel, email, name, phone,
         json.dumps(items or []), int(total_cents or 0), json.dumps(address or {}),
         status, int(get_cents or 0),
         (int(person_id) if person_id is not None else None),
         int(discount_cents or 0), int(points_redeemed_cents or 0), int(shipping_cents or 0)))
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


def set_order_status(cx, order_id, status):
    if status not in ORDER_STATUSES:
        raise ValueError(f"unknown status: {status}")
    cur = cx.execute("UPDATE orders SET status=?, updated_at=? WHERE id=?",
                     (status, _now(), order_id))
    cx.commit()
    return cur.rowcount > 0


def set_order_tracking(cx, order_id, tracking_number, shipment_id=None):
    cur = cx.execute("UPDATE orders SET tracking_number=?, shipment_id=?, updated_at=? WHERE id=?",
                     (tracking_number, shipment_id, _now(), order_id))
    cx.commit()
    return cur.rowcount > 0


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


def _invoice_paylink_enabled():
    return os.environ.get("INVOICE_PAYLINK_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")


def _send_invoice_exec(params, ctx):
    """OWNER: email the customer a tokenized pay-link for a proposed/confirmed
    invoice. Flag-gated (INVOICE_PAYLINK_ENABLED) — real customer email."""
    cx = (ctx or {}).get("cx") or (params or {}).get("cx")
    if cx is None:
        raise ValueError("no db connection")
    if not _invoice_paylink_enabled():
        raise ValueError("invoice pay-link is disabled (INVOICE_PAYLINK_ENABLED off)")
    oid = int(params["order_id"])
    order = get_order(cx, oid)
    if not order:
        raise ValueError(f"order #{oid} not found")
    if order.get("status") not in _PRE_FULFILL:
        raise ValueError(f"order #{oid} is '{order.get('status')}' — only a proposed/confirmed "
                         "invoice can be sent")
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
    plain = (f"Hi {name},\n\nYour invoice {ref} is ready ({total}). View, adjust, ask "
             f"questions, and pay here:\n{link}\n\nMahalo,\nDr. Glen Swartwout")
    html = (f"<p>Hi {name},</p><p>Your invoice <b>{ref}</b> is ready "
            f"(<b>{total}</b>). You can review it, adjust quantities, ask questions, "
            f"and pay securely here:</p>"
            f"<p><a href='{link}' style='background:#7c5cbf;color:#fff;padding:10px 18px;"
            f"border-radius:6px;text-decoration:none'>View &amp; pay your invoice</a></p>"
            f"<p>Mahalo,<br>Dr. Glen Swartwout</p>")
    try:
        _inbox.send_email(email, f"Your Remedy Match invoice {ref}", plain,
                          from_name="Dr. Glen Swartwout", html=html)
    except Exception as e:
        raise ValueError(f"could not send invoice email: {e}")
    mark_invoice_sent(cx, oid)
    return {"order_id": oid, "link": link,
            "message": f"Invoice {ref} emailed to {email}."}


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
    if order.get("status") not in _PRE_FULFILL:
        raise ValueError(f"order #{oid} is '{order.get('status')}'; payment is recorded on a "
                         "proposed/confirmed invoice")
    method = str(params.get("method", "")).strip()
    amount_cents = int(params.get("amount_cents") or order.get("total_cents") or 0)
    set_order_payment(cx, oid, method=method, amount_cents=amount_cents)
    return {"order_id": oid, "status": "new", "pay_status": "paid",
            "pay_method": method, "paid_cents": amount_cents,
            "message": f"Payment recorded for order #{oid}"
                       + (f" via {method}" if method else "") + " — now in fulfillment."}


action(key="orders.confirm", module="orders", title="Confirm proposed invoice",
       description="Mark a proposed invoice confirmed (customer agreed).",
       risk_tier=LOW_WRITE, permission=(OWNER,))(_confirm_exec)
action(key="orders.record_payment", module="orders", title="Record payment",
       description="Record payment on a proposed/confirmed invoice and move it into fulfillment.",
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
