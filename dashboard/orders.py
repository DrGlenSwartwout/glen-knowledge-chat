"""Business-OS Orders & Fulfillment model. One orders table unifies every order
flow (funnel/QBO retail+wholesale, GrooveKart, dispensary, manual) into a single
lifecycle: new -> packed -> shipped -> done (+ cancelled). Functions take a
sqlite connection for testability. Lifecycle actions + the Home signal register
on import (see Task 2)."""
import json
from datetime import datetime, timezone, timedelta

ORDER_STATUSES = ("new", "packed", "shipped", "done", "cancelled")
_OPEN = ("new", "packed")  # unfulfilled


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
    cx.commit()


def upsert_order(cx, *, source, external_ref, email="", name="", phone="",
                 items=None, total_cents=0, address=None, channel="retail",
                 status="new"):
    """Idempotent on (source, external_ref). Inserts a new order, or updates the
    soft fields of an existing one WITHOUT regressing its lifecycle status.
    items and address are only overwritten when explicitly provided (not None)."""
    ref = str(external_ref or "").strip()
    if not ref:
        raise ValueError("external_ref required")
    row = cx.execute("SELECT id FROM orders WHERE source=? AND external_ref=?",
                     (source, ref)).fetchone()
    if row:
        # Only overwrite items_json / address_json when caller provides them.
        sets = ["email=?", "name=?", "phone=?", "total_cents=?", "channel=?", "updated_at=?"]
        vals = [email, name, phone, int(total_cents or 0), channel, _now()]
        if items is not None:
            sets.insert(3, "items_json=?")
            vals.insert(3, json.dumps(items))
        if address is not None:
            sets.append("address_json=?")
            vals.append(json.dumps(address))
        vals.append(row[0])
        cx.execute(f"UPDATE orders SET {', '.join(sets)} WHERE id=?", vals)
        cx.commit()
        return row[0]
    cur = cx.execute(
        "INSERT INTO orders (created_at, source, external_ref, channel, email, name, "
        "phone, items_json, total_cents, address_json, status) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (_now(), source, ref, channel, email, name, phone,
         json.dumps(items or []), int(total_cents or 0), json.dumps(address or {}), status))
    cx.commit()
    return cur.lastrowid


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
