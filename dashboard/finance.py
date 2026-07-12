"""Business-OS Money & Finance. Pure aging/summary/signal logic (unit-tested) +
cached QBO-backed reads (production-verified) + the Money home signal + the safe
finance actions. Finance writes are owner/ops only (Shaira/va excluded)."""
import os
import time
from datetime import datetime, timezone, timedelta

from dashboard.signals import signal as _signal, RED, AMBER, GREEN, GRAY


def _parse_date(s):
    if not s:
        return None
    try:
        d = datetime.fromisoformat(str(s)[:10])
        return d.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _days_overdue(due_date, now):
    d = _parse_date(due_date)
    if d is None:
        return -9999
    return int((now - d).total_seconds() // 86400)


def aging(invoices, now=None):
    """Pure: QBO Invoice dicts -> AR rows with days_overdue, zero-balance dropped,
    most-overdue first."""
    now = now or datetime.now(timezone.utc)
    out = []
    for inv in invoices or []:
        try:
            bal = float(inv.get("Balance") or 0)
        except (TypeError, ValueError):
            bal = 0.0
        if bal <= 0:
            continue
        due = inv.get("DueDate") or ""
        out.append({
            "id": inv.get("Id"), "doc": inv.get("DocNumber"),
            "customer": (inv.get("CustomerRef") or {}).get("name", ""),
            "email": (inv.get("BillEmail") or {}).get("Address", ""),
            "total": float(inv.get("TotalAmt") or 0), "balance": bal,
            "due_date": due, "days_overdue": _days_overdue(due, now),
            "source": "qbo",
        })
    out.sort(key=lambda x: -x["days_overdue"])
    return out


def _inhouse_net_days():
    """Payment terms for in-house invoices, so aging matches QBO DueDate semantics
    (an invoice is 'overdue' only after its terms, not the moment it's created)."""
    try:
        return int(os.environ.get("FINANCE_INHOUSE_NET_DAYS", "14"))
    except (TypeError, ValueError):
        return 14


def inhouse_aging(orders, qbo_ids=None, now=None):
    """Pure: unpaid in-house order rows -> AR rows (source='inhouse'). Drops paid,
    cancelled, and zero-balance rows, and any order already represented on the QBO
    list (its external_ref matches an open QBO invoice Id). Due date = created +
    net terms. Most-overdue first."""
    now = now or datetime.now(timezone.utc)
    qbo_ids = {str(i) for i in (qbo_ids or set())}
    net = _inhouse_net_days()
    out = []
    for o in orders or []:
        if (o.get("pay_status") or "unpaid") == "paid":
            continue
        if (o.get("status") or "") == "cancelled":
            continue
        total = int(o.get("total_cents") or 0)
        paid = int(o.get("paid_cents") or 0)
        bal_cents = total - paid
        if bal_cents <= 0:
            continue
        ref = str(o.get("external_ref") or "").strip()
        if ref and ref in qbo_ids:
            continue  # already counted on the QBO list
        created = _parse_date(str(o.get("created_at") or "")[:10])
        due = (created + timedelta(days=net)).date().isoformat() if created else ""
        out.append({
            "id": o.get("id"), "order_id": o.get("id"),
            "doc": ref or ("#" + str(o.get("id"))),
            "customer": o.get("name") or "",
            "email": o.get("email") or "",
            "total": round(total / 100.0, 2),
            "balance": round(bal_cents / 100.0, 2),
            "due_date": due, "days_overdue": _days_overdue(due, now),
            "source": "inhouse", "pay_method": o.get("pay_method") or "",
        })
    out.sort(key=lambda x: -x["days_overdue"])
    return out


def _log_db_path():
    from pathlib import Path
    return Path(os.environ.get("DATA_DIR", str(Path(__file__).resolve().parent.parent))) / "chat_log.db"


def _inhouse_open_rows(qbo_ids):
    """Best-effort read of unpaid in-house order invoices from the local orders
    table, shaped as AR rows. Never raises (returns [] on any DB error) so the
    QBO receivables never go dark because of a local-read hiccup."""
    import sqlite3
    try:
        cx = sqlite3.connect(str(_log_db_path()))
    except Exception:
        return []
    try:
        cx.row_factory = sqlite3.Row
        cur = cx.execute(
            "SELECT id, created_at, external_ref, email, name, total_cents, "
            "paid_cents, pay_status, pay_method, status FROM orders "
            "WHERE COALESCE(pay_status,'unpaid')!='paid' "
            "AND COALESCE(status,'')!='cancelled'")
        orders = [dict(r) for r in cur.fetchall()]
    except Exception:
        return []
    finally:
        cx.close()
    return inhouse_aging(orders, qbo_ids=qbo_ids)


def summarize(aged, cash_total=0.0):
    overdue = [a for a in aged if a["days_overdue"] > 0]
    return {
        "open_count": len(aged),
        "open_total": round(sum(a["balance"] for a in aged), 2),
        "overdue_count": len(overdue),
        "overdue_total": round(sum(a["balance"] for a in overdue), 2),
        "cash_total": round(cash_total or 0.0, 2),
    }


def _cash_floor():
    try:
        return float(os.environ.get("FINANCE_CASH_FLOOR", "0"))
    except (TypeError, ValueError):
        return 0.0


def money_signal_from(summary, cash_floor=0.0):
    oc = summary.get("overdue_count", 0)
    opc = summary.get("open_count", 0)
    cash = summary.get("cash_total", 0)
    low_cash = cash_floor > 0 and cash < cash_floor
    if oc > 0 or low_cash:
        bits = []
        if oc:
            bits.append(f"{oc} overdue (${summary.get('overdue_total', 0):.0f})")
        if low_cash:
            bits.append(f"cash ${cash:.0f} low")
        return {"level": RED, "summary": ", ".join(bits),
                "top_actions": [{"label": "Open finance", "href": "/console/finance"}],
                "count": oc}
    if opc > 0:
        return {"level": AMBER, "summary": f"{opc} open (${summary.get('open_total', 0):.0f})",
                "top_actions": [{"label": "Open finance", "href": "/console/finance"}],
                "count": opc}
    return {"level": GREEN, "summary": "AR clear", "top_actions": [], "count": 0}


# --- QBO-backed reads (cached) + Money signal + void action ---
from dashboard.actions import action, LOW_WRITE, IRREVERSIBLE, MONEY_SEND
from dashboard.rbac import OWNER, OPS

_cache = {}


def _cached(key, ttl, fn):
    now = time.time()
    hit = _cache.get(key)
    if hit and now - hit[0] < ttl:
        return hit[1]
    val = fn()
    _cache[key] = (now, val)
    return val


def open_invoices():
    """Cached (10 min): open receivables from every source we can read —
    QBO open invoices (source='qbo', production-only) PLUS unpaid in-house order
    invoices (source='inhouse'), deduped against QBO by the order's external_ref.
    Most-overdue first. The in-house read is best-effort and never blocks the QBO
    rows. (PayPal/Practice Better/Authorize.net/Wise have no open-invoice API wired
    — they surface in the weekly reconciler, not here.)"""
    def _f():
        from dashboard import qbo_billing as qb
        rs = qb._query("SELECT * FROM Invoice WHERE Balance > '0' ORDER BY DueDate ASC")
        invs = (rs.get("QueryResponse") or {}).get("Invoice") or []
        qbo_rows = aging(invs)
        try:
            inh = _inhouse_open_rows({r["id"] for r in qbo_rows if r.get("id") is not None})
        except Exception:
            inh = []
        rows = qbo_rows + inh
        rows.sort(key=lambda x: -x["days_overdue"])
        return rows
    return _cached("open_invoices", 600, _f)


def finance_summary():
    """Cached: AR summary + cash position (sum of QBO bank balances)."""
    def _f():
        from dashboard import money as M
        aged = open_invoices()
        try:
            cash = sum(a.get("balance", 0) for a in (M.qb_banks().get("accounts") or []))
        except Exception:
            cash = 0.0
        return summarize(aged, cash)
    return _cached("finance_summary", 600, _f)


@_signal("money")
def money_signal(cx, actor=None):
    try:
        s = finance_summary()
    except Exception:
        return {"level": GRAY, "summary": "Not yet wired", "top_actions": [], "count": 0}
    res = money_signal_from(s, cash_floor=_cash_floor())
    # Recent Stripe card-payment failures take over the cell (revenue at risk).
    try:
        from dashboard import stripe_alerts as _sa
        n = _sa.recent_failure_count(cx, minutes=30)
    except Exception:
        n = 0
    if n:
        return {**res, "level": RED,
                "summary": f"⚠ Card payments failing ({n} in last 30m) — " + res.get("summary", ""),
                "top_actions": ([{"label": "Investigate card failures", "href": "/console/orders"}]
                                + (res.get("top_actions") or []))[:3]}
    return res


def _void_invoice_exec(params, ctx):
    from dashboard import qbo_billing as qb
    iid = str(params["invoice_id"])
    inv = qb.get_invoice(iid)
    if not inv:
        raise ValueError(f"invoice {iid} not found")
    qb.void_invoice(iid, inv.get("SyncToken"))
    _cache.clear()  # AR changed
    return {"invoice_id": iid, "doc": inv.get("DocNumber"),
            "message": f"Invoice {inv.get('DocNumber', iid)} voided."}


action(key="finance.void_invoice", module="money", title="Void invoice",
       description="Void an unpaid QBO invoice (zeroes it).", risk_tier=IRREVERSIBLE,
       permission=(OWNER, OPS))(_void_invoice_exec)


def _refund_confirm_summary(params):
    try:
        amt = f"${float(params.get('amount', 0)):.2f}"
    except (TypeError, ValueError):
        amt = "$?"
    target = params.get("invoice_id") or f"order #{params.get('order_id', '?')}"
    if (params.get("stripe_payment_intent") or "").strip():
        how = "This refunds the card via Stripe and records it in QuickBooks."
    else:
        how = ("This records a money-out refund in QuickBooks; if the order has a Stripe card "
               "payment on file it also refunds the card, otherwise you send the money (Zelle/Wise).")
    return f"Issue a {amt} refund against invoice {target}. {how} Confirm?"


def _refund_order_exec(params, ctx):
    from dashboard import qbo_billing as qb
    cx = (ctx or {}).get("cx") or (params or {}).get("cx")
    try:
        amount = float(params["amount"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("a positive amount (dollars) is required")
    if amount <= 0:
        raise ValueError("a positive amount is required")
    invoice_id = params.get("invoice_id")
    if not invoice_id and params.get("order_id"):
        from dashboard.orders import get_order
        order = get_order(cx, int(params["order_id"]))
        if not order:
            raise ValueError(f"order #{params['order_id']} not found")
        invoice_id = order.get("external_ref")
    if not invoice_id:
        raise ValueError("invoice_id or order_id required")
    inv = qb.get_invoice(str(invoice_id))
    if not inv:
        raise ValueError(f"invoice {invoice_id} not found")
    customer_id = (inv.get("CustomerRef") or {}).get("value")
    if not customer_id:
        raise ValueError("invoice has no customer")
    description = params.get("reason") or f"Refund for invoice {invoice_id}"

    # Resolve a Stripe PaymentIntent: explicit param, else from the captured order.
    pi = (params.get("stripe_payment_intent") or "").strip()
    if not pi and cx is not None:
        try:
            from dashboard.orders import find_order_by_external_ref
            o = find_order_by_external_ref(cx, invoice_id)
            pi = (o or {}).get("stripe_payment_intent") or ""
        except Exception:
            pi = ""

    card_msg = ""
    if pi:
        # Card refund FIRST: only book the QBO refund if real money actually went back.
        from dashboard import stripe_pay
        sr = stripe_pay.refund(pi, int(round(amount * 100)))
        card_msg = f" to the card (Stripe {sr.get('id')})"

    receipt = qb.create_refund_receipt(customer_id, amount, description=description)
    _cache.clear()
    return {"refund_receipt_id": receipt.get("Id"), "customer_id": customer_id,
            "amount": amount, "invoice_id": invoice_id, "stripe_refund": bool(pi),
            "message": f"Refund of ${amount:.2f}{card_msg} recorded in QuickBooks "
                       f"(RefundReceipt {receipt.get('DocNumber', receipt.get('Id'))})."}


action(key="finance.refund_order", module="money", title="Refund order",
       description="Record a customer refund in QuickBooks (money-out RefundReceipt).",
       risk_tier=MONEY_SEND, permission=(OWNER, OPS, "va"),
       confirm_summary=_refund_confirm_summary)(_refund_order_exec)


def _ship_credit_flag_on():
    return os.environ.get("SHIP_CREDIT_AUTOAPPLY_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")


def _refund_ship_credit_summary(params):
    try:
        amt = f"${float(params.get('_amount', 0)):.2f}"
    except (TypeError, ValueError):
        amt = "the"
    who = params.get("_who") or f"order #{params.get('order_id', '?')}"
    return (f"Refund {amt} shipping credit to {who} instead of applying it to their "
            f"next order. Card payments refund via Stripe; otherwise it books a "
            f"QuickBooks money-out receipt for you to send. Confirm?")


def _refund_ship_credit_exec(params, ctx):
    """One-click refund of an already-paid order's outstanding SHIPPING credit (from a
    combined-shipment recalc) to the customer, instead of auto-applying it to their
    next order. Refunds only the portion still outstanding in their ship_credit
    balance (never more than the order generated), then removes it from the ledger so
    it can't also auto-apply. Idempotent — a second click is a no-op (already-refunded
    guard, the gap the plain finance.refund_order lacks)."""
    from dashboard.orders import get_order
    from dashboard import ship_credit as _sc
    cx = (ctx or {}).get("cx") or (params or {}).get("cx")
    if cx is None:
        raise ValueError("no db connection")
    if not _ship_credit_flag_on():
        raise ValueError("shipping-credit feature is off")
    order = get_order(cx, int(params["order_id"]))
    if not order:
        raise ValueError(f"order #{params['order_id']} not found")
    email = (order.get("email") or "").strip().lower()
    source_ref = (order.get("external_ref") or "").strip()
    generated = int(order.get("overpay_credit_cents") or 0)
    if not email or not source_ref or generated <= 0:
        raise ValueError("this order has no shipping credit to refund")
    if _sc.already_refunded(cx, source_ref=source_ref):
        return {"order_id": order.get("id"), "already_refunded": True,
                "message": "This order's shipping credit was already refunded."}
    # Refund only what's still outstanding (not already spent on a later order),
    # bounded by what this order generated.
    outstanding = min(generated, _sc.balance(cx, email))
    if outstanding <= 0:
        raise ValueError("this shipping credit has already been used on another order")
    # Remove from the ledger FIRST (idempotent marker) so a mid-refund retry can't
    # re-refund; if the money-out fails the exception propagates and the marker stays,
    # which is the safe side (no double payout).
    _sc.mark_refunded(cx, email, outstanding, source_ref=source_ref)
    res = _refund_order_exec(
        {"amount": outstanding / 100.0, "order_id": order.get("id"),
         "reason": f"Shipping overpayment credit refund (order #{order.get('id')})",
         "cx": cx}, ctx)
    res["ship_credit_refunded_cents"] = outstanding
    res["message"] = (f"Refunded ${outstanding/100:.2f} shipping credit. " + res.get("message", ""))
    return res


action(key="finance.refund_ship_credit", module="money", title="Refund shipping credit",
       description="Refund an order's outstanding shipping-overpayment credit instead of auto-applying it.",
       risk_tier=MONEY_SEND, permission=(OWNER, OPS, "va"),
       confirm_summary=_refund_ship_credit_summary)(_refund_ship_credit_exec)


def _record_payment_confirm_summary(params):
    amt = float(params.get("amount") or 0)
    m = params.get("method")
    return "Record $%.2f against invoice %s%s?" % (amt, params.get("invoice_id"), (" via " + str(m)) if m else "")


def _record_payment_exec(params, ctx):
    invoice_id = params.get("invoice_id")
    try:
        amount = float(params.get("amount") or 0)
    except (TypeError, ValueError):
        amount = 0.0
    if not invoice_id or amount <= 0:
        return {"ok": False, "error": "invoice_id and a positive amount are required"}
    from dashboard import qbo_billing as qb
    inv = qb.get_invoice(invoice_id)
    if not inv:
        return {"ok": False, "error": "invoice %s not found" % invoice_id}
    customer_id = (inv.get("CustomerRef") or {}).get("value")
    qb.record_payment(customer_id, round(amount * 100), invoice_id, method=params.get("method"))
    _cache.clear()
    return {"ok": True, "invoice_id": invoice_id, "amount": amount}


action(key="finance.record_payment", module="money", title="Record payment",
       description="Record a customer payment against a QuickBooks invoice (partial/split supported).",
       risk_tier=MONEY_SEND, permission=(OWNER, OPS),
       confirm_summary=_record_payment_confirm_summary)(_record_payment_exec)
