"""Business-OS Money & Finance. Pure aging/summary/signal logic (unit-tested) +
cached QBO-backed reads (production-verified) + the Money home signal + the safe
finance actions. Finance writes are owner/ops only (Shaira/va excluded)."""
import os
import time
from datetime import datetime, timezone

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
        })
    out.sort(key=lambda x: -x["days_overdue"])
    return out


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
    """Cached (10 min): QBO open invoices as AR rows. Production-only (QBO)."""
    def _f():
        from dashboard import qbo_billing as qb
        rs = qb._query("SELECT * FROM Invoice WHERE Balance > '0' ORDER BY DueDate ASC")
        invs = (rs.get("QueryResponse") or {}).get("Invoice") or []
        return aging(invs)
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
    return money_signal_from(s, cash_floor=_cash_floor())


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
    return (f"Issue a {amt} refund against invoice {target}. This records a money-out "
            f"refund in QuickBooks (you still send the actual money for Zelle/Wise). Confirm?")


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
    receipt = qb.create_refund_receipt(customer_id, amount, description=description)
    _cache.clear()
    return {"refund_receipt_id": receipt.get("Id"), "customer_id": customer_id,
            "amount": amount, "invoice_id": invoice_id,
            "message": f"Refund of ${amount:.2f} recorded in QuickBooks "
                       f"(RefundReceipt {receipt.get('DocNumber', receipt.get('Id'))})."}


action(key="finance.refund_order", module="money", title="Refund order",
       description="Record a customer refund in QuickBooks (money-out RefundReceipt).",
       risk_tier=MONEY_SEND, permission=(OWNER, OPS, "va"),
       confirm_summary=_refund_confirm_summary)(_refund_order_exec)
