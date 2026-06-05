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
from dashboard.actions import action, LOW_WRITE, IRREVERSIBLE
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
