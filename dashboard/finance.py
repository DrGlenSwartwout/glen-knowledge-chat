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
