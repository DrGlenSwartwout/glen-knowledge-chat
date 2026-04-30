"""Studio.com sales monitor — aggregates sale notifications from Gmail.

Glen's Studio app (studio.com/dashboard?section=earnings) launched on
2026-04-28 as part of their Flagship Creator Program. Sales monitoring is
done via Gmail (Studio doesn't have a public API for creator earnings;
their internal API at api.studio.com requires session-cookie auth that
we couldn't automate).

Strategy:
- Watch for any incoming email from a `*@studio.com` sender
- Separate "program coordination" senders (jen@, craig@ — the human
  account managers from Studio's Flagship team) from transactional
  senders (notifications@, sales@, payments@ — the automated billing
  ones we'd expect when sales come in)
- Until the first transactional email arrives, the widget shows
  "Awaiting first sale" + the count of program emails as a heartbeat
- Once transactional emails start arriving, we'll learn the format
  and add amount/customer extraction

Public API used by the dashboard route:
    summary(days_back=30) → dashboard widget payload
"""

from __future__ import annotations

import re
from datetime import datetime, timezone, timedelta
from typing import Optional

from . import inbox as _inbox

# Known program-coordination senders we DON'T treat as sales notifications.
# Add to this set as we identify more humans-from-Studio addresses.
PROGRAM_SENDERS = {
    "jen@studio.com",
    "craig@studio.com",
}

# Cents-amount regex — matches $X.XX or $X (e.g., $19.99, $50, $1,200)
_AMOUNT_RE = re.compile(r"\$\s?([0-9][0-9,]*(?:\.[0-9]{2})?)")


def _normalize_sender_email(sender: str) -> str:
    if not sender:
        return ""
    m = re.search(r"<([^>]+)>", sender)
    return (m.group(1) if m else sender).strip().lower()


def _list_studio_messages(days_back: int = 30) -> list:
    """Pull metadata for all incoming Studio.com emails in the window."""
    svc = _inbox._get_gmail_service()
    res = svc.users().messages().list(
        userId="me",
        q=f"from:studio.com newer_than:{days_back}d",
        maxResults=100,
    ).execute()
    out = []
    for m in (res.get("messages") or []):
        full = svc.users().messages().get(
            userId="me", id=m["id"], format="metadata",
            metadataHeaders=["From", "Subject", "Date"],
        ).execute()
        headers = full.get("payload", {}).get("headers", [])
        out.append({
            "id": m["id"],
            "from": _inbox._header(headers, "From"),
            "from_email": _normalize_sender_email(_inbox._header(headers, "From")),
            "subject": _inbox._header(headers, "Subject"),
            "date": _inbox._header(headers, "Date"),
            "internal_date_ms": int(full.get("internalDate") or 0),
            "snippet": full.get("snippet", ""),
        })
    return out


def _looks_transactional(msg: dict) -> bool:
    """True if the message looks like a sale/payment notification, not coordination."""
    sender = (msg.get("from_email") or "").lower()
    if sender in PROGRAM_SENDERS:
        return False
    # If sender is a known automated address (notifications@, no-reply@, etc.)
    if any(prefix in sender for prefix in ["notifications@", "noreply@", "no-reply@",
                                            "billing@", "payments@", "sales@", "orders@",
                                            "receipts@", "stripe@"]):
        return True
    # If subject + snippet mention money or sale verbs, count it
    haystack = ((msg.get("subject") or "") + " " + (msg.get("snippet") or "")).lower()
    if _AMOUNT_RE.search(haystack):
        return True
    if any(kw in haystack for kw in ["purchase", "purchased", "new sale", "you got paid",
                                      "payout", "subscription", "order", "receipt",
                                      "earned", "earning"]):
        return True
    return False


def _extract_amount_cents(msg: dict) -> Optional[int]:
    """Pull the first $ amount out of subject+snippet. Returns cents or None."""
    text = (msg.get("subject") or "") + " " + (msg.get("snippet") or "")
    m = _AMOUNT_RE.search(text)
    if not m:
        return None
    s = m.group(1).replace(",", "")
    try:
        if "." in s:
            dollars, cents = s.split(".")
            return int(dollars) * 100 + int(cents.ljust(2, "0")[:2])
        return int(s) * 100
    except ValueError:
        return None


def summary(days_back: int = 30) -> dict:
    """Dashboard widget payload — sales counts + recent items + revenue."""
    try:
        msgs = _list_studio_messages(days_back=days_back)
    except Exception as e:
        return {"error": str(e), "configured": True, "awaiting_first_sale": True}

    now = datetime.now(timezone.utc)
    today_start_ms = int(datetime(now.year, now.month, now.day, tzinfo=timezone.utc).timestamp() * 1000)
    week_start_ms = int((now - timedelta(days=7)).timestamp() * 1000)

    transactional = [m for m in msgs if _looks_transactional(m)]
    program = [m for m in msgs if not _looks_transactional(m)]

    sales_today = [m for m in transactional if m["internal_date_ms"] >= today_start_ms]
    sales_week = [m for m in transactional if m["internal_date_ms"] >= week_start_ms]

    revenue_today_cents = sum((_extract_amount_cents(m) or 0) for m in sales_today)
    revenue_week_cents = sum((_extract_amount_cents(m) or 0) for m in sales_week)
    revenue_30d_cents = sum((_extract_amount_cents(m) or 0) for m in transactional)

    last_sale = transactional[0] if transactional else None
    last_sale_amount_cents = _extract_amount_cents(last_sale) if last_sale else None

    return {
        "as_of": now.isoformat(),
        "configured": True,
        "awaiting_first_sale": len(transactional) == 0,
        "sales_count_today": len(sales_today),
        "sales_count_week": len(sales_week),
        "sales_count_30d": len(transactional),
        "revenue_today_cents": revenue_today_cents,
        "revenue_week_cents": revenue_week_cents,
        "revenue_30d_cents": revenue_30d_cents,
        "last_sale": ({
            "from": last_sale["from"],
            "subject": last_sale["subject"],
            "date": last_sale["date"],
            "internal_date_ms": last_sale["internal_date_ms"],
            "amount_cents": last_sale_amount_cents,
        } if last_sale else None),
        "program_email_count_30d": len(program),
        "recent_program_emails": [
            {"from": p["from"], "subject": p["subject"], "date": p["date"]}
            for p in program[:3]
        ],
        "dashboard_link": "https://studio.com/dashboard?section=earnings",
        "inbox_search_link": "/console/inbox?key=&q=" + "from%3Astudio.com",
    }
