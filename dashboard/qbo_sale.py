"""Paid-only QBO booking: create ONE line-faithful Sales Receipt for an order the
moment it is confirmed paid. Idempotent (never re-books) and best-effort (never
raises into the payment path). See docs/superpowers/specs/2026-07-15-qbo-paid-only-
stage2-checkout-design.md."""
import json

from . import qbo_billing
from . import orders


def book_sale_on_payment(cx, order):
    """Book a QBO SalesReceipt for a PAID order from its stored qbo_lines_json.
    Returns the receipt Id (existing one if already booked; None on any failure or
    when there is nothing to book). Never raises."""
    try:
        existing = order.get("qbo_sales_receipt_id")
        if existing:
            return existing
        raw = order.get("qbo_lines_json")
        if not raw:
            return None
        payload = json.loads(raw) if isinstance(raw, str) else raw
        lines = payload.get("lines") or []
        if not lines:
            return None
        cust = qbo_billing.find_or_create_customer(order.get("email") or "",
                                                   order.get("name") or "")
        sr = qbo_billing.create_sales_receipt(
            cust, lines,
            discount_cents=int(payload.get("discount_cents") or 0),
            tax_cents=int(payload.get("tax_cents") or 0),
            email_to=order.get("email") or None)
        sr_id = sr.get("Id")
        if sr_id and order.get("id") is not None:
            orders.set_order_sales_receipt_id(cx, order["id"], sr_id)
        return sr_id
    except Exception as e:  # best-effort — must never break the payment path
        print(f"[qbo-sale] book_sale_on_payment skipped for order "
              f"{order.get('id')!r}: {e!r}", flush=True)
        return None
