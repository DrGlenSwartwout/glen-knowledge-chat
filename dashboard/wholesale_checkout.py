"""Practitioner wholesale checkout — the money path (Phase 3e).

Ties the pricing engine and the wallet to QBO: price a cart, create ONE invoice,
redeem Wellness Credit, and apply the redeemed amount as a discount line. Credit
is redeemed AFTER the invoice exists (so the redemption is authoritative against
the locked balance and idempotent on the invoice id), then the discount is set to
exactly what was committed. Training-first allocation lives at the route layer:
call build_module_order before build_order when both are pending.
"""

from __future__ import annotations

from typing import List, Optional

from dashboard import qbo_billing as qb
from dashboard import wallet
from dashboard import wholesale_pricing as pricing


def _qbo_line(ln: dict, catalog: Optional[dict]) -> dict:
    p = pricing._product_pricing(ln["slug"], catalog)
    return {
        "name": ln["name"],
        "amount": ln["unit_price_cents"] / 100.0,
        "qty": ln["qty"],
        "item_id": p.get("qbo_item_id"),
        "description": f'{ln["name"]} (wholesale)',
    }


def build_order(cart_items: List[dict], practitioner: dict, *,
                db_path=None, catalog=None, allow_override=False) -> dict:
    """Price + invoice a product cart, then redeem up to 50% of the order in credit.

    practitioner needs: id (uuid), modules_completed, email, name.
    Returns the checkout result dict (ok / error)."""
    quote = pricing.order_quote(cart_items, practitioner, db_path=db_path, catalog=catalog)
    if quote["total_bottles"] <= 0:
        return {"ok": False, "error": "empty_cart"}
    if not quote["margin_ok"] and not allow_override:
        return {"ok": False, "error": "margin_floor",
                "margin_warnings": quote["margin_warnings"], "quote": quote}

    lines = [_qbo_line(ln, catalog) for ln in quote["lines"]]
    cust = qb.find_or_create_customer(practitioner["email"], practitioner.get("name", ""))
    inv = qb.create_invoice(cust, lines, email_to=practitioner["email"])
    invoice_id = inv.get("Id")

    redeemed = wallet.redeem_for_order(practitioner["id"], quote["subtotal_cents"], invoice_id)
    if redeemed > 0:
        inv = qb.apply_invoice_discount(invoice_id, redeemed)

    return {
        "ok": True,
        "invoice_id": invoice_id,
        "sync_token": inv.get("SyncToken"),
        "doc_number": inv.get("DocNumber"),
        "total": inv.get("TotalAmt"),
        "subtotal_cents": quote["subtotal_cents"],
        "blended_unit_price_cents": quote["blended_unit_price_cents"],
        "credit_redeemed_cents": redeemed,
    }


def build_module_order(practitioner: dict, module_slug: str, *, today,
                       tuition_cents: int = wallet.MODULE_TUITION_CENTS) -> dict:
    """Invoice one certification module, then redeem up to 100% of tuition in credit
    (monthly-gated). practitioner needs: id, email, name."""
    cust = qb.find_or_create_customer(practitioner["email"], practitioner.get("name", ""))
    inv = qb.create_invoice(cust, [{
        "name": f"Certification Module — {module_slug}",
        "amount": round(tuition_cents / 100.0, 2),
        "qty": 1,
        "description": f"Functional Formulations Certification — {module_slug}",
    }], email_to=practitioner["email"])
    invoice_id = inv.get("Id")

    redeemed = wallet.redeem_for_module(practitioner["id"], module_slug,
                                        today=today, tuition_cents=tuition_cents)
    if redeemed > 0:
        inv = qb.apply_invoice_discount(invoice_id, redeemed)

    return {
        "ok": True,
        "invoice_id": invoice_id,
        "sync_token": inv.get("SyncToken"),
        "doc_number": inv.get("DocNumber"),
        "total": inv.get("TotalAmt"),
        "tuition_cents": tuition_cents,
        "credit_redeemed_cents": redeemed,
    }
