"""Practitioner wholesale checkout — the money path (Phase 3e).

``build_order`` (product cart) is QBO paid-only (Stage 4): it prices the cart and
resolves the Wellness Credit redemption up front against a freshly-minted
``checkout_ref`` token -- NOT a QBO invoice id, since no invoice is created at
checkout time. A real, line-faithful QBO Sales Receipt is booked later by the
return-handler (once payment is confirmed) from the ``qbo_payload`` this returns.
Redemption happens before booking (a Sales Receipt is final once posted) and is
idempotent on ``checkout_ref``.

``quote_module`` (certification tuition) is the paid-only module path: it
previews what a coach owes WITHOUT creating a QBO invoice or redeeming credit
at signup -- credit is redeemed only when the payment is recorded (a Sales
Receipt), matching the no-A/R-at-signup policy.
"""

from __future__ import annotations

import uuid
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


def build_order(cart_items: List[dict], practitioner: dict, *, method=None,
                db_path=None, catalog=None, allow_override=False,
                ship_to_state="", resale_ok=False) -> dict:
    """Price a product cart (paid-only -- no QBO invoice/customer at checkout
    time), then redeem up to 50% of the order in credit. When paid fee-free
    (method zelle/wise), also earn 3% Wellness Credit on the amount charged. A
    real, line-faithful QBO Sales Receipt is booked by the return-handler once
    payment is confirmed, from the ``qbo_payload`` this returns.

    Credit is resolved BEFORE booking (a Sales Receipt is final once posted),
    keyed on ``checkout_ref`` -- a fresh token minted here, not a QBO invoice id
    -- so the redemption is idempotent per checkout without QBO existing yet.

    practitioner needs: id (uuid), modules_completed, email, name.
    Returns the checkout result dict (ok / error)."""
    quote = pricing.order_quote(cart_items, practitioner, db_path=db_path, catalog=catalog)
    if quote["total_bottles"] <= 0:
        return {"ok": False, "error": "empty_cart"}
    if not quote["margin_ok"] and not allow_override:
        return {"ok": False, "error": "margin_floor",
                "margin_warnings": quote["margin_warnings"], "quote": quote}

    lines = [_qbo_line(ln, catalog) for ln in quote["lines"]]
    from dashboard import tax as _tax
    # Absorb-and-track: GET is computed and returned for the order ledger, NOT
    # charged to the practitioner (the wholesale price is all-in).
    get_cents = _tax.compute_get_cents(quote["subtotal_cents"], channel="wholesale",
                                       ship_to_state=ship_to_state, resale_ok=resale_ok)

    checkout_ref = uuid.uuid4().hex   # stable order/correlation key (no QBO invoice yet)
    redeemed = wallet.redeem_for_order(practitioner["id"], quote["subtotal_cents"], checkout_ref)
    charged = max(0, quote["subtotal_cents"] - redeemed)

    fee_free = 0
    if method in ("zelle", "wise"):
        fee_free = wallet.earn_fee_free(practitioner["id"], charged, checkout_ref)

    return {
        "ok": True,
        "invoice_id": checkout_ref,
        "customer_id": "",
        "doc_number": "",
        "total": round(charged / 100.0, 2),
        "subtotal_cents": quote["subtotal_cents"],
        "blended_unit_price_cents": quote["blended_unit_price_cents"],
        "credit_redeemed_cents": redeemed,
        "fee_free_credit_cents": fee_free,
        "get_cents": get_cents,
        "method": method,
        "qbo_payload": {"lines": lines, "discount_cents": redeemed, "tax_cents": 0},
    }


def quote_module(practitioner: dict, module_slug: str,
                 *, tuition_cents: int = wallet.MODULE_TUITION_CENTS) -> dict:
    """Paid-only quote for one certification module: what the coach owes, WITHOUT
    creating a QBO invoice or redeeming credit. Credit is previewed read-only and
    actually redeemed only when the payment is recorded (a Sales Receipt), matching
    the QBO paid-only policy of no A/R invoices at signup. practitioner needs: id."""
    balance = wallet.get_balance_cents(practitioner["id"])
    redeemable = wallet.redeem_amount_for_module_cents(balance, tuition_cents)
    due = tuition_cents - redeemable
    return {
        "ok": True,
        "tuition_cents": tuition_cents,
        "credit_available_cents": redeemable,
        "amount_due_cents": due,
        "total": round(due / 100.0, 2),
    }
