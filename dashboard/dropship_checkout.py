"""Practitioner-paid drop-ship: price each line at the drop-ship wholesale
(blended base + 33% of the RETAIL markup), invoice the practitioner, ship to the
patient. The practitioner bills the patient privately (margin off-platform)."""
from __future__ import annotations

from typing import List

from dashboard import practitioner_pricing as _pp
from dashboard import qbo_billing as qb
from dashboard import wallet


def _settings():
    # Drop-ship pricing settings; pairs with the pending console editor.
    return _pp.load_settings({})


def _retail_for(slug: str) -> int:
    """Retail price in cents for a product slug. Monkeypatchable in tests."""
    import app as _app
    return _app._get_product(slug)["price_cents"]


def dropship_line_cents(*, retail_cents, qty, modules, settings):
    """Per-line drop-ship economics. Fee is 33% of (retail - base) — RM's standard cut,
    since the patient price is private in practitioner-paid mode. Reuses Plan 1's
    quote_line with selling=retail."""
    q = _pp.quote_line(selling_cents=int(retail_cents), qty=int(qty),
                       modules=int(modules), settings=settings)
    unit = q["dropship_wholesale_cents"]          # base + fee
    return {
        "base_cents": q["base_cents"],
        "fee_cents": q["fee_cents"],
        "unit_cents": unit,
        "line_cents": unit * int(qty),
    }


def build_dropship_order(cart: List[dict], practitioner: dict, *,
                         patient_ship: dict, method=None) -> dict:
    """Price + invoice a drop-ship cart at wholesale, ship to the patient.

    Mirrors wholesale_checkout.build_order. Differences:
    - Price each line via dropship_line_cents (base + 33% of retail markup).
    - QBO customer is the PRACTITIONER (they pay), but ship-to is the PATIENT.
    - source = "dropship".
    - Wallet: redeem <= 50% + fee-free 3% earn on zelle/wise.
    - GET recorded-not-charged on the patient's ship-to state.

    practitioner needs: id (uuid), modules_completed, email, name.
    Returns dict with ok / invoice_id / total / customer_id / ship_to / source.
    """
    if not cart:
        return {"ok": False, "error": "empty_cart"}

    # Compute total bottles for the blended base curve (order-level, not per-line).
    total_bottles = sum(int(item.get("qty", 0)) for item in cart)
    if total_bottles <= 0:
        return {"ok": False, "error": "empty_cart"}

    modules = int(practitioner.get("modules_completed", 0))
    settings = _settings()

    # Build QBO lines — each priced at drop-ship wholesale (base + fee).
    lines = []
    subtotal_cents = 0
    for item in cart:
        slug = item["slug"]
        qty = int(item.get("qty", 1))
        retail = _retail_for(slug)
        dl = dropship_line_cents(retail_cents=retail, qty=total_bottles,
                                 modules=modules, settings=settings)
        unit_cents = dl["unit_cents"]
        subtotal_cents += unit_cents * qty
        lines.append({
            "name": slug,
            "amount": unit_cents / 100.0,
            "qty": qty,
            "description": f"{slug} (drop-ship wholesale)",
        })

    # Create QBO invoice billed to the PRACTITIONER.
    cust = qb.find_or_create_customer(practitioner["email"], practitioner.get("name", ""))

    from dashboard import tax as _tax
    get_cents = _tax.compute_get_cents(subtotal_cents, channel="wholesale",
                                       ship_to_state=patient_ship.get("state", ""))

    inv = qb.create_invoice(cust, lines, email_to=practitioner["email"])
    invoice_id = inv.get("Id")

    # Wallet: redeem up to 50% of the order.
    redeemed = wallet.redeem_for_order(practitioner["id"], subtotal_cents, invoice_id)
    if redeemed > 0:
        inv = qb.apply_invoice_discount(invoice_id, redeemed)

    # Fee-free 3% earn on zelle/wise.
    fee_free = 0
    if method in ("zelle", "wise"):
        charged = max(0, subtotal_cents - redeemed)
        fee_free = wallet.earn_fee_free(practitioner["id"], charged, invoice_id)

    return {
        "ok": True,
        "invoice_id": invoice_id,
        "sync_token": inv.get("SyncToken"),
        "doc_number": inv.get("DocNumber"),
        "total": inv.get("TotalAmt"),
        "subtotal_cents": subtotal_cents,
        "credit_redeemed_cents": redeemed,
        "fee_free_credit_cents": fee_free,
        "get_cents": get_cents,
        "method": method,
        "customer_id": cust.get("Id"),
        "ship_to": patient_ship,
        "source": "dropship",
    }
