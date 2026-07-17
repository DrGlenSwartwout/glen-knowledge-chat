"""Practitioner-paid drop-ship: price each line at the drop-ship wholesale
(blended base + 33% of the RETAIL markup), ship to the patient. The
practitioner bills the patient privately (margin off-platform).

``build_dropship_order`` is QBO paid-only (Stage 4): it prices the cart and
resolves the Wellness Credit redemption up front against a freshly-minted
``checkout_ref`` token -- NOT a QBO invoice id, since no invoice is created at
checkout time. A real, line-faithful QBO Sales Receipt is booked later by the
return-handler (once payment is confirmed) from the ``qbo_payload`` this returns.
Redemption happens before booking (a Sales Receipt is final once posted) and is
idempotent on ``checkout_ref``.

Also provides build_client_order — the patient-paid sibling where the patient is
invoiced at the practitioner's price S and the practitioner's margin (S - base - fee)
is returned for wallet crediting on payment. (Unchanged/out of scope here.)"""
from __future__ import annotations

import os
import sqlite3
import uuid
from pathlib import Path
from typing import List

from dashboard import practitioner_pricing as _pp
from dashboard import pricing as _pricing
from dashboard import qbo_billing as qb
from dashboard import wallet

# Resolve LOG_DB the same way practitioner_portal.py does — respects DATA_DIR in prod.
_LOG_DB = str(
    Path(os.environ.get("DATA_DIR", str(Path(__file__).resolve().parent.parent)))
    / "chat_log.db"
)


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
    """Price a drop-ship cart at wholesale (paid-only -- no QBO invoice/customer
    at checkout time), ship to the patient. A real, line-faithful QBO Sales
    Receipt is booked by the return-handler once payment is confirmed, from the
    ``qbo_payload`` this returns.

    Mirrors wholesale_checkout.build_order. Differences:
    - Price each line via dropship_line_cents (base + 33% of retail markup).
    - The PRACTITIONER pays (wallet credit is theirs), but ship-to is the PATIENT.
    - source = "dropship".
    - Wallet: redeem <= 50% (keyed on checkout_ref, resolved BEFORE booking) +
      fee-free 3% earn on zelle/wise.
    - GET recorded-not-charged on the patient's ship-to state.

    practitioner needs: id (uuid), modules_completed, email, name.
    Returns dict with ok / invoice_id (checkout_ref token) / total / customer_id
    (always "") / qbo_payload / ship_to / source.
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
        line_qty = int(item.get("qty", 1))                  # bottles on THIS line
        retail = _retail_for(slug)
        # base/fee use total_bottles for the blended curve; the line cost uses line_qty.
        dl = dropship_line_cents(retail_cents=retail, qty=total_bottles,
                                 modules=modules, settings=settings)
        unit_cents = dl["unit_cents"]
        subtotal_cents += unit_cents * line_qty
        lines.append({
            "name": slug,
            "amount": unit_cents / 100.0,
            "qty": line_qty,
            "description": f"{slug} (drop-ship wholesale)",
        })

    from dashboard import tax as _tax
    get_cents = _tax.compute_get_cents(subtotal_cents, channel="wholesale",
                                       ship_to_state=patient_ship.get("state", ""))

    # Paid-only: no QBO invoice/customer yet -- mint a stable order/correlation
    # key and resolve credit redemption against IT (never a QBO invoice id).
    checkout_ref = uuid.uuid4().hex
    redeemed = wallet.redeem_for_order(practitioner["id"], subtotal_cents, checkout_ref)
    charged = max(0, subtotal_cents - redeemed)

    # Fee-free 3% earn on zelle/wise.
    fee_free = 0
    if method in ("zelle", "wise"):
        fee_free = wallet.earn_fee_free(practitioner["id"], charged, checkout_ref)

    return {
        "ok": True,
        "invoice_id": checkout_ref,
        "customer_id": "",
        "doc_number": "",
        "total": round(charged / 100.0, 2),
        "subtotal_cents": subtotal_cents,
        "credit_redeemed_cents": redeemed,
        "fee_free_credit_cents": fee_free,
        "get_cents": get_cents,
        "method": method,
        "qbo_payload": {"lines": lines, "discount_cents": redeemed, "tax_cents": 0},
        "ship_to": patient_ship,
        "source": "dropship",
    }


# ── Patient-paid (client page) economics ─────────────────────────────────────


def _practitioner_price_cents(pid: str, slug: str, retail: int) -> int:
    """Return the practitioner's stored selling price for (pid, slug) in cents.

    Resolution: per-SKU override → default markup % → retail; clamped to MAP.
    Falls back to max(retail, MAP) on any error so a settings problem never
    crashes a checkout.
    """
    from dashboard import practitioner_settings as _ps
    settings = _settings()
    map_floor = int(settings.get("map_default_cents", 6700))
    try:
        cx = sqlite3.connect(_LOG_DB)
        cx.row_factory = sqlite3.Row
        _ps.init_settings_table(cx)
        try:
            return _ps.price_cents_for(
                cx, pid, slug,
                retail_cents=retail,
                map_cents=map_floor,
            )
        finally:
            cx.close()
    except Exception:
        # Best-effort fallback: never crash a checkout due to a settings error.
        return max(retail, map_floor)


def practitioner_price_for(pid: str, slug: str) -> int:
    """Return the practitioner's selling price for slug in cents (>= MAP)."""
    retail = _retail_for(slug)
    return _practitioner_price_cents(pid, slug, retail)


def build_client_order(cart: List[dict], practitioner: dict, *,
                       patient: dict, method=None,
                       points_to_redeem_cents=0, points_balance_cents=0,
                       effective_settings=None, program_member=False,
                       ship_credit_balance_cents=0) -> dict:
    """Price + invoice a patient-paid (dispensary) cart at the practitioner's price S.

    The patient is the QBO customer and pays S per bottle.  The practitioner's
    margin (S − base − fee) is summed and returned so the caller can credit it
    to the practitioner's wallet once payment clears.

    When ``effective_settings`` is provided the patient RECEIVES the
    practitioner's best-of volume discount off S (same engine helpers the direct
    channel uses): the discount comes OUT of the practitioner's margin, clamped
    to the house ceilings baked into ``effective_settings`` and floored via
    ``pricing.unit_floor_cents``.  When ``effective_settings`` is None the
    behavior is byte-identical to the flat-S baseline (no discount, unchanged
    margin) — the discount machinery is skipped entirely.

    ``build_client_order`` is QBO paid-only (Stage 4): no QBO invoice/customer
    is created at checkout time -- a fresh ``checkout_ref`` token is minted and
    a line-faithful ``qbo_payload`` is returned for the route to persist. The
    return-handler (already wired for kind="client") books a real QBO Sales
    Receipt from it once payment is confirmed. The points-redeem/ship-credit
    discount is already resolved before this point (never a QBO invoice id),
    so this is a straight Pattern-I conversion.

    Key differences from build_dropship_order:
    - QBO customer = the PATIENT (patient["email"]) -- but not created here;
      the return-handler resolves it when booking the Sales Receipt.
    - Each line is priced at S = practitioner_price_for(pid, slug) (>= MAP),
      optionally reduced by the practitioner-effective volume discount.
    - base/fee/margin computed via quote_line(selling_cents=S, qty=total_bottles).
    - Ship to patient["ship"]; source = "dispensary".
    - GET recorded-not-charged on the patient's ship-to state.
    - NO wallet redeem (the margin is credited on PAID, not here).

    Returns dict with ok / invoice_id (checkout_ref token) / total / customer_id
    (always "") / doc_number (always "") / qbo_payload / ship_to / source /
    margin_cents / get_cents / points_redeemed_cents / ship_credit_applied_cents.
    """
    if not cart:
        return {"ok": False, "error": "empty_cart"}

    total_bottles = sum(int(item.get("qty", 0)) for item in cart)
    if total_bottles <= 0:
        return {"ok": False, "error": "empty_cart"}

    pid = practitioner["id"]
    modules = int(practitioner.get("modules_completed", 0))
    settings = _settings()
    ship = patient["ship"]

    eff = effective_settings
    # Order-wide, best-of discount context (mirrors the direct engine). Only
    # computed when a practitioner discount config is in play; when eff is None
    # the loop charges flat S and this stays 0, preserving today's behavior.
    _app = None
    open_pct = prog_pct = 0.0
    if eff:
        import app as _app  # lazy (same pattern as _retail_for) — avoids a cycle
        total_ff = sum(int(i.get("qty", 0)) for i in cart
                       if _app._qty_eligible(_app._get_product(i["slug"]) or {}))
        open_pct = _pricing.open_total_pct(total_ff, eff)
        prog_pct = _pricing.program_total_pct(total_ff, eff, program_member)

    lines = []
    subtotal_cents = 0
    total_margin_cents = 0
    total_fee_cents = 0

    for item in cart:
        slug = item["slug"]
        line_qty = int(item.get("qty", 1))
        # S: practitioner's selling price for this slug (>= MAP)
        s_cents = practitioner_price_for(pid, slug)
        # base/fee/margin use total_bottles for the blended curve
        q = _pp.quote_line(selling_cents=s_cents, qty=total_bottles,
                           modules=modules, settings=settings)
        if eff:
            prod = _app._get_product(slug) or {}
            elig = bool(_app._qty_eligible(prod))
            if elig:
                t1 = _pricing.same_sku_pct(line_qty, eff)
                line_pct = max(t1, prog_pct, open_pct)   # non-additive: best single offer
            else:
                line_pct = 0.0
            floor = _pricing.unit_floor_cents(prod, s_cents, eff, "discount")
            paid_unit = _pricing.apply_discount(s_cents, line_pct, floor)
        else:
            paid_unit = s_cents   # baseline: patient pays flat S
        # The discount is taken out of the practitioner's margin.
        line_margin = q["margin_cents"] - (s_cents - paid_unit)
        subtotal_cents += paid_unit * line_qty
        total_margin_cents += line_margin * line_qty
        total_fee_cents += q["fee_cents"] * line_qty
        lines.append({
            "name": slug,
            "amount": paid_unit / 100.0,   # patient is charged the discounted S
            "qty": line_qty,
            "description": f"{slug} (dispensary)",
        })

    from dashboard import tax as _tax
    get_cents = _tax.compute_get_cents(subtotal_cents, channel="dispensary",
                                       ship_to_state=ship.get("state", ""))

    # Fee-capped patient points redemption: never below product base (RM keeps
    # selling at >= base + the practitioner's full margin); RM absorbs the discount.
    redeem_cents = max(0, min(int(points_to_redeem_cents or 0),
                              int(points_balance_cents or 0),
                              total_fee_cents))

    # Shipping credit (slice 2b, flag-gated by the caller which passes a 0 balance
    # when off): auto-apply the patient's outstanding ship_credit balance, bounded by
    # the payable after points (subtotal − points; GET is recorded, not charged).
    # Folded into the invoice discount so the charge drops; debited at payment settle.
    from dashboard import ship_credit as _sc
    ship_credit_applied = _sc.plan_application(
        int(ship_credit_balance_cents or 0), max(0, subtotal_cents - redeem_cents))

    # Paid-only: no QBO invoice/customer yet -- mint a stable order/correlation
    # key. The discount (points + ship-credit) is already resolved above (never
    # a QBO invoice id), so it folds straight into the qbo_payload discount.
    checkout_ref = uuid.uuid4().hex
    discount_cents = redeem_cents + ship_credit_applied
    charged = max(0, subtotal_cents - discount_cents)

    return {
        "ok": True,
        "invoice_id": checkout_ref,
        "customer_id": "",
        "doc_number": "",
        "total": round(charged / 100.0, 2),
        "ship_to": ship,
        "source": "dispensary",
        "subtotal_cents": subtotal_cents,
        "margin_cents": total_margin_cents,
        "points_redeemed_cents": redeem_cents,
        "ship_credit_applied_cents": ship_credit_applied,
        "get_cents": get_cents,
        "qbo_payload": {"lines": lines, "discount_cents": discount_cents, "tax_cents": 0},
    }
