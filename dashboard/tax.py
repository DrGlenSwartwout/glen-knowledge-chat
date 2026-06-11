"""Hawai'i GET computation (Project B — absorb-and-track).

GET is legally the seller's tax, so it is ABSORBED, not charged to the customer:
the app computes the GET owed per order and records it on the order ledger
(orders.get_cents) for remittance — it is NOT added to the invoice/Stripe total.
TAX_ENABLED therefore means "compute and record GET" (tracking), not "charge."

Rates + the enable flag are CONFIG (env), so Rae/CPA set them without code, and
the whole thing ships disabled (TAX_ENABLED=false → always 0) until the numbers
are confirmed. This module is pure and side-effect-free.

(A customer pass-through mode remains latent via qbo_billing.create_invoice's
tax_cents override hook, but is not wired in absorb mode.)

Tax-law note: rate values and the out-of-state rule are Rae/CPA determinations,
not encoded judgment. Defaults here are conservative placeholders.
"""

import os


def _rate(name, default):
    try:
        v = os.environ.get(name, "").strip()
        return float(v) if v else float(default)
    except (TypeError, ValueError):
        return float(default)


# Master switch — off until Rae/CPA confirm the rates. Off → tax is always 0,
# so merging/deploying this changes nothing until the flag is flipped.
def tax_enabled():
    return os.environ.get("TAX_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")


GET_HOME_STATE = (os.environ.get("GET_HOME_STATE", "HI").strip().upper() or "HI")


def retail_rate():
    # e.g. 0.045 (4.5%) or 0.04712 (the HI max pass-on). Rae/CPA to confirm.
    return _rate("GET_RETAIL_RATE", 0.045)


def wholesale_rate():
    # Hawai'i wholesale GET on sales for resale.
    return _rate("GET_WHOLESALE_RATE", 0.005)


def compute_get_cents(subtotal_cents, *, channel, ship_to_state, resale_ok=False):
    """Return the GET to add to an invoice, in cents.

    - Disabled (TAX_ENABLED off) → 0 (no behavior change).
    - Ship-to outside the home state (HI) → 0 (export; configured rule).
    - Wholesale channel AND a resale certificate on file → wholesale rate.
    - Otherwise → retail rate.
    """
    if not tax_enabled():
        return 0
    try:
        sub = int(subtotal_cents or 0)
    except (TypeError, ValueError):
        return 0
    if sub <= 0:
        return 0
    st = (ship_to_state or "").strip().upper()
    if st != GET_HOME_STATE:
        return 0
    rate = wholesale_rate() if (channel == "wholesale" and resale_ok) else retail_rate()
    return int(round(sub * rate))
