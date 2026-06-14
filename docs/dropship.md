# Practitioner-paid drop-ship

A wholesale-approved practitioner orders Functional Formulations **for a specific patient**,
pays the **drop-ship wholesale**, and we ship to the patient. The practitioner bills the
patient privately (their margin is off-platform).

## Pricing (`dashboard/dropship_checkout.py`)
- **Per-bottle = base + fee.** Base = the blended wholesale curve at the order's **total
  bottles** (`practitioner_pricing.drop_ship_base_cents` / `quote_line`); $50 @ 1 bottle →
  cert floor ($40 uncert → $25 certified) at 40 bottles.
- **Fee = 33% of (retail − base)** — RM's standard cut. Because the patient transaction is
  private in practitioner-paid mode, the fee is based on **retail**, not a declared patient
  price; the practitioner enters **no selling price**. (Design decision 2026-06-14; flip to
  a declared-price-with-floor model if desired.)
- `dropship_line_cents(retail_cents, qty, modules, settings) → {base_cents, fee_cents,
  unit_cents, line_cents}`.

## Order (`build_dropship_order`)
Mirrors `wholesale_checkout.build_order`: QBO customer = the **practitioner** (they pay);
the invoice + order **ship to the patient** (`patient_ship`); `source="dropship"`; wallet
redeem ≤50% + fee-free 3% earn on zelle/wise (same as wholesale); **GET recorded-not-charged**
on the patient's state; **no MAP** (private). Returns `ok / invoice_id / total / customer_id /
ship_to / source / get_cents`.

## Routes + page
- `GET/POST /api/practitioner/dropship/quote` — per-line drop-ship pricing + total for the
  cart (authed).
- `POST /api/practitioner/dropship/checkout` — body `{method, patient_address}`; requires
  `wholesale_unlocked`; clears cart; `_ingest_order(source="dropship")`; alt-pay or Stripe.
- `GET /practitioner/dropship` → `static/practitioner-dropship.html` (cart + patient address
  + live total + pay). No selling-price field. US ship-to only.

## Not here
The **patient-paid client page** (patient pays the practitioner's MAP-enforced price, margin
→ `wallet.earn_dropship_margin`) is **Plan 3**. White-label branding is **Plan 4**.
