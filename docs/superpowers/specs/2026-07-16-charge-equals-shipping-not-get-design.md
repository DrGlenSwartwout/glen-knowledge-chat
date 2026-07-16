# Charge = subtotal + shipping (not GET) — Design

**Date:** 2026-07-16
**Status:** Approved (design), pending spec review
**Owner:** Glen / RemedyMatch
**Context:** discovered while verifying the QBO paid-only migration — the Stripe charge and the booked Sales Receipt (and, pre-migration, the invoice) did not agree.

## Problem

For the retail/reorder/biofield checkout flows the amount charged to the customer
is `pc["priced"]["total_cents"]` = **subtotal + GET** (Hawaii General Excise Tax),
with **no shipping**. But the order's shipping is real and the Sales Receipt
(and the old invoice) is built as **subtotal + shipping**, `tax_cents = 0`.

So **charge ≠ receipt**: the customer is charged merch + GET (no shipping), while
QBO records merch + shipping (no GET). Two errors compound:
- **Shipping is never collected** (omitted from the charge) yet is recorded as
  revenue on the receipt.
- **GET is charged** to the customer, but Glen's policy is **GET is NOT charged**
  — it is absorbed and only recorded (for remittance).

Empirical (2× `ei8-microbes-liver-integrator`, ship CA): charge $77.83
(subtotal − discount, GET 0), receipt $100.83 (+$23 shipping) → $23 gap.

## Policy (confirmed with Glen)
- **GET is never charged** to the customer. It stays **recorded** on the order
  (`get_cents`, absorbed) for Glen's GET remittance — it is NOT on the charge and
  NOT a tax line on the receipt.
- **Shipping IS charged.** The customer pays merch + shipping.
- Therefore **charge == receipt == subtotal + shipping**.

## Design

**One change: the charged amount (and the order's stored total) become
`subtotal + shipping`.** Since `total_cents = subtotal + get_cents` and
`get_cents` is the absorbed GET, the corrected amount is:

```
charged_cents = int(pc["priced"]["total_cents"]) - int(pc["priced"]["get_cents"]) + int(pc["shipping_cents"])
```

i.e. remove GET, add shipping. Add a single helper (e.g.
`_charge_cents(pc)` in app.py) and use it everywhere the charge is currently
`pc["priced"]["total_cents"]`:
- `begin_checkout` — the `out["total"]`/charged amount fed to
  `_stripe_checkout_url_for_retail`, and the `_ingest_order(total_cents=...)`.
- `_checkout_cart` / `api_client_portal_checkout` / `reorder_subscribe` — the
  `out["total"]` fed to `_stripe_checkout_url_for_reorder`, and the order total.
- `biofield_checkout` — `charged_cents`; biofield is a service with
  `shipping_cents = 0`, so this reduces its charge to `subtotal` (drops the GET it
  should never have charged).

**The Sales Receipt payload is unchanged** — it is already `subtotal + shipping`
with `tax_cents = 0`, which now equals the charge exactly. `get_cents` remains
stored on the order (already the case) for remittance.

**Consistency:** after this, for every paid-only flow the Stripe amount, the
order's `total_cents`, and the booked Sales Receipt all equal `subtotal + shipping`.

## Stage 2 already-booked receipts (check + remediate)
Stage 2 (biofield + begin) went live 2026-07-16 ~15:41 UTC. Any order in the live
window that (a) booked a Sales Receipt (`qbo_sales_receipt_id` set) AND (b) had
`shipping_cents > 0` OR `get_cents > 0` has a receipt total ≠ amount charged.
- **Check:** query prod orders `WHERE source IN ('funnel','biofield') AND
  qbo_sales_receipt_id IS NOT NULL AND created_at >= '<stage2 deploy>' AND
  (shipping_cents > 0 OR get_cents > 0)`.
- Expectation: **likely none** (biofield is a service with no shipping; few/no
  physical funnel purchases completed in the ~short window). If any exist, list
  them for Glen (order id, email, charged vs receipt) to adjust the QBO receipt
  and/or collect the shipping difference. No automatic mutation.

## Testing
- Guard/pinning: for a shippable cart with `shipping_cents > 0`, the charged amount
  passed to Stripe == the Sales-Receipt payload total (== subtotal + shipping),
  and both exclude GET.
- For a HI order (`get_cents > 0`): the charge does NOT include GET (drops it), and
  `get_cents` is still recorded on the order.
- Biofield (shipping 0, service): charge == subtotal, GET not charged.
- Regression: existing checkout/pricing suites; the paid-only guard/pinning tests
  (Stage 2/3) still green.

## Out of scope
- Wholesale/dropship (their own GET-absorbed handling, Stage 4).
- The QBO Stage 3 conversions themselves (resumed after this lands).
