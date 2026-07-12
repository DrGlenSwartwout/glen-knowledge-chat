# Recalc shipping — overpayment credit for already-paid combined-shipment members

**Date:** 2026-07-12
**Status:** Approved (Approach A)
**Origin:** Combining a household shipment lowers the one-parcel shipping. Members
who already paid their standalone total are now overpaid by their shipping saving,
but "Recalc shipping" currently *skips* paid members and records nothing — the
overpayment is invisible. (Concrete case: Desiree, paid, in a household shipment.)

## Goal

When `Recalc shipping` lowers an already-paid member's fair shipping share below
what they paid, record the overpayment as a **credit on that order** and surface it
in the console so Rae can apply it to the client's next order or refund it. Do not
disturb the paid invoice.

## Non-goals

- No automatic money movement (no card refund on a button click).
- No Wellness Credit wallet posting — that ledger is keyed on `practitioner_id`;
  household members are retail buyers without one.
- No change to how *unpaid* members are re-billed (unchanged).

## Design (Approach A)

Paid orders are frozen by convention (every invoice-edit route guards
`if pay_status == "paid": return`). So the credit is recorded in a **dedicated,
informational column** rather than by editing `total_cents` / `adjustment_cents`
(which feed the invoice total and would re-open a paid, QBO-synced invoice).

### Data
- New column `orders.overpay_credit_cents INTEGER NOT NULL DEFAULT 0`, added via the
  existing `init_orders_table` ALTER-migration list. Flows to the console board
  automatically (`_row_to_dict` does `dict(row)` over `SELECT *`; `/api/orders`
  returns those dicts).

### Pure math — `combined_shipments.paid_member_overpay_cents(...)`
Lives beside `split_shipping_proportional` (the tested billing-math module), so it
is unit-testable without importing `app.py`.

```
paid_member_overpay_cents(paid_cents, total_cents, old_shipping_cents, new_share_cents):
    paid       = paid_cents or total_cents            # fallback for legacy paid rows
    fair_total = max(0, total_cents - old_shipping_cents + new_share_cents)
    return max(0, paid - fair_total)                  # never negative
```

Credit == what they paid minus their fair total. When `paid == total` (the normal
case) this reduces to `old_shipping − new_share` — exactly the shipping saving.
When combining did **not** lower their share, the result is 0.

### Setter — `orders.set_order_overpay_credit(cx, order_id, credit_cents)`
Idempotent REPLACE (clamped ≥ 0). Does not touch total/paid/shipping. Setting 0
clears a stale credit (self-heals if a later recalc raises the share back).

### Wiring — `app._recompute_combined_shipping`
Replace the blind `skipped:"paid"` branch with:
```
if pay_status == "paid":
    credit = C.paid_member_overpay_cents(paid_cents, total_cents, old_ship, share)
    O.set_order_overpay_credit(cx, id, credit)
    updates.append({..., "skipped":"paid", "shipping_cents":old_ship,
                    "fair_share_cents":share, "overpay_credit_cents":credit})
    continue
```
The paid order's `shipping_cents`/`total_cents` are still never written.

### UI — `static/console-orders.html`
1. Recalc summary alert: paid members with a credit show
   `$<old_ship> (paid) — credit $<credit> (overpaid)`.
2. Order card: when `overpay_credit_cents > 0`, a persistent badge
   `Credit $X (overpaid shipping)` next to the pay badge, so the credit is visible
   on the board (not just in the one-time alert).

## Idempotency & edge cases
- Re-running recalc recomputes and REPLACES the credit (never stacks).
- Combining only lowers a share, so paid members are never re-billed more; the
  clamp guarantees a non-negative credit.
- Legacy paid rows with `paid_cents == 0` fall back to `total_cents`.

## Testing (tests/test_combined_shipments.py, in-memory harness)
- `paid_member_overpay_cents`: normal saving, no-saving → 0, paid_cents fallback,
  never-negative.
- `set_order_overpay_credit`: sets, idempotent replace, clears on 0, leaves
  total/paid/shipping untouched.
- Regression: the existing `test_billing_never_changes_across_flow` still holds
  (paid total/pay_status untouched).

The `app._recompute_combined_shipping` wiring is verified by driving the live
console flow (no precedent for importing `app.py` in these unit tests).
