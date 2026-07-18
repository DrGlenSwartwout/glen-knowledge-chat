# Membership-on-Invoice — Design Spec

**Date:** 2026-07-17
**Status:** Draft for review
**Author:** Glen + Claude

## Problem / Motivation

Manual membership enroll grants the paid-member flag **immediately, before any payment**.
A comped or provisional enroll therefore prices someone as a paid member who has not paid
(this happened with Dana Tamraz: member pricing applied, membership never collected). The
undo shipped 2026-07-17 as `POST /api/console/membership/revoke`, but the real fix is to
stop granting-before-payment entirely.

The desired flow: a customer can **add a group-coaching membership to their invoice**. While
the membership sits on the (unpaid) invoice, the product lines price at the **member rate
provisionally** — computed from the cart, not from a persisted flag. The real membership
grant fires **only when the invoice is paid**. An "add membership & save" control shows the
economics (gross fee, net add, net savings) so the offer sells itself.

## Requirements (confirmed with Glen)

1. **Both surfaces:** staff order editor (`order-new.html`) and the customer-facing invoice
   (`/invoice/<token>`, `/api/invoice/<token>`).
2. **Configurable tier(s):** which tier(s) the button offers is driven by config; default
   `month` ($99 one-time, grant-only, no auto-renew). Tiers live in
   `dashboard/membership_products.py` (`month`, `year_monthly`, `year_prepay`).
3. **Auto-flip pricing:** adding the membership line re-prices the products to member rate;
   removing it reverts — **unless the buyer is already a paid member** (`owns_group`), in
   which case pricing is member either way and the offer is hidden.
4. **Lock-in on payment:** the real grant is written when the invoice reaches **fully paid**
   (balance $0) — the same signal that releases fulfillment/shipping, so a membership never
   grants before it is covered. A partial payment leaves the membership provisional. (Glen's
   framing: "grant on payment, since products ship only on full payment.")
5. **Cost framing:** show **gross** (tier price), **net add** (gross − product savings, floored
   at 0), and **net savings** (the product savings the membership unlocks). Example (Dana,
   pre-new-product): `$99 membership · $55.32 saved on this order · $43.68 more`.
6. **Customer invoice shows payments + balance:** the sent invoice lists payments made and the
   remaining balance alongside the updated product list and the join-membership option.

## Design

### 1. Membership as a line item

Represent the chosen tier as a reserved order line — slug `membership:<tier>` (e.g.
`membership:month`), priced at the tier's `price_cents`, carrying a `kind: "membership"`
marker. It:

- appears on the invoice and adds to the total like any line,
- is **excluded** from FF / volume-discount math (it is not a product),
- is the single source of truth for "this cart contains a pending membership."

Add/remove is a toggle that inserts or deletes this one line. At most one membership line per
order (adding replaces any existing one, e.g. switching tiers).

### 2. Provisional member pricing

The pricing pass changes its member test from:

```
member = _is_paid_member(email)
```

to:

```
member = _is_paid_member(email) or cart_has_membership_line(lines)
```

`cart_has_membership_line` is a pure function of the submitted lines, so `price-preview`,
the staff reprice (`/api/orders/<id>/edit`), and the customer invoice all compute identical
results. No persisted state changes until payment. Already-paid members are unaffected
(the left side is already true).

### 3. The add/remove control + net math

Reuses `price-preview`, which already returns `savings_cents` per line. For the cart:

- `gross_cents` = tier `price_cents`
- `savings_cents` = Σ over product lines of (list_cents − member_effective_cents)
- `net_add_cents` = max(0, gross − savings)
- `net_savings_cents` = savings

Rendering:

- Savings < gross → "Add a month of coaching for **$NET_ADD** more (you save $SAVINGS on
  this order)."
- Savings ≥ gross → "Add a month of coaching — **effectively free** / you come out $X ahead."

The control appears on both surfaces. On the customer invoice, toggling it live re-prices the
product lines and updates the amount due before payment.

### 4. Lock-in on payment

Hook the existing "order reached fully-paid" transition (the same path that drives
fulfillment). On that event, if the order carries a `membership:<tier>` line and the buyer is
not already a member:

- call the existing `_grant_membership(cx, email, grant_days(tier), tier_source)`,
- idempotent: guard on a per-order marker so redelivery / re-runs never double-grant,
- record a journey event for audit.

No grant is written on add, on preview, or on partial payment.

### 5. Config

A small config surface (Doppler var or a console setting row) lists the offered tier key(s);
default `["month"]`. The button reads this to decide which tier(s) to present. Changing the
offer is a config change, not a deploy.

### 6. Surfaces

- **Staff editor (`order-new.html`):** an "Add membership" control near the pricing section;
  shows the same gross/net-add/net-savings; toggling reprices the preview.
- **Customer invoice (`invoice.html` / `/api/invoice/<token>`):** the join-membership control
  plus the existing payments/balance panel. The invoice must render:
  updated product list → payments made (ledger rows) → balance due → join-membership offer.

### 7. Guards & edge cases

- **Already a paid member** (`owns_group`): offer hidden; pricing already member; adding a line
  must not double-grant.
- **Hard-override lines** (e.g. Dana's frozen #66 lines): explicit per-line overrides do **not**
  auto-flip with the membership toggle — by design, overrides are sticky. To let a whole order
  respond to the toggle, its lines must be non-override (float). See Dana prerequisites.
- **Partial payment:** membership stays provisional; no grant until balance $0.
- **Refund / void of the membership line after grant:** out of scope for v1 — handle manually
  with the `revoke` endpoint. (A future version may auto-revoke on membership-line refund.)
- **Tier switch:** replacing the membership line re-runs the net math; no grant side effects
  pre-payment.

## Dana as the first real use — prerequisites

1. **Clear the redundant overrides on order #66.** The 6 FF lines were frozen at $69.97 while
   she was still flagged a member. Now that she is a revoked non-member they compute to $69.97
   as floats too — but frozen they will not flip to member pricing if she accepts the offer.
   When the new product is added, clear those overrides so the whole order floats.
2. **Record her payments on the ledger** so the invoice shows "payments made + balance."
   Her card $222.91 + Zelle $131 + $88 are legacy/QBO only. Recording the card in the ledger
   pushes to QBO invoice 24455 and double-counts against the card still applied to duplicate
   invoice 24457 — so this is gated on the QBO duplicate-invoice delete (Rae's reconcile)
   landing first. After that: record all three; invoice reflects true paid + balance.

## Out of scope (v1)

- Auto-revoke on membership-line refund/void (manual via `/api/console/membership/revoke`).
- Changes to recurring auto-charge behavior for `year_monthly`.
- Multi-membership / gifting a membership to another email from the invoice.

## Open decisions — resolved

| Decision | Choice |
|---|---|
| Surface | Both staff editor and customer invoice |
| Tier(s) offered | Configurable; default `month` |
| Lock-in trigger | Invoice fully paid (= fulfillment signal) |
| Cost framing | Show gross + net add + net savings |
