# Per-Kind Settlement Parity on Closed Tabs — Design

**Date:** 2026-07-16
**Status:** Approved (design), pending spec review
**Depends on:** QBO paid-only migration complete (PR #931–#951). Webhook-back booking (`webhook_stripe` `checkout.session.completed`, PR #947). `dashboard/qbo_sale.book_sale_on_payment`, `orders.set_order_payment`/`set_order_stripe_pi`.
**Owner:** Glen / RemedyMatch

## Problem

Paid-only checkout has two confirmation paths:
- **Redirect** — `/begin/checkout-return` (and `/practitioner/checkout-return`). Fires when the browser returns from Stripe.
- **Webhook** — `webhook_stripe` `checkout.session.completed`. Fires server-to-server, even if the tab is closed.

The redirect handler performs **full per-kind settlement**. The webhook-back block performs only **mark-paid + book the QBO receipt + stamp the PaymentIntent**. So a **closed tab** (customer closes the browser before the redirect) leaves an order paid and booked but with its per-kind side-effects **silently unsettled**:

| Kind | What a closed tab misses today |
|---|---|
| **biofield** | **Readiness seed** (`biofield_store.seed_paid`) — the gate that unlocks scan-booking. A paying client literally cannot book their scan. Plus loyalty points + referral credit. |
| **subscribe** (Subscribe & Save) | **The entire recurring subscription row** (`subscriptions.create`) is never created — paid once, no autoship. Plus points + referral. |
| **client** (practitioner dispensary) | Practitioner **wallet margin** (`wallet.earn_dropship_margin`), **dispensary-order record** (`record_dispensary_order`), **patient points**, ship-credit consume. |
| **retail / reorder / portal-reorder** | Loyalty **points** (earn/redeem, referrer rewards) + **referral credit** (`_settle_referral`). |

Already webhook-safe (no gap): the subscription/membership *product* kinds with their own `_fulfill_*` fulfillers (membership_product, prepay_term, continuous_care_monthly, masterclass, coach_sub, family_plan, biofield_trial), and mark-paid + receipt for any `qbo_lines_json` order.

**Root cause:** settlement is inline per-kind in the redirect handler; the webhook hand-rolls a 3-operation subset. The two definitions drift. This bug *is* "the webhook forgot what the redirect does."

## Design (shared settlement helper — Glen chose Approach A)

A single per-kind settlement definition that **both** paths call, so the closed-tab path settles exactly what the redirect does and the two can never drift again. Modeled on the `qbo_heal` module we just shipped: a small, unit-testable module that receives its settler functions as injected dependencies, so it is testable without importing `app.py`.

### Part 1 — the settlement module (`dashboard/order_settlement.py`, new)

```
settle_paid_order_effects(order, session, *, deps, kind=None) -> dict
```

Given a paid `order` (dict) + its Stripe `session` (dict), runs the complete **per-kind side-effect settlement** idempotently and returns a summary dict `{"kind": ..., "settled": [names...], "skipped": [...]}`. `deps` is a bundle (dataclass or plain object) of the existing settler callables, injected by `app.py`:

- `settle_points(order, order_ref)` → `_settle_order_points`
- `settle_referral(order, order_ref)` → `_settle_referral`
- `ensure_subscription(order, session)` → guarded subscription-row create (Part 2)
- `grant_group_bundle(order, session)` → group-bundle window/membership grant
- `settle_client(order, session)` → wallet margin + dispensary record + patient points + ship-credit consume
- `settle_biofield(order, session)` → readiness seed + care taster

Dispatch by `kind` (derived from order/session, matching the redirect's current gating). **The plan MUST reproduce the redirect's exact per-kind calls verbatim** — the grouping below is the intended shape, but each kind's settler membership is defined by what the redirect actually does today, extracted from the code, not by this summary:
- **storefront points/referral** (retail / reorder / portal-reorder / subscribe / biofield): `settle_points`, `settle_referral`.
- **subscribe:** `ensure_subscription`, `grant_group_bundle`.
- **client:** `settle_client` **only** — client has its *own* points path (`points.redeem`/`credit`) and no referral in the redirect, so it must **not** also go through the common `settle_points`/`settle_referral` (that would double-settle). `settle_client` encapsulates the client-specific wallet + dispensary + patient-points + ship-credit.
- **biofield:** `settle_biofield` (readiness seed + care taster) in addition to the common points/referral above.

Each settler call is wrapped so one failing settler logs + is recorded in `skipped` but does not abort the rest (best-effort per side-effect). The module does **not** touch mark-paid / receipt-booking / PI-stamp — those already fire correctly in each path; this module is exactly the per-kind side-effects the webhook omits.

**Kind derivation:** the helper receives `kind` explicitly. Both paths already know the kind (redirect from its metadata/gating; webhook from `session.metadata.kind` / the order's stored kind). The plan defines the single derivation both callers use so redirect and webhook classify an order identically.

**Why injected deps, not direct imports:** the settlers (`_settle_order_points`, `_settle_referral`, biofield/client/subscribe logic) live in `app.py` with app-level dependencies. Injecting them (as `qbo_heal` took `find_receipt`/`book`/`stamp`) keeps the module free of `app.py` and fully unit-testable, and keeps `app.py` from growing.

### Part 2 — subscription dedup guard

`subscriptions.create` is the **only** non-idempotent primitive (plain INSERT, no dedup key — a redirect refresh already double-creates today). Add `subscriptions.has_subscription_for_order(cx, order_ref)` (or equivalent), and wrap creation in `ensure_subscription` so a given paid order yields **at most one** subscription row. This closes both the pre-existing redirect-refresh double and the new redirect-vs-webhook double. TDD, RED first.

### Part 3 — wire both paths

- **Webhook** (`webhook_stripe` `checkout.session.completed` book-back, ~app.py:27338): after the receipt books, build `deps` and call `settle_paid_order_effects(order, session, deps=...)`. Best-effort — a settler exception must never 500 the webhook or block the receipt (wrap in the existing try/except). **This is the parity fix.**
- **Redirect** (`/begin/checkout-return`, ~app.py:9457–9880): replace the inline per-kind settlement calls with a single call to `settle_paid_order_effects(...)`, same `deps`. **Single definition, no drift.** The redirect keeps its own mark-paid/receipt/PI/response-rendering; only the per-kind side-effect calls move into the helper.

### Safety / idempotency

Every settler the helper calls is already keyed on `order_ref` and guards double-apply (verified by audit): points `has_entry(order_ref, reason)`, referral `points.credit`/`accrue_cash` keyed on `order_ref`, wallet `earn_dropship_margin` idempotent per `qbo_invoice_id`, biofield `seed_paid` COALESCE first-write-wins, care taster + group bundle PK-claim — plus the new subscription guard. So calling from both paths and re-running is safe: no double-award.

## Testing

- **Helper unit tests** (`dashboard/order_settlement.py`, mocked deps): for each kind, asserts exactly the right settlers are invoked (retail→points+referral; subscribe→+ensure_subscription+group_bundle; client→settle_client; biofield→+settle_biofield); a second call is safe (deps are idempotent); one settler raising is recorded in `skipped` and does not abort the others.
- **Subscription dedup** (`tests/test_subscriptions_dedup.py`): two `ensure_subscription` calls for the same `order_ref` → one row.
- **Closed-tab webhook parity** (extend `tests/test_webhook_back_booking.py`): per kind, a webhook-only completion settles the missing side-effect — biofield readiness seeded, subscribe row created, client wallet credited, retail points+referral credited — and is idempotent if the redirect later also runs.
- **Redirect regression** (characterization first): capture current `/begin/checkout-return` per-kind settlement behavior, then confirm it is unchanged after the swap to the helper.
- **Best-effort webhook:** a settler raising still returns 200 and still books the receipt.

## Out of scope

- The mark-paid split (redirect does points/receipt, webhook does mark-paid) is a pre-existing architectural asymmetry; the order still gets marked paid by the webhook, so it is not broken. Not changed here.
- `/practitioner/checkout-return` (wholesale/dropship/personal) has no per-kind settlement on return (margin/wallet handled at checkout-creation) — already symmetric with the webhook, nothing to add.

## Files

- **Create:** `dashboard/order_settlement.py`, `tests/test_order_settlement.py`, `tests/test_subscriptions_dedup.py`.
- **Modify:** `dashboard/subscriptions.py` (dedup guard), `app.py` (build `deps`; call helper from webhook + redirect; replace redirect inline settlement), `tests/test_webhook_back_booking.py` (extend).
