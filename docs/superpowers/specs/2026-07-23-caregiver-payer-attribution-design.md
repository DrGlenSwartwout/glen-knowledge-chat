# Caregiver Payer Attribution — pay another adult's order, book it as your expense — Design

**Date:** 2026-07-23
**Status:** Approved in brainstorm with Glen 2026-07-23
**Repo:** deploy-chat

## Problem

A caregiver (real case: **Steve Fox**) wants to pay for an order that belongs to another **adult** household member (**Michael Hill**), and have that payment appear in **Steve's** portal as **Steve's** payment/expense — *not* in Michael's. The order itself (remedies, clinical record, fulfillment) must stay Michael's. Today the system cannot express this: every payment's identity is derived from the order's owner email, so paying for Michael's order books to Michael.

Both parties must control who-gets-what. These are two **capacitated adults**, so this is dual-consent (linked accounts), not guardian-controls-everything.

## Boundary vs. adjacent systems

- **`dashboard/family_plan.py`** is a *separate* concept: a recurring $147/mo household **entitlement** subscription that unlocks biofield via `covers(email)`. It is caregiver-pays-a-monthly-plan, not caregiver-pays-another-adult's-specific-order. This spec does **not** touch it and does not change `covers()`.
- **`dashboard/household.py`** already ships the directional caregiver→member link with member-controlled `share_consent` and caregiver-controlled `cc_enabled`, relationship words, and the **"a capacitated adult owns their own consent"** principle (see `2026-07-04-household-sharing-design.md`, `2026-07-17-household-caregiver-overlay-design.md`). This spec **reuses** that link and consent idiom rather than inventing a new authorization subsystem.
- **`dashboard/order_payments.py`** (the ledger, `2026-07-14-order-payments-ledger-design.md`) is where attribution lives. It is per-order today; this spec adds a payer identity to it.

## The three roles (the whole point)

Most of the system collapses these into one "client." This feature splits them:

- **Beneficiary / order owner** = Michael. Order, line items, clinical record, fulfillment stay his.
- **Payer** = Steve. The money and the expense line attach to him.
- **Portal viewer** = whoever holds the token.

Decision (confirmed with Glen): **payment → Steve, order → Michael.** Not "everything → Steve," not "duplicate in both."

## Data model

**Zero new tables.** Two columns on the shipped household link + two on the ledger. All additive `ALTER TABLE` (idempotent, mirroring the established pattern); every existing row keeps today's behavior with no backfill.

### 1. `household_members` — the standing authorization (member-controlled)

- **`pay_consent INTEGER DEFAULT 0`** — the **member** (Michael) authorizes this link's caregiver (`primary_email` = Steve) to pay the member's orders. Default **OFF** (money is more sensitive than the view/`share_consent` default of ON). Revocable at any time. Set **only from the member's own portal** — this satisfies the capacitated-adult-owns-consent rule: Steve can never self-authorize.
- **`pay_share_scope TEXT DEFAULT 'amount_only'`** — what Steve sees of Michael's payable orders: `amount_only` (default; total + status only) or `line_items` (itemized products + amounts, still **no** clinical/biofield/remedy-reasoning). Member-controlled, set alongside `pay_consent`.

Helpers in `dashboard/household.py` (mirroring `caregivers_for`/`viewable_members_for`):
- `can_pay(cx, payer_email, member_email) -> bool` — True iff an active link `(primary=payer, member=member)` has `pay_consent=1`. **The single enforcement predicate.**
- `payable_members_for(cx, payer_email) -> [ {member_email, pay_share_scope}, ... ]` — the members who have granted this payer pay-consent (drives Steve's "Orders I'm paying for" surface).
- `set_pay_consent(cx, *, payer_email, member_email, consent, share_scope)` — member-side write; asserts the caller token is the **member**.

**Self-pay guard:** reject `pay_consent` where `primary_email == member_email` (no-op that would only confuse the `COALESCE` read).

### 2. `order_payments` — the payer identity (attribution seam)

- **`payer_email TEXT`** (nullable) and **`payer_person_id`** (nullable). **NULL = "payer is the order's owner"** → every existing row and every ordinary self-paid order is unchanged. A non-NULL `payer_email` re-homes the money.

`add_payment(cx, order_id, amount_cents, method, ..., payer_email=None, payer_person_id=None)` — new optional params; when present, stamp them onto the row. New-row insert uses `RETURNING id` (Postgres-portable; no `lastrowid`).

## The attribution seam (the one surgical read change)

Today the payments/expenses read (`order_payments.py:57`, joining `orders` and reading `o.email`) filters by `o.email = ?`. Change it to:

```sql
COALESCE(op.payer_email, o.email) = ?
```

This one predicate does both jobs and is backward-compatible by construction:
- Rows Steve paid (`payer_email = Steve`) → appear in **Steve's** expenses, drop out of Michael's.
- Rows with `payer_email` NULL or = Michael → stay Michael's.
- Every existing row is `payer_email` NULL → `COALESCE` = `o.email` → **identical to today**.

The same `COALESCE(payer_email, o.email)` filter applies to the client-hub invoice/expense surfaces `_published_invoices_for` (`app.py:40918`) and `_past_invoices_for` (`app.py:40940`) so "past invoices" totals reconcile with the payments view.

## Payment flow

Thread an **explicit** `payer_email` from initiation to the ledger — never inferred.

1. **Steve's entry point.** A scoped "Orders I'm paying for" block in Steve's portal, built from `payable_members_for(Steve)` → their unpaid orders. He picks Michael's order and pays; the action carries `payer_email = Steve`, `order_id = Michael's order`.
2. **Enforcement point = checkout initiation.** The caregiver-pay endpoint calls `household.can_pay(Steve, Michael)`; if false (no consent / revoked), it refuses to start a payer-tagged checkout. After initiation, the metadata is the immutable snapshot.
3. **Stripe path.** `create_checkout_session(...)` (`stripe_pay.py:64`) sets `metadata.payer_email = Steve`, `metadata.caregiver = 1` (alongside `order_id`/`invoice_id`), and `customer_email = Steve` so **the Stripe receipt goes to Steve**. The Stripe return handler (`app.py:10883`) **and** the `/webhook/stripe` dispatch both read `metadata.payer_email` and pass it into `add_payment`. Idempotent claim on the session id (mirroring `family_plan`'s `session_id` PK claim) prevents double-delivery from redirect + webhook.
4. **Manual / Zelle path.** The console `api_order_payments_add` form (`app.py:42455`) gains an optional "paid by (caregiver)" selector, populated from `payable_members_for`/`can_pay` for that order; when set, it stamps `payer_email` the same way.
5. **Two email streams (must not collapse):** payment **receipt → payer (Steve)**; order confirmation / shipping notifications → **beneficiary (Michael)** — fulfillment is his.

## Portal display seams

- **Michael's portal (beneficiary).** Orders block is unchanged in sourcing — the order is his (`orders.email = Michael`), so he still sees it (`portal_view.py:76`). Add a **per-payment** "Paid by caregiver (Steve)" badge when a payment on the order carries `payer_email` ≠ his (per-payment, so a split reads "partially paid by Steve"). His expense total excludes those rows via the `COALESCE` filter. His biofield/points/consult data is untouched.
- **Steve's portal (payer).** A dedicated **thin** block from a purpose-built query returning only `{beneficiary_name, order_id, amount, pay_status, optional line_items}`. **Steve is never routed through `get_portal_view(Michael)`** (`portal_view.py:322`) — that function surfaces the whole email-keyed world (biofield, consult, points). The firewall is structural: a different, narrow query that *cannot* return clinical columns, not a permission bolted onto Michael's view. `pay_share_scope` gates itemization on this block **and** on the invoice/pay page Steve opens.

## Authorization & revocation rules

- **Beneficiary-only authorization** is the security anchor. Only Michael (member) can set `pay_consent`, from his own portal. Steve can request; nothing stamps until Michael consents. Attributing a payment to Steve also removes it from Michael's expense view, so that action must require Michael's consent — and it does.
- **No household-link *creation* barrier beyond what exists**, but `pay_consent` lives on a caregiver→member link; the operator/portal flow that creates the link is unchanged. No new gate on TOS/`covers()` — paying an order never grants entitlement or clinical view.
- **The public per-order pay-link is not weaponizable.** Holding `create_order_invoice_token` (`practitioner_portal.py:465`) lets anyone *pay*, but the payment books to the order owner unless a payer-tagged caregiver checkout (gated by `can_pay`) supplied `payer_email`.
- **Revocation is non-retroactive.** Michael sets `pay_consent=0` → future payer-tagged checkouts are refused at initiation; already-recorded ledger rows are untouched (real money moved). The revoke-mid-checkout race resolves to Steve: a checkout validly initiated under consent carries the `payer_email` snapshot in metadata, and the webhook stamps from metadata regardless of later consent state.

## Edge cases (decided rules)

1. **Refunds follow the payer.** A refund ledger row (`kind=refund`) copies the original payment's `payer_email` so it nets against Steve's expenses.
2. **Split / mixed payers.** Already supported (multiple ledger rows per order). Badge is computed per-payment.
3. **Idempotency intact.** Idempotency key stays `order_id + external_ref`; `payer_email` is not part of it, so a retried webhook can't double-write or flip attribution.
4. **Identity merge.** If Michael's email is merged/changed, the merge routine must remap `household_members` (already email-keyed) and any `order_payments.payer_email` alongside `orders.email`. Flag in the identity-merge path.
5. **Self-pay guard.** Reject `pay_consent` where payer == member.
6. **Reconciliation.** Money still ties to `order_id`/`external_ref`; `payer_email` is additive display metadata, so Rae's reconciler is unaffected.

## Testing (TDD — tests first per seam)

- **Attribution:** `add_payment` with `payer_email` stamps the row; without → NULL.
- **The `COALESCE` seam, verified by behavior + dual-store:** seed a Michael-owned order with a Steve payment; assert it **appears in Steve's** expense query **and is absent from Michael's**; a NULL-payer row appears in Michael's (backward-compat).
- **Gating:** a payer-tagged checkout is refused when `can_pay` is false (no consent / revoked); allowed when true.
- **Revocation non-retroactive; split payments; refund-follows-payer** each get a test.
- **Firewall:** assert Steve's caregiver query *cannot* surface biofield/consult/points columns — structurally, not just empty in the fixture.
- **Mutation-test the guards:** flip `COALESCE` back to `o.email` → Michael's-exclusion test must go **red**; drop the `can_pay` check → a "Steve stamps payer with no consent" test must go **red**.
- **Caller check:** confirm the new `payer_email` param has a live caller (the caregiver flow), not a dormant argument.

## CI & rollout hygiene

- Tests must **not** send live email — stub the receipt path. **Pin the catalog** (`$DATA_DIR` strips `products.json` in the full suite). New-column/new-row inserts use **`RETURNING id`**, not `lastrowid` (Postgres adapter). App-importing tests need dummy `OPENAI`/`PINECONE` keys.
- Gate the whole surface behind a Doppler flag (`CAREGIVER_PAY_ENABLED`), **ship dark**. Render-verify Steve's block + Michael's badge with a synthetic-payload headless harness, **then** flip — remembering a flag-flip is two deploys and flags read at startup, so it needs a Render restart; verify via the live `/api/portal/<token>/view`.

## Out of scope (YAGNI for v1)

- Per-order (vs per-relationship) share-scope granularity — one `pay_share_scope` per link is enough.
- `expense_target` override (expense always books to the payer — that's the feature).
- Caregiver paying via a standalone shadow order (rejected: double-entry / double-count risk).
- Any change to `family_plan.py`, `covers()`, or entitlement.
- SMS/other notification routing for caregiver payments.

## Key file seams (for the implementation plan)

- `dashboard/household.py` — add `pay_consent`/`pay_share_scope` columns + `can_pay`/`payable_members_for`/`set_pay_consent`.
- `dashboard/order_payments.py:216` `add_payment` (+payer params); `:57` read view (`COALESCE`).
- `dashboard/stripe_pay.py:64` `create_checkout_session` metadata/customer_email; `app.py:10883` Stripe return; `/webhook/stripe` dispatch.
- `app.py:42455` `api_order_payments_add` (manual payer selector); `app.py:40918`/`:40940` invoice/expense reads (`COALESCE`).
- `dashboard/portal_view.py:76`/`:322` (Michael's orders block badge; Steve's thin block is a *new* query, not this path); `dashboard/portal_identity.py` for token→identity on the new endpoints.
- New endpoints: caregiver-pay initiate (`POST /api/portal/<token>/caregiver-pay`), member pay-consent toggle (`POST /api/portal/<token>/pay-consent`).
