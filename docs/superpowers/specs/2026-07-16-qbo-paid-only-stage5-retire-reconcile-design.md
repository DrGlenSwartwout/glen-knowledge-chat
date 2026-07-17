# QBO Paid-Only Stage 5 — Retire the Reconcile Poller + Dead record_payment Calls — Design

**Date:** 2026-07-16
**Status:** Approved (design, Option A), pending spec review
**Depends on:** Stages 1–4 merged — no flow creates a QBO invoice anymore (only the `qbo_test_invoice` diagnostic). **Prerequisite:** the 6 legacy open invoices are VOIDED (operator-run; drains the 2 reconcile-relevant ones, 24439/24435).
**Owner:** Glen / RemedyMatch

## Problem

Now that every checkout/recurring flow is paid-only (books a Sales Receipt, never an invoice), the QBO **reconcile poller** (`dashboard/qbo_reconcile.py` + the `/api/console/reconcile-qbo` route) has nothing to do once the legacy in-flight invoices drain, and the two **return-handler `record_payment` calls** (invoice-apply) are dead — they only fire for legacy invoice orders (`cid` real), never for paid-only orders (`cid=""`). Retire the dead machinery.

**Scope (Option A — chosen):**
- Remove `dashboard/qbo_reconcile.py` and the `/api/console/reconcile-qbo` route (`app.py:5013-5043`) + its points-settle wrapper.
- Remove the two dead `if ... cid ...: record_payment(...)` blocks in `/begin/checkout-return` (`app.py:~9558-9562`) and `/practitioner/checkout-return` (`app.py:~25867-25884`).

**Explicitly KEPT (Option A, not Option B):**
- `qbo_billing.record_payment` (the function) — still used by the two *manual* tools.
- `finance.record_payment` action (console "record a payment against invoice #N").
- The **order-payments ledger** (`dashboard/order_payments.py:_push_payment`) — a live feature (#911). Untouched.

Rationale: this removes all *automatic* invoice-payment/reconcile machinery and the poller without disturbing the live payments-ledger feature. `record_payment` survives only as a manual operator escape hatch for a hand-made invoice — outside what the migration was about.

## Design

### 1. Delete the reconcile poller
- Remove `dashboard/qbo_reconcile.py`.
- Remove the `/api/console/reconcile-qbo` route (`console_reconcile_qbo`, `app.py:5013-5043`) and its `from dashboard import qbo_reconcile` import + the points-settle loop wrapper around it.
- Remove `tests/test_qbo_reconcile.py` (and any test importing `qbo_reconcile`).
- Grep for any other reference to `qbo_reconcile` / `reconcile_qbo_payments` / `list_open_qbo_orders` and remove.

### 2. Remove the two dead return-handler record_payment blocks
- `/begin/checkout-return` (`app.py:~9558-9562`): the `if cid and _kind != "in-house" and _kind not in (...): record_payment(cid, amount, inv)` block. This is legacy-invoice-only (paid-only has `cid=""`). Delete the block (and its now-unused `_qb_ret` import if local). The stripe-pi/points/referral/booking block (`if pi_id and _kind in (...)`) is SEPARATE and must remain untouched.
- `/practitioner/checkout-return` (`app.py:~25867-25884`): the `if inv and cid:` block (record_payment + `set_order_stripe_pi` for a legacy invoice). Delete it. The paid-only branch added in Stage 4 (`if inv: ... book_sale_on_payment ...`, guarded on `qbo_lines_json`) must remain — it handles the live path.

### 3. Keep the rest
`qbo_billing.record_payment`, `finance.record_payment`, `order_payments._push_payment` unchanged.

## Testing
- After removal, the app imports and starts (no dangling `qbo_reconcile` import).
- `/begin/checkout-return` and `/practitioner/checkout-return`: a PAID-ONLY order (card) still books exactly one Sales Receipt and marks the order paid (the paths that matter are unchanged); a session with a legacy shape (`cid` real) simply no-ops now (no record_payment) — acceptable since legacy invoices are voided/drained.
- The manual tools still work: `finance.record_payment` against a (hand-made) invoice still applies a QBO payment; `order_payments._push_payment` unchanged.
- Regression: the paid-only checkout/return suites (`test_begin_checkout_paid_only`, `test_practitioner_checkout_return_paid_only`, `test_biofield_checkout_paid_only`, `test_book_sale_on_payment`) stay green.
- Grep proves zero remaining references to the removed poller.

## Migration / prerequisite
- **Void the 6 legacy invoices first** (operator-run). That drains the reconcile-relevant orders so removing the poller can't strand a payment. The code change merges after the void is confirmed.
- After Stage 5, the only QBO-invoice code left is the `qbo_test_invoice` diagnostic + the manual `finance.record_payment`/ledger tools (both operator-gated).

## Out of scope (Option B, not chosen)
Removing `finance.record_payment` and reworking the order-payments ledger's QBO sync — a separate, larger project touching a live feature.
