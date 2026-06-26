# Money Inline Actions (Sub-project C2) — Design Spec

**Date:** 2026-06-26
**Status:** Design approved. Second slice of sub-project **C** (console inline "act-where-shown"
actions); follows C1 (reorder → draft PO). C3 (cross-links) is a separate cycle.

## Goal

Two money actions, each where the item is shown:
- **B — Record payment on a receivable** (Money board → Receivables): mark a QBO invoice paid from the
  console (supports **partial and split payments** — multiple payments of different amounts/methods
  against one invoice until its balance reaches zero).
- **C — Confirm a claimed payment** (Orders board): when a customer has claimed they paid (Zelle/Wise),
  confirm it in one click using the already-stated method — no re-prompting for method + amount.

**Deferred (not C2):** the failed-charge "retry/contact" — the `stripe_failures` table is an
owner-alert log of checkout-*session-creation* errors (no PaymentIntent to re-attempt, no
customer/email link), so "retry" is mis-modeled. A real "retry a declined subscription" is a separate
future effort (needs the subscription decline captured with a customer link).

## Current state (reuse-first)

- **Action substrate:** `dashboard/finance.py` registers money actions via the
  `action(key=, module="money", title=, description=, risk_tier=, permission=, confirm_summary=)(...)`
  decorator (e.g. `finance.refund_order` = `MONEY_SEND, (OWNER, OPS, "va")`, with a
  `confirm_summary`; `finance.void_invoice` = `IRREVERSIBLE`). RBAC (`rbac.py`): `MONEY_SEND` →
  CONFIRM for OWNER/OPS (the dispatcher returns `needs_confirmation` on the first call; the frontend
  re-POSTs `confirmed:true`). `LOW_WRITE` → AUTO (no dialog).
- **B's QBO write exists:** `qbo_billing.record_payment(customer_id, amount_cents, invoice_id)`
  (`qbo_billing.py:241`) posts a QBO `Payment` of `amount_cents/100` linked to the invoice; it
  **skips only when the invoice balance is already ≤ 0**, so recording a *partial* amount reduces the
  balance and leaves the invoice open for the next (split) payment. `qbo_billing.get_invoice(invoice_id)`
  (`:196`) returns the QBO invoice incl. `CustomerRef.value`. The AR cache is `finance._cache` (cleared
  to refresh `open_invoices`). AR rows from `/api/finance/ar` carry `{id, doc, customer, email, total,
  balance, due_date, days_overdue}` (id = QBO invoice id).
- **C's order action exists:** `orders.record_payment` (`orders.py:521`, `LOW_WRITE, (OWNER,)`) →
  `set_order_payment(cx, order_id, method=, amount_cents=)` writes `status='new', pay_status='paid',
  pay_method, paid_cents, paid_at` to the local `orders` table (no QBO). A claimed order has
  `pay_status='claimed'`, `pay_method` already set, and `status` in `('proposed','confirmed')` (so the
  executor's `_PRE_FULFILL` guard passes). The `/api/orders` response serializes `pay_status`,
  `pay_method`, `total_cents`.
- **UI:** Money board Receivables tab = the `MoneyReceivables` IIFE in `static/console-money.html`
  (its `rowHtml()` renders Send-reminder/Void; its `act()` already handles the `needs_confirmation`
  re-POST + reloads on success). Orders board = `static/console-orders.html` `cardHtml()` (the
  claimed badge at the `pay_status==='claimed'` branch; the generic `recordPay(id)` re-prompts).

## Design

### Component 1 — B backend

- **Extend `qbo_billing.record_payment(customer_id, amount_cents, invoice_id, method=None)`** — when
  `method` is given, set `body["PrivateNote"] = "Console payment — method: " + method` on the QBO
  Payment (free-text memo; captures the method for split payments). Backward-compatible: existing
  callers pass no method. No other behavior change (still skips when balance ≤ 0).
- **New action `finance.record_payment`** in `dashboard/finance.py`, mirroring `finance.refund_order`:
  `action(key="finance.record_payment", module="money", title="Record payment",
  description="Record a customer payment against a QuickBooks invoice (partial/split supported).",
  risk_tier=MONEY_SEND, permission=(OWNER, OPS), confirm_summary=_record_payment_confirm_summary)(_record_payment_exec)`.
  - `_record_payment_exec(params, ctx)`: `invoice_id = params["invoice_id"]`; `amount =
    float(params.get("amount"))` (dollars) — reject ≤ 0; `method = params.get("method")`. Fetch the
    invoice via `qbo_billing.get_invoice(invoice_id)` → error if missing; `customer_id =
    inv["CustomerRef"]["value"]`. Call `qbo_billing.record_payment(customer_id, round(amount*100),
    invoice_id, method=method)`; `finance._cache.clear()`; return `{"ok": True, "invoice_id":
    invoice_id, "amount": amount}`.
  - `_record_payment_confirm_summary(params)`: returns e.g. `"Record $%.2f against invoice %s%s?" %
    (amount, invoice_id, (" via "+method) if method else "")` — shown in the MONEY_SEND confirm dialog.

### Component 2 — B frontend (`static/console-money.html`, `MoneyReceivables`)

Add a **"Record payment"** button to each AR row in `MoneyReceivables.rowHtml()` (next to
Send-reminder/Void). On click: prompt the **amount** (default = the row's `balance`; accept a smaller
number for a partial) and the **method** (a short prompt/picker — Zelle / Wise / Card / Check / Cash /
Other; default Zelle). Call the existing `MoneyReceivables.act(invoice_id, 'finance.record_payment',
{invoice_id, amount, method})`. Because the action is `MONEY_SEND`, `act()` already shows the
confirm-and-retry dialog and reloads the AR list on success — so after a partial/split payment the row
reappears with the reduced balance, ready for the next payment. (Use the existing row-action wiring;
do not JSON-stringify objects into an onclick — follow the row's current button pattern.)

### Component 3 — C frontend (`static/console-orders.html`, `cardHtml`)

When `o.pay_status === 'claimed'`, render a dedicated **"Confirm payment (<method>)"** button (the
method = `o.pay_method`). On click: a lightweight `confirm("Confirm " + o.pay_method + " payment for
order #" + o.id + "?")`, then call the existing `act(o.id, 'orders.record_payment', {method:
o.pay_method})` — no amount/method prompts. (Confirm the `orders.record_payment` executor defaults
`amount_cents` to the order total when not passed; if it does **not**, also pass `amount_cents:
o.total_cents`.) Keep the generic `recordPay()` path for non-claimed orders. `orders.record_payment`
is `LOW_WRITE` (no server dialog), so the client `confirm()` is the only guard — intended.

## Out of scope

- The failed-charge retry/contact (deferred — see Goal).
- Any change to refund/void/reminder, the QBO Payment beyond the optional method memo, or the orders
  fulfillment flow. C3 (cross-links).

## Dependencies

- `dashboard/qbo_billing.py` (`record_payment`, `get_invoice`), `dashboard/finance.py` (action
  registry + `_cache`), `dashboard/orders.py` (`orders.record_payment`), the BOS dispatch/RBAC, and
  the B1 Money board (`MoneyReceivables`) + Orders board.

## Testing (run via [reference_deploy_chat_local_tests])

- **`finance.record_payment` executor (unit, mock QBO — no network):** monkeypatch
  `qbo_billing.get_invoice` → `{"CustomerRef": {"value": "42"}, "Balance": "100"}` and
  `qbo_billing.record_payment` → a sentinel; assert the executor calls `record_payment` with
  `customer_id="42"`, `amount_cents=5000` for a `$50` partial, the `method` passed through, and returns
  `{"ok": True, …}`. A missing invoice (`get_invoice → None`) returns an error (no `record_payment`
  call); `amount ≤ 0` rejected. Assert the action is registered `MONEY_SEND, (OWNER, OPS)`.
- **`record_payment` method memo (unit, mock `_post`):** monkeypatch `qbo_billing._post` to capture the
  body; assert `method="Zelle"` sets `PrivateNote` containing "Zelle"; no method → no PrivateNote;
  balance ≤ 0 → returns without posting (split-safe / idempotent).
- **Render-verify (headless, per the render-verify lesson) — mocked:** the Receivables tab row shows a
  **Record payment** button; clicking it (amount + method prompts auto-answered) posts
  `finance.record_payment` with `{invoice_id, amount, method}` and, on a mocked `needs_confirmation`
  then `done`, reloads the AR list — **zero JS errors**. The Orders board shows a **Confirm payment
  (zelle)** button on a `pay_status:'claimed'` order; clicking it (confirm auto-accepted) posts
  `orders.record_payment` with `{method:'zelle'}` (no amount/method prompts) — zero JS errors.

## Rollout

Additive: one extended `qbo_billing.record_payment` (optional `method`), one new `finance.record_payment`
action, and two frontend buttons. No schema change, no feature flag. MONEY_SEND (record-payment) is
RBAC-gated with a confirm dialog; the claimed-confirm is LOW_WRITE with a client confirm.
