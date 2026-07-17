# QBO Paid-Only — Webhook-back the Sales-Receipt Booking (closed-tab) — Design

**Date:** 2026-07-16
**Status:** Approved (design, Narrow scope), pending spec review
**Depends on:** QBO paid-only migration complete. `qbo_sale.book_sale_on_payment` (idempotent atomic-claim) + `orders.find_order_by_external_ref`/`set_order_payment`/`set_order_stripe_pi` exist.
**Owner:** Glen / RemedyMatch

## Problem

The interactive Stripe-redirect checkout flows (retail/reorder/portal-reorder/subscribe/biofield/wholesale/dispensary/personal) book their QBO Sales Receipt + mark the order paid ONLY in the redirect return handler (`/begin/checkout-return`, `/practitioner/checkout-return`). If the customer's browser closes before the redirect (or the redirect is dropped), Stripe still collects the money — but the receipt is never booked and the order stays "unpaid" on the board. There is no reconcile poller anymore to catch it. So: **money collected, nothing in QBO, order stuck unpaid.**

(Off-session flows — founding-ship, subscription cron — book inline and are already redirect-independent. Trials/memberships are handled by the webhook's existing `_fulfill_*` calls.)

## Design (Narrow scope — chosen)

Add a server-to-server booking to the Stripe webhook so the receipt is booked + the order marked paid regardless of the browser. **Money-critical path only** (mark paid + book); per-kind settlement (points/referral/sub-row/biofield-readiness) still happens on the redirect — that's a pre-existing, secondary gap tracked separately.

In `webhook_stripe` (`app.py:~27275`), inside the `checkout.session.completed` branch, AFTER the existing `_fulfill_biofield_trial(session_id)` + `_fulfill_membership_product(session_id)` calls, add a general paid-only booking that mirrors the `/practitioner/checkout-return` paid-only branch exactly:

```python
        # Webhook-back the paid-only Sales-Receipt booking so a closed browser tab
        # (dropped redirect) can't leave money collected with no QBO receipt + an
        # order stuck unpaid. Guarded on qbo_lines_json (paid-only checkout orders
        # only — trials/memberships have none) and idempotent via the atomic claim in
        # book_sale_on_payment, so it never double-books with the redirect handler.
        try:
            from dashboard import stripe_pay as _sp2
            sess = _sp2.get_session(session_id)
            if sess.get("payment_status") == "paid":
                inv = (sess.get("metadata") or {}).get("invoice_id")
                if inv:
                    _wcx = _sqlite3.connect(LOG_DB); _wcx.row_factory = _sqlite3.Row
                    try:
                        _wo = _bos_orders.find_order_by_external_ref(_wcx, inv)
                        if _wo and _wo["qbo_lines_json"] and not _wo["qbo_sales_receipt_id"]:
                            _wpi = sess.get("payment_intent")
                            if _wpi:
                                _bos_orders.set_order_stripe_pi(_wcx, _wo["id"], _wpi)
                            _bos_orders.set_order_payment(
                                _wcx, _wo["id"], method="card",
                                amount_cents=int(sess.get("amount_total") or 0))
                            from dashboard import qbo_sale as _wqs
                            _wqs.book_sale_on_payment(
                                _wcx, dict(_bos_orders.find_order_by_external_ref(_wcx, inv)))
                    finally:
                        _wcx.close()
        except Exception as _we:
            print(f"[stripe-webhook] paid-only book-back failed: {_we!r}", flush=True)
```

Best-effort (wrapped) — a failure here must not 500 the webhook (Stripe would retry, which is fine, but never break the ack).

### Why this is safe / idempotent
- **Guard `qbo_lines_json` present + `qbo_sales_receipt_id` absent:** only paid-only checkout orders enter (trials/memberships/biofield-trial have no `qbo_lines_json` → no-op; already-booked orders skip). Same guard the redirect uses.
- **Atomic claim** in `book_sale_on_payment` (`qbo_sales_receipt_id` NULL→'PENDING') means the redirect and the webhook can race and still produce exactly ONE receipt.
- **`set_order_payment` only runs when not yet booked** (guarded), so it can't regress an already-paid/shipped order's fulfillment status.

## Testing
- **Closed-tab (the point):** a `checkout.session.completed` for a paid-only order whose redirect NEVER ran → the webhook marks the order paid AND books exactly one Sales Receipt.
- **Idempotent with redirect:** redirect already booked → the webhook is a no-op (guard on `qbo_sales_receipt_id`); webhook first → the later redirect is a no-op. Both together → one receipt, order paid once.
- **Non-checkout sessions:** a trial / membership_product / biofield-trial `checkout.session.completed` (no `qbo_lines_json` on the order, or no matching order) → the new block no-ops; the existing `_fulfill_*` calls are unaffected.
- **Never 500s the webhook:** a booking exception is swallowed; the webhook still returns 200.
- Regression: `test_opens`/webhook-stripe tests, paid-only checkout/return suites stay green.

## Out of scope (separate follow-on)
Full per-kind settlement parity on the webhook (points/referral/subscribe sub-row/biofield readiness on a closed tab). Would mean extracting the redirect handler's per-kind logic into a shared function called by both — a larger refactor. The stuck-`PENDING` reconciler sweep is also separate.

## Verify at spec/impl time
Confirm the Stripe webhook is live in prod and receives `checkout.session.completed` (it already fulfills trials/memberships, so it is) and that `STRIPE_WEBHOOK_SECRET` gating doesn't drop legitimate events.
