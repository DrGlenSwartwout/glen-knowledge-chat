# Webhook-back the Paid-Only Booking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Book the QBO Sales Receipt + mark the order paid on the Stripe webhook (`checkout.session.completed`), so a closed browser tab (dropped redirect) can't leave money collected with no receipt and an unpaid order.

**Architecture:** One best-effort, idempotent block added to `webhook_stripe`, mirroring the `/practitioner/checkout-return` paid-only branch. Guarded on `qbo_lines_json` present + `qbo_sales_receipt_id` absent; the atomic claim in `book_sale_on_payment` guarantees exactly one receipt even if the redirect also fires.

**Tech Stack:** Python, Flask, SQLite, pytest, Stripe.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-16-qbo-webhook-back-booking-design.md`.
- Narrow scope: mark-paid + book only. Do NOT replicate per-kind settlement (points/referral/sub-row/biofield-readiness) — out of scope.
- Idempotent: never double-book with the redirect handlers (guard on `qbo_sales_receipt_id`, atomic claim in `book_sale_on_payment`). `set_order_payment` runs ONLY inside the not-yet-booked guard (so it can't regress a paid/shipped order).
- Best-effort: a booking exception must NOT 500 the webhook — swallow + log, still return 200.
- Only paid-only checkout orders (those with `qbo_lines_json`) are booked; trials/memberships/biofield-trial (no `qbo_lines_json`) must no-op and the existing `_fulfill_biofield_trial`/`_fulfill_membership_product` calls must be untouched.
- Do NOT touch `biofield_local_app.py`/`dashboard/biofield_report_html.py` (unrelated dirty WIP).
- Run tests: `doppler run --config dev -- python3 -m pytest <file>` (never bare pytest, never whole suite).

---

## Task 1: Add the webhook-back booking block

**Files:**
- Modify: `app.py` — `webhook_stripe` (~27275), inside the `if ... == "checkout.session.completed":` branch, after the existing `_fulfill_*` calls.
- Test: `tests/test_webhook_back_booking.py` (create)

**Interfaces:** Consumes `stripe_pay.get_session`, `orders.find_order_by_external_ref`/`set_order_payment`/`set_order_stripe_pi`, `qbo_sale.book_sale_on_payment`. All exist.

- [ ] **Step 1: Read** `webhook_stripe` (`app.py:~27275`) — find the `if (event or {}).get("type") == "checkout.session.completed":` block, `session_id`, and the existing `_fulfill_biofield_trial(session_id)` + `_fulfill_membership_product(session_id)` calls. Read the `/practitioner/checkout-return` paid-only branch (~`app.py:25845-25868`) as the template.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_webhook_back_booking.py` (DB-isolated — monkeypatch `app.LOG_DB` to a tmp db, seed orders via `dashboard.orders`). Monkeypatch `stripe_pay.verify_webhook` (or `STRIPE_WEBHOOK_SECRET`) so the event parses, and `stripe_pay.get_session` to return the session dict. Monkeypatch `_fulfill_biofield_trial`/`_fulfill_membership_product` to no-op so tests isolate the new block. Cover:

```python
# 1. Closed-tab: paid-only order, redirect never ran -> webhook marks paid + books ONE receipt.
def test_webhook_books_paid_only_order_when_redirect_missed(...):
    # seed order source="funnel", external_ref="tok1", qbo_lines_json set, qbo_sales_receipt_id NULL, pay_status unpaid
    # get_session -> {"payment_status":"paid","metadata":{"invoice_id":"tok1"},"amount_total":7000,"payment_intent":"pi_1"}
    # spy create_sales_receipt (via qbo_billing) -> count
    # POST /webhook/stripe with a checkout.session.completed event
    # assert order pay_status == "paid" AND create_sales_receipt called exactly once (qbo_sales_receipt_id set)

# 2. Idempotent: order already has qbo_sales_receipt_id -> webhook books NOTHING (no 2nd receipt), does not touch fulfillment status.
def test_webhook_noop_when_already_booked(...): ...

# 3. Non-checkout: order has no qbo_lines_json (or no matching order for the token) -> new block no-ops; _fulfill_* still callable.
def test_webhook_noop_for_non_paidonly_session(...): ...

# 4. Never 500s: get_session/book raises -> webhook still returns 200.
def test_webhook_swallows_booking_error_returns_200(...): ...
```

(Find the webhook route path — grep `@app.route` near `def webhook_stripe`; use it for the test client POST. Match the event JSON shape the handler parses.)

- [ ] **Step 3: Run → FAIL**

Run: `doppler run --config dev -- python3 -m pytest tests/test_webhook_back_booking.py -v`
Expected: test 1 fails (webhook doesn't book yet); others may pass trivially pre-change.

- [ ] **Step 4: Add the block**

In `webhook_stripe`, inside the `checkout.session.completed` branch, AFTER the `_fulfill_biofield_trial(session_id)` + `_fulfill_membership_product(session_id)` calls, insert (match indentation; confirm `_sqlite3`, `LOG_DB`, `_bos_orders` names in this file):

```python
        # Webhook-back the paid-only Sales-Receipt booking so a closed browser tab
        # (dropped redirect) can't leave money collected with no QBO receipt + an
        # order stuck unpaid. Guarded on qbo_lines_json (paid-only checkout orders
        # only) + idempotent via book_sale_on_payment's atomic claim -> never
        # double-books with the redirect handlers.
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

- [ ] **Step 5: Run → PASS**

Run: `doppler run --config dev -- python3 -m pytest tests/test_webhook_back_booking.py -v`
Expected: PASS (4 passed).

- [ ] **Step 6: Regression**

Run: `doppler run --config dev -- python3 -m pytest tests/test_opens.py tests/test_begin_checkout_paid_only.py tests/test_practitioner_checkout_return_paid_only.py tests/test_book_sale_on_payment.py -v`
Expected: PASS (webhook still fulfills trials/memberships; redirect booking unchanged; no double-book).

- [ ] **Step 7: Commit**

```bash
git add app.py tests/test_webhook_back_booking.py
git commit -m "feat(qbo): webhook-back the paid-only Sales-Receipt booking (closed-tab)"
```

---

## Full-suite gate

- [ ] Run: `doppler run --config dev -- python3 -m pytest tests/test_webhook_back_booking.py tests/test_begin_checkout_paid_only.py tests/test_practitioner_checkout_return_paid_only.py tests/test_wholesale_paid_only.py tests/test_book_sale_on_payment.py -v`
  Expected: PASS, or any failure also present on `main`.
- [ ] Post-deploy: verify the Stripe webhook still 200s and the block fires (a real paid-only checkout where you close the tab before redirect → the order flips to paid + a Sales Receipt appears in QBO within seconds).

## Notes
- Closes the "M3 webhook-back the booking" follow-on. The stuck-`PENDING` reconciler sweep and full per-kind settlement parity remain separate follow-ons.
