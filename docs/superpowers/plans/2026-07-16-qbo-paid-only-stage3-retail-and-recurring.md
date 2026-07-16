# QBO Paid-Only Stage 3 (Retail + Recurring) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the 6 remaining retail + recurring `create_invoice` sites to paid-only Sales Receipts and retire the reconcile poller's forward role, reusing Stage 2 machinery.

**Architecture:** Two patterns. **Pattern I (interactive Stripe redirect):** convert the checkout function (token re-key + drop `create_invoice` + persist `qbo_lines_json` + `customer_id=""`) and extend two kind-lists in the `/begin/checkout-return` handler so its existing generic booking/side-effect block fires for the new kinds. **Pattern II (off-session charge):** drop `create_invoice`(+`record_payment`) and book one Sales Receipt inline right after a successful `charge_off_session`.

**Tech Stack:** Python, Flask, SQLite, pytest, QuickBooks Online REST API, Stripe.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-16-qbo-paid-only-stage3-retail-and-recurring-design.md`.
- REUSE existing machinery — do NOT add columns/helpers: `orders.set_order_qbo_lines(cx, external_ref, payload)`, `orders.claim_sales_receipt_slot`, `orders.set_order_sales_receipt_id`, `qbo_sale.book_sale_on_payment(cx, order)` (idempotent, best-effort, line-faithful, atomic-claim).
- **Reference templates (merged code — read them):** Pattern I mirrors `begin_checkout` (commit `e01f8b09`, `git show e01f8b09 -- app.py`); Pattern II mirrors `_ship_founding_reservation`/the biofield block and `book_sale_on_payment`'s callers.
- Keep response/metadata FIELD names (`invoice_id`) with the token as value.
- Booking is best-effort — never break the checkout/charge path.
- Preserve cid-gated side-effects (`set_order_stripe_pi`, `_settle_order_points`, `_settle_referral`) for every converted interactive kind — pin with a spy test (the Stage 2 trap).
- Run tests: `doppler run --config dev -- python3 -m pytest <file>` (default doppler config = prd/DATA_DIR=/data breaks app-import collection). Never bare pytest, never the whole suite.
- Zero backfill; go-forward only.

---

## Task 1: Retire the reconcile poller's forward role (numeric-only filter)

**Files:**
- Modify: `dashboard/qbo_reconcile.py` — `list_open_qbo_orders` (~17-29)
- Test: `tests/test_qbo_reconcile.py` (extend, or create if absent)

**Interfaces:**
- Produces: `list_open_qbo_orders` returns only orders whose `external_ref` is an all-numeric QBO invoice id; token-based (paid-only) orders are excluded.

- [ ] **Step 1: Write the failing test**

Add to the reconcile test file (create `tests/test_qbo_reconcile.py` if none):

```python
import sqlite3
from dashboard import qbo_reconcile as R
from dashboard import orders as O


def _db():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    O.init_orders_table(cx)
    return cx


def test_poller_excludes_token_external_refs(tmp_path):
    cx = _db()
    O.upsert_order(cx, source="reorder", external_ref="24767", email="a@b.com", total_cents=100)   # legacy numeric invoice id
    O.upsert_order(cx, source="reorder", external_ref="3f6721cddeef4a1b9c0a1", email="b@b.com", total_cents=100)  # token (starts with digit)
    O.upsert_order(cx, source="portal-reorder", external_ref="c95ef29a9ccf4c55b5a3", email="c@b.com", total_cents=100)  # token (letter)
    rows = R.list_open_qbo_orders(cx)
    refs = {r["external_ref"] for r in rows}
    assert "24767" in refs
    assert "3f6721cddeef4a1b9c0a1" not in refs
    assert "c95ef29a9ccf4c55b5a3" not in refs
```

- [ ] **Step 2: Run to verify it fails**

Run: `doppler run --config dev -- python3 -m pytest tests/test_qbo_reconcile.py -v`
Expected: FAIL — the token starting with `3` matches the current `GLOB '[0-9]*'` and is returned.

- [ ] **Step 3: Tighten the filter**

In `dashboard/qbo_reconcile.py:list_open_qbo_orders`, replace the `external_ref GLOB '[0-9]*'` clause with an all-numeric + short guard (a QBO invoice id is all digits and short; a uuid hex token is 32 chars / has letters):

```python
        "AND external_ref NOT GLOB '*[^0-9]*' "   # all-numeric only (real QBO invoice id)
        "AND length(external_ref) < 20 "
```

(Keep the rest of the query identical.)

- [ ] **Step 4: Run to verify it passes**

Run: `doppler run --config dev -- python3 -m pytest tests/test_qbo_reconcile.py -v`
Expected: PASS. Existing reconcile tests still pass.

- [ ] **Step 5: Commit**

```bash
git add dashboard/qbo_reconcile.py tests/test_qbo_reconcile.py
git commit -m "fix(qbo): reconcile poller ignores token external_refs (Stage 3 forward-retire)"
```

---

## Task 2: `_checkout_cart` (kind `reorder`) → paid-only + return-handler kind-lists

**Files:**
- Modify: `app.py` — `_checkout_cart` (~25087) and the return-handler conditions at `app.py:9420` and `app.py:9426`
- Test: `tests/test_checkout_cart_paid_only.py` (create)

**Interfaces:**
- Consumes: `orders.set_order_qbo_lines`, `qbo_sale.book_sale_on_payment` (via the return block), the return handler's generic booking block.

- [ ] **Step 1: Read the reference + target**

Read `git show e01f8b09 -- app.py` (the begin_checkout paid-only conversion) as the template. Read `_checkout_cart` in full (`app.py:25087`+): note `create_invoice` (~25087), `external_ref=inv.get("Id")`, `_record_referral_if_any(_ref_ctx, email, inv.get("Id"))`, the `out={invoice_id,doc_number,customer_id,total}` dict, and the Stripe metadata built for `_stripe_checkout_url_for_reorder`.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_checkout_cart_paid_only.py` — mirror `tests/test_begin_checkout_paid_only.py` (DB-isolated `_isolate_db`): (a) guard — `qbo_billing.create_invoice` monkeypatched to raise, `_checkout_cart` (or its calling route) makes NO invoice; the returned `out["invoice_id"]` is a token; (b) the order (source `reorder`) is keyed on that token and has `qbo_lines_json` persisted. Use a real active reorder cart fixture (find one in `tests/test_begin_checkout_engine.py` or existing reorder tests).

- [ ] **Step 3: Run to verify fail** — `doppler run --config dev -- python3 -m pytest tests/test_checkout_cart_paid_only.py -v` → FAIL (create_invoice guard fires).

- [ ] **Step 4: Convert `_checkout_cart`**

Apply the Pattern I transformation (mirror begin_checkout `e01f8b09`): generate `checkout_ref = _uuid.uuid4().hex`; build `qbo_payload = {"lines": <the same lines passed to create_invoice>, "discount_cents": <same discount>, "tax_cents": 0}`; DROP the `create_invoice` call; set `external_ref=checkout_ref` in `_ingest_order`; `_record_referral_if_any(_ref_ctx, email, checkout_ref)`; `out = {"invoice_id": checkout_ref, "doc_number": "", "customer_id": "", "total": <charged dollars>}`; after `_ingest_order`, `set_order_qbo_lines(cx, checkout_ref, qbo_payload)`; set the Stripe metadata `invoice_id=checkout_ref` and `customer_id=""` (read `_stripe_checkout_url_for_reorder` to place these).

- [ ] **Step 5: Extend the return-handler kind-lists (covers reorder + portal-reorder + subscribe)**

In `app.py`, two edits:
- Line ~9420: `if cid and _kind != "in-house" and _kind not in ("biofield", "retail"):` → add the three kinds:
  `... and _kind not in ("biofield", "retail", "reorder", "portal-reorder", "subscribe"):`
- Line ~9426: `if pi_id and (cid or _kind in ("retail",)):` → add the three kinds:
  `if pi_id and (cid or _kind in ("retail", "reorder", "portal-reorder", "subscribe")):`

This makes the existing `set_order_stripe_pi`/points/referral/`book_sale_on_payment` block fire for these kinds; `book_sale_on_payment` is now live (no longer a no-op) because `qbo_lines_json` is persisted.

- [ ] **Step 6: Add the side-effect pinning test**

In the new test file, add a test that a `kind=="reorder"` checkout-return with a non-empty `pi_id` and a referral invokes `set_order_stripe_pi`, `_settle_order_points`, `_settle_referral`, and books exactly one Sales Receipt (spy/counter monkeypatches; mirror `test_begin_checkout_paid_only.py`'s pinning test).

- [ ] **Step 7: Run + regression**

Run: `doppler run --config dev -- python3 -m pytest tests/test_checkout_cart_paid_only.py tests/test_begin_checkout_paid_only.py -v`
Expected: PASS. Confirm no double-book and side-effects fire.

- [ ] **Step 8: Commit**

```bash
git add app.py tests/test_checkout_cart_paid_only.py
git commit -m "feat(qbo): _checkout_cart paid-only; return handler books reorder/portal-reorder/subscribe"
```

---

## Task 3: `api_client_portal_checkout` (kind `portal-reorder`) → paid-only

**Files:**
- Modify: `app.py` — `api_client_portal_checkout` (~20293)
- Test: `tests/test_portal_checkout_paid_only.py` (create)

**Interfaces:** the return-handler kinds are already covered by Task 2; this task is checkout-function-only.

- [ ] **Step 1: Read** `api_client_portal_checkout` in full (~20198-20206): `create_invoice(allow_online_pay=True)`, `external_ref=inv.get("Id")`, the `out={invoice_id,customer_id,doc_number,total}` dict, `_ingest_order(source="portal-reorder", external_ref=inv.get("Id"))`, and `_stripe_checkout_url_for_reorder(out, email)`.

- [ ] **Step 2: Write failing tests** — `tests/test_portal_checkout_paid_only.py`, DB-isolated: guard (no `create_invoice`), order (source `portal-reorder`) keyed on token, `qbo_lines_json` persisted, response `invoice_id` is the token.

- [ ] **Step 3: Run → FAIL.**

- [ ] **Step 4: Convert** `api_client_portal_checkout` with the same Pattern I transformation as Task 2 (token, drop create_invoice, persist qbo_lines, `external_ref=checkout_ref`, response + Stripe metadata carry the token, `customer_id=""`). The metadata `kind` for this flow must be `"portal-reorder"` (verify what `_stripe_checkout_url_for_reorder` sets and ensure the return handler sees `portal-reorder` — already in the Task-2 kind lists).

- [ ] **Step 5: Run + regression** — `doppler run --config dev -- python3 -m pytest tests/test_portal_checkout_paid_only.py tests/test_client_portal_routes.py -v` → PASS.

- [ ] **Step 6: Commit** — `git commit -m "feat(qbo): api_client_portal_checkout paid-only (portal-reorder)"`

---

## Task 4: `reorder_subscribe` (kind `subscribe`) → paid-only

**Files:**
- Modify: `app.py` — `reorder_subscribe` (~25340)
- Test: `tests/test_reorder_subscribe_paid_only.py` (create)

**Interfaces:** return-handler kind `subscribe` already covered by Task 2; the subscribe-specific block (`app.py:9469`) that writes the subscription row must keep working.

- [ ] **Step 1: Read** `reorder_subscribe` (~25295-25360): `create_invoice`, metadata `{"kind":"subscribe","invoice_id": inv.get("Id"), ...}` incl. the items/ship stashing, and the return-handler `if md.get("kind") == "subscribe" and pi_id:` block (`app.py:9469`) that writes the subscription row.

- [ ] **Step 2: Write failing tests** — DB-isolated: guard (no `create_invoice`), order keyed on token, `qbo_lines_json` persisted, metadata `invoice_id`=token; AND a test that the subscription row is still written on a `subscribe` return (spy on the subscription-create call).

- [ ] **Step 3: Run → FAIL.**

- [ ] **Step 4: Convert** with the Pattern I transformation; set metadata `invoice_id=checkout_ref`, `customer_id=""`, keep `kind:"subscribe"` and the items/ship stashing unchanged. Confirm the subscribe return block (9469) keys off metadata (not the invoice) so it is unaffected.

- [ ] **Step 5: Run + regression** — include `tests/test_subscribe_setup.py` and `tests/test_sub_cron_term_cap.py`. PASS.

- [ ] **Step 6: Commit** — `git commit -m "feat(qbo): reorder_subscribe paid-only (subscribe); sub row unchanged"`

---

## Task 5: `_ship_founding_reservation` (Pattern II — off-session inline)

**Files:**
- Modify: `app.py` — `_ship_founding_reservation` (~25228)
- Test: `tests/test_founding_ship_paid_only.py` (create)

**Interfaces:** Consumes `orders.set_order_qbo_lines`, `qbo_sale.book_sale_on_payment`. No return-handler involvement.

- [ ] **Step 1: Read** `_ship_founding_reservation` (~25218-25260): `create_invoice(allow_online_pay=False)`, `charge_off_session(...)`, `if res.get("status")=="succeeded":` → `_ingest_order(source="reorder", external_ref=res.get("id") or inv.get("Id"))`.

- [ ] **Step 2: Write failing tests** — DB-isolated, monkeypatch `charge_off_session` (return `{"status":"succeeded","id":"ch_1"}`) and `qbo_billing.create_invoice` to raise: (a) on success, NO `create_invoice`, order keyed on `res["id"]`, `qbo_lines_json` persisted, exactly ONE `book_sale_on_payment`/Sales Receipt; (b) on a FAILED charge (`status!="succeeded"`), no order and no booking; (c) re-run for the same charge books no second receipt (claim).

- [ ] **Step 3: Run → FAIL.**

- [ ] **Step 4: Convert** — drop `create_invoice`; build `qbo_payload` from the same lines/amount; keep `charge_off_session`; on success, after `_ingest_order` (external_ref stays `res["id"]`), `set_order_qbo_lines(cx, res["id"], qbo_payload)` then resolve the order and `book_sale_on_payment(cx, order)`. Best-effort wrap. Preserve any existing coaching/points bookkeeping in this function.

- [ ] **Step 5: Run + regression** — include the founding-reservation suite. PASS.

- [ ] **Step 6: Commit** — `git commit -m "feat(qbo): _ship_founding_reservation paid-only (off-session inline booking)"`

---

## Task 6: `cron_charge_subscriptions` ×2 (Pattern II — recurring off-session)

**Files:**
- Modify: `app.py` — `cron_charge_subscriptions` sites at ~32913 (source `membership`) and ~33011 (source `subscription`)
- Test: `tests/test_sub_cron_paid_only.py` (create)

**Interfaces:** Consumes `orders.set_order_qbo_lines`, `qbo_sale.book_sale_on_payment`.

- [ ] **Step 1: Read** both sites (~32895-32945 and ~32995-33045): each does `charge_off_session` → `if res.get("status")=="succeeded":` → `create_invoice` (+ likely `record_payment`) → `_ingest_order(source=..., external_ref=res.get("id") or inv_id)`.

- [ ] **Step 2: Write failing tests** — DB-isolated; monkeypatch `charge_off_session` (succeeded) and `qbo_billing.create_invoice`/`record_payment` to raise. For BOTH sources (`membership`, `subscription`): a successful recurring charge makes NO invoice/payment, order keyed on the charge id, `qbo_lines_json` persisted, exactly ONE Sales Receipt; a failed charge books none; re-running the same period books no second receipt.

- [ ] **Step 3: Run → FAIL.**

- [ ] **Step 4: Convert** both sites — drop `create_invoice`+`record_payment`; after a successful charge + `_ingest_order`, `set_order_qbo_lines(cx, <external_ref>, payload)` then `book_sale_on_payment(cx, order)` inline. Keep all non-QBO subscription/period bookkeeping. Apply the SAME transformation to both sites (they share the shape).

- [ ] **Step 5: Run + regression** — include `tests/test_sub_cron_term_cap.py` and membership-cron suites. PASS.

- [ ] **Step 6: Commit** — `git commit -m "feat(qbo): subscription cron books Sales Receipts per charge (paid-only)"`

---

## Full-suite gate (end of plan)

- [ ] Money-path + affected-flow regression (diff FAILED vs `main`):

Run: `doppler run --config dev -- python3 -m pytest tests/test_checkout_cart_paid_only.py tests/test_portal_checkout_paid_only.py tests/test_reorder_subscribe_paid_only.py tests/test_founding_ship_paid_only.py tests/test_sub_cron_paid_only.py tests/test_qbo_reconcile.py tests/test_begin_checkout_paid_only.py tests/test_biofield_checkout_paid_only.py tests/test_book_sale_on_payment.py tests/test_bos_finance.py -v`
Expected: PASS, or any failure also present on `main`.

- [ ] Post-deploy manual: a real reorder-cart and a real subscription charge each produce a QBO **Sales Receipt** (no open balance) and no invoice; confirm the reconcile poller no longer picks up new token orders. Local QBO 400s expected.
