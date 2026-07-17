# QBO Paid-Only Stage 4 (Wholesale + Dropship + Dispensary) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Convert the wholesale/dropship/dispensary `create_invoice` flows to paid-only QBO Sales Receipts, resolving the Wellness-Credit discount before booking.

**Architecture:** Reuse the Stage 1–3 machinery (`checkout_ref` token, `qbo_lines_json`, `book_sale_on_payment`, atomic claim). For the wallet-redeem flows, resolve the redemption up front keyed on the token (instead of `apply_invoice_discount` after `create_invoice`), persist a matching payload, drop the invoice, and book one Sales Receipt on payment (Stripe-return for card, operator-confirm for alt-pay). The dispensary flow already resolves its discount at creation.

**Tech Stack:** Python, Flask, SQLite, pytest, QuickBooks Online REST API, Stripe.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-16-qbo-paid-only-stage4-wholesale-dropship-design.md`.
- REUSE machinery — no new columns/helpers: `orders.set_order_qbo_lines`, `qbo_sale.book_sale_on_payment` (idempotent atomic-claim, best-effort, line-faithful).
- **Reference template (merged code):** the Stage 3 conversions — `git show 8d7c33d0 -- app.py` (cart Pattern I) and the return-handler booking block. For the charge basis, the pricing fixes are on main.
- Charge basis: wholesale/dropship charge = `subtotal − redeemed` (+ shipping if the quote has it); GET is absorbed — recorded on the order via `_ingest_order(get_cents=...)`, never charged, never a receipt tax line. Charge == booked Sales Receipt.
- Discount MUST be resolved before the receipt payload (a Sales Receipt is final). `wallet.redeem_for_order`/`redeem_for_module`, `wallet.earn_fee_free`, and `_pp.record_order` re-key from the QBO invoice Id → the `checkout_ref` token.
- Booking is best-effort (never breaks the charge/checkout).
- Run tests: `doppler run --config dev -- python3 -m pytest <file>` (default config = prd/DATA_DIR=/data breaks app-import collection). Never bare pytest, never the whole suite.
- Do NOT touch `biofield_local_app.py`/`biofield_report_html.py` (unrelated dirty WIP).
- Zero backfill; go-forward.

---

## Task 1: `wholesale_checkout.build_order` → paid-only + return-handler kind-list

**Files:**
- Modify: `dashboard/wholesale_checkout.py` — `build_order` (line 31)
- Modify: `app.py` — the wholesale route (~14028, `_ingest_order`/`_pp.record_order`) and the return-handler booking kind-list (`app.py:9565`)
- Test: `tests/test_wholesale_paid_only.py` (create)

**Interfaces:**
- Consumes: `orders.set_order_qbo_lines`, `qbo_sale.book_sale_on_payment`, `wallet.redeem_for_order`/`earn_fee_free`.

- [ ] **Step 1: Read** `build_order` (`dashboard/wholesale_checkout.py:31-79`) and the wholesale route (`app.py:~14005-14048`). Note: `create_invoice` (line 54), `redeem_for_order(pid, subtotal, invoice_id)` (57), `apply_invoice_discount(invoice_id, redeemed)` (59), `earn_fee_free(pid, charged, invoice_id)` (64), the `out` dict (66-78), and in the route `_pp.record_order(pid, invoice_id=out["invoice_id"])` + `_ingest_order(source="wholesale", external_ref=out["invoice_id"])` + `_stripe_checkout_url_for_order(out, ...)`.

- [ ] **Step 2: Write failing tests** (`tests/test_wholesale_paid_only.py`, DB-isolated — mirror `tests/test_checkout_cart_paid_only.py`'s `_isolate_db`):
  - Guard: `qbo_billing.create_invoice` and `apply_invoice_discount` monkeypatched to raise; `build_order` (or the route) makes NO invoice; `out["invoice_id"]` is a 32-hex token; `out["customer_id"]==""`.
  - Redeem fidelity: with a wallet balance, `redeemed > 0`, and the persisted `qbo_lines_json` has `discount_cents == redeemed`; `out["total"]` (the charge) == `subtotal − redeemed`.
  - `get_cents` recorded on the order; the order keyed on the token (source `wholesale`).

- [ ] **Step 3: Run → FAIL** (`create_invoice` guard fires).

- [ ] **Step 4: Convert `build_order`.** Rewrite the QBO block (lines 47-78):
  - `checkout_ref = _uuid()` (add `import uuid`; use `uuid.uuid4().hex`).
  - Keep the quote/lines/get_cents computation. DROP `find_or_create_customer` + `create_invoice` + `apply_invoice_discount`.
  - Resolve redeem up front keyed on the token: `redeemed = wallet.redeem_for_order(practitioner["id"], quote["subtotal_cents"], checkout_ref)`.
  - `charged = max(0, quote["subtotal_cents"] - redeemed)`; fee-free: `if method in ("zelle","wise"): fee_free = wallet.earn_fee_free(practitioner["id"], charged, checkout_ref)`.
  - Return `out` with `invoice_id=checkout_ref`, `customer_id=""`, `doc_number=""`, `total=round(charged/100.0, 2)`, plus `qbo_payload={"lines": lines, "discount_cents": redeemed, "tax_cents": 0}` (add a key so the route can persist it), `subtotal_cents`, `credit_redeemed_cents=redeemed`, `fee_free_credit_cents=fee_free`, `get_cents`, `method`.

- [ ] **Step 5: Wire the route** (`app.py:~14028`): after `out = _wc.build_order(...)`, persist the payload — `with _sqlite3.connect(LOG_DB) as _lcx: _bos_orders.set_order_qbo_lines(_lcx, out["invoice_id"], out["qbo_payload"])` (best-effort try/except). `_pp.record_order(pid, invoice_id=out["invoice_id"])` and `_ingest_order(source="wholesale", external_ref=out["invoice_id"], get_cents=out.get("get_cents",0), ...)` already read `out["invoice_id"]` (now the token) — confirm. The Stripe metadata (`_stripe_checkout_url_for_order`) carries `out["invoice_id"]` (token) + `customer_id=""`.

- [ ] **Step 6: Return-handler kind-list** (`app.py:9565`): `if pi_id and (cid or _kind in ("retail", "reorder", "portal-reorder", "subscribe")):` → add `"wholesale"` and `"client"`: `... _kind in ("retail", "reorder", "portal-reorder", "subscribe", "wholesale", "client")):`. This lets the shared block book `book_sale_on_payment` for wholesale/dispensary card payments. (Do NOT add these to the `record_payment` exclusion — the `if cid` guard already skips paid-only orders; unconverted invoice-based wholesale still needs `record_payment`.)

- [ ] **Step 7: Run** `doppler run --config dev -- python3 -m pytest tests/test_wholesale_paid_only.py tests/test_wholesale_checkout.py tests/test_checkout_cart_paid_only.py -v` → PASS.

- [ ] **Step 8: Commit** `git commit -m "feat(qbo): wholesale build_order paid-only; return handler books wholesale/client"`

---

## Task 2: `wholesale_checkout.build_module_order` → paid-only

**Files:** Modify `dashboard/wholesale_checkout.py` — `build_module_order` (line 82); its caller in `app.py` (~14102). Test: `tests/test_wholesale_module_paid_only.py`.

- [ ] **Step 1: Read** `build_module_order` (82-107) — `create_invoice` (87), `redeem_for_module(pid, slug, ...)` (95), `apply_invoice_discount` (98). Note `quote_module` (111+) is the paid-only-aware preview and already documents "redeemed only when the payment is recorded (a Sales Receipt)" — align with it.
- [ ] **Step 2: Write failing tests** — guard (no create_invoice), token key, `qbo_lines_json` `discount_cents == redeemed`, charge == `tuition − redeemed`.
- [ ] **Step 3: Run → FAIL.**
- [ ] **Step 4: Convert** with the Task-1 pattern: token, drop `create_invoice`+`apply_invoice_discount`, `redeemed = redeem_for_module(pid, slug, checkout_ref-keyed if the API takes a ref; else keep its own idempotency)`, build `qbo_payload` (single module line, `discount_cents=redeemed`), return `invoice_id=checkout_ref`/`total=(tuition−redeemed)/100`/`qbo_payload`. Persist + book at the caller like Task 1.
- [ ] **Step 5: Run + regression** (`tests/test_wholesale_checkout.py` module tests) → PASS.
- [ ] **Step 6: Commit** `git commit -m "feat(qbo): wholesale build_module_order paid-only"`

---

## Task 3: `dropship_checkout.build_dropship_order` → paid-only

**Files:** Modify `dashboard/dropship_checkout.py` — `build_dropship_order` (line 53); the dropship route (`app.py:~14210`). Test: `tests/test_dropship_paid_only.py`.

- [ ] **Step 1: Read** `build_dropship_order` (53-129) — same shape as wholesale: `create_invoice` (104), `redeem_for_order(pid, subtotal, invoice_id)` (108), `apply_invoice_discount` (110), `earn_fee_free` (114+), the `out` dict. And the route (`app.py:14181-14227`): `_ingest_order(source="dropship", external_ref=out["invoice_id"])`, `_stripe_checkout_url_for_order`.
- [ ] **Step 2: Write failing tests** — mirror Task 1 (guard, token key, `discount_cents==redeemed`, charge == subtotal−redeemed, get_cents recorded, source `dropship`).
- [ ] **Step 3: Run → FAIL.**
- [ ] **Step 4: Convert** with the identical Task-1 pattern (token, resolve redeem on token, drop create_invoice+apply_invoice_discount, `qbo_payload`, `out` fields). Persist the payload at the route + confirm `_ingest_order`/Stripe metadata carry the token. Return-handler already covers `kind="wholesale"` (from Task 1) — dropship route uses `_stripe_checkout_url_for_order` → kind `wholesale`; confirm.
- [ ] **Step 5: Run + regression** (`tests/test_dropship_checkout.py`, `tests/test_dropship_routes.py`) → PASS.
- [ ] **Step 6: Commit** `git commit -m "feat(qbo): dropship build_dropship_order paid-only"`

---

## Task 4: `dropship_checkout.build_client_order` (dispensary) → paid-only

**Files:** Modify `dashboard/dropship_checkout.py` — `build_client_order` (line 171); its caller (`app.py:~16028`). Test: `tests/test_dispensary_paid_only.py`.

**Note:** this flow already passes `discount_cents = redeem_cents + ship_credit_applied` at `create_invoice` (line 285) — the discount is pre-resolved, so this is a straight Pattern-I conversion (no redeem reorder).

- [ ] **Step 1: Read** `build_client_order` (171-300) — `create_invoice(cust, lines, discount_cents=redeem_cents + ship_credit_applied)` (285), points redeem (273), ship-credit (283), get_cents (268), the `out` dict; and the dispensary route (`app.py:~16010-16040`) metadata `kind="client"`.
- [ ] **Step 2: Write failing tests** — guard (no create_invoice), token key (source `dispensary`), `qbo_lines_json` `discount_cents == redeem_cents + ship_credit_applied`, charge == subtotal − that discount, points + ship-credit preserved, get_cents recorded.
- [ ] **Step 3: Run → FAIL.**
- [ ] **Step 4: Convert:** token; drop `create_invoice`; build `qbo_payload={"lines": lines, "discount_cents": redeem_cents + ship_credit_applied, "tax_cents": 0}`; `out` with `invoice_id=checkout_ref`, `customer_id=""`, charge == `subtotal − (redeem_cents + ship_credit_applied)`. Persist + book at the route (kind `client`, already in the Task-1 kind-list). Preserve the points/ship-credit/margin bookkeeping.
- [ ] **Step 5: Run + regression** (`tests/test_dispensary_*`, `tests/test_client_margin_credit.py`) → PASS.
- [ ] **Step 6: Commit** `git commit -m "feat(qbo): dispensary build_client_order paid-only"`

---

## Full-suite gate (end of plan)

- [ ] Run: `doppler run --config dev -- python3 -m pytest tests/test_wholesale_paid_only.py tests/test_wholesale_module_paid_only.py tests/test_dropship_paid_only.py tests/test_dispensary_paid_only.py tests/test_wholesale_checkout.py tests/test_dropship_checkout.py tests/test_checkout_cart_paid_only.py tests/test_book_sale_on_payment.py -v`
  Expected: PASS, or any failure also present on `main`.
- [ ] Post-deploy manual: a real wholesale/dropship/dispensary order (card + alt-pay) each produces one QBO Sales Receipt (no open balance) at `subtotal − credit`, no invoice; the Wellness-Credit redemption reflects on the receipt.

## Notes
- After Stage 4, the only remaining `create_invoice` caller is `qbo_test_invoice` (diagnostic). Stage 5 retires `qbo_reconcile.py` + `record_payment`.
- If `wallet.redeem_for_order`/`redeem_for_module`/`earn_fee_free`'s ref param is used for idempotency/audit, keying it on the token is correct; verify the signature and that the token is an acceptable ref (string).
