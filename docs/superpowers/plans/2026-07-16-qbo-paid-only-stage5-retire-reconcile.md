# QBO Paid-Only Stage 5 (Retire Reconcile Poller) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Delete the now-dead QBO reconcile poller and the two dead return-handler `record_payment` calls, keeping the live manual tools.

**Architecture:** Pure removal (Option A). No behavior change for any live flow — the reconcile poller had nothing to reconcile (legacy invoices voided/drained), and the `record_payment` blocks only fired for legacy invoice orders (`cid` real), never paid-only (`cid=""`).

**Tech Stack:** Python, Flask, pytest.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-16-qbo-paid-only-stage5-retire-reconcile-design.md`. **Prerequisite met:** the 6 legacy invoices are voided ($0) and their orders cancelled (drain complete).
- KEEP (do NOT remove): `qbo_billing.record_payment` (the function), `finance.record_payment` (console action), `dashboard/order_payments.py` (`_push_payment` ledger). These are live manual tools.
- Do NOT touch the paid-only booking blocks in either return handler (the `if pi_id and _kind in (...)` block in `/begin/checkout-return`; the `if inv: ... book_sale_on_payment` paid-only branch in `/practitioner/checkout-return`).
- Do NOT touch `biofield_local_app.py`/`dashboard/biofield_report_html.py` (unrelated dirty WIP).
- Run tests: `doppler run --config dev -- python3 -m pytest <file>` (never bare pytest, never whole suite).

---

## Task 1: Delete the reconcile poller + the two dead record_payment blocks

**Files:**
- Delete: `dashboard/qbo_reconcile.py`, `tests/test_qbo_reconcile.py`
- Modify: `app.py` — remove `console_reconcile_qbo` route (~5013-5043); remove the `/begin/checkout-return` record_payment sub-block (~9557-9564); remove the `/practitioner/checkout-return` legacy `if inv and cid:` block.

- [ ] **Step 1: Confirm the blast radius**

Run: `grep -rnE "qbo_reconcile|reconcile_qbo_payments|list_open_qbo_orders" app.py dashboard/*.py tests/*.py`
Expected: references only in `dashboard/qbo_reconcile.py`, `tests/test_qbo_reconcile.py`, and `app.py:5027/5039` (inside `console_reconcile_qbo`). If a reference exists anywhere else, STOP and report it.

- [ ] **Step 2: Delete the poller module + its test**

```bash
git rm dashboard/qbo_reconcile.py tests/test_qbo_reconcile.py
```

- [ ] **Step 3: Remove the reconcile route**

In `app.py`, delete the entire `@app.route("/api/console/reconcile-qbo", methods=["POST"])` + `def console_reconcile_qbo(): ...` block (~5013-5043, through its `return jsonify(...)`). Read the exact bounds first; remove the whole route function, nothing adjacent.

- [ ] **Step 4: Remove the `/begin/checkout-return` dead record_payment sub-block**

In `practitioner`... no — in `begin_checkout_return`, find:
```python
                if inv:
                    if cid and _kind != "in-house" and _kind not in (
                            "biofield", "retail"):
                        try:
                            from dashboard import qbo_billing as _qb_ret
                            _qb_ret.record_payment(cid, int(sess.get("amount_total") or 0), inv)
                        except Exception as e:
                            print(f"[begin-return] qbo payment failed: {e!r}", flush=True)
```
Delete ONLY the inner `if cid and _kind != "in-house" and _kind not in ("biofield","retail"): ... record_payment ...` block. KEEP the enclosing `if inv:` and everything else under it (the `if pi_id and (cid or _kind in (...))` booking block that does stripe-pi/points/referral/`book_sale_on_payment`). If removing the inner block leaves `if inv:` wrapping only the booking block, that's correct — leave `if inv:` in place.

- [ ] **Step 5: Remove the `/practitioner/checkout-return` legacy block**

In `practitioner_checkout_return`, find the legacy `if inv and cid:` block (record_payment + `set_order_stripe_pi` for a real-cid invoice) and delete it. KEEP the paid-only branch added in Stage 4 (`if inv: ... if _po["qbo_lines_json"] and not _po["qbo_sales_receipt_id"]: ... book_sale_on_payment ...`). Read both blocks first to remove only the legacy one.

- [ ] **Step 6: App imports + starts**

Run: `doppler run --config dev -- python3 -c "import app; print('import ok')"`
Expected: `import ok` (no `ModuleNotFoundError: qbo_reconcile`, no NameError).

- [ ] **Step 7: Paid-only return paths + money-path regression stay green**

Run: `doppler run --config dev -- python3 -m pytest tests/test_begin_checkout_paid_only.py tests/test_practitioner_checkout_return_paid_only.py tests/test_biofield_checkout_paid_only.py tests/test_book_sale_on_payment.py tests/test_bos_finance.py tests/test_order_payments_routes.py -v`
Expected: PASS (the removed code was dead; paid-only paths + the kept manual tools unaffected).

- [ ] **Step 8: Prove the poller is gone**

Run: `grep -rnE "qbo_reconcile|reconcile_qbo_payments|list_open_qbo_orders|console_reconcile_qbo" app.py dashboard/*.py`
Expected: no matches.

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "chore(qbo): retire reconcile poller + dead record_payment calls (Stage 5)"
```

---

## Full-suite gate (end of plan)

- [ ] Run: `doppler run --config dev -- python3 -m pytest tests/test_begin_checkout_paid_only.py tests/test_practitioner_checkout_return_paid_only.py tests/test_wholesale_paid_only.py tests/test_dispensary_paid_only.py tests/test_book_sale_on_payment.py tests/test_finance_record_payment.py tests/test_order_payments_routes.py -v`
  Expected: PASS, or any failure also present on `main`. (`test_finance_record_payment` + `test_order_payments_routes` confirm the KEPT manual tools still work.)

## Notes
- After Stage 5, the only QBO-invoice code left is `qbo_test_invoice` (diagnostic) + the operator-gated `finance.record_payment`/order-payments-ledger tools. No automatic invoice creation or reconcile anywhere.
- `build_module_order` (dead) remains a separate delete-candidate, out of scope here.
