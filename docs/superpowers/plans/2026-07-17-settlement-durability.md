# Settlement Durability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the two residual issues from per-kind settlement (#953): (I1) a crash between booking the receipt and settling strands settlement; (I2) the points ledger's check-then-insert guard can double-apply under prod's 2-worker concurrency.

**Architecture:** A `settled_at` marker on `orders` + moving the webhook's settlement out of its receipt-booked guard (so the always-delivered, Stripe-retried webhook backfills any crash-stranded settlement using the live session) closes I1. A UNIQUE index on `points_ledger(order_ref, reason, scope)` + `INSERT OR IGNORE` in `_add` makes every points write atomically idempotent (covering loyalty points, dispensary points, referral points, AND ship-credit — all rows in the same table), closing I2.

**Tech Stack:** Python, Flask, SQLite, pytest, Stripe.

## Global Constraints

- **Spec:** `docs/superpowers/specs/2026-07-17-settlement-durability-design.md`.
- **Money path, no CI:** deploy-chat merge = deploy. Correctness over speed.
- **Run tests:** `doppler run --config dev -- python3 -m pytest <file> -v` — NEVER bare `pytest` (prd Doppler config breaks app-import collection) and NEVER the whole suite (can send live email).
- **Do NOT touch** `biofield_local_app.py` or `dashboard/biofield_report_html.py`.
- **`balance()` is authoritative** = `SUM(delta_cents)` over the scope; `balance_after` is a non-authoritative per-row snapshot. So preventing the duplicate INSERT is the complete fix — the true balance self-corrects.
- **`INSERT OR IGNORE` is safe with OR without the index present:** with no UNIQUE constraint it behaves as a normal INSERT (identical to today); the index is what activates dedup. So deploy order does not matter for `_add`.
- **ship-credit is covered for free:** `ship_credit.grant/consume/refund` all delegate to `points.credit/spend` on `points_ledger` with `scope='ship_credit'` — the same UNIQUE index dedups them. No separate change.
- **Behavior-preserving redirect:** the redirect still settles exactly the same per-kind effects; this only ADDS a `mark_order_settled` call after its dispatch.
- **`order_ref`** for a paid-only order = `external_ref` = `md["invoice_id"]`.

---

## Task 1: `settled_at` marker on orders

**Files:**
- Modify: `dashboard/orders.py` — `init_orders_table` (the `ALTER TABLE orders ADD COLUMN` DDL tuple, ~line 65-122); add `mark_order_settled` near `set_order_sales_receipt_id` (~line 444).
- Test: `tests/test_settled_marker.py` (create)

**Interfaces:**
- Produces: `orders.mark_order_settled(cx, order_id) -> bool` (sets `settled_at` = now where currently NULL; returns True iff it set it). `settled_at` becomes a column on order rows (flows through `find_order_by_external_ref`'s `SELECT *`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_settled_marker.py`:

```python
import sqlite3
from dashboard import orders

def _mk():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    orders.init_orders_table(cx)
    return cx

def _seed(cx):
    cx.execute("INSERT INTO orders (source, external_ref, email, status) VALUES (?,?,?,?)",
               ("funnel", "tok1", "a@b.com", "new"))
    cx.commit()
    return cx.execute("SELECT id FROM orders WHERE external_ref='tok1'").fetchone()["id"]

def test_settled_at_column_exists_and_defaults_null():
    cx = _mk(); oid = _seed(cx)
    row = cx.execute("SELECT settled_at FROM orders WHERE id=?", (oid,)).fetchone()
    assert row["settled_at"] is None

def test_mark_order_settled_sets_once_and_is_idempotent():
    cx = _mk(); oid = _seed(cx)
    assert orders.mark_order_settled(cx, oid) is True
    row = cx.execute("SELECT settled_at FROM orders WHERE id=?", (oid,)).fetchone()
    assert row["settled_at"] is not None
    first = row["settled_at"]
    # second call is a no-op: does not overwrite, returns False
    assert orders.mark_order_settled(cx, oid) is False
    row2 = cx.execute("SELECT settled_at FROM orders WHERE id=?", (oid,)).fetchone()
    assert row2["settled_at"] == first

def test_find_order_exposes_settled_at():
    cx = _mk(); _seed(cx)
    o = orders.find_order_by_external_ref(cx, "tok1")
    assert "settled_at" in dict(o)
```

- [ ] **Step 2: Run → FAIL**

Run: `doppler run --config dev -- python3 -m pytest tests/test_settled_marker.py -v`
Expected: FAIL (no `settled_at` column; `mark_order_settled` undefined).

- [ ] **Step 3: Add the column migration**

In `init_orders_table`, add to the DDL tuple that is iterated with per-statement `try/except Exception: pass`:

```python
    "ALTER TABLE orders ADD COLUMN settled_at TEXT",
```

- [ ] **Step 4: Add `mark_order_settled`**

Near `set_order_sales_receipt_id` (mirror its shape; reuse the module's `_now()`):

```python
def mark_order_settled(cx, order_id):
    """Mark per-kind settlement as attempted for this order. Conditional on
    settled_at being NULL so a re-run (redirect + webhook, or a webhook retry)
    never overwrites the first timestamp. Returns True iff this call set it."""
    cur = cx.execute(
        "UPDATE orders SET settled_at=?, updated_at=? WHERE id=? AND settled_at IS NULL",
        (_now(), _now(), order_id))
    cx.commit()
    return cur.rowcount > 0
```

- [ ] **Step 5: Run → PASS**

Run: `doppler run --config dev -- python3 -m pytest tests/test_settled_marker.py -v`
Expected: PASS (3 passed).

- [ ] **Step 6: Regression — orders model tests**

Run: `doppler run --config dev -- python3 -m pytest tests/test_orders.py -v` (if present; else skip)
Expected: PASS (new nullable column doesn't change existing behavior).

- [ ] **Step 7: Commit**

```bash
git add dashboard/orders.py tests/test_settled_marker.py
git commit -m "feat(orders): settled_at marker + mark_order_settled (conditional, idempotent)"
```

---

## Task 2: Atomic points ledger + prod dedup route (closes I2)

**Files:**
- Modify: `dashboard/points.py` — `init_points_table` (add UNIQUE index), `_add` (INSERT OR IGNORE + authoritative return).
- Modify: `app.py` — add owner-gated `POST /api/console/points-dedup` route.
- Test: `tests/test_points_atomic.py` (create)

**Interfaces:**
- Consumes: nothing new.
- Produces: `_add` is now atomically idempotent per `(order_ref, reason, scope)`; route `POST /api/console/points-dedup` (dry-run reports duplicate groups; `?apply=1` deletes dupes keeping MIN(id) and ensures the UNIQUE index).

- [ ] **Step 1: Write the failing test**

Create `tests/test_points_atomic.py`:

```python
import sqlite3
from dashboard import points

def _mk():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    points.init_points_table(cx)
    return cx

def test_add_is_atomically_idempotent_bypassing_has_entry():
    # Simulate the cross-process race: two _add calls with the SAME
    # (order_ref, reason, scope) that both bypass the has_entry fast-path.
    cx = _mk()
    points._add(cx, "a@b.com", 500, "earn", "tok1")
    points._add(cx, "a@b.com", 500, "earn", "tok1")   # OR IGNORE -> no second row
    n = cx.execute("SELECT COUNT(*) FROM points_ledger "
                   "WHERE order_ref='tok1' AND reason='earn' AND scope='rm'").fetchone()[0]
    assert n == 1
    assert points.balance(cx, "a@b.com") == 500        # NOT 1000

def test_redeem_does_not_double_debit_on_same_ref():
    cx = _mk()
    points._add(cx, "a@b.com", 1000, "earn", "seed")
    points.redeem(cx, "a@b.com", value_cents=400, order_ref="tok1")
    # a racing duplicate redeem for the same order_ref is an atomic no-op
    points._add(cx, "a@b.com", -400, "redeem", "tok1")
    assert points.balance(cx, "a@b.com") == 600        # debited once, not twice

def test_distinct_keys_still_insert():
    cx = _mk()
    points._add(cx, "a@b.com", 500, "earn", "tokA")
    points._add(cx, "a@b.com", 500, "earn", "tokB")     # different order_ref
    points._add(cx, "a@b.com", 500, "referral", "tokA")  # different reason
    assert cx.execute("SELECT COUNT(*) FROM points_ledger").fetchone()[0] == 3

def test_scope_discriminates_ship_credit_from_rm():
    cx = _mk()
    points._add(cx, "a@b.com", 500, "earn", "tok1", scope="rm")
    points._add(cx, "a@b.com", 500, "ship_overpay", "tok1", scope="ship_credit")
    assert points.balance(cx, "a@b.com", scope="rm") == 500
    assert points.balance(cx, "a@b.com", scope="ship_credit") == 500

def test_add_returns_authoritative_balance_when_ignored():
    cx = _mk()
    assert points._add(cx, "a@b.com", 500, "earn", "tok1") == 500
    # the ignored duplicate must return the TRUE balance (500), not an optimistic 1000
    assert points._add(cx, "a@b.com", 500, "earn", "tok1") == 500
```

- [ ] **Step 2: Run → FAIL**

Run: `doppler run --config dev -- python3 -m pytest tests/test_points_atomic.py -v`
Expected: FAIL (`test_add_is_atomically_idempotent...` inserts twice; balance 1000; `_add` returns 1000 on the dup).

- [ ] **Step 3: Add the UNIQUE index in `init_points_table`**

After the existing `ix_points_email` index line, add (idempotent):

```python
    cx.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_points_order_ref_reason_scope "
               "ON points_ledger(order_ref, reason, scope)")
```

(NULL `order_ref` rows are exempt — SQLite treats each NULL as distinct under a UNIQUE index — so legacy manual credits with no ref never collide. On a fresh/dev DB this creates cleanly; on prod it is handled by the dedup route in Step 5 before it can matter, and `INSERT OR IGNORE` is a no-op-safe normal insert until the index exists.)

- [ ] **Step 4: Make `_add` atomic + authoritative return**

Replace `_add`'s INSERT with `INSERT OR IGNORE` and return the re-read authoritative balance:

```python
def _add(cx, email, delta_cents, reason, order_ref, scope="rm"):
    # balance_after is a NON-authoritative snapshot (balance() is SUM(delta_cents)).
    # INSERT OR IGNORE + the UNIQUE(order_ref,reason,scope) index makes this atomically
    # idempotent across processes: a concurrent duplicate (redirect + webhook settling
    # the same order under gunicorn's 2 workers) inserts exactly one row. Return the
    # re-read true balance so the result is correct whether we inserted or ignored.
    snapshot = balance(cx, email, scope=scope) + int(delta_cents)
    cx.execute("""INSERT OR IGNORE INTO points_ledger(email,delta_cents,reason,order_ref,balance_after,scope)
                  VALUES (?,?,?,?,?,?)""",
               (email, int(delta_cents), reason, order_ref, snapshot, scope))
    cx.commit()
    return balance(cx, email, scope=scope)
```

Leave `has_entry`, `credit`, `spend`, `earn`, `redeem` unchanged — `has_entry` stays as a cheap fast-path pre-check; the UNIQUE index is the atomic backstop for the cross-process race the pre-check can't cover. (`redeem`'s caller `_settle_order_points` already wraps it in `try/except ValueError`, so the rare post-commit balance-recheck raise on an idempotent re-call stays swallowed.)

- [ ] **Step 5: Add the owner-gated dedup route in `app.py`**

Find how other `X-Console-Key`-gated console routes are declared (grep `CONSOLE_SECRET` near an `@app.route("/api/console/...")`), match that auth exactly. Add:

```python
@app.route("/api/console/points-dedup", methods=["POST"])
def api_console_points_dedup():
    """Owner-gated one-off: remove duplicate (order_ref, reason, scope) points_ledger
    rows (keeping the earliest id) and ensure the UNIQUE index. Dry-run by default;
    ?apply=1 performs the deletion. Idempotent/re-runnable. Prereq for the UNIQUE
    index to create cleanly on prod."""
    if request.headers.get("X-Console-Key") != CONSOLE_SECRET:
        return jsonify({"error": "unauthorized"}), 401
    apply = request.args.get("apply") == "1"
    cx = _sqlite3.connect(LOG_DB); cx.row_factory = _sqlite3.Row
    try:
        dupes = cx.execute(
            "SELECT order_ref, reason, scope, COUNT(*) c, MIN(id) keep "
            "FROM points_ledger WHERE order_ref IS NOT NULL "
            "GROUP BY order_ref, reason, scope HAVING c > 1").fetchall()
        groups = [dict(r) for r in dupes]
        removed = 0
        if apply:
            for g in groups:
                cur = cx.execute(
                    "DELETE FROM points_ledger WHERE order_ref=? AND reason=? AND scope=? AND id<>?",
                    (g["order_ref"], g["reason"], g["scope"], g["keep"]))
                removed += cur.rowcount
            cx.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_points_order_ref_reason_scope "
                       "ON points_ledger(order_ref, reason, scope)")
            cx.commit()
        index_exists = cx.execute(
            "SELECT 1 FROM sqlite_master WHERE type='index' "
            "AND name='ux_points_order_ref_reason_scope'").fetchone() is not None
        return jsonify({"ok": True, "applied": apply, "duplicate_groups": len(groups),
                        "rows_removed": removed, "index_exists": index_exists,
                        "groups": groups[:50]})
    finally:
        cx.close()
```

- [ ] **Step 6: Test the dedup route (add to `tests/test_points_atomic.py`)**

Add a route test using the Flask test client (monkeypatch `app.LOG_DB` to a tmp db + `app.CONSOLE_SECRET`; INSERT two rows with the same (order_ref,reason,scope) DIRECTLY via SQL to bypass the index — do this BEFORE `init_points_table` creates the index, or drop the index first, so the seed can create the duplicate). Assert: dry-run reports `duplicate_groups==1, rows_removed==0`; `?apply=1` returns `rows_removed==1, index_exists==True`; one row remains; a second `apply` is a no-op (`duplicate_groups==0`). Read an existing console-route test for the client + auth-header fixture pattern.

- [ ] **Step 7: Run → PASS**

Run: `doppler run --config dev -- python3 -m pytest tests/test_points_atomic.py -v`
Expected: PASS (all).

- [ ] **Step 8: Regression — points + rewards + ship_credit tests**

Run: `doppler run --config dev -- python3 -m pytest tests/test_points.py tests/test_ship_credit.py -v` (whichever exist)
Expected: PASS (idempotent callers unchanged; distinct-key inserts unaffected).

- [ ] **Step 9: Commit**

```bash
git add dashboard/points.py app.py tests/test_points_atomic.py
git commit -m "feat(points): atomic ledger (UNIQUE index + INSERT OR IGNORE) + owner dedup route (closes I2)"
```

---

## Task 3: Decouple webhook settlement + mark settled (closes I1)

**Files:**
- Modify: `app.py` — `webhook_stripe` `checkout.session.completed` settlement block (~27427-27452); redirect `begin_checkout_return` settlement dispatch (~9879-9882).
- Test: extend `tests/test_webhook_back_booking.py`.

**Interfaces:**
- Consumes: `orders.mark_order_settled` (Task 1), the existing `order_settlement.settle_paid_order_effects` + `_SETTLEMENT_DEPS`.

- [ ] **Step 1: Write the failing tests (extend `tests/test_webhook_back_booking.py`)**

Reuse the file's existing seeding + monkeypatch harness. Add:

```python
# 1. Booked-but-unsettled backfill (I1): order already has qbo_sales_receipt_id set
#    (redirect booked) AND settled_at IS NULL -> webhook RUNS settlement (spy a settler,
#    e.g. subscriptions.create_once or _settle_order_points, fires) and settled_at is set.
# 2. Already-settled skip: order settled_at set -> webhook does NOT run settlement
#    (spy NOT called).
# 3. Total-settle failure leaves retry-able: monkeypatch settle_paid_order_effects to RAISE
#    -> webhook still returns 200, receipt still booked, and settled_at is STILL NULL
#    (so a Stripe redelivery can re-settle).
```

(Test 1 requires seeding an order with `qbo_sales_receipt_id` already set and `settled_at` NULL — bypasses the booking branch, exercises the new settlement branch.)

Run: `doppler run --config dev -- python3 -m pytest tests/test_webhook_back_booking.py -v`
Expected: test 1 FAILS (webhook skips settlement when already booked); test 3 FAILS (settled_at handling not present). Pre-existing tests still pass.

- [ ] **Step 2: Restructure the webhook block**

In `webhook_stripe`, change the guard structure so BOOKING stays gated on not-yet-booked but SETTLEMENT is gated independently on not-yet-settled. Replace the current `if _wo and _wo["qbo_lines_json"] and not _wo["qbo_sales_receipt_id"]:` block body with:

```python
                                if _wo and _wo["qbo_lines_json"]:
                                    # Book the receipt only if not already booked (atomic-claim guarded).
                                    if not _wo["qbo_sales_receipt_id"]:
                                        _wpi = sess.get("payment_intent")
                                        if _wpi:
                                            _bos_orders.set_order_stripe_pi(_wcx, _wo["id"], _wpi)
                                        _bos_orders.set_order_payment(
                                            _wcx, _wo["id"], method="card",
                                            amount_cents=int(sess.get("amount_total") or 0))
                                        from dashboard import qbo_sale as _wqs
                                        _wqs.book_sale_on_payment(
                                            _wcx, dict(_bos_orders.find_order_by_external_ref(_wcx, inv)))
                                    # Settle per-kind side-effects INDEPENDENTLY of booking, gated on
                                    # settled_at. Closes the crash-strand: a hard crash between booking
                                    # and settling leaves settled_at NULL + no 200 -> Stripe redelivers
                                    # -> this runs. A redirect crash after booking is backfilled here too
                                    # (the webhook always fires). Best-effort: a settler raise must not
                                    # 500 the webhook or block booking.
                                    if not _wo["settled_at"]:
                                        try:
                                            _wmd = sess.get("metadata") or {}
                                            _wro = _bos_orders.find_order_by_external_ref(_wcx, inv)
                                            from dashboard import order_settlement as _wosx
                                            _wosx.settle_paid_order_effects(
                                                kind=_wmd.get("kind") or "",
                                                order=(dict(_wro) if _wro else None),
                                                md=_wmd, pi_id=sess.get("payment_intent"),
                                                sid=session_id, deps=_SETTLEMENT_DEPS)
                                            _bos_orders.mark_order_settled(_wcx, _wo["id"])
                                        except Exception as _wse:
                                            print(f"[stripe-webhook] settlement failed: {_wse!r}",
                                                  flush=True)
```

Note: `mark_order_settled` is INSIDE the try, AFTER `settle_paid_order_effects` — so a total settle raise skips the mark (leaving settled_at NULL for a retry), while the orchestrator's own per-settler best-effort skips still let the mark run.

- [ ] **Step 3: Redirect marks settled**

In `begin_checkout_return`, immediately AFTER the `_osx.settle_paid_order_effects(...)` call (~9882), add:

```python
                if inv and _settle_order:
                    try:
                        _mcx = _sqlite3.connect(LOG_DB)
                        try:
                            _bos_orders.mark_order_settled(_mcx, _settle_order["id"])
                        finally:
                            _mcx.close()
                    except Exception as _me:
                        print(f"[begin-return] mark settled: {_me!r}", flush=True)
```

- [ ] **Step 4: Run → PASS**

Run: `doppler run --config dev -- python3 -m pytest tests/test_webhook_back_booking.py -v`
Expected: PASS (all — the 3 new + all pre-existing).

- [ ] **Step 5: Import + redirect regression**

Run: `doppler run --config dev -- python3 -c "import app; print('import ok')"` then
`doppler run --config dev -- python3 -m pytest tests/test_begin_return_settlement.py tests/test_begin_checkout_paid_only.py -v`
Expected: `import ok`; redirect settlement unchanged (now also marks settled — existing assertions still hold).

- [ ] **Step 6: Commit**

```bash
git add app.py tests/test_webhook_back_booking.py
git commit -m "feat(settlement): webhook settles on booked-but-unsettled + mark settled (closes I1 crash-strand)"
```

---

## Full-suite gate

- [ ] Run: `doppler run --config dev -- python3 -m pytest tests/test_settled_marker.py tests/test_points_atomic.py tests/test_webhook_back_booking.py tests/test_begin_return_settlement.py tests/test_begin_checkout_paid_only.py tests/test_order_settlement.py -v`
  Expected: PASS, or any failure also present on a `main` baseline run of the same files.

## Post-deploy (deployed env — in the PR checklist)

- [ ] **Prod dedup + index (do this right after deploy):** `POST /api/console/points-dedup` (dry-run) → read `duplicate_groups`. If > 0, `POST /api/console/points-dedup?apply=1` → confirm `rows_removed` and `index_exists: true`. If 0, still `?apply=1` once to ensure `index_exists: true`. This is the I2 activation on prod.
- [ ] **I1 backfill smoke:** seed (or find) a paid-only order with the receipt booked and `settled_at` NULL, replay/POST its `checkout.session.completed` → confirm the per-kind effect settles and `settled_at` is set; a second replay is a no-op.
- [ ] Confirm prod stays healthy after the redeploy (app imports the changed modules).

## Notes
- Closes the two deferred follow-ups from #953 (`feedback_qbo_paid_only_no_unpaid_invoices`).
- No new state persisted on the order beyond `settled_at`, and no heal cron — the always-delivered, Stripe-retried webhook is the settlement backfill (it carries the live session the settlers need).
- The UNIQUE index closes the SAME-`(order_ref,reason,scope)` double (the settlement double-settle). It does NOT serialize DIFFERENT-order concurrent redeems for one email (a pre-existing, separate `_add` concern, explicitly out of scope).
