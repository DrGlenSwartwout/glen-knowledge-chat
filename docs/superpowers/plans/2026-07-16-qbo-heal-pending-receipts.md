# QBO Auto-heal Stuck-PENDING Bookings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make every Sales Receipt self-identifying (order token in PrivateNote), and add a daily sweep that heals orders stuck at `qbo_sales_receipt_id='PENDING'` — stamp when the receipt exists (Case B), clear+rebook when it doesn't (Case A) — matched exactly by token, never double-booking.

**Architecture:** 4 tasks: (1) stamp PrivateNote; (2) a QBO lookup-by-token helper; (3) the `heal_pending_receipts` sweep (Case A/B, offline-testable via injected deps); (4) the cron endpoint + fold-in. QBO query semantics for PrivateNote are verified live (token 400s locally).

**Tech Stack:** Python, Flask, SQLite, pytest, QuickBooks Online REST API.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-16-qbo-heal-pending-receipts-design.md`. Prod has **0 stuck-PENDING orders** (going-forward only — no legacy to mis-heal).
- The atomic-claim contract is unchanged: a PENDING order never re-books via `book_sale_on_payment` (returns early); only this sweep resolves it.
- Exact-match ONLY (order token in PrivateNote). Never match on amount alone (double-book risk).
- Age guard: only sweep PENDING orders older than N min (default 10) so an in-flight booking isn't touched.
- Best-effort per order: one bad order is logged + skipped, never aborts the sweep.
- Do NOT touch `biofield_local_app.py`/`dashboard/biofield_report_html.py` (unrelated dirty WIP).
- Run tests: `doppler run --config dev -- python3 -m pytest <file>` (never bare pytest, never whole suite). QBO writes/queries 400 locally (token) — QBO-semantics steps are DEPLOYED-env verifications.

---

## Task 1: Stamp the order token into each Sales Receipt (PrivateNote)

**Files:** Modify `dashboard/qbo_billing.py` (`create_sales_receipt`), `dashboard/qbo_sale.py` (`book_sale_on_payment`). Test: `tests/test_sales_receipt_private_note.py` (create).

**Interfaces:** `create_sales_receipt(..., private_note: str|None = None)` — stamps `body["PrivateNote"]` when given.

- [ ] **Step 1: Write failing tests** — monkeypatch `qbo_billing._post` to capture body: `create_sales_receipt(..., private_note="order:tok1")` → `body["PrivateNote"] == "order:tok1"`; omitted → no `PrivateNote` key. And a `book_sale_on_payment` test asserting it calls `create_sales_receipt` with `private_note == f"order:{order['external_ref']}"`.
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement.** In `create_sales_receipt`, add `private_note=None` kwarg; after `body = {...}` (and before `_post`), add `if private_note: body["PrivateNote"] = str(private_note)[:1000]`. In `book_sale_on_payment`, pass `private_note=f"order:{order.get('external_ref')}"` to the `create_sales_receipt(...)` call (order dict has `external_ref`).
- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Regression** — `doppler run --config dev -- python3 -m pytest tests/test_qbo_sales_receipt.py tests/test_book_sale_on_payment.py tests/test_sales_receipt_private_note.py -v` → PASS.
- [ ] **Step 6: Commit** — `git commit -m "feat(qbo): stamp order token into SalesReceipt PrivateNote"`

---

## Task 2: QBO lookup — find a Sales Receipt by order token

**Files:** Modify `dashboard/qbo_billing.py` (add `find_sales_receipt_by_ref`). Test: `tests/test_find_sales_receipt_by_ref.py` (create).

**Interfaces:** `find_sales_receipt_by_ref(token, *, email=None, since_date=None) -> dict|None` — returns the SalesReceipt whose PrivateNote contains `order:<token>`, or None.

- [ ] **Step 1: Read** `qbo_billing._query` (the QBO query helper) + `find_or_create_customer`. Note QBO query is SQL-like via `/query`.
- [ ] **Step 2: Write failing tests** (monkeypatch `qbo_billing._query` to return canned QueryResponse dicts):
  - Primary path: `_query` returns a SalesReceipt whose `PrivateNote` contains `order:tok1` → returned.
  - Fallback path: if the LIKE query raises/returns empty, and a customer+date scan (`_query` for the customer's receipts) returns one with the matching PrivateNote → returned.
  - No match → None.
- [ ] **Step 3: Run → FAIL.**
- [ ] **Step 4: Implement.** Primary: `_query(f"SELECT * FROM SalesReceipt WHERE PrivateNote LIKE '%order:{_esc(token)}%'")`. Wrap in try/except; if it raises (QBO may not support LIKE on PrivateNote) OR returns nothing AND `email` given, fallback: resolve customer via `find_or_create_customer(email)`, `_query(f"SELECT * FROM SalesReceipt WHERE CustomerRef = '{cust_id}'" + (f" AND TxnDate >= '{since_date}'" if since_date else "") + " ORDERBY TxnDate DESC MAXRESULTS 50")`, and scan results client-side for `("order:"+token) in (r.get("PrivateNote") or "")`. Return the first exact-PrivateNote match or None. Use `_esc` for token safety.
- [ ] **Step 5: Run → PASS.**
- [ ] **Step 6: Commit** — `git commit -m "feat(qbo): find_sales_receipt_by_ref (PrivateNote token lookup + customer-scan fallback)"`
- [ ] **Step 7 (DEPLOYED-env, manual — the live-QBO verify):** after merge, on the deployed host, confirm the primary `PrivateNote LIKE` query actually returns a known receipt (create one via a real paid-only checkout, then query). If LIKE is unsupported by QBO, confirm the fallback path returns it. Record which path works. (Local runs 400 on QBO auth — cannot verify here; `feedback_verify_against_live_api`.)

---

## Task 3: The heal sweep

**Files:** Create `dashboard/qbo_heal.py`. Test: `tests/test_qbo_heal.py` (create).

**Interfaces:** `heal_pending_receipts(cx, *, find_receipt, book, stamp, older_than_min=10, now=None) -> list[dict]` — pure/injectable; the route wires the real deps.

- [ ] **Step 1: Write failing tests** (in-memory sqlite via `dashboard.orders`; inject fakes):
  - **Case B (receipt exists):** seed a PENDING order (updated_at 30 min ago, external_ref token, email); `find_receipt(token, email, since)` returns `{"Id":"SR9"}` → `stamp(cx, oid, "SR9")` called, `book` NOT called; result lists `{order_id, action:"stamped", receipt_id:"SR9"}`.
  - **Case A (no receipt):** `find_receipt` returns None → order `qbo_sales_receipt_id` cleared to NULL then `book(cx, order)` called (returns a new id) → result `{action:"rebooked", ...}`.
  - **Age guard:** a PENDING order updated 2 min ago is NOT in the sweep.
  - **Best-effort:** one order whose `find_receipt` raises is skipped (logged), the sweep continues and returns the others.
  - A non-PENDING order (real id or NULL) is never touched.
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement** `dashboard/qbo_heal.py`:
  ```python
  def heal_pending_receipts(cx, *, find_receipt, book, stamp, older_than_min=10, now=None):
      import datetime
      cutoff = (now or datetime.datetime.utcnow()) - datetime.timedelta(minutes=older_than_min)
      cx.row_factory = __import__("sqlite3").Row
      rows = cx.execute(
          "SELECT * FROM orders WHERE qbo_sales_receipt_id='PENDING' AND updated_at < ?",
          (cutoff.isoformat(),)).fetchall()
      out = []
      for r in rows:
          o = dict(r)
          try:
              token = o.get("external_ref")
              existing = find_receipt(token, email=o.get("email"), since_date=(o.get("created_at") or "")[:10])
              if existing and existing.get("Id"):
                  stamp(cx, o["id"], existing["Id"])
                  out.append({"order_id": o["id"], "action": "stamped", "receipt_id": existing["Id"]})
              else:
                  cx.execute("UPDATE orders SET qbo_sales_receipt_id=NULL WHERE id=?", (o["id"],)); cx.commit()
                  o["qbo_sales_receipt_id"] = None
                  sr = book(cx, o)
                  out.append({"order_id": o["id"], "action": "rebooked", "receipt_id": sr})
          except Exception as e:
              print(f"[qbo-heal] order {o.get('id')!r} skipped: {e!r}", flush=True)
      return out
  ```
  (`stamp` = `orders.set_order_sales_receipt_id`; `book` = a closure calling `qbo_sale.book_sale_on_payment`; `find_receipt` = `qbo_billing.find_sales_receipt_by_ref`.)
- [ ] **Step 4: Run → PASS.** — `doppler run --config dev -- python3 -m pytest tests/test_qbo_heal.py -v`
- [ ] **Step 5: Commit** — `git commit -m "feat(qbo): heal_pending_receipts sweep (Case A rebook / Case B stamp, exact token)"`

---

## Task 4: Cron endpoint + fold-in

**Files:** Modify `app.py` (new route), `scripts/run_briefings_cron.py`. Test: `tests/test_qbo_heal_route.py` (create).

- [ ] **Step 1: Read** an existing `/api/cron/*` route's auth (e.g. `/api/cron/household-holds/sweep`, `app.py:21600`) — mirror its `X-Cron-Secret`/CRON_SECRET gate exactly. Read `scripts/run_briefings_cron.py:main()` (curls `{WEB_URL}/cron/regenerate-briefings` with `X-Cron-Secret`).
- [ ] **Step 2: Write failing test** — `POST /api/cron/qbo-heal-pending` without the secret → 401; with it → 200 and calls `heal_pending_receipts` (monkeypatch the heal to return `[]`).
- [ ] **Step 3: Run → FAIL.**
- [ ] **Step 4: Add the route** in `app.py` mirroring the cron auth:
  ```python
  @app.route("/api/cron/qbo-heal-pending", methods=["POST"])
  def api_cron_qbo_heal_pending():
      # <same X-Cron-Secret / CRON_SECRET|CONSOLE_SECRET gate the other /api/cron/* routes use>
      from dashboard import qbo_heal as _heal, qbo_billing as _qb, qbo_sale as _qs, orders as _ord
      cx = _sqlite3.connect(LOG_DB); cx.row_factory = _sqlite3.Row
      try:
          healed = _heal.heal_pending_receipts(
              cx,
              find_receipt=_qb.find_sales_receipt_by_ref,
              book=lambda cx2, o: _qs.book_sale_on_payment(cx2, o),
              stamp=_ord.set_order_sales_receipt_id)
      finally:
          cx.close()
      return jsonify({"ok": True, "healed": healed, "count": len(healed)})
  ```
  (Confirm `_sqlite3`, `LOG_DB` names; use the exact gate helper the sibling `/api/cron/*` routes use.)
- [ ] **Step 5: Fold into the cron.** In `scripts/run_briefings_cron.py:main()`, after the existing briefings curl, add a second best-effort curl of `{WEB_URL}/api/cron/qbo-heal-pending` with the same `X-Cron-Secret` header (wrap so a heal failure doesn't fail the briefings cron). Do NOT add a new render.yaml service — this rides the existing `glen-briefings-daily` cron. Confirm that service is active (uncommented) in `render.yaml`.
- [ ] **Step 6: Run → PASS** — `doppler run --config dev -- python3 -m pytest tests/test_qbo_heal_route.py -v`
- [ ] **Step 7: Commit** — `git commit -m "feat(qbo): /api/cron/qbo-heal-pending + fold into daily briefings cron"`

---

## Full-suite gate

- [ ] `doppler run --config dev -- python3 -m pytest tests/test_sales_receipt_private_note.py tests/test_find_sales_receipt_by_ref.py tests/test_qbo_heal.py tests/test_qbo_heal_route.py tests/test_book_sale_on_payment.py tests/test_qbo_sales_receipt.py -v` → PASS.
- [ ] **DEPLOYED-env (post-merge):** (a) Task 2 Step 7 live-QBO PrivateNote lookup verify. (b) Seed one stuck order (`qbo_sales_receipt_id='PENDING'`, updated_at old) via a console/DB touch, POST `/api/cron/qbo-heal-pending`, confirm it books/stamps and clears PENDING. (c) Confirm the daily briefings cron's added curl 200s.

## Notes
- 0 legacy PENDING orders today → no backfill/migration risk.
- Stuck-`PENDING` visibility bonus: the heal endpoint's `{healed:[...]}` response is the operator's record of what it fixed.
