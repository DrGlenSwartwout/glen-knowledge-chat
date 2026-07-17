# Settlement Follow-ups Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the two remaining settlement follow-ups from #958: (A) make the live `_grant_group_bundle` grant atomic so a concurrent redirect+webhook settle can't double-grant a membership + double-email; (B) surface a silently-skipped settler as an actionable console todo.

**Architecture:** (A) claim the `group_bundle_grants` marker FIRST via `ON CONFLICT(invoice_id) DO NOTHING` + a `rowcount==0` bail, before any membership/email side effect. (B) both settlement callers capture `settle_paid_order_effects`'s return and, when `skipped` is non-empty, raise ONE deduped `todos` row via a new best-effort helper.

**Tech Stack:** Python, Flask, SQLite, pytest.

## Global Constraints

- **Spec:** `docs/superpowers/specs/2026-07-17-settlement-followups-design.md`.
- **Money/fulfillment path, no CI:** deploy-chat merge = deploy. Correctness over speed.
- **Run tests:** `doppler run --config dev -- python3 -m pytest <file> -v` — NEVER bare `pytest`, NEVER the whole suite.
- **Do NOT touch** `biofield_local_app.py` or `dashboard/biofield_report_html.py`.
- **Behavior-preserving except the fix:** `_grant_group_bundle` must still grant exactly one membership per invoice for the first (winning) run, extend an existing membership the same way, and stay flag-gated on `GROUP_BUNDLE_ENABLED`. The skipped-todo helper must be best-effort (never raise into the request path).
- **`order_ref`** at the settlement call sites = `inv` (= `md["invoice_id"]`).

---

## Task 1: Group-bundle claim-first (atomicity)

**Files:**
- Modify: `app.py` — `_grant_group_bundle` (~line 6245-6300)
- Test: `tests/test_group_bundle_atomic.py` (create)

**Interfaces:** no new signatures; internal reorder only.

- [ ] **Step 1: Write the failing test**

Create `tests/test_group_bundle_atomic.py`. DB-isolated (monkeypatch `app.LOG_DB` to a tmp db). Set `GROUP_BUNDLE_ENABLED=1`. Monkeypatch `app._member_join_welcome` to a spy (count calls) and `dashboard.stripe_pay.get_payment_intent` to return `{"customer":"cus_1","payment_method":"pm_1"}`. Call `app._grant_group_bundle(md, "pi_1")` with `md={"grant_group_months":"1","email":"a@b.com","invoice_id":"tok1"}`. Cover:

```python
# 1. First grant creates exactly one membership row + one welcome; the grant marker exists.
# 2. A SECOND call for the SAME invoice is a no-op: still one membership row, welcome NOT called again.
# 3. Concurrent-race sim: pre-INSERT the marker for tok1 (as if another run claimed it),
#    then _grant_group_bundle -> bails on rowcount==0: no membership created, welcome not called.
```

(Read `_grant_group_bundle` first for the exact table/migration calls it makes so the test seeds the same schema — it self-creates `group_bundle_grants` and calls `subscriptions.init_subscriptions_table` + migrations. Assert membership count via `subscriptions.active_memberships_by_email` or a direct `SELECT COUNT(*) FROM subscriptions`.)

Run: `doppler run --config dev -- python3 -m pytest tests/test_group_bundle_atomic.py -v`
Expected: test 2 and 3 FAIL against current code (current order creates membership + welcome BEFORE the marker check, so a race/second-run double-creates or the pre-claimed-marker case still creates).

- [ ] **Step 2: Reorder `_grant_group_bundle` to claim-first**

In `_grant_group_bundle`, after the `CREATE TABLE IF NOT EXISTS group_bundle_grants ...` + `_gcx.commit()`, REPLACE the current `already = SELECT ... ; if not already: <membership work> ; INSERT marker` block with a claim-first structure:

```python
                # Claim the grant marker ATOMICALLY before any side effect: exactly one
                # run wins ON CONFLICT, so a concurrent redirect+webhook settle (or a
                # second delivery) can't double-create the membership or double-send the
                # welcome. (Trade-off: a hard crash between this commit and create_membership
                # would strand this one free-window grant — rare, and strictly better than
                # the double-grant it replaces.)
                claim = _gcx.execute(
                    "INSERT INTO group_bundle_grants (invoice_id, created_at) VALUES (?,?) "
                    "ON CONFLICT(invoice_id) DO NOTHING", (g_invoice, _now_utc().isoformat()))
                _gcx.commit()
                if claim.rowcount == 0:
                    return  # another run already claimed/granted this invoice
                existing = _subs_gb.active_memberships_by_email(_gcx, g_email)
                if existing:
                    cur = existing[0]
                    _subs_gb.set_next_charge_date(
                        _gcx, cur["id"],
                        _subs_gb.add_months(cur["next_charge_date"], n))
                elif g_cus and g_pm:
                    start = _subs_gb.add_months(_date_gb.today().isoformat(), n)
                    _subs_gb.create_membership(
                        _gcx, email=g_email, stripe_customer_id=g_cus,
                        stripe_payment_method_id=g_pm,
                        amount_cents=_gb.MEMBERSHIP_AMOUNT_CENTS,
                        next_charge_date=start)
                    _member_join_welcome(_gcx, g_email, "subscription")
                _gcx.commit()
                print(f"[group-bundle] granted {n}mo to {g_email} inv={g_invoice}", flush=True)
```

Remove the old `already = _gcx.execute("SELECT 1 ...")` check and the trailing `INSERT INTO group_bundle_grants ...` (now done first as the claim). Keep everything ABOVE (flag gate, `get_payment_intent`, `g_cus`/`g_pm`, the table/migration setup) and the outer `try/except app.logger.exception` unchanged.

- [ ] **Step 3: Run → PASS**

Run: `doppler run --config dev -- python3 -m pytest tests/test_group_bundle_atomic.py -v`
Expected: PASS (3 passed).

- [ ] **Step 4: Regression + import**

Run: `doppler run --config dev -- python3 -m pytest tests/test_group_bundle_grant.py -v` (the existing group-bundle suite) and `doppler run --config dev -- python3 -c "import app; print('import ok')"`
Expected: PASS; `import ok`. (Existing grant behavior — one membership per new invoice, extend on existing — unchanged.)

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_group_bundle_atomic.py
git commit -m "fix(group-bundle): claim marker first (atomic) to prevent double-grant under concurrent settle"
```

---

## Task 2: Skipped-settler → console todo

**Files:**
- Modify: `app.py` — add `_raise_settlement_skip_todo`; capture the result at the redirect (`begin_checkout_return` ~9916) and webhook (`webhook_stripe` ~27498) settlement call sites.
- Test: `tests/test_settlement_skip_todo.py` (create)

**Interfaces:**
- Produces: `_raise_settlement_skip_todo(order_ref, kind, skipped)` — best-effort; inserts ONE deduped `todos` row (`dedup_key=f"settle-skip:{order_ref}"`); never raises.

- [ ] **Step 1: Write the failing test**

Create `tests/test_settlement_skip_todo.py`. DB-isolated (monkeypatch `app.LOG_DB` to a tmp db; ensure the `todos` table exists — call `app._init_todos_table(cx)` if that's how it's created, else the helper should `CREATE TABLE IF NOT EXISTS`). Cover:

```python
# 1. _raise_settlement_skip_todo("tok1", "subscribe", ["subscription"]) inserts one todos row
#    with dedup_key="settle-skip:tok1", owner="glen", source="settlement-skip".
# 2. A second call for the same order_ref -> still ONE row (ON CONFLICT(dedup_key) DO NOTHING).
# 3. The helper swallows a DB error (e.g. pass a bad/closed connection path) without raising.
```

(Grep how `todos` is created — `_init_todos_table` / the `_maybe_raise_cashout_review` insert at ~app.py:6031 — and match its columns exactly.)

Run: `doppler run --config dev -- python3 -m pytest tests/test_settlement_skip_todo.py -v`
Expected: FAIL (`_raise_settlement_skip_todo` undefined).

- [ ] **Step 2: Add the helper**

Near `_maybe_raise_cashout_review` (app.py ~6011), add:

```python
def _raise_settlement_skip_todo(order_ref, kind, skipped):
    """Best-effort: when per-kind settlement skipped one or more effects (a settler
    raised and was swallowed best-effort, then the order was marked settled and will
    NOT retry), raise ONE deduped console todo so the stranded effect is visible +
    actionable. Never raises into the request path."""
    try:
        if not order_ref or not skipped:
            return
        from datetime import datetime as _dt, timezone as _tz
        now = _dt.now(_tz.utc).isoformat()
        names = ", ".join(str(s) for s in skipped)
        dedup_key = f"settle-skip:{order_ref}"
        _tcx = _sqlite3.connect(LOG_DB)
        try:
            _init_todos_table(_tcx)
            _tcx.execute(
                """INSERT INTO todos (created_at, owner, category, title, body, priority, source, dedup_key)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(dedup_key) DO NOTHING""",
                (now, "glen", "Fulfillment",
                 f"Settlement skipped for order {order_ref}",
                 f"Order {order_ref} (kind={kind}) was marked settled but these effects were "
                 f"skipped and will not auto-retry: {names}. Re-run settlement manually if needed.",
                 "high", "settlement-skip", dedup_key))
            _tcx.commit()
        finally:
            _tcx.close()
    except Exception as _e:
        print(f"[settlement] skip-todo failed ref={order_ref!r}: {_e!r}", flush=True)
```

(Confirm the exact `todos` init function name via grep — the plan assumes `_init_todos_table`; if it differs, use the real one, or have the helper `CREATE TABLE IF NOT EXISTS todos (...)` matching the schema used by `_maybe_raise_cashout_review`.)

- [ ] **Step 3: Capture the result at both call sites**

Redirect (`begin_checkout_return`, ~app.py:9916) — change the bare call to capture + report:

```python
                _res = _osx.settle_paid_order_effects(
                    kind=_kind, order=_settle_order, md=md, pi_id=pi_id, sid=sid,
                    deps=_SETTLEMENT_DEPS)
                if _res and _res.get("skipped"):
                    _raise_settlement_skip_todo(inv, _res.get("kind"), _res["skipped"])
```

Webhook (`webhook_stripe`, ~app.py:27498) — same, inside the existing settlement try/except:

```python
                                            _res = _wosx.settle_paid_order_effects(
                                                kind=_wmd.get("kind") or "",
                                                order=(dict(_wro) if _wro else None),
                                                md=_wmd, pi_id=sess.get("payment_intent"),
                                                sid=session_id, deps=_SETTLEMENT_DEPS)
                                            if _res and _res.get("skipped"):
                                                _raise_settlement_skip_todo(inv, _res.get("kind"), _res["skipped"])
                                            _bos_orders.mark_order_settled(_wcx, _wo["id"])
```

(Keep `mark_order_settled` after — a skipped effect still marks settled; the todo is the retry signal. Do not reorder.)

- [ ] **Step 4: Run → PASS**

Run: `doppler run --config dev -- python3 -m pytest tests/test_settlement_skip_todo.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Regression + import**

Run: `doppler run --config dev -- python3 -c "import app; print('import ok')"` then
`doppler run --config dev -- python3 -m pytest tests/test_webhook_back_booking.py tests/test_begin_return_settlement.py -v`
Expected: `import ok`; PASS (settlement paths unchanged when `skipped` is empty — no todo raised).

- [ ] **Step 6: Commit**

```bash
git add app.py tests/test_settlement_skip_todo.py
git commit -m "feat(settlement): raise a deduped console todo when a settler is silently skipped"
```

---

## Full-suite gate

- [ ] Run: `doppler run --config dev -- python3 -m pytest tests/test_group_bundle_atomic.py tests/test_group_bundle_grant.py tests/test_settlement_skip_todo.py tests/test_webhook_back_booking.py tests/test_begin_return_settlement.py tests/test_order_settlement.py -v`
  Expected: PASS, or any failure also present on a `main` baseline of the same files.

## Post-deploy
- [ ] Confirm prod stays healthy after redeploy (app imports).
- [ ] (Optional) Verify a real group-window purchase still grants exactly one membership + one welcome.

## Notes
- Wallet margin was investigated and found already atomic (Postgres partial UNIQUE `(qbo_invoice_id, entry_type)` + `FOR UPDATE`) — no change; the #958 review's static flag did not account for the Postgres constraint.
- The skipped-todo makes a stranded effect visible + manually actionable; auto-retry is intentionally NOT built (would need settlement-claim machinery).
