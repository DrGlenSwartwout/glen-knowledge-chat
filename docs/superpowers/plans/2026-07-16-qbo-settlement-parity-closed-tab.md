# Per-Kind Settlement Parity on Closed Tabs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the closed-tab (Stripe-webhook-only) checkout path settle every per-kind side-effect the redirect handler settles, via one shared settlement definition both paths call — so a customer who closes the tab still gets their subscription row, biofield readiness, wallet credit, points, and referral.

**Architecture:** A new injected-deps orchestrator module `dashboard/order_settlement.py` holds the single per-kind dispatch (modeled on `dashboard/qbo_heal.py`). The four kind-specific side-effect blocks currently inline in `/begin/checkout-return` are extracted into named `app.py` functions and passed as `deps`. Both the redirect and the `webhook_stripe` book-back call the orchestrator. `subscriptions.create` gains a dedup guard (the one non-idempotent primitive).

**Tech Stack:** Python, Flask, SQLite, pytest, Stripe.

## Global Constraints

- **Spec:** `docs/superpowers/specs/2026-07-16-qbo-settlement-parity-closed-tab-design.md`.
- **Money path, no CI:** deploy-chat merge = deploy. Correctness over speed.
- **Run tests:** `doppler run --config dev -- python3 -m pytest <file>` — NEVER bare `pytest` (default Doppler config = prd, `DATA_DIR=/data`, breaks app-import collection) and NEVER the whole suite (it can send live email).
- **Do NOT touch** `biofield_local_app.py` or `dashboard/biofield_report_html.py` (unrelated dirty WIP).
- **Behavior-preserving extraction:** after the redirect refactor, `/begin/checkout-return` must settle each kind byte-for-byte as before. Characterization tests lock this.
- **Behavior-preserving per-kind membership (VERIFIED against code):** the redirect's shared gate `pi_id and (cid or _kind in ("retail","reorder","portal-reorder","subscribe","client"))` means ALL of retail/reorder/portal-reorder/subscribe/**client** get common `_settle_order_points`+`_settle_referral`, and biofield gets the same two via its own block. So `_COMMON_POINTS_KINDS` = all six. Do NOT drop client from the common set — a `client` order currently settles BOTH global-scope points (common) AND dispensary-scope points (its own block); that double-scope is pre-existing and MUST be preserved (the characterization tests lock it). Flagged to Glen as a possible latent oddity, not changed here.
- **Idempotent + best-effort:** every settler is keyed on `order_ref` and guards double-apply; the new subscription guard closes the last gap. In the webhook the orchestrator call is best-effort — a settler exception must never 500 the webhook or block the receipt.
- **`order_ref`** for a paid-only order = its `external_ref` = the `invoice_id` (checkout token) in session metadata `md`.

---

## Task 1: Subscription dedup guard

**Files:**
- Modify: `dashboard/subscriptions.py` — `init_subscriptions_table` (~line 94), `create` (~line 128)
- Test: `tests/test_subscriptions_dedup.py` (create)

**Interfaces:**
- Produces: `subscriptions.has_subscription_for_order(cx, order_ref) -> bool`; `subscriptions.create_once(cx, *, order_ref, **create_kwargs) -> int | None` (returns new rowid, or `None` if an order_ref row already existed); `create(cx, ..., order_ref=None)` now persists `order_ref`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_subscriptions_dedup.py`:

```python
import sqlite3
from dashboard import subscriptions as subs

def _mk():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    subs.init_subscriptions_table(cx)
    return cx

_KW = dict(email="a@b.com", stripe_customer_id="cus_1", stripe_payment_method_id="pm_1",
           items=[{"slug": "x", "qty": 1}], cadence_months=1, ship_address={}, next_charge_date="2026-08-16")

def test_create_once_dedups_on_order_ref():
    cx = _mk()
    first = subs.create_once(cx, order_ref="tok1", **_KW)
    second = subs.create_once(cx, order_ref="tok1", **_KW)
    assert first is not None
    assert second is None
    n = cx.execute("SELECT COUNT(*) FROM subscriptions WHERE order_ref='tok1'").fetchone()[0]
    assert n == 1

def test_has_subscription_for_order():
    cx = _mk()
    assert subs.has_subscription_for_order(cx, "tok9") is False
    subs.create_once(cx, order_ref="tok9", **_KW)
    assert subs.has_subscription_for_order(cx, "tok9") is True

def test_create_once_distinct_refs_both_insert():
    cx = _mk()
    assert subs.create_once(cx, order_ref="tokA", **_KW) is not None
    assert subs.create_once(cx, order_ref="tokB", **_KW) is not None
    assert cx.execute("SELECT COUNT(*) FROM subscriptions").fetchone()[0] == 2
```

- [ ] **Step 2: Run → FAIL**

Run: `doppler run --config dev -- python3 -m pytest tests/test_subscriptions_dedup.py -v`
Expected: FAIL (`create_once`/`has_subscription_for_order` undefined; no `order_ref` column).

- [ ] **Step 3: Add the column migration**

In `init_subscriptions_table` (after the existing `ALTER TABLE` migration blocks), add an idempotent column add mirroring the existing pattern (each wrapped so a re-run is a no-op):

```python
    try:
        cx.execute("ALTER TABLE subscriptions ADD COLUMN order_ref TEXT")
    except sqlite3.OperationalError:
        pass  # already migrated
```

- [ ] **Step 4: Persist `order_ref` in `create` and add the guards**

In `create(cx, *, email, ..., order_ref=None)` add the `order_ref=None` keyword param and include `order_ref` in the INSERT column list + values. Then add, after `create`:

```python
def has_subscription_for_order(cx, order_ref) -> bool:
    """True if a subscription row already exists for this originating order_ref."""
    if not order_ref:
        return False
    row = cx.execute(
        "SELECT 1 FROM subscriptions WHERE order_ref=? LIMIT 1", (order_ref,)).fetchone()
    return row is not None

def create_once(cx, *, order_ref, **create_kwargs):
    """Idempotent create keyed on order_ref: if a row already exists for this
    order_ref, no-op and return None; else create and return the new rowid.
    Closes the redirect-refresh and redirect-vs-webhook double-create."""
    if has_subscription_for_order(cx, order_ref):
        return None
    return create(cx, order_ref=order_ref, **create_kwargs)
```

(If `create` does not currently return the rowid, make it `return cx.execute(...).lastrowid` — confirm the existing return; keep it backward-compatible.)

- [ ] **Step 5: Run → PASS**

Run: `doppler run --config dev -- python3 -m pytest tests/test_subscriptions_dedup.py -v`
Expected: PASS (3 passed).

- [ ] **Step 6: Regression — existing subscription tests**

Run: `doppler run --config dev -- python3 -m pytest tests/test_subscriptions.py -v` (if it exists; else skip)
Expected: PASS (the new nullable column + params don't change existing behavior).

- [ ] **Step 7: Commit**

```bash
git add dashboard/subscriptions.py tests/test_subscriptions_dedup.py
git commit -m "feat(subs): order_ref column + create_once/has_subscription_for_order dedup guard"
```

---

## Task 2: The settlement orchestrator module

**Files:**
- Create: `dashboard/order_settlement.py`
- Test: `tests/test_order_settlement.py` (create)

**Interfaces:**
- Consumes: a `deps` object exposing callables `settle_points(order, order_ref)`, `settle_referral(order, order_ref)`, `ensure_subscription(md, pi_id)`, `grant_group_bundle(md, pi_id)`, `settle_client(md)`, `settle_biofield(md, sid)` (Task 3 supplies the real ones; tests use mocks).
- Produces: `settle_paid_order_effects(*, kind, order, md, pi_id, sid, deps) -> dict` returning `{"kind", "settled": [...], "skipped": [...]}`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_order_settlement.py`:

```python
from types import SimpleNamespace
from dashboard import order_settlement as osx

class _Deps:
    def __init__(self, raise_on=None):
        self.calls = []
        self._raise_on = raise_on or set()
    def _rec(self, name):
        self.calls.append(name)
        if name in self._raise_on:
            raise RuntimeError(f"boom:{name}")
    def settle_points(self, order, order_ref): self._rec("points")
    def settle_referral(self, order, order_ref): self._rec("referral")
    def ensure_subscription(self, md, pi_id): self._rec("subscription")
    def grant_group_bundle(self, md, pi_id): self._rec("group_bundle")
    def settle_client(self, md): self._rec("client")
    def settle_biofield(self, md, sid): self._rec("biofield")

_ORDER = {"id": 1, "email": "a@b.com"}
_MD = {"invoice_id": "tok1", "kind": "retail"}

def _run(kind, deps, order=_ORDER, md=None):
    return osx.settle_paid_order_effects(
        kind=kind, order=order, md=md or {"invoice_id": "tok1", "kind": kind},
        pi_id="pi_1", sid="sess_1", deps=deps)

def test_retail_settles_points_and_referral_only():
    d = _Deps(); out = _run("retail", d)
    assert d.calls == ["points", "referral"]
    assert set(out["settled"]) == {"points", "referral"}

def test_subscribe_adds_subscription_and_group_bundle():
    d = _Deps(); _run("subscribe", d)
    assert d.calls == ["points", "referral", "subscription", "group_bundle"]

def test_client_settles_common_points_and_client():
    # Behavior-preserving: client goes through the shared gate today, so it gets
    # common points+referral AND its own client settlement (dispensary-scope).
    d = _Deps(); out = _run("client", d)
    assert d.calls == ["points", "referral", "client"]
    assert set(out["settled"]) == {"points", "referral", "client"}

def test_biofield_settles_common_plus_biofield():
    d = _Deps(); _run("biofield", d)
    assert d.calls == ["points", "referral", "biofield"]

def test_reorder_and_portal_reorder_like_retail():
    for k in ("reorder", "portal-reorder"):
        d = _Deps(); _run(k, d)
        assert d.calls == ["points", "referral"]

def test_one_settler_raising_is_recorded_and_others_continue():
    d = _Deps(raise_on={"points"}); out = _run("subscribe", d)
    # points raises but referral/subscription/group_bundle still run
    assert d.calls == ["points", "referral", "subscription", "group_bundle"]
    assert "points" in out["skipped"]
    assert "referral" in out["settled"]

def test_no_order_skips_common_points_referral():
    d = _Deps(); out = _run("retail", d, order=None)
    assert d.calls == []
    assert out["settled"] == []

def test_unknown_kind_noop():
    d = _Deps(); out = _run("membership_product", d)
    assert d.calls == []
```

- [ ] **Step 2: Run → FAIL**

Run: `doppler run --config dev -- python3 -m pytest tests/test_order_settlement.py -v`
Expected: FAIL (`order_settlement` module does not exist).

- [ ] **Step 3: Write the orchestrator**

Create `dashboard/order_settlement.py`:

```python
"""Single per-kind settlement definition for a PAID checkout order, called by
BOTH the /begin/checkout-return redirect and the Stripe webhook book-back, so a
closed browser tab settles exactly what the redirect settles (no drift).

This module owns ONLY the dispatch. The actual side-effect functions live in
app.py (they need LOG_DB, Stripe, etc.) and are injected as `deps` -- mirroring
how qbo_heal takes find_receipt/book/stamp. It deliberately does NOT touch
mark-paid / receipt-booking / PI-stamp; those already fire correctly in each
caller. Every dep is idempotent per order_ref, so calling from both paths and
re-running is safe.
"""

# Kinds that earn loyalty points + referral credit the common way. VERIFIED
# against the redirect's shared gate (all of retail/reorder/portal-reorder/
# subscribe/client) plus biofield's own block -- so all six. client is INCLUDED
# on purpose: today it settles common (global-scope) points/referral AND its own
# dispensary-scope points; that pre-existing double-scope is preserved.
# Membership/subscription-product kinds are handled by their own _fulfill_*
# webhook fulfillers, not here.
_COMMON_POINTS_KINDS = {"retail", "reorder", "portal-reorder", "subscribe", "client", "biofield"}


def settle_paid_order_effects(*, kind, order, md, pi_id, sid, deps):
    """Run every per-kind side-effect for a paid order, idempotently, best-effort
    per effect. Returns {"kind", "settled": [names], "skipped": [names]}."""
    order_ref = (md or {}).get("invoice_id") or ""
    settled, skipped = [], []

    def _do(name, fn):
        try:
            fn()
            settled.append(name)
        except Exception as e:  # best-effort: one bad settler never aborts the rest
            print(f"[settlement] {name} failed kind={kind} ref={order_ref!r}: {e!r}", flush=True)
            skipped.append(name)

    if kind in _COMMON_POINTS_KINDS and order:
        _do("points", lambda: deps.settle_points(order, order_ref))
        _do("referral", lambda: deps.settle_referral(order, order_ref))

    if kind == "subscribe":
        _do("subscription", lambda: deps.ensure_subscription(md, pi_id))
        _do("group_bundle", lambda: deps.grant_group_bundle(md, pi_id))
    elif kind == "client":
        _do("client", lambda: deps.settle_client(md))
    elif kind == "biofield":
        _do("biofield", lambda: deps.settle_biofield(md, sid))

    return {"kind": kind, "settled": settled, "skipped": skipped}
```

- [ ] **Step 4: Run → PASS**

Run: `doppler run --config dev -- python3 -m pytest tests/test_order_settlement.py -v`
Expected: PASS (8 passed).

- [ ] **Step 5: Commit**

```bash
git add dashboard/order_settlement.py tests/test_order_settlement.py
git commit -m "feat(settlement): per-kind settlement orchestrator (injected deps, unit-tested)"
```

---

## Task 3: Extract redirect settlers into named deps + wire redirect to the orchestrator

This is the delicate task — it touches the working `/begin/checkout-return` money path. The change must be **behavior-preserving**: the redirect settles each kind exactly as before. Characterization tests (Step 1) lock current behavior and MUST pass unchanged after the refactor.

**Files:**
- Modify: `app.py` — extract four block bodies into module-level functions; build a `deps` object; replace the inline per-kind settler calls in `begin_checkout_return` (~9520–9810) with ONE `order_settlement.settle_paid_order_effects(...)` call.
- Test: `tests/test_begin_return_settlement.py` (create — characterization), plus re-run existing `tests/test_begin_checkout_paid_only.py`.

**Interfaces:**
- Consumes: `order_settlement.settle_paid_order_effects` (Task 2); `subscriptions.create_once`/`has_subscription_for_order` (Task 1).
- Produces (module-level in app.py):
  - `_ensure_subscription_row(md, pi_id)` — the subscribe block body (app.py ~9578–9620), calling `subscriptions.create_once(..., order_ref=md.get("invoice_id"))` instead of bare `create`.
  - `_grant_group_bundle(md, pi_id)` — the group-bundle block body (~9636–9678), unchanged logic.
  - `_settle_client_effects(md)` — the client block body (~9683–9760), unchanged logic.
  - `_settle_biofield_effects(md, sid)` — ONLY the biofield readiness seed (`biofield_store.seed_paid`, ~9737–9746) + care taster (`_fulfill_biofield_program(sid)`, ~9805). NOT points/referral/receipt/PI.
  - `_SETTLEMENT_DEPS` — a `SimpleNamespace` (or small class) with `settle_points=_settle_order_points_ref`, `settle_referral=_settle_referral_ref`, `ensure_subscription=_ensure_subscription_row`, `grant_group_bundle=_grant_group_bundle`, `settle_client=_settle_client_effects`, `settle_biofield=_settle_biofield_effects`. Provide thin adapters so signatures match the orchestrator's expected `settle_points(order, order_ref)` / `settle_referral(order, order_ref)` (they call `_settle_order_points(order, order_ref=order_ref)` / `_settle_referral(order, order_ref=order_ref)`).

- [ ] **Step 1: Write characterization tests FIRST (must pass BEFORE any refactor)**

Create `tests/test_begin_return_settlement.py`. DB-isolated (monkeypatch `app.LOG_DB` to a tmp db, seed orders via `dashboard.orders`), monkeypatch `stripe_pay.get_session`/`get_payment_intent` to return canned dicts, and SPY on the settler primitives to assert which fire per kind. Model the spies on the existing `tests/test_begin_checkout_paid_only.py` fixtures (read it first for the seeding + monkeypatch patterns). Cover, for a `payment_status="paid"` return:

```python
# For kind="retail": _settle_order_points AND _settle_referral called once each with order_ref=token;
#   book_sale_on_payment called; NO subscription/client/biofield effects.
# For kind="subscribe": points+referral called; subscriptions.create_once called once with order_ref=token.
# For kind="client": wallet.earn_dropship_margin called AND _settle_order_points called
#   (client passes the shared gate today — behavior-preserving; confirm this matches current code).
# For kind="biofield": biofield_store.seed_paid called; _settle_order_points AND _settle_referral called;
#   _fulfill_biofield_program called.
```

Run: `doppler run --config dev -- python3 -m pytest tests/test_begin_return_settlement.py -v`
Expected: PASS against the CURRENT (pre-refactor) code. If a spy assertion can't be made to pass against current code, the characterization is wrong — fix the test to match current behavior, not the code. These tests are the safety net; do not proceed until they are green on unchanged code. Commit them alone first:

```bash
git add tests/test_begin_return_settlement.py
git commit -m "test(begin-return): characterize per-kind settlement before refactor"
```

- [ ] **Step 2: Extract the four block bodies into module-level functions**

Move (cut, don't rewrite) the block bodies from `begin_checkout_return` into the four module-level functions named above, preserving every line of logic (the `try/except`, the `sqlite3.connect(LOG_DB)`, the metadata reads). Only change: in `_ensure_subscription_row`, swap the bare `_subs_ret.create(...)` for `_subs_ret.create_once(..., order_ref=(md.get("invoice_id") or ""))` and early-return if it returns `None`. Add the `_SETTLEMENT_DEPS` namespace and the two `settle_points`/`settle_referral` adapters near `_settle_order_points`.

- [ ] **Step 3: Replace the inline calls in `begin_checkout_return` with one orchestrator call**

In the paid block, after `pi_id`/`inv`/`cid`/`_kind` are computed and the PaymentIntent is stamped + the QBO receipt booked inline (LEAVE those inline — they are common, not per-kind), replace the removed per-kind settler calls with a single call. Look up the order once for the common-points path:

```python
                _settle_order = None
                if inv:
                    try:
                        _scx0 = _sqlite3.connect(LOG_DB); _scx0.row_factory = _sqlite3.Row
                        try:
                            _fo = _bos_orders.find_order_by_external_ref(_scx0, inv)
                            _settle_order = dict(_fo) if _fo else None
                        finally:
                            _scx0.close()
                    except Exception as _e:
                        print(f"[begin-return] settle order lookup: {_e!r}", flush=True)
                from dashboard import order_settlement as _osx
                _osx.settle_paid_order_effects(
                    kind=_kind, order=_settle_order, md=md, pi_id=pi_id, sid=sid,
                    deps=_SETTLEMENT_DEPS)
```

Keep the retail/biofield inline receipt-booking + PI-stamp exactly where they are. Remove ONLY the points/referral/subscribe/group-bundle/client/biofield-seed/care-taster calls that now live in the deps. The `in-house` and `founding_reserve` blocks are untouched (out of scope — not paid-only per-kind settlement).

**Preserve the existing gating:** the old common points/referral fired only under `pi_id and (cid or _kind in (...))`. Place the orchestrator call so that guard is preserved (paid card orders always have `pi_id`). The characterization tests (Step 1) are the arbiter — if moving the call changes which settlers fire for any kind, the gating was not preserved; fix the placement, not the test. If a characterization test written against current code contradicts this plan's stated per-kind expectation, current code wins — update the plan's expectation and tell the controller.

- [ ] **Step 4: Run characterization + paid-only regression → PASS unchanged**

Run: `doppler run --config dev -- python3 -m pytest tests/test_begin_return_settlement.py tests/test_begin_checkout_paid_only.py -v`
Expected: PASS — identical settlement behavior. If any characterization assertion changed, the extraction altered behavior; fix the extraction, not the test.

- [ ] **Step 5: Confirm import + no orphaned references**

Run: `doppler run --config dev -- python3 -c "import app; print('import ok')"`
Expected: `import ok` (no NameError from a half-moved reference). Grep to confirm the old inline settler calls are gone from `begin_checkout_return` and now only exist in the extracted functions.

- [ ] **Step 6: Commit**

```bash
git add app.py
git commit -m "refactor(begin-return): extract per-kind settlers to deps + dispatch via order_settlement (behavior-preserving)"
```

---

## Task 4: Wire the webhook to the orchestrator (the parity fix)

**Files:**
- Modify: `app.py` — `webhook_stripe` `checkout.session.completed` book-back block (~27338–27365), inside the existing `try/except`, after `book_sale_on_payment`.
- Test: `tests/test_webhook_back_booking.py` (extend)

**Interfaces:**
- Consumes: `order_settlement.settle_paid_order_effects`, `_SETTLEMENT_DEPS` (Task 3).

- [ ] **Step 1: Write the failing tests (extend the existing webhook test file)**

Add to `tests/test_webhook_back_booking.py` (reuse its existing seeding + monkeypatch harness). For a `checkout.session.completed` webhook where the redirect never ran (order unbooked), assert the per-kind side-effect fires:

```python
# 1. Closed-tab biofield: webhook books receipt AND seeds readiness (biofield_store.seed_paid called).
# 2. Closed-tab subscribe: webhook creates the subscription row (subscriptions.create_once called, one row).
# 3. Closed-tab client: webhook credits wallet (wallet.earn_dropship_margin called).
# 4. Closed-tab retail: webhook settles points + referral (_settle_order_points + _settle_referral called).
# 5. Idempotent: running the same webhook twice does not double-create the subscription / double-award.
# 6. Best-effort: a settler raising still returns 200 and the receipt is still booked.
```

Match the session metadata shape each kind uses (`kind`, `invoice_id`, and the kind-specific keys the extracted deps read: subscribe→cadence_months/items/ship/stash_key/email; client→practitioner_id/margin_cents/patient_email/subtotal_cents; biofield→email). Read the extracted deps (Task 3) to see exactly which `md` keys each reads, and set them in the test session.

Run: `doppler run --config dev -- python3 -m pytest tests/test_webhook_back_booking.py -v`
Expected: the new tests FAIL (webhook doesn't settle yet); the pre-existing webhook tests still PASS.

- [ ] **Step 2: Add the orchestrator call to the webhook book-back**

Inside the existing `if _wo and _wo["qbo_lines_json"] and not _wo["qbo_sales_receipt_id"]:` block, AFTER `book_sale_on_payment(...)`, add (still inside the outer `try/except` so it stays best-effort):

```python
                            # Closed-tab parity: settle the same per-kind side-effects the
                            # redirect settles, via the shared orchestrator. Idempotent +
                            # best-effort -- a settler raising must not 500 the webhook.
                            try:
                                _md = sess.get("metadata") or {}
                                _reo = _bos_orders.find_order_by_external_ref(_wcx, inv)
                                order_settlement.settle_paid_order_effects(
                                    kind=_md.get("kind") or "",
                                    order=(dict(_reo) if _reo else None),
                                    md=_md, pi_id=sess.get("payment_intent"),
                                    sid=session_id, deps=_SETTLEMENT_DEPS)
                            except Exception as _se:
                                print(f"[stripe-webhook] settlement failed: {_se!r}", flush=True)
```

Confirm `order_settlement` is imported (add `from dashboard import order_settlement` at the top of the block or module) and `_SETTLEMENT_DEPS` is in scope (module-level from Task 3).

- [ ] **Step 3: Run → PASS**

Run: `doppler run --config dev -- python3 -m pytest tests/test_webhook_back_booking.py -v`
Expected: PASS (all, including the new closed-tab per-kind + idempotency + best-effort tests).

- [ ] **Step 4: Import check**

Run: `doppler run --config dev -- python3 -c "import app; print('import ok')"`
Expected: `import ok`.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_webhook_back_booking.py
git commit -m "feat(webhook): settle per-kind side-effects on closed-tab completion (parity)"
```

---

## Full-suite gate

- [ ] Run: `doppler run --config dev -- python3 -m pytest tests/test_order_settlement.py tests/test_subscriptions_dedup.py tests/test_begin_return_settlement.py tests/test_begin_checkout_paid_only.py tests/test_webhook_back_booking.py tests/test_book_sale_on_payment.py -v`
  Expected: PASS, or any failure also present on `main` (diff against a `main` baseline run of the same files, per the "suite green ≠ task green" rule).

- [ ] Post-deploy (deployed env): trigger a real closed-tab checkout per kind (or replay a `checkout.session.completed` webhook for a seeded unbooked order) and confirm the side-effect lands — subscription row created, biofield readiness seeded (scan-booking unlocks), client wallet credited, retail points/referral credited — with no double-award when the redirect also fires.

## Notes

- Closes the "full per-kind settlement parity on closed tab (points/referral/sub-row)" follow-on tracked in `feedback_qbo_paid_only_no_unpaid_invoices`.
- The mark-paid split (redirect books receipt/points; webhook marks paid) is pre-existing and out of scope — the order still gets marked paid by the webhook.
- After Task 3, there is ONE per-kind settlement definition. A future new kind or settler is added in `order_settlement` + its dep, and both paths get it automatically — the drift that caused this bug cannot recur.
