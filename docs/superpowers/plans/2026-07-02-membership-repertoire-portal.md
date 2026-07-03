# Membership Repertoire + Portal Reorder + Analysis Cadence — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give paid members a $50 flat reorder price on their personal "repertoire" (SKUs they've bought), surfaced in the client portal with one-click reorder and a personalized upsell, and cap free-tier analysis to 1/month.

**Architecture:** Three additive, flag-dark slices on the existing SQLite/Flask app. (1) A new `dashboard/repertoire.py` persists an email-keyed set of SKUs, seeded from windowed order history on membership conversion and appended on each member purchase; its price is layered into the existing `pricing.compute()` engine, gated on `_is_paid_member` at read time (no per-SKU decay). (2) The portal reorder list is re-sourced from real order history (`orders` table) and shows member savings + grayed-out "unlock on commitment" rows + a personalized upsell. (3) A new `dashboard/analysis_quota.py` enforces 1 free analysis/calendar-month on the request route; paid members bypass.

**Tech Stack:** Python 3, Flask monolith (`app.py`), SQLite (`chat_log.db`), pure `dashboard/*.py` modules (take a `cx` connection), pytest.

## Global Constraints

- **DB:** SQLite `chat_log.db`. Writes: `with _db_lock, sqlite3.connect(LOG_DB) as cx:`. Reads: `with sqlite3.connect(LOG_DB) as cx:` (+ `cx.row_factory = sqlite3.Row` where dict rows needed). New logic modules are pure (take `cx`), following `dashboard/biofield_store.py` house style.
- **Emails** are stored/compared lowercased (`(email or "").strip().lower()`).
- **Idempotency** for any exactly-once effect: `INSERT OR IGNORE` into a PK'd claim table, act only if `rowcount == 1` (mirror `_fulfill_prepay_term`, app.py:7148).
- **Migrations** are idempotent Python fns (`init_*_table`, `migrate_add_*`) called at startup AND defensively where read/written. No framework runner. The `/migrations/*.sql` dir is Postgres-only — DO NOT use it.
- **Flags (default OFF → prod byte-unchanged):** `REPERTOIRE_ENABLED` (repertoire seeding + pricing + portal display), `ANALYSIS_QUOTA_ENABLED` (cadence gate). Read via `os.environ.get(...).strip().lower() in ("1","true","yes","on")` (house idiom, see app.py:4601).
- **Member =** `_is_paid_member(email)` (app.py:4761) — active membership AND category != 'trial'. This is the read-time gate for repertoire pricing and quota bypass.
- **Money** is integer cents. Never render a member price the checkout won't honor — all customer prices route through `pricing.compute()` / `_price_cart` (app.py:4911).
- **No per-SKU decay:** repertoire pricing is gated on *current* `_is_paid_member` at read time; the window only governs seed-depth + portal display, never eligibility.
- **Repertoire price = flat member reorder rate**, default `repertoire_reorder_pct = 0.29` (→ ~$50 on a $69.97 list), console-tunable, floored via `pricing.apply_discount`. It is additive/best-of with existing discounts (`max()`), never stacked.
- **Test runs:** `doppler run -p remedy-match -c dev -- python3 -m pytest tests/<file> -v` (app import needs Pinecone creds; dev config). Full suite is pollution/order-noisy — judge regressions by per-file isolation or a same-run `comm -23` diff vs merge-base, NEVER raw pass/fail counts.
- **DO NOT** revert #489 / change `same_sku_pct` gating — `same_sku` (same-SKU multiples) stays PUBLIC by design. Cross-SKU (`program_total`) is already members-only; `open_total` stays OFF.

---

## File Structure

**Slice A — Repertoire core (own PR, `REPERTOIRE_ENABLED`):**
- Create `dashboard/repertoire.py` — table + seed + add + read. Pure, takes `cx`.
- Create `tests/test_repertoire.py`.
- Modify `dashboard/pricing.py` — add `repertoire_reorder_pct` to DEFAULTS + `repertoire_slugs` param to `compute()`.
- Modify `tests/test_pricing*.py` (or new `tests/test_repertoire_pricing.py`).

**Slice B — Membership wiring (same PR as A or fast-follow):**
- Modify `app.py` — startup `init_repertoire_table`; seed in `_fulfill_prepay_term` (7148) + `_fulfill_continuous_care_monthly` (7217); append repertoire on order fulfillment; thread `repertoire_slugs` through `_price_cart` (4911) → `compute`.
- Modify `tests/test_prepay_checkout.py` / `tests/test_continuous_care_monthly.py` (seed assertions) + new `tests/test_repertoire_wiring.py`.

**Slice C — Portal reorder module (own PR, reuses `REPERTOIRE_ENABLED`):**
- Modify `app.py` — `api_client_portal` (12842) payload: reorder list from order history + repertoire price + savings + grayed unlock rows + upsell numbers.
- Modify `static/client-portal.html` — render list, savings, grayed rows, upsell CTA.
- Create `tests/test_portal_reorder_module.py`.

**Slice D — Analysis cadence gate (own PR, `ANALYSIS_QUOTA_ENABLED`):**
- Create `dashboard/analysis_quota.py` — per-email calendar-month claim + check.
- Create `tests/test_analysis_quota.py`.
- Modify `app.py` — gate `api_portal_biofield_request` (13680) + `biofield_request` (14827); paid bypass; startup init.

---

## Task 1: Repertoire store module

**Files:**
- Create: `dashboard/repertoire.py`
- Test: `tests/test_repertoire.py`

**Interfaces:**
- Produces:
  - `init_repertoire_table(cx) -> None`
  - `add_skus(cx, email, slugs, *, at=None) -> int` (idempotent; returns count newly inserted)
  - `repertoire_slugs(cx, email) -> set[str]` (all SKUs in the email's repertoire, lowercased)
  - `seed_from_history(cx, email, window_days, *, order_slugs_fn) -> int` (adds distinct non-cancelled SKUs bought within window; `order_slugs_fn(cx, email, window_days) -> list[str]` injected for testability)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_repertoire.py
import sqlite3
from dashboard import repertoire

def _cx():
    cx = sqlite3.connect(":memory:")
    repertoire.init_repertoire_table(cx)
    return cx

def test_add_and_read_skus_lowercased_and_deduped():
    cx = _cx()
    n1 = repertoire.add_skus(cx, "Glen@Example.com", ["Neuro-Mag", "neuro-mag", "terrain-restore"])
    assert n1 == 2  # deduped case-insensitively
    assert repertoire.repertoire_slugs(cx, "glen@example.com") == {"neuro-mag", "terrain-restore"}
    n2 = repertoire.add_skus(cx, "glen@example.com", ["neuro-mag"])  # already present
    assert n2 == 0

def test_seed_from_history_window():
    cx = _cx()
    def fake_history(cx_, email, window_days):
        assert window_days == 90
        return ["a", "b", "b"]  # b duplicated
    added = repertoire.seed_from_history(cx, "x@y.com", 90, order_slugs_fn=fake_history)
    assert added == 2
    assert repertoire.repertoire_slugs(cx, "x@y.com") == {"a", "b"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_repertoire.py -v`
Expected: FAIL — `ModuleNotFoundError` / `AttributeError: module 'dashboard.repertoire' has no attribute ...`

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/repertoire.py
"""Per-member SKU repertoire (email-keyed). Members get a flat reorder price on
these SKUs; first buy of a SKU is regular. No per-SKU decay — pricing eligibility
is gated on active membership at read time, not stored here. Pure: caller passes cx."""
from datetime import datetime, timezone


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _norm(email):
    return (email or "").strip().lower()


def init_repertoire_table(cx):
    cx.execute(
        """CREATE TABLE IF NOT EXISTS repertoire (
             email TEXT NOT NULL,
             slug  TEXT NOT NULL,
             added_at TEXT NOT NULL,
             PRIMARY KEY (email, slug)
           )"""
    )
    cx.execute("CREATE INDEX IF NOT EXISTS ix_repertoire_email ON repertoire(email)")
    cx.commit()


def add_skus(cx, email, slugs, *, at=None):
    email = _norm(email)
    at = at or _now_iso()
    seen, added = set(), 0
    for s in slugs:
        s = (s or "").strip().lower()
        if not s or s in seen:
            continue
        seen.add(s)
        if cx.execute(
            "INSERT OR IGNORE INTO repertoire(email, slug, added_at) VALUES (?,?,?)",
            (email, s, at),
        ).rowcount == 1:
            added += 1
    cx.commit()
    return added


def repertoire_slugs(cx, email):
    email = _norm(email)
    return {
        r[0]
        for r in cx.execute("SELECT slug FROM repertoire WHERE email=?", (email,))
    }


def seed_from_history(cx, email, window_days, *, order_slugs_fn):
    slugs = order_slugs_fn(cx, _norm(email), window_days) or []
    return add_skus(cx, email, slugs)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_repertoire.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add dashboard/repertoire.py tests/test_repertoire.py
git commit -m "feat(repertoire): email-keyed SKU repertoire store + windowed seed"
```

---

## Task 2: Repertoire price in the pricing engine

**Files:**
- Modify: `dashboard/pricing.py` (DEFAULTS L4-25; `compute` L123-184)
- Test: `tests/test_repertoire_pricing.py`

**Interfaces:**
- Consumes: `pricing.load_settings`, `pricing.apply_discount`, `pricing._ramp_pct` (existing).
- Produces: `compute(..., repertoire_slugs=None)` — when the buyer is a member (caller passes the set only for members) and a line's `slug` ∈ `repertoire_slugs` and the line is `volume_eligible`, that line's discount floor becomes at least `settings['repertoire_reorder_pct']` (best-of with existing via `max`). Non-members: caller passes `None` → behavior unchanged.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_repertoire_pricing.py
from dashboard import pricing

def _item(slug, qty=1, list_cents=6997):
    return {"slug": slug, "name": slug, "qty": qty, "list_cents": list_cents,
            "months": qty, "volume_eligible": True}

def test_member_repertoire_sku_gets_reorder_rate():
    s = pricing.load_settings({})
    # qty 1 => same_sku ramp gives 0%; regular price = list
    out_reg = pricing.compute([_item("neuro-mag", 1)], settings=s)
    assert out_reg["lines"][0]["unit_cents"] == 6997
    # member with slug in repertoire => ~29% off (repertoire_reorder_pct default)
    out_mem = pricing.compute([_item("neuro-mag", 1)], settings=s,
                              repertoire_slugs={"neuro-mag"})
    assert out_mem["lines"][0]["unit_cents"] < 5100  # ~$50

def test_non_repertoire_sku_unchanged_for_member():
    s = pricing.load_settings({})
    out = pricing.compute([_item("brand-new", 1)], settings=s,
                          repertoire_slugs={"neuro-mag"})
    assert out["lines"][0]["unit_cents"] == 6997  # first buy = regular
```

- [ ] **Step 2: Run test to verify it fails**

Run: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_repertoire_pricing.py -v`
Expected: FAIL — `compute() got an unexpected keyword argument 'repertoire_slugs'`

- [ ] **Step 3: Write minimal implementation**

In `dashboard/pricing.py` DEFAULTS (after `volume_anchors`, ~L15) add:
```python
    "repertoire_reorder_pct": 0.29,   # member flat reorder rate on repertoire SKUs (~$50 on $69.97)
```
Change `compute` signature (L123) to add `repertoire_slugs=None`. Inside the per-line loop (currently L148-162), where `line_pct = max(t1, order_pct, base_pct)` is computed (~L156), replace with:
```python
            rep_pct = 0.0
            if repertoire_slugs and eligible and (it.get("slug") or "").strip().lower() in repertoire_slugs:
                rep_pct = float(settings.get("repertoire_reorder_pct") or 0.0)
            line_pct = max(t1, order_pct, base_pct, rep_pct)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_repertoire_pricing.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Regression-check pricing + commit**

Run: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_pricing.py tests/test_repertoire_pricing.py -v` (expect existing pricing tests still green — `repertoire_slugs` defaults None).
```bash
git add dashboard/pricing.py tests/test_repertoire_pricing.py
git commit -m "feat(pricing): member repertoire reorder rate, additive best-of in compute"
```

---

## Task 3: Wire repertoire into checkout pricing (`_price_cart`) + startup init

**Files:**
- Modify: `app.py` — `_price_cart` (4911, forward new kwarg to `compute` at 4942), startup table init (near other `init_*` calls, e.g. app.py:6767), and the `_price_cart` callers that price for a known member (biofield preview 2830, `begin_checkout` 6983, `_checkout_cart` 15515, invoice edit 28153, reorder_subscribe 15762).
- Test: `tests/test_repertoire_wiring.py`

**Interfaces:**
- Consumes: `repertoire.repertoire_slugs(cx, email)`, `_is_paid_member(email)`, `REPERTOIRE_ENABLED`.
- Produces: `_price_cart(..., email=None)` resolves `repertoire_slugs` internally = `repertoire.repertoire_slugs(cx, email)` **iff** `REPERTOIRE_ENABLED and _is_paid_member(email)`, else `None`, and forwards to `compute`.

- [ ] **Step 1: Write the failing test** — a member with a seeded repertoire SKU is priced at the reorder rate through `_price_cart`; flag off → regular. (Use the app test harness pattern from `tests/test_prepay_checkout.py`; seed a membership + a repertoire row in the temp `chat_log.db`, monkeypatch `REPERTOIRE_ENABLED=True`.)

```python
# tests/test_repertoire_wiring.py  (sketch — follow test_prepay_checkout.py harness)
def test_price_cart_applies_repertoire_for_member(app_db, monkeypatch):
    import app as A
    monkeypatch.setattr(A, "REPERTOIRE_ENABLED", True)
    # seed active membership for member@x.com and repertoire slug 'neuro-mag'
    _seed_active_membership(app_db, "member@x.com")
    with A._db_lock, __import__("sqlite3").connect(A.LOG_DB) as cx:
        A.repertoire.add_skus(cx, "member@x.com", ["neuro-mag"])
    out = A._price_cart([{"slug": "neuro-mag", "qty": 1, "list_cents": 6997,
                          "months": 1, "volume_eligible": True}],
                        ship=None, email="member@x.com")
    assert out["lines"][0]["unit_cents"] < 5100
```

- [ ] **Step 2: Run to verify it fails** (`_price_cart` has no `email` param yet / no repertoire applied).

- [ ] **Step 3: Implement** — add `REPERTOIRE_ENABLED` global (near app.py:4601 flags); add `email=None` to `_price_cart` (if not already threaded — several callers already pass `email`/`program_member`); inside `_price_cart`, before calling `compute`:
```python
    rep = None
    if REPERTOIRE_ENABLED and email and _is_paid_member(email):
        with sqlite3.connect(LOG_DB) as _cx:
            rep = repertoire.repertoire_slugs(_cx, email)
    # ... pass repertoire_slugs=rep into _pricing.compute(...)
```
Add `import dashboard.repertoire as repertoire` and `repertoire.init_repertoire_table(cx)` at the startup init site. Ensure each member-known caller passes `email=` to `_price_cart` (verify 2830/6983/15515/28153/15762 already have the email in scope — they compute `_is_paid_member(email)` already, so email is available).

- [ ] **Step 4: Run to verify pass** (`tests/test_repertoire_wiring.py`), then flag-off regression: unset the monkeypatch → regular price.

- [ ] **Step 5: Commit** — `feat(repertoire): apply repertoire pricing in _price_cart for active members (flag-dark)`

---

## Task 4: Seed repertoire on membership conversion + append on purchase

**Files:**
- Modify: `app.py` — `_fulfill_prepay_term` (7148), `_fulfill_continuous_care_monthly` (7217): after the won-claim grant, seed repertoire from windowed history. Order-fulfillment path (`_ingest_order` / where a paid order is recorded, e.g. portal checkout 13342 + `_price_cart` callers) : append the ordered slugs to the buyer's repertoire when they are a member.
- Create helper: `_order_slugs_since(cx, email, window_days)` in app.py — distinct non-cancelled `orders.items_json[].slug` where `created_at >= now-window`.
- Test: extend `tests/test_repertoire_wiring.py`.

**Interfaces:**
- Consumes: `orders.list_orders_by_email` (orders.py:213), `prepay.term_days`/tier months, `repertoire.seed_from_history`, `repertoire.add_skus`.
- Window by plan: 1mo→90d, 6mo→180d, 12mo→365d (map from the grant's tier/term_months; monthly Continuous Care term_months drives it).

- [ ] **Step 1: Write failing test** — fulfilling a 6-month prepay for an email that has 2 non-cancelled orders (SKUs a,b) in the last 180d seeds {a,b} into repertoire; a cancelled order's SKU is excluded.

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement** `_order_slugs_since`:
```python
def _order_slugs_since(cx, email, window_days):
    import json
    from datetime import datetime, timedelta, timezone
    cutoff = (datetime.now(timezone.utc) - timedelta(days=int(window_days))).isoformat()
    cx.row_factory = sqlite3.Row
    rows = cx.execute(
        "SELECT items_json FROM orders WHERE lower(email)=? "
        "AND status!='cancelled' AND created_at>=?",
        ((email or "").strip().lower(), cutoff)).fetchall()
    out = []
    for r in rows:
        for it in (json.loads(r["items_json"] or "[]") or []):
            s = (it.get("slug") or "").strip().lower()
            if s:
                out.append(s)
    return out
```
In each fulfiller's won-claim branch, add (window from tier):
```python
        try:
            repertoire.init_repertoire_table(cx)
            repertoire.seed_from_history(cx, email, _window_days_for_term(term_months),
                                         order_slugs_fn=_order_slugs_since)
        except Exception:
            pass  # seeding is best-effort; never block fulfillment
```
Add `_window_days_for_term(m) -> 90 if m<=1 else 180 if m<=6 else 365`. In the order-recording path, when `_is_paid_member(email)`, `repertoire.add_skus(cx, email, [line slugs])`.

- [ ] **Step 4: Run to verify pass.**

- [ ] **Step 5: Commit** — `feat(repertoire): seed from windowed history on conversion + append on member orders`

---

## Task 5: Portal reorder module — payload

**Files:**
- Modify: `app.py` — `api_client_portal` (12842), the `reorder_items` build block (~12889-12928).
- Test: `tests/test_portal_reorder_module.py`

**Interfaces:**
- Consumes: `orders.list_orders_by_email`, `_get_product(slug)`, `_price_cart`/`_inhouse_line_unit_cents`, `repertoire.repertoire_slugs`, `portal_offers.next_offers`.
- Produces: payload key `reorder` = list of `{slug, name, qty, regular_cents, your_cents, is_member_price, in_repertoire}` sourced from the client's **portal-channel** order history ONLY (distinct SKUs, `source in ('portal-reorder','reorder')`, non-cancelled) — per Glen's explicit "bought here (their portal, not other channels)". NOTE the deliberate split: the DISPLAY list is portal-only, but repertoire **pricing** (Task 4) is all-channel (a proven remedy is proven regardless of purchase channel). Plus `membership_upsell = {reorders_30d, spend_30d_cents, member_would_pay_cents, savings_cents, net_after_fee_cents}` and `locked_rows` = repertoire-eligible SKUs from 90–365d ago shown grayed with the commitment tier that unlocks them.

- [ ] **Step 1: Write failing test** — portal payload for a client with order history returns a `reorder` list with distinct SKUs and, for a member, `your_cents < regular_cents` on repertoire SKUs; `membership_upsell.savings_cents` computed from last-30d history for a non-member.

- [ ] **Step 2: Run to verify fail.**

- [ ] **Step 3: Implement** — replace the curated-`reorder_items` source with a history-derived list: `orders.list_orders_by_email(cx, email)`, dedupe by slug (keep most recent qty), drop cancelled + non-purchasable, enrich each via `_get_product`; compute `regular_cents` and `your_cents` (member → through repertoire pricing; non-member → regular). Compute `membership_upsell` from last-30d orders: `spend_30d`, `member_would_pay` (re-price those lines as if member), `savings = spend - member_would_pay`, `net_after_fee = MONTHLY_ANCHOR_CENTS - savings`. Build `locked_rows` from 90–365d history SKUs not already discounted, tagged with the tier (6mo/12mo) whose window would seed them. Gate the whole block behind `REPERTOIRE_ENABLED` (fall back to today's `reorder_items` when off).

- [ ] **Step 4: Run to verify pass.**

- [ ] **Step 5: Commit** — `feat(portal): reorder module payload from real history + member savings + upsell`

---

## Task 6: Portal reorder module — UI

**Files:**
- Modify: `static/client-portal.html`
- Test: manual render-verify (documented) + reuse `tests/test_portal_reorder_module.py` for payload.

- [ ] **Step 1:** Add a "Your Remedies" section rendering `payload.reorder`: each row = name, qty stepper, `your_cents` (with `regular_cents` struck through + "you save $X" when `is_member_price`), and a **Reorder** button that adds to the existing portal checkout (`POST /api/portal/<token>/checkout`) — **add-then-confirm**, never silent charge (reuse the existing confirm step). Latch the button so it can't double-fire (one-shot `disabled` on click).
- [ ] **Step 2:** Render `locked_rows` grayed with forward-framed copy: *"Unlock $50 pricing on these when you commit to 6/12 months"* — never "you overpaid."
- [ ] **Step 3:** Render `membership_upsell` positively: their real 30-day savings + what the fee nets, benefits list (live Zoom Q&A + group coaching, expanded education, $50 repertoire). CTA → the existing membership/prepay checkout.
- [ ] **Step 4:** Render-verify locally (documented steps): load a `/portal/<token>` for a seeded member + non-member; confirm prices match `/api/portal/<token>` payload exactly and no console errors.
- [ ] **Step 5: Commit** — `feat(portal): render reorder module, member savings, forward-framed locked rows + upsell`

---

## Task 7: Analysis quota store

**Files:**
- Create: `dashboard/analysis_quota.py`
- Test: `tests/test_analysis_quota.py`

**Interfaces:**
- Produces:
  - `init_analysis_quota_table(cx) -> None`
  - `try_claim(cx, email, *, month=None) -> bool` — atomic; True on first claim in the calendar month (`YYYY-MM`), False if already claimed.
  - `claimed_this_month(cx, email, *, month=None) -> bool`

- [ ] **Step 1: Write failing test**

```python
# tests/test_analysis_quota.py
import sqlite3
from dashboard import analysis_quota as q

def _cx():
    cx = sqlite3.connect(":memory:"); q.init_analysis_quota_table(cx); return cx

def test_one_claim_per_calendar_month():
    cx = _cx()
    assert q.try_claim(cx, "a@b.com", month="2026-07") is True
    assert q.try_claim(cx, "a@b.com", month="2026-07") is False   # second in same month
    assert q.try_claim(cx, "a@b.com", month="2026-08") is True    # new month ok
    assert q.claimed_this_month(cx, "a@b.com", month="2026-08") is True
```

- [ ] **Step 2: Run to verify fail.**

- [ ] **Step 3: Implement** (mirror `biofield_free_unlocks` atomic-claim idiom):
```python
# dashboard/analysis_quota.py
"""Free-tier analysis cadence: 1 request per calendar month per email.
Paid members bypass entirely (checked by caller). Atomic INSERT OR IGNORE claim."""
from datetime import datetime, timezone

def _month(month=None):
    return month or datetime.now(timezone.utc).strftime("%Y-%m")

def _norm(email):
    return (email or "").strip().lower()

def init_analysis_quota_table(cx):
    cx.execute("""CREATE TABLE IF NOT EXISTS analysis_quota (
        email TEXT NOT NULL, month TEXT NOT NULL, claimed_at TEXT NOT NULL,
        PRIMARY KEY (email, month))""")
    cx.commit()

def try_claim(cx, email, *, month=None):
    ok = cx.execute(
        "INSERT OR IGNORE INTO analysis_quota(email, month, claimed_at) VALUES (?,?,?)",
        (_norm(email), _month(month), datetime.now(timezone.utc).isoformat())
    ).rowcount == 1
    cx.commit()
    return ok

def claimed_this_month(cx, email, *, month=None):
    return cx.execute("SELECT 1 FROM analysis_quota WHERE email=? AND month=?",
                      (_norm(email), _month(month))).fetchone() is not None
```

- [ ] **Step 4: Run to verify pass.**

- [ ] **Step 5: Commit** — `feat(analysis-quota): per-email calendar-month claim store`

---

## Task 8: Gate the analysis request routes

**Files:**
- Modify: `app.py` — `api_portal_biofield_request` (13680), `biofield_request` (14827); startup `init_analysis_quota_table`.
- Test: `tests/test_analysis_quota_gate.py`

**Interfaces:**
- Consumes: `analysis_quota.try_claim`, `_is_paid_member`, `_active_membership_for_email`, `ANALYSIS_QUOTA_ENABLED`.
- Behavior: if `ANALYSIS_QUOTA_ENABLED` and NOT `_active_membership_for_email(email)` (free tier) → `try_claim`; if False (already used this month) return a friendly 200 `{"ok": False, "reason": "monthly_quota", "next": "<1st of next month>"}` without enqueuing the request. Paid members always pass. Flag off → unchanged.

- [ ] **Step 1: Write failing test** — free-tier email: first request this month enqueues (status→requested); second returns `monthly_quota` and does NOT change status. Paid member: unlimited.
- [ ] **Step 2: Run to verify fail.**
- [ ] **Step 3: Implement** the guard at the top of both request handlers (after email resolved, before `_biofield_transition`/email send):
```python
    if ANALYSIS_QUOTA_ENABLED and not _active_membership_for_email(email):
        with _db_lock, sqlite3.connect(LOG_DB) as cx:
            analysis_quota.init_analysis_quota_table(cx)
            if not analysis_quota.try_claim(cx, email):
                return jsonify(ok=False, reason="monthly_quota"), 200
```
- [ ] **Step 4: Run to verify pass** + flag-off regression.
- [ ] **Step 5: Commit** — `feat(analysis-quota): gate free-tier analysis requests to 1/calendar-month (flag-dark)`

---

## Deferred (explicitly OUT of this plan — log, don't silently drop)

- **Auto-analysis-on-scan for members.** Generation runs off-server (`biofield_local_app.py` on Glen's Mac). "Unlimited" is covered (paid bypass); "automatic on every scan" needs the local engine to learn which emails are paid — cross-machine work, separate effort.
- **Christmas/seasonal prepay discount** — occasional, capped; build when scheduling that promo.
- **Console tuning UI** for window lengths + `repertoire_reorder_pct` — ships via existing `pricing-settings.json` editor; expose the new key there as a follow-up.

## Go-live checklist (per slice, after merge)

- Flip `REPERTOIRE_ENABLED=1` (Render env, per `render.yaml is NOT live-source` — set via API), render-verify: a seeded member sees $50 repertoire prices in portal == checkout price; a non-member sees regular; storefront `same_sku` bulk pricing UNCHANGED.
- Flip `ANALYSIS_QUOTA_ENABLED=1`, verify a free-tier second request in a month is blocked; a paid member is unlimited.
- Confirm no live storefront price moved (no #489 revert was performed).

---

## Self-Review

**Spec coverage:** repertoire (Tasks 1–4) ✓; reorders-not-first-buy (Task 2 — only `volume_eligible` lines in repertoire discount; first buy of a new SKU has slug∉repertoire) ✓; retroactive seed from window, no per-SKU decay (Task 4 seed + Task 3 read-time membership gate) ✓; portal reorder list + one-click + price-through-engine + grayed forward-framed rows + personalized upsell (Tasks 5–6) ✓; cadence gate 1/mo free, paid unlimited (Tasks 7–8) ✓; same-SKU public / cross-SKU gated unchanged (Global Constraints — no revert) ✓; auto-on-scan deferred (logged) ✓.

**Placeholder scan:** integration steps in Tasks 3–6 reference exact app.py line anchors + real function names; the app-harness test bodies (Tasks 3–5) are sketched against `tests/test_prepay_checkout.py` conventions rather than fully written because they need the repo's DB-temp fixture — the implementer must open that fixture. This is the one area to flesh out at execution time.

**Type consistency:** `repertoire_slugs` is a `set[str]` (lowercased) everywhere (Tasks 1→2→3). `try_claim`/`claimed_this_month` month arg is `YYYY-MM` string throughout (Tasks 7→8). `_order_slugs_since(cx, email, window_days)` signature matches `seed_from_history`'s `order_slugs_fn` contract (Tasks 1, 4).
