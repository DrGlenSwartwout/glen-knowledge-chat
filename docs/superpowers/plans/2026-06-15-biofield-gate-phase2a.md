# Biofield Checkout + Readiness Gate — Phase 2a Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Sell the $300 Biofield as a points-redeemable checkout item, then gate the consult booking on a hybrid readiness checklist (photo / intake / fresh voice scan ≤7 days); when all are satisfied, unlock a booking link and drop a 48-hour prep task. Ships dark behind `BIOFIELD_CHECKOUT_ENABLED`.

**Architecture:** A new `biofield_readiness` sqlite table holds per-email state (paid, photo-on-file, intake/scan self-confirm, booked). A pure `dashboard/biofield_gate.py` computes the three item states (intake auto-checked from `inbound_leads`; photo/scan/paid from the row). A `/biofield/checkout` route prices a single $300 item through the existing engine (points via `points_to_redeem_cents`), Stripe + a QBO line, no shipping; on paid it seeds readiness and redirects to the gate. The gate (`/biofield/ready` + `/api/biofield/*`) is magic-link/member auth (reuse the `auth_tokens` pattern), supports photo upload (PHI-safe storage) + self-confirm + PB-receipt, and reveals a configurable booking link when green — creating a `todos` prep task on booking.

**Tech Stack:** Python 3.11, Flask, sqlite, Stripe vault + QBO, pytest.

**Spec:** `docs/superpowers/specs/2026-06-15-biofield-checkout-readiness-gate-design.md`. Scan check is self-confirm in 2a (the scan DB is local to Glen's Mac; see [[project_e4l_scan_ingestion]]). Photo = a face photo, uploaded in the gate.

**Reuse (read these first):** magic-link `auth_tokens` + `_hash_token` (app.py ~180/218) + `send_magic_link_email` (~226) + the `/reorder/request`+`/reorder/auth/<token>`+cookie pattern; the intake query `SELECT first_name, raw_json FROM inbound_leads WHERE email=? AND source IN ('scoreapp','practice-better','concierge')` (app.py ~4603); the engine points path `_price_cart(..., points_to_redeem_cents=...)` + `_settle_order_points`; `_stripe_checkout_url_for_retail` + the `begin_checkout_return` handler; `qb.create_invoice`; `todos` insert (`INSERT INTO todos (created_at,owner,category,title,body,priority,source,dedup_key)`, app.py ~2580); the chat attachment upload infra for the photo (but PERSIST the photo, unlike chat).

**Test invocation:** pure modules → `~/.venvs/deploy-chat311/bin/python -m pytest <path> -q`. App/route tests → `doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" ~/.venvs/deploy-chat311/bin/python -m pytest <path> -q` (from inside the worktree; ignore the 2 known pre-existing failures).

---

### Task 1: `biofield_readiness` store

**Files:** Create `dashboard/biofield_store.py`; Test `tests/test_biofield_store.py`

- [ ] **Step 1: Failing test** — table init idempotent; `seed_paid(cx, email, via, order_ref)` upserts a row with `paid_at` set + `paid_via`; `set_photo_on_file`, `set_intake_confirmed`, `set_scan_confirmed`, `set_booked` flip flags; `get(cx, email)` returns the row dict (or None); calling `seed_paid` twice is idempotent (one row, paid_at unchanged).

```python
import sqlite3
from dashboard import biofield_store as bs

def _cx():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    bs.init_table(cx); return cx

def test_seed_paid_and_flags():
    cx = _cx()
    bs.seed_paid(cx, "p@x.com", via="stripe", order_ref="INV1")
    r = bs.get(cx, "p@x.com")
    assert r["paid_at"] and r["paid_via"] == "stripe" and r["order_ref"] == "INV1"
    assert not r["photo_on_file"]
    bs.set_photo_on_file(cx, "p@x.com", "data/biofield-photos/p_x_com.jpg")
    bs.set_intake_confirmed(cx, "p@x.com", True)
    bs.set_scan_confirmed(cx, "p@x.com", True)
    r = bs.get(cx, "p@x.com")
    assert r["photo_on_file"] and r["intake_confirmed"] and r["scan_confirmed"]
    assert r["photo_path"].endswith(".jpg")

def test_seed_paid_idempotent():
    cx = _cx()
    bs.seed_paid(cx, "p@x.com", via="stripe", order_ref="INV1")
    first = bs.get(cx, "p@x.com")["paid_at"]
    bs.seed_paid(cx, "p@x.com", via="pb", order_ref="INV2")  # no-op on paid_at
    assert bs.get(cx, "p@x.com")["paid_at"] == first

def test_set_booked():
    cx = _cx()
    bs.seed_paid(cx, "p@x.com", via="stripe", order_ref="INV1")
    bs.set_booked(cx, "p@x.com")
    assert bs.get(cx, "p@x.com")["booked_at"]
```

- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement** `dashboard/biofield_store.py` — pure (cx passed). Schema:
```sql
CREATE TABLE IF NOT EXISTS biofield_readiness (
  email TEXT PRIMARY KEY,
  paid_at TEXT, paid_via TEXT, order_ref TEXT,
  photo_on_file INTEGER NOT NULL DEFAULT 0, photo_path TEXT,
  intake_confirmed INTEGER NOT NULL DEFAULT 0,
  scan_confirmed INTEGER NOT NULL DEFAULT 0,
  booked_at TEXT, created_at TEXT, updated_at TEXT
);
```
`seed_paid` uses `INSERT ... ON CONFLICT(email) DO UPDATE` but leaves `paid_at` unchanged if already set (COALESCE). Flag setters `UPDATE ... WHERE email=?` (insert a bare row first if missing). `get` returns `dict(row)` or None. All `cx.commit()`.

- [ ] **Step 4: Run → pass.** **Step 5: Commit** — `feat(biofield): readiness store`

---

### Task 2: readiness state logic

**Files:** Create `dashboard/biofield_gate.py`; Test `tests/test_biofield_gate.py`

- [ ] **Step 1: Failing test** — `gate_state(cx, email, *, scan_window_days=7)` returns:
```
{"paid": bool, "items": {
   "photo": {"status": "green"|"needed"},
   "intake": {"status": ...},
   "scan": {"status": ...}},
 "booking_unlocked": bool}
```
Rules: `photo` green if `photo_on_file`; `intake` green if `intake_confirmed` OR an `inbound_leads` intake row exists (auto); `scan` green if `scan_confirmed` (self-confirm in 2a); `booking_unlocked = paid AND all three green AND not already booked-blocking`. Inject the intake auto-check as a callable so the test stays pure:

```python
import sqlite3
from dashboard import biofield_store as bs, biofield_gate as bg

def _cx():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    bs.init_table(cx); return cx

def test_gate_unlocks_only_when_all_green():
    cx = _cx()
    bs.seed_paid(cx, "p@x.com", via="stripe", order_ref="INV1")
    st = bg.gate_state(cx, "p@x.com", has_intake=lambda e: False)
    assert st["paid"] and not st["booking_unlocked"]
    assert st["items"]["photo"]["status"] == "needed"
    bs.set_photo_on_file(cx, "p@x.com", "x.jpg")
    bs.set_scan_confirmed(cx, "p@x.com", True)
    # intake auto-detected
    st = bg.gate_state(cx, "p@x.com", has_intake=lambda e: True)
    assert st["items"]["intake"]["status"] == "green"
    assert st["booking_unlocked"] is True

def test_not_paid_never_unlocks():
    cx = _cx()
    st = bg.gate_state(cx, "nobody@x.com", has_intake=lambda e: True)
    assert st["paid"] is False and st["booking_unlocked"] is False
```

- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement** `dashboard/biofield_gate.py`: `gate_state(cx, email, *, has_intake, scan_window_days=7)`. Read the `biofield_store.get` row; compute each item; `booking_unlocked = bool(paid and photo and intake and scan)`. `has_intake` is a callable (the route passes a real `inbound_leads` lookup; tests pass a lambda). Keep it pure.

- [ ] **Step 4: Run → pass.** **Step 5: Commit** — `feat(biofield): gate_state logic`

---

### Task 3: $300 Biofield checkout (points-redeemable) → seed readiness

**Files:** Modify `app.py`; Test `tests/test_biofield_checkout.py`

- [ ] **Step 1: Failing test** — `POST /biofield/checkout` (flag on) with `{email, name, points_to_redeem_cents?}` → builds a single $300 line priced through the engine (points reduce it, floor-protected), creates a Stripe session (stub `stripe_pay.create_checkout_session`) with metadata `kind="biofield"` + email, returns the stripe url. On the `begin_checkout_return` for `kind=="biofield"` (stub a paid session), `biofield_store.seed_paid(...via="stripe")` is recorded and points are settled. Flag off → 404/disabled.

- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement** — a `BIOFIELD_PRICE_CENTS = 30000` constant + a `_biofield_item()` (a service line: name "Causal Biofield Analysis", no shipping, `volume_eligible=False`, points-eligible). Route `POST /biofield/checkout`: gate on `BIOFIELD_CHECKOUT_ENABLED`; price via the engine for a one-item cart with `points_to_redeem_cents` (reuse `_price_cart` but skip shipping — pass a flag or a service path that returns shipping_cents=0); QBO invoice one line; Stripe session metadata `{kind:"biofield", email, points_redeemed_cents, invoice_id}` with `save_card` not required. In `begin_checkout_return`, add a `kind=="biofield"` branch: `biofield_store.seed_paid(cx, email, via="stripe", order_ref=inv)`, settle points (reuse the existing `_settle_order_points` path with the recorded order), then the success page redirects to `/biofield/ready`. Reuse the engine floor so points can't take $300 below the points floor.

- [ ] **Step 4: Run → pass.** **Step 5: Commit** — `feat(biofield): $300 checkout + points + seed readiness`

---

### Task 4: readiness gate — auth, API, page

**Files:** Modify `app.py`; Create `static/biofield-ready.html`; Test `tests/test_biofield_gate_routes.py`

- [ ] **Step 1: Failing test** — `GET /api/biofield/ready` (magic-link/member email) returns the `gate_state` JSON; `POST /api/biofield/photo` (multipart) stores the file + sets `photo_on_file`+`photo_path`; `POST /api/biofield/confirm {item: "scan"|"intake"|"payment", receipt?}` flips the flag (payment via PB-receipt upload or self-attest → `seed_paid(via="pb")`); unauth → 401; flag off → disabled. Stub file storage to a tmp dir.

- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement.**
  - **Auth:** reuse the `auth_tokens` magic-link with `purpose="biofield"` and a `rm_biofield_email` cookie (mirror `/reorder/request` + `/reorder/auth/<token>`), plus accept an existing member session. A helper `_biofield_email()` reads the cookie/session.
  - `GET /api/biofield/ready`: `gate_state(cx, email, has_intake=<inbound_leads lookup>)` where the lookup runs the documented intake query. Include `booking_url` (from `BIOFIELD_BOOKING_URL` env) only when `booking_unlocked`.
  - `POST /api/biofield/photo`: accept an uploaded image; **persist to a private path** `DATA_DIR/biofield-photos/<hashed-email>.<ext>` (NOT a public/static dir — PHI); `biofield_store.set_photo_on_file(cx, email, path)`. Validate content-type + size.
  - `POST /api/biofield/confirm`: `item="scan"` → `set_scan_confirmed`; `item="intake"` → `set_intake_confirmed`; `item="payment"` → store/great a receipt (reuse the chat OCR upload to read the receipt, best-effort) then `seed_paid(via="pb")`.
  - `/biofield/ready` page (`static/biofield-ready.html`): shows the three items with green/needed states, the photo upload, "complete intake" + "record a fresh voice scan (within 7 days)" links with self-confirm checkboxes, the PB-receipt upload for PB payers, and a **Book your session** button that appears only when `booking_unlocked` (links to `booking_url`). Match the funnel page styling; no em dashes / ALL CAPS.

- [ ] **Step 4: Run → pass.** **Step 5: Commit** — `feat(biofield): readiness gate routes + page`

---

### Task 5: booking unlock → 48h prep task

**Files:** Modify `app.py`; Test `tests/test_biofield_gate_routes.py` (append)

- [ ] **Step 1: Failing test** — `POST /api/biofield/book` when `booking_unlocked` records `set_booked` and inserts ONE `todos` row (category e.g. "biofield", title "Biofield prep due 48h — <email>", `dedup_key="biofield-prep-<email>-<order_ref>"`); idempotent (second call no dup); returns the `booking_url`. When not unlocked → 409 with the gate state.

- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement** `POST /api/biofield/book`: recompute `gate_state`; if not `booking_unlocked` → 409 + state; else `biofield_store.set_booked`, insert the `todos` row (dedup_key prevents duplicates) with a body noting the 48h window + what's on file, optionally `append_event`, return `{ok, booking_url}`.

- [ ] **Step 4: Run → pass.** **Step 5: Commit** — `feat(biofield): booking unlock + 48h prep task`

---

### Task 6: flag + doc + suite

**Files:** Create `docs/biofield-gate.md`

- [ ] **Step 1:** Confirm every new route + the funnel entry is gated by `BIOFIELD_CHECKOUT_ENABLED` (default off). Add `BIOFIELD_BOOKING_URL` (booking link) + the photo storage dir to config/env docs.
- [ ] **Step 2:** `docs/biofield-gate.md`: the flow (pay $300 in checkout w/ points, or PB + receipt → readiness gate → photo upload / intake auto-or-confirm / fresh scan ≤7d self-confirm → booking unlock → 48h task), the PHI photo storage note, the self-confirm-scan-now / auto-later (2b) plan, the flag.
- [ ] **Step 3:** Suite green:
`doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_store.py tests/test_biofield_gate.py tests/test_biofield_checkout.py tests/test_biofield_gate_routes.py -q` + the begin-checkout/points regression.
- [ ] **Step 4:** Commit — `docs(biofield): checkout + readiness gate`

---

## Self-review
- **Spec coverage:** $300 checkout + points (Task 3); hybrid gate — photo upload, intake auto, scan self-confirm ≤7d (Task 1,2,4); PB payment via receipt/self-attest (Task 4); booking unlock + 48h task (Task 5); `BIOFIELD_CHECKOUT_ENABLED` dark flag (all). Magic-link/member auth reuses `auth_tokens`.
- **Type consistency:** `biofield_store` (init/seed_paid/get/set_*); `gate_state(cx,email,*,has_intake,scan_window_days=7)->{paid,items,booking_unlocked}`; metadata `kind="biofield"`; routes `/biofield/checkout`, `/api/biofield/ready|photo|confirm|book`, page `/biofield/ready`.
- **PHI (call out):** photos persist to a private DATA_DIR path, access-controlled, never a public/static dir; only a reference + flag in the DB. Review before go-live.
- **Deferred (2b):** real E4L scan-freshness auto-verify; Practice Better intake/photo/payment API; auto-drafting the analysis via the matcher agent; refund/expiry handling.
- **Risk:** money + PHI. Mitigations — dark flag, points floor-protected, idempotent seed/book, photos private + referenced, self-confirm gate tightened in 2b. Auth via the proven magic-link pattern.

## Done
Biofield is a points-redeemable $300 checkout that routes into a readiness gate (photo/intake/scan), unlocking a booking link + a 48h prep task — shipped dark behind `BIOFIELD_CHECKOUT_ENABLED`.
