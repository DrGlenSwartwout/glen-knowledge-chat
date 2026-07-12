# Family Plan — self-serve recurring billing — Design

**Date:** 2026-07-11
**Status:** Approved in brainstorm with Glen 2026-07-11 (billing-only; mirror the shipped coach subscription; caregiver-initiated from their portal; $147/mo card-on-file + monthly cron; entitlement `covers()` unchanged; the three ancillary benefits and the benefit-copy rework are deferred).
**Repo:** deploy-chat

**Relates to / reuses:**
- `dashboard/family_plan.py` (already shipped #747, live behind `FAMILY_PLAN_ENABLED`): `PLAN` (`amount_cents=14700`, `value_cents=19700`, `label="Family Plan"`), `activate(cx, caregiver_email, *, next_charge_at, customer_id, payment_method_id, source)`, `get`, `set_status`, `is_active`, `covers(cx, email)`, `ACTIVE_STATUSES=("active","past_due")`. This slice ADDS the billing helpers it lacks vs the coach module.
- `dashboard/household.py`: `caregivers_for`, `viewable_members_for`, `share_consent` (consent already gates `covers()` in both directions).
- `dashboard/stripe_pay.py`: `create_checkout_session(amount_cents, *, customer_email, description, metadata, success_url, cancel_url, save_card=True)`, `get_session(session_id)`, `get_payment_intent(pi_id)`, `charge_off_session(customer_id, payment_method_id, amount_cents, *, description, metadata)`, `_STRIPE_ACTIVE`.
- `dashboard/subscriptions.py:add_months(yyyy_mm_dd, n)` (date math); `dashboard/portal_identity.py:resolve_identity` (caregiver auth from the portal token); `_db_lock`, `LOG_DB`, `PUBLIC_BASE_URL`, `GLEN_CONSULT_EMAIL`, `CONSOLE_SECRET`/`CRON_SECRET`, `send_evox_email`.
- **The direct sibling (mirror it 1:1):** `dashboard/coach_subscriptions.py` + `POST /api/community/coach-subscribe` + `_fulfill_coach_sub` + `/coach-subscribe/return` + the `/webhook/stripe` dispatch + `POST /api/cron/coach-subscriptions/charge` (`app.py` ~19086-19256). This slice is the family analog with one flat tier and no per-cycle service grant.
- Memory: [[reference_family_plan_entitlement]], [[project_family_accounts_per_scan]], [[feedback_flag_flip_backfill_window]] (avoid mass-charge on first cron run), [[feedback_verify_review_findings_money_path]].

## Context and boundary

The Family Plan **entitlement** already shipped: one plan per caregiver, and a single predicate `covers(email)` (they hold a plan, OR a caregiver who holds one is linked to them with `share_consent=1`) wired into the live paywall `_portal_biofield_unlocked`. Operator enrollment + comp already exist (`POST /api/console/family-plan`, `source:"comp"`). What is missing is the **billing**: a caregiver cannot buy the plan themselves, and nothing charges a plan's `next_charge_at` (`run_subscriptions_cron.py`/`/api/cron/charge-subscriptions` is a separate membership subsystem; the coach cron is a separate endpoint). A "plan" today is therefore only ever a comp.

This slice adds self-serve recurring billing so the Family Plan becomes a real paid product. It does **not** change entitlement (`covers()` untouched), does **not** wire the three ancillary benefits (monthly family shipment, member product pricing, group coaching access — they still just hang off `covers()`), and does **not** rework the plan copy.

**Money-path invariants (load-bearing):**
- The caregiver's first month is charged ONCE at signup, idempotent against webhook + redirect double-delivery (claim on `family_sub_grants(session_id)` PRIMARY KEY).
- The cron never double-charges: `next_charge_at` advances only on a successful charge, inside the same locked write, and only rows with `next_charge_at <= today` are charged.
- `next_charge_at` is seeded to signup + 1 month, so the first cron run does not mass-charge new subscribers ([[feedback_flag_flip_backfill_window]]).
- A **comped** plan (`source='comp'`, `next_charge_at IS NULL`) is NEVER charged — `due()` excludes it explicitly.
- Card data is captured only on Stripe's hosted page; the app stores the Stripe customer id + payment_method id only, never a card number.

## Scope

**Subscribe (card on file) → first month charged + plan active (whole consented household un-blurs via `covers()`) → monthly cron recharges → cancel anytime.** Unlike the coach subscription there is **no per-cycle service grant** — a successful charge simply keeps `status='active'`, which is what `covers()` reads. One flat tier ($147/mo, $197 anchor) from `family_plan.PLAN`; one plan per caregiver. The enrollment surface is the caregiver's existing portal Family Plan block (today display-only, "Just reply to arrange it").

**Deferred:** the three ancillary benefits and their wiring; the benefit-copy rework (see go-live gate); proration; annual/prepay tiers; a self-serve card-update UI; richer dunning than the bounded retry below.

## Components

### 1. Store additions (`dashboard/family_plan.py`)

Existing: `family_subscriptions(caregiver_email PK, amount_cents, stripe_customer_id, payment_method_id, status, source, started_at, next_charge_at, last_charged_at, fail_count)` + `PLAN`/`activate`/`get`/`set_status`/`is_active`/`covers`. `status` ∈ {`active`,`past_due`,`cancelled`}.

Add (mirroring `coach_subscriptions`):
- `family_sub_charges(id INTEGER PK AUTOINCREMENT, caregiver_email TEXT, amount_cents INTEGER, pi_id TEXT, status TEXT, charged_at TEXT)` — one ledger row per charge attempt (audit + idempotency evidence). Created in `init_family_plan_table`.
- `due(cx, today) -> [dict]`: `SELECT * FROM family_subscriptions WHERE status='active' AND next_charge_at IS NOT NULL AND next_charge_at <= ? AND source != 'comp' ORDER BY next_charge_at`. The `IS NOT NULL` + `source != 'comp'` guards ensure a comped plan (no card, no charge date) is never billed. Only `active` is charged — `past_due`/`cancelled` are not auto-retried here (dunning handled in the cron).
- `mark_charged(cx, caregiver_email, next_charge_at)`: set `next_charge_at`, `last_charged_at=now`, `fail_count=0`, `status='active'`.
- `mark_failed(cx, caregiver_email)`: `fail_count += 1`, `status='past_due'`.
- `record_charge(cx, *, caregiver_email, amount_cents, pi_id, status)`: insert a `family_sub_charges` row.

`activate()` already upserts the card ids, `status='active'`, `fail_count=0` — it is the create/restart path for a paid signup (`source='stripe'`) and the comp path (`source='comp'`, no card, no `next_charge_at`).

### 2. Subscribe + activation (`app.py`)

- `POST /api/portal/<token>/family-plan/subscribe`: gated on `_family_plan_enabled()` (404 if off) and `_STRIPE_ACTIVE` (503 if off). Resolve the caregiver via `portal_identity.resolve_identity` (404 if no identity). `create_checkout_session(family_plan.PLAN["amount_cents"], customer_email=caregiver, description="Family Plan", metadata={"kind":"family_plan","email":caregiver}, success_url=f"{base}/family-plan/return?session_id={{CHECKOUT_SESSION_ID}}", cancel_url=<portal url>, save_card=True)`. Return `{ok, url}`. Month 1 is charged by this checkout; the card is vaulted for months 2..N.
- `_fulfill_family_plan(session_id)` (never raises; mirrors `_fulfill_coach_sub`): `get_session` → guard `metadata.kind == "family_plan"`; require `email` + `payment_intent`; `get_payment_intent` must be `succeeded` with a `customer` + `payment_method`. Idempotent claim: `INSERT OR IGNORE INTO family_sub_grants(session_id PK, email, created_at)` — if not newly claimed, return already-fulfilled (no re-activate, no re-charge). On claim: `activate(caregiver, next_charge_at=add_months(today,1), customer_id=customer, payment_method_id=pm, source="stripe")`; `record_charge(..., status="succeeded")`; best-effort confirmation email. Callable from BOTH the return redirect AND the webhook (closed-tab / dropped-redirect safety).
- `GET /family-plan/return`: `_fulfill_family_plan(session_id)` then redirect to the caregiver's portal.
- `/webhook/stripe` dispatch: add a `metadata.kind == "family_plan"` branch calling `_fulfill_family_plan`.

### 3. Monthly charge cron (`app.py`)

- `POST /api/cron/family-plan/charge` (auth: `X-Cron-Secret`/`X-Console-Key` matching `CRON_SECRET`/`CONSOLE_SECRET`, like the coach cron; `?dry_run=1` computes without charging). For each `due(cx, today)` row: `charge_off_session(customer, pm, PLAN["amount_cents"], description="Family Plan", metadata={"kind":"family_plan_cycle","email":caregiver})` in a per-sub `try/except` (a raised charge is treated as a failure for THAT sub only, never aborting the batch). On success: `record_charge(succeeded)` + `mark_charged(add_months(today,1))`. On failure: `record_charge(failed)` + `mark_failed` (→ `past_due`), notify the caregiver + Glen; **bounded dunning** — when `fail_count >= 3`, `set_status('cancelled')` so grace ends and `covers()` stops (a `past_due` plan still entitles the household mid-cycle per `ACTIVE_STATUSES`, so grace must be bounded or a failing card would cover forever). Returns `{charged, failed, cancelled, dry_run}`.
- A Render `cron_job` hits this monthly, mirroring the coach cron's wiring (post from the cron container to this web endpoint so the charge runs where `LOG_DB` lives).

### 4. Cancel (`app.py`)

- `POST /api/portal/<token>/family-plan/cancel`: resolve the caregiver via `resolve_identity`; `set_status('cancelled')`. `covers()` stops for the household at the next read; no refund, no proration (the current paid cycle simply is not renewed). Returns `{ok}`. (Operator cancel already exists at `/api/console/family-plan/cancel`.)

### 5. Caregiver surface (`static/client-portal.html`)

The Family Plan block (~line 992) is display-only today. Wire it:
- `fpl && !fpl.active` → a **Subscribe** button → `POST /api/portal/<token>/family-plan/subscribe` → redirect to the returned Stripe `url`.
- `fpl.active` → keep the "you're on it" line and add a **Cancel** affordance → `POST /api/portal/<token>/family-plan/cancel` → reload.
- Copy rules: no em dashes, no ALL CAPS; price/value from the payload, never hardcoded. **The current benefit copy is subject to the go-live gate below.**

## Config

Reuses Stripe keys, `PUBLIC_BASE_URL`, `GLEN_CONSULT_EMAIL`, `CONSOLE_SECRET`/`CRON_SECRET`, `_STRIPE_ACTIVE`, and `FAMILY_PLAN_ENABLED` (already on). Amounts stay in `family_plan.PLAN`. Adds one Render monthly `cron_job` for the charge endpoint. No new feature flag — the subscribe route being reachable is itself gated by `_family_plan_enabled()` + `_STRIPE_ACTIVE`.

## Money-path safety

- **First charge once:** activation claims `family_sub_grants(session_id)` before activating/recording; a webhook retry or a return-redirect racing the webhook finds the row claimed and no-ops.
- **No mass-charge on first cron:** signup seeds `next_charge_at = today + 1 month`; `due()` only returns `next_charge_at <= today`.
- **No cron double-charge:** `mark_charged` advances `next_charge_at` by a month only after a successful charge, in the same locked write, so an overlapping / re-run cron skips already-charged rows.
- **Comps never charged:** `due()` excludes `next_charge_at IS NULL` and `source='comp'`.
- **Failed charge / dunning:** `mark_failed` increments `fail_count`, sets `past_due` (still entitles mid-cycle — grace), does NOT advance `next_charge_at`; the caregiver + Glen are notified; at `fail_count >= 3` the plan is `cancelled` and cover stops.
- Card capture is Stripe-hosted; only the customer id + payment_method id are stored.

## Go-live gate (blocking — honesty on a money path)

The Family Plan benefit copy (portal block, lines ~994/998) promises four benefits — full household scan results, member product pricing, a monthly family shipment, and group coaching with Dr. Glen — but only the household analysis-unlock is wired ([[reference_family_plan_entitlement]]). The billing machinery may be **built and tested** now, but **flipping self-serve checkout live to charge real caregivers $147/mo is blocked until the copy matches delivered value** — either trim the copy to the Causal Biofield Analysis household unlock, or wire the other three benefits first. Building the mechanism does not require resolving this; charging real money does. (Test-mode Stripe validation is unaffected.)

Separately, **seed Karin now via a comp** (`POST /api/console/family-plan {caregiver_email: permanentlyyours@hawaii.rr.com, source:"comp"}`) so her live household is covered by one entitlement immediately, independent of this build.

## Testing

- **Pure sqlite (`dashboard/family_plan.py`):** `due()` returns only `active` rows with `next_charge_at <= today` and excludes future-dated, `past_due`, `cancelled`, `source='comp'`, and `next_charge_at IS NULL`; `mark_charged` advances `next_charge_at` + resets `fail_count`; `mark_failed` increments + sets `past_due`; `record_charge` writes a ledger row; `covers()` unchanged (regression: holder + consented-caregiver true, revoked-consent false).
- **Route/api (Stripe + email mocked):** subscribe returns a checkout url (404 when `_family_plan_enabled()` false, 503 when `_STRIPE_ACTIVE` false, 404 on a bad token); `_fulfill_family_plan` on a mocked succeeded session activates the plan + records the charge, and a SECOND delivery of the same `session_id` neither re-activates nor re-charges; the charge cron charges a due sub once and advances `next_charge_at`; a due sub whose charge fails goes `past_due` without advancing; a third consecutive failure sets `cancelled` and `covers()` then returns false; a comped sub is never returned by `due()` nor charged; cancel sets `cancelled` and the cron then skips it.
- **Regression:** entitlement `covers()`, the console comp/cancel endpoints, and the coach subscription flow are untouched (this slice reads/writes only `family_subscriptions` / `family_sub_charges` / `family_sub_grants` and reuses shared Stripe/date helpers).
- **Go-live (test mode):** subscribe as a test caregiver with a Stripe test card from the portal → confirm the month-1 charge and that a linked member's report un-blurs (`covers()` true); back-date `next_charge_at` and run the cron → confirm a second charge + advanced date; cancel → confirm the cron skips and the member re-blurs; comp a caregiver → confirm the cron never charges them.

## Deferred

- The three ancillary benefits (monthly family shipment, member product pricing, group coaching access) and their wiring off `covers()`.
- The benefit-copy rework (the go-live gate above resolves the honesty blocker; the full copy pass is separate).
- A self-serve card-update UI, proration, annual/prepay tiers, and richer dunning (multiple retries, grace windows, a paid-subscriber operator view).
