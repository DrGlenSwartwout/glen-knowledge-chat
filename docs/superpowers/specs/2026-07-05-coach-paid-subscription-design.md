# Coaching — paid coaching subscription (coaching arc, slice 2b) — Design

**Date:** 2026-07-05
**Status:** Approved in brainstorm with Glen 2026-07-05 (month-to-month; card-on-file + monthly cron; first month at signup; EVOX credit / Biofield entitlement per cycle; cancel keeps the current cycle).
**Repo:** deploy-chat

**Relates to / reuses:**
- Slice 2: the upsell offer + `coaching_interest` (the "I'm interested" flag becomes a real Subscribe).
- `dashboard/stripe_pay.py`: `create_setup_session(*, customer_email, metadata, success_url, cancel_url)` (Stripe-hosted card capture), `get_setup_intent(si_id)`, `_find_or_create_customer(email)`, `charge_off_session(customer_id, payment_method_id, amount_cents, *, description, metadata)`, the `/webhook/stripe` `_fulfill_*` dispatch, `_STRIPE_ACTIVE`.
- Entitlement grants: `dashboard/evox.py:add_session_credits(cx, email, n)` (Rae → 1 EVOX credit/cycle), `dashboard/consult.py:set_consult_ready(cx, email, True)` (Glen → Causal Biofield entitlement/cycle).
- `_evox_ident` (member auth), `send_evox_email` + `GLEN_CONSULT_EMAIL` (notify), `CONSOLE_SECRET`/`CRON_SECRET` (cron auth), `PUBLIC_BASE_URL`, `_db_lock`, `LOG_DB`.
- [[project_membership_prepay_ladder]], [[reference_captured_unpaid_reconcile]] (off-session-charge precedent), [[reference_render_prod_ops_trigger]] (cron pattern), [[feedback_flag_flip_backfill_window]] (avoid mass-charge on first cron run).

## Context and boundary

The coaching arc's paid upsell (slice 2 offered it; this slice charges it). A member subscribes to month-to-month coaching with **Rae ($100/mo, includes one EVOX session per cycle)** or **Dr. Glen ($200/mo, includes one Causal Biofield Analysis per cycle)**. Recurring billing is **card on file + a monthly cron** (there is no Stripe Subscription object in this integration; the wrapper is raw one-time/off-session charges).

**Money-path invariants (load-bearing):** the member's first month is charged ONCE at signup (idempotent against webhook retries); the cron never double-charges (advances `next_charge_at` only on a successful charge, and only charges rows whose `next_charge_at <= now`); `next_charge_at` is seeded to signup + 1 month so the first cron run does not mass-charge new subscribers. Card data is captured only on Stripe's hosted page — the app never receives raw card numbers.

## Scope

**Subscribe (card on file) → first month charged + this cycle's service granted → monthly cron recharges + regrants → cancel anytime.** One subscription store + a subscribe/activation flow (Stripe setup session + webhook fulfillment) + a monthly charge cron + a cancel route + the member surface. The paid coaching *conversation* still uses slice 3's thread (or the EVOX call / Biofield consult); this slice is the billing + monthly entitlement.

**Deferred:** slice 3 (the 1:1 coaching thread), annual/prepay tiers, proration, dunning beyond a simple retry+suspend, carryover of unused monthly services.

## Components

### 1. Subscription store (`dashboard/coach_subscriptions.py`)

- `coach_subscriptions(member_email TEXT PRIMARY KEY, tier TEXT, amount_cents INTEGER, stripe_customer_id TEXT, payment_method_id TEXT, status TEXT, started_at TEXT, next_charge_at TEXT, last_charged_at TEXT, fail_count INTEGER DEFAULT 0)` — one subscription per member. `tier` ∈ {`rae`,`glen`}; `status` ∈ {`active`,`past_due`,`canceled`}.
- `coach_sub_charges(id INTEGER PK, member_email TEXT, tier TEXT, amount_cents INTEGER, pi_id TEXT, status TEXT, charged_at TEXT)` — a ledger row per charge attempt (for idempotency + audit).
- Functions (pure sqlite, no app imports): `init_sub_tables(cx)`; `TIERS = {"rae": {"amount_cents":10000,"service":"evox","label":"Rae"}, "glen": {"amount_cents":20000,"service":"biofield","label":"Dr. Glen"}}`; `get_sub(cx, email)`; `upsert_sub(cx, *, email, tier, customer_id, payment_method_id, next_charge_at)`; `set_status(cx, email, status)`; `mark_charged(cx, email, pi_id, next_charge_at)` (advance + reset fail_count + last_charged_at); `mark_failed(cx, email)` (fail_count += 1, status past_due); `record_charge(cx, email, tier, amount_cents, pi_id, status)`; `due(cx, now_iso) -> [dict]` (active subs with `next_charge_at <= now`).

### 2. Subscribe + activation (`app.py`)

- `POST /api/community/coach-subscribe {tier}` (member portal-token; `tier` ∈ rae/glen else 400; 503 if not `_STRIPE_ACTIVE`): `create_setup_session(customer_email=member, metadata={kind:"coach_sub", tier, email}, success_url, cancel_url)`; return `{checkout_url}`. The member enters their card on Stripe's page.
- **Activation** — add `_fulfill_coach_sub(session_id)` to the `/webhook/stripe` dispatch (kind-guarded on `metadata.kind == "coach_sub"`): resolve the completed setup session → `setup_intent` → `payment_method` + `customer`; **idempotency guard** (if a sub already exists for this member with a charge this cycle, no-op); `upsert_sub`; charge the first month via `charge_off_session(customer, pm, TIERS[tier].amount_cents, ...)`; on success `record_charge` + `_grant_cycle_service(cx, email, tier)` + `mark_charged(next_charge_at = now+1 month)` + confirmation email; on failure `mark_failed` + notify. Best-effort, never breaks the webhook.
- `_grant_cycle_service(cx, email, tier)`: `rae` → `evox.add_session_credits(cx, email, 1)`; `glen` → `consult.set_consult_ready(cx, email, True)`. (One included service per successful cycle; use-it-or-lose-it — no stacking guard needed since EVOX credits are a counter and the Biofield flag is idempotent.)

### 3. Monthly charge cron (`app.py`)

- `POST /api/cron/coach-subscriptions/charge` (CONSOLE_SECRET/CRON_SECRET-gated, like the EVOX reminders cron): for each `due(cx, now)` sub — `charge_off_session`; on success `record_charge` + `_grant_cycle_service` + `mark_charged(next_charge_at = now+1 month)`; on failure `mark_failed` + notify the member + Glen, and if `fail_count >= 2` `set_status("past_due")` stays and the coaching entitlement is not regranted (member notified to update their card). Returns `{charged, failed}`. Idempotent: only charges `next_charge_at <= now` and advances on success, so a re-run in the same window does not double-charge.
- A Render cron_job hits this monthly (e.g. `0 17 1 * *`), mirroring the `glen-evox-reminders` cron wiring.

### 4. Cancel (`app.py`)

- `POST /api/community/coach-subscribe/cancel` (member portal-token): `set_status("canceled")`; the cron skips canceled subs; the member keeps the service already granted for the current cycle (no refund, no proration). Returns `{ok}`.

### 5. Member surface (`static/client-portal.html`)

- The slice-2 upsell "I'm interested" buttons become **Subscribe** buttons → `POST /api/community/coach-subscribe {tier}` → redirect to `checkout_url`. On return, a "Your coaching subscription" card shows: tier + price, next charge date, this cycle's included service and whether it's been used (EVOX credit balance / Biofield entitlement), and a Cancel button. Copy: no em dashes, no ALL CAPS; the price and "includes one EVOX session / Causal Biofield Analysis per month" stated plainly.

## Config

- Reuses the Stripe keys, `PUBLIC_BASE_URL`, `GLEN_CONSULT_EMAIL`, `CONSOLE_SECRET`/`CRON_SECRET`. Optional `COACH_SUB_RAE_CENTS` (default 10000) / `COACH_SUB_GLEN_CENTS` (default 20000). A new Render cron_job for the monthly charge.
- Go-live prerequisite: `_STRIPE_ACTIVE` (already set on prod).

## Money-path safety

- **First charge once:** activation is webhook-driven and idempotent (guard on an existing sub + a same-cycle charge ledger row) so a webhook retry cannot double-charge month 1.
- **No mass-charge on first cron:** signup seeds `next_charge_at = now + 1 month`; the cron only charges `next_charge_at <= now`, so newly-subscribed members are not charged again by the first cron run ([[feedback_flag_flip_backfill_window]]).
- **No cron double-charge:** `mark_charged` advances `next_charge_at` by a month only after a successful charge, inside the same locked write, so an overlapping/re-run cron skips already-charged rows.
- **Failed charge:** `mark_failed` increments `fail_count` and does NOT advance `next_charge_at` or grant the service; the member is notified; after 2 failures the sub sits `past_due` (skipped until they resubscribe/update the card).
- Card capture is Stripe-hosted; the app stores only the Stripe customer id + payment_method id, never a card number.

## Testing

- Pure/sqlite (`dashboard/coach_subscriptions.py`): `upsert_sub`/`get_sub` round-trip; `TIERS` amounts; `due` returns only active subs with `next_charge_at <= now` (not future, not canceled/past_due); `mark_charged` advances `next_charge_at` + resets fail_count; `mark_failed` increments + past_due; `record_charge` ledger.
- Route/api (Stripe + grants mocked): subscribe returns a checkout_url (503 when `_STRIPE_ACTIVE` false; 400 bad tier); `_fulfill_coach_sub` on a mocked completed setup session creates the sub, charges once, grants the cycle service (mock `add_session_credits`/`set_consult_ready`), and a second webhook delivery for the same session does NOT double-charge; the charge cron charges a due sub once, grants, advances `next_charge_at`, and a due sub whose charge fails goes `past_due` without a grant; cancel sets canceled and the cron then skips it. Assert the cron does NOT charge a sub whose `next_charge_at` is in the future.
- Regression: EVOX/consult/Community/coaching slices 1-2 untouched (this reads/writes its own tables + reuses `add_session_credits`/`set_consult_ready`/`charge_off_session`).
- Go-live: subscribe as a test member (Stripe test card), confirm month-1 charge + an EVOX credit (Rae) / consult-ready (Glen); run the cron with a back-dated `next_charge_at` and confirm a second charge + regrant + advanced date; cancel and confirm the cron skips.

## Deferred (coaching arc, later)

- Slice 3: the 1:1 coaching thread (the paid coaching conversation) + report/block/moderation.
- Annual/prepay tiers, proration, richer dunning (multiple retries, grace period), unused-service carryover, and a coach-side view of paid subscribers.
