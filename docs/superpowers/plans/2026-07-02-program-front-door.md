# Program → Deposit Front Door Implementation Plan (sub-project 3)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]` checkboxes. Branch: `program-front-door` (off main). NOTE: 5 owner-decision notes at the bottom must be confirmed before/during build.

**Goal:** Wire the $1-deposit wedge to the recurring engine — a deposit buyer buys their first program (scalable $100 / premium $300), gets the analysis + a 30-day Continuous Care taster (member pricing during it), with the $1 credited toward the program; near the window's end they're invited to continue on Continuous Care. Ships DARK behind `PROGRAM_CARE_TASTER_ENABLED`.

**Architecture:**
- `PROGRAM_TIERS` dict (module scope, app.py): `premium`=$300 (`BIOFIELD_PRICE_CENTS`, unchanged), `scalable`=$100 (10000¢). `_price_biofield` + `biofield_checkout` gain a `tier` param defaulting to premium (existing behavior byte-for-byte unchanged); front door passes `tier=scalable`.
- 30-day care window: on program fulfillment (the `kind=="biofield"` branch of `begin_checkout_return`), `_grant_membership(cx, email, 30, "care_taster")` — NEW source (reads as PAID, not the discount-withheld `biofield_trial` "trial" state), idempotent via new `care_taster_grants(order_ref PK)` table.
- $1 credit on the points rail: `points.credit(value_cents=100, reason="deposit_credit", order_ref=pi_id)` at `_fulfill_biofield_trial` (idempotent on order_ref+reason); auto-redeem at `biofield_checkout` via the existing `points_to_redeem_cents` arg when flag on + no explicit redeem.
- Continuous Care invite: new `source=="care_taster"` branch in `cron_membership_renewals` (mirrors the `prepay_%` branch), links `/prepay?renew=3mo`, no auto-charge.
- Front-door CTA: flag-gated CTA on the reveal page (`static/begin-biofield.html`) POSTing `{tier:"scalable"}` to `/biofield/checkout`.

**Tech stack:** Flask monolith app.py, SQLite LOG_DB, dashboard/*.py, pytest. Run: `cd /tmp/wt-deploy-chat-0cdf9c99 && doppler run -p remedy-match -c dev -- python3 -m pytest tests/<file> -q` (app import needs Pinecone/dev creds; python3; cwd resets → cd each time).

## Global Constraints
- premium=`BIOFIELD_PRICE_CENTS=30000` (unchanged); scalable=`10000`; care window=`30` days, source=`"care_taster"`; deposit credit=`100`¢ ($1 off — see owner decision #2); deposit preview grant stays `"biofield_trial"`/90d; flag `PROGRAM_CARE_TASTER_ENABLED` one-liner env pattern, default OFF; renewal deep-link `/prepay?renew=3mo` (3mo is a real prepay TIER key); idempotency: care grant via `care_taster_grants`, deposit credit via `points.has_entry(order_ref,reason)`; points floor 43% — $1 off $100/$300 safe.

## Tasks (bite-sized; full step code in the agent transcript — reproduce there)
1. **PROGRAM_TIERS + parametrize biofield_checkout by tier** (default premium unchanged). Tests: `test_biofield_checkout_scalable_tier_charges_100`, `test_biofield_checkout_defaults_premium_unchanged`. Impl: add `PROGRAM_TIERS`/`PROGRAM_PREMIUM_TIER`/`PROGRAM_SCALABLE_TIER` after `_BIOFIELD_ITEM_NAME`; `_price_biofield(points_to_redeem_cents=0, tier=PROGRAM_PREMIUM_TIER)` resolves price/name from the dict; `biofield_checkout` reads `data.get("tier")`, passes it, adds `"tier"` to metadata + uses tier name in Stripe description.
2. **Flag + 30-day care_taster grant on program fulfillment.** New `tests/test_program_front_door.py`: grants 30-day grant, idempotent, flag-off no-op, deposit+care_taster reads paid. Impl: flag block + `PROGRAM_CARE_TASTER_DAYS=30` + `CARE_TASTER_SOURCE="care_taster"`; in `begin_checkout_return` `kind=="biofield"` branch, claim-then-create on `care_taster_grants` then `_grant_membership(...,30,"care_taster")`.
3. **$1 points credit at deposit + auto-redeem at program checkout.** `PROGRAM_DEPOSIT_CREDIT_CENTS=100`; in `_fulfill_biofield_trial` won-claim lock, `points.credit(...,value_cents=100,reason="deposit_credit",order_ref=pi_id)`; in `biofield_checkout`, when `redeem==0` and flag on, auto-redeem `min(100, balance)`. Tests: deposit grants $1 credit (idempotent on replay), scalable auto-redeems → amount 9900.
4. **Continuous Care invite branch in `cron_membership_renewals`** for `source=="care_taster"` (no auto-charge, links `/prepay?renew=3mo`, gated by flag). Tests in `test_membership.py` mirror the prepay renewal test.
5. **Front-door CTA on reveal page.** Reveal payload gains `program_enabled`/`program_tier`; `static/begin-biofield.html` flag-gated CTA POSTs `{tier:"scalable"}` to `/biofield/checkout`. Test: reveal HTML contains `program_enabled` + `/biofield/checkout` + `scalable`.

## Owner decisions needed (CONFIRM before/during build)
1. **Front-door CTA surface** — plan puts it on the reveal page (`static/begin-biofield.html`); alt = readiness gate / portal / dedicated route.
2. **Deposit credit magnitude** — plan credits `value_cents=100` = exactly **$1 off**. If "100 points" was meant literally (=$5 at 5¢/pt), set `PROGRAM_DEPOSIT_CREDIT_CENTS=500`. Confirm the dollar credit.
3. **Scalable = tier param on `/biofield/checkout`** (chosen) vs a distinct public route/slug for the $100 program.
4. **Scalable program display name** — plan uses `"Biofield Program"` (premium keeps `"Causal Biofield Analysis"`). Confirm/supply.
5. **Care-taster idempotency key** — plan keys on QBO `invoice_id` (one taster per purchase) vs email (one ever per person).

Full per-step test+impl code is in the plan-agent transcript (2026-07-02). Reproduce verbatim when executing.
