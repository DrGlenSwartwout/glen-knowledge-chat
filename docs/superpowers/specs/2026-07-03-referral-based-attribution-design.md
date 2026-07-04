# Patient-Portal Attribution via the Referral Graph — Design

**Date:** 2026-07-03
**Status:** design; Part 1 (read/reporting) approved to build, Part 2 (capture) needs Dr. Glen's reward-policy decision.
**Supersedes:** `2026-07-03-patient-portal-attribution-design.md` (Approaches A/B). Dr. Glen's insight: treat the referral of a patient to the doctor's online ordering portal like an **affiliate referral** in the existing tracking system, so all that patient's orders (first purchase AND reorders) attribute to the referring doctor, with no requirement they ever bought through the doctor before.

## The key realization

The system already has a durable, first-touch practitioner↔client link: the **referral graph** (`dashboard/referrals.py`).
- `referral_redemptions(referee_email PK, owner_email, code, order_ref, …)` — one referrer per referee (PK ⇒ first-touch, single owner).
- `owner_of_referee(cx, referee_email)` → the referrer's email.
- `record_redemption(...)` already fires **on order** (app.py:12860 general, 7111 gift), and the `rm_ref` cookie carries the referrer code site-wide.

So attribution = "who referred this patient" (`referral_redemptions.owner_email = practitioner`), independent of purchase history, covering the first order and every reorder, resolved to a single owner. Strictly better than Approach A (dispensary-history-only) and cheaper than Approach B (no new table/UX — the referral system IS the durable link).

## Part 1 — Attribution / reporting (read-only, safe, BUILD NOW)

Rewrite the Patient-portal column to read the referral graph instead of dispensary history.

`patient_portal_items(practitioner_email, *, db_path=None) -> {slug: units}`:
1. `referees = SELECT DISTINCT lower(referee_email) FROM referral_redemptions WHERE lower(owner_email)=?`
2. sum `items_json` across `orders WHERE lower(email) IN (referees) AND source IN ('portal-reorder','reorder') AND status != 'cancelled'` (reuse `_add_items`).

`dispense_stats` gains a `practitioner_email` param (threaded from `portal_data`'s `row["email"]`) and passes it through. No money is touched — this is display/reporting only.

**Limit (until Part 2):** Part 1 attributes only patients already in the referral graph (those who came through the doctor's affiliate referral link). Correct and forward-working — the column grows as more patients are captured as referrals. It no longer requires a prior dispensary purchase.

## Part 2 — Capture unification (NEEDS DR. GLEN'S DECISION — not built yet)

To capture *every* patient a doctor sends to their portal (not only those who used an affiliate `?ref=` link), make the doctor's **online-ordering-portal / dispensary link double as their affiliate referral link**, so a referred patient's first order writes a durable `referral_redemptions(referee, owner=doctor)` via the existing `record_redemption`. Recommended: one "your patient link" that drives both the drop-ship credit and the durable referral attribution.

**Decision required — reward interaction:** today a dispensary drop-ship earns the doctor **$20/bottle Wellness Credit** (`dispensary_orders`), and an affiliate referral earns **affiliate commission** (`referral_redemptions` → reward). Unifying the links means one sale could write both. Dr. Glen must confirm the policy:
- (a) **Attribution only** — write the referral row for *attribution/reporting* but suppress the affiliate *reward* on dispensary-originated sales (no double-pay); or
- (b) **Both rewards** — the doctor earns dispensary credit AND affiliate commission (intended stacking); or
- (c) **Replace** — dispensary credit gives way to the affiliate commission model.

Until this is decided, Part 2 is not wired — writing referral rows could trigger commission payouts. Part 1 stands alone and is safe.

## Testing (Part 1)

- `patient_portal_items` attributes a referred patient's portal orders (via `referral_redemptions.owner_email`), case-insensitive; excludes a non-referred email, a non-portal source, and a cancelled order; a referee listed once (PK) isn't double-counted.
- `dispense_stats` threads `practitioner_email` and fills the third channel from the referral graph.
- The old dispensary-history attribution for the column is removed; dispensed/dropship channels are unchanged.

## Out of scope

Part 2 capture wiring + the reward-policy change; unifying *all three* channels under referral attribution (dispensed/dropship keep their current, correct attribution for now).
