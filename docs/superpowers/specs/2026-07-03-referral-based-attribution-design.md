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

## Part 1 — Attribution / reporting (read-only, safe, BUILT)

The **Patient portal** column (name kept for continuity with the client-portal pages) counts a practitioner's patients' **own** portal orders — **first purchase and reorders**.

`patient_portal_items(practitioner_email, *, practitioner_id=None, db_path=None) -> {slug: units}`:
1. patient set = **UNION**, deduped by email, of:
   - referred patients — `SELECT DISTINCT lower(referee_email) FROM referral_redemptions WHERE lower(owner_email)=practitioner_email` (the new referral model, first-touch, single-owner), and
   - dispensary clients — `SELECT DISTINCT lower(customer_email) FROM dispensary_orders WHERE practitioner_id=?` (so an existing dispensary-based practitioner is not stranded before Part 2 unifies the links).
2. sum `items_json` across `orders WHERE lower(email) IN (set) AND source IN _PORTAL_SOURCES AND status != 'cancelled'`, where `_PORTAL_SOURCES = ('portal-reorder','reorder','funnel')` — reorders plus the **funnel first purchase** (so first orders count, per Glen). Not wholesale (dispensed) or dispensary (drop-ship). Reuses `_add_items`.

`dispense_stats` gains `practitioner_email` (threaded from `portal_data`'s `row["email"]`) and `practitioner_id`, passed through. No money is touched — display/reporting only. Adversarial review caught (and this fixes) two earlier semantic bugs: first orders were excluded and dispensary-based practitioners read empty.

## Part 2 — Capture unification (NEEDS DR. GLEN'S DECISION — not built yet)

To capture *every* patient a doctor sends to their portal (not only those who used an affiliate `?ref=` link), make the doctor's **online-ordering-portal / dispensary link double as their affiliate referral link**, so a referred patient's first order writes a durable `referral_redemptions(referee, owner=doctor)` via the existing `record_redemption`. Recommended: one "your patient link" that drives both the drop-ship credit and the durable referral attribution.

**Reward policy — DECIDED by Dr. Glen 2026-07-03:** a drop-ship / portal sale is paid at **wholesale by the practitioner**, so their compensation is the **markup** — like any wholesale/retail sale. Therefore **no L1 affiliate commission** is paid on it (that would double-pay on top of the markup). **Only the L2 override is tracked, and only as points** (the practitioner's upline referrer accrues L2 points). The sale **is tracked for the practitioner's sales reporting** (the dispense table, Part 1) but generates **no L1 points**.

So sales-tracking and the points ledger are separate systems: writing the referral row on the portal link is for **attribution + L2 points only** and can never trigger an L1 commission — there is no double-pay. This is the "(a) attribution-only" posture for L1, plus L2 points. Part 2 wiring: make the doctor's portal/dispensary link write the durable `referral_redemptions` row on first order (via `record_redemption`), ensure the reward path pays **L2 points only, never L1**, on drop-ship/portal sales.

## Testing (Part 1)

- `patient_portal_items` attributes a referred patient's portal orders (via `referral_redemptions.owner_email`), case-insensitive; excludes a non-referred email, a non-portal source, and a cancelled order; a referee listed once (PK) isn't double-counted.
- `dispense_stats` threads `practitioner_email` and fills the third channel from the referral graph.
- The old dispensary-history attribution for the column is removed; dispensed/dropship channels are unchanged.

## Out of scope

Part 2 capture wiring + the reward-policy change; unifying *all three* channels under referral attribution (dispensed/dropship keep their current, correct attribution for now).
