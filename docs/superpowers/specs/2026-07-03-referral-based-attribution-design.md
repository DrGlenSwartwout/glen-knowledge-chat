# Patient-Portal Attribution via the Referral Graph — Design

**Date:** 2026-07-03
**Status:** Part 1 (read/reporting) BUILT + MERGED (PR #560). Part 2 (capture) design APPROVED by Dr. Glen 2026-07-04; ready to write the implementation plan.
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
2. sum `items_json` across `orders WHERE lower(email) IN (set) AND source IN _PORTAL_SOURCES AND status != 'cancelled'`, where `_PORTAL_SOURCES = ('portal-reorder','reorder')` — the patient's own portal-page orders, first order and reorders alike. The first portal order lands under `reorder` via the biofield/portal checkout path, so it is counted. `funnel` is **deliberately excluded**: it tags every `/begin` retail purchase by anyone, unbounded in time and across practitioners, so including it would cross-attribute unrelated sales. Not wholesale (dispensed) or dispensary (drop-ship). Reuses `_add_items`.

`dispense_stats` gains `practitioner_email` (threaded from `portal_data`'s `row["email"]`) and `practitioner_id`, passed through. No money is touched — display/reporting only. Adversarial review caught (and this fixes) semantic bugs: dispensary-based practitioners read empty, and a first pass over-broadened `_PORTAL_SOURCES` to `funnel` (every retail purchase). The kept sources capture the first portal order (via the `reorder`/biofield path) without the funnel's cross-attribution.

**Known tradeoff (honest edge):** a patient who makes their genuine *first* purchase through the general `/begin` funnel — not the portal checkout — before any `reorder`/`portal-reorder` order is not counted, because that path is source `funnel` and we cannot distinguish this doctor's referred patient from an unrelated retail buyer without a time/attribution bound. A future refinement could count a funnel order only when the buyer's email is in the practitioner's referred/dispensary set *and* the order carries the practitioner's `rm_ref` — but that is Part 2 (capture) territory, not a read-only report.

## Part 2 — Capture unification (design APPROVED by Dr. Glen 2026-07-04; not built yet)

### Correction to the original model (code-grounded)

The first draft assumed the `rm_ref` cookie "carries the referrer code site-wide" into the referral graph. It does not. The code has **three independent identifier systems** that never cross today:

| System | Identifier | Set by | Cookie | Writes | Pays |
|---|---|---|---|---|---|
| **A. Referrals** (`referrals.py`) | 8-char code | manual `referral_code` field at checkout | none | `referral_redemptions` | L1 points + optional L2 override (`_settle_referrer_reward`, app.py:5172) |
| **B. Affiliate** (`rewards.py`) | slug | `?ref=slug` link | `rm_ref` (90d) | `affiliate_earnings` | cash or points (`_settle_referral`, app.py:5092) |
| **C. Dispensary** (`practitioner_portal.py`) | dispensary code | `/dispensary/<code>` visit | `rm_dispensary` (90d) | `dispensary_orders` | wellness credit only |

"The doctor's online ordering portal" in Dr. Glen's framing is **System C** (the dispensary/drop-ship link, patient pays retail, doctor pays wholesale, keeps the markup). Only System A rows enter the reward graph and the L2 path. **Part 2's core is the missing bridge: System C → System A** — write a durable `referral_redemptions` row when a patient orders through a practitioner's dispensary/portal link, and gate the reward so these rows pay L2 points only, never L1.

### Reward policy (DECIDED by Dr. Glen 2026-07-03)

A drop-ship / portal sale is paid at **wholesale by the practitioner**, so their compensation is the **markup**, like any wholesale/retail sale. Therefore **no L1** is paid on it (that would double-pay on top of the markup). **Only the L2 override is tracked, and only as points** — the person who referred the *practitioner* into the system accrues L2 points. The sale is tracked for the practitioner's sales reporting (Part 1 dispense table) but generates no L1 points.

### The four decisions (all confirmed by Dr. Glen 2026-07-04)

1. **Capture point = first dispensary/portal order.** On a patient's first order through a practitioner-linked checkout (the `rm_dispensary` cookie is present, or the order is source `dispensary`/`dropship`/`portal-reorder` resolvable to a practitioner), write `referral_redemptions(referee=patient_email, owner=practitioner_email, kind='dispensary_portal')`. Reuses the existing on-order write; no publish-time console UX needed.
2. **First-touch wins.** `referral_redemptions.referee_email` is the PK and `record_redemption` is `INSERT OR IGNORE`, so a patient already referred by an Ambassador keeps that owner; the dispensary link does not overwrite it. Single-owner, simplest, matches the table.
3. **Backfill from `dispensary_orders`.** A one-time pass writes `kind='dispensary_portal'` referral rows for existing dispensary clients (owner resolved practitioner_id → email), so current clients become L2-eligible and durably attributed. Idempotent via the PK.
4. **L2 boundary confirmed.** On a drop-ship/portal sale, L2 points go to the practitioner's upline referrer; the practitioner themselves gets no L1 (their pay is the markup).

### Wiring (grounded in the code map)

- **Schema:** add `kind TEXT DEFAULT 'referral'` to `referral_redemptions` (lazy additive ALTER, matching the existing `reward_cents`/`rewarded_at` pattern, referrals.py:21). Existing rows read as `'referral'` (System A, keeps L1).
- **`record_redemption`** gains a `kind='referral'` param, written into the new column. First-touch/idempotency unchanged (`INSERT OR IGNORE` on the PK).
- **practitioner_id ↔ email bridge:** the dispensary link resolves `code → practitioner_id` (`practitioner_id_by_dispensary_code`); Part 2 maps `practitioner_id → email` from the practitioners table to fill `owner_email`.
- **Capture hook:** at the dispensary/dropship/portal checkout, after the order is ingested, resolve the owning practitioner and call `record_redemption(..., kind='dispensary_portal')`. The `reorder`/`portal-reorder` paths already run near a referral hook (app.py:16081); the dispensary/dropship path (app.py:11468/11504) needs the call added.
- **L1 suppression:** in `_settle_referrer_reward` (app.py:5172), skip the L1 `_points.credit` (app.py:5190) when the redemption's `kind='dispensary_portal'`, but still run the L2 branch (app.py:5195-5202) and still `mark_rewarded` (so the row is guarded and never pays L1 on a later settlement). `kind` is read from the redemption row fetched by `redemption_by_order_ref`.
- **Backfill script:** one-time, reads `dispensary_orders(practitioner_id, customer_email)`, resolves practitioner_id → email, `INSERT OR IGNORE` a `kind='dispensary_portal'` row per (customer_email) with `order_ref` = the dispensary invoice. Does not backfill rewards (no retroactive L2 payout); attribution + go-forward L2 only.

### Non-goals / guardrails

- Does not change System B (affiliate cash) or the existing System-A Ambassador L1 flow.
- Does not retroactively pay L2 on historical sales; backfill establishes attribution only.
- `reward_cents` stays a paid-stamp, not a trigger; the payout guard remains `rewarded_at IS NULL` + row existence.

## Testing

### Part 1 (built)
- `patient_portal_items` attributes a referred patient's portal orders (via `referral_redemptions.owner_email`), case-insensitive; excludes a non-referred email, a non-portal source, and a cancelled order; a referee listed once (PK) isn't double-counted.
- `dispense_stats` threads `practitioner_email` and fills the third channel from the referral graph.
- The old dispensary-history attribution for the column is removed; dispensed/dropship channels are unchanged.

### Part 2 (to build)
- **Schema:** the `kind` ALTER is idempotent and additive; a pre-existing row reads `kind='referral'`.
- **`record_redemption`:** writes the passed `kind`; still first-touch (a second call for the same referee is ignored, original owner and kind preserved).
- **Capture hook:** a first dispensary/dropship/portal order by a practitioner-linked patient writes one `kind='dispensary_portal'` row with `owner_email` = that practitioner; a second order by the same patient writes nothing new; a patient already owned by an Ambassador keeps the Ambassador owner (first-touch).
- **L1 suppression:** `_settle_referrer_reward` on a `kind='dispensary_portal'` redemption credits **no** L1 points, credits L2 points to the upline when tier-2 is enabled, and stamps `rewarded_at` (no L1 on replay). A `kind='referral'` redemption still pays L1 (Ambassador flow unchanged).
- **Backfill:** running it writes one row per existing dispensary client, is idempotent on re-run (PK), and pays no reward (attribution only).

## Out of scope

Changing System B (affiliate cash) or the existing Ambassador L1 flow; retroactive L2 payout on historical sales; unifying the dispensed/dropship display channels under referral attribution (they keep their current, correct source-scoped attribution).
