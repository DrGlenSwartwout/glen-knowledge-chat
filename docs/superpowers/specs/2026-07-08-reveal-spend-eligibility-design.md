# Reveal spend-eligibility — design

**Date:** 2026-07-08
**Author:** Glen + Claude
**Status:** approved (design), pending implementation plan

## Goal

A **free member** who places a **paid order of $100 or more since their last biofield reveal** earns their **next reveal for free** (fully un-blurred). This closes a loyalty loop: buying remedies keeps a free member's readings unlocked without a paid membership.

## Rules (confirmed with Glen 2026-07-08)

- **Who:** a *free* member — `_active_membership_for_email(email)` is falsey (not a paid/continuous-care member).
- **Trigger:** a **single paid order with `total_cents >= 10000`** (not cumulative across smaller orders).
- **Order sources:** any — in-house, GrooveKart, funnel — since they all funnel through `_ingest_order` into the BOS orders table. Must be **paid** (money actually captured), not merely created/abandoned.
- **"Since their last reveal":** the order is recorded **after** the email's previous `biofield_reveals` row.
- **Reward:** the **single next reveal** is unlocked free. **No banking, no stacking** — any qualifying spend in the period since the last reveal unlocks exactly the one next reveal.
- Grants the *entitlement* only; does **not** force-trigger a new E4L scan (scans stay client-initiated).

## Mechanism — a single derived check at reveal creation (no new table)

Because eligibility is per-period and non-stacking, there's nothing to persist: it is fully determined by "was there a qualifying order between the previous reveal and this one?" So we compute it **at the moment a reveal is created**, reusing the existing free-unlock path.

Today `dashboard/biofield_reveals.py` un-blurs a reveal for a free member when `biofield_free_unlocks(email, reveal_id, granted_at)` has a row for it (`free_unlock_reveal_id` / `record_free_unlock`, surfaced via `free_available` in `_reveal_view_state` ~`app.py:2904`). This feature calls `record_free_unlock` for the new reveal when the period qualifies.

### The check (hook at the reveal-creation chokepoint)

Where a new `biofield_reveals` row is created for an email (single insert chokepoint — to be located in the plan), after insert, run one best-effort helper `maybe_unlock_for_spend(cx, email, new_reveal_id)` that:

1. Returns early if `_active_membership_for_email(email)` (paid member — already fully unlocked).
2. Finds the **previous** reveal for the email (the `biofield_reveals` row immediately before `new_reveal_id`), and its timestamp `since_ts`. If there is no previous reveal, `since_ts` = epoch (any prior qualifying order counts toward the first reveal).
3. Queries the BOS orders table for **any paid order** with `total_cents >= 10000` for this email dated **after `since_ts`** (and ≤ now). `EXISTS` — one row is enough.
4. If found: `record_free_unlock(email, new_reveal_id)` (idempotent `INSERT OR IGNORE`).

No `_ingest_order` hook, no credit table, no lifecycle. Each reveal independently evaluates its own preceding window, which is exactly "one per period, no banking."

### Helper location

`dashboard/biofield_reveals.py` — a new `maybe_unlock_for_spend(cx, email, reveal_id)` next to `record_free_unlock`. The orders query is a thin read against the orders table (paid + total_cents + email + created_at); confirm exact column/status names in the plan.

## Edge cases

- **No prior reveal:** window is open-ended; any prior qualifying paid order unlocks the first reveal. (Buying $100 before any reveal earns the first one — acceptable.)
- **Paid member at reveal creation:** skipped (already unlocked). If they later downgrade, their *next* reveal re-evaluates normally.
- **Multiple qualifying orders in the period:** irrelevant — the check is `EXISTS`, so one or many both unlock exactly the single next reveal.
- **Order refunded before the reveal:** if the refund flips the order's paid status/total in the orders table, it naturally stops qualifying (the check reads live order state). No separate claw-back needed.
- **Reveal re-created / re-synthesized for the same period:** `record_free_unlock` is `INSERT OR IGNORE`, so re-running is safe.

## Testing

- Free member + paid $100 order after last reveal ⇒ next reveal is free-unlocked.
- Order < $100, or unpaid, or before the last reveal ⇒ not unlocked.
- Paid member ⇒ skipped (helper no-ops).
- Two qualifying orders in the period ⇒ still exactly one unlock on the single next reveal (no banking).
- No-prior-reveal free member + $100 paid order ⇒ first reveal free-unlocked.
- Helper re-run for the same reveal ⇒ idempotent (one free_unlock row).

## Out of scope (v1)

- Forcing/scheduling a new E4L scan on qualification.
- Cumulative-spend (sub-$100 orders adding up to $100).
- A persisted "credit" or banking of multiple free reveals.
