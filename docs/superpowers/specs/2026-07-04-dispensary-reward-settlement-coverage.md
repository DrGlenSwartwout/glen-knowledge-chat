# Follow-up: Reward-Settlement Coverage for Dispensary Sales

**Date:** 2026-07-04
**Status:** decisions APPROVED by Dr. Glen 2026-07-04 — build now. Plan to follow.

**Decisions (Dr. Glen 2026-07-04):** (1) L2 accrues on **every** dispensary order, across **all** pay methods (card, Zelle, Wise) — close Gap A. (2) L2 accrues on **reorders**, not once — close Gap B. (3) L1 stays off (markup is the practitioner's pay). (4) Record `pay_method` on dispensary ingest so the split is exact going forward. **Why build now:** prod has ZERO dispensary sales as of 2026-07-04 (verified via `/api/console/dispensary-pay-mix` and the backfill dry-run, both 0), so there is no history to migrate and no live sales at risk — the safest possible moment to change the shared settlement path. No backfill needed.
**Parent:** `2026-07-03-referral-based-attribution-design.md` (Part 2, MERGED). Part 2 built durable attribution + L1 suppression + L2-on-first-card-sale + backfill. This follow-up covers two settlement-layer gaps the Part 2 whole-branch review surfaced. Both are pre-existing infrastructure limits, not regressions Part 2 introduced.

## The two gaps

### Gap A — alt-pay dispensary sales credit no L2

Reward settlement runs only on the Stripe card-return path (`app.py` ~7685-7714, `if inv and cid:` → `_settle_order_points` → `_settle_referrer_reward`). Zelle/Wise orders return pay-instructions only and never reach `_settle_order_points`; the Stripe webhook (`app.py` ~17964) routes only to biofield/prepay/care fulfillers; manual mark-paid does not call it. So a dispensary sale paid by **Zelle or Wise** gets correct attribution and correct Part-1 reporting, but the practitioner's upline **never receives L2 points**.

This affects *all* alt-pay orders (the Ambassador L1 flow has the same limit), so any fix touches the shared reward path — it must not double-credit a card order that later also settles, and must not alter existing Ambassador behavior unexpectedly.

**Candidate fix:** call `_settle_order_points(order, order_ref=invoice)` from the manual mark-paid action (and/or a Zelle/Wise confirmation step), guarded by the same idempotency the card path relies on (`_settle_referrer_reward` is already idempotent via `rewarded_at`; `_settle_order_points`'s earn/redeem branches guard on `points.has_entry`). Verify no path can settle the same order twice.

**Decision needed:** is a material fraction of dispensary sales alt-pay? If yes, build the fix. If most dispensary sales are card, accept the gap and document it.

### Gap B — L2 fires once (first sale), not per reorder

The reward is keyed to the single first-touch `referral_redemptions` row, whose `order_ref` is frozen to the **first** dispensary invoice. `_settle_referrer_reward` does `redemption_by_order_ref(this_order_ref)`, so only the first order matches; a reorder has a new invoice id → no row → no L2. This matches the existing one-time Ambassador reward model. Attribution and Part-1 reporting DO cover every reorder; only the L2 **points reward** is once-only.

**Decision needed:** should L2 points accrue on **every** dispensary reorder, or once at first attribution? "First purchase and reorders attribute" (Part 1) is already satisfied; this is specifically about the points ledger. Per-reorder L2 is a real change to the settlement model — it would need to look up the owner via `owner_of_referee(patient)` on every paid dispensary order rather than via the frozen redemption row, and re-confirm the anti-cycle / first-touch guarantees. Same shared-path caution as Gap A.

## Non-goals

- Changing L1 policy (Part 2's decision stands: no L1 on drop-ship/portal, ever).
- Retroactive payout on historical sales.
- Touching System B (affiliate cash).

## Open questions for Dr. Glen

1. Share of dispensary sales that are alt-pay (Zelle/Wise) vs card — drives whether Gap A is worth building.
2. L2 points: once at first attribution, or on every reorder (Gap B)?
3. If both are built, they share one settlement-path change and should be one plan with one review, because they also touch the Ambassador flow.
