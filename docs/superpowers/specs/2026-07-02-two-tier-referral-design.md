# Sub-project 5 — Two-Tier Referral Points — Design

**Date:** 2026-07-02
**Part of:** the healing-first offer redesign (see `project_membership_prepay_ladder`).
**Decisions locked by Glen 2026-07-02:** (1) points rail only, add Tier 2, leave the cash
pro-affiliate rail untouched; (2) Tier-2 rate = half of Tier-1; (3) tracking = wiring + extend
the existing `/api/pif/summary` (no new page).

## Goal

Extend the existing non-cashable **points** referral rail with a second economic tier: when a
buyer you were referred-by-your-referral (an L2 relationship) makes a **paid** purchase, the
Tier-2 referrer earns non-cashable points — half the Tier-1 rate. Everything else (sales-tied,
non-cashable, idempotent) is already guaranteed by the existing rail.

## What already exists (reused unchanged)

- `dashboard/referrals.py`: `referral_redemptions(referee_email PK, code, owner_email, order_ref,
  rewarded_at, reward_cents)` — the sole giver→receiver edge (owner_email = referrer,
  referee_email = buyer). One referrer per buyer, ever.
- `_settle_referrer_reward(cx, order, order_ref)` (app.py:5071): on a paid order, looks up the
  redemption by order_ref, credits the direct referrer (L1) `product_cents * pct // 100` as
  non-cashable points (`points.credit reason="referral_reward"`), stamps `mark_rewarded`
  (rewarded_at) for idempotency, returns early on replay. Tier-1 rate `pct = _referrer_reward_pct()`
  = flat `REFERRER_REWARD_PCT` env (integer %).
- `points.credit(cx, email, value_cents, reason, order_ref)` — idempotent via `has_entry(order_ref,
  reason)`. All `points_ledger` credits are structurally non-cashable (`actions_rewards.py` hard-
  refuses to cash points; cash flows only via the separate `affiliate_earnings` pro-affiliate rail).
- `/api/pif/summary` (app.py:25804) — member-gated read: points balance + `chain_summary`
  (L1/L2 reach counts) + gift wallet + healer_level.

## What's new (the gap)

No L2 economic propagation exists — the L2 walk today is read-only/cosmetic. This build adds it.

## Components

### 1. Lineage helper — `dashboard/referrals.py`
```python
def owner_of_referee(cx, referee_email):
    """Who referred this person (their Tier-1 referrer), or '' if none. The L2 hop."""
    row = cx.execute("SELECT owner_email FROM referral_redemptions WHERE referee_email=?",
                     (_norm(referee_email),)).fetchone()
    return (row[0] or "") if row else ""
```

### 2. Tier-2 settlement — `app.py`, flag `REFERRAL_TIER2_ENABLED`
- New flag near the reward flags (app.py ~4615), same idiom, default off.
- In `_settle_referrer_reward`, AFTER the existing L1 credit + `mark_rewarded` (so on replay the
  top `rewarded_at` early-return skips BOTH tiers — forward-only, no retroactive backfill on a
  flag flip), add:
```python
    if REFERRAL_TIER2_ENABLED:
        l2_owner = _rf.owner_of_referee(cx, red["owner_email"])   # who referred L1
        # No self-dealing / cycles: L2 must exist and differ from the buyer AND from L1.
        if l2_owner and l2_owner != red["referee_email"] and l2_owner != red["owner_email"]:
            reward_l2 = product_cents * pct // 200                 # half the Tier-1 rate
            if reward_l2 > 0:
                _points.credit(cx, l2_owner, value_cents=reward_l2,
                               reason="referral_reward_l2",
                               order_ref=f"referral_l2:{red['referee_email']}")
```
- Idempotency for L2 is free via `points.has_entry(order_ref, reason)` (order_ref keyed on the
  buyer's email, reason `referral_reward_l2`). Non-cashable free (same ledger). Cash rail untouched.
- The function's return value (Tier-1 cents) is unchanged — L2 is a side credit.

### 3. Per-tier earnings in the summary
- `dashboard/points.py`: `earned_by_reason(cx, email, reason)` → `SUM(delta_cents)` for positive
  credits of that reason (0 if none).
- `/api/pif/summary`: add `tier1_earned_cents` (reason `referral_reward`) and `tier2_earned_cents`
  (reason `referral_reward_l2`) to the payload. (The chain L1/L2 reach counts already exist.)

## Edge cases / invariants
- **No L2 when L1 has no referrer** → `owner_of_referee` returns '' → skip.
- **No self-dealing** → L2 != buyer and L2 != L1 guards (blocks A→B→A cycles crediting the buyer).
- **Idempotent** → replay hits the top `rewarded_at` early-return (and `points.has_entry` as a
  second line of defense). Exactly one L2 credit per qualifying order.
- **Forward-only** → flipping the flag does not retroactively pay L2 on already-settled orders
  (rewarded_at already set) — intentional, avoids a backfill blast.
- **Flag-dark** → `REFERRAL_TIER2_ENABLED` off ⇒ Tier-1 behavior byte-identical; no L2 credits.
- **Non-cashable** → L2 points land in `points_ledger`; `actions_rewards.py` refuses to cash them.

## Testing
- `owner_of_referee`: returns the referrer for a referee; '' when none.
- Settlement: on a paid order with a full A→B→C chain (C buys), B gets Tier-1, A gets Tier-2 =
  half; A gets nothing if A has no referrer; no credit when L2 == buyer or == L1 (cycle);
  idempotent on replay (exactly one L2 credit); flag-off → no L2 + Tier-1 unchanged;
  `reason="referral_reward_l2"`, order_ref `referral_l2:<buyer>`.
- Summary: `/api/pif/summary` returns `tier1_earned_cents`/`tier2_earned_cents` matching the ledger.

## Not in scope
- The cash pro-affiliate rail (no multi-level cash — FTC/MLM boundary preserved).
- A dedicated referral dashboard page (deferred; reuse `/api/pif/summary`).
- L3+ propagation (two tiers only).
- A separately-tunable Tier-2 percent (hardcoded half; trivial to make an env later).
