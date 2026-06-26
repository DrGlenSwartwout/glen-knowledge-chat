# Gift Coupons & Ambassador Gifting — Design Spec (1b-B)

**Date:** 2026-06-25
**Status:** Design (pre-build). Increment **1b-B**; builds on the merged **1b-A** self-coupon layer
([[2026-06-25-virtual-remedies-coupons-1b]]).
**Model (agreed):** per-product **gift coupons** (a giftable twin per collected remedy), unlocked by
**instant self-serve gifting activation** (two-tier — gifting now, commission stays gated), with
redemptions **attributed via the existing referral rails**.

## Goal

Turn collectors into referrers: each remedy a visitor collected (their 1b-A self-coupon) gains a
**giftable twin** — a `kind='gift'` coupon they can share so a friend gets **15% off that product**,
with the redemption attributed back to the gifter through the existing `referral_redemptions`
system. Activation is instant; *earning commission* on it stays behind the existing approved-affiliate
gate.

## Current state (reuse-first)

- **Coupon registry** (`dashboard/coupons.py`, 1b-A): `coupons` table already has a **`kind`** column
  (default `'self'`) and `email` (owner). Extend with `kind='gift'` minting + gift validation.
- **Affiliate / Ambassador** (`app.py`): `affiliate_signups` (email UNIQUE, `status DEFAULT 'approved'`,
  requires name/slug/token). **`_is_ambassador` = a row with `status='approved'`** (the commission gate).
  `referrals.py`: `record_redemption(cx, code, owner_email, referee_email, order_ref)` +
  `resolve(...)` (blocks self-referral + already-redeemed referee). Checkout already records referrals
  (`_record_referral_if_any`) and resolves codes → `coupon_pct`.
- **Shell/wallet** (1b-A): `static/shell.js` wallet panel + the Beacon land ("the gift of healing")
  are the natural homes for the gift UI + the activation CTA.

## Design

### Gifting activation (instant, two-tier)

- **`gifting_activated_at TEXT`** added to `affiliate_signups` (nullable; additive migration).
- **`POST /api/journey/activate-gifting`** — requires the funnel **email+TOS** gate (via `get_state`).
  Upserts the visitor's `affiliate_signups` row: if absent, insert with their email + generated
  `slug`/`token` + name (from funnel state) + **`status='pending'`** (explicit — the table default is
  `'approved'`, which we must NOT inherit) + `gifting_activated_at=now`; if present, set
  `gifting_activated_at=now` only. Idempotent. **`_is_ambassador` is unchanged** (`status='approved'`),
  so gifting does not grant commission-earning.
- **Gifting-enabled signal:** `gifting_activated_at IS NOT NULL` for the gifter's email — a *separate*
  check from `_is_ambassador`.

### Gift coupons (extend `dashboard/coupons.py`)

- **`mint_gift(cx, *, email, product_slug, pct=15, days=30) -> dict`** — `kind='gift'`, `email`=owner
  (gifter). Idempotent per `(email, product_slug, kind='gift')` for life (mirrors self earn-once: one
  gift per remedy per gifter).
- **`validate_gift(cx, code, *, referee_email) -> dict | None`** — returns the coupon if `kind='gift'`,
  unexpired, unredeemed, **and `referee_email != owner email`** (no self-gift); else None. Unlike
  self-coupons it is **redeemable by any email** (validate-by-code, not redeemer-bound).
- Reuse `mark_redeemed`. Gift coupons surface in the **wallet only when gifting is active**; minted
  on activation (and on each new collection thereafter while active).

### Wallet + sharing (client)

- **`GET /api/journey/wallet`** extended: alongside self-coupons, include the owner's **gift coupons**
  (`kind='gift'`) each with a **share URL** `"/begin/buy/<slug>?gift=<code>"`.
- **`static/shell.js`** — the Beacon land shows an **"Unlock gifting"** CTA → `POST activate-gifting`
  (routes to the email+TOS gate on `409`). The wallet renders, per remedy: the self-coupon + (when
  active) a **giftable coupon with a "Share" button** (copies the share URL).

### Checkout — friend redeems, gifter gets attribution

Extend the checkout coupon resolution (the same `begin_checkout` path 1b-A wired):
- Read a gift code from `data.get("gift")` (and the `?gift=` link → posted by `begin-buy.html`).
- `validate_gift(code, referee_email=checkout_email)` → its `pct` → applied via the existing
  `coupon_pct` path (**clamped to the 57% wholesale floor**; non-stacking `max()` with referral/self).
- On order completion: `mark_redeemed(code, order_ref)` **and**
  `referrals.record_redemption(code, owner_email=coupon.email, referee_email=checkout_email, order_ref)`
  — so the gift redemption feeds the **existing** attribution (`_has_referred_friend`, and the
  approved-affiliate commission path when the gifter is approved).
- Anti-abuse: `validate_gift` blocks self-gift; reuse the referral guard (referee hasn't already
  redeemed) so one new customer can't farm multiple gift discounts.

### Defaults (agreed)

- Gift coupon: **15% off**, **single-use** (one friend per remedy), valid **30 days** from mint/share.
- **One gift coupon per (gifter, product)**.

## Compliance

Gifting a discount is low-stakes and instant. **Earning** on a gift redemption stays behind the
existing **approved-affiliate** (`status='approved'`) + FTC-compliant commission structure — no
payout-earning status is granted unreviewed. The gift redemption is *recorded* for attribution
regardless; commission only flows for approved affiliates via the existing engine.

## Out of scope (1b-B)

- Commission **payout** mechanics (existing approved-affiliate track, unchanged).
- Multi-friend / multi-use gift codes; a gift/referral analytics dashboard.
- Auto-approving affiliates for commission (stays manual/console).

## Data / rollout

- New flag **`REWARDS_1B_GIFT_ENABLED`** (Doppler), **dark by default**, independent of
  `REWARDS_1B_ENABLED`. Client reads it via `window.__SHELL__`.
- Additive: one nullable column (`affiliate_signups.gifting_activated_at`); `coupons` unchanged
  (uses existing `kind`); reuses `referrals.py`. `_is_ambassador`, the referral path, `points.py`,
  and the `begin_funnel` engine untouched.

## Testing (run via [[reference_deploy_chat_local_tests]])

- `coupons.mint_gift`: idempotent per (owner, slug); `validate_gift` blocks self-gift
  (`referee==owner`), expired, redeemed, and wrong-kind; redeemable by a different email.
- `activate-gifting`: 409 without email/TOS; sets `gifting_activated_at` + inserts `status='pending'`
  (assert **`_is_ambassador` stays False** afterwards); idempotent.
- wallet: includes gift coupons + share URL only when gifting active.
- checkout: a gift code yields its pct **clamped at the floor**, marks redeemed, **records a
  `referral_redemption`** (owner=gifter, referee=buyer); self-gift (buyer==gifter) is rejected;
  referral-code path still resolves unchanged (regression).
- flag off → activate-gifting + gift wallet entries absent; client gift UI hidden.

## Inputs from Glen (not blocking)

- Confirm the Beacon land is the right home for the "Unlock gifting" CTA (vs the wallet header).
