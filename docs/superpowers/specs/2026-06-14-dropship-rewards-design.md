# Practitioner Drop-ship Rewards — Design

**Date:** 2026-06-14
**Status:** Design approved (Glen, two AskUserQuestion rounds). Spec for review → plan → build.
**Depends on:** PR #110 (console settings editor: `dashboard/pricing_settings.py`, `_pricing_settings()`/`_rewards_settings()` live-reload accessors, `/console/pricing-settings` editor). Stack this branch on #110; retarget to main once #110 merges.

## Problem

"What incentives reward a practitioner for sourcing a sale, and should patients earn points on drop-ship?" Confirmed baseline in code:
- A normal RM **retail** sale earns the buyer loyalty points (`_settle_order_points`) AND fires a referral reward to the referring practitioner (`_settle_referral`, 5% flat).
- A **drop-ship** order (client page / practitioner-paid) earns the practitioner their **margin only** (`wallet.earn_dropship_margin`) — no points to anyone.

## Decisions (locked)

1. **Drop-ship stays margin-only.** The practitioner already captures the markup as margin, and that margin already scales with certification via the wholesale base (lower base at higher cert → bigger margin). A points bonus on drop-ship would pay twice for the same behavior.
2. **The cert-tiered "higher reward" rides the referral channel** (patient buys at RM retail via the practitioner's link), where the practitioner otherwise captures nothing but the reward %.
3. **Patients earn channel-locked, RM-absorbed loyalty points on the client page.**

## Piece 1 — Cert-tiered referral reward

The referral reward % scales with the referring practitioner's certification, using the same anchor-curve pattern as the volume curve: `[modules_completed, pct]` knots, linear-interpolated, flat beyond the last, **console-editable** (a new setting in `pricing-settings.json` under `rewards`).

- **Curve (approved):** `referral_cert_anchors = [[0, 5], [6, 10], [12, 15]]` → 0 mods 5%, 6 mods 10%, 12 mods 15% (3 mods 7.5%, 9 mods 12.5%).
- **Who it applies to:** only referrers who are **practitioners** (resolve referrer slug → affiliate email → practitioner record with `modules_completed`). A regular affiliate or a pro-influencer with no practitioner record stays at the **base `referral_reward_pct` (5%)**.
- **Mode is orthogonal:** the curve sets the *percentage*; whether it pays as points (client/doctor referrer) or cash (pro-influencer) is unchanged. (Pro-influencers typically have no practitioner cert record, so in practice they stay at base 5% cash — acceptable; cert scaling is a practitioner-channel lever.)
- **Where it plugs in:** `_settle_referral` currently does `referral_reward_pct = float(settings["referral_reward_pct"])`. Replace the flat read with a resolver: if the referrer maps to a practitioner, interpolate their `modules_completed` through `referral_cert_anchors`; else use base `referral_reward_pct`. The interpolation reuses the existing `pricing.volume_pct`-style linear interp (extract a shared `interp_anchors(x, anchors)` helper or reuse `pricing.volume_pct`'s math).
- **Settings + validation:** add `referral_cert_anchors` to the rewards section; `pricing_settings.validate` checks ascending `[modules>=0, pct 0-100]` pairs (same shape rule as `volume_anchors` but allowing modules 0). Surface it in the editor page as another anchor table ("Referral reward by certification (modules → %)").
- **Defaults unchanged behavior:** with the default curve starting at 5% for 0 modules, an uncertified practitioner and a regular affiliate both earn 5% — identical to today. Nothing changes until a practitioner certifies.

### Edge cases
- Referrer has a practitioner record but `modules_completed` is null/0 → 5% (curve[0]).
- Referrer slug resolves to an email with no practitioner record → base 5%.
- `referral_cert_anchors` absent from settings → fall back to flat `referral_reward_pct` (no cert scaling) so the feature is safe-by-default before first configuration.

## Piece 2 — Patient loyalty points on the client page (channel-locked, RM-absorbed)

The patient earns RM loyalty points on a client-page (patient-paid drop-ship) order, redeemable only within that practitioner's channel.

- **Earn rate:** reuse the existing `points_earn_pct` (5%, already console-editable). No new rate knob. Earn on product spend only (net of shipping + GET), mirroring `_settle_order_points`.
- **Channel-locked:** points are scoped to the practitioner's dispensary code. The points ledger gets a **scope key** (e.g. a `scope` column = `dispensary:<code>`, default `"rm"` for normal retail points). A patient's channel points are redeemable **only** when checking out on that same dispensary code's client page; they never apply to RM-direct retail/funnel checkout.
- **RM-absorbed:** the redemption discount comes out of RM's margin (the practitioner is still paid their full margin on the order). The existing points floor (`points_floor_pct`) still protects the unit price.
- **Identity:** the consent email the client page already collects (`is_member` ToS gate) is the points key — no new gate.
- **Earn wiring:** in the `kind == "client"` branch of `/begin/checkout-return` (where `earn_dropship_margin` is credited today), also call a scoped points-earn for the patient (`points.credit(..., scope="dispensary:<code>")`).
- **Redeem wiring:** the client-page checkout (`build_client_order` / `/api/client/<code>/checkout`) accepts `points_to_redeem_cents`, looks up the patient's balance **for that dispensary scope only**, and applies it through the same engine points path (floor-protected). The client page shows the patient's available channel-points balance.

### Edge cases
- Patient with retail (`scope="rm"`) points opens a client page → those retail points are NOT offered there (different scope); only `dispensary:<code>` points apply.
- Patient shops two different practitioners → two separate scoped balances (no cross-redemption). Acceptable and intended.
- Refund/reversal of a client order → out of scope v1 (matches the existing referral/points refund-reversal gap already noted as deferred).

## Architecture / reuse

- `dashboard/rewards.py` — add `referral_cert_anchors` to DEFAULTS (or keep absent + fall back); add the practitioner-cert resolver helper (referrer slug → modules → pct). Reuse `referrer_email_for_slug`.
- `dashboard/points.py` — add an optional `scope` to credit/balance/redeem (default `"rm"`; existing rows treated as `"rm"`). A light schema migration (add column with default) or a parallel scoped table.
- `dashboard/practitioner_portal.py` — `modules_completed` / practitioner lookup by email (exists).
- `app.py` — `_settle_referral` (cert resolver), the `kind=="client"` return branch (scoped patient earn), `build_client_order` path + `/api/client/<code>/checkout` (scoped redeem + balance), client-page catalog/balance surfacing.
- `dashboard/pricing_settings.py` + editor — validate + expose `referral_cert_anchors`.

## Flags
Additive and flag-gated like the rest of the portal. Cert-tiered referral gated by `REWARDS_TIERS_ENABLED` (the existing rewards flag — `_settle_referral` already checks `_rewards_enabled()`). Patient channel-points gated behind the portal / a small flag so it can ship dark.

## Out of scope (v1)
- Refund-driven reversal of referral credits or patient channel-points (existing deferred gap).
- Practitioner-set patient-points earn rate (reuse global 5%).
- Cross-channel points portability.
- Pro-influencer cert scaling (they have no module record).

## Testing
- Cert resolver: 0/3/6/9/12 modules → 5/7.5/10/12.5/15%; non-practitioner slug → 5%; anchors absent → flat 5%.
- `_settle_referral` credits the cert-scaled amount (points and cash modes), still idempotent per order_ref, still full-price-only.
- Scoped points: credit/balance/redeem isolate by scope; retail points unaffected; client checkout redeems only `dispensary:<code>` points, floor-protected; RM-direct checkout never sees channel points.
- `pricing_settings.validate` accepts/rejects `referral_cert_anchors`; editor round-trips it.
- Regression: existing referral/points/retail tests stay green.
