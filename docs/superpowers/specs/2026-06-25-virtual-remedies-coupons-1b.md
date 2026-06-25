# Virtual Remedies & Coupon Wallet — Design Spec (1b)

**Date:** 2026-06-25
**Status:** Design (pre-build). Increment **1b** of the navigation/journey arc; sibling of the
shipped **1a** shell ([[2026-06-24-navigation-shell-journey-map-design]]).
**Reward model:** product **savings coupons** (NOT loyalty points). Coupon = a normal promotion →
cleaner compliance, works without points/identity-currency, self-limiting liability, tighter funnel.

> **Phasing (agreed):**
> - **1b-A (this spec):** virtual remedies (see) → **self-coupon** (earn, email+TOS-gated) → wallet →
>   avatar → coupon registry → checkout wiring. Self-contained; ships the core loop + lead capture.
> - **1b-B (separate spec, later):** the **gift coupon** — activating **Ambassador** unlocks a
>   shareable coupon per collected remedy, redemptions tracked + attributed. A bridge onto the
>   **existing** affiliate/referral system (`dashboard/referrals.py`, `affiliate_*` tables,
>   `_is_ambassador`, `_resolve_checkout_coupon_pct`).

## Goal

Turn the journey map into a **guided, collectible tour of the real catalog**: each land reveals a
real Functional Formulation as a "virtual remedy" (pre-frame), and completing the stage earns a
**15%-off, 10-day coupon** for that product (reward), claimed via the funnel's existing email+TOS
gate (lead capture). Delight + education + conversion, with compliance-clean promo mechanics.

## Current state (reuse-first — most rails exist)

- **Pricing already supports a coupon:** `dashboard/pricing.py:compute(..., coupon_pct=…)` applies a
  single % and **clamps to the 57% wholesale floor** (margin can't be eroded). Orders carry
  `discount_cents`.
- **Checkout already resolves a code → pct:** `app.py:_price_cart(..., coupon_pct=…)` (3479) and
  `begin_checkout` (5076) call **`_resolve_checkout_coupon_pct(code, email)`** (10091). 1b extends
  that resolver to also recognize self-coupon codes. (1b-B's gift coupons reuse the referral path
  unchanged.)
- **Email+TOS gate exists:** `begin_funnel` tracks `name`/`email`/`tos` gates + `tos_agreed_at`;
  `GET /begin/state` exposes them. Claiming a coupon requires this gate — no new account system.
- **Catalog:** `data/products.json` — 328 products keyed by slug (`name`, `price_cents`,
  `bottle_type`). Featured products reference real slugs.
- **Affiliate/referral system already exists** (for 1b-B): `referrals.py` codes + redemptions +
  attribution; affiliate apply/login; `affiliate_signups/offers/conversions/earnings`.

## Design

### The loop: see → earn → use

1. **See (open to all, cosmetic).** In the map overlay each land shows its **featured virtual
   remedy** (real formulation name + a *compliant* "healing power" line). Pre-frame only; no coupon.
2. **Earn (reward for engagement).** When a land's stage completes (its gate goes `done` per the
   journey engine), the self-coupon becomes **claimable**. **Claiming requires the email+TOS gate.**
   On claim: mint a **15% / 10-day** coupon bound to the email, drop it in the **wallet**, and
   **brighten the avatar**. Earned by doing + identity-gated → farm-proof and a lead-capture carrot.
3. **Use (convert).** The **wallet** panel lists earned coupons with a gentle countdown + a link to
   the product. Redeemed at checkout via the existing `coupon_pct` path (floored at 57%).

### New subsystem — coupon registry (`dashboard/coupons.py`)

The only net-new economy piece. Mirrors `points.py` / `referrals.py` patterns (pure-ish, one sqlite
connection, idempotent).

- **Table `coupons`:** `code TEXT PK, product_slug, pct INTEGER, kind TEXT (= 'self'),
  email, session_id, minted_at, expires_at, redeemed_at, order_ref`.
- **`mint_self(cx, *, email, product_slug, pct=15, days=10) -> dict`** — idempotent per
  `(email, product_slug, kind='self')` while an unexpired unredeemed one exists (returns the
  existing one); else generates a code (`SELF-<base32>`), `expires_at = now + days`.
- **`validate(cx, code, *, product_slug=None) -> dict|None`** — returns the coupon if it exists,
  matches the product (when given), is unexpired and unredeemed; else None.
- **`mark_redeemed(cx, code, *, order_ref)`** — idempotent stamp.
- **`wallet(cx, *, email) -> list`** — active (unexpired, unredeemed) coupons for an email.

### Endpoints (new)

- **`POST /api/journey/claim-coupon`** `{land}` — resolve session via `amg_session`; load
  `begin_funnel` state; **require `email` + `tos_agreed_at`** (else `409 {needs: "email_tos"}` so the
  shell can route to the funnel gate). Map `land → featured product_slug` from `shell-map.json`,
  require the land's stage `done` (re-check engine state server-side), then `coupons.mint_self`.
  Returns the coupon.
- **`GET /api/journey/wallet`** — `coupons.wallet(email)` for the session's email (empty if not yet
  identified).
- **Checkout:** extend **`_resolve_checkout_coupon_pct`** (app.py:10091) — if the code matches a
  `coupons.validate`, return its `pct` (+ a context tag) so the existing `_price_cart`/`compute`
  path applies it (floored). On order completion, `coupons.mark_redeemed(code, order_ref)`.

### Presentational config (`static/shell-map.json`, extended)

Each land gains `featured`: `{product_slug, product_name, healing_power}` (compliant copy). 1b-A
covers **scan / find / heal** (3 self-coupons); **give** is the gift land → **1b-B**.

| Land | Featured remedy (draft — Glen finalizes) | "Healing power" (compliant draft) |
|---|---|---|
| scan | Terrain Restore | supports your foundational terrain |
| find | Magnesium Taurate | supports calm, clear focus |
| heal | Endocrine Restore *(or Gut Terrain Program)* | supports your body's healing terrain |
| give | — (gift coupon → 1b-B) | — |

### Client (`static/shell.js` / `shell.css`)

- **See:** overlay land cards show the featured remedy + healing-power line (+ "claim" affordance
  when the stage is `done`).
- **Claim:** "Claim 15% off" → if `409 needs email_tos`, route to the funnel gate, then retry; on
  success, toast + wallet update + avatar brighten.
- **Wallet:** a panel (off the shell, like My Path) listing coupons with countdown + product link.
- **Avatar:** a small **biofield orb** (CSS/SVG) that gains coherence/glow per claimed coupon —
  **not a body.** N claimed = N rings lit.

### Avatar & copy guardrails (compliance)

- Avatar improves as a **brightening, more coherent biofield**, never a changing body.
- "Healing power" lines + any reward copy are **metaphor + the catalog's compliant language**, run
  through the **authority/compliance guardian** before ship. The coupon is "complete a step → earn a
  product discount" (an onboarding promo), not cash-for-claims.

## Out of scope (1b-A)

- The **gift coupon**, Ambassador activation, referral attribution → **1b-B**.
- Per-pavilion (8) collectibles — 1b-A is per-land (scan/find/heal).
- Real loyalty **points** (the `points.py` economy) — unchanged, untouched.
- Cross-device wallet persistence beyond email binding (identity depth is SP2).

## Data / rollout

- New flag **`REWARDS_1B_ENABLED`** (Doppler), **dark by default** — independent of
  `JOURNEY_SHELL_ENABLED` so 1a stays live while 1b pilots. Client reads it via `window.__SHELL__`.
- Additive: one new table (`coupons`); `begin_funnel`, `points.py`, pricing floors untouched;
  `_resolve_checkout_coupon_pct` extended (referral path unchanged).

## Testing

Python (sqlite tmp + the monkeypatch-`LOG_DB` client fixture; run via
[[reference_deploy_chat_local_tests]]):
- `coupons.py`: mint idempotency (same email+slug → same code while active); expiry; `validate`
  (product mismatch / expired / redeemed → None); `mark_redeemed` idempotent; `wallet` filters
  active only.
- `claim-coupon`: **409 when email/TOS missing**; mints when present + stage `done`; rejects a land
  whose stage isn't `done`.
- checkout: a valid self-coupon yields its `pct`, **clamped at the 57% floor**; redeemed coupon is
  marked on order completion; referral-code path still resolves unchanged (regression).
- `shell-map.json`: every `featured.product_slug` exists in `data/products.json`.

Client: manual/visual QA via the playwright screenshot workflow (light+dark, overlay claim flow,
wallet, avatar) per [[reference_deploy_chat_local_tests]].

## Inputs from Glen (during planning, not blocking)

1. Finalize the **3 featured formulations** (scan/find/heal) + their healing-power lines.
2. Confirm **avatar** treatment (biofield orb) is the right visual.
