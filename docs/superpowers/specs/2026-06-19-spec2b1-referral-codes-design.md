# Spec 2b-1 — Referral codes + referee redemption (single-sided)

**Date:** 2026-06-19
**Status:** Approved (design); ready for implementation plan
**Repo:** deploy-chat (Flask, Render `glen-knowledge-chat`)
**Parent:** Spec 2 (reviews + referral). Final piece. The referrer reward is the fast-follow **2b-2**.

---

## Problem

We want shareable, tracked referral codes: every customer has one stable code (highlighted as the headline reward when their review is approved); a friend (referee) enters it at checkout for ~10% off, one time ever, with self-referral blocked. The discount flows through the existing pricing engine (best-of-one with the volume curve, respecting per-SKU floors). The referrer reward is deferred to 2b-2 — this is the **referee side only**.

## Scope (2b-1)

A per-person referral code store + retrieval; a redemption validator with anti-abuse guards (self-referral block, one-redemption-per-referee-ever); checkout integration so a referee's valid code resolves to a `coupon_pct` and records a single redemption on order creation. Ships dark behind `REFERRALS`.

**Out of scope:** the referrer reward / double-sided payout (→ 2b-2); code expiry (codes are evergreen for now); restricting referees to brand-new customers (any non-self, never-redeemed email qualifies).

---

## Confirmed decisions (Glen, 2026-06-19)

- **Issuance:** one stable code **per person** (email), lazily created, **evergreen** (no expiry in 2b-1). Available to every customer; **highlighted as the review reward** when a review is approved.
- **Discount:** **10%** off for the referee (configurable).
- **One redemption per referee, ever** — a given email gets a referral price exactly once (DB-enforced).
- **Self-referral blocked** (referee email ≠ the code's owner).
- **Single-sided** — no referrer reward in 2b-1.

---

## Architecture

### Store — `dashboard/referrals.py` (SQLite `chat_log.db`)
- `referral_codes(email TEXT PRIMARY KEY, code TEXT UNIQUE, created_at TEXT)`.
- `referral_redemptions(referee_email TEXT PRIMARY KEY, code TEXT, owner_email TEXT, order_ref TEXT, created_at TEXT)` — `referee_email` PK enforces **one redemption per referee, ever**.
- Functions:
  - `init_tables(cx)`
  - `get_or_create_code(cx, email) -> str` — returns the email's existing code or mints a new short uppercase code (e.g. `secrets.token_urlsafe`-derived, deduped on the UNIQUE column).
  - `owner_of(cx, code) -> str|None` — the email that owns a code.
  - `has_redeemed(cx, referee_email) -> bool`.
  - `resolve(cx, code, referee_email, *, pct) -> {"owner_email": str, "coupon_pct": int} | None` — returns None unless: `owner_of(code)` exists, owner ≠ referee (self-referral block, case-insensitive), and `not has_redeemed(referee)`. On success returns the owner + `pct`.
  - `record_redemption(cx, code, owner_email, referee_email, order_ref) -> bool` — `INSERT OR IGNORE` on `referee_email` PK; returns True if it inserted (False if the referee already had a redemption — race-safe).

### Discount percent
`_referral_pct()` reads a configurable value (env `REFERRAL_PCT`, default `10`) → integer percent. (Future: move into `pricing-settings.json`.)

### Retrieval surface
- `GET /api/referral/my-code` (flag-gated): resolves the caller's email (`get_authenticated_user` / the reorder cookie), `get_or_create_code`, returns `{"code": ...}` (404/empty when no email or flag off). Surfaced on the reorder/account page as "Your referral code: …, give a friend 10% off." (Front-end = manual visual pass.)
- **Review-reward highlight:** when a review is approved, the reviewer's referral code is shown as part of the reward (a small hook reading `get_or_create_code` for the reviewer's email — display only, no new money path).

### Checkout integration (the money path)
A referee may pass a `referral_code` at checkout (funnel `begin-buy` + reorder). A new helper resolves the effective coupon percent, preferring the customer's best option:
- `_resolve_checkout_coupon_pct(referral_code, referee_email) -> (pct, referral_ctx|None)`: if `REFERRALS` and a `referral_code` validates via `referrals.resolve(... pct=_referral_pct())`, the effective `coupon_pct = max(referral_pct, _active_coupon_pct() or 0)` and `referral_ctx = {code, owner_email}`. Otherwise `pct = _active_coupon_pct()`, `referral_ctx = None`.
- The resolved `coupon_pct` flows into `_price_cart(..., coupon_pct=...)` / `pricing.compute` exactly as today — best-of-one with the volume curve, never stacking with the subscriber tier, and bounded by the per-SKU floors. No change to pricing internals.
- **Redemption is recorded once, at order creation**: when an order is created from a checkout that used a valid `referral_ctx`, call `referrals.record_redemption(cx, code, owner_email, referee_email, order_ref)`. The `referee_email` PK guarantees a referee consumes their one-time discount on the first such order; a second attempt is a no-op insert (and, by then, `has_redeemed` already blocks the discount at resolve time).

### Flag
`_REFERRALS = os.environ.get("REFERRALS", ...)`. Gates the code surface, the resolve/redemption, and the checkout `referral_code` handling. Fully inert when off (no code minted, no referral discount, no recording).

---

## Data flow
1. A customer fetches `GET /api/referral/my-code` (or sees it on review approval) → their stable code.
2. They share it; a friend enters it at checkout.
3. Checkout resolves the code → `coupon_pct = max(10, daily)` (if valid: exists, not self, referee hasn't redeemed) → pricing applies it (best-of-one, floors).
4. On order creation, the redemption is recorded (referee PK) — the referee's one-time discount is consumed; future checkouts by that referee no longer validate.

## Error handling
- Invalid / self / already-redeemed code → `resolve` returns None → checkout falls back to the normal `_active_coupon_pct()` (no error shown beyond "code not applied"); the order proceeds.
- `record_redemption` is `INSERT OR IGNORE` (race-safe); a concurrent double-submit can't double-grant because the discount itself is gated by `has_redeemed` at resolve and the PK at record.
- All referral reads/writes wrapped so a referral failure never blocks checkout/order creation (degrade to the normal coupon).
- Floors in `pricing.compute` cap the maximum discount regardless of the referral percent.
- Flag off → no code, no referral discount, no recording.

## Testing
- **Store:** `get_or_create_code` stable per email + unique codes; `owner_of`; `has_redeemed`; `resolve` returns pct for a valid referee, None for self-referral, None for an already-redeemed referee, None for an unknown code; `record_redemption` one-per-referee (second insert no-ops, `has_redeemed` true after).
- **Discount/resolve helper:** `_resolve_checkout_coupon_pct` returns `max(referral, daily)` for a valid code, the daily pct (referral_ctx None) for invalid/self/redeemed, and the daily pct when the flag is off.
- **Checkout integration (Flask test client):** a referee checkout with a valid code prices at the referral percent (or better) and records exactly one redemption tied to the order; a second order by the same referee gets no referral discount; a self-referral checkout gets the normal price + no redemption; pricing floors still cap the discount.
- **Retrieval:** `GET /api/referral/my-code` returns a stable code for an email; 404/empty with no email or flag off.
- Follow deploy-chat test isolation (tmp `$DATA_DIR`; mock Supabase; importorskip playwright; `importlib.reload`). Front-end (code display, checkout field) = manual visual pass. NO emoji; no em dashes.

## Flags
`REFERRALS` (default off). With it off, 2b-1 is fully inert (the funnel/reorder behave exactly as today).

## Notes
- Reuses `pricing.compute(coupon_pct=...)` + `_active_coupon_pct` + `_price_cart` (no pricing-internals change), `get_authenticated_user` / the reorder cookie, and the dispatch/console patterns. No new external dependency.
- The referral discount is a `coupon_pct`, so it inherits the engine's guarantees: best-of-one with volume, no stacking with the subscriber tier, and the per-SKU floors as the hard cap.
- 2b-2 (fast follow) adds the **referrer reward** (double-sided): on a referee's first paid order with a code, credit the referrer (e.g. store-credit points), once per distinct referee, uncapped on super-referrers. The `referral_redemptions` table already records `owner_email` + `order_ref`, so 2b-2 can reward off it without schema change.
