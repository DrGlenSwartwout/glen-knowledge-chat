# Begin #4b — $1 Biofield Unlock -> $99/mo Trial

**Date:** 2026-06-20
**Status:** Approved (design); ready for implementation plan
**Repo:** deploy-chat (Flask, Render `glen-knowledge-chat`, illtowell.com)
**Parent:** Begin-page redesign #4. #4a (Biofield reveal, PR #194) blurred the deeper remedies behind a stub "Unlock your full Biofield Analysis" CTA. #4b is that unlock: a $1 tripwire that vaults the card, grants the full paid membership, releases the deeper remedies, and auto-converts to $99/mo.

---

## Problem

The Biofield reveal (#4a) shows the visitor their top remedy free and withholds the deeper remedies server-side (only a `blurred_count`). The "Unlock your full Biofield Analysis" CTA is a stub. We want it to be a **$1 tripwire** into the **$99/mo tier**: pay $1 now, become a full paid member immediately (the deeper Biofield remedies release, and everything gated on `_active_membership_for_email` lights up - ingredient pages, deep AI, monthly scan interpretation), card vaulted, auto-converts to $99/mo at +30 days, one-click cancel.

## Goal

Wire the reveal's unlock CTA to a $1 Stripe checkout that vaults the card; on return, create the membership (billing subscription + access grant) so the visitor is a full paid member with the deeper remedies released; bill $99/mo from +30 days via the existing daily cron; one-click cancel. Behind a flag (`BIOFIELD_TRIAL_ENABLED`, default off) because it is live money.

## Scope (#4b)

The $1 checkout endpoint, the return handler that creates the membership (subscription + grant) idempotently, the reveal's deeper-remedy release for paid members, the one-click cancel, and the unlock CTA wiring - all behind `BIOFIELD_TRIAL_ENABLED`.

**Out of scope:** a general "$1 trial" entry point not originating from the Biofield reveal; changing the $99/mo billing engine, the daily charge cron, or the grant-extend-on-charge logic (we reuse them); any change to #4a's free top-remedy flow (it stays).

---

## Confirmed decisions (Glen, 2026-06-20)

- **$1 trial of the FULL $99/mo tier:** $1 charged now + card vaulted + full paid membership active immediately; auto-converts to $99/mo at +30 days; one-click cancel.
- **The $1 unlocks everything gated on `_active_membership_for_email`** - the deeper Biofield remedies now, plus ingredient pages, deep AI, monthly scan interpretation. The Biofield depth is the hook into the whole tier.
- **Behind `BIOFIELD_TRIAL_ENABLED`** (default off, dark until flipped in Doppler) - live money.
- **One-click cancel** included in #4b (reuse `subscriptions.set_status(... 'cancelled')`).
- No emoji, no em dashes.

---

## Architecture

### The two membership records (the integration point)
"Paid member" = `_active_membership_for_email(email)` returns active, which reads the **`memberships`** GRANT table (`expires_at > now`). Recurring billing lives in the **`subscriptions`** table (`kind='membership'`, `amount_cents=9900`), charged by the daily cron. The existing flow EXTENDS the grant when a membership charge succeeds (app.py ~4390). So #4b's return handler creates BOTH: the subscription (billing) AND a `memberships` grant (access) - otherwise the depth would not release.

### 1. Flag + the unlock CTA
`BIOFIELD_TRIAL_ENABLED = os.environ.get("BIOFIELD_TRIAL_ENABLED", "").lower() in (...)`. In `static/begin-biofield.html`, the "Unlock your full Biofield Analysis" CTA: when the page-data says trial is enabled (a `trial_enabled` boolean in the reveal payload) the button reads "Unlock your full analysis ($1)" and POSTs to the checkout endpoint; otherwise it stays the disabled "unlocking soon" stub (current #4a behavior).

### 2. The $1 checkout - `POST /begin/biofield/<token>/unlock-checkout`
Verify the token + resolve the `biofield_reveals` row (reuse `_biofield_verify_token` / `get_by_token_hash`). Guards: `BIOFIELD_TRIAL_ENABLED` and `_STRIPE_ACTIVE` (else 404/`{ok:false}`); if `_active_membership_for_email(row.email)` is ALREADY active -> `{ok:true, already:true}` (the page just reloads to the full reveal, no charge). Else create a $1 Stripe checkout: `stripe_pay.create_checkout_session(amount_cents=100, customer_email=row.email, description="Biofield Analysis - full unlock", metadata={"email": row.email, "kind": "biofield_trial", "token": <token>}, save_card=True, success_url=f"{BASE}/begin/checkout-return?kind=biofield_trial&session_id={{CHECKOUT_SESSION_ID}}", cancel_url=f"{BASE}/begin/biofield/<token>")`. Return `{ok:true, url:<checkout_url>}`; the page redirects.

### 3. The return - `/begin/checkout-return?kind=biofield_trial`
Extend the existing `begin_checkout_return` (app.py ~4268) with a `kind=="biofield_trial"` branch (mirror `studio_claim_return`): read the session (`stripe_pay.get_session`), confirm the $1 PaymentIntent succeeded and pull `customer` + `payment_method` (the vaulted card); resolve `email` from the session metadata. Then, **idempotently** (guard on a per-session marker, mirroring studio's `already_granted`, so a reloaded return URL never double-creates):
- `subscriptions.create_membership(cx, email=email, stripe_customer_id=customer, stripe_payment_method_id=pm, amount_cents=MEMBERSHIP_AMOUNT_CENTS, next_charge_date=add_months(today, 1))` - the $99/mo billing, first charge at +30 days.
- Grant `memberships` access: INSERT a `memberships` row (the same shape as the admin grant ~18943) with `source="biofield_trial"`, `expires_at = +31 days` (a 1-day buffer past the first charge so access is continuous; the cron's extend-on-charge keeps it going; a failed $99 charge lets it lapse). Use/extract a small `_grant_membership(cx, email, days, source)` helper consistent with the existing INSERT.
Best-effort, never 500; on success redirect to `/begin/biofield/<token>` (the full reveal). On a payment that did not succeed -> redirect back to the reveal with no membership.

### 4. Release the deeper remedies for paid members
In `begin_biofield_reveal` (the page-data) and `_biofield_top_payload` area: compute `paid = _active_membership_for_email(row.email) is not None`. When `paid`:
- emit the FULL ranked remedies (top + `remedies[1:]`), each as `{name, meaning, buy_url, page_url}` (the deeper ones gain the same `_blank` product-page name-link + Order button as the top), with `blurred_count = 0`.
When NOT paid: unchanged from #4a (top via the one-time free button if approved; the rest a `blurred_count`, withheld). Add `trial_enabled` (the flag) and `paid` to the payload so the page renders the right CTA. Anti-bypass preserved: deeper remedy details still leave the server ONLY when the email is a paid member.

### 5. One-click cancel
On the post-unlock confirmation (and reachable from the reveal once paid), a "Cancel anytime - no charge" link to `GET /membership/cancel/<token>` where `<token>` is a tokened cancel link (mint an `auth_tokens` row `purpose="membership_cancel"` for the email at grant time, emailed in the "you're in" confirmation and/or rendered on the reveal for the verified owner). The route verifies the token, finds the email's active `kind='membership'` subscription, and calls `subscriptions.set_status(cx, sub_id, "cancelled")` (stops the +30d charge; the `memberships` grant runs out its paid window). Confirm with a calm "your trial is cancelled, no further charge; access continues until <date>" page. Idempotent (cancel twice = still cancelled).

### Reuse / untouched
- Stripe: `stripe_pay.create_checkout_session` / `get_session` / the PI read; `_STRIPE_ACTIVE`.
- Subscriptions: `create_membership`, `set_status`, `add_months`, `MEMBERSHIP_AMOUNT_CENTS`; the daily charge cron + the grant-extend-on-charge logic (UNCHANGED).
- Membership access: the `memberships` grant table + `_active_membership_for_email` (the same check the ingredient page uses).
- The #4a reveal store/route/token; `auth_tokens` magic-link; the checkout-return handler.
- Untouched: #4a's free top-remedy flow, the billing engine, the pricing engine, the journey/funnel.

---

## Data flow
1. Paid-tier flag on. The reveal shows "Unlock your full analysis ($1)".
2. Visitor clicks -> `unlock-checkout` -> $1 Stripe checkout (card vaulted).
3. Stripe return (`kind=biofield_trial`) -> confirm $1 paid -> create subscription (+30d $99) + grant `memberships` access (+31d) idempotently -> redirect to the reveal.
4. The reveal (now paid) releases ALL remedies (top + deep, each linked); ingredient pages + deep AI also unlock (same gate).
5. +30 days: the daily cron charges $99 and extends the grant. Cancel before then -> no charge; access until the grant expires.

## Error handling
- Flag off OR Stripe inactive -> the CTA stays the "unlocking soon" stub; `unlock-checkout` returns `{ok:false}`; no charge path.
- Already a paid member -> no checkout; the reveal already shows the full depth.
- Return with an unpaid/failed PI -> no membership created; redirect to the reveal unchanged (still blurred). The $1 is only ever charged by Stripe on a completed checkout.
- Idempotent membership creation: a re-loaded return URL never double-creates the subscription or grant (per-session marker, like studio `already_granted`).
- The whole return is best-effort and never 500s; a grant/subscription failure logs and the visitor can retry.
- Cancel: invalid/expired token -> friendly page; cancelling an already-cancelled sub is a no-op; access continues until the paid window ends (we do not refund or revoke the paid period).
- Anti-bypass: the deeper remedy content is released ONLY for a paid member (server-side); a non-paid visitor cannot obtain it via the page-data or `reveal-top`.

## Testing
- **Checkout:** `unlock-checkout` with the flag off -> `{ok:false}`; flag on + non-member -> creates a $1 checkout (mock `stripe_pay.create_checkout_session`, assert amount_cents=100 + save_card + the metadata) and returns the url; an already-paid member -> `{ok:true, already:true}`, no checkout.
- **Return:** the `kind=biofield_trial` branch with a mocked paid session (customer+pm) -> a `subscriptions` membership row (amount 9900, next +30d) AND a `memberships` grant (expires ~+31d) exist, and `_active_membership_for_email(email)` is active; a re-run of the same session does NOT create a second subscription/grant (idempotent); a non-succeeded PI creates neither.
- **Depth release:** the Biofield reveal page-data for a PAID member -> full remedies (top + deep, each with `page_url`/`buy_url`), `blurred_count=0`; for a NON-paid member -> unchanged #4a (top-or-free-button, blurred count, NO deep content). `reveal-top` unchanged.
- **Cancel:** a valid cancel token -> the active membership subscription status becomes `cancelled`; an invalid token -> friendly page, no change; double-cancel -> still cancelled.
- Stripe is mocked throughout (no live charge in tests). deploy-chat test isolation (tmp `LOG_DB`; init the subscriptions + memberships + biofield_reveals + auth_tokens tables; mock `_active_membership_for_email` where asserting the gate vs reading the real grant as appropriate). No emoji; no em dashes.
- Front-end (the $1 button, the post-unlock confirmation, the cancel link) = manual visual pass.

## Notes
- **Live money, behind `BIOFIELD_TRIAL_ENABLED` (default off).** Merge is dark; go-live = flip the flag in Doppler (with `STRIPE_ACTIVE` already on). Test the $1 charge end-to-end in Stripe before flipping.
- The $1 is charged NOW (the checkout); the $99 recurs from +30 days via the existing cron - month 1 is effectively $1.
- This is the conversion engine for the whole paid tier: the same `_active_membership_for_email` gate already governs the ingredient pages (#ingredient) and will govern deep AI / monthly scan interpretation, so the $1 unlock lights all of them at once.
- All copy (the button, the confirmation, the cancel page) is provisional - BNSN pass later.
