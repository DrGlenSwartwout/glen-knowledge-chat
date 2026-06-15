# Program-bundled live group coaching (Mechanic 1)

A remedy-program purchase can bundle **live-group coaching free**, then auto-continue at
$99/mo on the Stripe vault. Ships dark behind `GROUP_BUNDLE_ENABLED` (default off).

## The offer
- **1 free live-group month per program month purchased, capped at 3** (the max recommended
  program length). A Biofield-only / no-program purchase grants 0. Past 3 months the patient
  re-Biofields and a new program restarts the window (`group_bundle.included_group_months`).
- After the free window, the membership **auto-continues at $99/mo** (founders) unless
  cancelled. Net effect: group is free while on the quarterly Biofield → program cycle, and
  the $99/mo charge monetizes the gap when the patient lapses.

## Opt-in + compliance (required)
Auto-charging after a free trial is a negative-option offer, so it is **explicit opt-in, not
silent enrollment**: a checkbox at the funnel checkout ("Add live group coaching — up to 3
months free with your program, then $99/mo, cancel anytime"), a 3-day pre-charge reminder
email, and one-click cancel via the member portal. Card payments only (Zelle/Wise don't vault
a card, so there's no payment method to auto-continue).

## Rail
Stripe vault + the Subscribe-and-Grow scheduler — `subscriptions` rows with `kind='membership'`,
flat `amount_cents` (9900), `cadence_months=1`. The charge cron (`/api/cron/charge-subscriptions`)
has a membership branch that charges the flat amount off-session and writes a one-line
"Live Group Coaching" invoice. QBO recurring is the deferred alternative (pending QBO Payments
approval).

## Flow
1. **Checkout** (`/begin/checkout/<slug>`, card path via `_stripe_checkout_url_for_retail`): when
   `GROUP_BUNDLE_ENABLED` and the buyer opted in and program months ≥ 1, the Stripe session is
   created with `save_card=True` and metadata `grant_group_months` (= included months) + `email`.
2. **Return** (`begin_checkout_return`): a flag-gated grant block reads the vaulted card from the
   PaymentIntent and creates a `kind='membership'` subscription with the first charge set to
   `today + N months` (the end of the free window). A per-invoice marker
   (`group_bundle_grants`) makes a literal re-run a no-op; a genuinely new program order for a
   member with an active membership **extends** the next charge date by N months (window-stacking).
3. **Charge cron**: when the free window ends, charges $99/mo off-session, advances, sends a
   membership-flavored receipt; the heads-up pass sends a membership-flavored 3-day reminder.
4. **Cancel**: `set_status(...,'cancelled')` — excluded from `list_due`/`list_heads_up_due`.

## Flag + open item
- `GROUP_BUNDLE_ENABLED` (env, default off) gates the checkout opt-in, the grant, and the page checkbox.
- **Before flipping it on:** confirm what qualifies as a program order — every remedy order, only
  Biofield-designed programs, or a minimum size — so a one-bottle buyer isn't offered a $99/mo
  membership. (Deferred; the dark flag keeps it safe until decided.)

## Deferred (v2)
QBO-Payments recurring rail; proration; more than one membership per email; richer member-portal
management of the trial window.
