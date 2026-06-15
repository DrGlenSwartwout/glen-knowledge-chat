# Studio.com bridge — Flow B (free first month of live group)

A customer who joined Studio.com at **studio.com/drglen** can claim their **first month of
live group coaching free**, then auto-continue at $99/mo unless cancelled. Ships dark behind
`STUDIO_BRIDGE_ENABLED` (default off). This is Mechanic 2 (Flow B) of the upgrade ladder.

## Why both directions pay
Studio.com pays us creator rev-share on every participant, including ones we refer. So driving
people to studio.com/drglen earns us rev-share, and the free group month (near-zero marginal
cost) is the carrot that pulls them into our paid live group + the rest of the ecosystem.

## Flow
1. **Claim** (`/studio/claim`): the customer enters name + email + ToS consent (so the member
   gate passes — claiming also makes them our free member), self-attests "I joined at
   studio.com/drglen" (optional receipt reference), and sees the disclosure: "After your free
   month, live group continues at $99/mo unless you cancel."
2. `POST /api/studio/claim` — consent-gated (`is_member`, 403 `need_optin` otherwise); records a
   pending claim; creates a Stripe **mode=setup** Checkout session that **vaults a card with no
   charge** (the customer paid Studio.com, not us); returns the Stripe URL.
3. `GET /studio/claim-return` — reads the session's `setup_intent` → `customer` + `payment_method`;
   creates a `kind='membership'` subscription ($99/mo, `next_charge_date = today + 1 month` — the
   free month), marks the claim granted. Idempotent (one grant per email). The existing membership
   charge cron bills the $99 off-session after the free month.

## Confirmation
The Studio.com creator dashboard exposes no subscriber emails, so the signup is confirmed by
receipt/self-attest (not an automated match). The downside is trivial — the free month is ~$0.

## Compliance
Auto-charging after the free month is a negative-option offer: explicit opt-in + the disclosure
above + the Mechanic 1 pre-charge reminder + one-click cancel (the membership cancel path).

## Flags / config
- `STUDIO_BRIDGE_ENABLED` (default off) — gates the claim API + the return grant.
- Reuses the Mechanic 1 membership rail + charge cron + `BIOFIELD`/group `MEMBERSHIP_AMOUNT_CENTS` ($99).

## Flow A — clinical-wedge welcome (`/studio`)
A welcome landing page (`/studio` → `static/studio-welcome.html`) that layers Dr. Glen's clinical
wedge on Studio.com users: a free Biofield voice scan + remedy match (`Truly.VIP/E4L`, the thing a
phone app can't do), the deeper remedy-aware AI Q&A (`/begin`), free membership + courses
(`/begin`), and the Flow B free first month of live group (`/studio/claim`). It reuses existing
destinations — no new backend. It stamps a last-touch `rm_ref=studio` attribution cookie (only when
no real affiliate ref is set) + an `amg_session`, so any later membership/order is attributed
`source=studio`. The flywheel: Studio.com pays us rev-share on participants, and the wedge converts
their users into our Biofield / remedy / live-group revenue.

Note: the `/studio` page itself is public positioning (harmless before launch); the "first month
free" card links to `/studio/claim`, which is gated by `STUDIO_BRIDGE_ENABLED` and degrades
gracefully (disabled) until the flag is on. Gate the whole page too if a fully-dark launch is wanted.

## Deferred
- A dedicated Studio-user onboarding sequence / Studio-specific concierge mode (the generic funnel +
  chat suffice for now).
- Auto-verify of the Studio signup (no email export); refund/clawback of an un-charged free month.
