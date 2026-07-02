# Sub-project 4 — Two-Door "See & Choose" Surface — Design

**Date:** 2026-07-02
**Part of:** the healing-first offer redesign (see `project_membership_prepay_ladder`).
**Status:** flag-dark build for Glen's review/merge. Placement is Glen's taste call; recommended option below chosen while he was away, reversible before merge.

## Goal

After a client has seen their Biofield analysis and their matched remedies, give them one
clear moment to choose **how they want to continue healing**:

- **Door A — On your own (à la carte):** order remedies as you need them, re-scanned only
  when you're ready. Cost-based quantity pricing, open to all.
- **Door B — Continuous Care (~$99/mo):** ongoing guided healing — protocol re-matched each
  month, live group coaching, AI ally, Terrain Restore. Term-based prepay, never auto-renews.

This is the "→ choose solo vs Continuous Care" step of Glen's two-door customer path.

## Placement decision

**A new flag-gated page `GET /begin/choose` → `static/begin-choose.html`**, cloned from the
existing `static/begin-path.html` aesthetic (dark healing palette, Raleway/Open Sans, card
grid). Reached by a **handoff CTA on the reveal page** (`static/begin-biofield.html`).

Rejected alternatives: extending the reveal page (already dense: unlock deposit + à-la-carte
cart + $100 program CTA; token-scoped/no-store — a third decision block muddies it) and a
standalone non-`/begin` route (loses the free journey-engine + nav-shell integration).

## Key constraint that shapes the wiring

À-la-carte remedy ordering is **token-scoped**: it runs through
`POST /begin/biofield/<token>/order-checkout`, which needs the reveal token. Continuous Care
(`/prepay`) and the $100 program (`/biofield/checkout`) are email-scoped, not token-scoped.

Therefore:
- The **reveal page is the origin** (it holds the token). Its CTA links to
  `/begin/choose?token=<token>`.
- The `/begin/choose` route **verifies the token server-side** (same `_biofield_verify_token`
  path as the reveal route) and injects `window.__CHOOSE__` with the reveal URL + flags. Email
  is **not** placed in the URL — the token (already in the user's URL bar on the reveal) is the
  only bearer value carried forward, consistent with existing exposure.
- **Door A** → links back to the reveal cart (`__CHOOSE__.reveal_url` = `/begin/biofield/<token>`),
  where the token-scoped cart already lives.
- **Door B** → `/prepay` (the Continuous Care picker, which collects its own email).

If `?token=` is missing/invalid, the page still renders both doors with a generic payload
(Door A points at `/begin` since there's no token; Door B → `/prepay`).

## Components

### 1. Flag: `TWO_DOOR_ENABLED`
New env flag near the other funnel flags (app.py ~4500), same idiom
(`os.environ.get("TWO_DOOR_ENABLED","").strip().lower() in ("1","true","yes","on")`). Default
off → the whole surface is inert (route redirects, reveal CTA hidden).

### 2. Route: `GET /begin/choose`
- Flag off → `redirect("/")` (mirrors `/prepay` when `PREPAY_LADDER_ENABLED` off).
- Flag on: read `?token=`. If a valid reveal token → inject
  `window.__CHOOSE__ = {token, reveal_url, program_enabled, program_tier, prepay_enabled}`
  (email intentionally omitted from client payload; not needed by either door). If token
  absent/invalid → inject `{token:null, reveal_url:"/begin", program_enabled, program_tier,
  prepay_enabled}`.
- `no-store` headers (same as reveal + prepay).
- JSON injection escapes `<`,`>`,`&` before `</head>` (same helper pattern as reveal route).

### 3. Page: `static/begin-choose.html`
Cloned from `begin-path.html`: brandbar, eyebrow "Choose how you continue", two `.card`s in a
grid. Reads `window.__CHOOSE__`. Door A card → `reveal_url`; Door B card → `/prepay`. Fires a
`care_fork` analytics unlock via `POST /begin/unlock` on load and on each door click with
`{care_choice:'solo'|'care'}` — **does not** pass a `path` kwarg, so the `journey_state.path`
column (private/pay_forward semantics) is untouched. Inherits the nav shell automatically under
`JOURNEY_SHELL_ENABLED`. Includes the standard `theme-toggle.js` + `ref-capture.js` + `widget.js`.

### 4. Reveal-page handoff CTA
In `static/begin-biofield.html`, add a CTA "Choose how you want to continue →" that navigates
to `/begin/choose?token=<token>` (the reveal already has `token` in scope). Gated by a new
payload flag `choose_enabled` (= `TWO_DOOR_ENABLED`), added to **both** member payload branches
(paid + non-paid) in `begin_biofield_reveal`. When the flag is off, `choose_enabled` is false →
CTA hidden → reveal page byte-unchanged. Deploy-safe.

## What this does NOT change

- No pricing-engine change. No new checkout endpoint. No `subscriptions`/`memberships` writes.
- The `path` column and `paid_fork`/`choose_path` semantics are untouched (à-la-carte-vs-care
  is a distinct decision from private-vs-pay-forward; it uses its own `care_fork` trigger).
- With `TWO_DOOR_ENABLED` off, prod behavior is byte-for-byte unchanged.

## Testing

- **Route:** flag off → 302 to `/`; flag on + valid token → 200 with `__CHOOSE__` carrying the
  right `reveal_url` and flag values; flag on + invalid/absent token → 200 with generic payload
  (`reveal_url == "/begin"`); `Cache-Control: no-store` present.
- **Reveal CTA:** `begin_biofield_reveal` member payload includes `choose_enabled` reflecting
  `TWO_DOOR_ENABLED` (patched true → present/true; default → false), for both paid and non-paid.
- **Page:** the served route returns `begin-choose.html` content (door hooks present).

## Open question for Glen (default chosen, changeable before merge)

Door A "On your own" currently loops back to the reveal cart. If Glen would rather Door A lead
to the **$100 program front door** (`/biofield/checkout`) or present both à-la-carte AND program,
that's a one-card copy/destination change. Default = back-to-reveal-cart (simplest, uses an
endpoint that already works end-to-end).
