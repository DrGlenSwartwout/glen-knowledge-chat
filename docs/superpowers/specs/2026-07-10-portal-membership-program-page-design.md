# Client-portal Membership Program page — design

**Date:** 2026-07-10 · **Status:** design for review · **Author:** Glen + Claude

A new, separate page inside the client portal that presents and sells the whole
membership program (Free → Paid → Family → Ambassador), gives enrolled
ambassadors their affiliate links and share tools, and lightly introduces the
practitioner / coach / certification paths. Personalized to the viewer's current
tier. Sell/convert leads; the affiliate hub unlocks for ambassadors.

---

## 1. Goal & framing

- **Primary job: sell/convert.** Move a client up the ladder (Free → Paid →
  Family) and invite advocacy (Ambassador). Benefits-first copy, not a cold
  pricing matrix — consistent with the offer-architecture principle "a path, not
  a pricing matrix" ([[2026-07-02-offer-architecture-design]]).
- **Secondary: affiliate hub** for enrolled ambassadors — their referral link,
  recruit link, and share tools, on their own page.
- **Tertiary: grow-with-us intro** to practitioner / coach / certification,
  closing the CTA gap where those funnels exist as standalone routes but are not
  linked from the client portal today.
- **Personalized by tier.** Every viewer is already *someone* in the portal;
  the page must reflect their real standing, never pitch what they already own.

## 2. Relationship to existing surfaces (reconciled with Glen 2026-07-10)

- **Coexists with the simple "Options & Pricing" card**
  ([[2026-07-08-portal-options-pricing-page-design]]). That card stays as the
  same-for-everyone 3-item at-a-glance orientation. This program page is the
  richer "see everything / grow with us" destination. Do NOT merge the two
  pricing surfaces.
- **"Paid" = the guided recurring membership** (~$99/mo Continuous Care),
  surfaced through the existing live-group / subscription checkout
  (`portal_offers._LADDER` `live_group` rung, checkout `/portal/offer/live-group/checkout`).
- **Family Plan ($147/mo) stays on the page** as the household version of Paid,
  even though the offer-architecture doc predates it — it is live in code
  (`dashboard/family_plan.py::PLAN`) and billable today.
- **Ambassador uses the affiliate-slug system only** (`affiliate_signups` +
  `dashboard/portal_view.py::_ambassador_block`). The separate referral-*code*
  system (`dashboard/referrals.py`) is left untouched to avoid conflating two
  link systems.

## 3. Route & architecture

Mirror the existing `/portal/<token>/analyze` pattern (`app.py:16211`):

- `GET /portal/<token>/program` → serves new `static/portal-program.html`
  (static shell + JS that renders from the API, same as `client-portal.html`).
- `GET /api/portal/<token>/program` → returns the personalized `program` JSON.
- Resolves for both the emailed token link and logged-in `/portal/me`
  (reuse the same identity resolution `api_client_portal` uses — token-first,
  see [[reference_portal_token_vs_session]]).
- **Entry point:** one card near the top of the main portal stack in
  `static/client-portal.html` — copy e.g. *"See everything your membership
  unlocks"* — linking to `/portal/<token>/program`. Single doorway; no top nav
  for v1.

## 4. Personalization — state per tier

The API derives each tier's **state** at request time from the predicates that
already exist. No new stored "tier" field.

| Tier | State source | States rendered |
|---|---|---|
| Free | baseline — everyone in the portal has it | `owned` (✓ "you have this" + benefits) |
| Paid (guided ~$99/mo) | `_active_membership_for_email(email)` / `_portal_biofield_unlocked` membership branch (`app.py:10647`) | `owned` ✓ **or** `available` (sell card + CTA) |
| Family ($147/mo) | `family_plan.covers(cx, email)` (`dashboard/family_plan.py:95`) | `owned` ✓ **or** `available` (sell card + CTA) |
| Ambassador | `_ambassador_block(cx, email, quiz_url, base)` states `none` / `pending` / `enrolled` (`dashboard/portal_view.py:132`) | `available` (apply CTA) / `pending` (status) / `enrolled` (**affiliate hub**) |
| Practitioner / Coach / Cert | always shown as intro band (no per-viewer state in v1) | `intro` (blurb + link out) |

Each `available` sell card's CTA also respects the tier's own **commerce flag**
(`SUBSCRIPTIONS_ENABLED` for Paid, `FAMILY_PLAN_ENABLED` for Family). If a
tier's commerce is off, render it as an info / "coming soon" card — **never a
dead buy button**.

## 5. Page sections (top → bottom)

1. **Hero** — greeting + a badge showing current standing (Free / Paid / Family /
   Ambassador). Benefits framing, not a price table.
2. **Free membership** — what being in the portal already gives them (analysis,
   remedies access, referral tracking). Renders `owned` ✓ so the value is
   visible and the paid jump feels earned.
3. **Paid — guided membership (~$99/mo Continuous Care)** — benefits + CTA reusing
   the `live_group` checkout. `owned` → ✓ state.
4. **Family Plan ($147/mo)** — whole consented household; benefits + CTA reusing
   the existing family-plan checkout. `covers()` → ✓ state.
5. **Ambassador** — not enrolled → apply CTA (`{base}/affiliate/apply-form`);
   pending → status; enrolled → **affiliate hub (v1 = links + tools only)**:
   `referral_url`, `recruit_url`, copy-to-clipboard, prebuilt share text (all
   already produced by `_ambassador_block` / `affiliate_dashboard`). A light
   stats line is a **fast-follow, out of scope for v1**. Link out to the
   existing full affiliate dashboard for leads/earnings.
6. **Grow with us** — light intro band: Practitioner (wholesale) → `/practitioner/register`;
   Coach training; Certification → `/cert`. Short blurb + link each.
7. **Footer** — "Questions? Ask Dr. Glen" (portal concierge) + back-to-portal link.

## 6. Single source of truth for tiers

New module **`dashboard/program_tiers.py`** — pure/cx-based, never imports `app`
(same discipline as `dashboard/portal_offers.py`). Holds each tier's:

- `key`, `name`, benefit bullets (`copy`), `cta_label`, `checkout_path`
- **price REFERENCED from existing constants, never re-hardcoded:**
  - Paid: `portal_offers.MEMBERSHIP_PRICE_CENTS` (= `prepay.MONTHLY_ANCHOR_CENTS`)
  - Family: `family_plan.PLAN` (`amount_cents` / `value_cents` / `label`)
- a `state(cx, email)` resolver per tier calling the predicates in §4.

Rationale: prices have drifted repeatedly; a hardcoded page silently goes stale
([[feedback_confirm_channel_live_before_fixing]]). This mirrors how the current
Family Plan card already reads `family_plan.PLAN` rather than hardcoding $147.

The `/api/portal/<token>/program` endpoint composes `program_tiers` output +
`_ambassador_block` into one `program` object. **Best-effort block** — a failure
in one tier resolver must not break the page load (same pattern as
`payload["invoices"]`). A field reaches the page only if it is BOTH in the
payload dict AND rendered ([[feedback_portal_content_payload_surface]]).

## 7. Flags & safety

- Whole page behind a new flag `PORTAL_PROGRAM_PAGE_ENABLED` (default off; ships
  dark, Glen flips it). **Check presence, never print the value** — prod flags
  have been observed *deleted* rather than set false
  ([[reference_prod_flags_deleted_not_off]]).
- Entry-point card in `client-portal.html` also gated by the same flag.
- Per-tier commerce flags respected as in §4 (no dead buy buttons).

## 8. Testing

- API returns a `program` object with per-tier `state` correct for: a free
  client, a paid/member client (Paid shows ✓), a Family-covered caregiver
  (Family shows ✓), a household dependent covered via `covers()`, and each of
  the three ambassador states.
- No dollar figure is hardcoded in `portal-program.html`; Paid/Family prices
  trace to `MEMBERSHIP_PRICE_CENTS` / `family_plan.PLAN`.
- Commerce flag off for a tier → that tier renders info/coming-soon, not a buy
  button.
- Page flag off → route 404/redirect and no entry card; on → both present.
- Render the actual page in a headless browser and confirm the personalized
  states show, not just that the payload is correct
  ([[feedback_render_the_page_not_the_payload]]).

## 9. Out of scope (v1 — YAGNI)

- No new checkout/payment code — reuse existing `live_group` and family-plan
  checkout paths.
- No new affiliate dashboard — the affiliate hub is links + tools only; full
  leads/earnings stays at the existing affiliate portal, linked out.
- Affiliate stats line on the hub — fast-follow, not v1.
- No changes to the referral-*code* system (`referrals.py`).
- No editing of tier prices from this page.
- No per-viewer personalization of the practitioner/coach/cert band (static
  intro in v1).
- Does not replace or modify the simple Options & Pricing card.
