# Practitioner-Portal Affiliate Dashboard Render (read-only)

**Date:** 2026-06-26
**Status:** Approved (Glen: "read-only")
**Parent:** the personal-portal unification program. 2b-1 brought the enrolled-affiliate dashboard into the *client* portal; #332 attached the same `ambassador` block (incl. `dashboard`) to the *practitioner* portal payload but the practitioner render stayed thin (only the two links). This closes that gap.

## Problem

An enrolled practitioner-ambassador opening their portal sees only the referral + invite links — none of the stats/offers/recent/recruit/social that the client portal shows. The data is already present in `d.ambassador.dashboard` (build_dashboard via `_ambassador_block`); only the `static/practitioner-portal.html` enrolled-branch render omits it.

## Design

Single-file frontend change to `static/practitioner-portal.html`, enrolled branch only. Mirror the client-portal enrolled render (client-portal.html ~472–516) adapted to the practitioner card idiom (`escapeHtml`, `--gold`/`--border`/`--muted`, inline styles, innerHTML string-build). Renders, after the existing two links:
- **Stats** — Leads / Last lead / Conversions / Member since (each shown only if present).
- **Offers** — name + description + view link + instructions.
- **Recent referrals** — name + score + received_at, or "No referrals yet."
- **Recruited** — count linking the recruit_url (if present).
- **Social shares** — list of url + points/views/likes/shares, or "No social shares yet."

## Non-goals / read-only decision

- **No add-social-share form.** The POST `/api/portal/<token>/social-links` authenticates via `resolve_identity` on the *client* portal token/session; a practitioner-portal token won't resolve, so an add-form would fail. Adding shares stays in the client portal (or a later increment that wires a practitioner-identity→slug auth path).
- No server-side change, no affiliate-engine change, no change to pending/none branches or the rest of the practitioner portal.

## Testing

- JS syntax-check the inline script (node --check on the extracted script).
- Render-verify: headless-load the static page with a mocked enrolled `d.ambassador.dashboard` payload → assert the dashboard DOM (stats/offers/recent/social) renders + zero console errors. (A live practitioner session is session-gated and can't be curled; the mocked-payload headless render exercises the exact new code path.)
- Confirm non-enrolled (none/pending) still render the plain CTA / review notice.

## Rollout

Ships on merge → Render deploy. Pure static asset; no migration, no flag.
