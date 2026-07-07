# Migrate client portals to myhealingoasis.com

**Date:** 2026-07-06
**Status:** Draft for review
**Owner:** Glen (decisions) + Claude (implementation)

## Goal

Serve the client portal experience on its own domain, **myhealingoasis.com**, instead of
`illtowell.com/portal/*`. The funnel / home / chat stay on illtowell.com. Portal magic-links,
the "My Healing Oasis" button, and portal emails all point to the new domain.

## Current state (verified 2026-07-06)

- **myhealingoasis.com** is registered at Namecheap (exp. 2026-09-01) and is **already on
  Cloudflare nameservers** (`dora`/`major.ns.cloudflare.com`). No nameserver migration is needed.
- Cloudflare currently proxies the domain to a **Groove** origin serving a live "My Healing Oasis"
  page. Glen has confirmed that page is **safe to replace**.
- The myhealingoasis.com Cloudflare zone lives in a **different Cloudflare account** than the one
  our API token manages (token covers `this.elf@gmail.com`'s account: illtowell.com + 25 others).
  → The one DNS record change is gated on access to the account that currently holds the zone.
- The app (`glen-knowledge-chat` on Render, `srv-d7n83o7avr4c73ff7gig`) already serves `/portal/*`
  on any Host. Portal links are built from a single global `PUBLIC_BASE_URL` (= illtowell.com),
  used across ~15 externally-delivered link sites (welcome email, claim link, resend, reveal-push
  analyze link, EVOX, the Healing Oasis front door, console link lookup, etc.).

## Approach

One app serving two hosts. Introduce a portal-specific base URL so **only portal links** move to
the new domain while everything else stays on illtowell.com. Default it to `PUBLIC_BASE_URL` so the
code deploy is a behavioral no-op until we flip the env var — the migration is a config flip, not a
big-bang code change.

### 1. Code (Claude)

- **New env `PORTAL_BASE_URL`**, defaulting to `PUBLIC_BASE_URL`. Add a helper
  `portal_base()` / `_portal_link(token)` and route every **externally-delivered** portal link
  through it. Sites to convert (all current `f"{PUBLIC_BASE_URL}/portal/..."` and the hardcoded
  `https://illtowell.com/portal/login` strings):
  - `send_portal_welcome` login + token links (~578, 609, 652)
  - Healing Oasis front door link (the `ensure_portal_token` link in `/api/healing-oasis/request`)
  - `_portal_claim_url` signed claim link
  - `/admin/portal/get-or-create-link`, `/admin/portal/rollout-enroll`
  - `/portal/login-request` verify link + `/api/console/portal-link[/resend]`
  - reveal-push / notify links (`.../portal/{token}`, `.../portal/{token}/analyze`)
  - booking `portal_url` payloads (~15460, 15562, 15961, 17393)
  - Leave **in-app** relative redirects (`/portal/login`, `redirect(f"/portal/{token}")`) as-is —
    they resolve against whatever host the user is already on, which is correct.
- **Oasis-host root:** when `Host == myhealingoasis.com`, `/` redirects to `/portal/login`
  (portal front door) instead of the funnel. Minimal special-casing — all other routes keep
  working on both hosts. Inert until DNS points the host at us.
- **Backward compatibility:** `illtowell.com/portal/*` keeps working unchanged, so every link
  already emailed still resolves. (Optional Phase 2 nicety: 301 `illtowell.com/portal/<token>`
  → new host for token links only; not session pages, to avoid cross-domain cookie confusion.)

### 2. Infra (Claude via API, except the one DNS edit)

- **Render:** add `myhealingoasis.com` (+ `www`) as custom domains on `srv-d7n83o7avr4c73ff7gig`
  (Render API). Render returns the DNS target (A `216.24.57.x` / CNAME to `glen-knowledge-chat.onrender.com`)
  and an ACME validation record, and issues the TLS cert once DNS points at it.
- **Cloudflare DNS (one edit, gated on account access):** point `myhealingoasis.com` at the Render
  target. Because Cloudflare-proxy + Render-managed-cert can deadlock on first issue, the safe
  recipe is: set the record **DNS-only (grey cloud)** until Render's cert validates and HTTPS works
  on the host, then flip to **proxied + SSL Full (strict)**.
  - If the zone is moved into the main account first, Claude can do this edit via API.
  - Otherwise Glen makes this single edit in the account that holds the zone; Claude supplies the
    exact record values.

### 3. Cutover sequence (safe, reversible)

1. Ship the code with `PORTAL_BASE_URL` **unset** → identical behavior (still illtowell). No-op deploy.
2. Add `myhealingoasis.com` to Render custom domains.
3. Point the Cloudflare DNS record at Render (grey cloud); confirm cert issues and
   `https://myhealingoasis.com/portal/login` loads; then enable proxy + Full (strict).
4. Set `PORTAL_BASE_URL=https://myhealingoasis.com` in prod env; redeploy.
5. Verify end-to-end: request a Healing Oasis link, confirm the emailed link is on the new domain
   and resolves to a working portal.
6. (Optional) add the illtowell → oasis token redirect.

### Rollback

Unset `PORTAL_BASE_URL` (all links revert to illtowell instantly on redeploy). DNS can revert to the
Groove origin. The Render custom domain can be left in place harmlessly.

## Testing / verification

- Unit: `portal_base()` returns `PORTAL_BASE_URL` when set, else `PUBLIC_BASE_URL`; a representative
  set of link builders (welcome, claim, Healing Oasis, resend) emit the new host when set.
- Host routing: request `/` with `Host: myhealingoasis.com` → 302 to `/portal/login`; with
  `Host: illtowell.com` → funnel unchanged.
- Regression: existing portal-link + portal-route tests stay green.
- Live: browser render on the new host after cutover — portal login loads over valid HTTPS; a
  real Healing Oasis request emails a `myhealingoasis.com` link that opens the portal.

## Who does what

| Step | Owner |
|------|-------|
| All code (`PORTAL_BASE_URL`, host routing, link conversion, tests) | Claude |
| Add Render custom domain (API) | Claude |
| The one Cloudflare DNS edit | Glen (or Claude, if the zone is moved to the main account) |
| Optional: move the zone to the main account (add-site + release-old + Namecheap NS change) | Glen (Claude can do the new-account zone/record API parts) |
| Flip `PORTAL_BASE_URL` in prod + verify | Claude |

## Out of scope

- Moving the funnel / home / chat off illtowell.com (portals only).
- Any Groove content preservation (page confirmed disposable).
- Changing the "My Healing Oasis" button UX (still a name+email modal; only its emailed link's host changes).

## Open items

- Which Cloudflare account currently holds myhealingoasis.com (needed for the DNS edit or the zone move).
- Decide now vs later on the optional zone consolidation.
