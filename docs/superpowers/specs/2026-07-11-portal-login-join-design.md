# Add a "join" path to the portal login page (myhealingoasis.com front door)

**Date:** 2026-07-11
**Status:** Approved for implementation
**Owner:** Glen (decisions) + Claude (implementation)

## Goal

`myhealingoasis.com` root redirects to `/portal/login`, served by
`static/client-login.html`. Today that page is **sign-in only** — one email field →
`/portal/login-request`, which by design does NOT provision a portal for an unknown
email (account-enumeration protection). So a **brand-new visitor has no way to join**
from the portal front door. Add a second, quieter *join* action so the front door
serves both returning clients and new visitors.

## Current state (verified 2026-07-11)

- `/portal/login` (GET) serves `static/client-login.html`, gated by
  `CLIENT_LOGIN_ENABLED`. It POSTs to `/portal/login-request` (sign-in link for
  KNOWN emails only; non-enumerating).
- The "provision anyone" plumbing already exists and needs **no backend change**:
  - `POST /api/healing-oasis/request` — provisions a portal for any `{name, email}`
    and emails the magic link. Gated by `HEALING_OASIS_ENABLED`. Rate-limited,
    non-enumerating, never returns the token.
  - `GET /api/healing-oasis/status` — `{enabled: bool}`, lets a static page reveal
    the join affordance only when the feature is live.
- Precedent: `static/begin.html` already reveals its "My Healing Oasis" button off
  `/api/healing-oasis/status` and submits to `/api/healing-oasis/request`.

## Approach — two distinct actions (approved)

Keep **"Send my sign-in link"** as the primary action for returning clients.
Below it, a visually secondary **join** block: a "New here?" separator, a name
input, and a **"Create my Healing Oasis"** button that POSTs `{name, email}` to
`/api/healing-oasis/request`. The email input is **shared** with sign-in (no
duplicate email entry). Both paths just email a link, so the UX is symmetric.

Rejected: a single combined "Continue" button — it would have to merge the two
endpoints and weaken the deliberate no-account-enumeration property of sign-in.

### Frontend (only change)

`static/client-login.html`:

- Add a `.join` block (hidden by default) after the sign-in button + message:
  a "New here?" separator, one-line sub, `#joinName` input, and a secondary-styled
  `#joinGo` button. Reuses the existing `#email` input and shared `#msg` line.
- On load, `fetch('/api/healing-oasis/status')`; reveal the `.join` block only when
  `enabled` (mirrors begin.html). Use `[hidden]{display:none}` guards — including
  `.btn[hidden]{display:none!important}` to beat the `.btn{display:block}` rule
  (the known hidden-vs-display gotcha).
- `#joinGo` handler: validate email, POST `{name, email}` to
  `/api/healing-oasis/request`, show `j.message` in `#msg`.

### Gating (unchanged behavior)

- Sign-in stays dark behind `CLIENT_LOGIN_ENABLED` (the whole page 404s when off).
- The join affordance stays dark behind `HEALING_OASIS_ENABLED` (revealed only when
  `/api/healing-oasis/status` reports enabled). Both flags flip in Doppler `prd`.

## Testing

- Extend `tests/test_client_portal_routes.py`: with `CLIENT_LOGIN_ENABLED` on, GET
  `/portal/login` returns 200 and the HTML contains the join wiring
  (`/api/healing-oasis/status`, `/api/healing-oasis/request`, `joinGo`).
- Existing `tests/test_healing_oasis.py` already covers the request/status backend.

## Out of scope

- No backend/endpoint changes.
- No email-copy changes.
- No DNS / domain-migration work (tracked separately in the portal-domain-migration
  spec). This change is inert on both hosts until the flags are on.
