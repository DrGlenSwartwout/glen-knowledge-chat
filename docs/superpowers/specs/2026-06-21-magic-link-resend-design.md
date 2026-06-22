# Request a Fresh Magic Link (invalid-link recovery)

**Date:** 2026-06-21
**Status:** Approved (design); ready for implementation plan
**Repo:** deploy-chat (Flask, illtowell.com).
**Trigger:** Glen hit the Biofield reveal invalid-link page ("This link is no longer valid. Please request a fresh one.") and there was no way to request one - a dead end. Every magic-link page has the same gap.

---

## Problem

When a magic link is expired, consumed, or stale, the page tells the user to "request a fresh one" but gives no way to do it. The user is stranded. This affects every `auth_tokens`-based magic link: the Biofield reveal, the reorder pay-link, the sign-in / portal links, the affiliate login, and the practitioner claim/optout/share links.

## Goal

Add a "Request a fresh link" button to each invalid magic-link surface. It uses the **expired token already in the URL** (which carries the user's email and purpose) to mint a fresh token of the same purpose and email it - no typing, no email-enumeration. One backend endpoint + a small per-purpose registry + the button on each surface.

## Scope

- One endpoint `POST /link/resend {token}` that resolves the token's `auth_tokens` row (even expired/consumed) and dispatches by `purpose`.
- A `RESEND_HANDLERS` registry: most purposes use a generic mint-and-send; the Biofield reveal is custom (must still exist; bespoke email).
- The button wired onto each invalid-link surface (some static `.html`, some inline-HTML routes).

**Out of scope:** **inquiry-reply** - its link is keyed by `inquiry_id`/`practitioner_id` in the path, NOT an `auth_tokens` token, so it cannot use this mechanism (a separate, rare, practitioner-facing case; note as a possible follow-up). `membership_cancel` is included as a purpose but has no standalone invalid-link page to wire (its link is one-click; covered by the endpoint if ever hit). No change to how links are originally minted/sent.

---

## Confirmed decisions (Glen, 2026-06-21)

- All (token-based) magic-link pages get the button.
- Frictionless: the expired token identifies the user; no email-entry form.
- Always return a generic ok ("if that link was valid, a fresh one is on its way") - no enumeration, no leak of purpose/email/existence.
- Reuse each purpose's existing sender + TTL; do not reinvent email plumbing.
- No emoji, no em dashes.

---

## Architecture

### 1. Endpoint - `POST /link/resend` (app.py)
- Body `{token}` (the page's expired token). Hash it; `SELECT email, purpose FROM auth_tokens WHERE token_hash=?` (no expiry/consumed filter - we WANT expired rows).
- If found and `purpose` is in `RESEND_HANDLERS`: call the handler with `email`; it mints a fresh token (same purpose, that purpose's TTL) and sends the matching email. Wrap in try/except (best-effort).
- Always respond `200 {"ok": true}` with the generic message - found-or-not, sent-or-not. (A light per-token throttle: skip a resend if one was issued for this token in the last few minutes; the email only ever reaches the token's own owner, so abuse risk is low.)

### 2. `RESEND_HANDLERS` registry (app.py)
A dict `purpose -> handler(email) -> None`. A generic factory builds most:

```
_generic_resend(url_template, ttl_min) -> handler:
    mint fresh token (purpose, now, now+ttl) -> auth_tokens
    url = PUBLIC_BASE_URL + url_template.format(token=fresh)
    send_magic_link_email(email, "", url)
```

| purpose | url template | TTL | handler |
| --- | --- | --- | --- |
| `biofield_reveal` | `/begin/biofield/{token}` | 30 d | **custom**: only if a reveal exists for `email` (else ok, no send); mint 30 d token; send the bespoke "Your Biofield Analysis is ready" email (the `/api/e4l/reveal-draft` notify copy). |
| `reorder` | `/reorder/auth/{token}` | `AUTH_TOKEN_TTL_MIN` | generic |
| `magic_link` | `/auth/magic-link/verify?token={token}` | `AUTH_TOKEN_TTL_MIN` | generic |
| `portal` | `/portal/login-verify?token={token}` | `AUTH_TOKEN_TTL_MIN` | generic |
| `affiliate_magic_link` | `/affiliate/login-verify?token={token}` | `AUTH_TOKEN_TTL_MIN` | generic |
| `practitioner_claim` | `/practitioner-claim/{token}` | (existing practitioner TTL) | generic |
| `practitioner_optout` | `/practitioner-optout/{token}` | (existing) | generic |
| `practitioner_share` | `/share-with-practitioner/{token}` | (existing) | generic |

(The plan pins each exact TTL by reading its current mint site. Unknown/absent purpose -> not in the registry -> generic ok, nothing sent.)

### 3. Front-end - the button on each invalid surface
A tiny shared behavior: a "Request a fresh link" button that POSTs `{token}` (read from the page URL) to `/link/resend`, disables itself, and shows "Check your email for a fresh link." Wired onto:
- `static/begin-biofield.html` - State 1 (the `reveal === null` block): append the button (the token is already parsed as `_token`).
- `static/coaching.html` - the sign-in error block.
- `static/practitioner-claim.html`, `static/practitioner-optout.html`, `static/practitioner-share.html` - the invalid-link block.
- The inline-HTML invalid responses in the **reorder** route (~app.py:9562) and the **sign-in** route (~app.py:10976): add the same button markup + a small inline script (or point them at a shared snippet).

All XSS-safe (textContent / static markup); the token comes from the URL path/query the page already has.

### Reuse / untouched
- `send_magic_link_email`, the reveal notify email, `_hash_token`, `auth_tokens`, `PUBLIC_BASE_URL`, `AUTH_TOKEN_TTL_MIN` - reused.
- Original mint/send flows, verify routes, inquiry-reply - untouched.

---

## Data flow
1. User opens an expired magic link; the page shows the invalid state + a "Request a fresh link" button.
2. Button POSTs the expired token to `/link/resend`.
3. Server resolves the token -> email + purpose -> handler mints a fresh token + emails the right link (reveal/reorder/sign-in/etc.).
4. Server returns the generic ok; the page shows "Check your email."
5. The user clicks the fresh link and proceeds.

## Error handling
- Token bogus / not found / unknown purpose -> generic ok, nothing sent (no enumeration).
- Underlying object gone (reveal deleted) -> generic ok, nothing sent.
- Sender throws -> caught, still ok (best-effort), logged.
- Repeated clicks -> the light per-token throttle suppresses duplicate sends; the endpoint still returns ok.

## Testing
`tests/test_link_resend.py`:
- A valid expired `reorder` token -> a fresh `reorder` `auth_tokens` row is minted and `send_magic_link_email` is called with the `/reorder/auth/...` URL; response is `{ok:true}` generic.
- `biofield_reveal` token with an existing reveal -> fresh 30 d token + the reveal email sent; with the reveal deleted -> ok, no send.
- `magic_link` / `affiliate_magic_link` / `practitioner_claim` -> the generic handler mints + sends the right URL.
- Bogus token -> ok, no mint, no send. Unknown purpose -> ok, no send.
- Throttle: a second resend for the same token within the window does not send again.
- Front-end serve: each invalid surface ships the "Request a fresh link" button + the `/link/resend` POST wiring. (Senders + `auth_tokens` mocked/tmp `LOG_DB`; no emoji; no em dashes.)

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_link_resend.py -v`

## Notes
- Ships live on merge (no flag); purely additive (a new endpoint + buttons). The generic-ok contract means a misconfigured purpose degrades to "nothing sent," never an error to the user.
- inquiry-reply is the one listed page that cannot use this (non-token); flag to Glen as out-of-scope here, a separate follow-up if wanted.
