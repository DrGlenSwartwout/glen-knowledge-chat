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

- One endpoint `POST /link/resend` accepting EITHER `{token}` (an `auth_tokens` magic link) OR `{inquiry_id, practitioner_id}` (the inquiry-reply link, whose token lives in `inquiry_reply_tokens`).
- A `RESEND_HANDLERS` registry for the `auth_tokens` purposes: most generic mint-and-send; the Biofield reveal is custom (must still exist; bespoke email).
- A dedicated inquiry-reply branch (mints a fresh `inquiry_reply_tokens` row + emails the practitioner).
- The button wired onto each invalid-link surface (some static `.html`, some inline-HTML routes).

**Out of scope:** `membership_cancel` is registered as a purpose but has no standalone invalid-link page to wire (its link is one-click; covered by the endpoint if ever hit). No change to how links are originally minted/sent.

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
- **Two input shapes**, branched by body:
  - `{inquiry_id, practitioner_id}` present -> the **inquiry-reply branch** (section 2b).
  - else `{token}` -> the **auth_tokens branch**: hash it; `SELECT email, purpose FROM auth_tokens WHERE token_hash=?` (no expiry/consumed filter - we WANT expired rows). If found and `purpose` is in `RESEND_HANDLERS`, call the handler with `email` (mints a fresh same-purpose token at that purpose's TTL + sends the matching email). Wrap in try/except (best-effort).
- Always respond `200 {"ok": true}` with the generic message - found-or-not, sent-or-not. (A light per-key throttle: skip a resend if one was issued for this token / inquiry+practitioner in the last few minutes; the email only ever reaches the owner, so abuse risk is low.)

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

### 2b. Inquiry-reply branch (app.py)
The reply link is `/inquiries/{inquiry_id}/{practitioner_id}/reply?token=<t>`; its token lives in `inquiry_reply_tokens (token_hash, inquiry_id, practitioner_id, created_at, expires_at)` (30 d, `UNIQUE(inquiry_id, practitioner_id)`, INSERT OR REPLACE). The fresh link must reach the **practitioner**, whose email is `inquiry_practitioners.practitioner_email` for that `(inquiry_id, practitioner_id)`. Handler `_resend_inquiry_reply(inquiry_id, practitioner_id)`:
- Resolve `practitioner_email` from `inquiry_practitioners`; if none -> generic ok, nothing sent. (Optionally confirm the inquiry still exists.)
- Mint a fresh reply token: `INSERT OR REPLACE INTO inquiry_reply_tokens (...)` with a new 30 d `expires_at` (mirrors the original fan-out mint).
- Build `reply_url = {PUBLIC_BASE_URL}/inquiries/{inquiry_id}/{practitioner_id}/reply?token={fresh}` and email the practitioner the fresh secure-reply link (reuse the existing reply-link email copy via the inquiry sender).
- Best-effort; generic ok.

### 3. Front-end - the button on each invalid surface
A tiny shared behavior: a "Request a fresh link" button that POSTs `{token}` (read from the page URL) to `/link/resend`, disables itself, and shows "Check your email for a fresh link." Wired onto:
- `static/begin-biofield.html` - State 1 (the `reveal === null` block): append the button (the token is already parsed as `_token`).
- `static/coaching.html` - the sign-in error block.
- `static/practitioner-claim.html`, `static/practitioner-optout.html`, `static/practitioner-share.html` - the invalid-link block.
- The inline-HTML invalid responses in the **reorder** route (~app.py:9562) and the **sign-in** route (~app.py:10976): add the same button markup + a small inline script (or point them at a shared snippet).
- `static/inquiry-reply.html` - the `status="error"` block: the button reads `inquiry_id` + `practitioner_id` from the URL path (`/inquiries/<id>/<pid>/reply`) and POSTs `{inquiry_id, practitioner_id}` to `/link/resend` (no `auth_tokens` token involved).

All XSS-safe (textContent / static markup); the token (or inquiry/practitioner ids) come from the URL path/query the page already has.

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
- **Inquiry-reply:** `{inquiry_id, practitioner_id}` with a known practitioner -> a fresh `inquiry_reply_tokens` row is minted and the practitioner is emailed the `/inquiries/.../reply?token=...` URL; an unknown practitioner -> ok, no send.
- Bogus token -> ok, no mint, no send. Unknown purpose -> ok, no send.
- Throttle: a second resend for the same token (or inquiry+practitioner) within the window does not send again.
- Front-end serve: each invalid surface ships the "Request a fresh link" button + the `/link/resend` POST wiring. (Senders + `auth_tokens` mocked/tmp `LOG_DB`; no emoji; no em dashes.)

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_link_resend.py -v`

## Notes
- Ships live on merge (no flag); purely additive (a new endpoint + buttons). The generic-ok contract means a misconfigured purpose degrades to "nothing sent," never an error to the user.
- All listed pages are covered: the `auth_tokens` magic links via the registry, and inquiry-reply via its own branch (separate token table + practitioner email target).
