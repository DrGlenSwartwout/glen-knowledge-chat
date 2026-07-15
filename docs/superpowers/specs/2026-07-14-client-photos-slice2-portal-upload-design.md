# Client Photos — Slice 2: Portal Self-Upload

**Date:** 2026-07-14
**Depends on:** Slice 1 (store + `/client-photo` serve). Same store, keyed by email.

## Goal

Let a client upload their own photo from their portal. It shows in their portal and
flows to the same `client_photos` store, so it appears on the intake page and console
reveals like any other source.

## Components

- **`POST /api/portal/<token>/photo`** — resolve the client's email from the portal
  token via `portal_identity.identity_from_token` (the pattern the other
  `/api/portal/<token>/*` writes use). Validate the upload (allowed types
  jpg/png/webp; size cap, e.g. 5 MB), then `client_photos.put(email, blob, ctype,
  source="portal-self")`. Prod-only (portals are prod).
- **`GET /api/portal/<token>/photo`** — serve the photo for the email resolved from the
  token. **Token-scoped**: a client can only ever fetch their own (never another
  client's by guessing an email). 404 when none.
- **`static/client-portal.html`** — a photo control in the profile/greeting header:
  show the current photo (via the token-scoped GET) with a "Change photo" / "Add photo"
  file picker that POSTs and refreshes on success.

## Data flow

Client picks a file in their portal → `POST /api/portal/<token>/photo` (email from
token) → `client_photos` (source `portal-self`) → visible in their portal, and to
console/reveals/intake via the existing serve routes.

## Source precedence (shared across Slices 2–4)

**`portal-self` > `fmp` > `ghl`.** A client's own upload is the most authoritative — it
overwrites an FMP or GHL photo. To enforce this, `client_photos.put` gains an optional
precedence check: a lower-rank source does not overwrite a higher-rank existing photo
(a `force` flag bypasses it). `portal-self` always writes.

## Decisions to confirm

1. **Moderation:** a client-uploaded image is shown to Glen in console/reveals. Trust
   clients (validate type/size only, no human review) — acceptable? (Recommended: yes;
   it's their own health portal.)
2. **Size/type cap:** jpg/png/webp, ≤ 5 MB (recommended).
3. **Replace vs keep:** a new upload replaces the client's previous photo (one photo per
   client). Recommended.

## Testing

- Token → email resolution drives the write/serve.
- Upload stores with `source="portal-self"` and overwrites an existing `fmp` photo.
- Serve is token-scoped: token A cannot fetch client B's photo.
- Oversized / non-image upload rejected with a clear message.

## Out of scope

Cropping/rotation, multiple photos, avatars/initials fallback.
