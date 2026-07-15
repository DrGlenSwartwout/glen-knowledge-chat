# Client Photos — Design Spec

**Date:** 2026-07-14
**Status:** Approved (Slice 1 to implement first)

## Goal

Show each client's photo where we work with their records: **large on the Biofield
Intake page**, and as **thumbnails wherever a client record is opened** (reveals,
console). Photos come from three sources over time — FMP (FileMaker desktop, no live
API), GHL, and client self-upload in the portal — all feeding one shared store.

## Architecture (Approach A: central store, many feeders)

- **Identity key:** lowercased `email`. The one key shared across FMP
  (`fmp_snap_clients.email`), GHL, reveals, portals, and console.
- **Store:** `client_photos(email PK, image_blob BLOB, content_type TEXT,
  source TEXT, updated_at TEXT)` — present in both local (`~/deploy-chat/chat_log.db`)
  and prod DBs. Original bytes only; display size via CSS/`<img>` (no server-side
  thumbnailing until byte size proves it necessary).
- **Serve:** `GET /client-photo/<email>` returns the image, gated. Same image
  everywhere; the surface decides render size.
- **Local↔prod:** the local intake page reads/writes the local store and pushes the
  blob to prod (same transport as the FMP report push in
  `biofield_portal_publish`). Prod serves console/reveal/portal surfaces.

## Slices

1. **Store + intake display + operator upload + console/reveal thumbnails** (this spec).
2. Portal self-upload — client adds their own photo (adds portal-token serve path).
3. FMP folder bulk sync — export FileMaker container fields to a folder; local script
   pushes all, keyed FMP id → email.
4. GHL pull — fetch contact photos for clients who have them in GHL.

## Slice 1 detail

### Components
- **`dashboard/client_photos.py`** — store module: `init_table(cx)`,
  `put(cx, email, blob, content_type, source)`, `get(cx, email) -> {blob, content_type}`,
  `has(cx, email)`. One purpose: persist/fetch a photo by email. No HTTP, no rendering.
- **Prod endpoints (`app.py`):**
  - `POST /api/console/client-photo` — console-key gated. Body: `email`, base64 `image`,
    `content_type`, `source`. Upserts into prod `client_photos`. Returns `{ok, email}`.
  - `GET /client-photo/<email>` — serves the blob with its content_type. Gated:
    console-key / owner session (portal-token path added in Slice 2). 404 when absent.
- **Local intake page (`biofield_local_app.py` + `biofield_report_html.py`):**
  - Photo panel in the intake header (`render_report_html`), shown when a photo exists
    for the test's `client_email`; renders `<img>` at large size.
  - Upload control: file picker → `POST /test/<id>/photo` (local route) → saves to
    **local** `client_photos` (keyed by the test's email) AND pushes to prod via
    `POST /api/console/client-photo`. On success the panel refreshes.
- **Console + reveal thumbnails:** wherever a client record is opened (reveal detail,
  console client views), add `<img src="/client-photo/<email>">` sized small; render
  nothing when no photo exists (no broken image).

### Data flow (Slice 1)
Operator exports Michael's photo from FileMaker (right-click container → Export Field
Contents) → uploads on his intake page → local store (intake shows it immediately) +
pushed to prod store → console/reveal thumbnails read `/client-photo/<email>`.

### Gating / privacy
Photos are personal. `/client-photo/<email>` requires console-key or owner session for
console surfaces. The portal-token-scoped read (a client sees only their own) arrives
with Slice 2. Upload endpoint is console-key gated.

### Error handling
- Missing photo → serve route 404; UI renders no image element (never a broken img).
- Prod push failure on upload → local save still succeeds (intake shows it); surface a
  non-fatal "saved locally, prod push failed" note so it can be retried.
- Oversized/non-image upload → reject with a clear message; cap accepted size.

### Testing
- Store round-trip: `put` then `get` by email returns identical bytes + content_type.
- Serve gating: `GET /client-photo/<email>` without console key → 401/403; with key → 200 + bytes.
- Serve absent: unknown email → 404.
- Visual: photo renders large on the intake page and as a thumbnail on one reveal.

### Explicitly out of scope for Slice 1
Portal self-upload, FMP bulk sync, GHL pull, server-side thumbnail resizing,
multiple photos per client, cropping/editing.
