# Body Map Multi-Photo + Alignment — Slice 1 (Foundation) — Design

**Date:** 2026-07-25
**Status:** Approved in brainstorm with Glen 2026-07-25
**Repo:** deploy-chat
**Branch:** `sess/72339d9d-bodymap` (off `main`)

## Problem

Two limits in the personalized Body Map today, both confirmed in code:

1. **One photo per client.** `dashboard/client_photos.py` is keyed `email TEXT PRIMARY KEY` — a client can store exactly one photo. The portal auto-warps it onto the **face** map only; every other system (iris, sclera, hand, foot, ear, meridian…) shows the reference figure. A client can locally upload a photo for another system, but it is a browser object URL — never persisted, gone on reload.
2. **The alignment is never saved.** `state.transform` in `static/body-map.js` is computed on every page load (MediaPipe auto-detect, or manual anchor taps) and reset to `null` on mode switch (`body-map.js:505,510`). Nothing persists it, so the map re-detects every single time.

Glen wants **multiple photos (different parts/views), each with a saved alignment**, ultimately editable by the **client** (self-service, approximate) and curated by **Glen** (console, precise).

## Decomposition — this is Slice 1 of 3

The full feature is too large for one spec. Decided with Glen:

- **Slice 1 (this spec) — Foundation.** A photo store holding many photos per client keyed by slot, each carrying a saved transform; the HTTP endpoints; and wiring so the *existing* alignment persists. Audience-agnostic plumbing both surfaces need. Visible win: **face alignment stops re-detecting on every load.**
- **Slice 2 — Client surface.** Portal UI to upload a photo per slot and nudge/rotate/scale to adjust, then save. (Later spec.)
- **Slice 3 — Console curation.** Glen reviews and fine-tunes the canonical alignment per client per slot. (Later spec.)

Slices 2 and 3 both consume Slice 1's store and endpoints. This spec deliberately builds **no adjust UI and no console screen** — only the persistence layer and the auto-persist of the transform the code already computes.

## The slot key

A photo slot is **`(email, system, side)`** (confirmed with Glen: per system **+ side**).

- `system` — a Body Map system id: `face`, `iris`, `sclera`, `hand`, `foot`, `ear`, `meridian`, `neurotome`, `lymph`, `eav`, … (the existing `bodymap_store.system_catalog()` ids).
- `side` — `'left'`, `'right'`, `'foot'`, or `''` (none). Distinguishes left-eye vs right-eye iris, left vs right foot. Most systems use `''`.

The slot is the primary key: a new upload for the same slot **replaces** the old photo and clears its transform (a new photo needs re-aligning).

## Boundary vs. `client_photos` — do NOT repurpose it

`client_photos` is read by `biofield_local_app.py`, `app.py` (biofield/reveal/console), `dashboard/portal_onboarding.py`, and `scripts/sync_client_photos.py`. It is the client's single **identity portrait**. Changing its key would break all of those. This spec adds a **separate** table and leaves `client_photos` completely untouched.

**The one bridge:** the `face` slot's GET **falls back to `client_photos.get(email)`** when no `body_map_photos` face row exists. This preserves today's face-warp for every existing client with zero migration — their identity portrait keeps driving the face map until/unless they upload a dedicated face slot photo.

## Data model — new table `body_map_photos`

```
body_map_photos
  email          TEXT     -- lowercased
  system         TEXT     -- body-map system id
  side           TEXT     -- 'left' | 'right' | 'foot' | ''
  image_blob     BYTEA    -- the photo (BYTEA, never BLOB — runtime pgcompat does
                          --   not translate BLOB; it fails outright on Postgres)
  content_type   TEXT
  transform_json TEXT     -- {"mx":.., "my":.., "tx":.., "ty":..} or NULL (unaligned)
  source         TEXT     -- 'portal-self' | 'console'
  updated_at     TEXT
  PRIMARY KEY (email, system, side)
```

Runtime-created via `init_table(cx)`, following the `client_documents` precedent (which established the `BYTEA` requirement the hard way).

### Why `{mx, my, tx, ty}` is the right transform representation

The map operates in a **fixed 600×600 viewBox** (`body-map.html`: `viewBox="0 0 600 600" preserveAspectRatio="xMidYMid meet"`; `body-map.js:4`: `VIEW=600`). Screen pixels scale via the SVG viewBox, so a transform expressed in viewBox coordinates is **resolution-independent** — a saved alignment renders identically on any screen size. No device/pixel data is stored.

Both alignment code paths reduce to the same 4-parameter similarity (translation + rotation + uniform scale):
- `fitSimilarity` (`body-map.js:528`) returns `(n) => ({x: mx*n.x - my*n.y + tx, y: my*n.x + mx*n.y + ty})`.
- `computeSimilarity` (the 3-anchor iris fallback) returns `(n) => ({x: P.x + s*(n.x*cos - n.y*sin), y: P.y + s*(n.x*sin + n.y*cos)})`, which is the **same** form with `mx=s*cos, my=s*sin, tx=P.x, ty=P.y`.

So the persistable transform is exactly `{mx, my, tx, ty}`, and reconstruction is a single closure:
```js
const T = state.savedTransform;  // {mx,my,tx,ty}
state.transform = (n) => ({ x: T.mx*n.x - T.my*n.y + T.tx, y: T.my*n.x + T.mx*n.y + T.ty });
```

## Storage module — `dashboard/body_map_photos.py`

Persistence only. No HTTP, no rendering. Mirrors `client_photos.py` / `client_documents.py` house style (tuple indexing, dicts built explicitly, ids/keys read back explicitly, `BYTEA`).

```
init_table(cx)
put(cx, email, system, side, blob, content_type, source) -> bool
    # upsert the slot's photo; CLEARS transform_json (new photo => needs re-align)
get(cx, email, system, side) -> {blob, content_type, transform, source} | None
set_transform(cx, email, system, side, transform) -> bool
    # transform is {mx,my,tx,ty} (validated to 4 finite numbers) or None to clear
get_transform(cx, email, system, side) -> {mx,my,tx,ty} | None
list_for_email(cx, email) -> [{system, side, has_transform, updated_at}, ...]  # no blobs
```

`side` normalizes `None`→`''`. `transform` is stored as JSON; `set_transform` rejects anything that is not four finite numbers (a malformed transform must never persist — it would render the photo somewhere absurd).

## Endpoints (app.py)

All photo bytes are served through the existing `_doc_response_content_type` / `_doc_safe_filename` / `_DOC_INLINE_TYPES` helpers (added by #1172) plus `X-Content-Type-Options: nosniff` and `Cache-Control: private, no-store` — a client-influenced image must never render inline as `text/html`/`svg`.

**Client, token-scoped** (resolve token → owner email via `_portal_record_for`, write only that email):
- `POST /api/portal/<token>/bodymap-photo?system=&side=` — upload a slot photo. `source='portal-self'`. Validates type (image/\* + pdf-not-relevant here → images only) and a size cap (5 MB, matching the existing `/photo` route). Missing/invalid `system` → 400.
- `GET /api/portal/<token>/bodymap-photo?system=&side=` — serve the slot photo. **`system=face` with no slot row falls back to `client_photos.get(email)`.** 404 (bare, indistinguishable) when neither exists.
- `PUT /api/portal/<token>/bodymap-transform?system=&side=` — body `{mx,my,tx,ty}`; saves it (validated). `GET` returns it or 404.

**Console, gated** (the established console-key/owner-token check): twins `POST/GET /api/console/bodymap-photo` and `PUT/GET /api/console/bodymap-transform`, taking an explicit `email`. `source='console'`. The endpoints land in this slice; Slice 3's UI consumes them. No console *screen* is built here.

## Body Map wiring (`static/body-map.js`) — the visible win

Extend the bodymap data payload and `bootstrapPortal`/`loadPortalPhoto` (`body-map.js:382,404`):

1. The portal bodymap-data payload (`_portal_bodymap_data`, `app.py:21279`) already returns `has_photo` for the face. Extend it to report, for the **current system+side**, whether a slot photo exists and whether it has a **saved transform**.
2. When a system loads with a slot photo:
   - **If the slot has a saved transform**, reconstruct `state.transform` from `{mx,my,tx,ty}` and render — **skip `beginAnchoring()`/`autoDetect()` entirely**. This is the "stop re-detecting every load" win.
   - **Else** run today's path (`beginAnchoring` → `autoDetect` for face/hand/pose; manual tap otherwise).
3. When an alignment is newly established (auto-detect succeeds, or the manual anchors resolve in `placeOverlay`, `body-map.js:542`), **extract `{mx,my,tx,ty}` from the resolved transform and PUT it** to `bodymap-transform` for the current slot. Silent, best-effort (a failed save just means it re-detects next time — no worse than today).

No new visible controls in Slice 1 — the nudge/adjust affordance is Slice 2. This slice makes the existing alignment persist and serves per-slot photos.

## Error handling

- Upload with no file / empty bytes → 400. Over-cap or non-image type → 400 (never a truncated store).
- Unknown/expired token → bare 404 (reveals nothing).
- Missing `system` param → 400; unknown `system` value → 400 (validate against `bodymap_store.system_catalog()` ids so junk slots can't be created).
- Malformed transform body (not 4 finite numbers) → 400; never persisted.
- A save/transform failure in the JS is swallowed (best-effort) — the map falls back to re-detection, exactly as today.
- Serving a slot whose stored `content_type` isn't on the inline allowlist → served as `attachment`/`octet-stream`, never inline.

## Testing

**Store (`body_map_photos`)**
- put → get round-trip through a `BYTEA` column (bytes lossless, pins the BLOB/BYTEA trap).
- the slot key is `(email, system, side)`: same email + system, different side → two rows; re-put same slot → one row, and **transform is cleared** on re-put.
- `set_transform` rejects a non-4-number / non-finite transform; `get_transform` round-trips a valid one; `None` clears it.
- `list_for_email` excludes blobs and reports `has_transform` correctly.

**Endpoints**
- token-scoping: token A cannot upload to or read B's slots.
- **face fallback:** a client with only a `client_photos` portrait and no `body_map_photos` row → `GET bodymap-photo?system=face` serves the portrait; a client with a face slot → serves the slot (slot wins).
- a non-face system with no slot → 404 (no fallback).
- size/type rejection; unknown `system` → 400.
- transform PUT validates; GET returns saved or 404.
- console twins require the console key.
- serving uses the allowlist (a slot stored as `text/html` is served `attachment`, `nosniff` present) — mirror the doc-ingestion serving tests.

**Wiring (JS, node render test where feasible)**
- given a payload with a saved transform, the reconstruct-from-`{mx,my,tx,ty}` closure maps a known point identically to the original `fitSimilarity` closure (pins the persistence round-trip math).

**CI conventions:** pin the product catalog per the `$DATA_DIR`-strips-catalog rule; dummy `OPENAI`/`PINECONE` keys for the app-importing test module.

## Explicitly out of scope (later slices / non-goals)

- **The adjust/nudge/rotate/scale editor** — Slice 2.
- **The console curation screen** — Slice 3 (its endpoints land here, its UI does not).
- **Any change to `client_photos`** — untouched; only read via the face fallback.
- **Auto-detection improvements / new detectors** — this slice persists what the existing MediaPipe/manual paths already produce.
- **Migrating existing single photos into slots** — none needed; the face fallback covers existing clients.
- **Non-similarity transforms** (perspective/warp mesh) — the stored model is a 4-param similarity, matching what the map computes today.
