# Client Photos — Slice 4: GHL Contact-Photo Pull

**Date:** 2026-07-14
**Depends on:** Slice 1 (store + prod push). **Gated by a feasibility spike (below).**

## Goal

Fill in photos for clients who have one in GHL (GoHighLevel) but not from FMP or their
own upload — so more console/reveal records show a face.

## ⚠️ Feasibility spike FIRST (gates the whole slice)

The app talks to GHL via the **v1 REST API** (`https://rest.gohighlevel.com/v1`, Bearer
`GHL_API_KEY`); the current sync only reads email/name/phone/tags. **It is unknown whether
the GHL API exposes a contact profile photo / avatar URL at all** (v1 may not; v2 /
LeadConnector might). Before building:

1. Fetch a few known contacts via the API and inspect the JSON for any photo/avatar/image
   URL field.
2. If none in v1, check the v2 / LeadConnector contact endpoint.
3. If no API-accessible photo exists, **stop** — GHL is not a viable source; record the
   finding and close the slice.

The rest of this spec assumes the spike finds a usable photo URL.

## Components (post-spike)

- **`scripts/pull_ghl_photos.py`** — for each GHL contact that has a photo URL **and** an
  email matching a known client:
  1. GET the image bytes (+ content-type).
  2. Respect precedence: **`ghl` is lowest** — write only if the client has **no** photo
     yet (never overwrite `fmp` or `portal-self`).
  3. `client_photos.put(email, blob, ctype, source="ghl")` → push to prod.
  4. Log per contact: pulled / skipped-has-photo / skipped-no-match / no-photo.
- May run prod-side (GHL creds + store both on prod) or as a local script pushing to prod;
  choose after the spike based on where GHL creds live.

## Decisions to confirm

1. **Spike outcome** decides go/no-go (does GHL expose photos via API?).
2. **Precedence:** GHL fills gaps only — lowest trust (a GHL avatar may be a logo/initial).
   Confirm.
3. **Match key:** email (skip contacts whose email isn't a known client). Confirm.

## Testing

- Spike documented (photo field present? which API version?).
- For a contact with a photo + no existing store photo → pulled and stored `source="ghl"`.
- A client who already has an `fmp`/`portal-self` photo is NOT overwritten.
- Contact with no email match is skipped.

## Out of scope

Real-time GHL webhook photo sync, de-duping generic/logo avatars, non-contact GHL objects.
