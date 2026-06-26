# Portal PDF — One-Click Download

**Date:** 2026-06-26
**Status:** Approved (design)
**Author:** Glen + Claude

## Problem

The portal's "Download your report (PDF)" link opens the PDF in a new tab instead of saving it, because the asset is served cross-origin (portal on illtowell.com, asset on `*.onrender.com`) and browsers ignore the `<a download>` attribute cross-origin.

## Goal

Clicking the link downloads the PDF in one click, from any origin. The audio walkthrough must keep streaming inline in the `<audio>` player (no change).

## Design

Browsers always honor a `Content-Disposition: attachment` **response header** (unlike the `download` attribute, which is ignored cross-origin). So serve the PDF as an attachment.

In `app.py` `portal_asset_serve` (`/portal-asset/<filename>`, ~line 16092), make the response depend on the extension (already captured as `m.group(1)`):
- **`.pdf`** → `send_from_directory(..., mimetype="application/pdf", as_attachment=True, download_name="Biofield-Analysis.pdf")` (Flask 3 sets `Content-Disposition: attachment; filename="Biofield-Analysis.pdf"`).
- **`.mp3`** → unchanged: `send_from_directory(..., mimetype="audio/mpeg")` (inline, so the `<audio>` player streams it). **Must NOT be `as_attachment`** — that would stop inline playback.

Implementation: `is_pdf = m.group(1) == "pdf"`; pass `as_attachment=is_pdf` and, only when pdf, `download_name`.

## Non-goals

- Same-origin asset migration (unnecessary — the header fixes it).
- Per-client download filenames (a generic "Biofield-Analysis.pdf" is fine inside the client's own portal).
- Any change to the portal page, the connector, or the asset upload route / URLs.

## Error handling

The existing `400` on a bad filename regex is unchanged. No new failure modes.

## Testing

`app.py` cannot be imported offline (Pinecone at import), so this is verified LIVE post-deploy:
1. `GET` an existing `/portal-asset/<…>.pdf` → response header `Content-Disposition: attachment` (and `Content-Type: application/pdf`).
2. `GET` an existing `/portal-asset/<…>.mp3` → NO `Content-Disposition: attachment`; still `Content-Type: audio/mpeg` (player still works).
3. On Karin's portal, click "Download your report (PDF)" → it downloads in one click (no new tab); the audio still plays inline.

## Rollout

Ships on merge → Render deploy. Re-check Karin's portal: PDF one-click downloads, audio still plays.
