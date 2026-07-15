# Client Photos — Slice 3: FMP Folder Bulk Sync

**Date:** 2026-07-14
**Depends on:** Slice 1 (store + prod push endpoint). Runs locally, pushes to prod.

## Goal

Backfill FMP-hosted client photos in bulk. FileMaker is desktop (no live API) and CSV
export drops container binaries, so Glen exports the container photos to a folder and a
local script loads them into the store for all clients at once — the bulk counterpart to
Slice 1's one-at-a-time intake upload.

## FileMaker side (Glen, one-time setup)

A FileMaker step that exports each client's photo container to an image file named by the
client's **`id_pk`** (e.g. `21459.jpg`) into a folder (e.g. `~/fmp-photos/`). Options in
FileMaker Pro: a script looping records with `Export Field Contents`, or a container batch
export. **The exact FileMaker mechanism is Glen's to set up; this spec assumes a folder of
`<id_pk>.<ext>` image files.**

## Our side

- **`scripts/sync_client_photos.py`** — for each `<id>.<ext>` in the folder:
  1. Resolve `id_pk → email` via `fmp_clients` (skip + log if no email on file).
  2. Read bytes + infer content-type from extension.
  3. Respect precedence: skip if the stored photo's source is `portal-self` (a client's
     own choice wins); otherwise `client_photos.put(email, blob, ctype, source="fmp")`
     locally **and** push to prod (`POST /api/console/client-photo`).
  4. Log per file: pushed / skipped-no-email / skipped-precedence / error.
- Idempotent; re-runnable. Optional content-hash skip to avoid re-pushing unchanged files
  when the folder is large (defer unless it proves slow).
- Run on-demand: `bash ~/deploy-chat/sync_client_photos.sh <folder>` (doppler prd for
  CONSOLE_SECRET + local DB via `DATA_DIR` override — same pattern as
  `fulfill_requests.sh`).

## Decisions to confirm

1. **Filename convention:** `id_pk` (recommended — stable, always present) vs email. Which
   can your FileMaker export produce?
2. **Feasibility:** can FileMaker Pro bulk-export the containers to a folder (script /
   Export Field Contents loop)? If not, this slice needs a different extraction path.
3. **Precedence:** FMP does not overwrite a client's own `portal-self` photo. Confirm.

## Testing

- `id_pk → email` resolution; files with no resolvable email are skipped + logged.
- `portal-self` photo is NOT overwritten; an older `fmp`/`ghl` photo IS.
- Mixed folder (valid images, junk, missing-email ids) handled without aborting the run.
- Prod push round-trip for one file.

## Out of scope

Auto-watching the folder, the FileMaker-side export scripting, per-photo cropping.
