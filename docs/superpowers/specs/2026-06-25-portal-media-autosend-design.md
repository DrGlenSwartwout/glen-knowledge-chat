# Portal-Hosted Audio + PDF, Auto-Email on Publish

**Date:** 2026-06-25
**Status:** Approved (design)
**Author:** Glen + Claude
**Parent:** extends the Biofield Intake → Client Portal Publish connector (PR #320, `dashboard/biofield_portal_publish.py`).

## Problem

The publish connector puts a client's authored Biofield Analysis (layers/remedies/narrative) on their portal, but the **audio walkthrough** and the **PDF report** still have to be delivered by hand (manual email attachments). Glen wants: the audio **playable inline** on the portal page, the PDF **downloadable** there, and the publish to **auto-email** the client the link — so "they get those files" via their portal.

## Goal

Make "Publish to portal" (1) upload the client's audio mp3 + report PDF to prod, (2) render them on the portal page (inline `<audio>` player + PDF download button), and (3) auto-send the existing "your personal healing home is ready" link email. Unlike PR #320, this **touches prod** (a file-hosting route, the portal page, the `/api/portal` payload) and deploys to Render.

## Non-goals

- Attaching files to the email (decided: link-only; files live on the portal).
- Token-gating asset downloads (v1 uses opaque/random filenames, same trust model as the existing `/clips` route; can add gating later).
- Re-send on re-publish (the upsert only emails when a NEW token is minted — i.e. first publish; re-publishing an existing client updates content silently, which is the desired behavior).
- Video (the existing `video` slot is untouched).

## Design

### 1. Prod file hosting (`app.py`) — mirror the existing `/clips` routes

- `_PORTAL_ASSETS_DIR = Path(os.environ.get("DATA_DIR", str(Path(__file__).parent))) / "portal-assets"` (created at import, like `_CLIPS_DIR` at app.py:16043). On Render this is the persistent disk.
- `PUT /portal-asset/upload?filename=<name>` — **console-gated** (`X-Console-Key` == `CONSOLE_SECRET`, 401 otherwise; mirrors the existing portal/console gates). `filename` must match `^[\w\-]+\.(mp3|pdf)$` (400 otherwise). Writes `request.data` (raw bytes) to `_PORTAL_ASSETS_DIR/<filename>`. Returns `{"ok": True, "url": f"{base}/portal-asset/{filename}"}` where `base = os.environ.get("RENDER_EXTERNAL_URL", "https://glen-knowledge-chat.onrender.com")`.
- `GET /portal-asset/<filename>` — validates the same regex (400 otherwise), serves from `_PORTAL_ASSETS_DIR` with mimetype `audio/mpeg` for `.mp3`, `application/pdf` for `.pdf`. Public (unguessable filename is the protection, as with `/clips`).

### 2. Prod portal payload (`/api/portal/<token>` in `app.py`)

In the return dict (next to `"video": bf_content.get("video") or {}` at ~app.py:10891), add — **gated on `bf_confirmed`** because the audio/PDF name remedies:
- `"audio": (bf_content.get("audio") or {}) if bf_confirmed else {}`
- `"report_pdf": (bf_content.get("report_pdf") or {}) if bf_confirmed else {}`

### 3. Prod portal render (`static/client-portal.html`)

Near the existing video-card render (~line 248), add two cards, each only when its field is present:
- **Audio:** when `d.audio && d.audio.url` → a card "Your audio walkthrough" containing `<audio controls preload="none" src="${esc(d.audio.url)}"></audio>` (+ the `d.audio.label` as caption).
- **PDF:** when `d.report_pdf && d.report_pdf.url` → a card "Your written report" with `<a class="btn" href="${esc(d.report_pdf.url)}" target="_blank" rel="noopener" download>Download your report (PDF)</a>`.

### 4. Local connector (`dashboard/biofield_portal_publish.py` + `biofield_local_app.py`)

- `upload_asset(data_bytes, filename, *, base_url, console_key, http_put=None) -> str` — `PUT {base_url}/portal-asset/upload?filename=<filename>` with header `X-Console-Key`, body = raw bytes, timeout 60; returns the `url` from the JSON; raises `RuntimeError` on non-2xx. `http_put` injectable (defaults `requests.put`).
- `_asset_name(test_id, ext) -> str` — opaque filename, e.g. `f"biofield-{secrets.token_hex(8)}.{ext}"` (no PHI in the name).
- `publish_to_portal` gains a `send=False` parameter (replaces the hard-coded `send:False`); the route passes `send=True`.
- `build_portal_content` gains optional kwargs `audio_url=None`, `report_pdf_url=None`; when set, adds to content: `audio = {"url": audio_url, "label": "Listen to your walkthrough"}` and `report_pdf = {"url": report_pdf_url}`. (Omitted/empty → keys absent or empty, page renders neither.)
- Route `POST /test/<id>/publish-portal` flow becomes:
  1. Resolve/build as today; if `unresolved`, 409 (unchanged).
  2. Ensure the PDF exists: reuse the existing report-pdf path (`render_present` + `save_report_pdf` to `~/biofield-reports/report_<id>_<date>.pdf`); read its bytes. Read the audio from `AUDIO_DIR/test_<id>.mp3` (skip audio if the file is absent — publish still proceeds).
  3. `upload_asset` the PDF (always) and the audio (if present) → URLs.
  4. `build_portal_content(..., audio_url=…, report_pdf_url=…)`.
  5. `publish_to_portal(payload, base_url=…, console_key=…, send=True)`.
  6. Return `{"ok": True, "url": …, "updated": …, "note": …, "emailed": bool(res.get("emailed")), "unresolved": []}`.

## Error handling

- Asset upload failure → route returns 502 with the error (no partial portal publish: upload before the upsert).
- Missing audio file → log/skip audio, still publish PDF + content (don't block the report on a missing mp3).
- Prod route: 401 (bad key), 400 (bad filename/ext), as above.

## Testing

**Offline (tmp sqlite, injected HTTP) — the connector:**
1. `upload_asset` — injected `http_put` captures the URL (`…/portal-asset/upload?filename=…`), the `X-Console-Key` header, and the raw body; returns the JSON `url`; non-2xx raises.
2. `_asset_name` — matches `^biofield-[0-9a-f]{16}\.(mp3|pdf)$`; two calls differ.
3. `build_portal_content` with `audio_url`/`report_pdf_url` → `content["audio"] == {"url":…, "label":"Listen to your walkthrough"}` and `content["report_pdf"] == {"url":…}`; without them → keys absent.
4. `publish_to_portal(send=True)` → injected post body has `send: True`; default still `False`.
5. Route — monkeypatch `upload_asset` (returns fake URLs) + `publish_to_portal` (captures payload); seed a test with a remedy + write a dummy mp3 into `AUDIO_DIR`; assert 200, the published `content` carries both asset URLs, and `publish_to_portal` was called with `send=True`. Plus a missing-audio variant: no mp3 → still 200, `report_pdf` present, `audio` absent.

**Live post-deploy (after merge; `app.py` can't import offline):**
6. `curl -X PUT …/portal-asset/upload?filename=test-<rand>.pdf` with the console key → 200 + url; GET it back → 200 `application/pdf`; no-key → 401; bad ext → 400.
7. **Render-verify the portal page in a headless browser** (per the project's hard rule): publish a real test, load `/portal/<token>`, assert the `<audio>` element and the PDF download link are present and the audio `src` resolves (network 200), with **zero console errors**.

## Rollout

Prod pieces (routes, portal render, api) ship on merge → Render deploy. Then run the connector's publish for Karin (test `a3`) and do the live render-verify. The Gmail draft becomes unnecessary (the app auto-sends); delete it.
