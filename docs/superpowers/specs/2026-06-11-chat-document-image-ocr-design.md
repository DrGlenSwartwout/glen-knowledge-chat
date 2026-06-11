# Chat Document & Image Upload for OCR — Design

**Date:** 2026-06-11
**Status:** Approved for implementation
**Repo:** deploy-chat

## Goal

Let visitors attach **images and PDF documents** to the chat and have their
content read (OCR + visual understanding) so the assistant can answer about
lab results, supplement labels, E4L scan PDFs, photos, and handwritten notes.

The image path already exists on the `/chat` route. This work (a) adds PDF
("document") support, (b) ports the upload UI from the retired `index.html`
into a shared module used by the live chat pages, and (c) wires attachment
handling into all three live chat routes.

## Current state (as built)

- **`/chat` route** (`app.py`): already accepts an `images` array (base64 /
  data-url), gated behind `images_consented`, runs a single Haiku vision pass
  (`extract_image_content`) to extract text, embeds the extracted text for
  retrieval, joins it to the user question, and **discards the image bytes** —
  only the extracted text (`extracted_image_data`) and an `image_count` are
  written to `query_log`. Caps: 3 images, 5 MB each, png/jpeg/webp/gif.
- **Frontend UI**: a complete attach UI (file picker, drag-drop, thumbnail
  previews, consent checkbox) exists **only in `index.html`**, which is retired
  for chat. The live chat pages — `embed.html`, `concierge.html`,
  `begin-match.html` — have **no upload UI**.
- **`/begin/concierge/chat`** and **`/begin/match/chat`** routes: **no**
  attachment handling at all.
- **PDFs**: not supported anywhere.

## Claude PDF facts (verified 2026-06-11, platform docs)

- PDFs are sent as a `document` content block:
  `{"type":"document","source":{"type":"base64","media_type":"application/pdf","data":"<b64>"}}`
- Processing: each page is converted to an image **and** text-extracted, then
  both are given to the model — true OCR + visual understanding.
- **All current models support PDF**, including Haiku 4.5.
- Base64 document blocks need **no beta header** (only the Files API path does).
- Limits: **32 MB per request**; **100 pages** on 200K-context models (Haiku
  4.5). Standard pricing, ~1.5–3k tokens/page.

## Design

### Privacy model (unchanged, extended to PDFs)

Attachment bytes are sent to Claude for one extraction pass and then
**discarded**. Only the extracted text and a count are persisted to
`query_log`. The consent gate (`images_consented`) applies to documents too.

### Backend (`app.py`)

1. **`_normalize_attachments(images, documents)`** — generalizes the existing
   `_normalize_image_payload`. Returns `(blocks, errors)` where `blocks` is an
   ordered list of Anthropic content blocks:
   - images → existing image-block shape (unchanged behavior).
   - PDFs → `{"type":"document","source":{"type":"base64",
     "media_type":"application/pdf","data":<b64>}}`.
   - Accepts the same entry shapes already supported (`data_url` string,
     `{data_url}` dict, `{data, media_type}` dict).
   - Caps:
     - images: 3 max, ~5 MB raw each — unchanged.
     - PDFs: **2 max, 10 MB (raw) each**, `application/pdf` only.
     - **combined cap**: total base64 size across all attachments ≤ ~25 MB
       (keeps the whole request under Claude's 32 MB limit); over-cap entries
       are rejected with a clear error string.
   - The existing `_normalize_image_payload` is kept as a thin wrapper (or
     folded in) so nothing else that calls it breaks.

2. **`extract_attachment_content(blocks, query)`** — renames/generalizes
   `extract_image_content`. Sends all image **and** document blocks in one
   Haiku call with the same "extract everything visible, label each item, do
   not diagnose" instruction (instruction text extended to mention document
   pages). Returns the extraction string. Bytes are not persisted. Model stays
   `claude-haiku-4-5-20251001` to match surrounding code (deliberate cost
   choice for high-volume public chat). The old name is kept as a wrapper if
   anything else references it.

3. **Route wiring** — a small shared inline helper reads `images`,
   `documents`, and `images_consented` from the request JSON, enforces the
   consent gate, and calls `_normalize_attachments` →
   `extract_attachment_content`. Applied to:
   - `/chat` (extend existing image handling to also pass `documents`).
   - `/begin/concierge/chat` (new).
   - `/begin/match/chat` (new).
   In each route the extracted text is (a) joined to the embedding input for
   retrieval and (b) appended to the user turn as context, mirroring the
   existing `/chat` behavior. Each route logs the extracted text + a combined
   `image_count` (renamed conceptually to attachment count, column reused).

### Frontend — new shared module `static/chat-attachments.js`

Lifts the upload UI out of `index.html` into a reusable module matching the
existing shared-module pattern (`op-nav.js`, `mic-input.js`, `tts-output.js`).

- Renders: attach button, hidden `<input type="file"
  accept="image/png,image/jpeg,image/webp,image/gif,application/pdf" multiple>`,
  consent checkbox (relabeled "Allow attaching **documents and images**
  (lab results, supplement labels, scan PDFs, photos) — content is extracted as
  text; the original file is not saved."), drag-drop zone, and preview chips
  (image thumbnail or a PDF file-name chip) each with a remove button.
- Holds pending files in memory as `{data_url, name, kind}` (`kind` =
  `image`|`document`). Enforces the same client-side caps as the backend and
  shows inline errors.
- Public API:
  - `ChatAttach.mount(opts)` — wires the UI into a given container; persists
    consent in `localStorage` under the existing `amg_images_consented` key.
  - `ChatAttach.getPayload()` → `{ images:[{data_url}],
    documents:[{data_url, name}], consented:Boolean }`.
  - `ChatAttach.clear()` — called after a successful send.
- Each page (`embed.html`, `concierge.html`, `begin-match.html`) adds one
  `<script src="/static/chat-attachments.js">`, a mount call, and merges
  `getPayload()` into its existing chat POST body
  (`images`, `documents`, `images_consented`).
- The retired `index.html` is left unchanged.

## Error handling

- Attachments present but `images_consented` false → 400 with a clear message
  (existing behavior, now covers documents).
- Per-entry errors (bad media type, over-size, over-count, combined-cap) are
  collected and surfaced; valid attachments still process.
- Extraction failure returns an `[attachment-extraction-error: …]` marker
  (existing pattern) rather than failing the whole chat turn.

## Testing

New `tests/test_chat_attachments.py`:
- `_normalize_attachments`: PDF block shaping; rejects non-pdf/non-image media
  types; enforces image count/size, PDF count/size, and combined-payload caps;
  preserves existing image behavior.
- Consent gate: a route receiving `documents` without `images_consented`
  returns 400 (mirror the existing image gate; reuse the `test_chat_tts.py`
  route-test style with the Anthropic client mocked).
- `extract_attachment_content`: with the Anthropic client mocked, asserts both
  image and document blocks are forwarded in the single call.

## Out of scope (YAGNI)

- Office formats (.docx/.xlsx/.txt) — images + PDF only for now.
- Files API upload / persistence — base64 inline only; bytes still discarded.
- Refactoring the retired `index.html` onto the shared module.
- Storing or re-displaying uploaded files.
