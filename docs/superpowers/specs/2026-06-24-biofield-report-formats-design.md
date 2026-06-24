# Biofield Analysis — Report Formats Design

**Date:** 2026-06-24
**Status:** Design approved (pending spec review)
**Context:** The local Biofield Analysis tool (`biofield_local_app.py` + `dashboard/biofield_*`) already produces a three-section report — **Causal Chain table**, **Remedy Schedule**, **Narrative** (with optional your-voice audio + talking-head video) — but only as one combined *authoring* screen. This design adds clean, client-facing **output formats**: printed/PDF, email, and an online client **portal** with audio, on-request video, and a Q&A chat.

---

## Goals

- One report, rendered cleanly in multiple delivery formats — no edit chrome in client-facing output.
- A **printed PDF** Glen/Rae can print and ship with the order, AND that the client can download from the portal (identical file).
- An **email** that securely notifies the client their report is ready.
- A **client portal** page: the three sections + audio narration + on-request video + a report-grounded Q&A chat, with per-client UI memory (collapse states, light/dark).
- Keep PHI local where possible; reuse existing server infrastructure (magic-link auth, `/chat` RAG, `portal_biofield_reports`).

## Non-goals

- Replacing the authoring screen (it stays as-is).
- Changing how tests are authored/interpreted.
- Auto-generating video for every client (cost control: video is on-request only).

---

## Architecture: one content model → one renderer → three skins

```
Local authoring tool ── build_report(test) ──▶ report object (3 sections + media refs)
        │                                              │
        │                                  presentation_render(report, opts)   ← shared, no edit chrome
        │                                              │
        ├─ PRINT:  render + print.css → headless Chromium → report.pdf   (saved locally to ship; uploaded to portal)
        ├─ EMAIL:  short "report ready — view securely" + magic-link button (no PHI in body)
        └─ PORTAL: publish to server → report row + audio upload → /begin/biofield/<token>
                       served behind magic-link + report-grounded /chat + on-request server-side video
```

**Shared content model.** A `build_report(test)` produces a plain dict: client/date header, `layers[]` (causal chain rows incl. depth tags), `schedule` (time-of-day grid), `narrative` text, and `media` refs (audio path/url, video status). All three skins consume this — they never re-query.

**Shared presentation renderer.** `presentation_render(report, opts)` emits clean semantic HTML for the three sections with **no buttons/textareas/pickers**. `opts` controls skin concerns (print vs screen, which sections, theme). This is the single source of truth for layout; print/portal differ only by stylesheet + wrapper.

---

## Format 1 — Printed / PDF

- **Order: schedule-forward.** Letterhead → **Remedy Schedule** ("How to take your remedies") → **Narrative** → **Causal Chain table** → footer. Rationale: the sheet physically ships with the bottles, so the actionable schedule leads.
- **Real file, not browser-only.** Generated from the presentation HTML via **headless Chromium** (Playwright is already in the stack) → `report.pdf`. Saved **locally** (so Glen/Rae print + ship it) and **uploaded to the portal** (so the client downloads the identical file).
- **Letterhead/footer** (see Branding, below): centered logo + wordmark + client name + date; footer sign-off + contact/site.
- All three sections at **full detail** (client sees the same causal chain table — Glen's choice: maximum transparency).

## Format 2 — Email

- **Thin secure notification**, not embedded PHI: a short "Your Biofield Analysis is ready — view it securely" message with a **magic-link button** to the portal.
- Because the body carries **no PHI** (just a tokened link), the tool **sends it directly** at publish time, reusing the existing `send_magic_link_email` pattern — no copy/paste step.

## Format 3 — Client Portal (the rich view)

- **Layout: single-column, narrative-first** (mobile-friendly, matches Glen's warm voice): letterhead → 🎧 Narration → 🎥 Video → Narrative → Remedy Schedule → Causal Chain → Download PDF.
- **Served behind magic-link auth** at `/begin/biofield/<token>`, reusing the existing `auth_tokens` `biofield_reveal` purpose (24h TTL; re-mintable). The email's link is this token.
- **Published from local → server.** On "Publish", the local tool POSTs the report object + uploads the generated **PDF** and **audio mp3** to the server, which stores it (extending `portal_biofield_reports` with a `manual_biofield` report type) and mints the magic link.
- **Collapsible sections** with an unobtrusive chevron; **per-client remembered open/closed state** per section (stored per report+section). ▾ = open, ▸ = closed.
- **Light/dark mode** toggle, **remembered per client** (same UI-state store as collapse).

### Portal media

- **Audio narration: pre-made + reviewed.** Glen generates + reviews the ElevenLabs your-voice audio locally (as today), uploaded at publish. Section is **collapsed by default**; the player **lazy-loads on open** (page stays light). Cost ≈ pennies; Glen keeps review. Remembered open → starts open next visit.
- **Video: on-request, AI-generated, server-side.** No auto-spend. The client taps **"Generate video walkthrough"**; the **server** generates it (chosen over a local worker because a sleeping Mac can't reliably service the request), then notifies the client. Once made, it's cached + remembered-open.

#### Video format

Talking-head bookends + voice-over B-roll middle (educational, avoids avatar fatigue):

1. **Talking-head INTRO (~10s, personalized):** "Aloha [Name]…" — Glen's avatar + cloned voice.
2. **VO + B-roll body (per causal-chain layer):** Glen's VO (from the approved narrative) over **B-roll keyed to the client's remedies** — botanical + mechanism/"how it heals" visuals.
3. **Talking-head OUTRO (~10s, reusable):** "Stay in touch with any questions…"

- **B-roll source: per-remedy asset library.** Botanical + mechanism visuals are built **once per formulation** via the `formulation-image-studio` (Flux) + **HyperFrames** motion, stored, and **reused across every client** on that remedy. Per-client video = personalized intro + the library clips matching their causal chain + VO + outro, stitched (HyperFrames). This makes the heavy creative amortized and keeps per-client server assembly light. The library doubles as marketing assets.
- A `remedy → b-roll assets` mapping grows as formulations get the image-studio treatment; a remedy with no assets yet falls back to a captioned/title card.

### Portal chat (Q&A)

- **Report-grounded + knowledge base, under compliance guardrails.** Reuses the existing `/chat` RAG pipeline (embed → Pinecone → context) with the **client's report injected as context** (their causal chain, remedies, schedule) so it can answer "what is this remedy / when do I take it / what does this layer mean" AND draw on the broader Accelerated Self Healing knowledge base. Compliance scan applies as on the main chat.

---

## Branding (one open item — confirm at review)

Default placeholders used in mockups, to confirm:
- **Wordmark:** "DR. GLEN & RAE"
- **Tagline:** "accelerated self healing™"
- **Logo:** text wordmark for now (swap in an image file if provided)
- **Footer sign-off:** "In wellness, Dr. Glen & Rae" + contact/site line

---

## Build phases (for the implementation plan)

1. **Shared core + Print/PDF (local):** `build_report` content model + `presentation_render` + print stylesheet + headless-Chromium PDF generation, saved locally. Clean on-screen report. *All local; no server changes.*
2. **Publish + Portal shell:** publish local→server (extend `portal_biofield_reports`), `/begin/biofield/<token>` page with layout A, magic-link auth, email notification, PDF upload/download, collapse + remembered state + light/dark memory.
3. **Audio:** upload pre-made audio at publish; lazy-loaded player in the portal.
4. **Chat Q&A:** report-grounded `/chat` integration + compliance.
5. **On-request video (server-side):** per-remedy B-roll asset library + assembly pipeline (intro + library clips + VO + outro) + request/generate/notify flow.

Phases 1–4 ship the full text+audio+chat experience; Phase 5 adds the rich video.

---

## Testing

- `build_report` / `presentation_render`: pure functions, unit-tested on authored + FMP tests (golden-HTML for the 3 sections, schedule-forward order, collapse markup).
- PDF generation: smoke test that a non-empty PDF is produced from sample HTML.
- Publish + portal: token mint/verify, report round-trip, per-client UI-state persistence (collapse + theme).
- Chat: report-context injection + compliance gate.
- Video (Phase 5): remedy→asset mapping resolution, assembly manifest, request state machine.
- Reuse existing biofield suite patterns (injectable LLM/TTS, local PHI never committed).
