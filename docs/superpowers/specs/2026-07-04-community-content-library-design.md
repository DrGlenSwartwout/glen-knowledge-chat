# Community — content library + AI cataloging (PB→illtowell, Community subsystem, slice 1) — Design

**Date:** 2026-07-04
**Status:** Approved in brainstorm with Glen 2026-07-04.
**Repo:** deploy-chat
**Session:** PB→illtowell · EVOX (continued)

**Relates to / reuses:**
- `_is_paid_member(email)` (app.py:5120) — the member gate for full content.
- `_evox_ident` / `portal_identity.resolve_identity` (portal-token auth), `PUBLIC_BASE_URL`, `CONSOLE_SECRET` (console/publish auth), `STATIC`, `_db_lock`, `LOG_DB`, sqlite conventions.
- `dashboard/video_trim.py` (ffmpeg dead-air/clip cutting) — reused to cut out-take clips.
- `dashboard/openai_failover.py` / `openai` client — Whisper transcription + Claude cataloging suggestions.
- `biofield_local_app.py` (local Flask authoring app on 127.0.0.1:8011) — the pattern for the local cataloging tool.
- `dashboard/biofield_portal_publish.py` — the pattern for a console-authed publish-to-prod endpoint.
- [[reference_video_hosting_rumble]] (Rumble unlisted, not YouTube — health-content censorship), [[project_pb_to_illtowell_evox]] (Community is the last PB→illtowell subsystem), [[reference_audio_production_workflow]] (existing Whisper workflow), [[project_portal_element_backdrop]] (existing in-page media serving).

## Context and boundary

Community is the last PB→illtowell subsystem: one place where members are supported and encouraged to interact with chat and video. **Courses fold into Community** — a course session and a weekly group-coaching call are the same shape (a gated full recording with free out-take clips that tease it by interest), so they are two content *types* in one library, not separate subsystems.

The full Community vision has three layers, built bottom-up:
- **Layer A — content library + gating (THIS SLICE).** Gated full recordings (paid) + free out-take teasers, with AI-assisted cataloging.
- **Layer B — interaction surface.** Members post/react/comment, like/block people and topics.
- **Layer C — AI curation + matchmaking.** Mine private chat/journal signal to surface relevant content to a member, and (opt-in only) introduce like-minded people.

**Hard privacy line (binds all layers, stated here so C is designed correctly):** private journals and chat may be used to surface content *to* a member (one-directional, safe), but must never surface a member *to* other members, or expose their words, without an explicit per-item opt-in. Journals are confessional.

This slice is **Layer A only**: the content library, its member-facing surface with the free/paid gate, and the local AI-assisted cataloging tool that publishes into it.

## Scope

**Catalog a recording → publish (full = paid Rumble embed, out-takes = free self-hosted clips) → members watch, gated by tier.** One prod-side content store + member library page + console/publish API, plus a local Mac cataloging tool (Whisper + Claude + ffmpeg + approve + push).

**Deferred / out of scope:**
- Layer B (react/like/block, comments, the interaction surface) and Layer C (AI curation feed, opt-in member matchmaking).
- The fully automated pipeline: Zoom cloud-recording auto-fetch (blocked until Glen re-enables the Zoom S2S app), YouTube back-catalog bulk import, scheduling.
- Member-featured out-takes (clips of members) and their consent/approval flow — slice-1 out-takes are Glen-only, needing no consent. YouTube is a future *source* to clip from, never the durable host (censorship risk).
- Leak-proof paywalling of full replays (see Media hosting) — slice 1 uses a soft page-gate.

## Components

### 1. Content store (prod, deploy-chat)

- `community_content(id INTEGER PK, type TEXT, title TEXT, description TEXT, video_ref TEXT, tier TEXT, interest_tags TEXT, parent_id INTEGER, transcript TEXT, published INTEGER DEFAULT 0, published_at TEXT, created_at TEXT)`.
  - `type` ∈ {`coaching_replay`, `course_session`, `outtake`}. Coaching replays and course sessions are identical in shape; only `type` differs (drives labeling/filtering).
  - `tier` ∈ {`paid`, `free`}. Full recordings are `paid`; out-takes are `free`.
  - `video_ref` — for a full item, the Rumble unlisted embed URL/id; for an out-take, the app-relative path to the self-hosted clip file.
  - `parent_id` — an out-take's `parent_id` points at the full item it teases (null for full items).
  - `interest_tags` — JSON array of topic/interest strings (stored now; Layer C sorts on them later).
  - `transcript` — Whisper transcript text (stored on full items; supports future search + Layer C).
- Module `dashboard/community.py` (pure + sqlite): `init_community_tables(cx)`; `create_content(cx, *, type, title, description, video_ref, tier, interest_tags, parent_id=None, transcript=None) -> int`; `get_content(cx, content_id) -> dict|None`; `publish(cx, content_id)`; `list_full(cx) -> [dict]` (published `paid` full items, newest first); `list_outtakes(cx, parent_id=None) -> [dict]` (published `free` out-takes, optionally for one parent); `outtakes_for(cx, parent_id) -> [dict]`. Emails/text sanitized per existing conventions; tags round-trip as JSON.

### 2. Member-facing library (prod)

- `GET /community` serves `static/community.html` (self-contained page).
- `GET /api/community/library?token=…` → tier-aware payload. `_evox_ident` resolves the token; `_is_paid_member(email)` decides tier.
  - **Paid member:** `{tier:"paid", full:[{id,type,title,description,video_ref,interest_tags,published_at, outtakes:[…]}], outtakes:[…]}` — full replays and course sessions with their Rumble embeds, plus out-takes.
  - **Free member (or no active membership):** `{tier:"free", full:[{id,type,title,description,interest_tags, teaser_outtakes:[…]}]}` — **no `video_ref` for full items** (field allowlist; the full Rumble link never reaches a non-member), only the metadata and the free out-take clips, each with a "become a member for the full session" tease pointing at the paid parent (`parent_id`).
- `GET /community/clip/<id>` — serves a self-hosted out-take clip file (free, ungated). Full recordings are NOT served by the app; they play via the Rumble embed inside the gated page.
- Copy on the page and teases: warm, no em dashes, no ALL CAPS.

### 3. Media hosting split

- **Full recordings (paid):** hosted on **Rumble unlisted** (Glen's censorship-safe workflow; YouTube/Vimeo remove health content). The `video_ref` is the Rumble embed; the member-library PAGE is gated on `_is_paid_member`, and the full `video_ref` is withheld from the free payload (soft page-gate). Accepted caveat: an unlisted Rumble URL is shareable, so a paid member could leak a full replay. Acceptable for a trust-based community; leak-proof self-hosted streaming is a deferred hardening step.
- **Out-take clips (free):** short files cut by ffmpeg from the source recording, **self-hosted** in the app and served by `GET /community/clip/<id>` (small, free, no paywall, trivial egress). This gives the out-take pipeline a home without any Rumble upload API (Rumble has no reliable public upload API).

### 4. Console / publish API (prod)

- `POST /api/console/community/publish` (`CONSOLE_SECRET`-gated, mirrors `biofield_portal_publish`): accepts a finished catalog entry from the local tool — full-item fields + Rumble `video_ref` + transcript + interest tags, and a list of out-take clips (each: title, tags, the clip file). Creates the full `community_content` row, stores each out-take clip file under the app's clip directory and creates its `outtake` row with `parent_id` set, and marks all published. Idempotent on the full item's Rumble `video_ref` (unique per recording): re-publishing the same session updates the existing full row and replaces its out-takes rather than duplicating.
- Multipart upload for the out-take clip files (small). The full recording file is never uploaded to prod (it lives on Rumble).

### 5. Local cataloging tool (Glen's Mac)

- A local Flask app (pattern: `biofield_local_app.py`, 127.0.0.1:8011-style, its own port), NOT part of the deployed app. Flow:
  1. Glen points it at the source recording file (local path) and pastes the Rumble unlisted URL of the full published replay, and picks the type (coaching_replay / course_session).
  2. **Whisper** (`openai` client via `openai_failover`) transcribes the source audio → transcript.
  3. **Claude** reads the transcript and suggests: a title, interest tags, and out-take moments (timestamp ranges with a one-line reason each).
  4. Glen reviews in a simple local view: edit the title/tags, accept/reject/adjust each suggested out-take range.
  5. **ffmpeg** (`video_trim`) cuts the approved out-take ranges from the source file → short mp4 clips locally.
  6. The tool POSTs the finished entry (full metadata + Rumble ref + transcript + tags + the out-take clip files) to `POST /api/console/community/publish` with the `CONSOLE_SECRET`.
- Runs where the large source file already is (Glen's Mac), so no multi-GB upload to Render. Only the short out-take clips travel to prod.

## Config

- Prod: no new required env (reuses `CONSOLE_SECRET`, `PUBLIC_BASE_URL`, `_is_paid_member`). A clip-storage directory under the app's data dir (e.g. `DATA_DIR/community_clips/`).
- Local tool: `OPENAI_API_KEY` (Whisper, already in Doppler), `ANTHROPIC_API_KEY` (Claude suggestions), `PUBLIC_BASE_URL` + `CONSOLE_SECRET` (to publish), `ffmpeg` on PATH.

## Copy guidance

Client-facing copy (library page, teases, out-take captions): no em dashes, no ALL CAPS. Warm, inviting. The free-member tease names the value of the full session and points at membership, without disparaging.

## Testing

- Pure/sqlite (`dashboard/community.py`): create/get/publish; `list_full` returns only published paid full items newest-first; `list_outtakes`/`outtakes_for` return only published free out-takes and filter by parent; tags round-trip as JSON; unpublished items are excluded.
- Route/api (prod): `GET /api/community/library` — paid member sees full `video_ref` + out-takes; free member sees NO full `video_ref` (allowlist), only metadata + free out-take teasers with parent linkage; bad token → 404. `GET /community/clip/<id>` serves a free out-take and 404s an unknown/paid id. `POST /api/console/community/publish` — `CONSOLE_SECRET`-gated (401 without), creates the full row + out-take rows with `parent_id`, stores clip files, idempotent on re-publish.
- Regression: EVOX/consult/triage/masterclass/onboarding untouched; the member gate reuse (`_is_paid_member`) is read-only.
- Local tool: transcription and Claude-suggestion functions are unit-tested with a mocked OpenAI/Claude client; ffmpeg cut ranges validated by `video_trim`'s existing tests; the publish POST is integration-tested against a mocked endpoint.
- Go-live: publish one real coaching replay + 2 out-takes via the local tool; verify a paid member sees the Rumble full embed + out-takes on `/community`, and a free member sees only the out-takes with the membership tease and no full link.

## Deferred (future Community slices)

- **Layer B:** interaction surface — post/comment, react, like/block people and topics.
- **Layer C:** AI curation feed (surface relevant content to a member from their private journal/chat signal, one-directional) + opt-in like-minded-member introductions (honoring the hard privacy line).
- **Pipeline automation:** Zoom cloud-recording auto-fetch (needs the Zoom S2S app re-enabled), YouTube back-catalog bulk import (as a clip source, not durable host), scheduling.
- **Member-featured out-takes** + their consent/approval flow (record → clip → member approves → publish).
- **Leak-proof full-replay paywalling** via self-hosted token-streamed video (only if unlisted-Rumble leakage becomes a real problem).
