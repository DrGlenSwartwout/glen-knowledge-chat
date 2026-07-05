# Community — community-aware "Ask Dr. Glen" chat (PB→illtowell Community, Layer C, slice C3) — Design

**Date:** 2026-07-05
**Status:** Approved in brainstorm with Glen 2026-07-05 (augment the members' portal chat with tier-aware related-community cards, reuse C1, portal-only).
**Repo:** deploy-chat
**Session:** PB→illtowell · EVOX (continued)

**Relates to / reuses:**
- The existing members' chat: `POST /api/portal/<token>/chat` (app.py:15159) — already embeds the query (`embed(query)`), does knowledge-base RAG (`_match_query_namespaces` + `build_context`), injects a `CONTEXT:` block, and streams a Claude answer via `sse(...)` events.
- C1: `dashboard/community.py:get_embeddings(cx, content_ids, model)` + `list_full(cx)`; `dashboard/community_feed.py:cosine`; the app.py `_community_candidates(cx, is_paid)` tier-shaping helper and `COMMUNITY_FEED_MODEL`.
- `_is_paid_member`, `_portal_record_for`, `sse`, `LOG_DB`, the portal chat UI (in `static/client-portal.html`).
- [[project_pb_to_illtowell_evox]], [[project_portal_top_chat]], [[reference_portal_chat_triage]].

## Context and boundary

Community Layer C has three slices: C1 (curated feed, LIVE), **C3 = community-aware chat (THIS SLICE)**, C2 (opt-in introductions). C3 reuses C1's relevance engine: when a member asks the "Ask Dr. Glen" chat something, the chat also surfaces the Community content most relevant to their question.

**Hard privacy line (governs all of C):** the member's query is their own; matching content to it is one-directional (surfacing content TO them). No other member's data is involved, and no full Rumble link reaches a non-member.

## Scope

**Augment the existing members' portal chat with tier-aware "related from the community" cards.** One retrieval helper (reusing C1), a small addition to the chat route (inject a context note + emit a final `related` SSE event), and the related-cards render in the portal chat UI.

**Deferred:** making the public pre-member widget/funnel chat community-aware (a different marketing motion), per-item deep links (cards link to `/community`), and C2 (introductions).

## Components

### 1. Related-content retrieval (app.py helper)

- `_community_related(cx, query_vec, is_paid, *, k=2, min_sim=0.72) -> [dict]`:
  - Reuse `_community_candidates(cx, is_paid)` to get the tier-visible content items (paid → full items; free → teaser dicts with no `video_ref`).
  - Load their vectors via `community.get_embeddings(cx, ids, COMMUNITY_FEED_MODEL)` (items already embedded lazily by the C1 feed; any not-yet-embedded item is simply skipped here — no lazy embed on the chat path, to keep chat latency low).
  - Score each by `community_feed.cosine(query_vec, vec)`, keep those `>= min_sim`, take the top `k`.
  - Return a **card-shaped** list: `{"id", "title", "kind": "full"|"teaser", "url": PUBLIC_BASE_URL + "/community?token=" + <portal token>}`. The card carries NO `video_ref` (it links to the gated `/community` page). `kind` drives the label ("Watch the replay" vs "Preview and become a member").
  - Never raises (returns `[]` on any error — the chat must not break).

### 2. Chat route wiring (`POST /api/portal/<token>/chat`)

- After the existing `qvec = embed(query)` (the route already embeds the query for KB RAG — reuse that vector; do not embed twice), call `_community_related(cx, qvec, _is_paid_member(email), k=2)`.
- **Context injection:** if any related items, prepend a short line to the existing `CONTEXT:` block, e.g. `"Relevant community sessions the member can open: <title1>; <title2>."` so the assistant can naturally acknowledge them.
- **Structured emit:** after the answer stream completes, emit one final `sse({"related": items})` event before the stream closes, so the frontend can render cards. This is additive to the existing token events.
- All of this is best-effort/fail-open: a retrieval failure logs and skips, never breaks the chat.

### 3. Portal chat UI (`static/client-portal.html`)

- On the chat's SSE stream, handle the new `related` event: render each item as a small "From the community" card under the just-finished answer, showing the title and a link to `item.url` (the member's `/community` page). Free-tier cards (`kind: "teaser"`) show the "become a member for the full session" nudge.
- Cards inserted via `textContent` for titles (no innerHTML injection). Copy: no em dashes, no ALL CAPS.

## Config

No new env. Reuses `COMMUNITY_FEED_MODEL`, `PUBLIC_BASE_URL`, the embeddings keys, and C1's stores.

## Copy guidance

Client-facing copy (card labels, the assistant's acknowledgement): no em dashes, no ALL CAPS. Warm and helpful. The free-tier nudge names the value without disparaging.

## Testing

- Pure/route (`_community_related`): with two published+embedded content items and a query vector near one of them, returns that item first; items below `min_sim` are excluded; returns `[]` when no item clears the bar or when no embeddings exist; free tier returns teaser-kind cards whose `url` is the `/community` link and which carry NO `video_ref`; paid returns full-kind cards; never raises on a malformed store.
- Chat route: `POST /api/portal/<token>/chat` still streams an answer (unchanged), and when related items exist it emits a final `related` SSE event with the cards; a retrieval failure (mock `_community_related` to raise) does NOT break the answer stream (fail-open). The related items reflect the member's tier (free → teaser, no full link).
- Regression: the existing KB RAG, triage, and persistence in the chat route are unchanged; C1/B/A untouched; the query is embedded once (reused, not re-embedded).
- Go-live: as a paid member, ask the portal chat about a topic with a matching replay and confirm a "From the community" card appears linking to `/community`; as a free member, confirm the card is the teaser with the membership nudge and no full Rumble link.

## Deferred (future Community slices)

- **C2:** opt-in like-minded-member introductions.
- Public widget/funnel chat community-awareness (marketing motion).
- Per-item deep links / scroll-to-item on `/community` (cards currently link to the page).
- Lazy-embedding on the chat path (C3 relies on the C1 feed having embedded items; a never-opened-feed member's brand-new content may lack vectors until the feed runs).
