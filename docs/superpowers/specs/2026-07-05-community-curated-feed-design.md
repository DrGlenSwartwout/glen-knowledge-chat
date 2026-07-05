# Community — curated "for you" feed (PB→illtowell Community, Layer C, slice C1) — Design

**Date:** 2026-07-05
**Status:** Approved in brainstorm with Glen 2026-07-05 (in-app cosine, transparency line, free top-3 / paid full).
**Repo:** deploy-chat
**Session:** PB→illtowell · EVOX (continued)

**Relates to / reuses:**
- Layer A: `dashboard/community.py` (`community_content`, `list_full`, `list_outtakes`, `get_content`), `GET /api/community/library`, `static/community.html`.
- Layer B: `dashboard/community_signals.py` (`my_signals` → likes/blocks; feeds the feed's boost + filter).
- `dashboard/journal_store.py:select(cx, *, since_iso, order, limit)` (member's own reflections = interest signal), `dashboard/openai_failover.py:build_openai_client` + the existing `text-embedding-3-small` embeddings pattern (app.py:21497), `_evox_ident`, `_is_paid_member` (tier depth), `LOG_DB`, `_db_lock`.
- [[project_pb_to_illtowell_evox]], [[reference_e4l_ingestion_pipeline]] (existing Pinecone/embeddings usage), [[reference_video_hosting_rumble]].

## Context and boundary

Community has three layers: A = content library (LIVE), B = signal layer (LIVE), C = curation + matchmaking. Layer C decomposes into three slices: **C1 = the curated "for you" feed (THIS SLICE)**, C3 = community-aware AI chat, C2 = opt-in like-minded-member introductions. C1 is the foundation the other two build on, and it works at any scale (a single member with no social graph still gets a feed).

**Hard privacy line (governs all of C):** a member's private journal and chat may be used to surface content *to* them (one-directional, safe), but never to surface *them* to anyone else. C1 is entirely one-directional: the interest vector is built only from the member's own data, used only to rank content for them, and never exposed. No other member's data touches a member's feed.

## Scope

**Rank the Community content a member can see, by relevance to them, and surface a short "for you" feed with a transparency line.** One ranking module + a content-embedding store (lazy) + one feed route + a "For you" section on `/community`. No matchmaking, no chat changes.

**Deferred:** C3 (community-aware chat), C2 (introductions), Pinecone-backed retrieval (in-app cosine is the C1 implementation; the ranking interface is swappable), and chat-history as an interest input beyond a best-effort include.

## Relevance model

Two inputs combine into a per-member ranking of the content items the member can see (Layer A already tier-gates visibility: paid sees full recordings + out-takes; free sees out-take teasers):

1. **Explicit (Layer B `my_signals`):** content whose `interest_tags` intersect the member's **liked** topics gets a score boost; content whose tags intersect a **blocked** topic is filtered out entirely. (Person-blocks are stored but moot for C1 since all content is Glen's; the filter hook is kept for C2.)
2. **Implicit (the core magic):** a member **interest vector** = embedding of the member's own recent journal entries + their liked topic names + (best-effort) recent chat text. Each content item has a **content vector** = embedding of its title + tags + a transcript snippet. Rank by cosine similarity between the member vector and each content vector.

**Combined score** per item = cosine similarity + a fixed boost when a liked topic matches. Blocked-topic items are removed before ranking.

**Cold start:** if the member has no journal, no chat, and no likes (no interest vector), the feed falls back to newest content, tie-broken by reaction count. The feed is never empty.

**Transparency line** per item ("why you're seeing this"), chosen by the strongest contributing signal:
- a liked topic matched → "Because you liked {topic}"
- otherwise, member had an interest vector → "Related to your recent reflections"
- cold-start fallback → "New in the community" (or "Popular right now" for the reaction-ranked tail)

## Components

### 1. Content embedding store + lazy embed (`dashboard/community.py` extension)

- Sidecar table `community_embeddings(content_id INTEGER PK, vec TEXT, model TEXT, updated_at TEXT)` — `vec` is the JSON-encoded float list; `model` records the embedding model so a model change forces re-embed.
- `set_embedding(cx, content_id, vec, model)`, `get_embeddings(cx, content_ids) -> {content_id: [float]}` (only rows whose `model` matches the current model).
- **Lazy embed:** the feed embeds any visible published item that lacks a current-model vector at feed time and stores it (self-healing; no change to the already-shipped publish path). A content item's embed text = `title + " " + tags joined + " " + transcript[:2000]`.

### 2. Ranking module (`dashboard/community_feed.py`, pure + injectable embedder)

- `cosine(a, b) -> float` (0 if either is empty/zero-norm).
- `build_interest_text(journal_texts, liked_topics, chat_texts) -> str` (concatenate; empty when the member has no signal).
- `rank(candidates, member_vec, content_vecs, liked_topics, blocked_topics, *, now, boost=0.15) -> [ranked_item]` — filter out items whose tags intersect `blocked_topics`; if `member_vec` is non-empty, score = cosine + boost·(liked-tag match); else cold-start order = newest then reaction count. Returns items with a `score` and a `reason`.
- `reason_for(item, liked_topics, has_vec, cold_start) -> str` — the transparency line per the rules above.
- Embedding calls are injected (a callable `embed(text)->[float]`) so the module is unit-testable without network. In app.py the injected embedder wraps the existing `text-embedding-3-small` client.

### 3. Feed route (`app.py`)

- `GET /api/community/feed?token=…` — `_evox_ident` (bad token → 404). Build the visible candidate set from Layer A (`list_full` if `_is_paid_member` else the free teaser set), drop blocked-topic items, lazy-embed missing vectors, build the member interest vector (journal via `journal_store.select` over the last N days + liked topics + best-effort chat), rank, and return the top **K** items each with its tier-appropriate fields + `reason`. **Depth: free K=3, paid K=10.** Response: `{items: [...], cold_start: bool}`. The free payload carries NO full `video_ref` (same allowlist rule as Layer A — the feed reuses the library's per-tier item shaping).
- Member interest vector is cached in `member_interest(email PK, vec TEXT, model TEXT, built_at TEXT)` and rebuilt when older than a TTL (e.g. 24h) or absent; a rebuild failure falls back to cold-start rather than erroring.

### 4. Member surface (`static/community.html`)

- A "For you" section at the top of `/community` that fetches `GET /api/community/feed` and renders the ranked items (reusing the existing card renderer) each with its transparency line. If `cold_start`, a gentle header ("New in the community"); otherwise "For you". The full browsable library stays below, unchanged.
- Copy: no em dashes, no ALL CAPS. The transparency line is quiet and honest.

## Config

- No new required env (reuses the OpenAI embeddings keys already configured for the E4L/journal Pinecone pipeline). `COMMUNITY_FEED_MODEL` optional (default `text-embedding-3-small`). Feed depths `COMMUNITY_FEED_FREE_K` (default 3) / `COMMUNITY_FEED_PAID_K` (default 10) optional.

## Privacy

- The interest vector is built ONLY from the member's own journal/chat/likes, stored keyed to their own email, used ONLY to rank their own feed, and never returned to the client or any other member. Blocked topics/people are hard-filtered. The feed never reveals who else liked/reacted to anything (it reuses Layer A/B shaping, which is already aggregate/self-scoped).

## Testing

- Pure (`dashboard/community_feed.py`): `cosine` (orthogonal→0, identical→1, empty→0); `build_interest_text` (empty when no signal; concatenates journal+likes+chat); `rank` (blocked-topic items removed; liked-topic boost changes order; cold-start falls back to newest-then-reactions when member_vec empty; deterministic with an injected fake embedder); `reason_for` (each branch).
- Store: `set_embedding`/`get_embeddings` round-trip; `get_embeddings` skips rows whose `model` differs from the current model (forces re-embed).
- Route: feed for a paid member returns full items + reasons; free member returns ≤3 teaser items with NO full `video_ref`; a member with a blocked topic never sees content tagged with it; cold-start member (no journal/likes) gets newest-first; bad token → 404. Embedding + journal reads mocked.
- Regression: Layer A/B untouched; the feed is read-only over their stores (only new writes are the embedding + member_interest caches).
- Go-live: as a member with journal entries + a liked topic, load `/community` and confirm the "For you" section ranks relevant content with correct transparency lines; a fresh member sees the cold-start feed; a free member sees ≤3 items and no full Rumble link.

## Deferred (future Community slices)

- **C3:** community-aware AI chat (reuses this relevance engine to point a member to a replay in chat).
- **C2:** opt-in like-minded-member introductions (needs consent, matching, and a connection channel; its own design).
- Pinecone-backed content retrieval (swap in behind the `rank`/embedding interface once the library outgrows in-app cosine).
- Richer chat-history mining and per-signal feedback ("show me less like this").
