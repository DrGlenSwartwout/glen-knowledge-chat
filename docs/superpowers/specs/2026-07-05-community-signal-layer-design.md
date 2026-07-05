# Community — signal layer (PB→illtowell Community, Layer B) — Design

**Date:** 2026-07-05
**Status:** Approved in brainstorm with Glen 2026-07-05 (signal-layer-only scope; chat-awareness deferred to Layer C).
**Repo:** deploy-chat
**Session:** PB→illtowell · EVOX (continued)

**Relates to / reuses:**
- `dashboard/community.py` (Layer A content store: `community_content`, `get_content`), `static/community.html` (Layer A member page — extended here), `GET /api/community/library`.
- `_evox_ident` (portal-token → identity), `_is_paid_member` (tier, read-only), `LOG_DB`, `_db_lock`, sqlite conventions, `dashboard/product_reviews.py` (UGC-store pattern).
- [[project_pb_to_illtowell_evox]] (Community is the last PB→illtowell subsystem), [[reference_video_hosting_rumble]].

## Context and boundary

Community has three layers: A = content library (SHIPPED, live), B = signal layer (THIS SLICE), C = AI curation feed + opt-in matchmaking. Layer B's job is narrow: capture the private like/block/react signals that Layer C's curation runs on, and make the library feel alive, without becoming a forum (a moderation and safety liability for a solo practice, and empty-feeling at low volume).

**Explicitly NOT in Community:** free-form discussion threads, member-to-member chat or DMs, and public posts. Glen chose the signal-layer-only scope.

**Hard privacy line (binds all layers):** private signals stay private. Reaction *counts* are shown in aggregate; **who** reacted is never exposed (in a health community, showing that a named member reacted "this is me" to a replay about a stigmatized condition would out them). Like/block signals are visible only to the member who set them.

## Scope

**React to content (anonymous aggregate counts) + privately like/block topics and people.** One new signal store + a few portal-token-gated routes + reaction and interest affordances added to the existing `/community` page. These signals are the input Layer C will later curate on; Layer B does not itself build a feed.

**Deferred to Layer C:** the curated "for you" feed, opt-in like-minded-member introductions, and making the AI chat community-aware (a surfacing behavior that needs B's signals to draw on).

## Components

### 1. Signal store (`dashboard/community_signals.py`)

- `community_reactions(id INTEGER PK, email TEXT, content_id INTEGER, reaction TEXT, created_at TEXT, UNIQUE(email, content_id, reaction))` — one row per (member, content item, reaction type). Toggling adds or removes the row.
- `community_signals(id INTEGER PK, email TEXT, target_type TEXT, target_key TEXT, signal TEXT, created_at TEXT, UNIQUE(email, target_type, target_key))` — a member's like/block on a topic or a person. One row per (member, target); setting a new signal on the same target replaces the old (upsert). Clearing deletes the row.
- `REACTIONS = ["helpful", "inspiring", "this_is_me"]` (fixed set). `TARGET_TYPES = ["topic", "person"]`. `SIGNALS = ["like", "block"]`.
- Functions (pure sqlite, emails lowercased, no app imports):
  - `init_signal_tables(cx)`
  - `toggle_reaction(cx, email, content_id, reaction) -> bool` (returns True if now on, False if just removed)
  - `reaction_counts(cx, content_id) -> {reaction: count}` — aggregate only, **never returns emails**
  - `my_reactions(cx, email, content_id) -> [reaction]` — the caller's own reactions on one item
  - `set_signal(cx, email, target_type, target_key, signal)` — upsert like/block on a target
  - `clear_signal(cx, email, target_type, target_key)`
  - `my_signals(cx, email) -> {"likes": [{target_type, target_key}], "blocks": [...]}` — the caller's own only

### 2. Routes (portal-token gated, any member)

All authed via `_evox_ident(cx, token)` (bad token → 404 `{"error":"not_found"}`). Gate is **any logged-in member** (free or paid) — signals feed each member's own (later, tier-capped) curation. Validation rejects unknown reaction/target_type/signal values with 400.

- `POST /api/community/react {content_id, reaction}` → validate `reaction in REACTIONS` and the content exists + is published; toggle; return `{ok, on: bool, counts: {reaction: count}}`.
- `GET /api/community/reactions?content_id=…` → `{counts: {reaction: count}, mine: [reaction]}` (aggregate counts + the caller's own toggles; no identities).
- `POST /api/community/signal {target_type, target_key, signal}` → validate `target_type in TARGET_TYPES`, `signal in SIGNALS`; `set_signal`; return `{ok}`. (A `signal` of `"none"`/null clears via `clear_signal`.)
- `GET /api/community/signals` → `{likes: [...], blocks: [...]}` (the caller's own topics/people only).

Content-visibility note: reactions are allowed on any *published* content id; a free member cannot watch a paid full item anyway (Layer A withholds its `video_ref`), and a reaction is a harmless private-plus-aggregate signal, so tier is not re-checked here. Counts never leak identity, so no privacy exposure.

### 3. Member surface (extend `static/community.html`)

- On each content card: the reaction set as small toggle buttons showing aggregate counts and the member's own on/off state (fetch `GET /api/community/reactions` per item or fold counts into the library payload; posting toggles via `POST /api/community/react`).
- On each topic tag: a like / block affordance (`POST /api/community/signal`), reflecting the member's current signal.
- A small "Your interests" section listing the member's liked and blocked topics with a way to clear each (`GET /api/community/signals` + clear).
- Copy: no em dashes, no ALL CAPS. Counts and affordances are quiet and warm, not gamified-loud.

## Config

No new env. Reuses the member gate, the Layer A store, and the existing `/community` page.

## Copy guidance

Client-facing copy on reactions and interests: no em dashes, no ALL CAPS. The reaction labels are gentle ("Helpful", "Inspiring", "This is me"). Nothing shames a member for blocking; blocking is framed as tuning their own experience.

## Testing

- Pure/sqlite (`dashboard/community_signals.py`): `toggle_reaction` on then off (row added then removed); `reaction_counts` aggregates and returns NO emails; `my_reactions` returns only the caller's; `set_signal` upsert (like then block on the same target replaces, one row); `clear_signal` deletes; `my_signals` splits likes/blocks and returns only the caller's.
- Route/api: react toggle returns `on` + counts; `GET reactions` returns aggregate counts + `mine` with no identity fields; unknown reaction → 400; signal set/clear round-trips; unknown target_type/signal → 400; bad token → 404; a second member's reactions/signals are never visible to the first (privacy).
- Regression: Layer A (`/api/community/library`, the store, publish) untouched; the member gate reuse is read-only; EVOX/consult/triage/masterclass/onboarding untouched.
- Go-live: as a member, react to an out-take and see the count rise; like a topic and block another; confirm a different member cannot see who reacted (counts only) and cannot see the first member's likes/blocks.

## Deferred (future Community slices)

- **Layer C:** the curated "for you" feed built from these signals, opt-in like-minded-member introductions (honoring the privacy line), and the community-aware AI chat.
- Reactions/likes on *people* become meaningful only once C surfaces people; Layer B stores person-target signals but the person-facing UI lands with C.
- Moderated comments, member-to-member chat, and any public-post surface (explicitly out of Community scope per Glen).
