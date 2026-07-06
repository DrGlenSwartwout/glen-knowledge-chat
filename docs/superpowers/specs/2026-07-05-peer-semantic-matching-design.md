# Community — peer matching v2: semantic interest-vector gap-filler — Design

**Date:** 2026-07-05
**Status:** Approved in brainstorm with Glen 2026-07-05 (gap-filler, not co-ranked: exact shared-topics stay primary; semantic only fills an otherwise-empty card; soft why-line "You seem to be walking a similar path."; conservative similarity threshold; member vectors built once and cached in the feed's `member_interest`; one candidate at a time; downstream reveal/thread identical to an exact match).
**Repo:** deploy-chat

**Relates to / reuses:**
- Peer matching v1 (PR #623/#624): `dashboard/peer_connect.py` (opt-in pool, `next_candidate` exact-shared-topics matcher, `member_ref`, `resolve_ref`, `liked_topics`, all the exclusion logic). Member routes `/api/peer/*`.
- Community feed C1 (`dashboard/community_feed.py`): `cosine(a,b)`, `build_interest_text(journal, liked, chat)`. `dashboard/community.py`: `member_interest(email PK, vec, model, built_at)` store — `get_member_interest(cx, email, model)`, `set_member_interest(cx, email, vec, model)`. The SAME embedding model constant the feed uses (so peer vectors and feed vectors are one cache).
- `app.py`: `embed(text)` (the embedding helper), `_evox_ident`, `_is_paid_member`, `_db_lock`, `LOG_DB`.
- [[project_pb_to_illtowell_evox]], the peer-matching privacy line (anonymous until mutual; no email in payloads; blocks/skips/matches/non-paid excluded).

## Context and boundary

Peer matching v1 proposes the opted-in paid member who shares the most **exact** liked-topics. A member with unusual or few likes, or who has already connected/skipped everyone who shares a topic, sees an empty card. This slice adds a **semantic gap-filler**: when no eligible candidate shares an exact topic, surface the eligible member whose **interest vector** (built from their liked topics, reused from the feed) is closest to the member's own, above a conservative threshold, with an honest softer why-line.

**Design stance (load-bearing):** semantic matching is a FALLBACK, never a co-ranked signal. Exact shared-topic candidates are always returned first, unchanged, with their clear why-line. Semantic only runs when the exact matcher returns nothing. This preserves every trusted v1 behavior and only adds reach.

**Privacy invariants (unchanged from v1):** the semantic candidate is returned anonymously (`{member_ref, shared_topics: [], semantic: true}`) — no email, no name, no vector. All v1 exclusions (self, non-opted, matched, acted, skipped-me, person-blocked, non-paid) apply identically. Downstream (connect → mutual reveal → thread) is byte-identical to an exact match; a semantic candidate is just another opted-in member resolvable by `resolve_ref`.

## Scope

**Exact matcher unchanged → when it returns nothing, rank the eligible pool by interest-vector cosine → return the closest above threshold with a soft why-line → connect/reveal/thread identical to v1.** A small refactor of `peer_connect.py` to expose the eligible pool, an `app.py` semantic-fallback helper (embedding-dependent, so it lives in the route layer, not the pure store), wiring into `/api/peer/proposal`, and a one-line why-line branch in the portal card.

**Deferred / non-goals:** co-ranked blended scoring; naming the nearest shared theme in the why-line (requires vector→topic inversion); semantic matching in `peer_state`'s cheap `has_proposal` hint (kept exact-only to avoid embedding on every portal load); rebuilding a member's vector when their likes change (the feed's existing staleness behavior applies — a cached vector is reused until the model changes; a periodic rebuild is a separate future item).

## Components

### 1. Expose the eligible pool (`dashboard/peer_connect.py`)

Refactor the exclusion logic out of `next_candidate` into a reusable helper, with no behavior change to `next_candidate`:
- `eligible_candidates(cx, me, is_paid=None) -> [email]` — every opted-in member that passes ALL v1 exclusions (self; existing `peer_matches` row; `interest_kind(me,n)` set; `interest_kind(n,me)=='skip'`; person-blocked either direction; `is_paid` predicate when provided) — WITHOUT the shared-topic requirement. `next_candidate` calls this, then keeps only those with a non-empty shared-topic set and ranks by count (identical output to today).

This gives the route the exact-less eligible pool for semantic ranking. The pure store gains no embedding dependency.

### 2. Semantic fallback (`app.py`, route layer — embedding lives here)

- `PEER_SEMANTIC_MIN_COSINE = float(os.environ.get("PEER_SEMANTIC_MIN_COSINE", "0.80"))` — conservative floor so a gap-filler is shown only when the two are genuinely close.
- `_peer_member_vec(cx, email) -> list|None` — get-or-build a member's interest vector, cached in the feed's `member_interest`:
  1. `v = community.get_member_interest(cx, email, MODEL)`; if present, return it.
  2. `topics = peer_connect.liked_topics(cx, email)`; if empty → return None (nothing to vectorize).
  3. `vec = embed(community_feed.build_interest_text([], sorted(topics), []))`; `community.set_member_interest(cx, email, vec, MODEL)` (under `_db_lock`); return `vec`. `MODEL` = the same embedding-model constant the feed uses.
- `_peer_semantic_candidate(cx, me, pool) -> {member_ref, shared_topics: [], semantic: True}|None` — build `my_vec = _peer_member_vec(me)` (None → no candidate); for each `n` in `pool`, `v = _peer_member_vec(n)` (skip None); `score = community_feed.cosine(my_vec, v)`; pick the max with `score >= PEER_SEMANTIC_MIN_COSINE`; return it anonymized (member_ref, empty shared_topics, `semantic: True`), else None. Best-effort: any embedding error → return None (the card just stays empty, never 500s).

### 3. Wire into `/api/peer/proposal` (`app.py`)

Unchanged auth/eligibility. Then:
```
cand = peer_connect.next_candidate(cx, email, is_paid=_is_paid_member)   # exact, unchanged
if cand is None:
    pool = peer_connect.eligible_candidates(cx, email, is_paid=_is_paid_member)
    cand = _peer_semantic_candidate(cx, email, pool)                     # gap-filler
return {"candidate": cand}
```
`peer_interest` (connect/skip) is UNCHANGED — a semantic candidate's `member_ref` resolves via the existing `resolve_ref` (it is an opted-in member), so mutual detection, reveal, and thread creation are identical. `peer_state`'s `has_proposal` stays exact-only (a cheap hint); the portal card fetches `/proposal` directly when opted-in, so a semantic-only member still sees their card.

### 4. Portal why-line (`static/client-portal.html`)

In the peer proposal render, when `candidate.shared_topics` is empty (a semantic match), show the soft line "You seem to be walking a similar path." instead of the "A member who also resonates with ..." topic line. Everything else (Connect / Not now / anonymity) is unchanged. Copy via `textContent`; no em dashes, no ALL CAPS.

## Data flow

1. `/api/peer/proposal` runs the exact matcher (v1). If it returns a candidate → return it with the topic why-line.
2. Only if exact returns nothing: fetch the eligible pool, lazily build/cache each member's interest vector (reusing the feed's `member_interest`), cosine-rank, and return the closest above threshold with `semantic: true` + empty shared_topics.
3. The card shows the soft why-line for a semantic candidate; Connect/skip and the mutual-reveal-then-thread flow are identical to v1.

## Error handling

- `_peer_semantic_candidate` and `_peer_member_vec` are best-effort: an `embed()` failure or a missing vector returns None (the card is simply empty), never a 500.
- Vector writes (`set_member_interest`) happen under `_db_lock`.
- A member with no liked topics has no vector and is never a semantic candidate nor gets a semantic proposal.
- All v1 exclusions are applied by `eligible_candidates` before any embedding work, so no blocked/skipped/matched/non-paid member is ever embedded-and-ranked.
- Threshold gate: below `PEER_SEMANTIC_MIN_COSINE`, return None (an empty card is correct — better than a weak filler).

## Config

- Optional `PEER_SEMANTIC_MIN_COSINE` (default 0.80). No other new env. Reuses the feed's embedding model + `embed()`.

## Testing

- **Store (`eligible_candidates`):** returns the full eligible pool (all v1 exclusions applied, NO shared-topic filter); excludes self/non-opted/matched/acted/skipped-me/person-blocked/non-paid; `next_candidate` output is unchanged (still exact-only, ranked by shared count).
- **Semantic fallback (mock `embed` to return controlled vectors):** when an exact overlap exists, `/proposal` returns the EXACT candidate (semantic not consulted); when no exact overlap exists but two members have vectors with cosine ≥ threshold, `/proposal` returns `{member_ref, shared_topics: [], semantic: true}` (anonymous — assert no email/name); below threshold → `candidate: null`; a member with no liked topics → no semantic candidate; a blocked/skipped/non-paid member is never the semantic candidate; the returned member_ref resolves and a mutual connect opens a thread exactly as in v1.
- **Vector reuse:** `_peer_member_vec` populates `member_interest` and reuses an existing row (assert `embed` is called once per member, then cached; a member with a feed-built vector is reused without re-embedding).
- **Frontend:** a semantic candidate (empty shared_topics) renders the soft why-line, not a topic list; parse check clean.
- **Privacy regression:** no email/name in a semantic proposal payload; the semantic path applies every v1 exclusion.
- **Go-live:** two paid members with related-but-not-identical likes (no exact overlap), both opted in, each sees the other as a "similar path" proposal; both connect → reveal + thread. A pair with an exact shared topic still sees the exact topic line (semantic never shown).

## Deferred (later)

- Co-ranked blended scoring; naming the nearest shared theme in the semantic why-line; periodic member-vector rebuild when likes change; semantic-aware `has_proposal`; tuning the threshold from live match-acceptance data.
