# Community — peer matching v3: co-ranked blend (replace the gap-filler) — Design

**Date:** 2026-07-05
**Status:** Approved in brainstorm with Glen 2026-07-05 (replace the gap-filler with a co-ranked blend; strong semantic can outrank a lone shared topic; 2+ shared topics always win; `score = len(shared) + PEER_BLEND_WEIGHT · cosine` with W=1.75; keep the 0.80 cosine floor for zero-shared candidates; why-line stays honest and follows the winner's actual shared topics).
**Repo:** deploy-chat

**Relates to / reuses:**
- Peer matching v2 (PR #626, just shipped): the gap-filler `_peer_semantic_candidate` (app.py) — THIS slice replaces it. `PEER_SEMANTIC_MIN_COSINE` (0.80) is kept as the zero-shared floor. `_member_interest_vec` (get-or-build cached vector, `member_interest` / `COMMUNITY_FEED_MODEL`), `community_feed.cosine`.
- `dashboard/peer_connect.py`: `eligible_candidates(cx, me, is_paid)` (the exclusion pool), `liked_topics`, `blocked_topics`, `member_ref`, `next_candidate` (kept ONLY for `peer_state`'s cheap `has_proposal` hint), `is_opted_in`.
- `_peer_ident_paid`, `_is_paid_member`, `_evox_ident`, `_db_lock`, `LOG_DB`.
- [[project_pb_to_illtowell_evox]], the peer-matching privacy line.

## Context and boundary

v2 shipped semantic matching as a **gap-filler**: exact shared-topics always won, and interest-vector similarity only ran when the exact matcher found nobody. Glen has decided to go further: a **co-ranked blend** where a genuinely aligned semantic match can outrank a shallow single-topic overlap, while still guaranteeing that a candidate with multiple shared topics (the clearest, most explainable match) always wins.

**The blend (load-bearing):** over the full eligible pool, `score = len(shared_topics) + PEER_BLEND_WEIGHT · cosine`. With `PEER_BLEND_WEIGHT = 1.75`: a zero-shared semantic candidate (score `1.75·cos_sem`, `cos_sem ≥ 0.80` floor → 1.40 to 1.75) can outrank a lone-shared candidate (score `1 + 1.75·cos_lone`) whenever `cos_sem − cos_lone > 1/1.75 ≈ 0.57` — i.e. a deeply-aligned peer beats a shallow single-overlap whose own vector is not close. Any candidate sharing 2+ topics (score ≥ 2.0) tops every pure-semantic candidate (max ≈ 1.75), guaranteeing multi-overlap always wins. Among several exact matches, cosine now breaks ties sensibly. A zero-shared candidate qualifies only if `cosine >= PEER_SEMANTIC_MIN_COSINE` (0.80) — the floor keeps a barely-related stranger out.

**Why-line honesty (unchanged UI):** the winning candidate carries its real `shared_topics`. The portal already branches on empty-vs-nonempty `shared_topics` — a winner with shared topics shows "resonates with liver detox and sleep"; a semantic-only winner shows "You seem to be walking a similar path." No frontend change.

**Privacy (unchanged from v1/v2):** anonymous until mutual — the candidate payload is `{member_ref, shared_topics, semantic}` with no email/name/vector/score. Every exclusion (self, non-opted, matched, acted, skipped-me, person-blocked, non-paid) is applied by `eligible_candidates` before any embedding. Connect/reveal/thread unchanged (`resolve_ref`).

## Scope

**Replace the gap-filler branch in `/api/peer/proposal` with one blended ranking over the eligible pool.** A route-layer helper rewrite (`_peer_semantic_candidate` → `_peer_blended_candidate`), the proposal rewire (drop the "exact first, else semantic" two-step for a single blend call), one new config constant, and updated/added tests. No store change (the exact `next_candidate` stays as the `has_proposal` hint), no frontend change.

**Non-goals / deferred:** a UI-exposed weight slider; naming the nearest shared theme for a semantic winner; a "why this ranking" explainer; per-member weight personalization; short-circuiting the embed when a 2+ exact match already guarantees the top (a possible perf optimization — YAGNI at the current pool size, vectors are cached).

## Components

### 1. Blended candidate helper (`app.py`) — replaces `_peer_semantic_candidate`

`_peer_blended_candidate(cx, me, pool) -> {member_ref, shared_topics, semantic}|None`:
- `PEER_BLEND_WEIGHT = float(os.environ.get("PEER_BLEND_WEIGHT", "1.75"))`; keep `PEER_SEMANTIC_MIN_COSINE` (0.80).
- `my_liked = peer_connect.liked_topics(me) - peer_connect.blocked_topics(me)`.
- `my_vec = _member_interest_vec(cx, me, sorted(my_liked))` (may be `[]` — then cosine terms are 0; a member with no topics AND no vector yields no qualifying candidate).
- For each `n` in `pool`:
  - `shared = my_liked & (liked_topics(n) - blocked_topics(n))`.
  - `cos = community_feed.cosine(my_vec, _member_interest_vec(cx, n, sorted(liked_topics(n))))` (cosine returns 0.0 on an empty vector).
  - **Qualify** only if `shared` is non-empty OR `cos >= PEER_SEMANTIC_MIN_COSINE`.
  - `score = len(shared) + PEER_BLEND_WEIGHT * cos`.
  - Track the max by `(score, then member_ref(n) lexicographic)`.
- Return the winner as `{member_ref, shared_topics: sorted(shared_of_winner), semantic: len(shared_of_winner) == 0}` or None.
- Entire body wrapped in try/except → None (best-effort; never raises, never 500s).

### 2. Rewire `/api/peer/proposal` (`app.py`)

Replace:
```
cand = next_candidate(...)
if cand is None:
    cand = _peer_semantic_candidate(cx, email, eligible_candidates(...))
```
with a single blended call:
```
pool = peer_connect.eligible_candidates(cx, email, is_paid=_is_paid_member)
cand = _peer_blended_candidate(cx, email, pool)
return {"candidate": cand}
```
Auth/eligibility gates unchanged (`_evox_ident`→404, free/non-opted → `{candidate: None}`). `peer_interest`/`peer_state`/peer-thread routes unchanged; `next_candidate` (exact) remains only in `peer_state`'s `has_proposal` hint.

### 3. Config

- New `PEER_BLEND_WEIGHT` (default 1.75). Keep `PEER_SEMANTIC_MIN_COSINE` (default 0.80). No other new env.

## Data flow

1. `/api/peer/proposal` builds the eligible pool (all exclusions applied).
2. For each candidate, compute shared-topic set + interest-vector cosine (vectors lazily built once and cached in `member_interest`, shared with the feed).
3. Qualify (≥1 shared topic OR cosine ≥ 0.80), score `len(shared) + 1.75·cosine`, return the top anonymized. The winner's `shared_topics` drive the why-line (topic line, or the soft "similar path" line when empty).
4. Connect/reveal/thread identical to v1/v2.

## Error handling

- `_peer_blended_candidate` is best-effort: any embedding/DB failure → None (empty card, never 500).
- A member with no vector and no shared topics yields no candidate.
- The 0.80 floor prevents a weak zero-topic match; below it and with no shared topic, a candidate does not qualify.
- All exclusions run in `eligible_candidates` before any embedding, so no blocked/skipped/matched/non-paid member is embedded or offered.

## Config / cost note

The blend embeds every eligible candidate (not only when exact is empty, as the gap-filler did). Vectors are cached in `member_interest` (one build per member, ever), so steady-state cost is cosine math over a small paid pool. The first proposal after new opt-ins builds those members' vectors lazily. Acceptable at current scale; a "skip embed when a 2+ exact match already tops" optimization is deferred.

## Testing

- **Blend ranking (mock `_member_interest_vec` with controlled vectors):**
  - A candidate with 0 shared topics but high cosine (≥ ~0.6) OUTRANKS a candidate with 1 shared topic but low cosine — proves strong semantic beats a lone overlap.
  - A candidate with 2 shared topics OUTRANKS any pure-semantic candidate (even cosine ~0.99) — proves multi-overlap always wins (`2.0 > 1.75·1.0`).
  - Among two candidates who each share 1 topic, the higher-cosine one wins (cosine tiebreak).
  - A zero-shared candidate with cosine < 0.80 does NOT qualify (floor holds); if it is the only candidate → `candidate: null`.
- **Why-line / payload:** a semantic-only winner returns `shared_topics: []` + `semantic: true`; an exact/blended winner returns its real `shared_topics` (non-empty) + `semantic: false`. Assert no email/name/score/vector in any payload.
- **Update the v2 gap-filler test** `test_exact_overlap_never_triggers_semantic` → the blend DOES embed everyone, but an exact (shared-topic) candidate still wins the top slot; rename/rewrite to assert the exact candidate is returned with its `shared_topics` (the old "must not embed" premise no longer holds under a blend).
- **Regression:** `peer_interest`/`peer_state`/peer-thread routes untouched; `test_peer_match_api.py`, `test_peer_thread_api.py`, `test_peer_connect_store.py`, `test_peer_eligible_pool.py` still green (the store is unchanged).
- **Privacy:** every exclusion applied before embedding; anonymous payload.
- **Go-live:** a pair with one coincidental shared topic vs a pair with deep semantic alignment (no exact overlap) — confirm the deeply-aligned peer is surfaced; a pair sharing 2+ topics always leads.

## Deferred (later)

- Skip-embed optimization when a 2+ exact match already tops; UI weight control; nearest-theme naming for semantic winners; per-member weight; threshold/weight tuning from live match-acceptance data.
