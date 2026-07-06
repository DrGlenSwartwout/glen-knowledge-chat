# Community — peer matching: anchored why-line for a semantic winner — Design

**Date:** 2026-07-05
**Status:** Approved in brainstorm with Glen 2026-07-05 (anchor the semantic why-line on the ASKER's OWN interest — never the winner's — to stay honest and privacy-safe; deterministic first liked topic; copy "Another member exploring themes close to your interest in <topic>.").
**Repo:** deploy-chat

**Relates to / reuses:**
- Peer matching co-ranked blend (`app.py` `_peer_blended_candidate`), `peer_connect.liked_topics`/`blocked_topics`, the proposal frontend `renderPeerProposal` in `static/client-portal.html`.

## Context and boundary

A semantic-only winner (co-ranked blend, `shared_topics == []`) currently shows the generic "You seem to be walking a similar path." Naming a specific theme is tempting but was found unsafe: a semantic match shares NO topic, so naming the winner's topic would leak the winner's private interest pre-mutual, and naming an inferred "shared theme" would be fabricated. The one honest, privacy-safe framing is to anchor the line on the ASKER's OWN interest — a topic the asker already likes, never the winner's — which reveals nothing new and states only a truth.

This slice replaces the generic line (for semantic winners only) with an anchored one that names one of the asker's own liked topics.

**Privacy invariant (load-bearing):** the added field carries ONLY the asker's own liked topic (their own data, returned to them). It NEVER carries the winner's topics, name, email, or any of the winner's private signals. Exact-match lines are unchanged (they already name the shared topic).

## Scope

**One payload field on a semantic winner (`reason_topic` = the asker's first liked topic) + a three-way frontend why-line + tests.** No matcher change, no store change, no new env.

**Non-goals:** naming the winner's topic; an inferred/LLM shared theme; ranking the asker's topics by relevance (would need per-topic embeddings — YAGNI; deterministic first-sorted is enough).

## Components

### 1. `_peer_blended_candidate` payload (`app.py`)

`my_liked = liked_topics(me) - blocked_topics(me)` is already computed. For the returned winner, when it is semantic-only (`shared_topics == []`) AND `my_liked` is non-empty, add `reason_topic = sorted(my_liked)[0]` (deterministic, stable across refreshes):
```
shared = best[2]
out = {"member_ref": best[1], "shared_topics": shared, "semantic": len(shared) == 0}
if not shared and my_liked:
    out["reason_topic"] = sorted(my_liked)[0]
return out
```
An exact winner (non-empty `shared_topics`) carries NO `reason_topic`. (A semantic winner always has at least one liked topic — the asker's interest vector is built from them — so `reason_topic` is present in practice; the `if my_liked` guard is a fail-safe.)

### 2. Proposal why-line (`static/client-portal.html`)

`renderPeerProposal`'s `desc.textContent` becomes three-way (via `textContent`):
- `shared_topics` non-empty → "A member who also resonates with " + topics.join(" and ")  (unchanged).
- else `candidate.reason_topic` present → "Another member exploring themes close to your interest in " + reason_topic + "."
- else → "You seem to be walking a similar path."  (unchanged fallback).

No em dashes, no ALL CAPS; the topic is server-supplied but set via `textContent`.

## Data flow

1. The blend picks a semantic winner (0 shared topics, cosine >= floor).
2. The route payload includes `reason_topic` = the asker's own first liked topic.
3. The card renders the anchored line naming the asker's own interest. Connect/skip/block/thread are unchanged.

## Error handling

- If the asker somehow has no liked topics (edge case; a semantic winner shouldn't occur then, since their vector would be empty), `reason_topic` is omitted and the frontend falls back to the generic line.
- `_peer_blended_candidate` stays best-effort (the new lines are inside the existing try/except).

## Config

- No new env.

## Testing

- **Route/store:** a semantic winner payload includes `reason_topic` equal to the asker's first sorted liked topic, and does NOT include the winner's topics/email/name (assert the winner's liked-topic string is absent from the JSON). An exact winner payload has NO `reason_topic` (still carries its `shared_topics`). A member with no liked topics → no `reason_topic` (fail-safe).
- **Privacy:** the `reason_topic` is one of the ASKER's own liked topics, never the winner's (seed the two members with disjoint topics and assert the field equals an asker topic).
- **Frontend:** parse check clean; a candidate with `reason_topic` and empty `shared_topics` renders the anchored line; a candidate with `shared_topics` renders the topic line; a candidate with neither renders the generic line.

## Deferred (later)

- Ranking the asker's topics by relevance to the winner (needs per-topic embeddings); an inferred shared-theme line; per-member copy variation.
