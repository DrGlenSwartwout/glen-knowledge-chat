# Community — peer matching: person-block action — Design

**Date:** 2026-07-05
**Status:** Approved in brainstorm with Glen 2026-07-05 (a permanent "never show me this person again" as a third action on the anonymous proposal; nothing new at the thread; mutual invisibility; supersedes a prior skip's cool-off; no unblock UI).
**Repo:** deploy-chat

**Relates to / reuses:**
- Peer matching v1-v3 + skip cool-off: `app.py` `peer_interest` (adds a `block` kind), `dashboard/peer_connect.py` `resolve_ref`/`member_ref`, `_person_blocked` (already excludes on a `community_signals` person-block, both matcher passes, both directions), `eligible_candidates`.
- `dashboard/community_signals.py`: `set_signal(cx, email, target_type, target_key, signal)`, `TARGET_TYPES=['topic','person']`, `SIGNALS=['like','block']`. The person-block plumbing already exists end to end — this slice only adds the trigger.
- Frontend `renderPeerProposal`/`peerInterest` in `static/client-portal.html` (Connect / Not now → gets a third "Not a fit").

## Context and boundary

The signal store supports a `person` block and the peer matcher already honors it (`_person_blocked` excludes a member from proposals permanently, in both the fresh and stale-skip passes), but no control ever writes one. Meanwhile the only proposal exits today are **Connect** and **Not now** — and after the cool-off work, "Not now" (skip) can resurface a person after 30 days. There is no way to say "definitely not this person, ever."

This slice adds that: a third proposal action that writes a person-block, permanently excluding the candidate (surviving the cool-off). Post-match blocking is unchanged — the peer thread's existing Block ends the match, which already excludes the pair from future proposals.

**Key properties:**
- **Permanent + supersedes skip:** a person-block excludes the candidate regardless of any prior skip's cool-off (the block check is independent of the interest row).
- **Mutual invisibility (intended, falls out for free):** `eligible_candidates` excludes when `_person_blocked(me, n) OR _person_blocked(n, me)`. So once I block someone, that OR also removes me from *their* proposals — after a block, neither of us is ever shown the other.
- **Anonymous:** the block is keyed on the candidate's opaque `member_ref` (`target_key=member_ref(target)`); no email/name is stored or shown. Because proposals are anonymous, there is no practical unblock UI — a block is a deliberate, unreviewable "never."

## Scope

**A `kind:"block"` branch in `POST /api/peer/interest` (writes a person-block signal) + a "Not a fit" action on the proposal card + tests.** No matcher change, no thread change, no store change, no new env.

**Non-goals / deferred:** an unblock/review-your-blocks screen; blocking from anywhere but the proposal; a block reason/report; rate-limiting blocks.

## Components

### 1. `POST /api/peer/interest` — add the `block` kind (`app.py`)

- Accept `kind in ("connect", "skip", "block")` (was connect/skip).
- After auth/eligibility and `resolve_ref` (404 on a stale/forged ref — unchanged), branch BEFORE the connect/skip `record_interest` logic:
  ```
  if kind == "block":
      community_signals.set_signal(cx, email, "person", peer_connect.member_ref(target), "block")
      return {"ok": True, "matched": False}
  ```
  (`target_key = member_ref(target)` is the canonical ref the matcher checks; `resolve_ref` already validated the ref maps to an opted-in member. No `peer_interest` row is written for a block. `set_signal` runs under the route's existing `_db_lock`.)
- connect/skip behavior is unchanged. The response shape `{ok, matched}` is unchanged (block returns `matched:false`, like a skip).

### 2. Proposal surface (`static/client-portal.html`)

- In `renderPeerProposal`, add a third quiet action **"Not a fit"** beside Connect / Not now. On click: a light confirm ("You will not be shown this person again.") then `peerInterest(host, candidate.member_ref, "block", ...)`.
- `peerInterest` already handles `connect`/`skip`; `block` follows the skip path in the client (on `ok`, load the next proposal). On a non-ok/error, re-enable and show a plain retry message. All strings via `textContent`; no em dashes, no ALL CAPS.

## Data flow

1. Member sees an anonymous proposal (`{member_ref, shared_topics, semantic}`).
2. Taps **Not a fit** → confirm → `POST /api/peer/interest {member_ref, kind:"block"}`.
3. The route resolves the ref → writes `community_signals` person-block (`email`, `person`, `member_ref`, `block`) → returns `{ok, matched:false}`.
4. The card loads the next proposal. That person is now excluded from this member's proposals forever (both fresh and cool-off passes), and — via the `_person_blocked` OR — this member is excluded from that person's proposals too.

## Error handling

- `_evox_ident`→404, `_is_paid_member`/opted-in→403, bad kind→400, stale/forged ref→404 — all unchanged from the existing route.
- `set_signal` is a plain upsert under the existing `_db_lock`; re-blocking the same ref is idempotent (one row per email+target_type+target_key).
- A block on someone already matched (unlikely from an anonymous proposal, since matched people are excluded from proposals) is harmless — the person-block simply adds to the existing exclusion.

## Config

- No new env. Reuses the member-token auth + the community_signals store.

## Testing

- **Route:** `kind:"block"` on a valid ref writes a `community_signals` row (`target_type='person'`, `target_key=member_ref(target)`, `signal='block'`) and returns `{ok, matched:false}`; a stale/forged ref → 404; a free/non-opted member → 403; a bad kind → 400.
- **Matcher effect (integration):** after a member blocks candidate N, N is not proposed to the member even when N would otherwise be the only/best candidate (fresh AND stale-skip passes); AND the blocking member is not proposed to N (mutual). A person-block supersedes a prior skip (a stale skip that would resurface stays excluded once blocked).
- **Privacy:** no email/name in the block request/response; the stored `target_key` is the opaque `member_ref`, not an email.
- **Frontend:** parse check clean; "Not a fit" posts `kind:"block"` and advances to the next proposal; copy has no em dashes/ALL CAPS.
- **Go-live:** as a paid opted-in member, block a proposed candidate, confirm they never reappear (and, with a second test member, that the blocker no longer appears to the blocked person).

## Deferred (later)

- A review-your-blocks / unblock screen; block-with-reason or report; blocking from other surfaces; rate-limiting.
