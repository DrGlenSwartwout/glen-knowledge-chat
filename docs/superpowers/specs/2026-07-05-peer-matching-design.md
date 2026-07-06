# Community — peer matching (like-minded member intros) — Design

**Date:** 2026-07-05
**Status:** Approved in brainstorm with Glen 2026-07-05 (paid-only + upgrade tease; passive curated proposals; shared-liked-topics match with a plain why line; anonymous until mutual; reveal first name only; multiple concurrent connections; skipped candidates not re-proposed).
**Repo:** deploy-chat

**Relates to / reuses:**
- Community signal layer (`dashboard/community_signals.py`): `community_signals(email, target_type ['topic'|'person'], target_key, signal ['like'|'block'])`. Liked topics = `target_type='topic' AND signal='like'`; blocked topics = `signal='block'`. `my_signals`, `set_signal`, `_lc`.
- Coaching slice 3 (`dashboard/coach_threads.py`): the generic 1:1 thread + moderation + epochs. Peer matches open a `source='peer'` thread and inherit report/block/owner-moderation. Reused as-is with a slot convention (see below).
- `_evox_ident(cx, token)` (member auth; `.email`), `_is_paid_member(email)` (eligibility), `client_portal.get_portal_content_by_email` (first name at reveal), `send_evox_email` (reveal nudge), `_portal_console_ok` (owner), `_db_lock`, `LOG_DB`, `PUBLIC_BASE_URL`.
- Owner console `/api/console/coach-threads*` already lists ALL threads by `source` — peer threads surface there automatically. The owner **unmatch** route needs a `source` branch (see Components 5).
- [[project_pb_to_illtowell_evox]], [[project_chat_surface_model]] (TOS/paywall gating), the HARD community privacy line: never surface a member to others, or expose their words, without per-item opt-in.

## Context and boundary

The last Community piece. Coaching (slices 1-3) pairs a member with a cert-student coach. Peer matching pairs two **members** who are on a similar path — a symmetric, anonymous, double-opt-in intro that ends in the same 1:1 thread. It reuses the shipped thread + moderation channel (that dependency, previously the reason this arc was on hold, is now done).

**Privacy invariants (load-bearing):**
- Only **opted-in paid members** are matchable. A member is never surfaced to anyone until they opt in.
- Proposals are **anonymous** — a candidate is shown as a stable opaque `member_ref` + the shared topics, never a name or email — until BOTH members independently express connect-interest.
- There is **no "you were passed on" signal**, ever. A skip is private to the skipper.
- Person/topic **blocks**, prior **skips**, already-**matched** pairs, and pairs with a **blocked peer thread** are excluded from proposals.
- No email in any payload; **first name only** at reveal (via `client_portal`).
- Peer threads inherit slice-3 thread privacy (no counterpart email, blocked history owner-only, all bodies rendered via `textContent`).

**Non-goals (YAGNI / deferred):** semantic interest-vector matching (v1 is shared-liked-topics only); group intros; scheduled/batched match runs (proposals computed on demand); a "you have N candidates" teaser count; re-proposing skipped candidates after a cool-off; person-block UI (the signal store supports it; peer matching honors it forward-compatibly, but building the block-a-person UI is out of scope).

## Scope

**Opt in → see one anonymous like-minded candidate → connect or skip → on mutual connect, reveal first names + open a peer thread → converse with report/block, owner-moderated.** One store, member routes (opt-in / proposal / interest / connections), thin peer-thread routes (member↔member) over the slice-3 store, one owner-route branch, and the portal surface.

## Components

### 1. Store (`dashboard/peer_connect.py`) — pure sqlite, no app imports

- `member_ref(email) -> sha256(lower(email))[:16]` (anonymized handle; same construction as `coach_connect.coach_ref`; used for anonymous proposals and forward-compatible person-block keying).
- Tables (`init_peer_tables(cx)`):
  - `peer_optin(member_email TEXT PRIMARY KEY, active INTEGER NOT NULL DEFAULT 1, updated_at TEXT)`.
  - `peer_interest(from_email TEXT, to_email TEXT, kind TEXT ['connect'|'skip'], created_at TEXT, UNIQUE(from_email, to_email))` — a directional intent.
  - `peer_matches(id INTEGER PK, a_email TEXT, b_email TEXT, thread_id INTEGER, status TEXT ['active'|'ended'], created_at TEXT, UNIQUE(a_email, b_email))` — the pair is normalized `a_email < b_email` so it is order-independent.
- Functions:
  - `set_optin(cx, email, active)`, `is_opted_in(cx, email) -> bool`, `opted_in_members(cx) -> [email]`.
  - `liked_topics(cx, email) -> set`, `blocked_topics(cx, email) -> set` (read `community_signals`).
  - `record_interest(cx, from_email, to_email, kind)` (upsert), `interest_kind(cx, from_email, to_email) -> str|None`.
  - `next_candidate(cx, me) -> {member_ref, shared_topics:[...]}|None` — the matcher (below).
  - `create_match(cx, a_email, b_email, thread_id)` (normalized), `match_for_pair(cx, e1, e2) -> dict|None`, `matches_for(cx, me) -> [{other_email, thread_id, status}]`, `end_match(cx, thread_id)`.
  - `resolve_ref(cx, me, member_ref) -> email|None` — resolve an anonymous ref to an email by scanning `opted_in_members` (refs are one-way hashes); returns None if the ref is not a currently-opted-in member (guards a stale/forged ref).

**The matcher `next_candidate(cx, me)`** (anonymized, privacy-filtered):
- `mine = liked_topics(me) - blocked_topics(me)`; if empty → None.
- For each other `n` in `opted_in_members` (`n != me`), skip when: a `peer_matches` row exists for the pair; `interest_kind(me, n)` is already set (I connected or skipped them); a `coach_threads` `source='peer'` thread for the pair is `blocked`; either side person-blocked the other in `community_signals` (keyed by `member_ref`).
- `shared = mine ∩ (liked_topics(n) - blocked_topics(n))`; if non-empty, candidate with `score = len(shared)`.
- Return the highest-score candidate (stable tiebreak by `member_ref`) as `{member_ref: member_ref(n), shared_topics: sorted(shared)}` — no email, no name.

### 2. Member routes (`app.py`, `_evox_ident` + `_is_paid_member` gated)

- `GET /api/peer/state` → `{eligible, opted_in, has_proposal}`; `eligible=false` (free member) drives the upgrade tease. 404 on bad token.
- `POST /api/peer/optin {active}` → paid-only (403 `not_eligible` for free); `set_optin`. Toggling off keeps existing matches.
- `GET /api/peer/proposal` → paid + opted-in: `{candidate: {member_ref, shared_topics} | null}` (anonymous). Not opted-in / free → `{candidate: null, ...}`.
- `POST /api/peer/interest {member_ref, kind}` (`connect`|`skip`) → paid + opted-in. `resolve_ref`→ target email (404 if stale). `record_interest(me→target, kind)`. On `connect`, if `interest_kind(target, me) == 'connect'` → **mutual**: open a `source='peer'` `coach_threads` thread for the pair, `create_match(a,b,thread_id)`, best-effort reveal nudge to both. Returns `{matched: bool}` (true only on a fresh mutual; never reveals the target's prior intent otherwise).
- `GET /api/peer/connections` → `[{first_name, thread_id, status}]` for the member's `peer_matches` (first name via `client_portal`, never email).

### 3. Peer thread routes (`app.py`, `_evox_ident`, member↔member over the slice-3 store)

The peer thread reuses `dashboard/coach_threads.py` with a **slot convention**: for a normalized pair (`a_email < b_email`), `a_email` occupies the store's `coach_email` slot and `b_email` the `member_email` slot; the caller's role is `'coach'` if they are `a_email`, else `'member'` (purely internal — the UI only ever shows the other person's first name). This avoids duplicating the whole thread/moderation/epoch machinery. (A future refactor may rename the store's columns to `participant_a/participant_b`; out of scope here.)
- `GET /api/peer-thread/<int:thread_id>` → resolve `_evox_ident`; confirm the caller is a participant via `peer_matches` (else 403); `mark_read` for the caller's slot-role; return `{other_first_name, status, can_post, messages}` (blocked → `messages:[]`). No email.
- `POST /api/peer-thread/<int:thread_id>/message {body}` → participant-check, 400 empty/oversized, 409 blocked, else post + best-effort nudge the other side.
- `POST /api/peer-thread/<int:thread_id>/block` → `block_thread(caller-role)` + `end_match(thread_id)` (removes the pair from each other's future proposals) + owner alert; history hidden from both (owner-only), same as coaching.
- `POST /api/peer-thread/<int:thread_id>/report {reason}` → `report_thread` + owner alert.

### 4. Member surface (`static/client-portal.html`)

A "Connect with members" card:
- **Free member:** a locked tease + upgrade nudge ("Membership opens like-minded member connections.") — no proposal, no opt-in control.
- **Paid, not opted-in:** an opt-in control ("Open to meeting like-minded members") with a one-line explanation of the anonymous, mutual-only flow.
- **Paid, opted-in:** the current anonymous proposal ("A member who also resonates with liver detox and sleep. Open to connecting?") with **Connect** / **Not now**; on a mutual `matched:true`, surface the new connection; plus a **Your connections** list (first name → opens the peer thread) and the peer thread panel (messages + compose + Report + Block), mirroring the coaching thread panel. All dynamic strings via `textContent`; no em dashes, no ALL CAPS.

### 5. Owner moderation (existing `/api/console/coach-threads*`, one branch added)

Peer threads (`source='peer'`) already appear in `list_all_threads` and the transcript route works unchanged. The **owner unmatch** route (`/api/console/coach-threads/<id>/unmatch`) currently ends a coaching `coach_requests` pairing; add a `source` branch: for `source='peer'`, `block_thread('owner')` + `peer_connect.end_match(thread_id)` (NOT `set_request_status`), then notify both. No new owner UI — the existing console panel drives it.

## Data flow

1. Paid member opts in → enters the pool.
2. `GET /api/peer/proposal` computes the top anonymous like-minded candidate on demand (privacy-filtered).
3. Member connects or skips. Skip = private, never re-proposed. Connect = a stored directional intent.
4. When both members have `connect` intents toward each other → a `source='peer'` thread opens, a `peer_matches` row is written, both get a reveal nudge, and the connection appears in each member's list with the other's first name.
5. Conversation reuses the slice-3 thread. Block ends the match + excludes the pair from future proposals; report + owner unmatch behave as in coaching (with the peer branch).

## Error handling

- Every route resolves `_evox_ident`→404 first, then `_is_paid_member`→403 (`not_eligible`) for the paid-gated ones, then body validation (a bad token never returns 400/403).
- `resolve_ref` returns None (404) for a stale/forged/non-opted ref so interest can only be recorded toward a currently-matchable member.
- Mutual-match creation runs under `_db_lock` (thread create + `create_match` + interest read in one write) so a simultaneous mutual connect from both sides cannot create two threads/matches (guarded by `peer_matches UNIQUE(a_email,b_email)` and `coach_threads UNIQUE`).
- Peer thread routes 403 a non-participant; message routes 400 empty/`>COACH_MESSAGE_MAX_CHARS`, 409 blocked.
- Nudge/owner emails best-effort (own try/except, never raise).
- Opting out does not delete existing matches or threads (they persist until blocked/unmatched).

## Config

- No new env. Reuses `CONSOLE_SECRET`, `GLEN_CONSULT_EMAIL`, `PUBLIC_BASE_URL`, `COACH_MESSAGE_MAX_CHARS`, the email sender.

## Testing

- **Store:** optin set/clear + `opted_in_members`; `liked_topics`/`blocked_topics` read `community_signals`; `next_candidate` excludes self, non-opted, already-matched, already-skipped/connected, person-blocked, and blocked-thread pairs, ranks by shared-topic count, and returns `member_ref`+shared_topics with NO email/name; empty when no shared topics; `record_interest` upsert; mutual detection via `interest_kind`; `create_match` normalizes the pair; `matches_for` resolves the other side; `resolve_ref` maps a ref to a currently-opted-in email and returns None for a stale ref.
- **Member routes (mocked `_is_paid_member`, signals seeded):** free member → `eligible:false` + opt-in 403; proposal is anonymous (assert no email/name in the JSON); connect without a reciprocal intent → `matched:false`, nothing revealed; a reciprocal connect → `matched:true`, a peer thread + match row created, both nudged; skip removes that candidate from the next proposal; `connections` returns first name never email.
- **Peer thread routes:** a non-participant member → 403; participant sees the other's first name (never email); message 400/409; block ends the match (pair no longer proposed) + hides history from both.
- **Owner branch:** a `source='peer'` thread appears in the console list; owner unmatch on a peer thread ends the `peer_match` (asserts the pair can be re-proposed only per policy) and does NOT touch `coach_requests`.
- **Privacy regression:** no email in any proposal/connections/thread payload; a non-opted or free member is never returned as a candidate; "skip" produces no signal to the skipped member.
- **Go-live:** two paid test members like an overlapping topic, both opt in, both see each other anonymously, both connect → reveal + thread; exchange a message; one blocks → connection ends, neither re-proposed.

## Deferred (later)

- Semantic interest-vector matching (reuse `community_feed` embeddings) as a second candidate source; re-proposing skipped candidates after a cool-off; a person-block UI; batched/scheduled match runs + a candidate-count teaser; generalizing the `coach_threads` column names to `participant_a/participant_b`.
