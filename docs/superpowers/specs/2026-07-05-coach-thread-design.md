# Coaching — 1:1 coaching thread + report/block/moderation (coaching arc, slice 3) — Design

**Date:** 2026-07-05
**Status:** Approved in brainstorm with Glen 2026-07-05 (owner reads every thread; member block ends the pairing + hides history from both, readable by owner; either side reports; text-only async; reply nudge to the other side; no live chat; no media).
**Repo:** deploy-chat

**Relates to / reuses:**
- Coaching slice 2 (PR #618): `dashboard/coach_connect.py` — the accepted `coach_requests` row (coach_email ↔ member_email, `status` pending/accepted/withdrawn/declined) is the matched pair this thread hangs off. Reuses `set_request_status`, `accepted_count`, `member_has_accepted`, `coach_ref`. **Needs a new helper** `accepted_pair(cx, member_email) -> {request_id, coach_email}|None` (resolve the member's single accepted coach + its request id) so the thread can find the counterpart and end the pairing on block; the coach side resolves its accepted members via a sibling `accepted_members(cx, coach_email) -> [{request_id, member_email, member_name}]`.
- Coaching slice 1 (PR #616): `dashboard/coach_directory.py` — `get_volunteer`/`list_active_full` for the coach display name/focus shown to the member.
- Auth: member = `_evox_ident(cx, token)` (portal token; `.email`, no `.name`; first name via `client_portal.get_portal_content_by_email(cx, email)["name"]` split on space). Coach/practitioner = `_practitioner_session_pid()` (token→pid) + `practitioner_email_by_id(pid)`. Owner console = `X-Console-Key == CONSOLE_SECRET` (the `_portal_console_ok` idiom + console-biofield-portal.html surface).
- `send_evox_email(to, name, subject, html, text, ics_bytes)` for the reply/block/report nudges (best-effort, never raises). `GLEN_CONSULT_EMAIL` for owner alerts.
- `_db_lock`, `LOG_DB`, sqlite `?` placeholders, `cx.row_factory = sqlite3.Row`; DATA_DIR override for tests.
- [[project_pb_to_illtowell_evox]], [[feedback_identity_merge_review]] (never expose the other party's email), [[reference_video_hosting_rumble]] (n/a — no media this slice).

## Context and boundary

Coaching slices 1-2 shipped the directory + double-opt-in matching. A member and a cert-student coach are matched when a `coach_requests` row reaches `status='accepted'`. This slice is the **place they actually talk**: a text-only, async 1:1 message thread, with the safety layer a solo practice needs — either side can report, the member can block, and the practice owner (Glen) can read and moderate every thread because his cert students are coaching his paying clients.

**The thread is designed as the shared channel** the on-hold peer-matching arc (community-c2a) will also use. It is therefore built generically: two participants each carry a **role** (`coach` / `member`), and the thread carries a **source** tag (`coaching` now, `peer` later). This slice ships the coaching source only.

**Privacy invariants (load-bearing):**
- Neither participant ever sees the other's email or any contact info — messages stay inside the thread. The coach sees the member's first name only; the member sees the coach's display name only.
- A blocked thread's message history is hidden from BOTH participants and readable only by the owner console.
- Aggregate/other-member exposure rules from the community privacy line still hold: a thread is visible only to its two participants (pre-block) and the owner — never to any third member.

**Non-goals (YAGNI / deferred):** live/real-time chat, typing indicators, websockets; media/attachments (text only); group threads (1:1 only); coach-initiated block (a coach ending a pairing is a capacity/management action, not in this slice); read receipts shown to the other party (unread is tracked server-side only for the badge + nudge); the peer (`source='peer'`) routes and member↔member auth.

## Scope

**Materialize a thread for an accepted pair → both sides post text async → reply nudge to the other side → member can block (ends pairing, hides history) → either side can report (flags for owner) → owner reads/unmatches from console.** One thread store, member thread routes, coach thread routes, owner moderation routes, and the two portal surfaces.

## Components

### 1. Thread store (`dashboard/coach_threads.py`) — pure sqlite, no app imports

Tables (lazy `init_thread_tables(cx)`):
- `coach_threads(id INTEGER PK, source TEXT, coach_email TEXT, member_email TEXT, status TEXT, blocked_by TEXT, reported INTEGER DEFAULT 0, created_at TEXT, coach_last_read_at TEXT, member_last_read_at TEXT, UNIQUE(coach_email, member_email))` — `source` ∈ {`coaching`}; `status` ∈ {`active`,`blocked`}.
- `coach_messages(id INTEGER PK, thread_id INTEGER, sender_role TEXT, body TEXT, created_at TEXT)` — `sender_role` ∈ {`coach`,`member`}.
- `coach_thread_reports(id INTEGER PK, thread_id INTEGER, reporter_role TEXT, reason TEXT, created_at TEXT, resolved INTEGER DEFAULT 0)`.

Functions (pure; emails lowercased; `_now()` UTC iso):
- `get_or_create_thread(cx, *, coach_email, member_email, source="coaching") -> dict` — upsert on `UNIQUE(coach_email, member_email)`; returns the thread row. (Callers gate on an accepted pair BEFORE calling; the store does not check matching.)
- `get_thread(cx, thread_id) -> dict|None`; `thread_for_pair(cx, coach_email, member_email) -> dict|None`.
- `post_message(cx, *, thread_id, sender_role, body) -> int` — inserts; returns message id. (Caller enforces `status='active'` + participant identity.)
- `messages(cx, thread_id) -> [dict]` — `[{id, sender_role, body, created_at}]`, chronological.
- `mark_read(cx, thread_id, role)` — set `<role>_last_read_at = now`.
- `unread_count(cx, thread_id, role) -> int` — messages from the OTHER role newer than this role's `last_read_at`.
- `block_thread(cx, thread_id, blocked_by_role)` — `status='blocked'`, `blocked_by=<role>`.
- `report_thread(cx, *, thread_id, reporter_role, reason)` — insert a report row + set `reported=1`.
- `list_all_threads(cx) -> [dict]` — owner console: every thread with `source, coach_email, member_email, status, reported, created_at, message count, last_message_at` (reported/blocked sort first).

### 2. Member thread routes (`app.py`, `_evox_ident` portal-token gated)

The member's counterpart is their accepted coach (from `coach_connect`). All routes resolve the member email from the token, then resolve the accepted pair; 404 if the member has no accepted coach.
- `GET /api/coach-thread/member` → `{coach_name, status, messages:[{sender_role, body, created_at}], can_post}` — materializes the thread; `mark_read(member)`; if `status='blocked'`, returns `messages:[]` + a "this conversation has ended" state (history hidden). `coach_name` = coach display name (directory), never email.
- `POST /api/coach-thread/member/message {body}` → `post_message(sender_role='member')` (400 empty/oversized body; 409 if blocked); best-effort nudge to the coach.
- `POST /api/coach-thread/member/block` → resolve `accepted_pair` → `block_thread(blocked_by='member')` + `coach_connect.set_request_status(pair.request_id, 'ended')` (frees coach capacity, lets the member re-apply) + owner email. Idempotent.
- `POST /api/coach-thread/member/report {reason}` → `report_thread(reporter_role='member')` + owner email.

### 3. Coach thread routes (`app.py`, `_practitioner_session_pid` session gated)

The coach's counterpart(s) are their accepted members. Resolve coach email from pid; a coach may have several members.
- `GET /api/coach-thread/coach/list` → `[{member_first_name, thread_id, status, unread}]` for each accepted member (resolved via `coach_connect.accepted_members`; first name via `client_portal.get_portal_content_by_email`, never email).
- `GET /api/coach-thread/coach/<thread_id>` → `{member_first_name, status, messages, can_post}` — 403 unless the coach owns the thread; `mark_read(coach)`; blocked → history hidden.
- `POST /api/coach-thread/coach/<thread_id>/message {body}` → `post_message(sender_role='coach')` (owner-check; 409 if blocked); nudge to the member.
- `POST /api/coach-thread/coach/<thread_id>/report {reason}` → `report_thread(reporter_role='coach')` + owner email. (No coach block route this slice.)

### 4. Owner moderation routes (`app.py`, `X-Console-Key == CONSOLE_SECRET`)

- `GET /api/console/coach-threads` → `list_all_threads` (reported/blocked first). Owner sees both emails (moderation context).
- `GET /api/console/coach-threads/<thread_id>` → full transcript incl. blocked history + any report rows.
- `POST /api/console/coach-threads/<thread_id>/unmatch` → owner-initiated block (same as member block: `status='blocked'`, `blocked_by='owner'`, the pair's accepted `coach_requests` row → `ended` via `accepted_pair`/its request id, both parties emailed the pairing ended).
- `POST /api/console/coach-threads/<thread_id>/resolve-report` → clear `reported` + mark report rows resolved (owner reviewed, no action).
- Surfaced on console-biofield-portal.html (a "Coaching threads" panel).

### 5. Portal surfaces

- **client-portal.html:** the existing "Your coach" card (when the member has an accepted coach) gains a **Message** thread — reverse-chronological messages (`textContent`), a compose box, and quiet **Block** and **Report** links (with a confirm + reason prompt). Coach shown by display name. Blocked → a calm "This coaching conversation has ended" note, no history.
- **practitioner-portal.html:** a **"Your members"** panel listing each accepted member (first name + unread badge); selecting one opens the thread (compose + **Report**). Member shown by first name.
- All server/dynamic strings via `textContent`; copy has no em dashes, no ALL CAPS; calm, consultative tone.

## Data flow

1. Member and coach match (slice 2 accept) → an accepted `coach_requests` row exists.
2. Either side opens their thread surface → `get_or_create_thread` materializes it once (unique on the pair).
3. A post inserts a `coach_messages` row and fires a best-effort nudge to the other side's email (portal link, no email exposed).
4. Member block → thread `blocked`, `coach_requests`→`ended` (capacity freed), history hidden from both, owner emailed.
5. Report → `reported=1` + report row + owner emailed; thread keeps working.
6. Owner console reads any thread, resolves reports, or unmatches.

## Error handling

- Every route resolves identity first (member `_evox_ident`→404, coach `_practitioner_session_pid`→401, owner key→401), THEN validates the body — a bad token never leaks a 400 (mirrors the Layer B fix).
- Coach routes 403 unless the coach owns `<thread_id>`; member routes operate only on the member's own accepted pair.
- `post_message` rejects empty/whitespace and over-long bodies (cap, e.g. 4000 chars) with 400; a blocked thread rejects posts with 409.
- Nudge/owner emails are best-effort (own try/except, never break the request or raise).
- Writes under `with _db_lock, sqlite3.connect(LOG_DB)`; block + `coach_requests` status change happen in one locked write so capacity/pairing stay consistent.
- `'ended'` is a new `coach_requests.status` value: verify blast radius — `accepted_count` and `member_has_accepted` filter `status='accepted'` (so `ended` correctly frees capacity + lets re-apply); `member_applications` lists `pending`/`accepted` (ended drops off); no other reader treats unknown statuses as accepted ([[project_pb_to_illtowell_evox]] enum-blast-radius check).

## Config

- No new env. Reuses `CONSOLE_SECRET`, `GLEN_CONSULT_EMAIL`, `PUBLIC_BASE_URL`, the email sender.
- Optional `COACH_MESSAGE_MAX_CHARS` (default 4000).

## Testing

- **Pure store** (`dashboard/coach_threads.py`): get_or_create idempotent on the pair; post/messages round-trip chronological; unread_count counts only the other role's newer messages; mark_read zeroes it; block sets status+blocked_by; report sets reported + a report row; list_all_threads sorts reported/blocked first.
- **Member routes** (mocked identity + accepted pair): GET materializes + returns coach display name never email; POST message rejects empty (400), rejects when blocked (409), nudges coach; block sets blocked + flips `coach_requests`→`ended` (assert `accepted_count` drops, member can re-apply) + hides history on next GET; report flags + owner emailed; a member with no accepted coach → 404.
- **Coach routes** (mocked practitioner session): list shows first name + unread per accepted member never email; GET/POST 403 on a thread the coach does not own; blocked thread hides history; post nudges member.
- **Owner routes**: list flags reported first; transcript shows blocked history; unmatch blocks + ends pairing + emails both; resolve-report clears the flag; all 401 without the key.
- **Privacy regression**: no member/coach route response payload ever contains the counterpart email (assert on serialized JSON); a third member's token cannot read a pair's thread.
- **Enum blast radius**: adding `coach_requests.status='ended'` does not make an ended pair count as accepted anywhere.
- **Go-live**: as a matched member, open the thread, message, see the coach reply, block, confirm the pairing frees + history hides; as owner, read the transcript + unmatch.

## Deferred (coaching arc / peer arc, later)

- The peer-matching arc (community-c2a): `source='peer'` threads with two `member` participants reusing these tables/plumbing + member↔member auth on both sides.
- Media/attachments; live chat; group threads; coach-initiated block/end; "show unread since" receipts to the other party; per-thread mute; a members-can-rate-their-coach loop.
