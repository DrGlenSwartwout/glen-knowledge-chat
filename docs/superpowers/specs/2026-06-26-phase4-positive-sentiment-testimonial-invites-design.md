# Testimonial Collection — Phase 4: Positive-Sentiment-Triggered Invitations

**Date:** 2026-06-26
**Status:** Approved (brainstorm) — ready for implementation plan when scheduled
**Owner:** Glen
**Parent project:** In-house testimonial collection (`2026-06-26-in-house-testimonial-collection-design.md`)

## Context

Phase 1 brought the testimonial *collection form* in-house (`/results`, `POST /api/testimonials`,
console moderation; PR #353). Today an invitation to leave a testimonial only goes out post-purchase
for a **product** (`_send_review_invite` → `/review/<token>`). But clients express positive results
all the time in places we already capture — website chat, email replies, intake/inquiries, journal
entries — and those moments pass without an ask.

Phase 4 closes that gap: **detect when a client expresses a positive result/success in their recent
communications, and trigger a testimonial invitation** (a link to the Phase-1 `/results` form, asking
for a rating or a quick video). Per Glen's decision it is **review-queue-first**: a detected positive
result becomes a *suggested invite* that a human approves before any email sends — a false-positive
must never auto-email a client. Auto-send can be added later once the queue proves its hit rate.

This is a **prod** feature (the comms data, the console, and the send all live on prod), unlike the
local biofield tooling. It ships dark behind a flag.

## Sources (all four, per Glen)

Detection reads, for a given client email:
- **Website chat** — `query_log` (the client's own questions to the assistant).
- **Email feedback** — `personal_email_feedback`, **`ai_summary` / `extracted_topics` /
  `extracted_conditions` only**. `raw_text` is stored but is NEVER read or surfaced (privacy stance
  carried from B3b-2a).
- **Practitioner inquiries / intake** — `inquiries` (`main_challenge`, `main_goal`) and
  `inbound_leads` intake summaries.
- **Journal entries** — the illtowell.com/journal entries (already run through emotion/TCM analysis).

The first three are already aggregated by `dashboard/recent_comms.py recent_comms(cx, email,
days_window=7)`. Journal is added as a fourth section (read from its store, summary/analysis text, not
raw audio).

## Components

### 1. Detector — `dashboard/testimonial_signals.py` (new)
- `classify_positive_result(comms_text, complete) -> {positive: bool, confidence: float,
  quote: str, kind: str}` where `kind ∈ {"remedy","service","general"}`. Mirrors the
  `dashboard/biofield_interpret.py interpret_stresses(transcript, complete)` pattern: an **injected
  `complete(system, user)`** completer returning **strict JSON**, so the classifier is fully testable
  offline. `quote` is the client's own positive phrase, used as the evidence shown in the queue.
- `gather_comms_text(cx, email)` — assembles `recent_comms(...)` output + journal into one text blob
  (reusing `recent_comms`'s no-raw_text guarantee for the email-feedback portion).

### 2. Candidate store — `testimonial_invite_candidates` table (new, in LOG_DB)
Columns: `id`, `email`, `name`, `quote` TEXT, `source` TEXT, `kind` TEXT, `confidence` REAL,
`status` TEXT DEFAULT 'pending' (`pending` / `approved` / `sent` / `dismissed`), `detected_at`,
`decided_at`, `decided_by`, `sent_at`. Unique on `email` (one open candidate per person; re-detect
updates the pending row).

Helper module `dashboard/testimonial_invites.py`: `upsert_candidate`, `pending_queue`, `set_status`,
and the de-dup predicate `should_skip(cx, email)`.

**De-dup / cooldown — skip generating a candidate when any holds:**
- the person already submitted a testimonial (`product_reviews` where `kind='testimonial'` and
  `email=?`);
- they were invited within the cooldown — default **`TESTIMONIAL_INVITE_COOLDOWN_DAYS=180`** (a prior
  candidate with `status='sent'` inside the window);
- they already have a `pending` candidate;
- their email is on the suppression list (`email_suppression`) — the send path checks this too, but we
  skip early to keep the queue clean.

### 3. Scan job — prod cron
A daily job (same shape as the existing `glen-qbo-reconcile` / reveal crons): find emails with comms
activity in the last 7 days (distinct across `query_log` / `inquiries` / `personal_email_feedback` /
journal), and for each not-skipped email run the detector and `upsert_candidate` when
`positive and confidence >= TESTIMONIAL_INVITE_MIN_CONFIDENCE` (default 0.6). Bounded by recent
activity. Exposed as a console-key-gated `POST /api/console/testimonial-invites/scan` (with
`?dry_run=1`) following the prod-ops trigger pattern, plus a Render cron that calls it.

### 4. Console queue — `/console/testimonial-invites`
Folds into the existing **Approvals hub** (the count-card + nav-group pattern from sub-project B3).
Cards show: client name/email, the positive **quote**, source, confidence, detected_at, and two
actions — **Approve & send** / **Dismiss**. Console-gated (`require_console_key`). The console shows
AI summaries/quotes only; it never renders `personal_email_feedback.raw_text`. Actions go through the
existing dispatch/actions registry (`testimonial_invite.approve` / `.dismiss`).

### 5. Send — `_send_testimonial_invite(email, name)` (new, in app.py)
Mirrors `_send_review_invite` but links to **`/results`** (the Phase-1 in-house form), not the
product-bound `/review/<token>`. Sends via suppression-checked `dashboard/inbox.py send_email`
(`from_name="Dr. Glen Swartwout"`, dash-stripped body, Aloha/In-wellness voice). On success it sets
the candidate `status='sent'`, `sent_at=now` (this is the cooldown record). Best-effort: a send
failure leaves the candidate `approved` for retry, never crashes the action.

## Gating & rollout
- New flag **`TESTIMONIAL_INVITES_ENABLED`** (default off) gates the scan endpoint, the cron, the
  console page, and the send. Ships dark.
- Depends on Phase 1 being live (`TESTIMONIALS_ENABLED` on) so `/results` actually accepts the invited
  submissions. **Recommendation: flip Phase 1 live and watch real submissions before enabling
  Phase 4.**

## Testing
- `classify_positive_result`: positive body → `positive=true` with a quote; neutral/negative →
  `positive=false`; bad JSON → safe default `positive=false` (mirror the `interpret_stresses` tests,
  injected `complete`).
- `should_skip`: true for already-submitted, within-cooldown, pending-exists, suppressed; false for a
  fresh active client.
- Scan: a seeded positive comms blob creates one pending candidate; a second scan does not duplicate;
  a person who already submitted a testimonial yields no candidate.
- Console queue: `pending_queue` returns candidates; `testimonial_invite.approve` flips status to
  `sent` and calls the (mocked) send; `.dismiss` flips to `dismissed`. Queue never exposes `raw_text`.
- Send: `_send_testimonial_invite` calls `send_email` with a `/results` link and does not raise;
  suppressed email → no send.
- Flag off → scan endpoint and console page 404; no cron effect.

## Out of scope (Phase 4)
- **Practitioner `?p=` attribution** on auto-invites — we don't reliably know the practitioner per
  client yet; the invite link omits `?p=` for now. (Attribution arrives with the Phase 2/3 share-link
  minting.)
- **Any auto-send tier** — every invite is human-approved in Phase 4. A high-confidence auto-send tier
  is a follow-on once the queue's precision is known.
- Re-analyzing historical comms beyond the rolling activity window (no backfill).
