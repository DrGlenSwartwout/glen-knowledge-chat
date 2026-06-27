# Chat Abuse / IP-Protection Rate Limiting — Design

**Date:** 2026-06-27
**Status:** Approved (design), pending implementation plan
**Author:** Glen + Claude (brainstorming session)

## Problem

The public AI chatbot (`/chat`) exposes Glen's synthesized clinical knowledge (RAG over the
content/education Pinecone namespaces). Two distinct threats:

1. **IP scraping** — someone systematically interrogating the bot to extract the long-form
   clinical synthesis (the actual IP). Signal = velocity + breadth, not total volume.
2. **Cost abuse** — running up the Anthropic API bill. Signal = sustained monthly volume.

A naive "N words/month per user" cap fails both: it can't be applied to anonymous traffic
(no durable identity to count against — `amg_session` cookies reset trivially), and a
per-account cap is defeated by minting fresh accounts. It mostly just annoys real users.

## Key design decisions (resolved during brainstorming)

- **Gate answer *depth*, not access.** The protectable asset is the long-form synthesized
  answer, not the act of asking. Short answers stay free and frictionless; the deep answer
  becomes the metered/captured/throttled surface.
- **Anonymous limit behavior = soft email gate**, not a hard wall — it's a funnel tool.
- **Email gate strength = hybrid**: email accepted as-typed normally; magic-link
  verification triggered only when velocity/IP signals flag a session as likely abuse.
- **Short answers are uncapped** (bounded only by per-IP velocity).
- **Funnel chat endpoints** (`/begin/match/chat`, `/begin/concierge/chat`) get the per-IP
  velocity floor only — not the depth gate (already ToS-gated, lower scrape risk).

## Architecture: three enforcement layers

### Layer 1 — Per-IP velocity (the anti-scrape floor, all tiers, all chat endpoints)

The real scrape defense. Survives cookie/email resets because it keys on IP. A human can't
exceed it; a bot hits it regardless of how it rotates identity (short of rotating IPs, which
is the accepted residual — see Non-goals).

- Lift the **existing proven pattern** from the TTS endpoint (`app.py:15981–16012`): an
  in-memory dict keyed on client IP → recent request timestamps, guarded by a
  `threading.Lock`, returning HTTP 429 on exceed.
- IP obtained via the established pattern:
  `request.headers.get("X-Forwarded-For", request.remote_addr or "").split(",")[0].strip()`.
- Two windows per IP: a short burst window (per-minute) and a daily window.
- IPv6: key on the **/64 prefix**, not the full address (one user owns the whole /64).
- In-memory state is acceptable: velocity windows are short, and a Render restart resetting
  them is harmless (worst case a scraper gets one fresh short window post-restart). The
  daily window is best-effort; the durable backstop is Layer 3.

### Layer 2 — Depth gate (short free / full-via-email, anonymous only, `/chat` only)

Reuses existing plumbing: `/chat` already accepts a `mode` field (`"brief"` | `"full"`), and
`query_log` already has an `email_sent_at` column.

- **Short answer** (`mode="brief"`): always streamed inline to everyone. Uncapped (Layer 1
  velocity is its only bound).
- **Full answer** (`mode="full"`) for an **anonymous** caller (no email on file): instead of
  streaming, respond with an offer — "I'll email you the full answer" — capture the email,
  and deliver the full answer **asynchronously by email** (existing email path; stamp
  `email_sent_at`). This:
  - captures every lead,
  - makes the crown-jewel long-form answer un-scrapable at volume (async trickle, not a
    streaming firehose),
  - moves high-value output to a channel that is per-email metered and logged.
- **Hybrid verification:** the captured email is accepted as-typed and the answer sent
  normally. Only when the session is **flagged** (Layer 1 velocity tripped recently, or
  other abuse signal) is a magic-link confirmation required before the full answer is sent.
- Callers with an email on file (registered) or an active membership skip the gate — they
  get full answers inline (subject to Layer 3 / fair-use).

### Layer 3 — Monthly word ceiling on full answers (cost backstop, per email / per member)

Durable, identity-anchored, applies to **full-answer** consumption only.

- Counted from `query_log`, which already records `ts`, `session_id`, `email`, `mode`, and
  the answer text. Add an additive `word_count` column (populated at `log_query()` time) so
  we sum a number instead of re-tokenizing text on every request.
- Registered (email on file, no active membership): ceiling ~10,000 full-answer words / 30d.
- Paid member (`memberships.expires_at > now`, via `_active_membership_for_email`): fair-use
  unlimited, with an **abuse flag** raised at ~100,000 words / 30d for human review (no hard
  wall a real member hits).
- On exceed (registered tier): soft message ("you've reached this month's full-answer
  limit — here's the short answer / become a member for more"), fall back to short answers.

## Tier model (anchored on identity signals that already exist)

| Tier | Anchor (existing) | Short ans. | Full answer | Monthly full-ans words | Per-IP velocity |
|---|---|---|---|---|---|
| Anonymous | `amg_session` cookie + IP | uncapped* | via email (async) | n/a (email-gated) | 10/min, 40/day |
| Registered | `email` on file | uncapped* | inline | ~10,000 | 15/min, 60/day |
| Paid member | `memberships.expires_at > now` | uncapped* | inline | fair-use (flag ~100k) | 30/min, 150/day |

\*bounded only by per-IP velocity. **All numbers are tunable defaults**, defined as
constants in one place.

## Components / data

- **`dashboard/chat_limits.py`** (new) — the limiter logic, isolated and unit-testable:
  - `client_ip(request)` — IP extraction incl. IPv6 /64 normalization.
  - `check_velocity(ip, tier)` — in-memory window check; returns allow / retry-after.
  - `tier_for(session_id, email)` — resolves anonymous | registered | member using
    existing `is_member` / `_active_membership_for_email` / email presence.
  - `monthly_full_words(email)` — sums `query_log.word_count WHERE email=? AND
    mode='full' AND ts >= now-30d`.
  - `is_flagged(session_id, ip)` — whether to require magic-link verification.
  - Pure functions + an in-memory velocity store object; no Flask coupling, so it tests in
    isolation.
- **`query_log`** — additive migration: `ALTER TABLE query_log ADD COLUMN word_count
  INTEGER DEFAULT 0` (try/except pattern per `app.py:945`); populate in `log_query()`.
- **`abuse_flags`** (new table, optional v1) — `session_id, ip, reason, ts` for
  flagged-session tracking and the verify-on-suspicion decision. Created via the
  `_init_*_tables()` + `_db_lock` pattern.
- **`/chat`** wiring — call the limiter at the top of the handler; branch the full-answer
  path through the depth gate; pass `word_count` into `log_query()`.
- **`/begin/match/chat`, `/begin/concierge/chat`** — call `check_velocity` only.

## Error handling

- Velocity exceeded → HTTP 429 with a `Retry-After`-style hint in the SSE/JSON payload; the
  UI shows a friendly "slow down a moment" message, not an error.
- Monthly ceiling exceeded (registered) → graceful fallback to short answer + upgrade nudge.
- Email send failure (depth gate) → fall back to streaming the full answer inline (fail
  open: never deny a real user their answer because email broke), and log the failure.
- Limiter internal error → **fail open** (allow the request). A bug in abuse-prevention must
  never take down the chatbot.

## Testing

- Unit tests for `chat_limits.py`: velocity windows (per-min + per-day, IPv6 /64), tier
  resolution for each identity combination, monthly word summation (boundary at 30d), flag
  logic. Pure functions → fast offline tests (the `import begin_funnel`-style plain-pytest
  path, no network/secrets).
- Endpoint tests: anonymous full-answer request returns the email-capture branch; registered
  over-ceiling falls back to short; member never hard-walled; velocity 429 path.
- Real-shape mocks for `query_log` rows (avoid mock-masked-green: assert on the actual
  summed count, not a stubbed total).

## Non-goals / accepted residual risk

- **IP rotation** (residential proxy pools) defeats Layer 1 — accepted. What leaks is
  short-answer educational RAG, not formulations/sources (those aren't in the chat-facing
  namespaces). The depth gate keeps long-form synthesis off the freely-streamable surface
  regardless.
- **No `flask-limiter` dependency** — the in-memory pattern is already proven in-repo.
- **Console-configurable limits** — v1 uses constants; a settings UI is a possible follow-on.
- **Device fingerprinting** — out of scope.

## Open items for the implementation plan

- Confirm how the UI currently chooses `mode` (brief/full) and where the "email me the full
  answer" affordance renders.
- Exact magic-link flow for verify-on-suspicion (reuse existing auth/magic-link if one
  exists; otherwise scope minimally).
