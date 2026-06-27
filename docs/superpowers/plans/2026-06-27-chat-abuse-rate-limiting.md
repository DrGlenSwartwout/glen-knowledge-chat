# Chat Abuse / IP-Protection Rate Limiting — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use `- [ ]` checkboxes.
> **Canonical copy:** on approval, also save this to `deploy-chat/docs/superpowers/plans/2026-06-27-chat-abuse-rate-limiting.md` and work in worktree `/tmp/wt-deploy-chat-9a09cc4e` (branch `sess/9a09cc4e`).
> **Spec:** `docs/superpowers/specs/2026-06-27-chat-abuse-rate-limiting-design.md` (commit `3fcf38d`).

## Context

The public AI chatbot (`/chat`) streams Glen's synthesized clinical answers — the actual IP. Today there is **no rate limiting** on `/chat` (only the TTS endpoint is throttled), so it can be bulk-scraped, and the long-form ("full") answer streams freely to anonymous callers. A naive per-account word cap fails: anonymous `amg_session` cookies reset trivially and fresh accounts defeat per-account caps.

This builds a **three-layer** defense that gates *answer depth*, not access, reusing existing plumbing wherever possible:

1. **Per-IP velocity floor** — anti-scrape, all chat endpoints (copy the proven TTS pattern).
2. **Depth gate** — short answers stream free to everyone; for anonymous callers the *full* answer is delivered async via the existing `/full-report` email flow (lead capture + un-scrapable trickle).
3. **Monthly word ceiling on full answers** — per email / per member, counted from `query_log`; a generous cost backstop.
4. **Verify-on-suspicion** (hybrid) — flagged sessions must confirm via the existing magic-link flow before a full answer is emailed.

Outcome: humans never feel it; bulk scraping of the long-form synthesis becomes slow, captured, and detectable.

## Global Constraints

- **No new dependencies.** Use stdlib + existing helpers. `flask-limiter` is explicitly out.
- **Fail open.** Any internal error in the limiter path must allow the request — a bug in abuse-prevention must never break the chatbot.
- **All numbers are tunable constants in one place** (`LIMITS` in `dashboard/chat_limits.py`).
- **DB:** raw `sqlite3` against `LOG_DB`, guarded by `_db_lock`; additive `ALTER TABLE` wrapped in try/except (idiom at `app.py:_init_log_db` ~917).
- **IP extraction idiom (verbatim):** `request.headers.get("X-Forwarded-For", request.remote_addr or "").split(",")[0].strip()`.
- **Tests:** pytest, `tmp_path`+`monkeypatch`, `monkeypatch.setenv("DATA_DIR", str(tmp_path))`, import the dashboard module directly (pure-logic tests need no app reload / no network).
- **Default limits (verbatim):**

  | Tier | per-min | per-day | monthly full-answer words |
  |---|---|---|---|
  | anonymous | 10 | 40 | n/a (email-gated) |
  | registered | 15 | 60 | 10,000 |
  | member | 30 | 150 | flag at 100,000 (no hard wall) |

## File Structure

- **Create `dashboard/chat_limits.py`** — pure, Flask-free limiter logic: IP normalization, in-memory velocity store, tier policy, monthly-word summation. One responsibility: "given facts, decide allow/deny/gate." Unit-tested in isolation.
- **Create `tests/test_chat_limits.py`** — unit tests for the above.
- **Modify `app.py`** — `query_log` migration (`word_count`), `log_query()` populates it, wire the limiter into `/chat` + `/begin/match/chat` + `/begin/concierge/chat`, depth gate + monthly ceiling at `/chat`, optional `abuse_flags` table + verify-on-suspicion.
- **Modify `static/embed.html`** — render the "enter your email for the full answer" affordance when `/chat` returns a gated full-answer signal (route to existing `/full-report`).

---

### Task 1: `chat_limits.py` — IP normalization + velocity limiter

**Files:**
- Create: `dashboard/chat_limits.py`
- Test: `tests/test_chat_limits.py`

**Interfaces — Produces:**
- `client_ip(xff: str, remote_addr: str) -> str` — first XFF hop; IPv6 normalized to /64 prefix.
- `LIMITS: dict` — per-tier policy (see Global Constraints).
- `class VelocityLimiter:` with `__init__(self, clock=time.time)` and `check(self, ip: str, per_min: int, per_day: int) -> tuple[bool, int]` returning `(allowed, retry_after_seconds)`. Thread-safe (own `threading.Lock`). In-memory.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_chat_limits.py
from dashboard.chat_limits import client_ip, VelocityLimiter, LIMITS

def test_client_ip_takes_first_xff_hop():
    assert client_ip("1.2.3.4, 5.6.7.8", "9.9.9.9") == "1.2.3.4"

def test_client_ip_falls_back_to_remote_addr():
    assert client_ip("", "9.9.9.9") == "9.9.9.9"

def test_client_ip_ipv6_normalized_to_64():
    # two addresses in the same /64 collapse to one key
    a = client_ip("2001:db8:abcd:1234:1::1", "")
    b = client_ip("2001:db8:abcd:1234:ffff:: ff", "")
    assert a == b == "2001:db8:abcd:1234::/64"

def test_velocity_allows_under_limit():
    t = [1000.0]
    v = VelocityLimiter(clock=lambda: t[0])
    for _ in range(5):
        allowed, _ = v.check("ip", per_min=10, per_day=40)
        assert allowed

def test_velocity_blocks_over_per_minute():
    t = [1000.0]
    v = VelocityLimiter(clock=lambda: t[0])
    for _ in range(10):
        assert v.check("ip", 10, 40)[0]
    allowed, retry = v.check("ip", 10, 40)
    assert not allowed and retry > 0

def test_velocity_recovers_after_window():
    t = [1000.0]
    v = VelocityLimiter(clock=lambda: t[0])
    for _ in range(10):
        v.check("ip", 10, 40)
    t[0] += 61  # past the per-minute window
    assert v.check("ip", 10, 40)[0]

def test_limits_has_three_tiers():
    assert set(LIMITS) == {"anonymous", "registered", "member"}
    assert LIMITS["anonymous"]["per_min"] == 10
```

- [ ] **Step 2: Run to confirm failure** — `python3 -m pytest tests/test_chat_limits.py -q` → FAIL (module missing).

- [ ] **Step 3: Implement `dashboard/chat_limits.py`**

```python
"""Pure, Flask-free rate-limit + tier policy for the public chat.
Keep this importable with no app/network deps so it unit-tests in isolation."""
import ipaddress
import threading
import time

# Tunable in ONE place. per_min/per_day are per-IP velocity; monthly_full_words
# is the per-email/per-member full-answer ceiling (None = no hard wall).
LIMITS = {
    "anonymous":  {"per_min": 10, "per_day": 40,  "monthly_full_words": None,    "flag_full_words": None},
    "registered": {"per_min": 15, "per_day": 60,  "monthly_full_words": 10_000,  "flag_full_words": None},
    "member":     {"per_min": 30, "per_day": 150, "monthly_full_words": None,    "flag_full_words": 100_000},
}

def client_ip(xff: str, remote_addr: str) -> str:
    """First X-Forwarded-For hop, else remote_addr. IPv6 collapsed to /64."""
    raw = (xff or "").split(",")[0].strip() or (remote_addr or "").strip()
    if not raw:
        return "anon"
    try:
        ip = ipaddress.ip_address(raw)
    except ValueError:
        return raw
    if ip.version == 6:
        net = ipaddress.ip_network(f"{raw}/64", strict=False)
        return f"{net.network_address}/64"
    return raw

class VelocityLimiter:
    """In-memory per-IP sliding-window counter. Two windows: 60s and 86400s."""
    _MIN_WINDOW = 60
    _DAY_WINDOW = 86_400

    def __init__(self, clock=time.time):
        self._clock = clock
        self._hits: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def check(self, ip: str, per_min: int, per_day: int) -> tuple[bool, int]:
        now = self._clock()
        with self._lock:
            hits = [t for t in self._hits.get(ip, []) if now - t < self._DAY_WINDOW]
            minute = [t for t in hits if now - t < self._MIN_WINDOW]
            if len(minute) >= per_min:
                self._hits[ip] = hits
                return (False, self._MIN_WINDOW - int(now - minute[0]))
            if len(hits) >= per_day:
                self._hits[ip] = hits
                return (False, self._DAY_WINDOW - int(now - hits[0]))
            hits.append(now)
            self._hits[ip] = hits
            return (True, 0)
```

- [ ] **Step 4: Run tests** → PASS.
- [ ] **Step 5: Commit** — `feat(chat-limits): IP normalization + per-IP velocity limiter`

---

### Task 2: `chat_limits.py` — tier resolver + monthly full-answer words

**Files:** Modify `dashboard/chat_limits.py`; Modify `tests/test_chat_limits.py`

**Interfaces — Produces:**
- `tier_for(has_auth: bool, has_membership: bool, has_email: bool) -> str` — returns `"member" | "registered" | "anonymous"`. (Caller computes the bools from existing app.py helpers; this stays Flask-free.)
- `monthly_full_words(cx, email: str, now_iso: str) -> int` — sums `query_log.word_count WHERE email=? AND mode='full' AND ts >= (now-30d)`. `cx` is an open sqlite connection.

- [ ] **Step 1: Add failing tests**

```python
import sqlite3
from datetime import datetime, timedelta, timezone
from dashboard.chat_limits import tier_for, monthly_full_words

def test_tier_precedence():
    assert tier_for(True, True, True) == "member"
    assert tier_for(True, False, True) == "registered"
    assert tier_for(False, False, True) == "registered"   # email alone = registered
    assert tier_for(False, False, False) == "anonymous"

def _db(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "t.db"))
    cx.execute("CREATE TABLE query_log (ts TEXT, email TEXT, mode TEXT, word_count INTEGER DEFAULT 0)")
    return cx

def test_monthly_full_words_sums_only_full_in_window(tmp_path):
    cx = _db(tmp_path)
    now = datetime(2026, 6, 27, tzinfo=timezone.utc)
    recent = now.isoformat(); old = (now - timedelta(days=40)).isoformat()
    rows = [(recent,"a@x","full",300),(recent,"a@x","brief",999),
            (recent,"a@x","full",200),(old,"a@x","full",500),(recent,"b@x","full",111)]
    cx.executemany("INSERT INTO query_log VALUES (?,?,?,?)", rows); cx.commit()
    assert monthly_full_words(cx, "a@x", now.isoformat()) == 500  # 300+200 only
```

- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement** (append to `chat_limits.py`):

```python
from datetime import datetime, timedelta

def tier_for(has_auth: bool, has_membership: bool, has_email: bool) -> str:
    if has_membership:
        return "member"
    if has_auth or has_email:
        return "registered"
    return "anonymous"

def monthly_full_words(cx, email: str, now_iso: str) -> int:
    if not email:
        return 0
    try:
        cutoff = (datetime.fromisoformat(now_iso) - timedelta(days=30)).isoformat()
    except Exception:
        return 0
    row = cx.execute(
        "SELECT COALESCE(SUM(word_count),0) FROM query_log "
        "WHERE email=? AND mode='full' AND ts >= ?",
        (email, cutoff),
    ).fetchone()
    return int(row[0] or 0)
```

- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** — `feat(chat-limits): tier resolver + monthly full-answer word count`

---

### Task 3: `query_log.word_count` migration + `log_query` populates it

**Files:** Modify `app.py` (`_init_log_db` ~917, `log_query` ~1086); Modify a test (extend `tests/test_chat_limits.py` or new `tests/test_log_query_wordcount.py`).

- [ ] **Step 1:** Add `"word_count          INTEGER DEFAULT 0"` to the `col_def` list in `_init_log_db` (the try/except ALTER loop).
- [ ] **Step 2:** Add `word_count: int = 0` param to `log_query()` and include it in the INSERT column list + values tuple. Where `log_query` is called for an answer, compute `word_count=len(answer.split())` at the call sites in `/chat`, `/full-report`, and the funnel chats (or compute inside `log_query` from `answer` if the param is 0 — simpler: compute inside `log_query`: `wc = word_count or len((answer or "").split())`).
- [ ] **Step 3:** Test — open a tmp DB, run the app's init, insert via `log_query`, assert the row's `word_count` matches `len(answer.split())`. Use the `tmp_path`+`monkeypatch.setenv("DATA_DIR", ...)`+reload-`app` convention (skip if app import fails, per existing tests).
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5: Commit** — `feat(query_log): add word_count column, populate in log_query`

---

### Task 4: Wire per-IP velocity into all three chat endpoints

**Files:** Modify `app.py` — module-level singleton + a guard at the top of `chat()` (~2822), `/begin/match/chat` (~3257), `/begin/concierge/chat` (~6323).

**Interfaces — Consumes:** `client_ip`, `VelocityLimiter`, `LIMITS`, `tier_for` from Task 1–2; `get_authenticated_user` (520), `is_member` (546), `_active_membership_for_email` (7845).

- [ ] **Step 1:** Near the other module globals, add:

```python
from dashboard.chat_limits import (client_ip, VelocityLimiter, LIMITS,
                                    tier_for, monthly_full_words)
_chat_velocity = VelocityLimiter()

def _resolve_chat_tier(req, session_id, email):
    """Best-effort; fail open to 'anonymous' on any error."""
    try:
        auth = get_authenticated_user(req)
        eff_email = (email or (auth or {}).get("email") or "").strip()
        has_membership = bool(eff_email) and _active_membership_for_email(eff_email) is not None
        has_email = bool(eff_email) or is_member(session_id, eff_email)
        return tier_for(bool(auth), has_membership, has_email), eff_email
    except Exception as e:
        print(f"[chat-limit] tier resolve failed: {e!r}", flush=True)
        return "anonymous", (email or "")

def _velocity_guard(req, tier):
    """Return a Flask 429 response if over limit, else None. Fail open."""
    try:
        ip = client_ip(req.headers.get("X-Forwarded-For", ""), req.remote_addr or "")
        pol = LIMITS.get(tier, LIMITS["anonymous"])
        allowed, retry = _chat_velocity.check(ip, pol["per_min"], pol["per_day"])
        if not allowed:
            return jsonify({"error": "rate_limited", "retry_after": retry}), 429
    except Exception as e:
        print(f"[chat-limit] velocity guard failed: {e!r}", flush=True)
    return None
```

- [ ] **Step 2:** At the top of each of the three handlers (after `session_id`/`email`/`mode` are parsed, before any retrieval/streaming begins), insert:

```python
    _tier, _eff_email = _resolve_chat_tier(request, session_id, email)
    _blocked = _velocity_guard(request, _tier)
    if _blocked is not None:
        return _blocked
```

(For the two funnel endpoints that may not parse `email`, pass `email=""`.)

- [ ] **Step 3:** Test — using the app test client (reload-`app` convention), POST the same IP to `/chat` more than `per_min` times within the window and assert a 429 appears; assert one request under the limit returns 200/stream. Monkeypatch `_chat_velocity` with a fast clock or lower the limit via monkeypatching `LIMITS` if needed.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5: Commit** — `feat(chat): per-IP velocity limit on all chat endpoints`

---

### Task 5: Depth gate — anonymous full answers route to email (`/chat`)

**Files:** Modify `app.py` `chat()` (the `mode`/synth branch ~2960 and the streaming generator ~3010); Modify `static/embed.html` (full-report button path ~1595 + a gated-response handler).

**Behavior:** When `tier == "anonymous"` **and** `mode == "full"`, do **not** stream the full answer. Instead emit a single SSE signal telling the client to capture an email and call the existing `/full-report` flow.

- [ ] **Step 1:** In `chat()`, after tier resolution, before building `synth_instr`:

```python
    if _tier == "anonymous" and mode == "full":
        def _gated():
            yield sse({"gated": "email_required",
                       "message": "Enter your email and I'll send you the full report.",
                       "session_id": session_id})
            yield sse({"done": True})
        return Response(stream_with_context(_gated()), content_type="text/event-stream")
```

- [ ] **Step 2:** In `static/embed.html`, where SSE payloads are handled, branch on `data.gated === "email_required"`: render the existing email-capture UI, and on submit POST to `/full-report` (the endpoint that already regenerates full + emails + stamps `email_sent_at`). The funnel already has an email field and a `/full-report` path — reuse it; do not invent a new endpoint.
- [ ] **Step 3:** Backend test — POST `/chat` with `mode:"full"`, no auth cookie, no email → assert the SSE body contains `"gated": "email_required"` and the full answer text is **absent**. POST with `mode:"brief"` anonymous → assert tokens stream normally.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5: Render-verify the frontend** (per `feedback_render_verify_not_just_inject`): load the chat widget headless, trigger a full-answer request as anonymous, assert the email-capture UI renders and **zero console errors**. Do NOT rely on "script injected" alone.
- [ ] **Step 6: Commit** — `feat(chat): gate anonymous full answers to email via /full-report`

---

### Task 6: Monthly ceiling enforcement on full answers (`/chat`)

**Files:** Modify `app.py` `chat()`.

**Behavior:** For `mode == "full"` and `tier in ("registered","member")`, before generating, check `monthly_full_words`. Registered over `monthly_full_words` → downgrade to brief + nudge. Member over `flag_full_words` → still answer, but record a flag (Task 7 table; until then, `print` a flag line).

- [ ] **Step 1:** After tier resolution, for full mode:

```python
    if mode == "full" and _tier in ("registered", "member") and _eff_email:
        pol = LIMITS[_tier]
        try:
            with _db_lock, sqlite3.connect(LOG_DB) as _cx:
                used = monthly_full_words(_cx, _eff_email, datetime.now(timezone.utc).isoformat())
        except Exception:
            used = 0
        cap = pol.get("monthly_full_words")
        if cap is not None and used >= cap:
            mode = "brief"            # graceful downgrade
            _ceiling_hit = True       # used below to append an upgrade nudge
        flag = pol.get("flag_full_words")
        if flag is not None and used >= flag:
            print(f"[chat-limit] FULL-WORD FLAG email={_eff_email} used={used}", flush=True)
```

- [ ] **Step 2:** When `_ceiling_hit`, append a one-line nudge to the streamed answer (e.g. a final SSE token: "You've reached this month's full-report limit — here's the summary. Members get unlimited.").
- [ ] **Step 3:** Test — seed `query_log` for an email with `mode='full'` words ≥ 10,000 in-window, POST `/chat` `mode:"full"` with that email → assert the response is brief-length and includes the nudge; seed an email under the cap → assert full answer streams.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5: Commit** — `feat(chat): monthly full-answer word ceiling per email/member`

---

### Task 7 (final, deferrable): Verify-on-suspicion + `abuse_flags`

**Files:** Modify `app.py` — new `_init_abuse_flags()` table (CREATE TABLE pattern, called at load), record a flag when velocity trips, and in the depth-gate email path require magic-link confirmation for flagged sessions.

**Behavior (hybrid gate):** Normal anonymous → `/full-report` sends the email as-typed. If the session/IP is in `abuse_flags` (recent velocity trip), the gated response asks them to confirm via the **existing** `/auth/magic-link/request` flow before the full report is emailed.

- [ ] **Step 1:** Add `abuse_flags(session_id TEXT, ip TEXT, reason TEXT, ts TEXT)` via the `_init_*` + `_db_lock` idiom; insert a row inside `_velocity_guard` when a request is blocked.
- [ ] **Step 2:** Add `is_flagged(cx, session_id, ip) -> bool` (recent row within, say, 24h) to `chat_limits.py` with a unit test.
- [ ] **Step 3:** In the depth gate (Task 5), when `is_flagged`, emit `{"gated": "verify_email_required"}` instead; frontend calls `/auth/magic-link/request`. Reuse existing route — no new auth code.
- [ ] **Step 4:** Tests for `is_flagged` + the gated-vs-verify branch.
- [ ] **Step 5: Commit** — `feat(chat): verify-on-suspicion via magic-link for flagged sessions`

> If scope needs trimming, ship Tasks 1–6 first (velocity + depth gate + ceiling already deliver the core protection) and land Task 7 as a fast-follow.

---

## Verification (end-to-end)

- **Unit:** `python3 -m pytest tests/test_chat_limits.py -q` — all pure-logic tests pass with no network/secrets.
- **Endpoint (offline):** app-test-client tests for 429 velocity, anonymous-full gating, registered-ceiling downgrade (reload-`app` convention; skips cleanly if app import needs secrets).
- **Render-verify (Task 5):** headless-load the chat widget, run an anonymous full-answer request, confirm the email-capture UI renders with zero console errors.
- **Manual smoke before deploy:** brief answers stream for everyone; anonymous full → email-capture; a real member is never hard-walled.
- **Tuning pass:** before locking the 10k/100k numbers, pull the actual full-answer word distribution from prod `query_log` and adjust `LIMITS`.
- **Pre-PR hygiene:** `git diff --name-only origin/main..HEAD | grep -i superpowers` and `git rm --cached` any leaked SDD scratch (per `feedback_sdd_scratch_git_leak`).

## Decisions folded in (from brainstorming)

- Anonymous limit = soft email gate (funnel tool). Email gate = hybrid (verify only on suspicion). Short answers uncapped (velocity-only). Funnel chat endpoints get velocity-only.
- Numbers are tunable defaults, not load-bearing — Glen tunes after seeing real volume.
