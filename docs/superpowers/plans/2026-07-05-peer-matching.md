# Peer Matching (community arc c2a) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Paid members opt in to anonymous, like-minded peer intros matched on shared liked-topics; a mutual connect reveals first names and opens a `source='peer'` 1:1 thread with the slice-3 moderation baked in.

**Architecture:** New pure-sqlite store `dashboard/peer_connect.py` (opt-in pool, directional connect/skip intents, mutual matches). Member routes (`_evox_ident` + `_is_paid_member`) for opt-in / anonymous proposal / interest / connections. Thin peer-thread routes reuse the shipped `dashboard/coach_threads.py` store via a slot convention (member↔member). One `source='peer'` branch added to the existing owner unmatch route. Portal surface in `client-portal.html`.

**Tech Stack:** Python 3 / Flask (`app.py` + `dashboard/*.py`), sqlite (`chat_log.db`, `?` placeholders, `_db_lock`, `cx.row_factory = sqlite3.Row`), reuses `community_signals`, `coach_threads`, `client_portal`, `send_evox_email`.

## Global Constraints

- **Privacy (load-bearing):** only opted-in paid members are matchable. Proposals are ANONYMOUS — `{member_ref, shared_topics}` only, never a name or email — until a mutual connect. No "you were passed on" signal ever. No email in any payload; first name only at reveal. Peer threads inherit slice-3 privacy (no counterpart email, blocked history owner-only, bodies via `textContent`).
- **Anonymized handle:** `member_ref(email) = sha256(lower(email))[:16]` (identical construction to `coach_connect.coach_ref`). Refs are one-way; resolve by scanning opted-in members.
- **Exclusions from a proposal for member M and candidate N:** a `peer_matches` row exists for the pair (active OR ended → never re-proposed); M already acted on N (`interest_kind(M,N)` set, connect or skip); N skipped M (`interest_kind(N,M)=='skip'`); either person-blocked the other in `community_signals` (`target_type='person'`, `signal='block'`, `target_key=member_ref(other)`); no shared liked-topic. N who *connected* to M (and M hasn't acted) MUST still appear so M can reciprocate.
- **Mutual match:** created under one `_db_lock` write; `peer_matches UNIQUE(a_email,b_email)` (normalized `a<b`) + `coach_threads UNIQUE` prevent a double thread/match on simultaneous connects.
- **Eligibility:** `_is_paid_member(email)` for opt-in / proposal / interest (free → 403 `not_eligible`, drives the upgrade tease). Auth order: `_evox_ident`→404, then `_is_paid_member`→403, then body.
- **Peer thread slot convention:** for a normalized pair (`a_email < b_email`), `a_email` = the store's `coach_email` slot, `b_email` = `member_email` slot; caller role = `'coach'` if caller==a_email else `'member'` (internal only; UI shows the other's first name).
- No new env. Copy: no em dashes, no ALL CAPS. sqlite writes under `_db_lock`; emails lowercased. DRY, YAGNI, TDD.

**Repo facts (verified anchors):**
- `dashboard/community_signals.py`: `community_signals(email, target_type, target_key, signal, UNIQUE(email,target_type,target_key))`; liked topics = `target_type='topic' AND signal='like'`; blocked = `signal='block'`; `init_signal_tables`, `set_signal`, `_lc`.
- `dashboard/coach_threads.py`: `init_thread_tables`, `get_or_create_thread(cx,*,coach_email,member_email,source='coaching')`, `get_thread`, `thread_for_pair`, `post_message(cx,*,thread_id,sender_role,body)`, `messages(cx,thread_id,epoch=None)`, `mark_read(cx,thread_id,role)`, `unread_count`, `block_thread(cx,thread_id,role)`, `report_thread(cx,*,thread_id,reporter_role,reason)`. Thread rows carry `source`, `status`, `active_epoch`.
- `app.py`: `_evox_ident(cx,token)->ident|None` (`.email`); `_is_paid_member(email)->bool`; `send_evox_email(to,name,subject,html,text,ics)`; `GLEN_CONSULT_EMAIL`; `PUBLIC_BASE_URL`; `COACH_MESSAGE_MAX_CHARS`; `_coach_thread_nudge(to,label)` / `_coach_thread_owner_alert(subj,detail)` (slice-3 helpers, reusable); `_portal_console_ok()`; `_db_lock`; `LOG_DB`. Owner unmatch route = `console_coach_thread_unmatch` (has `t=get_thread`; add a `source` branch). `client_portal.get_portal_content_by_email(cx,email)->{name,...}`.
- `coach_connect.coach_ref(email) = hashlib.sha256(_lc(email).encode("utf-8")).hexdigest()[:16]` — mirror for `member_ref`.

**Testing note (READ FIRST):**
- Pure store tests (Task 1) — plain `python3 -m pytest <path> -q`.
- Route tests (Tasks 2-3) `import app`; override DATA_DIR, and run EACH new test file in its OWN fresh DATA_DIR (the route test files share one `chat_log.db` per pytest session):
  ```
  export DATA_DIR="$(mktemp -d)" && doppler run -p remedy-match -c prd -- env DATA_DIR="$DATA_DIR" python3 -m pytest <one path> -q
  ```
- Frontend (Task 4): `node --check` on the extracted `<script>` blocks.

---

### Task 1: Store (`dashboard/peer_connect.py`)

**Files:**
- Create: `dashboard/peer_connect.py`
- Test: `tests/test_peer_connect_store.py`

**Interfaces:**
- Consumes: `dashboard/community_signals.py` (reads `community_signals` directly), `dashboard/coach_threads.py` (reads `coach_threads` for blocked-pair exclusion — via the `peer_matches` row, so no direct dependency needed at match time).
- Produces: `member_ref(email)`; `init_peer_tables(cx)`; `set_optin(cx,email,active)`; `is_opted_in(cx,email)`; `opted_in_members(cx)`; `liked_topics(cx,email)`; `blocked_topics(cx,email)`; `record_interest(cx,from_email,to_email,kind)`; `interest_kind(cx,from_email,to_email)`; `next_candidate(cx,me)`; `create_match(cx,a_email,b_email,thread_id)`; `match_for_pair(cx,e1,e2)`; `matches_for(cx,me)`; `end_match(cx,thread_id)`; `resolve_ref(cx,me,member_ref)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_peer_connect_store.py
import sqlite3
from dashboard import peer_connect as _pc
from dashboard import community_signals as _cs


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    _pc.init_peer_tables(cx)
    _cs.init_signal_tables(cx)
    return cx


def _like(cx, email, *topics):
    for t in topics:
        _cs.set_signal(cx, email, "topic", t, "like")


def test_member_ref_matches_coach_ref_shape():
    import hashlib
    assert _pc.member_ref("A@x.com") == hashlib.sha256(b"a@x.com").hexdigest()[:16]


def test_optin_pool():
    cx = _cx()
    _pc.set_optin(cx, "A@x.com", True)
    assert _pc.is_opted_in(cx, "a@x.com") is True
    _pc.set_optin(cx, "a@x.com", False)
    assert _pc.is_opted_in(cx, "a@x.com") is False
    assert "a@x.com" not in _pc.opted_in_members(cx)


def test_liked_and_blocked_topics():
    cx = _cx()
    _like(cx, "m@x.com", "liver", "sleep")
    _cs.set_signal(cx, "m@x.com", "topic", "keto", "block")
    assert _pc.liked_topics(cx, "m@x.com") == {"liver", "sleep"}
    assert _pc.blocked_topics(cx, "m@x.com") == {"keto"}


def test_next_candidate_anonymous_shared_topics_ranked():
    cx = _cx()
    _pc.set_optin(cx, "me@x.com", True); _like(cx, "me@x.com", "liver", "sleep", "detox")
    _pc.set_optin(cx, "one@x.com", True); _like(cx, "one@x.com", "liver")           # 1 shared
    _pc.set_optin(cx, "two@x.com", True); _like(cx, "two@x.com", "liver", "sleep")  # 2 shared
    _pc.set_optin(cx, "off@x.com", False); _like(cx, "off@x.com", "liver", "sleep") # not opted in
    c = _pc.next_candidate(cx, "me@x.com")
    assert c["member_ref"] == _pc.member_ref("two@x.com")     # highest overlap
    assert c["shared_topics"] == ["liver", "sleep"]
    assert "two@x.com" not in str(c) and "email" not in c     # anonymous


def test_next_candidate_excludes_acted_skipped_matched():
    cx = _cx()
    _pc.set_optin(cx, "me@x.com", True); _like(cx, "me@x.com", "liver")
    _pc.set_optin(cx, "skipme@x.com", True); _like(cx, "skipme@x.com", "liver")
    _pc.record_interest(cx, "me@x.com", "skipme@x.com", "skip")   # I skipped them
    assert _pc.next_candidate(cx, "me@x.com") is None
    _pc.set_optin(cx, "theyskip@x.com", True); _like(cx, "theyskip@x.com", "liver")
    _pc.record_interest(cx, "theyskip@x.com", "me@x.com", "skip") # they skipped me
    assert _pc.next_candidate(cx, "me@x.com") is None


def test_next_candidate_keeps_someone_who_connected_to_me():
    cx = _cx()
    _pc.set_optin(cx, "me@x.com", True); _like(cx, "me@x.com", "liver")
    _pc.set_optin(cx, "keen@x.com", True); _like(cx, "keen@x.com", "liver")
    _pc.record_interest(cx, "keen@x.com", "me@x.com", "connect")  # they want to connect
    c = _pc.next_candidate(cx, "me@x.com")
    assert c["member_ref"] == _pc.member_ref("keen@x.com")       # still shown so I can reciprocate


def test_resolve_ref_and_match():
    cx = _cx()
    _pc.set_optin(cx, "a@x.com", True); _pc.set_optin(cx, "b@x.com", True)
    assert _pc.resolve_ref(cx, "a@x.com", _pc.member_ref("b@x.com")) == "b@x.com"
    assert _pc.resolve_ref(cx, "a@x.com", "deadbeefdeadbeef") is None
    _pc.create_match(cx, "b@x.com", "a@x.com", 42)               # normalized a<b
    m = _pc.match_for_pair(cx, "a@x.com", "b@x.com")
    assert m["thread_id"] == 42 and m["a_email"] == "a@x.com" and m["b_email"] == "b@x.com"
    assert [x["other_email"] for x in _pc.matches_for(cx, "a@x.com")] == ["b@x.com"]
    _pc.end_match(cx, 42)
    assert _pc.match_for_pair(cx, "a@x.com", "b@x.com")["status"] == "ended"
    # an ended match is never re-proposed
    _like(cx, "a@x.com", "liver"); _like(cx, "b@x.com", "liver")
    assert _pc.next_candidate(cx, "a@x.com") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_peer_connect_store.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/peer_connect.py
"""Peer matching store (community arc c2a). Pure sqlite. Paid, opted-in members are
matched on shared liked-topics (read from community_signals); a mutual 'connect'
opens a source='peer' coach_threads thread. Privacy: proposals are anonymous
(member_ref only) until mutual; no 'you were passed on' signal; blocks/skips/matches
are excluded. Eligibility (paid) + reveal + thread live in the routes."""
import hashlib


def _now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _lc(email):
    return (email or "").strip().lower()


def member_ref(email):
    return hashlib.sha256(_lc(email).encode("utf-8")).hexdigest()[:16]


_DDL = """
CREATE TABLE IF NOT EXISTS peer_optin (
    member_email TEXT PRIMARY KEY,
    active INTEGER NOT NULL DEFAULT 1,
    updated_at TEXT
);
CREATE TABLE IF NOT EXISTS peer_interest (
    from_email TEXT NOT NULL,
    to_email TEXT NOT NULL,
    kind TEXT NOT NULL,
    created_at TEXT,
    UNIQUE(from_email, to_email)
);
CREATE TABLE IF NOT EXISTS peer_matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    a_email TEXT NOT NULL,
    b_email TEXT NOT NULL,
    thread_id INTEGER,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT,
    UNIQUE(a_email, b_email)
);
"""


def init_peer_tables(cx):
    cx.executescript(_DDL)
    cx.commit()


def set_optin(cx, email, active):
    cx.execute("INSERT INTO peer_optin (member_email, active, updated_at) VALUES (?,?,?) "
               "ON CONFLICT(member_email) DO UPDATE SET active=excluded.active, "
               "updated_at=excluded.updated_at", (_lc(email), 1 if active else 0, _now()))
    cx.commit()


def is_opted_in(cx, email):
    row = cx.execute("SELECT active FROM peer_optin WHERE member_email=?", (_lc(email),)).fetchone()
    return bool(row and row["active"])


def opted_in_members(cx):
    return [r["member_email"] for r in
            cx.execute("SELECT member_email FROM peer_optin WHERE active=1").fetchall()]


def _topics(cx, email, signal):
    return {r["target_key"] for r in cx.execute(
        "SELECT target_key FROM community_signals WHERE email=? AND target_type='topic' "
        "AND signal=?", (_lc(email), signal)).fetchall()}


def liked_topics(cx, email):
    return _topics(cx, email, "like")


def blocked_topics(cx, email):
    return _topics(cx, email, "block")


def record_interest(cx, from_email, to_email, kind):
    cx.execute("INSERT INTO peer_interest (from_email, to_email, kind, created_at) "
               "VALUES (?,?,?,?) ON CONFLICT(from_email, to_email) DO UPDATE SET "
               "kind=excluded.kind, created_at=excluded.created_at",
               (_lc(from_email), _lc(to_email), kind, _now()))
    cx.commit()


def interest_kind(cx, from_email, to_email):
    row = cx.execute("SELECT kind FROM peer_interest WHERE from_email=? AND to_email=?",
                     (_lc(from_email), _lc(to_email))).fetchone()
    return row["kind"] if row else None


def _person_blocked(cx, blocker, blocked):
    row = cx.execute("SELECT 1 FROM community_signals WHERE email=? AND target_type='person' "
                     "AND target_key=? AND signal='block'",
                     (_lc(blocker), member_ref(blocked))).fetchone()
    return row is not None


def _pair_has_match(cx, e1, e2):
    a, b = sorted([_lc(e1), _lc(e2)])
    return cx.execute("SELECT 1 FROM peer_matches WHERE a_email=? AND b_email=?",
                      (a, b)).fetchone() is not None


def next_candidate(cx, me):
    me = _lc(me)
    mine = liked_topics(cx, me) - blocked_topics(cx, me)
    if not mine:
        return None
    best = None
    for n in opted_in_members(cx):
        if n == me:
            continue
        if _pair_has_match(cx, me, n):
            continue
        if interest_kind(cx, me, n) is not None:            # I already acted on them
            continue
        if interest_kind(cx, n, me) == "skip":              # they passed on me
            continue
        if _person_blocked(cx, me, n) or _person_blocked(cx, n, me):
            continue
        shared = mine & (liked_topics(cx, n) - blocked_topics(cx, n))
        if not shared:
            continue
        score = len(shared)
        if best is None or score > best[0] or (score == best[0] and member_ref(n) < best[1]):
            best = (score, member_ref(n), sorted(shared))
    if best is None:
        return None
    return {"member_ref": best[1], "shared_topics": best[2]}


def resolve_ref(cx, me, ref):
    me = _lc(me)
    for n in opted_in_members(cx):
        if n != me and member_ref(n) == ref:
            return n
    return None


def create_match(cx, a_email, b_email, thread_id):
    a, b = sorted([_lc(a_email), _lc(b_email)])
    cx.execute("INSERT OR IGNORE INTO peer_matches (a_email, b_email, thread_id, status, "
               "created_at) VALUES (?,?,?, 'active', ?)", (a, b, thread_id, _now()))
    cx.commit()


def match_for_pair(cx, e1, e2):
    a, b = sorted([_lc(e1), _lc(e2)])
    row = cx.execute("SELECT * FROM peer_matches WHERE a_email=? AND b_email=?", (a, b)).fetchone()
    return dict(row) if row else None


def matches_for(cx, me):
    me = _lc(me)
    rows = cx.execute("SELECT a_email, b_email, thread_id, status FROM peer_matches "
                      "WHERE a_email=? OR b_email=? ORDER BY id", (me, me)).fetchall()
    out = []
    for r in rows:
        other = r["b_email"] if r["a_email"] == me else r["a_email"]
        out.append({"other_email": other, "thread_id": r["thread_id"], "status": r["status"]})
    return out


def end_match(cx, thread_id):
    cx.execute("UPDATE peer_matches SET status='ended' WHERE thread_id=?", (thread_id,))
    cx.commit()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_peer_connect_store.py -q`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add dashboard/peer_connect.py tests/test_peer_connect_store.py
git commit -m "feat(community): peer matching store (opt-in, shared-topic matcher, matches)"
```

---

### Task 2: Peer match member routes (`app.py`)

**Files:**
- Modify: `app.py` (opt-in / proposal / interest / connections routes)
- Test: `tests/test_peer_match_api.py`

**Interfaces:**
- Consumes: `dashboard/peer_connect.py` (all), `dashboard/coach_threads.py` (`get_or_create_thread` for `source='peer'`), `dashboard/client_portal.py` (`get_portal_content_by_email`), `_evox_ident`, `_is_paid_member`, `_coach_thread_nudge`, `_db_lock`, `LOG_DB`.
- Produces: `GET /api/peer/state`; `POST /api/peer/optin`; `GET /api/peer/proposal`; `POST /api/peer/interest`; `GET /api/peer/connections`; `_peer_first_name(cx, email)`.

**Contract:** auth `_evox_ident`→404 then `_is_paid_member`→403 (except `state`, which returns `eligible:false` for a free member rather than erroring). Proposal returns `{candidate}` anonymous. Interest `connect`: resolve ref (404 stale), record intent; if the target already `connect`ed me, open a `source='peer'` thread (slot: `coach_email=min, member_email=max`), `create_match`, nudge both, return `{matched:true}`; else `{matched:false}`. `skip`: record, `{matched:false}`. Connections return first names only.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_peer_match_api.py
import json, sqlite3
from unittest import mock
import app as appmod
from dashboard import peer_connect as _pc, community_signals as _cs, coach_threads as _ct


def _client():
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()


def _member(email, *topics):
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        from dashboard import evox as _ev, client_portal as _cp
        _ev.init_evox_tables(cx); _cp.init_client_portal_table(cx)
        _pc.init_peer_tables(cx); _cs.init_signal_tables(cx); _ct.init_thread_tables(cx)
        _cp.ensure_token(cx, email, email.split("@")[0].title())
        for t in topics:
            _cs.set_signal(cx, email, "topic", t, "like")
        tok = _ev.ensure_portal_token(cx, email, email.split("@")[0]); cx.commit()
    return tok


def test_free_member_not_eligible():
    c = _client(); tok = _member("free@x.com", "liver")
    with mock.patch.object(appmod, "_is_paid_member", return_value=False):
        assert c.get(f"/api/peer/state?token={tok}").get_json()["eligible"] is False
        assert c.post(f"/api/peer/optin?token={tok}", json={"active": True}).status_code == 403


def test_proposal_is_anonymous():
    c = _client(); a = _member("a@x.com", "liver", "sleep"); b = _member("b@x.com", "liver", "sleep")
    with mock.patch.object(appmod, "_is_paid_member", return_value=True):
        c.post(f"/api/peer/optin?token={a}", json={"active": True})
        c.post(f"/api/peer/optin?token={b}", json={"active": True})
        d = c.get(f"/api/peer/proposal?token={a}").get_json()
    assert d["candidate"]["member_ref"] == _pc.member_ref("b@x.com")
    assert "b@x.com" not in json.dumps(d) and "Bob" not in json.dumps(d)   # anonymous


def test_mutual_connect_reveals_and_opens_thread():
    c = _client(); a = _member("a@x.com", "liver"); b = _member("b@x.com", "liver")
    with mock.patch.object(appmod, "_is_paid_member", return_value=True), \
         mock.patch.object(appmod, "send_evox_email"):
        c.post(f"/api/peer/optin?token={a}", json={"active": True})
        c.post(f"/api/peer/optin?token={b}", json={"active": True})
        ref_b = _pc.member_ref("b@x.com"); ref_a = _pc.member_ref("a@x.com")
        r1 = c.post(f"/api/peer/interest?token={a}", json={"member_ref": ref_b, "kind": "connect"})
        assert r1.get_json()["matched"] is False                # not yet mutual
        r2 = c.post(f"/api/peer/interest?token={b}", json={"member_ref": ref_a, "kind": "connect"})
        assert r2.get_json()["matched"] is True                 # mutual
    conns = c.get(f"/api/peer/connections?token={a}").get_json()
    assert conns[0]["first_name"] == "B" and "b@x.com" not in json.dumps(conns)
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row; _pc.init_peer_tables(cx)
        assert _pc.match_for_pair(cx, "a@x.com", "b@x.com")["thread_id"] is not None


def test_skip_removes_from_next_proposal():
    c = _client(); a = _member("a@x.com", "liver"); b = _member("b@x.com", "liver")
    with mock.patch.object(appmod, "_is_paid_member", return_value=True):
        c.post(f"/api/peer/optin?token={a}", json={"active": True})
        c.post(f"/api/peer/optin?token={b}", json={"active": True})
        c.post(f"/api/peer/interest?token={a}", json={"member_ref": _pc.member_ref("b@x.com"),
                                                       "kind": "skip"})
        assert c.get(f"/api/peer/proposal?token={a}").get_json()["candidate"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `export DATA_DIR="$(mktemp -d)" && doppler run -p remedy-match -c prd -- env DATA_DIR="$DATA_DIR" python3 -m pytest tests/test_peer_match_api.py -q`
Expected: FAIL — routes missing.

- [ ] **Step 3: Write minimal implementation**

Add to `app.py` (near the community/coach routes):

```python
def _peer_first_name(cx, email):
    from dashboard import client_portal as _cp
    row = _cp.get_portal_content_by_email(cx, email) or {}
    return ((row.get("name") or "").strip().split() or ["A member"])[0]


def _peer_ident_paid(cx, token, *, require_paid=True):
    """(email or None, eligible bool). eligible=False for a free member."""
    ident = _evox_ident(cx, token)
    if ident is None:
        return None, False
    return ident.email, _is_paid_member(ident.email)


@app.route("/api/peer/state")
def peer_state():
    from dashboard import peer_connect as _pc
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _pc.init_peer_tables(cx)
        email, eligible = _peer_ident_paid(cx, request.args.get("token", ""))
        if email is None:
            return jsonify({"error": "not_found"}), 404
        opted = _pc.is_opted_in(cx, email) if eligible else False
        has_prop = bool(eligible and opted and _pc.next_candidate(cx, email))
        return jsonify({"eligible": eligible, "opted_in": opted, "has_proposal": has_prop})


@app.route("/api/peer/optin", methods=["POST"])
def peer_optin():
    from dashboard import peer_connect as _pc
    active = bool((request.get_json(silent=True) or {}).get("active"))
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _pc.init_peer_tables(cx)
        email, eligible = _peer_ident_paid(cx, request.args.get("token", ""))
        if email is None:
            return jsonify({"error": "not_found"}), 404
        if not eligible:
            return jsonify({"error": "not_eligible"}), 403
        _pc.set_optin(cx, email, active)
        return jsonify({"ok": True, "opted_in": active})


@app.route("/api/peer/proposal")
def peer_proposal():
    from dashboard import peer_connect as _pc
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _pc.init_peer_tables(cx)
        email, eligible = _peer_ident_paid(cx, request.args.get("token", ""))
        if email is None:
            return jsonify({"error": "not_found"}), 404
        if not (eligible and _pc.is_opted_in(cx, email)):
            return jsonify({"candidate": None})
        return jsonify({"candidate": _pc.next_candidate(cx, email)})


@app.route("/api/peer/interest", methods=["POST"])
def peer_interest():
    from dashboard import peer_connect as _pc, coach_threads as _ct
    body = request.get_json(silent=True) or {}
    ref = (body.get("member_ref") or "").strip()
    kind = (body.get("kind") or "").strip()
    if kind not in ("connect", "skip"):
        kind = None
    matched = False
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _pc.init_peer_tables(cx); _ct.init_thread_tables(cx)
        email, eligible = _peer_ident_paid(cx, request.args.get("token", ""))
        if email is None:
            return jsonify({"error": "not_found"}), 404
        if not (eligible and _pc.is_opted_in(cx, email)):
            return jsonify({"error": "not_eligible"}), 403
        if not kind:
            return jsonify({"error": "bad_kind"}), 400
        target = _pc.resolve_ref(cx, email, ref)
        if target is None:
            return jsonify({"error": "not_found"}), 404
        _pc.record_interest(cx, email, target, kind)
        if kind == "connect" and _pc.interest_kind(cx, target, email) == "connect" \
                and not _pc.match_for_pair(cx, email, target):
            a, b = sorted([email, target])                       # slot: a->coach, b->member
            t = _ct.get_or_create_thread(cx, coach_email=a, member_email=b, source="peer")
            _pc.create_match(cx, a, b, t["id"])
            matched = True
            both = (target, email)
    if matched:
        for who in both:
            _coach_thread_nudge(who, "a member you connected with")
    return jsonify({"ok": True, "matched": matched})


@app.route("/api/peer/connections")
def peer_connections():
    from dashboard import peer_connect as _pc
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _pc.init_peer_tables(cx)
        email, eligible = _peer_ident_paid(cx, request.args.get("token", ""))
        if email is None:
            return jsonify({"error": "not_found"}), 404
        out = [{"first_name": _peer_first_name(cx, m["other_email"]),
                "thread_id": m["thread_id"], "status": m["status"]}
               for m in _pc.matches_for(cx, email)]
        return jsonify(out)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `export DATA_DIR="$(mktemp -d)" && doppler run -p remedy-match -c prd -- env DATA_DIR="$DATA_DIR" python3 -m pytest tests/test_peer_match_api.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_peer_match_api.py
git commit -m "feat(community): peer match routes (opt-in, anonymous proposal, mutual reveal)"
```

---

### Task 3: Peer thread routes + owner peer branch (`app.py`)

**Files:**
- Modify: `app.py` (peer-thread routes + a `source` branch in `console_coach_thread_unmatch`)
- Test: `tests/test_peer_thread_api.py`

**Interfaces:**
- Consumes: `dashboard/coach_threads.py` (`get_thread`, `messages`, `mark_read`, `post_message`, `block_thread`, `report_thread`), `dashboard/peer_connect.py` (`match_for_pair`, `matches_for`, `end_match`), `_evox_ident`, `_peer_first_name` (Task 2), `_coach_thread_nudge`/`_coach_thread_owner_alert`, `COACH_MESSAGE_MAX_CHARS`, `_portal_console_ok`, `_db_lock`, `LOG_DB`.
- Produces: `GET /api/peer-thread/<int:thread_id>`; `POST /api/peer-thread/<int:thread_id>/message`; `POST /api/peer-thread/<int:thread_id>/block`; `POST /api/peer-thread/<int:thread_id>/report`; `_peer_thread_role(cx, thread_id, email)`; the `source='peer'` branch in `console_coach_thread_unmatch`.

**Contract:** `_evox_ident`→404; the caller must be a participant of the peer thread (via a `peer_matches` row for the pair) else 403; role = `'coach'` if caller==`coach_email` slot else `'member'`. GET `mark_read` + blocked hides history + returns `other_first_name` (never email). Message 400/409. Block → `block_thread(role)` + `end_match(thread_id)` + owner alert. Owner unmatch on a `source='peer'` thread ends the `peer_match` instead of a coaching request.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_peer_thread_api.py
import json, sqlite3
from unittest import mock
import app as appmod
from dashboard import peer_connect as _pc, coach_threads as _ct


def _client():
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()


def _matched_pair(a="a@x.com", b="b@x.com"):
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        from dashboard import evox as _ev, client_portal as _cp
        _ev.init_evox_tables(cx); _cp.init_client_portal_table(cx)
        _pc.init_peer_tables(cx); _ct.init_thread_tables(cx)
        _cp.ensure_token(cx, a, "Ana Smith"); _cp.ensure_token(cx, b, "Ben Jones")
        lo, hi = sorted([a, b])
        t = _ct.get_or_create_thread(cx, coach_email=lo, member_email=hi, source="peer")
        _pc.create_match(cx, a, b, t["id"])
        ta = _ev.ensure_portal_token(cx, a, "Ana"); tb = _ev.ensure_portal_token(cx, b, "Ben")
        cx.commit()
    return t["id"], ta, tb


def test_participant_sees_other_first_name_not_email():
    c = _client(); tid, ta, tb = _matched_pair()
    d = c.get(f"/api/peer-thread/{tid}?token={ta}").get_json()
    assert d["other_first_name"] == "Ben" and "b@x.com" not in json.dumps(d)


def test_non_participant_403():
    c = _client(); tid, ta, tb = _matched_pair()
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        from dashboard import evox as _ev
        _ev.init_evox_tables(cx); tc = _ev.ensure_portal_token(cx, "c@x.com", "Cy"); cx.commit()
    assert c.get(f"/api/peer-thread/{tid}?token={tc}").status_code == 403


def test_message_then_block_ends_match():
    c = _client(); tid, ta, tb = _matched_pair()
    with mock.patch.object(appmod, "send_evox_email"):
        assert c.post(f"/api/peer-thread/{tid}/message?token={ta}",
                      json={"body": "hello peer"}).get_json()["ok"] is True
        assert c.post(f"/api/peer-thread/{tid}/block?token={ta}").get_json()["ok"] is True
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row; _pc.init_peer_tables(cx); _ct.init_thread_tables(cx)
        assert _pc.match_for_pair(cx, "a@x.com", "b@x.com")["status"] == "ended"
        assert _ct.get_thread(cx, tid)["status"] == "blocked"
    assert c.post(f"/api/peer-thread/{tid}/message?token={tb}",
                  json={"body": "still there?"}).status_code == 409


def test_owner_unmatch_peer_ends_match_not_coaching():
    c = _client(); tid, ta, tb = _matched_pair()
    with mock.patch.object(appmod, "_portal_console_ok", return_value=True), \
         mock.patch.object(appmod, "send_evox_email"):
        assert c.post(f"/api/console/coach-threads/{tid}/unmatch").get_json()["ok"] is True
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row; _pc.init_peer_tables(cx); _ct.init_thread_tables(cx)
        assert _pc.match_for_pair(cx, "a@x.com", "b@x.com")["status"] == "ended"
        assert _ct.get_thread(cx, tid)["status"] == "blocked"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `export DATA_DIR="$(mktemp -d)" && doppler run -p remedy-match -c prd -- env DATA_DIR="$DATA_DIR" python3 -m pytest tests/test_peer_thread_api.py -q`
Expected: FAIL — routes missing.

- [ ] **Step 3: Write minimal implementation**

Add the peer-thread routes to `app.py`:

```python
def _peer_thread_role(cx, thread_id, email):
    """(thread, role) if `email` is a participant of this peer thread, else (thread, None).
    role = 'coach' for the coach_email slot, 'member' for the member_email slot."""
    from dashboard import coach_threads as _ct, peer_connect as _pc
    t = _ct.get_thread(cx, thread_id)
    if not t or t["source"] != "peer":
        return None, None
    e = (email or "").strip().lower()
    if not _pc.match_for_pair(cx, t["coach_email"], t["member_email"]):
        return t, None
    if e == t["coach_email"]:
        return t, "coach"
    if e == t["member_email"]:
        return t, "member"
    return t, None


@app.route("/api/peer-thread/<int:thread_id>")
def peer_thread_get(thread_id):
    from dashboard import coach_threads as _ct
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _ct.init_thread_tables(cx)
        ident = _evox_ident(cx, request.args.get("token", ""))
        if ident is None:
            return jsonify({"error": "not_found"}), 404
        t, role = _peer_thread_role(cx, thread_id, ident.email)
        if role is None:
            return jsonify({"error": "forbidden"}), 403
        _ct.mark_read(cx, thread_id, role)
        other = t["member_email"] if role == "coach" else t["coach_email"]
        blocked = t["status"] == "blocked"
        return jsonify({"other_first_name": _peer_first_name(cx, other), "status": t["status"],
                        "can_post": not blocked,
                        "messages": [] if blocked else _ct.messages(cx, thread_id,
                                                                    epoch=t["active_epoch"])})


@app.route("/api/peer-thread/<int:thread_id>/message", methods=["POST"])
def peer_thread_message(thread_id):
    from dashboard import coach_threads as _ct
    body = ((request.get_json(silent=True) or {}).get("body") or "").strip()
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _ct.init_thread_tables(cx)
        ident = _evox_ident(cx, request.args.get("token", ""))
        if ident is None:
            return jsonify({"error": "not_found"}), 404
        t, role = _peer_thread_role(cx, thread_id, ident.email)
        if role is None:
            return jsonify({"error": "forbidden"}), 403
        if not body or len(body) > COACH_MESSAGE_MAX_CHARS:
            return jsonify({"error": "bad_body"}), 400
        if t["status"] == "blocked":
            return jsonify({"error": "blocked"}), 409
        _ct.post_message(cx, thread_id=thread_id, sender_role=role, body=body)
        other = t["member_email"] if role == "coach" else t["coach_email"]
    _coach_thread_nudge(other, "a member you connected with")
    return jsonify({"ok": True})


@app.route("/api/peer-thread/<int:thread_id>/block", methods=["POST"])
def peer_thread_block(thread_id):
    from dashboard import coach_threads as _ct, peer_connect as _pc
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _ct.init_thread_tables(cx); _pc.init_peer_tables(cx)
        ident = _evox_ident(cx, request.args.get("token", ""))
        if ident is None:
            return jsonify({"error": "not_found"}), 404
        t, role = _peer_thread_role(cx, thread_id, ident.email)
        if role is None:
            return jsonify({"error": "forbidden"}), 403
        _ct.block_thread(cx, thread_id, role)
        _pc.end_match(cx, thread_id)
    _coach_thread_owner_alert("A peer connection ended",
                              "A member blocked a peer connection; it has ended.")
    return jsonify({"ok": True})


@app.route("/api/peer-thread/<int:thread_id>/report", methods=["POST"])
def peer_thread_report(thread_id):
    from dashboard import coach_threads as _ct
    reason = ((request.get_json(silent=True) or {}).get("reason") or "").strip()
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _ct.init_thread_tables(cx)
        ident = _evox_ident(cx, request.args.get("token", ""))
        if ident is None:
            return jsonify({"error": "not_found"}), 404
        t, role = _peer_thread_role(cx, thread_id, ident.email)
        if role is None:
            return jsonify({"error": "forbidden"}), 403
        _ct.report_thread(cx, thread_id=thread_id, reporter_role=role, reason=reason)
    _coach_thread_owner_alert("A peer thread was reported",
                              "A member reported a peer thread. Review it in the console.")
    return jsonify({"ok": True})
```

Then add the `source='peer'` branch to `console_coach_thread_unmatch` — after `_ct.block_thread(cx, thread_id, "owner")`, before the coaching pair flip:

```python
        _ct.block_thread(cx, thread_id, "owner")
        if t["source"] == "peer":
            from dashboard import peer_connect as _pc
            _pc.init_peer_tables(cx)
            _pc.end_match(cx, thread_id)
            member_email, coach_email = t["member_email"], t["coach_email"]
        else:
            pair = _cc.accepted_pair(cx, t["member_email"])
            if pair and pair["coach_email"] == t["coach_email"]:
                _cc.set_request_status(cx, pair["request_id"], "ended")
            member_email, coach_email = t["member_email"], t["coach_email"]
```

(The existing both-parties best-effort email loop stays; the note copy already reads generically enough, or make it source-aware — keep it best-effort and copy-clean.)

- [ ] **Step 4: Run test to verify it passes**

Run: `export DATA_DIR="$(mktemp -d)" && doppler run -p remedy-match -c prd -- env DATA_DIR="$DATA_DIR" python3 -m pytest tests/test_peer_thread_api.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_peer_thread_api.py
git commit -m "feat(community): peer thread routes + owner unmatch peer branch"
```

---

### Task 4: Member surface (`static/client-portal.html`)

**Files:**
- Modify: `static/client-portal.html` (a "Connect with members" card + peer thread panel)
- Test: JS parse check.

**Interfaces:**
- Consumes: `GET /api/peer/state`, `POST /api/peer/optin`, `GET /api/peer/proposal`, `POST /api/peer/interest`, `GET /api/peer/connections`, and the peer-thread routes.

**Design note:** Mirror the coaches card. Near the `coaches-card` placeholder (~line 1025) add `html += '<div class="card" id="peer-card" hidden></div>';` and call `initPeerCard();` where `initCoachesCard();` is called (~line 1056). `initPeerCard`:
- `GET /api/peer/state?token=...`. If `eligible` is false → render a calm locked tease ("Membership opens like-minded member connections.") with the existing upgrade nudge; done.
- If eligible + not opted in → an "Open to meeting like-minded members" toggle (one-line explanation of the anonymous, mutual-only flow) → `POST /api/peer/optin {active:true}` → re-render.
- If eligible + opted in → fetch `GET /api/peer/proposal`: if a candidate, render the anonymous card ("A member who also resonates with " + shared topics joined by " and ", via `textContent`) with **Connect** (`POST /api/peer/interest {member_ref, kind:'connect'}` → on `matched:true` show "You are connected" and refresh connections; else "We will let you know if it is mutual.") and **Not now** (`kind:'skip'` → fetch the next proposal). Also fetch `GET /api/peer/connections` → a "Your connections" list (first name → opens the peer thread panel).
- Peer thread panel: mirror the coaching member thread panel but hit the `/api/peer-thread/<id>` routes; show `other_first_name`, messages (via `textContent`), compose + Send, **Report**, **Block** (confirm "End this connection?"). Blocked → "This connection has ended." only.

All dynamic strings via `textContent`; no em dashes, no ALL CAPS. Wrap the new script in `<!-- BEGIN peer-match script -->` / `<!-- END peer-match script -->`.

- [ ] **Step 1: Add the peer card + panel**

Read the coaches-card markup + `initCoachesCard` and add the `peer-card` placeholder, the `initPeerCard()` call, and the new script as described.

- [ ] **Step 2: Verify the page JS parses**

Run: `cd /tmp/wt-deploy-chat-cca589e9 && node --check <(python3 -c "import re;h=open('static/client-portal.html').read();print('\n;\n'.join(re.findall(r'<script>(.*?)</script>',h,re.S)))")`
Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add static/client-portal.html
git commit -m "feat(community): peer matching card + thread panel"
```

---

## Definition of Done

- A paid member opts in, sees one anonymous like-minded candidate (shared topics as the why-line), and connects or skips. A mutual connect reveals first names, opens a `source='peer'` thread, and nudges both. They converse with report/block; a block ends the connection; the owner moderates from the existing console (unmatch ends the peer match).
- Privacy holds: proposals carry no name/email; free/non-opted members are never candidates; a skip produces no signal to the skipped member; no email in any payload; peer threads inherit slice-3 privacy.
- All new tests pass; coaching (slices 1-3), the signal layer, and the appointment loop are untouched (additive tables + routes + reuse of the thread store).

## Deferred (not in this plan)

- Semantic interest-vector matching as a second candidate source; re-proposing skipped candidates after a cool-off; a person-block UI; batched match runs + candidate-count teaser; renaming `coach_threads` columns to `participant_a/participant_b`.
