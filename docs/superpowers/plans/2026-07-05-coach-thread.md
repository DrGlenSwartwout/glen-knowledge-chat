# 1:1 Coaching Thread + Moderation (coaching arc, slice 3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A text-only async 1:1 message thread between a matched coach and member, with a reply nudge, member block (ends the pairing + hides history), either-side report, and owner console moderation — built generically so the peer-matching arc reuses it.

**Architecture:** New pure-sqlite store `dashboard/coach_threads.py` (thread keyed on `UNIQUE(coach_email, member_email)`, two roles, a `source` tag). Two small `dashboard/coach_connect.py` helpers resolve counterparts by the accepted `coach_requests` row. Member routes (`_evox_ident`), coach routes (`_coach_session_email`), and owner routes (`_portal_console_ok`) sit in `app.py`. Portal surfaces in `client-portal.html` (member), `practitioner-portal.html` (coach), `console-biofield-portal.html` (owner).

**Tech Stack:** Python 3 / Flask (`app.py` + `dashboard/*.py`), sqlite (`chat_log.db`, `?` placeholders, `_db_lock`, `cx.row_factory = sqlite3.Row`), `send_evox_email` for nudges.

## Global Constraints

- **Privacy (load-bearing):** no member/coach route response payload ever contains the counterpart's email or contact info. Coach sees the member first name only; member sees the coach display name only. A blocked thread's history is hidden from BOTH participants and returned only by the owner console. A third member's token can never read a pair's thread.
- **Block ends the pairing:** a member block sets the thread `blocked` AND flips the pair's accepted `coach_requests` row to `status='ended'` in the SAME locked write, freeing the coach's capacity and letting the member re-apply. `'ended'` is a NEW status value — it must NOT be treated as accepted anywhere (`accepted_count`/`member_has_accepted` filter `status='accepted'`; `member_applications` lists `pending`/`accepted`; verify no other reader).
- **Auth ordering:** resolve identity FIRST (member `_evox_ident`→404, coach `_coach_session_email`→401, owner key→401), THEN validate the body — a bad token never returns a 400.
- **Text only, async:** no media, no websockets. Message body rejects empty/whitespace (400) and over-`COACH_MESSAGE_MAX_CHARS` (default 4000) (400). A blocked thread rejects posts (409).
- **Nudges + owner emails are best-effort:** own try/except, never raise, never break the request.
- **Copy:** no em dashes, no ALL CAPS; calm consultative tone; all server/dynamic strings via `textContent`.
- sqlite writes under `with _db_lock, sqlite3.connect(LOG_DB)`; emails lowercased. No new env (reuses `CONSOLE_SECRET`, `GLEN_CONSULT_EMAIL`, `PUBLIC_BASE_URL`).
- DRY, YAGNI, TDD, frequent commits.

**Repo facts the implementer needs (verified anchors):**
- `dashboard/coach_connect.py`: `coach_requests(id, coach_email, member_email, member_name, note, member_video_url, status, created_at, decided_at, UNIQUE(coach_email,member_email))`; `init_connect_tables(cx)`, `set_request_status(cx, request_id, status)`, `accepted_count(cx, coach_email)` (filters `status='accepted'`), `member_has_accepted(cx, member_email)` (filters `'accepted'`), `member_applications` (lists `pending`/`accepted`), `requests_for_coach(cx, coach_email, status)`, `_now()`, `_lc(email)`.
- `app.py`: `_evox_ident(cx, token) -> ident|None` (`.email`, NO `.name`); `_coach_session_email() -> email|None` (practitioner session → coach email, used by the slice-2 coach routes; 401 when None); `_portal_console_ok() -> bool` (X-Console-Key / owner token); `send_evox_email(to_email, name, subject, html_body, text_body, ics_bytes)`; `GLEN_CONSULT_EMAIL`; `PUBLIC_BASE_URL`; `_db_lock`; `LOG_DB`.
- `dashboard/coach_directory.py`: `get_volunteer(cx, email) -> dict|None` (full row incl `name`, `focus`); `init_coach_tables(cx)`.
- `dashboard/client_portal.py`: `get_portal_content_by_email(cx, email) -> {name, ...}|None` (first name = `name.split()[0]`).
- Slice-2 coach route to mirror for auth shape: `practitioner_coach_requests` (app.py:16527) uses `_coach_session_email()` then `_cc`/`_cd`. Member route to mirror: `community_coaches` (app.py:16216) uses `_evox_ident`.

**Testing note (READ FIRST):**
- Pure store/helper tests (Task 1) — plain `python3 -m pytest <path> -q`.
- Route tests (Tasks 2-4) `import app`; override DATA_DIR:
  ```
  export DATA_DIR="$(mktemp -d)" && doppler run -p remedy-match -c prd -- env DATA_DIR="$DATA_DIR" python3 -m pytest <paths> -q
  ```
- Frontend tasks (5-6): JS parse check via `node --check` on the extracted `<script>` blocks.

---

### Task 1: Thread store + counterpart helpers

**Files:**
- Create: `dashboard/coach_threads.py`
- Modify: `dashboard/coach_connect.py` (add `accepted_pair`, `accepted_members`)
- Test: `tests/test_coach_threads_store.py`

**Interfaces:**
- Produces (coach_threads): `init_thread_tables(cx)`; `get_or_create_thread(cx, *, coach_email, member_email, source="coaching") -> dict`; `get_thread(cx, thread_id) -> dict|None`; `thread_for_pair(cx, coach_email, member_email) -> dict|None`; `post_message(cx, *, thread_id, sender_role, body) -> int`; `messages(cx, thread_id) -> [dict]`; `mark_read(cx, thread_id, role)`; `unread_count(cx, thread_id, role) -> int`; `block_thread(cx, thread_id, blocked_by_role)`; `report_thread(cx, *, thread_id, reporter_role, reason)`; `list_all_threads(cx) -> [dict]`.
- Produces (coach_connect): `accepted_pair(cx, member_email) -> {request_id, coach_email}|None`; `accepted_members(cx, coach_email) -> [{request_id, member_email, member_name}]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_coach_threads_store.py
import sqlite3
from dashboard import coach_threads as _ct
from dashboard import coach_connect as _cc


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    _ct.init_thread_tables(cx)
    _cc.init_connect_tables(cx)
    return cx


def test_get_or_create_idempotent():
    cx = _cx()
    t1 = _ct.get_or_create_thread(cx, coach_email="C@x.com", member_email="M@x.com")
    t2 = _ct.get_or_create_thread(cx, coach_email="c@x.com", member_email="m@x.com")
    assert t1["id"] == t2["id"] and t1["status"] == "active" and t1["source"] == "coaching"


def test_post_and_messages_chronological():
    cx = _cx()
    t = _ct.get_or_create_thread(cx, coach_email="c@x.com", member_email="m@x.com")
    _ct.post_message(cx, thread_id=t["id"], sender_role="coach", body="hello")
    _ct.post_message(cx, thread_id=t["id"], sender_role="member", body="hi back")
    ms = _ct.messages(cx, t["id"])
    assert [m["sender_role"] for m in ms] == ["coach", "member"]
    assert ms[0]["body"] == "hello"


def test_unread_counts_only_other_role_newer():
    cx = _cx()
    t = _ct.get_or_create_thread(cx, coach_email="c@x.com", member_email="m@x.com")
    _ct.post_message(cx, thread_id=t["id"], sender_role="coach", body="a")
    _ct.post_message(cx, thread_id=t["id"], sender_role="coach", body="b")
    assert _ct.unread_count(cx, t["id"], "member") == 2   # 2 from coach, unread by member
    assert _ct.unread_count(cx, t["id"], "coach") == 0     # own messages don't count
    _ct.mark_read(cx, t["id"], "member")
    assert _ct.unread_count(cx, t["id"], "member") == 0


def test_block_sets_status():
    cx = _cx()
    t = _ct.get_or_create_thread(cx, coach_email="c@x.com", member_email="m@x.com")
    _ct.block_thread(cx, t["id"], "member")
    b = _ct.get_thread(cx, t["id"])
    assert b["status"] == "blocked" and b["blocked_by"] == "member"


def test_report_sets_flag_and_row():
    cx = _cx()
    t = _ct.get_or_create_thread(cx, coach_email="c@x.com", member_email="m@x.com")
    _ct.report_thread(cx, thread_id=t["id"], reporter_role="member", reason="rude")
    assert _ct.get_thread(cx, t["id"])["reported"] == 1
    row = cx.execute("SELECT * FROM coach_thread_reports").fetchone()
    assert row["reporter_role"] == "member" and row["reason"] == "rude"


def test_list_all_sorts_flagged_first():
    cx = _cx()
    ok = _ct.get_or_create_thread(cx, coach_email="c1@x.com", member_email="m1@x.com")
    rep = _ct.get_or_create_thread(cx, coach_email="c2@x.com", member_email="m2@x.com")
    _ct.report_thread(cx, thread_id=rep["id"], reporter_role="coach", reason="x")
    ids = [t["id"] for t in _ct.list_all_threads(cx)]
    assert ids[0] == rep["id"]                              # reported first


def test_accepted_pair_and_members():
    cx = _cx()
    _cc.create_request(cx, "coach@x.com", "mem@x.com", "Mel P", "help me")
    rid = cx.execute("SELECT id FROM coach_requests").fetchone()["id"]
    _cc.set_request_status(cx, rid, "accepted")
    p = _cc.accepted_pair(cx, "mem@x.com")
    assert p["coach_email"] == "coach@x.com" and p["request_id"] == rid
    ms = _cc.accepted_members(cx, "coach@x.com")
    assert ms == [{"request_id": rid, "member_email": "mem@x.com", "member_name": "Mel P"}]
    _cc.set_request_status(cx, rid, "ended")
    assert _cc.accepted_pair(cx, "mem@x.com") is None       # ended is not accepted
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_coach_threads_store.py -q`
Expected: FAIL — modules/functions missing.

- [ ] **Step 3: Write minimal implementation**

Create `dashboard/coach_threads.py`:

```python
"""1:1 coaching thread store (arc slice 3). Pure sqlite. A thread hangs off a
matched pair (coach_email, member_email); two roles (coach/member) + a source tag
('coaching' now, 'peer' later) so the peer-matching arc reuses these tables. Text
only, async. Privacy + block/report policy live in the routes; this module is state."""

_DDL = """
CREATE TABLE IF NOT EXISTS coach_threads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL DEFAULT 'coaching',
    coach_email TEXT NOT NULL,
    member_email TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    blocked_by TEXT,
    reported INTEGER NOT NULL DEFAULT 0,
    created_at TEXT,
    coach_last_read_at TEXT,
    member_last_read_at TEXT,
    UNIQUE(coach_email, member_email)
);
CREATE TABLE IF NOT EXISTS coach_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id INTEGER NOT NULL,
    sender_role TEXT NOT NULL,
    body TEXT NOT NULL,
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_cmsg_thread ON coach_messages(thread_id, id);
CREATE TABLE IF NOT EXISTS coach_thread_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id INTEGER NOT NULL,
    reporter_role TEXT NOT NULL,
    reason TEXT,
    created_at TEXT,
    resolved INTEGER NOT NULL DEFAULT 0
);
"""


def _now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _lc(email):
    return (email or "").strip().lower()


def init_thread_tables(cx):
    cx.executescript(_DDL)
    cx.commit()


def get_or_create_thread(cx, *, coach_email, member_email, source="coaching"):
    ce, me = _lc(coach_email), _lc(member_email)
    cx.execute("INSERT OR IGNORE INTO coach_threads (source, coach_email, member_email, "
               "status, created_at) VALUES (?,?,?, 'active', ?)", (source, ce, me, _now()))
    cx.commit()
    return dict(cx.execute("SELECT * FROM coach_threads WHERE coach_email=? AND member_email=?",
                           (ce, me)).fetchone())


def get_thread(cx, thread_id):
    row = cx.execute("SELECT * FROM coach_threads WHERE id=?", (thread_id,)).fetchone()
    return dict(row) if row else None


def thread_for_pair(cx, coach_email, member_email):
    row = cx.execute("SELECT * FROM coach_threads WHERE coach_email=? AND member_email=?",
                     (_lc(coach_email), _lc(member_email))).fetchone()
    return dict(row) if row else None


def post_message(cx, *, thread_id, sender_role, body):
    cur = cx.execute("INSERT INTO coach_messages (thread_id, sender_role, body, created_at) "
                     "VALUES (?,?,?,?)", (thread_id, sender_role, body, _now()))
    cx.commit()
    return cur.lastrowid


def messages(cx, thread_id):
    rows = cx.execute("SELECT id, sender_role, body, created_at FROM coach_messages "
                      "WHERE thread_id=? ORDER BY id", (thread_id,)).fetchall()
    return [dict(r) for r in rows]


def mark_read(cx, thread_id, role):
    col = "coach_last_read_at" if role == "coach" else "member_last_read_at"
    cx.execute(f"UPDATE coach_threads SET {col}=? WHERE id=?", (_now(), thread_id))
    cx.commit()


def unread_count(cx, thread_id, role):
    col = "coach_last_read_at" if role == "coach" else "member_last_read_at"
    other = "member" if role == "coach" else "coach"
    row = cx.execute(f"SELECT {col} AS lr FROM coach_threads WHERE id=?", (thread_id,)).fetchone()
    last = (row["lr"] if row else None) or ""
    return cx.execute("SELECT COUNT(*) FROM coach_messages WHERE thread_id=? AND sender_role=? "
                      "AND created_at > ?", (thread_id, other, last)).fetchone()[0]


def block_thread(cx, thread_id, blocked_by_role):
    cx.execute("UPDATE coach_threads SET status='blocked', blocked_by=? WHERE id=?",
               (blocked_by_role, thread_id))
    cx.commit()


def report_thread(cx, *, thread_id, reporter_role, reason):
    cx.execute("INSERT INTO coach_thread_reports (thread_id, reporter_role, reason, created_at) "
               "VALUES (?,?,?,?)", (thread_id, reporter_role, (reason or "")[:500], _now()))
    cx.execute("UPDATE coach_threads SET reported=1 WHERE id=?", (thread_id,))
    cx.commit()


def list_all_threads(cx):
    rows = cx.execute(
        "SELECT t.*, "
        "(SELECT COUNT(*) FROM coach_messages m WHERE m.thread_id=t.id) AS message_count, "
        "(SELECT MAX(created_at) FROM coach_messages m WHERE m.thread_id=t.id) AS last_message_at "
        "FROM coach_threads t "
        "ORDER BY (t.reported=1 OR t.status='blocked') DESC, last_message_at DESC").fetchall()
    return [dict(r) for r in rows]
```

Add to `dashboard/coach_connect.py` (near the other accepted-row readers):

```python
def accepted_pair(cx, member_email):
    """The member's single accepted coach + that request id, or None."""
    row = cx.execute("SELECT id, coach_email FROM coach_requests WHERE member_email=? "
                     "AND status='accepted' ORDER BY id LIMIT 1", (_lc(member_email),)).fetchone()
    return {"request_id": row["id"], "coach_email": row["coach_email"]} if row else None


def accepted_members(cx, coach_email):
    """The coach's accepted members: [{request_id, member_email, member_name}]."""
    rows = cx.execute("SELECT id, member_email, member_name FROM coach_requests "
                      "WHERE coach_email=? AND status='accepted' ORDER BY id",
                      (_lc(coach_email),)).fetchall()
    return [{"request_id": r["id"], "member_email": r["member_email"],
             "member_name": r["member_name"]} for r in rows]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_coach_threads_store.py -q`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add dashboard/coach_threads.py dashboard/coach_connect.py tests/test_coach_threads_store.py
git commit -m "feat(coaching): thread store + accepted-pair counterpart helpers"
```

---

### Task 2: Member thread routes (`app.py`)

**Files:**
- Modify: `app.py` (member thread routes + a shared nudge helper)
- Test: `tests/test_coach_thread_member_api.py`

**Interfaces:**
- Consumes: `dashboard/coach_threads.py` (all), `dashboard/coach_connect.py` (`accepted_pair`, `set_request_status`, `accepted_count`), `dashboard/coach_directory.py` (`get_volunteer`), `_evox_ident`, `send_evox_email`, `GLEN_CONSULT_EMAIL`, `PUBLIC_BASE_URL`, `_db_lock`, `LOG_DB`.
- Produces: `GET /api/coach-thread/member`; `POST /api/coach-thread/member/message`; `POST /api/coach-thread/member/block`; `POST /api/coach-thread/member/report`; `_coach_thread_nudge(to_email, from_label)`; `COACH_MESSAGE_MAX_CHARS`.

**Contract:** all resolve `_evox_ident`→404, then `accepted_pair(email)`→404 if unmatched. GET materializes the thread, `mark_read(member)`, returns `{coach_name, status, messages, can_post}`; when `status='blocked'` returns `messages:[]` + `can_post:false` (history hidden). `coach_name` = `get_volunteer(coach_email)["name"]`, never email. POST message: 400 empty/oversized, 409 blocked, else post + best-effort nudge to the coach. Block: `block_thread('member')` + `set_request_status(pair.request_id,'ended')` in ONE locked write + owner email; idempotent. Report: `report_thread('member')` + owner email.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_coach_thread_member_api.py
import json, sqlite3
from unittest import mock
import app as appmod
from dashboard import coach_connect as _cc, coach_directory as _cd, coach_threads as _ct


def _client():
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()


def _matched(member="m@x.com", coach="c@x.com"):
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        from dashboard import evox as _ev, client_portal as _cp
        _ev.init_evox_tables(cx); _cp.init_client_portal_table(cx)
        _cc.init_connect_tables(cx); _cd.init_coach_tables(cx); _ct.init_thread_tables(cx)
        _cd.upsert_volunteer(cx, email=coach, name="Coach Kai", focus="terrain", capacity=5)
        _cc.create_request(cx, coach, member, "Mel P", "help")
        rid = cx.execute("SELECT id FROM coach_requests").fetchone()["id"]
        _cc.set_request_status(cx, rid, "accepted")
        t = _ev.ensure_portal_token(cx, member, "Mel"); cx.commit()
    return t


def test_get_materializes_shows_coach_name_not_email():
    c = _client(); tok = _matched()
    r = c.get(f"/api/coach-thread/member?token={tok}")
    d = r.get_json()
    assert d["coach_name"] == "Coach Kai" and d["status"] == "active"
    assert "c@x.com" not in json.dumps(d)          # never the coach email


def test_unmatched_member_404():
    c = _client()
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        from dashboard import evox as _ev, client_portal as _cp
        _ev.init_evox_tables(cx); _cp.init_client_portal_table(cx); _ct.init_thread_tables(cx)
        _cc.init_connect_tables(cx)
        tok = _ev.ensure_portal_token(cx, "solo@x.com", "Solo"); cx.commit()
    assert c.get(f"/api/coach-thread/member?token={tok}").status_code == 404


def test_post_message_rejects_empty_and_nudges():
    c = _client(); tok = _matched()
    assert c.post(f"/api/coach-thread/member/message?token={tok}",
                  json={"body": "  "}).status_code == 400
    with mock.patch.object(appmod, "send_evox_email") as sender:
        r = c.post(f"/api/coach-thread/member/message?token={tok}", json={"body": "hi coach"})
    assert r.get_json()["ok"] is True and sender.called


def test_block_ends_pairing_and_hides_history():
    c = _client(); tok = _matched()
    c.post(f"/api/coach-thread/member/message?token={tok}", json={"body": "hello"})
    with mock.patch.object(appmod, "send_evox_email"):
        assert c.post(f"/api/coach-thread/member/block?token={tok}").get_json()["ok"] is True
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row; _cc.init_connect_tables(cx)
        assert _cc.accepted_count(cx, "c@x.com") == 0          # capacity freed
        assert _cc.member_has_accepted(cx, "m@x.com") is False  # can re-apply
    d = c.get(f"/api/coach-thread/member?token={tok}").get_json()
    assert d["status"] == "blocked" and d["messages"] == [] and d["can_post"] is False
    assert c.post(f"/api/coach-thread/member/message?token={tok}",
                  json={"body": "more"}).status_code == 409     # blocked rejects posts


def test_report_flags_and_emails_owner():
    c = _client(); tok = _matched()
    with mock.patch.object(appmod, "send_evox_email") as sender:
        r = c.post(f"/api/coach-thread/member/report?token={tok}", json={"reason": "rude"})
    assert r.get_json()["ok"] is True and sender.called
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row; _ct.init_thread_tables(cx)
        assert _ct.thread_for_pair(cx, "c@x.com", "m@x.com")["reported"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `export DATA_DIR="$(mktemp -d)" && doppler run -p remedy-match -c prd -- env DATA_DIR="$DATA_DIR" python3 -m pytest tests/test_coach_thread_member_api.py -q`
Expected: FAIL — routes missing.

- [ ] **Step 3: Write minimal implementation**

Add to `app.py` (near the other community/coach routes ~16216-16560):

```python
COACH_MESSAGE_MAX_CHARS = int(os.environ.get("COACH_MESSAGE_MAX_CHARS", "4000"))


def _coach_thread_nudge(to_email, from_label):
    """Best-effort 'you have a new message' nudge. Never raises, never exposes the
    other party's email."""
    try:
        base = PUBLIC_BASE_URL.rstrip("/")
        html = (f"<p>You have a new message from {from_label}. Open your portal to read and "
                f"reply:</p><p><a href=\"{base}/\">Go to your portal</a></p>")
        send_evox_email(to_email, "", f"A new message from {from_label}", html, html, b"")
    except Exception:
        app.logger.exception("coach thread nudge failed")


def _coach_thread_owner_alert(subject, detail):
    try:
        send_evox_email(GLEN_CONSULT_EMAIL, "Glen", subject, f"<p>{detail}</p>",
                        f"<p>{detail}</p>", b"")
    except Exception:
        app.logger.exception("coach thread owner alert failed")


def _member_thread_ctx(cx, token):
    """(ident_email, pair) for a matched member, or (None, None)."""
    ident = _evox_ident(cx, token)
    if ident is None:
        return None, None
    from dashboard import coach_connect as _cc
    return ident.email, _cc.accepted_pair(cx, ident.email)


@app.route("/api/coach-thread/member")
def coach_thread_member_get():
    from dashboard import coach_threads as _ct, coach_directory as _cd
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _ct.init_thread_tables(cx); _cd.init_coach_tables(cx)
        email, pair = _member_thread_ctx(cx, request.args.get("token", ""))
        if email is None or pair is None:
            return jsonify({"error": "not_found"}), 404
        t = _ct.get_or_create_thread(cx, coach_email=pair["coach_email"], member_email=email)
        _ct.mark_read(cx, t["id"], "member")
        vol = _cd.get_volunteer(cx, pair["coach_email"]) or {}
        blocked = t["status"] == "blocked"
        return jsonify({"coach_name": vol.get("name") or "Your coach", "status": t["status"],
                        "can_post": not blocked,
                        "messages": [] if blocked else _ct.messages(cx, t["id"])})


@app.route("/api/coach-thread/member/message", methods=["POST"])
def coach_thread_member_message():
    from dashboard import coach_threads as _ct
    body = ((request.get_json(force=True) or {}).get("body") or "").strip()
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _ct.init_thread_tables(cx)
        email, pair = _member_thread_ctx(cx, request.args.get("token", ""))
        if email is None or pair is None:
            return jsonify({"error": "not_found"}), 404
        if not body or len(body) > COACH_MESSAGE_MAX_CHARS:
            return jsonify({"error": "bad_body"}), 400
        t = _ct.get_or_create_thread(cx, coach_email=pair["coach_email"], member_email=email)
        if t["status"] == "blocked":
            return jsonify({"error": "blocked"}), 409
        _ct.post_message(cx, thread_id=t["id"], sender_role="member", body=body)
        coach_email = pair["coach_email"]
    _coach_thread_nudge(coach_email, "your client")
    return jsonify({"ok": True})


@app.route("/api/coach-thread/member/block", methods=["POST"])
def coach_thread_member_block():
    from dashboard import coach_threads as _ct, coach_connect as _cc
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _ct.init_thread_tables(cx); _cc.init_connect_tables(cx)
        email, pair = _member_thread_ctx(cx, request.args.get("token", ""))
        if email is None or pair is None:
            return jsonify({"error": "not_found"}), 404
        t = _ct.get_or_create_thread(cx, coach_email=pair["coach_email"], member_email=email)
        _ct.block_thread(cx, t["id"], "member")
        _cc.set_request_status(cx, pair["request_id"], "ended")
    _coach_thread_owner_alert("A member ended a coaching pairing",
                              "A member blocked their coach; the pairing has ended.")
    return jsonify({"ok": True})


@app.route("/api/coach-thread/member/report", methods=["POST"])
def coach_thread_member_report():
    from dashboard import coach_threads as _ct
    reason = ((request.get_json(force=True) or {}).get("reason") or "").strip()
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _ct.init_thread_tables(cx)
        email, pair = _member_thread_ctx(cx, request.args.get("token", ""))
        if email is None or pair is None:
            return jsonify({"error": "not_found"}), 404
        t = _ct.get_or_create_thread(cx, coach_email=pair["coach_email"], member_email=email)
        _ct.report_thread(cx, thread_id=t["id"], reporter_role="member", reason=reason)
    _coach_thread_owner_alert("A coaching thread was reported",
                              "A member reported their coaching thread. Review it in the console.")
    return jsonify({"ok": True})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `export DATA_DIR="$(mktemp -d)" && doppler run -p remedy-match -c prd -- env DATA_DIR="$DATA_DIR" python3 -m pytest tests/test_coach_thread_member_api.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_coach_thread_member_api.py
git commit -m "feat(coaching): member thread routes (read/post/block/report)"
```

---

### Task 3: Coach thread routes (`app.py`)

**Files:**
- Modify: `app.py` (coach thread routes)
- Test: `tests/test_coach_thread_coach_api.py`

**Interfaces:**
- Consumes: `dashboard/coach_threads.py`, `dashboard/coach_connect.py` (`accepted_members`), `dashboard/client_portal.py` (`get_portal_content_by_email`), `_coach_session_email`, `_coach_thread_nudge` (Task 2), `_coach_thread_owner_alert` (Task 2), `_db_lock`, `LOG_DB`.
- Produces: `GET /api/coach-thread/coach/list`; `GET /api/coach-thread/coach/<int:thread_id>`; `POST /api/coach-thread/coach/<int:thread_id>/message`; `POST /api/coach-thread/coach/<int:thread_id>/report`.

**Contract:** `_coach_session_email()`→401 first. List: for each `accepted_members(coach_email)`, materialize the thread, return `[{member_first_name, thread_id, status, unread}]` (first name via `get_portal_content_by_email(member_email)["name"].split()[0]`, never email). GET/POST `<thread_id>`: 403 unless the thread's `coach_email == coach_email`; GET `mark_read(coach)` + blocked hides history; POST 400 empty/oversized, 409 blocked, else post + nudge the member. Report: `report_thread('coach')` + owner alert.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_coach_thread_coach_api.py
import json, sqlite3
from unittest import mock
import app as appmod
from dashboard import coach_connect as _cc, coach_directory as _cd, coach_threads as _ct


def _client():
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()


def _accepted(coach="c@x.com", member="m@x.com", name="Mel P"):
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        from dashboard import client_portal as _cp
        _cp.init_client_portal_table(cx); _cc.init_connect_tables(cx)
        _cd.init_coach_tables(cx); _ct.init_thread_tables(cx)
        _cp.upsert_portal_content(cx, email=member, name="Mel Palakiko", content="{}")
        _cc.create_request(cx, coach, member, name, "help")
        rid = cx.execute("SELECT id FROM coach_requests WHERE member_email=?", (member,)).fetchone()["id"]
        _cc.set_request_status(cx, rid, "accepted")
        t = _ct.get_or_create_thread(cx, coach_email=coach, member_email=member); cx.commit()
    return t["id"]


def test_list_shows_first_name_not_email():
    c = _client(); _accepted()
    with mock.patch.object(appmod, "_coach_session_email", return_value="c@x.com"):
        d = c.get("/api/coach-thread/coach/list").get_json()
    assert d[0]["member_first_name"] == "Mel" and "m@x.com" not in json.dumps(d)


def test_unauthorized_401():
    c = _client()
    with mock.patch.object(appmod, "_coach_session_email", return_value=None):
        assert c.get("/api/coach-thread/coach/list").status_code == 401


def test_get_post_403_when_not_owner():
    c = _client(); tid = _accepted()
    with mock.patch.object(appmod, "_coach_session_email", return_value="other@x.com"):
        assert c.get(f"/api/coach-thread/coach/{tid}").status_code == 403
        assert c.post(f"/api/coach-thread/coach/{tid}/message",
                      json={"body": "hi"}).status_code == 403


def test_post_message_nudges_member():
    c = _client(); tid = _accepted()
    with mock.patch.object(appmod, "_coach_session_email", return_value="c@x.com"), \
         mock.patch.object(appmod, "send_evox_email") as sender:
        r = c.post(f"/api/coach-thread/coach/{tid}/message", json={"body": "welcome"})
    assert r.get_json()["ok"] is True and sender.called


def test_blocked_hides_history_for_coach():
    c = _client(); tid = _accepted()
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row; _ct.init_thread_tables(cx)
        _ct.post_message(cx, thread_id=tid, sender_role="member", body="hi")
        _ct.block_thread(cx, tid, "member")
    with mock.patch.object(appmod, "_coach_session_email", return_value="c@x.com"):
        d = c.get(f"/api/coach-thread/coach/{tid}").get_json()
    assert d["status"] == "blocked" and d["messages"] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `export DATA_DIR="$(mktemp -d)" && doppler run -p remedy-match -c prd -- env DATA_DIR="$DATA_DIR" python3 -m pytest tests/test_coach_thread_coach_api.py -q`
Expected: FAIL — routes missing.

- [ ] **Step 3: Write minimal implementation**

Add to `app.py` (after the Task 2 member routes):

```python
def _coach_first_name(cx, member_email):
    from dashboard import client_portal as _cp
    row = _cp.get_portal_content_by_email(cx, member_email) or {}
    return ((row.get("name") or "").strip().split() or ["Your client"])[0]


@app.route("/api/coach-thread/coach/list")
def coach_thread_coach_list():
    email = _coach_session_email()
    if not email:
        return jsonify({"error": "unauthorized"}), 401
    from dashboard import coach_threads as _ct, coach_connect as _cc
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _ct.init_thread_tables(cx); _cc.init_connect_tables(cx)
        out = []
        for m in _cc.accepted_members(cx, email):
            t = _ct.get_or_create_thread(cx, coach_email=email, member_email=m["member_email"])
            out.append({"member_first_name": _coach_first_name(cx, m["member_email"]),
                        "thread_id": t["id"], "status": t["status"],
                        "unread": _ct.unread_count(cx, t["id"], "coach")})
        return jsonify(out)


def _coach_owns(cx, thread_id, coach_email):
    from dashboard import coach_threads as _ct
    t = _ct.get_thread(cx, thread_id)
    return t if (t and t["coach_email"] == (coach_email or "").strip().lower()) else None


@app.route("/api/coach-thread/coach/<int:thread_id>")
def coach_thread_coach_get(thread_id):
    email = _coach_session_email()
    if not email:
        return jsonify({"error": "unauthorized"}), 401
    from dashboard import coach_threads as _ct
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _ct.init_thread_tables(cx)
        t = _coach_owns(cx, thread_id, email)
        if not t:
            return jsonify({"error": "forbidden"}), 403
        _ct.mark_read(cx, thread_id, "coach")
        blocked = t["status"] == "blocked"
        return jsonify({"member_first_name": _coach_first_name(cx, t["member_email"]),
                        "status": t["status"], "can_post": not blocked,
                        "messages": [] if blocked else _ct.messages(cx, thread_id)})


@app.route("/api/coach-thread/coach/<int:thread_id>/message", methods=["POST"])
def coach_thread_coach_message(thread_id):
    email = _coach_session_email()
    if not email:
        return jsonify({"error": "unauthorized"}), 401
    from dashboard import coach_threads as _ct
    body = ((request.get_json(force=True) or {}).get("body") or "").strip()
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _ct.init_thread_tables(cx)
        t = _coach_owns(cx, thread_id, email)
        if not t:
            return jsonify({"error": "forbidden"}), 403
        if not body or len(body) > COACH_MESSAGE_MAX_CHARS:
            return jsonify({"error": "bad_body"}), 400
        if t["status"] == "blocked":
            return jsonify({"error": "blocked"}), 409
        _ct.post_message(cx, thread_id=thread_id, sender_role="coach", body=body)
        member_email = t["member_email"]
    _coach_thread_nudge(member_email, "your coach")
    return jsonify({"ok": True})


@app.route("/api/coach-thread/coach/<int:thread_id>/report", methods=["POST"])
def coach_thread_coach_report(thread_id):
    email = _coach_session_email()
    if not email:
        return jsonify({"error": "unauthorized"}), 401
    from dashboard import coach_threads as _ct
    reason = ((request.get_json(force=True) or {}).get("reason") or "").strip()
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _ct.init_thread_tables(cx)
        t = _coach_owns(cx, thread_id, email)
        if not t:
            return jsonify({"error": "forbidden"}), 403
        _ct.report_thread(cx, thread_id=thread_id, reporter_role="coach", reason=reason)
    _coach_thread_owner_alert("A coaching thread was reported",
                              "A coach reported their coaching thread. Review it in the console.")
    return jsonify({"ok": True})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `export DATA_DIR="$(mktemp -d)" && doppler run -p remedy-match -c prd -- env DATA_DIR="$DATA_DIR" python3 -m pytest tests/test_coach_thread_coach_api.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_coach_thread_coach_api.py
git commit -m "feat(coaching): coach thread routes (list/read/post/report)"
```

---

### Task 4: Owner moderation routes + console panel

**Files:**
- Modify: `app.py` (owner moderation routes)
- Modify: `static/console-biofield-portal.html` (a "Coaching threads" panel)
- Test: `tests/test_coach_thread_owner_api.py`

**Interfaces:**
- Consumes: `dashboard/coach_threads.py` (`list_all_threads`, `get_thread`, `messages`, `block_thread`), `dashboard/coach_connect.py` (`accepted_pair`/its request id, `set_request_status`), `_portal_console_ok`, `_coach_thread_owner_alert`/`send_evox_email`, `_db_lock`, `LOG_DB`.
- Produces: `GET /api/console/coach-threads`; `GET /api/console/coach-threads/<int:thread_id>`; `POST /api/console/coach-threads/<int:thread_id>/unmatch`; `POST /api/console/coach-threads/<int:thread_id>/resolve-report`.

**Contract:** all gate on `_portal_console_ok()`→401. List returns `list_all_threads` (reported/blocked first; owner may see both emails — moderation context). Transcript returns the full thread incl blocked history + report rows. Unmatch: `block_thread('owner')` + set the pair's accepted request to `ended` (resolve request id via the thread's coach+member email) + email both parties; idempotent (no-op if already blocked/ended). Resolve-report: clear `reported` + mark report rows resolved.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_coach_thread_owner_api.py
import sqlite3
from unittest import mock
import app as appmod
from dashboard import coach_connect as _cc, coach_threads as _ct


def _client():
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()


def _seed(coach="c@x.com", member="m@x.com"):
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _cc.init_connect_tables(cx); _ct.init_thread_tables(cx)
        _cc.create_request(cx, coach, member, "Mel P", "help")
        rid = cx.execute("SELECT id FROM coach_requests WHERE member_email=?", (member,)).fetchone()["id"]
        _cc.set_request_status(cx, rid, "accepted")
        t = _ct.get_or_create_thread(cx, coach_email=coach, member_email=member)
        _ct.post_message(cx, thread_id=t["id"], sender_role="member", body="secret")
        _ct.block_thread(cx, t["id"], "member"); cx.commit()
    return t["id"]


def test_all_owner_routes_401_without_key():
    c = _client()
    with mock.patch.object(appmod, "_portal_console_ok", return_value=False):
        assert c.get("/api/console/coach-threads").status_code == 401
        assert c.get("/api/console/coach-threads/1").status_code == 401
        assert c.post("/api/console/coach-threads/1/unmatch").status_code == 401


def test_transcript_shows_blocked_history():
    c = _client(); tid = _seed()
    with mock.patch.object(appmod, "_portal_console_ok", return_value=True):
        d = c.get(f"/api/console/coach-threads/{tid}").get_json()
    assert d["status"] == "blocked"
    assert any(m["body"] == "secret" for m in d["messages"])   # owner sees hidden history


def test_unmatch_blocks_and_ends_pairing():
    c = _client()
    # fresh active pair (not pre-blocked)
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _cc.init_connect_tables(cx); _ct.init_thread_tables(cx)
        _cc.create_request(cx, "c2@x.com", "m2@x.com", "Ana", "hi")
        rid = cx.execute("SELECT id FROM coach_requests WHERE member_email='m2@x.com'").fetchone()["id"]
        _cc.set_request_status(cx, rid, "accepted")
        t = _ct.get_or_create_thread(cx, coach_email="c2@x.com", member_email="m2@x.com"); cx.commit()
    with mock.patch.object(appmod, "_portal_console_ok", return_value=True), \
         mock.patch.object(appmod, "send_evox_email"):
        assert c.post(f"/api/console/coach-threads/{t['id']}/unmatch").get_json()["ok"] is True
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row; _cc.init_connect_tables(cx); _ct.init_thread_tables(cx)
        assert _ct.get_thread(cx, t["id"])["status"] == "blocked"
        assert _cc.accepted_count(cx, "c2@x.com") == 0


def test_resolve_report_clears_flag():
    c = _client()
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row; _cc.init_connect_tables(cx); _ct.init_thread_tables(cx)
        t = _ct.get_or_create_thread(cx, coach_email="c3@x.com", member_email="m3@x.com")
        _ct.report_thread(cx, thread_id=t["id"], reporter_role="member", reason="x"); cx.commit()
    with mock.patch.object(appmod, "_portal_console_ok", return_value=True):
        assert c.post(f"/api/console/coach-threads/{t['id']}/resolve-report").get_json()["ok"] is True
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row; _ct.init_thread_tables(cx)
        assert _ct.get_thread(cx, t["id"])["reported"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `export DATA_DIR="$(mktemp -d)" && doppler run -p remedy-match -c prd -- env DATA_DIR="$DATA_DIR" python3 -m pytest tests/test_coach_thread_owner_api.py -q`
Expected: FAIL — routes missing.

- [ ] **Step 3: Write minimal implementation**

Add to `app.py` (near the other console routes):

```python
@app.route("/api/console/coach-threads")
def console_coach_threads():
    if not _portal_console_ok():
        return jsonify({"error": "unauthorized"}), 401
    from dashboard import coach_threads as _ct
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _ct.init_thread_tables(cx)
        return jsonify({"threads": _ct.list_all_threads(cx)})


@app.route("/api/console/coach-threads/<int:thread_id>")
def console_coach_thread_transcript(thread_id):
    if not _portal_console_ok():
        return jsonify({"error": "unauthorized"}), 401
    from dashboard import coach_threads as _ct
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _ct.init_thread_tables(cx)
        t = _ct.get_thread(cx, thread_id)
        if not t:
            return jsonify({"error": "not_found"}), 404
        reports = [dict(r) for r in cx.execute(
            "SELECT reporter_role, reason, created_at, resolved FROM coach_thread_reports "
            "WHERE thread_id=? ORDER BY id", (thread_id,)).fetchall()]
        return jsonify({**t, "messages": _ct.messages(cx, thread_id), "reports": reports})


@app.route("/api/console/coach-threads/<int:thread_id>/unmatch", methods=["POST"])
def console_coach_thread_unmatch(thread_id):
    if not _portal_console_ok():
        return jsonify({"error": "unauthorized"}), 401
    from dashboard import coach_threads as _ct, coach_connect as _cc
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _ct.init_thread_tables(cx); _cc.init_connect_tables(cx)
        t = _ct.get_thread(cx, thread_id)
        if not t:
            return jsonify({"error": "not_found"}), 404
        _ct.block_thread(cx, thread_id, "owner")
        pair = _cc.accepted_pair(cx, t["member_email"])
        if pair and pair["coach_email"] == t["coach_email"]:
            _cc.set_request_status(cx, pair["request_id"], "ended")
        member_email, coach_email = t["member_email"], t["coach_email"]
    try:
        note = ("<p>Your coaching pairing has ended. You are welcome to choose another coach "
                "from your portal whenever you are ready.</p>")
        send_evox_email(member_email, "", "Your coaching pairing has ended", note, note, b"")
        send_evox_email(coach_email, "", "A coaching pairing has ended", note, note, b"")
    except Exception:
        app.logger.exception("unmatch notify failed")
    return jsonify({"ok": True})


@app.route("/api/console/coach-threads/<int:thread_id>/resolve-report", methods=["POST"])
def console_coach_thread_resolve_report(thread_id):
    if not _portal_console_ok():
        return jsonify({"error": "unauthorized"}), 401
    from dashboard import coach_threads as _ct
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _ct.init_thread_tables(cx)
        cx.execute("UPDATE coach_threads SET reported=0 WHERE id=?", (thread_id,))
        cx.execute("UPDATE coach_thread_reports SET resolved=1 WHERE thread_id=?", (thread_id,))
        cx.commit()
    return jsonify({"ok": True})
```

- [ ] **Step 4: Add the console panel**

In `static/console-biofield-portal.html`, add a "Coaching threads" panel: a button that fetches `GET /api/console/coach-threads` (sending the console key the page already uses for its other console fetches), renders a list (coach/member emails, status, reported flag, message count) with a "View" action that fetches the transcript into a readable pane, plus "Unmatch" and "Resolve report" buttons posting the respective routes. All dynamic strings via `textContent`. Mirror the page's existing console-fetch helper and panel styling. Wrap in `<!-- BEGIN coach-threads console panel -->` / `<!-- END coach-threads console panel -->`.

- [ ] **Step 5: Run test + verify console JS parses**

Run: `export DATA_DIR="$(mktemp -d)" && doppler run -p remedy-match -c prd -- env DATA_DIR="$DATA_DIR" python3 -m pytest tests/test_coach_thread_owner_api.py -q`
Expected: PASS (4 passed)
Run: `cd /tmp/wt-deploy-chat-cca589e9 && node --check <(python3 -c "import re;h=open('static/console-biofield-portal.html').read();print('\n;\n'.join(re.findall(r'<script>(.*?)</script>',h,re.S)))")`
Expected: no output.

- [ ] **Step 6: Commit**

```bash
git add app.py static/console-biofield-portal.html tests/test_coach_thread_owner_api.py
git commit -m "feat(coaching): owner thread moderation routes + console panel"
```

---

### Task 5: Member portal thread surface (`static/client-portal.html`)

**Files:**
- Modify: `static/client-portal.html` (thread UI in the matched-coach state)
- Test: JS parse check.

**Interfaces:**
- Consumes: `GET /api/coach-thread/member`, `POST /api/coach-thread/member/message`, `POST /api/coach-thread/member/block`, `POST /api/coach-thread/member/report`.

**Design note:** In `renderCoachesCard` (the coaches card, ~line 1304), the matched branch currently shows "You are matched with a coach." Extend that branch: when `d.matched` is true, fetch `GET /api/coach-thread/member?token=...` and render a thread panel below the message — the coach display name as a heading, the messages reverse-chronological (each as a row with `sender_role` label "You"/coach name and body via `textContent`), a compose box + Send button posting `/message` then re-fetching, and two quiet links: **Block** (confirm "End this coaching pairing? This cannot be undone." → POST `/block` → on ok show "This coaching conversation has ended." and clear the panel) and **Report** (prompt for a short reason → POST `/report` → "Thank you, we will review this."). When `status==='blocked'`, render only the calm "This coaching conversation has ended." note (no history, no compose). All strings via `textContent`; no em dashes, no ALL CAPS. Wrap in `<!-- BEGIN coach-thread member script -->` / `<!-- END coach-thread member script -->`.

- [ ] **Step 1: Add the member thread panel**

Read `renderCoachesCard`/the matched branch and add the thread fetch + panel + compose + block/report handlers as described.

- [ ] **Step 2: Verify the page JS parses**

Run: `cd /tmp/wt-deploy-chat-cca589e9 && node --check <(python3 -c "import re;h=open('static/client-portal.html').read();print('\n;\n'.join(re.findall(r'<script>(.*?)</script>',h,re.S)))")`
Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add static/client-portal.html
git commit -m "feat(coaching): member coaching thread panel"
```

---

### Task 6: Coach portal thread surface (`static/practitioner-portal.html`)

**Files:**
- Modify: `static/practitioner-portal.html` (thread panel in the coach-requests area)
- Test: JS parse check.

**Interfaces:**
- Consumes: `GET /api/coach-thread/coach/list`, `GET /api/coach-thread/coach/<id>`, `POST /api/coach-thread/coach/<id>/message`, `POST /api/coach-thread/coach/<id>/report`.

**Design note:** Below the existing `coach-requests-coachees` list (~line 371), add a **"Messages"** panel: on load (and after the coach-requests fetch), call `GET /api/coach-thread/coach/list` (with the practitioner session token the page already uses) and render one row per member — first name + an unread badge — each opening the thread via `GET /api/coach-thread/coach/<thread_id>`. The open thread shows messages (via `textContent`, "You"/member first name labels), a compose box + Send posting `/message` then re-fetching, and a quiet **Report** link (reason prompt → POST `/report`). `status==='blocked'` → calm "This conversation has ended." note, no compose. No block button on the coach side. All strings via `textContent`; no em dashes, no ALL CAPS. Mirror the page's existing fetch/token helper and panel styling. Wrap in `<!-- BEGIN coach-thread coach script -->` / `<!-- END coach-thread coach script -->`.

- [ ] **Step 1: Add the coach thread panel**

Read the coach-requests panel and add the list + open-thread + compose + report handlers as described.

- [ ] **Step 2: Verify the page JS parses**

Run: `cd /tmp/wt-deploy-chat-cca589e9 && node --check <(python3 -c "import re;h=open('static/practitioner-portal.html').read();print('\n;\n'.join(re.findall(r'<script>(.*?)</script>',h,re.S)))")`
Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add static/practitioner-portal.html
git commit -m "feat(coaching): coach messages panel in practitioner portal"
```

---

## Definition of Done

- A matched member and coach exchange text messages, each nudged by email when the other replies (no email ever exposed). The member can block (ends the pairing, frees the coach's capacity, hides history from both, notifies the owner) or report; the coach can report. The owner reads any thread from the console, resolves reports, and unmatches.
- Privacy holds: no counterpart email in any member/coach payload; blocked history is owner-only; a third member cannot read a pair's thread; the coach sees only the member first name.
- `coach_requests.status='ended'` frees capacity and lets the member re-apply, and is treated as accepted nowhere.
- All new tests pass; coaching slices 1-2, community, and the appointment loop are untouched (additive tables + routes + reused helpers).

## Deferred (not in this plan)

- Peer-matching arc (`source='peer'` threads, two `member` participants, member↔member auth) reusing these tables/routes.
- Media/attachments, live chat, group threads, coach-initiated block, per-thread mute, unread receipts shown to the other party, a coach-rating loop.
