# Coaching Request/Accept + Waitlist + Upsell Offer (coaching arc, slice 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A member applies to a free student coach (double opt-in: apply → accept up to capacity), overflow goes to a waitlist, and when all coaches are full an upsell offer presents paid coaching (Rae $100/EVOX, Glen $200/Causal Biofield) with an "I'm interested" flag.

**Architecture:** A new pure-sqlite `dashboard/coach_connect.py` (requests, waitlist, interest, opaque coach ref) plus a server-side `list_active_full` on the slice-1 `coach_directory`; member routes (apply/waitlist/interest + the extended directory) and coach routes (review/respond) in `app.py`; the two portal surfaces. No payments in this slice (the paid tiers are display-only copy).

**Tech Stack:** Python 3 / Flask (single `app.py` + `dashboard/*.py`), sqlite (`chat_log.db`, `?` placeholders, `_db_lock`, `cx.row_factory = sqlite3.Row`), `hashlib` for the opaque ref, vanilla JS/HTML.

## Global Constraints

- **Privacy (load-bearing):** a coach's **email is never exposed** to a member — coaches are referenced by an opaque `ref` (`sha256(lower(email))[:16]`). A member is shown to a requested coach as **first name + note only**, never email. No coach payload contains a member email; no member payload contains a coach email.
- **Member application video (optional):** the member may also upload a short intro video ("what I'm looking for help with"), self-hosted via the existing `/portal-asset` mechanism as `member-<hex>.mp4` (same pattern as the coach intro video). It is shown ONLY to the coach the member applied to (the member consented by applying). Filming in-browser is a fast-follow; this slice is upload.
- **Double opt-in, multi-apply, first-accept-wins:** the member APPLIES to one or MORE coaches (each a pending request + note + optional video); a coach ACCEPTS. A member may hold multiple **pending** applications, but at most one **accepted** coach — a new application is blocked only once the member is already matched (`already_matched`). When a coach accepts, that **claims** the member: the member's other pending applications are withdrawn (status `withdrawn`). If two coaches race, the first accept wins; a later accept of an already-matched member is refused (`member_taken`). A coach accepts only while `accepted_count < capacity` (else `at_capacity`). Full coaches (accepted_count >= capacity) are excluded from the member directory. (The accept path runs under `_db_lock`, which serializes the claim within a process.)
- **No payment in this slice:** the paid tiers (Rae $100/mo incl. an EVOX session; Dr. Glen $200/mo incl. a Causal Biofield Analysis) are display-only copy + an "I'm interested" flag that notifies Glen. No charge, no checkout (that is slice 2b).
- **Copy:** no em dashes, no ALL CAPS; server strings via `textContent`. Warm, low-pressure ("Not now" not "Reject").
- sqlite writes under `with _db_lock, sqlite3.connect(LOG_DB)`; emails lowercased; the member note is trimmed and capped at 500 chars.
- DRY, YAGNI, TDD, frequent commits.

**Repo facts the implementer needs:**
- Slice 1 (`dashboard/coach_directory.py`): `init_coach_tables(cx)`, `list_active(cx)` (member-safe `{name,focus,intro_video_url}`), `_lc(email)`. The `coach_volunteers` table has `email,name,focus,intro_video_url,capacity,active,cert_ok`.
- `app.py`: `community_coaches` route (app.py:16217) currently returns `{eligible, coaches: list_active(cx)}`; `_evox_ident(cx, token)` (member) → `.email` or None; `coaching.active_window(cx, email)`; `_practitioner_session_pid()` + `dashboard.practitioner_portal.practitioner_email_by_id`; `send_evox_email(to, name, subject, html, text, ics_bytes)` + `GLEN_CONSULT_EMAIL`; `_db_lock`; `LOG_DB`.
- Member/practitioner portal pages: `static/client-portal.html` (the "Meet your coaches" card from slice 1; member `token` var), `static/practitioner-portal.html` (the "Coaching volunteer" control from slice 1; practitioner session token).

**Testing note (READ FIRST):**
- Pure/store tests (Task 1) do NOT import app — plain `python3 -m pytest <path> -q`.
- Route tests (Tasks 2-3) `import app`; override DATA_DIR:
  ```
  export DATA_DIR="$(mktemp -d)" && doppler run -p remedy-match -c prd -- env DATA_DIR="$DATA_DIR" python3 -m pytest <paths> -q
  ```

---

### Task 1: Connect store + server-side coach list (`dashboard/coach_connect.py`, `coach_directory.py`)

**Files:**
- Create: `dashboard/coach_connect.py`
- Modify: `dashboard/coach_directory.py` (append `list_active_full`)
- Test: `tests/test_coach_connect.py`

**Interfaces:**
- Consumes: `dashboard/coach_directory.py:list_active_full` (this task adds it).
- Produces:
  - `coach_directory.list_active_full(cx) -> [dict]` — active+cert_ok volunteers with `{email, name, focus, intro_video_url, capacity}` (server-side; includes email + capacity).
  - `coach_connect.init_connect_tables(cx)`; `coach_ref(email) -> str`; `email_for_ref(cx, ref) -> str|None`; `create_request(cx, coach_email, member_email, member_name, note, member_video_url="") -> int|None` (multi-apply allowed; None if already matched or already applied to this coach); `member_has_accepted(cx, member_email) -> bool`; `member_applications(cx, member_email) -> [{coach_email,status}]`; `request_member(cx, request_id) -> email|None`; `request_owner(cx, request_id) -> coach_email|None`; `withdraw_other_pendings(cx, member_email, keep_request_id)`; `requests_for_coach(cx, coach_email, status="pending") -> [dict]` (each `{id,member_name,note,member_video_url,status}`); `accepted_count(cx, coach_email) -> int`; `set_request_status(cx, request_id, status)`; `join_waitlist(cx, member_email)`; `on_waitlist(cx, member_email) -> bool`; `record_interest(cx, member_email, tier)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_coach_connect.py
import sqlite3
from dashboard import coach_directory as _cd
from dashboard import coach_connect as _cc


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    _cd.init_coach_tables(cx)
    _cc.init_connect_tables(cx)
    return cx


def test_coach_ref_stable_and_resolves():
    cx = _cx()
    _cd.upsert_volunteer(cx, email="Coach@X.com", name="Cora", focus="sleep",
                         intro_video_url="u", capacity=2, cert_ok=1)
    ref = _cc.coach_ref("coach@x.com")
    assert ref == _cc.coach_ref("COACH@X.com") and len(ref) == 16
    assert _cc.email_for_ref(cx, ref) == "coach@x.com"
    assert _cc.email_for_ref(cx, "deadbeefdeadbeef") is None


def test_email_for_ref_only_active():
    cx = _cx()
    _cd.upsert_volunteer(cx, email="off@x.com", name="Off", focus="f", intro_video_url="u",
                         capacity=2, cert_ok=1)
    _cd.set_active(cx, "off@x.com", 0)
    assert _cc.email_for_ref(cx, _cc.coach_ref("off@x.com")) is None  # inactive not resolvable


def test_multi_apply_allowed_until_matched():
    cx = _cx()
    r1 = _cc.create_request(cx, "c1@x.com", "m@x.com", "Mel", "trouble sleeping")
    r2 = _cc.create_request(cx, "c2@x.com", "m@x.com", "Mel", "and adrenals")
    assert r1 is not None and r2 is not None                    # two pendings OK
    dup = _cc.create_request(cx, "c1@x.com", "m@x.com", "Mel", "dup")
    assert dup is None                                          # already applied to c1
    assert [a["coach_email"] for a in _cc.member_applications(cx, "m@x.com")] == ["c1@x.com", "c2@x.com"]
    # once matched, no new applications
    _cc.set_request_status(cx, r1, "accepted")
    assert _cc.member_has_accepted(cx, "m@x.com") is True
    assert _cc.create_request(cx, "c3@x.com", "m@x.com", "Mel", "more") is None


def test_first_accept_withdraws_other_pendings():
    cx = _cx()
    r1 = _cc.create_request(cx, "c1@x.com", "m@x.com", "Mel", "n1")
    r2 = _cc.create_request(cx, "c2@x.com", "m@x.com", "Mel", "n2")
    _cc.set_request_status(cx, r1, "accepted")
    _cc.withdraw_other_pendings(cx, "m@x.com", r1)
    apps = {a["coach_email"]: a["status"] for a in _cc.member_applications(cx, "m@x.com")}
    assert apps == {"c1@x.com": "accepted"}                     # c2 pending withdrawn (not active)
    assert _cc.request_member(cx, r2) == "m@x.com"


def test_accept_flow_and_count():
    cx = _cx()
    rid = _cc.create_request(cx, "c@x.com", "m@x.com", "Mel", "note")
    assert _cc.accepted_count(cx, "c@x.com") == 0
    _cc.set_request_status(cx, rid, "accepted")
    assert _cc.accepted_count(cx, "c@x.com") == 1
    pend = _cc.requests_for_coach(cx, "c@x.com", status="pending")
    assert pend == []                                     # no longer pending
    _cc.set_request_status(cx, rid, "declined")
    assert _cc.accepted_count(cx, "c@x.com") == 0


def test_requests_for_coach_has_name_note_video_no_email():
    cx = _cx()
    _cc.create_request(cx, "c@x.com", "m@x.com", "Mel", "working on adrenals",
                       member_video_url="/portal-asset/member-ab.mp4")
    pend = _cc.requests_for_coach(cx, "c@x.com")
    assert pend[0]["member_name"] == "Mel" and pend[0]["note"] == "working on adrenals"
    assert pend[0]["member_video_url"] == "/portal-asset/member-ab.mp4"
    assert "member_email" not in pend[0] and "email" not in pend[0]


def test_waitlist_and_interest():
    cx = _cx()
    assert _cc.on_waitlist(cx, "m@x.com") is False
    _cc.join_waitlist(cx, "M@x.com")
    assert _cc.on_waitlist(cx, "m@x.com") is True
    _cc.record_interest(cx, "m@x.com", "glen")
    _cc.record_interest(cx, "m@x.com", "glen")             # idempotent per (member,tier)
    assert cx.execute("SELECT COUNT(*) FROM coaching_interest").fetchone()[0] == 1


def test_list_active_full_has_email_and_capacity():
    cx = _cx()
    _cd.upsert_volunteer(cx, email="c@x.com", name="Cora", focus="sleep",
                         intro_video_url="u", capacity=3, cert_ok=1)
    full = _cd.list_active_full(cx)
    assert full[0]["email"] == "c@x.com" and full[0]["capacity"] == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_coach_connect.py -q`
Expected: FAIL — `dashboard.coach_connect` / `list_active_full` missing.

- [ ] **Step 3: Write minimal implementation**

Append to `dashboard/coach_directory.py`:

```python
def list_active_full(cx):
    """Server-side: active+cert_ok volunteers WITH email + capacity (for ref +
    capacity composition in the route). Never send this shape to a member."""
    rows = cx.execute("SELECT email, name, focus, intro_video_url, capacity "
                      "FROM coach_volunteers WHERE active=1 AND cert_ok=1 "
                      "ORDER BY updated_at DESC").fetchall()
    return [{"email": r["email"], "name": r["name"], "focus": r["focus"],
             "intro_video_url": r["intro_video_url"], "capacity": r["capacity"]}
            for r in rows]
```

Create `dashboard/coach_connect.py`:

```python
# dashboard/coach_connect.py
"""Coaching pairing (arc slice 2): member applications, waitlist, paid-tier
interest, and the opaque coach ref. Pure sqlite; no app-layer imports. Double
opt-in: the member applies (pending request + note), the coach accepts.
Privacy: coaches are referenced by an opaque ref (sha256 of email), and a
member is shown to a coach as first name + note only."""

import hashlib

_DDL = """
CREATE TABLE IF NOT EXISTS coach_requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    coach_email TEXT NOT NULL,
    member_email TEXT NOT NULL,
    member_name TEXT,
    note TEXT,
    member_video_url TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT,
    decided_at TEXT,
    UNIQUE(coach_email, member_email)
);
CREATE INDEX IF NOT EXISTS ix_creq_coach ON coach_requests(coach_email, status);
CREATE INDEX IF NOT EXISTS ix_creq_member ON coach_requests(member_email, status);
CREATE TABLE IF NOT EXISTS coach_waitlist (
    member_email TEXT PRIMARY KEY,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS coaching_interest (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    member_email TEXT NOT NULL,
    tier TEXT NOT NULL,
    created_at TEXT,
    UNIQUE(member_email, tier)
);
"""


def _now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _lc(email):
    return (email or "").strip().lower()


def init_connect_tables(cx):
    cx.executescript(_DDL)
    cx.commit()


def coach_ref(email):
    return hashlib.sha256(_lc(email).encode("utf-8")).hexdigest()[:16]


def email_for_ref(cx, ref):
    """Resolve an opaque ref to an ACTIVE+cert_ok coach email, or None."""
    from dashboard import coach_directory as _cd
    for c in _cd.list_active_full(cx):
        if coach_ref(c["email"]) == ref:
            return c["email"]
    return None


def member_has_accepted(cx, member_email):
    """True if the member is already matched (has an accepted coach)."""
    return cx.execute("SELECT 1 FROM coach_requests WHERE member_email=? AND status='accepted' "
                      "LIMIT 1", (_lc(member_email),)).fetchone() is not None


def member_applications(cx, member_email):
    """The member's pending + accepted applications: [{coach_email, status}]."""
    rows = cx.execute("SELECT coach_email, status FROM coach_requests WHERE member_email=? "
                      "AND status IN ('pending','accepted') ORDER BY id", (_lc(member_email),)).fetchall()
    return [{"coach_email": r["coach_email"], "status": r["status"]} for r in rows]


def request_member(cx, request_id):
    """The member_email that owns a request id, or None."""
    row = cx.execute("SELECT member_email FROM coach_requests WHERE id=?", (request_id,)).fetchone()
    return row["member_email"] if row else None


def withdraw_other_pendings(cx, member_email, keep_request_id):
    """When a member is claimed (a coach accepts), withdraw their OTHER pending
    applications so first-accept-wins."""
    cx.execute("UPDATE coach_requests SET status='withdrawn', decided_at=? "
               "WHERE member_email=? AND status='pending' AND id!=?",
               (_now(), _lc(member_email), keep_request_id))
    cx.commit()


def create_request(cx, coach_email, member_email, member_name, note, member_video_url=""):
    """Create a pending application. A member may hold MULTIPLE pending
    applications, but not once already matched: returns None if the member
    already has an accepted coach, or if they already applied to THIS coach."""
    member_email = _lc(member_email)
    if member_has_accepted(cx, member_email):
        return None
    if cx.execute("SELECT 1 FROM coach_requests WHERE coach_email=? AND member_email=? "
                  "AND status IN ('pending','accepted') LIMIT 1",
                  (_lc(coach_email), member_email)).fetchone():
        return None  # already applied to this coach
    cur = cx.execute(
        "INSERT INTO coach_requests (coach_email,member_email,member_name,note,"
        "member_video_url,status,created_at) VALUES (?,?,?,?,?, 'pending', ?) "
        "ON CONFLICT(coach_email,member_email) DO UPDATE SET status='pending', "
        "member_name=excluded.member_name, note=excluded.note, "
        "member_video_url=excluded.member_video_url, created_at=excluded.created_at",
        (_lc(coach_email), member_email, member_name, (note or "")[:500],
         member_video_url or "", _now()))
    cx.commit()
    return cur.lastrowid


def requests_for_coach(cx, coach_email, status="pending"):
    rows = cx.execute("SELECT id, member_name, note, member_video_url, status "
                      "FROM coach_requests WHERE coach_email=? AND status=? ORDER BY created_at",
                      (_lc(coach_email), status)).fetchall()
    return [{"id": r["id"], "member_name": r["member_name"], "note": r["note"],
             "member_video_url": r["member_video_url"], "status": r["status"]} for r in rows]


def accepted_count(cx, coach_email):
    return cx.execute("SELECT COUNT(*) FROM coach_requests WHERE coach_email=? "
                      "AND status='accepted'", (_lc(coach_email),)).fetchone()[0]


def set_request_status(cx, request_id, status):
    cx.execute("UPDATE coach_requests SET status=?, decided_at=? WHERE id=?",
               (status, _now(), request_id))
    cx.commit()


def request_owner(cx, request_id):
    """The coach_email that owns a request id (for authorization), or None."""
    row = cx.execute("SELECT coach_email FROM coach_requests WHERE id=?",
                     (request_id,)).fetchone()
    return row["coach_email"] if row else None


def join_waitlist(cx, member_email):
    cx.execute("INSERT OR IGNORE INTO coach_waitlist (member_email,created_at) VALUES (?,?)",
               (_lc(member_email), _now()))
    cx.commit()


def on_waitlist(cx, member_email):
    return cx.execute("SELECT 1 FROM coach_waitlist WHERE member_email=?",
                      (_lc(member_email),)).fetchone() is not None


def record_interest(cx, member_email, tier):
    cx.execute("INSERT OR IGNORE INTO coaching_interest (member_email,tier,created_at) "
               "VALUES (?,?,?)", (_lc(member_email), tier, _now()))
    cx.commit()
```

Note: the test file also needs `requests_for_coach` to exclude accepted rows after acceptance — the `status="pending"` filter handles that. `request_owner` is added for Task 3's authorization.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_coach_connect.py -q`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add dashboard/coach_connect.py dashboard/coach_directory.py tests/test_coach_connect.py
git commit -m "feat(coaching): connect store (applications, waitlist, interest, coach ref)"
```

---

### Task 2: Member routes (`app.py`)

**Files:**
- Modify: `app.py` (extend `community_coaches`; add apply / waitlist / interest routes)
- Test: `tests/test_coach_request_api.py`

**Interfaces:**
- Consumes: `dashboard/coach_directory.py:list_active_full`, `dashboard/coach_connect.py` (all), `dashboard/coaching.py:active_window`, `_evox_ident`, `send_evox_email`, `GLEN_CONSULT_EMAIL`, `_db_lock`, `LOG_DB`.
- Produces: extended `GET /api/community/coaches`; `POST /api/community/coach-request`; `POST /api/community/coach-waitlist`; `POST /api/community/coaching-interest`.

**Contract:**
- `GET /api/community/coaches` → bad token 404; no active window → `{eligible:false, coaches:[]}`; else build the visible list from `list_active_full` dropping coaches with `accepted_count >= capacity`, mapping each to `{ref, name, focus, intro_video_url}` (NO email). Response `{eligible:true, coaches:[...], applications:[{ref,status}], matched:bool, all_full:bool}` where `all_full = (there are cert-ok volunteers) and (visible list is empty)`. `applications` = `member_applications` mapped to refs; `matched` = `member_has_accepted`.
- `POST /api/community/coach-request {coach_ref, note}` → 404 bad token; 403 if no active window; resolve `coach_ref` (404 `coach_unavailable` if it does not resolve to an active coach or that coach is full); `create_request` → None means already-active → 409 `already_requested`; else `{ok, status:"pending"}`.
- `POST /api/community/coach-waitlist` → member active window required; `join_waitlist`; `{ok}`.
- `POST /api/community/coaching-interest {tier}` → `tier in {"rae","glen"}` else 400; `record_interest`; best-effort notify Glen via `send_evox_email(GLEN_CONSULT_EMAIL, "Glen", subject, html, html, b"")`; `{ok}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_coach_request_api.py
import sqlite3
from unittest import mock
import app as appmod
from dashboard import coach_directory as _cd, coach_connect as _cc


def _client():
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()


def _seed_member(email, *, window=True, capacity=1):
    from datetime import datetime, timezone, timedelta
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        from dashboard import evox as _ev, client_portal as _cp, coaching as _co
        _ev.init_evox_tables(cx); _cp.init_client_portal_table(cx)
        _cd.init_coach_tables(cx); _cc.init_connect_tables(cx); _co.init_coaching_table(cx)
        _cd.upsert_volunteer(cx, email="coach1@x.com", name="Cora", focus="sleep",
                             intro_video_url="/portal-asset/c.mp4", capacity=capacity, cert_ok=1)
        if window:
            now = datetime.now(timezone.utc)
            ends = (now + timedelta(days=10)).isoformat()
            cx.execute("INSERT INTO coaching_windows (email,order_id,started_at,ends_at,"
                       "source,created_at) VALUES (?,?,?,?,?,?)",
                       (email, 1, now.isoformat(), ends, "test", now.isoformat()))
        token = _ev.ensure_portal_token(cx, email, "Mel")
        cx.commit()
    return token


def _coach_ref():
    return _cc.coach_ref("coach1@x.com")


def _add_coach(email, capacity=1):
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _cd.init_coach_tables(cx)
        _cd.upsert_volunteer(cx, email=email, name="C2", focus="adrenals",
                             intro_video_url="/portal-asset/c2.mp4", capacity=capacity, cert_ok=1)
        cx.commit()
    return _cc.coach_ref(email)


def test_coaches_lists_ref_no_email():
    c = _client(); tok = _seed_member("m@x.com")
    d = c.get(f"/api/community/coaches?token={tok}").get_json()
    assert d["eligible"] and d["coaches"][0]["ref"] == _coach_ref()
    assert "email" not in d["coaches"][0]
    assert d["all_full"] is False and d["applications"] == [] and d["matched"] is False


def test_apply_then_status_shows_in_applications():
    c = _client(); tok = _seed_member("m@x.com")
    r = c.post(f"/api/community/coach-request?token={tok}",
               json={"coach_ref": _coach_ref(), "note": "sleep trouble"})
    assert r.get_json()["status"] == "pending"
    d = c.get(f"/api/community/coaches?token={tok}").get_json()
    assert {"ref": _coach_ref(), "status": "pending"} in d["applications"] and d["matched"] is False


def test_apply_multiple_coaches_allowed_same_coach_409():
    c = _client(); tok = _seed_member("m@x.com")
    ref2 = _add_coach("coach2@x.com")
    a1 = c.post(f"/api/community/coach-request?token={tok}", json={"coach_ref": _coach_ref(), "note": "x"})
    a2 = c.post(f"/api/community/coach-request?token={tok}", json={"coach_ref": ref2, "note": "y"})
    assert a1.get_json()["status"] == "pending" and a2.get_json()["status"] == "pending"  # both OK
    dup = c.post(f"/api/community/coach-request?token={tok}", json={"coach_ref": _coach_ref(), "note": "z"})
    assert dup.status_code == 409                              # same coach again


def test_apply_with_video_is_stored():
    import io
    c = _client(); tok = _seed_member("m@x.com")
    data = {"coach_ref": _coach_ref(), "note": "n",
            "video": (io.BytesIO(b"\x00\x01vid"), "me.mp4")}
    r = c.post(f"/api/community/coach-request?token={tok}", data=data,
               content_type="multipart/form-data")
    assert r.get_json()["status"] == "pending"
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row; _cc.init_connect_tables(cx)
        pend = _cc.requests_for_coach(cx, "coach1@x.com")
        assert pend[0]["member_video_url"].startswith("/portal-asset/member-")


def test_full_coach_dropped_and_all_full():
    c = _client(); tok = _seed_member("m@x.com", capacity=1)
    # fill the only coach's single slot with an accepted request from another member
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _cc.init_connect_tables(cx)
        rid = _cc.create_request(cx, "coach1@x.com", "other@x.com", "Oth", "n")
        _cc.set_request_status(cx, rid, "accepted"); cx.commit()
    d = c.get(f"/api/community/coaches?token={tok}").get_json()
    assert d["coaches"] == [] and d["all_full"] is True


def test_waitlist_and_interest():
    c = _client(); tok = _seed_member("m@x.com")
    assert c.post(f"/api/community/coach-waitlist?token={tok}").get_json()["ok"] is True
    with mock.patch.object(appmod, "send_evox_email") as sent:
        r = c.post(f"/api/community/coaching-interest?token={tok}", json={"tier": "glen"})
    assert r.get_json()["ok"] is True and sent.called
    bad = c.post(f"/api/community/coaching-interest?token={tok}", json={"tier": "nope"})
    assert bad.status_code == 400
```

- [ ] **Step 2: Run test to verify it fails**

Run: `export DATA_DIR="$(mktemp -d)" && doppler run -p remedy-match -c prd -- env DATA_DIR="$DATA_DIR" python3 -m pytest tests/test_coach_request_api.py -q`
Expected: FAIL — routes 404 / missing `ref`+`all_full`.

- [ ] **Step 3: Write minimal implementation**

Replace the `community_coaches` route body and add the three POST routes:

```python
@app.route("/api/community/coaches")
def community_coaches():
    from dashboard import coach_directory as _cd, coaching as _co, coach_connect as _cc
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _cd.init_coach_tables(cx); _co.init_coaching_table(cx); _cc.init_connect_tables(cx)
        ident = _evox_ident(cx, request.args.get("token", ""))
        if ident is None:
            return jsonify({"error": "not_found"}), 404
        if not _co.active_window(cx, ident.email):
            return jsonify({"eligible": False, "coaches": []})
        candidates = _cd.list_active_full(cx)
        coaches = []
        for cand in candidates:
            if _cc.accepted_count(cx, cand["email"]) >= (cand["capacity"] or 0):
                continue
            coaches.append({"ref": _cc.coach_ref(cand["email"]), "name": cand["name"],
                            "focus": cand["focus"], "intro_video_url": cand["intro_video_url"]})
        apps = _cc.member_applications(cx, ident.email)
        applications = [{"ref": _cc.coach_ref(a["coach_email"]), "status": a["status"]}
                        for a in apps]
        return jsonify({"eligible": True, "coaches": coaches, "applications": applications,
                        "matched": _cc.member_has_accepted(cx, ident.email),
                        "all_full": bool(candidates) and not coaches})


@app.route("/api/community/coach-request", methods=["POST"])
def community_coach_request():
    from dashboard import coaching as _co, coach_connect as _cc
    # multipart: coach_ref + note + optional `video` file (member intro). Falls back
    # to JSON when no file is sent.
    ref = (request.form.get("coach_ref") or (request.get_json(silent=True) or {}).get("coach_ref") or "").strip()
    note = (request.form.get("note") or (request.get_json(silent=True) or {}).get("note") or "").strip()
    member_video_url = ""
    vf = request.files.get("video")
    if vf is not None and vf.filename:
        fname = f"member-{secrets.token_hex(8)}.mp4"
        (_PORTAL_ASSETS_DIR / fname).write_bytes(vf.read())
        member_video_url = f"/portal-asset/{fname}"
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _cc.init_connect_tables(cx)
        ident = _evox_ident(cx, request.args.get("token", ""))
        if ident is None:
            return jsonify({"error": "not_found"}), 404
        if not _co.active_window(cx, ident.email):
            return jsonify({"error": "not_eligible"}), 403
        coach_email = _cc.email_for_ref(cx, ref)
        if not coach_email:
            return jsonify({"error": "coach_unavailable"}), 404
        first_name = ((getattr(ident, "name", "") or "").split(" ")[0])
        rid = _cc.create_request(cx, coach_email, ident.email, first_name, note,
                                 member_video_url=member_video_url)
        if rid is None:
            # already matched, or already applied to this coach
            return jsonify({"error": "cannot_apply"}), 409
        return jsonify({"ok": True, "status": "pending"})


@app.route("/api/community/coach-waitlist", methods=["POST"])
def community_coach_waitlist():
    from dashboard import coaching as _co, coach_connect as _cc
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _cc.init_connect_tables(cx)
        ident = _evox_ident(cx, request.args.get("token", ""))
        if ident is None:
            return jsonify({"error": "not_found"}), 404
        if not _co.active_window(cx, ident.email):
            return jsonify({"error": "not_eligible"}), 403
        _cc.join_waitlist(cx, ident.email)
        return jsonify({"ok": True})


@app.route("/api/community/coaching-interest", methods=["POST"])
def community_coaching_interest():
    from dashboard import coach_connect as _cc
    body = request.get_json(force=True) or {}
    tier = (body.get("tier") or "").strip()
    if tier not in ("rae", "glen"):
        return jsonify({"error": "bad_tier"}), 400
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _cc.init_connect_tables(cx)
        ident = _evox_ident(cx, request.args.get("token", ""))
        if ident is None:
            return jsonify({"error": "not_found"}), 404
        _cc.record_interest(cx, ident.email, tier)
        email = ident.email
    who = "Dr. Glen ($200/mo, Causal Biofield Analysis)" if tier == "glen" \
        else "Rae ($100/mo, EVOX session)"
    try:
        html = f"<p>{email} is interested in paid coaching with {who}.</p>"
        send_evox_email(GLEN_CONSULT_EMAIL, "Glen", f"Coaching interest: {tier}", html, html, b"")
    except Exception:
        app.logger.exception("coaching interest notify failed")
    return jsonify({"ok": True})
```

Note: if `_evox_ident`'s identity object has no `.name`, the member first name falls back to `""`; verify the identity shape (`grep -n "class .*Ident\|_evox_ident" app.py`) and use whatever first-name field it exposes (or pull the portal record name). The note in this task is that the member first name must be a FIRST name only (split on space), never the email.

- [ ] **Step 4: Run test to verify it passes**

Run: `export DATA_DIR="$(mktemp -d)" && doppler run -p remedy-match -c prd -- env DATA_DIR="$DATA_DIR" python3 -m pytest tests/test_coach_request_api.py -q`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_coach_request_api.py
git commit -m "feat(coaching): member apply + waitlist + interest routes"
```

---

### Task 3: Coach routes (`app.py`)

**Files:**
- Modify: `app.py` (add coach review + respond routes)
- Test: `tests/test_coach_respond_api.py`

**Interfaces:**
- Consumes: `dashboard/coach_connect.py` (`requests_for_coach`, `accepted_count`, `set_request_status`, `request_owner`), `dashboard/coach_directory.py:get_volunteer` (for capacity), `_practitioner_session_pid`, `practitioner_email_by_id`, `_db_lock`, `LOG_DB`.
- Produces: `GET /api/practitioner/coach-requests`; `POST /api/practitioner/coach-request/respond`.

**Contract:**
- `GET /api/practitioner/coach-requests?token=…` → practitioner session (401 no pid/email); `{pending: requests_for_coach(email), coachees: [{member_name} from accepted], capacity, slots_left}` where `slots_left = capacity - accepted_count` (capacity from `get_volunteer`; 0 if no profile). Never exposes member email.
- `POST /api/practitioner/coach-request/respond {request_id, accept}` → practitioner session (401); verify `request_owner(request_id) == this coach email` (else 404); on `accept`: 409 `at_capacity` if `accepted_count >= capacity`, else `set_request_status("accepted")`; on decline: `set_request_status("declined")`. Returns `{ok, status}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_coach_respond_api.py
import sqlite3
from unittest import mock
import app as appmod
from dashboard import coach_directory as _cd, coach_connect as _cc


def _client():
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()


def _seed(coach_email="coach1@x.com", *, capacity=1):
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _cd.init_coach_tables(cx); _cc.init_connect_tables(cx)
        _cd.upsert_volunteer(cx, email=coach_email, name="Cora", focus="sleep",
                             intro_video_url="u", capacity=capacity, cert_ok=1)
        rid = _cc.create_request(cx, coach_email, "mem@x.com", "Mel", "sleep help")
        cx.commit()
    return rid


def test_requests_list_first_name_note_no_email():
    c = _client(); rid = _seed()
    with mock.patch.object(appmod, "_practitioner_session_pid", return_value="pid1"), \
         mock.patch("dashboard.practitioner_portal.practitioner_email_by_id", return_value="coach1@x.com"):
        d = c.get("/api/practitioner/coach-requests?token=t").get_json()
    assert d["pending"][0]["member_name"] == "Mel" and d["pending"][0]["note"] == "sleep help"
    assert "member_email" not in d["pending"][0] and "email" not in repr(d["pending"][0])
    assert d["capacity"] == 1 and d["slots_left"] == 1


def test_respond_accept_then_at_capacity():
    c = _client(); rid = _seed(capacity=1)
    with mock.patch.object(appmod, "_practitioner_session_pid", return_value="pid1"), \
         mock.patch("dashboard.practitioner_portal.practitioner_email_by_id", return_value="coach1@x.com"):
        r1 = c.post("/api/practitioner/coach-request/respond?token=t",
                    json={"request_id": rid, "accept": True})
        assert r1.get_json()["status"] == "accepted"
        # a second pending request now cannot be accepted (capacity 1 filled)
        with sqlite3.connect(appmod.LOG_DB) as cx:
            cx.row_factory = sqlite3.Row; _cc.init_connect_tables(cx)
            rid2 = _cc.create_request(cx, "coach1@x.com", "mem2@x.com", "Mo", "n"); cx.commit()
        r2 = c.post("/api/practitioner/coach-request/respond?token=t",
                    json={"request_id": rid2, "accept": True})
        assert r2.status_code == 409


def test_first_accept_wins_second_coach_member_taken():
    c = _client()
    # member applied to TWO coaches; coach1 accepts first, coach2 then gets member_taken
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _cd.init_coach_tables(cx); _cc.init_connect_tables(cx)
        for e in ("coach1@x.com", "coach2@x.com"):
            _cd.upsert_volunteer(cx, email=e, name="C", focus="f", intro_video_url="u",
                                 capacity=1, cert_ok=1)
        rid1 = _cc.create_request(cx, "coach1@x.com", "mm@x.com", "Mel", "n")
        rid2 = _cc.create_request(cx, "coach2@x.com", "mm@x.com", "Mel", "n")
        cx.commit()
    with mock.patch.object(appmod, "_practitioner_session_pid", return_value="p1"), \
         mock.patch("dashboard.practitioner_portal.practitioner_email_by_id", return_value="coach1@x.com"):
        assert c.post("/api/practitioner/coach-request/respond?token=t",
                      json={"request_id": rid1, "accept": True}).get_json()["status"] == "accepted"
    with mock.patch.object(appmod, "_practitioner_session_pid", return_value="p2"), \
         mock.patch("dashboard.practitioner_portal.practitioner_email_by_id", return_value="coach2@x.com"):
        r2 = c.post("/api/practitioner/coach-request/respond?token=t",
                    json={"request_id": rid2, "accept": True})
    assert r2.status_code == 409 and r2.get_json()["error"] == "member_taken"


def test_respond_non_owner_404():
    c = _client(); rid = _seed()
    with mock.patch.object(appmod, "_practitioner_session_pid", return_value="pid9"), \
         mock.patch("dashboard.practitioner_portal.practitioner_email_by_id", return_value="other@x.com"):
        r = c.post("/api/practitioner/coach-request/respond?token=t",
                   json={"request_id": rid, "accept": True})
    assert r.status_code == 404


def test_requests_requires_session():
    with mock.patch.object(appmod, "_practitioner_session_pid", return_value=None):
        assert _client().get("/api/practitioner/coach-requests?token=x").status_code == 401
```

- [ ] **Step 2: Run test to verify it fails**

Run: `export DATA_DIR="$(mktemp -d)" && doppler run -p remedy-match -c prd -- env DATA_DIR="$DATA_DIR" python3 -m pytest tests/test_coach_respond_api.py -q`
Expected: FAIL — routes 404.

- [ ] **Step 3: Write minimal implementation**

Add to `app.py`:

```python
def _coach_session_email():
    pid = _practitioner_session_pid()
    if not pid:
        return None
    from dashboard.practitioner_portal import practitioner_email_by_id
    return (practitioner_email_by_id(pid) or "").strip().lower() or None


@app.route("/api/practitioner/coach-requests")
def practitioner_coach_requests():
    email = _coach_session_email()
    if not email:
        return jsonify({"error": "unauthorized"}), 401
    from dashboard import coach_connect as _cc, coach_directory as _cd
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _cc.init_connect_tables(cx); _cd.init_coach_tables(cx)
        vol = _cd.get_volunteer(cx, email)
        capacity = (vol or {}).get("capacity", 0) or 0
        accepted = _cc.accepted_count(cx, email)
        coachees = [{"member_name": r["member_name"]}
                    for r in _cc.requests_for_coach(cx, email, status="accepted")]
        return jsonify({"pending": _cc.requests_for_coach(cx, email, status="pending"),
                        "coachees": coachees, "capacity": capacity,
                        "slots_left": max(0, capacity - accepted)})


@app.route("/api/practitioner/coach-request/respond", methods=["POST"])
def practitioner_coach_request_respond():
    email = _coach_session_email()
    if not email:
        return jsonify({"error": "unauthorized"}), 401
    from dashboard import coach_connect as _cc, coach_directory as _cd
    body = request.get_json(force=True) or {}
    rid = body.get("request_id")
    accept = bool(body.get("accept"))
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _cc.init_connect_tables(cx); _cd.init_coach_tables(cx)
        if _cc.request_owner(cx, rid) != email:
            return jsonify({"error": "not_found"}), 404
        if accept:
            vol = _cd.get_volunteer(cx, email)
            capacity = (vol or {}).get("capacity", 0) or 0
            if _cc.accepted_count(cx, email) >= capacity:
                return jsonify({"error": "at_capacity"}), 409
            member_email = _cc.request_member(cx, rid)
            if _cc.member_has_accepted(cx, member_email):
                return jsonify({"error": "member_taken"}), 409   # another coach won the race
            _cc.set_request_status(cx, rid, "accepted")
            _cc.withdraw_other_pendings(cx, member_email, rid)   # first-accept-wins
            return jsonify({"ok": True, "status": "accepted"})
        _cc.set_request_status(cx, rid, "declined")
        return jsonify({"ok": True, "status": "declined"})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `export DATA_DIR="$(mktemp -d)" && doppler run -p remedy-match -c prd -- env DATA_DIR="$DATA_DIR" python3 -m pytest tests/test_coach_respond_api.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_coach_respond_api.py
git commit -m "feat(coaching): coach review + respond routes (capacity-gated)"
```

---

### Task 4: Member card — apply / status / waitlist / upsell (`static/client-portal.html`)

**Files:**
- Modify: `static/client-portal.html` (the slice-1 "Meet your coaches" card)
- Test: manual JS parse check.

**Interfaces:**
- Consumes: extended `GET /api/community/coaches` (`{coaches:[{ref,name,focus,intro_video_url}], my_request, all_full}`), `POST /api/community/coach-request {coach_ref, note}`, `POST /api/community/coach-waitlist`, `POST /api/community/coaching-interest {tier}`.

**Design note:** extend the existing "Meet your coaches" card (from slice 1). The directory items now carry `ref`, and the payload has `applications:[{ref,status}]`, `matched:bool`, `all_full:bool`. Wrap new JS in `<!-- BEGIN coach-apply script -->` / `<!-- END coach-apply script -->`:
- Per coach card: an "Apply to this coach" control that reveals a small form — a textarea ("What are you working on?") and an OPTIONAL video **file input** (`accept="video/mp4,video/*"`, a short intro "what you are looking for help with"). Submit builds a `FormData` (`coach_ref`, `note`, and `video` file only if chosen — omit otherwise) and `POST /api/community/coach-request?token=...` (no Content-Type header). Members MAY apply to **multiple** coaches: only disable a coach's apply button if `matched` is true or that coach's `ref` is already in `applications`. Show each applied coach's status from `applications` ("Applied, waiting for a reply" / "Matched"); when `matched`, show "You are matched with a coach" and hide all apply controls.
- If `d.all_full`: below the (empty) list, show a "Join the waitlist" button (`POST /api/community/coach-waitlist`) AND the upsell card: heading and copy "Our student coaches are full right now. Join the waitlist, or start coaching now with Rae ($100 a month, includes an EVOX session) or Dr. Glen ($200 a month, includes a Causal Biofield Analysis)." with two "I'm interested" buttons → `POST /api/community/coaching-interest` with `{tier:"rae"}` / `{tier:"glen"}`; on click show "Thanks, we will be in touch."
- All server strings via `textContent`; the note textarea value is user input sent in the POST body (not rendered back via innerHTML). Copy: no em dashes, no ALL CAPS.

- [ ] **Step 1: Extend the card + add apply/waitlist/upsell**

Read the slice-1 "Meet your coaches" card. Add the apply control (textarea + submit), the `my_request` status/disable, and the `all_full` waitlist + upsell block per the design note.

- [ ] **Step 2: Verify the page JS parses**

Run: `cd /tmp/wt-deploy-chat-cca589e9 && node --check <(python3 -c "import re; h=open('static/client-portal.html').read(); print('\n;\n'.join(re.findall(r'<script>(.*?)</script>', h, re.S)))")`
Expected: no output (clean parse).

- [ ] **Step 3: Commit**

```bash
git add static/client-portal.html
git commit -m "feat(coaching): member apply + waitlist + upsell offer on the coaches card"
```

---

### Task 5: Practitioner applications list (`static/practitioner-portal.html`)

**Files:**
- Modify: `static/practitioner-portal.html` (the slice-1 "Coaching volunteer" control)
- Test: manual JS parse check.

**Interfaces:**
- Consumes: `GET /api/practitioner/coach-requests` (`{pending:[{request_id, member_name, note}], coachees, capacity, slots_left}`), `POST /api/practitioner/coach-request/respond {request_id, accept}`.

**Design note:** extend the slice-1 "Coaching volunteer" control. Wrap new JS in `<!-- BEGIN coach-requests script -->` / `<!-- END coach-requests script -->`:
- On load (and after any respond), `GET /api/practitioner/coach-requests?token='+TOKEN` (the practitioner session token the page already uses). Render a "slots left: N of M" line, then each pending application as a row: member first name + their note (both via `textContent`), and if `member_video_url` is set, the member's intro video as a `<video controls>` with that src. Accept and Not now buttons. On Accept → `POST /api/practitioner/coach-request/respond` `{request_id, accept:true}`; on Not now → `{request_id, accept:false}`; on a 409, show "You are at capacity." (`at_capacity`) or "That member has already matched with another coach." (`member_taken`). Refresh the list after each action. List accepted coachees (first names) below.
- Copy: no em dashes, no ALL CAPS. All member data via `textContent`.

- [ ] **Step 1: Extend the control + add the applications list**

Read the slice-1 "Coaching volunteer" control; add the requests fetch + render + accept/decline per the design note.

- [ ] **Step 2: Verify the page JS parses**

Run: `cd /tmp/wt-deploy-chat-cca589e9 && node --check <(python3 -c "import re; h=open('static/practitioner-portal.html').read(); print('\n;\n'.join(re.findall(r'<script>(.*?)</script>', h, re.S)))")`
Expected: no output (clean parse).

- [ ] **Step 3: Commit**

```bash
git add static/practitioner-portal.html
git commit -m "feat(coaching): coach applications list + accept/decline in the portal control"
```

---

## Definition of Done

- A member with an active coaching window applies to a free student coach (by opaque ref, with a note), capped at one active coach; the coach reviews the application (first name + note, never email) and accepts up to capacity; full coaches drop from the directory; when all are full the member sees a waitlist + the paid upsell offer (Rae/Glen) with an "I'm interested" flag that notifies Glen. No charge anywhere.
- All new tests pass; slice-1 directory/signup untouched except the additive `list_active_full` + the ref/full-exclusion in the directory route; Community A/B/C untouched.

## Deferred (not in this plan)

- Slice 2b: the recurring paid-coaching subscription (Rae $100/mo + EVOX, Glen $200/mo + Causal Biofield) reusing membership + EVOX + Biofield.
- Slice 3: the 1:1 coaching thread on an accepted pairing + report/block + moderation.
- Waitlist auto-assignment/notification when a slot frees.
