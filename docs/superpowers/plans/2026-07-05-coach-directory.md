# Coach Volunteer Directory (coaching arc, slice 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A certification student volunteers to coach from their practitioner portal (sets capacity 0-12, uploads an intro video, gated on cert completion), and members with an active coaching window browse a coach directory.

**Architecture:** A sqlite `coach_volunteers` store (denormalized name/focus/video so the member directory never queries Postgres); a practitioner-session-authed self-service signup route that stores the uploaded intro video via the existing asset mechanism and runs a fail-closed cert-completion check; a member-portal-gated directory route gated on an active coaching window; a member directory card + a practitioner-portal control. Cert check + coaching eligibility are both sqlite; the coach roster is the volunteer table (not Postgres).

**Tech Stack:** Python 3 / Flask (single `app.py` + `dashboard/*.py`), sqlite (`chat_log.db`, `?` placeholders, `_db_lock`, `cx.row_factory = sqlite3.Row`), Rumble unlisted for intro video, vanilla JS/HTML.

## Global Constraints

- **Privacy:** the member directory exposes only `{name, focus, intro_video_url}` — NEVER a coach's email. A coach who volunteers consents to being shown; a member never gets coach contact in this slice.
- **Cert eligibility is fail-closed:** a volunteer is only listed if `cert_ok=1`, set from `cert_rules.evaluate(approved submissions).complete`; any lookup error → `cert_ok=0` (not listed).
- **Member gate:** the directory is only served to members with an active coaching window (`coaching.active_window`); others get `{eligible:false, coaches:[]}`.
- **Signup is practitioner-portal self-service:** a cert student, authed by their practitioner session token (`_practitioner_session_pid()` → practitioner_id → email via `practitioner_email_by_id`), sets their capacity (0-12) and uploads a short intro video from their own portal. Capacity 0 = not taking members (drops them from the directory). In-browser recording is a fast-follow; this slice is file **upload**.
- **Intro video self-hosts** via the existing `_PORTAL_ASSETS_DIR` (written as `coach-<hex>.mp4`, served by `/portal-asset/<filename>` which already accepts `.mp4`); `intro_video_url` = `/portal-asset/<filename>`. No Rumble for these short intros.
- **Video hosting = Rumble unlisted** (the `intro_video_url` is a Rumble link embedded in the card).
- **Copy:** no em dashes, no ALL CAPS.
- sqlite writes under `with _db_lock, sqlite3.connect(LOG_DB)`; emails lowercased.
- DRY, YAGNI, TDD, frequent commits.

**Repo facts the implementer needs:**
- `dashboard/cert_submissions.py:list_for_email(cx, email) -> [dict]` (each submission dict has `status` and `credited_modules`/`formats`); approved submissions have `status == "approved"`.
- `dashboard/cert_rules.py:evaluate(submissions) -> dict` with a `complete` bool — pass ONLY approved submissions.
- `dashboard/coaching.py:active_window(cx, email) -> dict | None` (a member's active coaching window, or None).
- `app.py`: `_evox_ident(cx, token)` → identity with `.email` or None; `CONSOLE_SECRET`; the console-gate idiom `if request.headers.get("X-Console-Key") != CONSOLE_SECRET: return jsonify({"error":"unauthorized"}), 401`; `LOG_DB`; `_db_lock`; `send_from_directory`; `STATIC`.
- `static/client-portal.html`: the portal page; the `token` var (member portal token) and the card-building idiom used by existing cards.

**Testing note (READ FIRST):**
- Pure store test (Task 1) does NOT import app — plain `python3 -m pytest tests/test_coach_directory_store.py -q`.
- Route tests (Tasks 2-3) `import app`; override DATA_DIR (prd Doppler points it at a nonexistent prod path):
  ```
  export DATA_DIR="$(mktemp -d)" && doppler run -p remedy-match -c prd -- env DATA_DIR="$DATA_DIR" python3 -m pytest <paths> -q
  ```

---

### Task 1: Volunteer store (`dashboard/coach_directory.py`)

**Files:**
- Create: `dashboard/coach_directory.py`
- Test: `tests/test_coach_directory_store.py`

**Interfaces:**
- Consumes: nothing (pure sqlite).
- Produces: `init_coach_tables(cx)`; `upsert_volunteer(cx, *, email, name, focus, intro_video_url, capacity, cert_ok) -> None`; `set_active(cx, email, active)`; `get_volunteer(cx, email) -> dict|None`; `list_active(cx) -> [dict]` (only `active=1 AND cert_ok=1`, newest first, each `{name, focus, intro_video_url}` — NO email/capacity).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_coach_directory_store.py
import sqlite3
from dashboard import coach_directory as _cd


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    _cd.init_coach_tables(cx)
    return cx


def test_upsert_and_get():
    cx = _cx()
    _cd.upsert_volunteer(cx, email="C@X.com", name="Cora", focus="sleep",
                         intro_video_url="https://rumble.com/v-c", capacity=3, cert_ok=1)
    row = _cd.get_volunteer(cx, "c@x.com")
    assert row["name"] == "Cora" and row["cert_ok"] == 1 and row["active"] == 1


def test_upsert_is_idempotent_on_email():
    cx = _cx()
    _cd.upsert_volunteer(cx, email="c@x.com", name="Cora", focus="a",
                         intro_video_url="u1", capacity=3, cert_ok=1)
    _cd.upsert_volunteer(cx, email="c@x.com", name="Cora B", focus="sleep",
                         intro_video_url="u2", capacity=5, cert_ok=1)
    assert cx.execute("SELECT COUNT(*) FROM coach_volunteers").fetchone()[0] == 1
    row = _cd.get_volunteer(cx, "c@x.com")
    assert row["focus"] == "sleep" and row["intro_video_url"] == "u2" and row["capacity"] == 5


def test_list_active_member_safe_no_email():
    cx = _cx()
    _cd.upsert_volunteer(cx, email="c@x.com", name="Cora", focus="sleep",
                         intro_video_url="u", capacity=3, cert_ok=1)
    lst = _cd.list_active(cx)
    assert lst == [{"name": "Cora", "focus": "sleep", "intro_video_url": "u"}]
    assert "email" not in lst[0] and "capacity" not in lst[0]


def test_list_active_excludes_inactive_and_uncertified():
    cx = _cx()
    _cd.upsert_volunteer(cx, email="ok@x.com", name="Ok", focus="f", intro_video_url="u",
                         capacity=3, cert_ok=1)
    _cd.upsert_volunteer(cx, email="nocert@x.com", name="NoCert", focus="f",
                         intro_video_url="u", capacity=3, cert_ok=0)   # not certified
    _cd.upsert_volunteer(cx, email="off@x.com", name="Off", focus="f", intro_video_url="u",
                         capacity=3, cert_ok=1)
    _cd.set_active(cx, "off@x.com", 0)                                  # deactivated
    names = [c["name"] for c in _cd.list_active(cx)]
    assert names == ["Ok"]


def test_set_active_toggles():
    cx = _cx()
    _cd.upsert_volunteer(cx, email="c@x.com", name="Cora", focus="f", intro_video_url="u",
                         capacity=3, cert_ok=1)
    _cd.set_active(cx, "c@x.com", 0)
    assert _cd.get_volunteer(cx, "c@x.com")["active"] == 0
    assert _cd.list_active(cx) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_coach_directory_store.py -q`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/coach_directory.py
"""Coach volunteer directory (coaching arc, slice 1). Pure sqlite; no app-layer
imports. Members see only {name, focus, intro_video_url} — a coach's email is
never exposed. Only active AND cert_ok volunteers are listed."""

_DDL = """
CREATE TABLE IF NOT EXISTS coach_volunteers (
    email TEXT PRIMARY KEY,
    name TEXT,
    focus TEXT,
    intro_video_url TEXT,
    capacity INTEGER DEFAULT 3,
    active INTEGER DEFAULT 1,
    cert_ok INTEGER DEFAULT 0,
    created_at TEXT,
    updated_at TEXT
);
"""


def _now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _lc(email):
    return (email or "").strip().lower()


def init_coach_tables(cx):
    cx.executescript(_DDL)
    cx.commit()


def upsert_volunteer(cx, *, email, name, focus, intro_video_url, capacity, cert_ok):
    email = _lc(email)
    now = _now()
    cx.execute(
        "INSERT INTO coach_volunteers (email,name,focus,intro_video_url,capacity,"
        "active,cert_ok,created_at,updated_at) VALUES (?,?,?,?,?,1,?,?,?) "
        "ON CONFLICT(email) DO UPDATE SET name=excluded.name, focus=excluded.focus, "
        "intro_video_url=excluded.intro_video_url, capacity=excluded.capacity, "
        "cert_ok=excluded.cert_ok, updated_at=excluded.updated_at",
        (email, name, focus, intro_video_url, int(capacity), int(cert_ok), now, now))
    cx.commit()


def set_active(cx, email, active):
    cx.execute("UPDATE coach_volunteers SET active=?, updated_at=? WHERE email=?",
               (1 if active else 0, _now(), _lc(email)))
    cx.commit()


def get_volunteer(cx, email):
    row = cx.execute("SELECT * FROM coach_volunteers WHERE email=?", (_lc(email),)).fetchone()
    return dict(row) if row else None


def list_active(cx):
    rows = cx.execute("SELECT name, focus, intro_video_url FROM coach_volunteers "
                      "WHERE active=1 AND cert_ok=1 ORDER BY updated_at DESC").fetchall()
    return [{"name": r["name"], "focus": r["focus"], "intro_video_url": r["intro_video_url"]}
            for r in rows]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_coach_directory_store.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add dashboard/coach_directory.py tests/test_coach_directory_store.py
git commit -m "feat(coach-directory): volunteer store"
```

---

### Task 2: Practitioner self-service signup + video upload + cert eligibility (`app.py`)

**Files:**
- Modify: `app.py` (add `_coach_cert_ok` helper + the practitioner-authed self-service route)
- Test: `tests/test_coach_signup_api.py`

**Interfaces:**
- Consumes: `dashboard/coach_directory.py:upsert_volunteer`/`set_active`, `dashboard/cert_submissions.py:list_for_email`, `dashboard/cert_rules.py:evaluate`, `_practitioner_session_pid()` (app.py — resolves the practitioner session token to a practitioner_id), `dashboard/practitioner_portal.py:practitioner_email_by_id`, `_PORTAL_ASSETS_DIR` + `secrets` (app.py), `_db_lock`, `LOG_DB`.
- Produces: `_coach_cert_ok(cx, email) -> bool`; `POST /api/practitioner/coach-profile`.

**Contract:** `POST /api/practitioner/coach-profile?token=<practitioner session token>` — authed via `_practitioner_session_pid()` → practitioner_id → email via `practitioner_email_by_id` (no pid/email → 401). Multipart form: `name`, `focus`, `capacity` (int), and an optional file field `video`. Behavior: clamp `capacity` to 0..12; if a `video` file is present, write it to `_PORTAL_ASSETS_DIR` as `coach-<hex>.mp4` and set `intro_video_url = "/portal-asset/<that filename>"` (else keep the `intro_video_url` form field if provided); `cert_ok = _coach_cert_ok(cx, email)`; `upsert_volunteer(email, name, focus, intro_video_url, capacity, cert_ok)`; then `set_active(email, 1 if capacity > 0 else 0)` (capacity 0 = not taking members → inactive). Returns `{ok:true, cert_ok:bool, capacity:int, listed:bool, intro_video_url:str}` where `listed == (cert_ok and capacity > 0)`. `_coach_cert_ok` filters to approved submissions, runs `evaluate`, returns its `complete`, and returns `False` on any error (fail-closed).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_coach_signup_api.py
import io
import sqlite3
from unittest import mock
import app as appmod
from dashboard import coach_directory as _cd


def _client():
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()


def _form(*, name="Cora", focus="sleep", capacity=3, with_video=True):
    data = {"name": name, "focus": focus, "capacity": str(capacity)}
    if with_video:
        data["video"] = (io.BytesIO(b"\x00\x01fakevideo"), "clip.mp4")
    return data


def test_signup_requires_practitioner_session():
    with mock.patch.object(appmod, "_practitioner_session_pid", return_value=None):
        r = _client().post("/api/practitioner/coach-profile?token=bad",
                           data=_form(), content_type="multipart/form-data")
    assert r.status_code == 401


def test_signup_certified_uploads_video_and_lists():
    c = _client()
    with mock.patch.object(appmod, "_practitioner_session_pid", return_value="pid1"), \
         mock.patch("dashboard.practitioner_portal.practitioner_email_by_id", return_value="ok@x.com"), \
         mock.patch.object(appmod, "_coach_cert_ok", return_value=True):
        r = c.post("/api/practitioner/coach-profile?token=t", data=_form(),
                   content_type="multipart/form-data")
    d = r.get_json()
    assert d["ok"] and d["cert_ok"] is True and d["listed"] is True and d["capacity"] == 3
    assert d["intro_video_url"].startswith("/portal-asset/coach-") and d["intro_video_url"].endswith(".mp4")
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _cd.init_coach_tables(cx)
        assert any(v["name"] == "Cora" for v in _cd.list_active(cx))


def test_capacity_clamped_and_zero_unlists():
    c = _client()
    with mock.patch.object(appmod, "_practitioner_session_pid", return_value="pid2"), \
         mock.patch("dashboard.practitioner_portal.practitioner_email_by_id", return_value="cap@x.com"), \
         mock.patch.object(appmod, "_coach_cert_ok", return_value=True):
        hi = c.post("/api/practitioner/coach-profile?token=t", data=_form(capacity=99),
                    content_type="multipart/form-data").get_json()
        zero = c.post("/api/practitioner/coach-profile?token=t",
                      data=_form(capacity=0, with_video=False),
                      content_type="multipart/form-data").get_json()
    assert hi["capacity"] == 12                    # clamped to 12
    assert zero["capacity"] == 0 and zero["listed"] is False   # 0 = not taking members
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _cd.init_coach_tables(cx)
        assert _cd.get_volunteer(cx, "cap@x.com")["active"] == 0


def test_signup_uncertified_not_listed():
    c = _client()
    with mock.patch.object(appmod, "_practitioner_session_pid", return_value="pid3"), \
         mock.patch("dashboard.practitioner_portal.practitioner_email_by_id", return_value="no@x.com"), \
         mock.patch.object(appmod, "_coach_cert_ok", return_value=False):
        d = c.post("/api/practitioner/coach-profile?token=t", data=_form(),
                   content_type="multipart/form-data").get_json()
    assert d["cert_ok"] is False and d["listed"] is False
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _cd.init_coach_tables(cx)
        assert _cd.get_volunteer(cx, "no@x.com")["cert_ok"] == 0  # stored, not listed


def test_coach_cert_ok_fail_closed():
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        assert appmod._coach_cert_ok(cx, "nobody@nowhere.com") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `export DATA_DIR="$(mktemp -d)" && doppler run -p remedy-match -c prd -- env DATA_DIR="$DATA_DIR" python3 -m pytest tests/test_coach_signup_api.py -q`
Expected: FAIL — route 404 / `_coach_cert_ok` undefined.

- [ ] **Step 3: Write minimal implementation**

Add to `app.py` (near the other community routes). `secrets` and `_PORTAL_ASSETS_DIR` already exist in app.py:

```python
def _coach_cert_ok(cx, email):
    """True only if the practitioner's APPROVED cert submissions satisfy the
    completion rules. Fail-closed: any error → False (an unverified student is
    never listed)."""
    try:
        from dashboard import cert_submissions as _cs, cert_rules as _cr
        subs = [s for s in _cs.list_for_email(cx, email) if s.get("status") == "approved"]
        return bool(_cr.evaluate(subs).get("complete"))
    except Exception:
        app.logger.exception("coach cert check failed for %s", email)
        return False


@app.route("/api/practitioner/coach-profile", methods=["POST"])
def practitioner_coach_profile():
    pid = _practitioner_session_pid()
    if not pid:
        return jsonify({"error": "unauthorized"}), 401
    from dashboard.practitioner_portal import practitioner_email_by_id
    email = (practitioner_email_by_id(pid) or "").strip().lower()
    if not email:
        return jsonify({"error": "unauthorized"}), 401
    from dashboard import coach_directory as _cd
    name = (request.form.get("name") or "").strip()
    focus = (request.form.get("focus") or "").strip()
    try:
        capacity = max(0, min(12, int(request.form.get("capacity") or 0)))
    except (TypeError, ValueError):
        capacity = 0
    intro_video_url = (request.form.get("intro_video_url") or "").strip()
    vf = request.files.get("video")
    if vf is not None and vf.filename:
        fname = f"coach-{secrets.token_hex(8)}.mp4"
        (_PORTAL_ASSETS_DIR / fname).write_bytes(vf.read())
        intro_video_url = f"/portal-asset/{fname}"
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _cd.init_coach_tables(cx)
        cert_ok = _coach_cert_ok(cx, email)
        _cd.upsert_volunteer(cx, email=email, name=name, focus=focus,
                             intro_video_url=intro_video_url, capacity=capacity,
                             cert_ok=cert_ok)
        _cd.set_active(cx, email, 1 if capacity > 0 else 0)
    return jsonify({"ok": True, "cert_ok": bool(cert_ok), "capacity": capacity,
                    "listed": bool(cert_ok and capacity > 0),
                    "intro_video_url": intro_video_url})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `export DATA_DIR="$(mktemp -d)" && doppler run -p remedy-match -c prd -- env DATA_DIR="$DATA_DIR" python3 -m pytest tests/test_coach_signup_api.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_coach_signup_api.py
git commit -m "feat(coach-directory): practitioner self-service signup + video upload + cert check"
```

---

### Task 3: Member directory route (`app.py`)

**Files:**
- Modify: `app.py` (add the member-gated directory route)
- Test: `tests/test_coach_directory_api.py`

**Interfaces:**
- Consumes: `dashboard/coach_directory.py:list_active`, `dashboard/coaching.py:active_window`, `_evox_ident`, `LOG_DB`.
- Produces: `GET /api/community/coaches`.

**Contract:** `GET /api/community/coaches?token=…` — bad token → 404. If the member has NO active coaching window → `{eligible:false, coaches:[]}`. Else → `{eligible:true, coaches:[{name,focus,intro_video_url}]}` from `list_active`. Never exposes a coach email.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_coach_directory_api.py
import sqlite3
from unittest import mock
import app as appmod
from dashboard import coach_directory as _cd


def _client():
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()


def _seed(email, *, with_window):
    from datetime import datetime, timezone, timedelta
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        from dashboard import evox as _ev, client_portal as _cp, coaching as _co
        _ev.init_evox_tables(cx); _cp.init_client_portal_table(cx)
        _cd.init_coach_tables(cx); _co.init_coaching_table(cx)
        _cd.upsert_volunteer(cx, email="coach@x.com", name="Cora", focus="sleep",
                             intro_video_url="https://rumble.com/v-c", capacity=3, cert_ok=1)
        if with_window:
            now = datetime.now(timezone.utc)
            started = now.isoformat()
            ends = (now + timedelta(days=10)).isoformat()
            cx.execute("INSERT INTO coaching_windows (email,order_id,started_at,ends_at,"
                       "source,created_at) VALUES (?,?,?,?,?,?)",
                       (email, 1, started, ends, "test", started))
        token = _ev.ensure_portal_token(cx, email, "Mem")
        cx.commit()
    return token


def test_directory_member_with_window_sees_coaches():
    c = _client(); tok = _seed("m@x.com", with_window=True)
    d = c.get(f"/api/community/coaches?token={tok}").get_json()
    assert d["eligible"] is True
    assert d["coaches"][0]["name"] == "Cora"
    assert "email" not in d["coaches"][0]           # no coach email exposed


def test_directory_member_without_window_ineligible():
    c = _client(); tok = _seed("m2@x.com", with_window=False)
    d = c.get(f"/api/community/coaches?token={tok}").get_json()
    assert d["eligible"] is False and d["coaches"] == []


def test_directory_bad_token_404():
    assert _client().get("/api/community/coaches?token=nope").status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `export DATA_DIR="$(mktemp -d)" && doppler run -p remedy-match -c prd -- env DATA_DIR="$DATA_DIR" python3 -m pytest tests/test_coach_directory_api.py -q`
Expected: FAIL — route 404.

- [ ] **Step 3: Write minimal implementation**

Add to `app.py`:

```python
@app.route("/api/community/coaches")
def community_coaches():
    from dashboard import coach_directory as _cd, coaching as _co
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _cd.init_coach_tables(cx); _co.init_coaching_table(cx)
        ident = _evox_ident(cx, request.args.get("token", ""))
        if ident is None:
            return jsonify({"error": "not_found"}), 404
        if not _co.active_window(cx, ident.email):
            return jsonify({"eligible": False, "coaches": []})
        return jsonify({"eligible": True, "coaches": _cd.list_active(cx)})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `export DATA_DIR="$(mktemp -d)" && doppler run -p remedy-match -c prd -- env DATA_DIR="$DATA_DIR" python3 -m pytest tests/test_coach_directory_api.py -q`
Expected: PASS (3 passed). If `coaching_windows` insert columns differ, first `grep -n "CREATE TABLE IF NOT EXISTS coaching_windows" dashboard/coaching.py` and match the seed to the real columns.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_coach_directory_api.py
git commit -m "feat(coach-directory): member-gated coach directory route"
```

---

### Task 4: Portal card (`static/client-portal.html`)

**Files:**
- Modify: `static/client-portal.html`
- Test: manual JS parse check.

**Interfaces:**
- Consumes: `GET /api/community/coaches?token=…` → `{eligible, coaches:[{name,focus,intro_video_url}]}`.

**Design note:** read `static/client-portal.html` first (the `token` var, the card idiom). Add a "Meet your coaches" card, wrapped in `<!-- BEGIN coaches script -->` / `<!-- END coaches script -->`:
- On load, `fetch('/api/community/coaches?token='+token)`. If `!d.eligible` OR `d.coaches` is empty, hide the card.
- Else render each coach: name + focus via `textContent`, and the intro video as a Rumble `<iframe>` whose `src` is `intro_video_url`. Add one quiet line: "Choosing your coach is coming soon."
- Card heading: "Meet your coaches". Copy: no em dashes, no ALL CAPS. Server strings via `textContent` (only the video uses the URL as an iframe src).

- [ ] **Step 1: Read the page and add the card**

Read `static/client-portal.html`. Add the "Meet your coaches" card + its fetch/render per the design note.

- [ ] **Step 2: Verify the page JS parses**

Run: `cd /tmp/wt-deploy-chat-cca589e9 && node --check <(python3 -c "import re; h=open('static/client-portal.html').read(); print('\n;\n'.join(re.findall(r'<script>(.*?)</script>', h, re.S)))")`
Expected: no output (clean parse).

- [ ] **Step 3: Commit**

```bash
git add static/client-portal.html
git commit -m "feat(coach-directory): Meet your coaches portal card"
```

---

### Task 5: Practitioner-portal coaching control (`static/practitioner-portal.html`)

**Files:**
- Modify: `static/practitioner-portal.html`
- Test: manual JS parse check.

**Interfaces:**
- Consumes: `POST /api/practitioner/coach-profile?token=…` (multipart: `name`, `focus`, `capacity`, `video` file) → `{ok, cert_ok, capacity, listed, intro_video_url}` (Task 2).

**Design note:** read `static/practitioner-portal.html` first — how it obtains the practitioner **session token** (the value passed as `?token=` to practitioner APIs) and its card idiom. Add a "Coaching volunteer" card, wrapped in `<!-- BEGIN coach-volunteer script -->` / `<!-- END coach-volunteer script -->`:
- Fields: display name (text), focus (short text), **capacity** (a number input or slider, min 0 max 12, integer; label it so 0 reads as "not taking new members right now"), and a **file input** `accept="video/mp4,video/*"` for the intro video (upload; in-browser recording is a fast-follow).
- On submit: build a `FormData` with `name`, `focus`, `capacity`, and the selected `video` file (omit `video` if none chosen), and `fetch('/api/practitioner/coach-profile?token='+TOKEN, {method:'POST', body: formData})` (do NOT set Content-Type; the browser sets the multipart boundary).
- On the response: show a quiet status line built with `textContent` — if `!d.cert_ok`, "Your profile is saved. You will be listed once your certification is complete." If `d.cert_ok && d.listed`, "You are listed for members to find." If `d.capacity === 0`, "Saved. You are set to not taking new members right now."
- Copy: no em dashes, no ALL CAPS. All status/response text via `textContent`.

- [ ] **Step 1: Read the page and add the card**

Read `static/practitioner-portal.html`; add the "Coaching volunteer" card + its FormData submit per the design note, using the page's existing practitioner session token var.

- [ ] **Step 2: Verify the page JS parses**

Run: `cd /tmp/wt-deploy-chat-cca589e9 && node --check <(python3 -c "import re; h=open('static/practitioner-portal.html').read(); print('\n;\n'.join(re.findall(r'<script>(.*?)</script>', h, re.S)))")`
Expected: no output (clean parse).

- [ ] **Step 3: Commit**

```bash
git add static/practitioner-portal.html
git commit -m "feat(coach-directory): practitioner portal coaching-volunteer control"
```

---

## Definition of Done

- A cert-complete student sets capacity (0-12) and uploads an intro video from their own practitioner portal; a non-cert-complete one can save but is not listed; capacity 0 unlists them.
- Members with an active coaching window see the coach directory (name, focus, intro video) on their portal; ineligible members do not; no coach email is ever exposed.
- All new tests pass; Community A/B/C1/C3 and cert/coaching stores are untouched (the directory reads them, writes only `coach_volunteers`).

## Deferred (not in this plan)

- **In-browser video recording** (camera capture) in the practitioner portal — this slice ships file **upload**; recording is the immediate fast-follow.
- Verifying `portal_role='coach'` in Postgres at signup (the practitioner-session auth already proves they're a practitioner; cert-completeness is the substantive automated gate).
- Slice 2: member requests a coach → student accepts up to `capacity` (pairing).
- Slice 3: the 1:1 coaching thread + report/block + moderation.
- Coach-side dashboard; interest-matching of coaches to members; capacity enforcement.
