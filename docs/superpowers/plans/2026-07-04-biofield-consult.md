# Biofield Consult Booking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let an eligible client book a 30-minute Biofield Consult with Dr. Glen over Zoom, from inside their existing portal, gated by a manual "consult ready" unlock.

**Architecture:** Generalize the shipped EVOX booking engine (`dashboard/evox.py`) to carry a `session_type` + `medium`, add a fresh Zoom Server-to-Server module (`dashboard/zoom.py`), a consult-eligibility module (`dashboard/consult.py`), consult routes + confirmations in `app.py`, a "Mark consult ready" console flip, and a consult card in the client portal. Reuses EVOX's availability math, `create_booking`, ICS, portal-token auth, and email rail.

**Tech Stack:** Python 3 / Flask, sqlite (`chat_log.db`, `?` placeholders, `_db_lock`), Zoom Server-to-Server OAuth, existing `send_evox_email` rail, `resolve_identity` portal auth, vanilla-JS portal. Tests: pytest via `app.app.test_client()`.

## Global Constraints

- **DB = sqlite** `chat_log.db` via `cx = sqlite3.connect(LOG_DB)`; placeholders `?`; wrap writes in `with _db_lock, sqlite3.connect(LOG_DB) as cx:` and set `cx.row_factory = sqlite3.Row`. Each module owns an idempotent `init_*` it calls at the top of routes. Additive migrations use guarded `try: cx.execute("ALTER TABLE ... ADD COLUMN ...") except Exception: pass`.
- **Timestamps** use `datetime.now(timezone.utc).isoformat()` (NOT deprecated `datetime.utcnow()`). Slot times are naive ISO `YYYY-MM-DDTHH:MM:SS`, HST; pure functions get `now` passed in.
- **Auth:** consult client APIs authenticate via portal token: `dashboard.portal_identity.resolve_identity(cx, token=..., session_token=request.cookies.get("rm_portal_session",""), client_login_enabled=_client_login_enabled())`; `ident.email` (lowercased) keys everything. Console flip uses `_portal_console_ok()` (accepts `X-Console-Key` header, `?key=`, or owner token).
- **Session config (verbatim):** `session_type = "biofield-consult"`, `practitioner = "glen"`, `duration_min = 30`, `medium = "video"`, gate = manual `consult_ready` flag, payment = none at booking.
- **Env (read at module load in `app.py`, passed into helpers):**
  - `GLEN_CONSULT_HOURS` — default `"1-7:09:00-17:00"` (all days, 09:00–17:00 HST, 30-min grid — Glen's answer).
  - `GLEN_ZOOM_USER` — Glen's Zoom user id/email; default `"me"`.
  - Zoom creds already in env: `ZOOM_ACCOUNT_ID`, `ZOOM_CLIENT_ID`, `ZOOM_CLIENT_SECRET`.
- **Zoom failure never blocks a booking** — if meeting creation fails, the booking still succeeds; confirmation says "Zoom link to follow" and the error is logged.
- **Client-facing copy** follows Glen's standing rules: **no em dashes, no ALL CAPS**. Warm, clear, consultative.
- **Test command:** `mkdir -p /tmp/dc-test && doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/dc-test python3 -m pytest tests/<file>.py -q`. Pure tests that import only `dashboard/*` (no `app`) run with plain `python3 -m pytest`. Keep pure functions importable without importing `app`.
- **EVOX regression:** the shipped EVOX flow must stay green after the `session_type` generalization — `evox_bookings` rows default `session_type='evox'`, `medium='phone'`.

---

## File Structure

- **Modify `dashboard/evox.py`** — generalize `create_booking` to accept `session_type`/`medium`, vary the mirror `calendar_events` row's `location`/`summary`, and add lazy columns (`session_type`, `medium`, `zoom_join_url`, `zoom_meeting_id`) to `evox_bookings`. Booking core stays here.
- **Create `dashboard/consult.py`** — consult session config, `consult_eligibility` table + `set_consult_ready`/`consult_is_ready`, `has_paid_purchase(cx, email, slug)`.
- **Create `dashboard/zoom.py`** — Server-to-Server OAuth `get_token()` + `create_meeting(...)`.
- **Modify `app.py`** — config globals; `/api/console/consult-ready`; `/api/consult/state|availability|book`; `_consult_send_confirmations`.
- **Modify `dashboard/portal_view.py`** — add a `consult` block to `get_portal_view`.
- **Modify `static/client-portal.html`** — a "Biofield Consult" card + slot-picker/book JS.
- **Modify `static/console-biofield-portal.html`** — a "Mark consult ready" toggle.
- **Create tests** — `tests/test_consult_pure.py` (no doppler), `tests/test_consult_api.py` (doppler).

---

### Task 1: Generalize `create_booking` for session_type + medium

**Files:**
- Modify: `dashboard/evox.py` (`init_evox_tables` ~line 25; `create_booking` lines 155-184)
- Test: `tests/test_consult_pure.py`

**Interfaces:**
- Produces: `create_booking(cx, email, start_ts, *, duration_min=60, prepaid=False, practitioner="rae", session_type="evox", medium="phone", tag_fn=None) -> dict` (adds `session_type`, `medium` to the return dict). The mirror `calendar_events` row uses `location="Video" if medium=="video" else "Phone"` and `summary=f"{'Biofield Consult' if session_type=='biofield-consult' else 'EVOX'} — {email}"`. New lazy columns on `evox_bookings`: `session_type TEXT DEFAULT 'evox'`, `medium TEXT DEFAULT 'phone'`, `zoom_join_url TEXT`, `zoom_meeting_id TEXT`.
- Consumes: nothing.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_consult_pure.py
import sqlite3
from dashboard import evox

def _cx():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    evox.init_evox_tables(cx)
    cx.execute("""CREATE TABLE calendar_events (id INTEGER PRIMARY KEY AUTOINCREMENT,
        pushed_at TEXT, google_cal_id TEXT, google_event_id TEXT, calendar_name TEXT,
        summary TEXT, start TEXT, end TEXT, location TEXT, owner TEXT, status TEXT,
        cal_alert INTEGER, UNIQUE(google_cal_id, google_event_id))""")
    cx.commit(); return cx

def test_consult_booking_sets_type_medium_and_video_calendar_row():
    cx = _cx()
    b = evox.create_booking(cx, "c@x.com", "2026-07-06T13:00:00",
                            duration_min=30, practitioner="glen",
                            session_type="biofield-consult", medium="video")
    assert b["end_ts"] == "2026-07-06T13:30:00"
    assert b["session_type"] == "biofield-consult" and b["medium"] == "video"
    row = cx.execute("SELECT owner, location, summary, session_type, medium "
                     "FROM evox_bookings JOIN calendar_events "
                     "ON calendar_events.google_event_id='biofield-consult-'||evox_bookings.id").fetchone()
    assert row["owner"] == "glen" and row["location"] == "Video"
    assert "Biofield Consult" in row["summary"]
    assert row["session_type"] == "biofield-consult" and row["medium"] == "video"

def test_evox_booking_defaults_unchanged():
    cx = _cx()
    b = evox.create_booking(cx, "e@x.com", "2026-07-06T11:00:00")
    assert b["session_type"] == "evox" and b["medium"] == "phone"
    row = cx.execute("SELECT location, summary FROM calendar_events").fetchone()
    assert row["location"] == "Phone" and row["summary"] == "EVOX — e@x.com"
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_consult_pure.py -q`
Expected: FAIL (`create_booking() got an unexpected keyword argument 'session_type'`).

- [ ] **Step 3: Generalize `init_evox_tables` columns + `create_booking`**

In `init_evox_tables(cx)` (after the `evox_bookings` CREATE), add guarded lazy columns:

```python
    for _col, _decl in (("session_type", "TEXT DEFAULT 'evox'"),
                        ("medium", "TEXT DEFAULT 'phone'"),
                        ("zoom_join_url", "TEXT"), ("zoom_meeting_id", "TEXT")):
        try:
            cx.execute(f"ALTER TABLE evox_bookings ADD COLUMN {_col} {_decl}")
        except Exception:
            pass
```

Rewrite `create_booking`’s signature + body (keep the `SlotTaken`/rollback logic identical):

```python
def create_booking(cx, email: str, start_ts: str, *, duration_min: int = 60,
                   prepaid: bool = False, practitioner: str = "rae",
                   session_type: str = "evox", medium: str = "phone", tag_fn=None) -> dict:
    email = (email or "").strip().lower()
    start_dt = datetime.fromisoformat(start_ts[:19])
    end_ts = (start_dt + timedelta(minutes=duration_min)).isoformat()
    now = datetime.now(timezone.utc).isoformat()
    ics_uid = f"{session_type}-{secrets.token_hex(8)}@illtowell.com"
    try:
        cur = cx.execute(
            "INSERT INTO evox_bookings (email,practitioner,start_ts,end_ts,status,"
            "prepaid,ics_uid,created_at,session_type,medium) "
            "VALUES (?,?,?,?,'booked',?,?,?,?,?)",
            (email, practitioner, start_ts, end_ts, 1 if prepaid else 0, ics_uid, now,
             session_type, medium))
    except sqlite3.IntegrityError as e:
        cx.rollback()
        if "UNIQUE" in str(e).upper():
            raise SlotTaken(start_ts)
        raise
    booking_id = cur.lastrowid
    ev_id = f"{session_type}-{booking_id}"
    location = "Video" if medium == "video" else "Phone"
    label = "Biofield Consult" if session_type == "biofield-consult" else "EVOX"
    cx.execute(
        "INSERT INTO calendar_events (pushed_at,google_cal_id,google_event_id,"
        "calendar_name,summary,start,end,location,owner,status,cal_alert) "
        "VALUES (?, 'delegated', ?, ?, ?, ?, ?, ?, ?, 'visible', 0)",
        (now, ev_id, f"{label} booking", f"{label} — {email}", start_ts, end_ts,
         location, practitioner))
    cx.execute("UPDATE evox_bookings SET calendar_event_id=? WHERE id=?", (ev_id, booking_id))
    cx.commit()
    if tag_fn:
        tag_fn(email, ["evox-client", "evox-ready"])
    return {"id": booking_id, "email": email, "start_ts": start_ts, "end_ts": end_ts,
            "ics_uid": ics_uid, "prepaid": prepaid, "session_type": session_type,
            "medium": medium}
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_consult_pure.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Run the EVOX pure suite to confirm no regression**

Run: `python3 -m pytest tests/test_evox_pure.py -q`
Expected: PASS (all prior EVOX tests still green — the calendar row now uses `EVOX — {email}` which matches the existing EVOX assertions).

- [ ] **Step 6: Commit**

```bash
git add dashboard/evox.py tests/test_consult_pure.py
git commit -m "feat(consult): generalize create_booking for session_type + medium"
```

---

### Task 2: `dashboard/consult.py` — eligibility + paid-purchase check

**Files:**
- Create: `dashboard/consult.py`
- Test: `tests/test_consult_pure.py`

**Interfaces:**
- Produces:
  - `CONSULT = {"session_type": "biofield-consult", "practitioner": "glen", "duration_min": 30, "medium": "video", "test_slug": "biofield-analysis"}`
  - `init_consult_tables(cx)` — creates `consult_eligibility(email TEXT, session_type TEXT, ready INTEGER DEFAULT 0, marked_at TEXT, PRIMARY KEY(email, session_type))`.
  - `set_consult_ready(cx, email, ready: bool, session_type="biofield-consult") -> bool` (returns new state).
  - `consult_is_ready(cx, email, session_type="biofield-consult") -> bool`.
  - `has_paid_purchase(cx, email, slug) -> bool` — True iff an `orders` row for that lowercased email has a line with that slug AND the order is paid (`pay_status='paid'` OR `paid_cents>0`). Absent `orders` table → False.
- Consumes: nothing.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_consult_pure.py
from dashboard import consult
import json

def _ccx():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    consult.init_consult_tables(cx); return cx

def test_consult_ready_roundtrip():
    cx = _ccx()
    assert consult.consult_is_ready(cx, "A@x.com") is False
    assert consult.set_consult_ready(cx, "a@x.com", True) is True
    assert consult.consult_is_ready(cx, "A@x.com") is True      # lowercased
    assert consult.set_consult_ready(cx, "a@x.com", False) is False
    assert consult.consult_is_ready(cx, "a@x.com") is False

def test_has_paid_purchase():
    cx = _ccx()
    cx.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, email TEXT, items_json TEXT, "
               "pay_status TEXT, paid_cents INTEGER)")
    cx.execute("INSERT INTO orders (email, items_json, pay_status, paid_cents) VALUES (?,?,?,?)",
               ("buyer@x.com", json.dumps([{"slug": "biofield-analysis"}]), "paid", 30000))
    cx.execute("INSERT INTO orders (email, items_json, pay_status, paid_cents) VALUES (?,?,?,?)",
               ("unpaid@x.com", json.dumps([{"slug": "biofield-analysis"}]), "unpaid", 0))
    cx.commit()
    assert consult.has_paid_purchase(cx, "BUYER@x.com", "biofield-analysis") is True
    assert consult.has_paid_purchase(cx, "unpaid@x.com", "biofield-analysis") is False
    assert consult.has_paid_purchase(cx, "nobody@x.com", "biofield-analysis") is False
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_consult_pure.py -q`
Expected: FAIL (`ModuleNotFoundError: dashboard.consult`).

- [ ] **Step 3: Create `dashboard/consult.py`**

```python
"""Biofield Consult: eligibility gate + paid-test detection. Stdlib-only; import
without importing app."""
import sqlite3, json as _json
from datetime import datetime, timezone

CONSULT = {"session_type": "biofield-consult", "practitioner": "glen",
           "duration_min": 30, "medium": "video", "test_slug": "biofield-analysis"}


def init_consult_tables(cx) -> None:
    cx.execute("""CREATE TABLE IF NOT EXISTS consult_eligibility (
        email TEXT NOT NULL, session_type TEXT NOT NULL,
        ready INTEGER DEFAULT 0, marked_at TEXT,
        PRIMARY KEY (email, session_type))""")
    cx.commit()


def set_consult_ready(cx, email: str, ready: bool,
                      session_type: str = "biofield-consult") -> bool:
    email = (email or "").strip().lower()
    now = datetime.now(timezone.utc).isoformat()
    cx.execute("INSERT INTO consult_eligibility (email, session_type, ready, marked_at) "
               "VALUES (?,?,?,?) ON CONFLICT(email, session_type) "
               "DO UPDATE SET ready=excluded.ready, marked_at=excluded.marked_at",
               (email, session_type, 1 if ready else 0, now))
    cx.commit()
    return bool(ready)


def consult_is_ready(cx, email: str, session_type: str = "biofield-consult") -> bool:
    email = (email or "").strip().lower()
    r = cx.execute("SELECT ready FROM consult_eligibility WHERE email=? AND session_type=?",
                   (email, session_type)).fetchone()
    return bool(r[0]) if r else False


def has_paid_purchase(cx, email: str, slug: str) -> bool:
    email = (email or "").strip().lower()
    try:
        rows = cx.execute("SELECT items_json, pay_status, paid_cents FROM orders "
                          "WHERE lower(email)=?", (email,)).fetchall()
    except sqlite3.OperationalError:
        return False
    for items, pay_status, paid_cents in rows:
        paid = (str(pay_status or "").lower() == "paid") or (int(paid_cents or 0) > 0)
        if not paid:
            continue
        try:
            for line in _json.loads(items or "[]"):
                if (line.get("slug") or "") == slug:
                    return True
        except Exception:
            continue
    return False
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_consult_pure.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add dashboard/consult.py tests/test_consult_pure.py
git commit -m "feat(consult): eligibility flag + paid-test detection module"
```

---

### Task 3: `dashboard/zoom.py` — Server-to-Server OAuth + create_meeting

**Files:**
- Create: `dashboard/zoom.py`
- Test: `tests/test_consult_pure.py`

**Interfaces:**
- Produces:
  - `get_token(account_id, client_id, client_secret, *, _now=None) -> str` — S2S OAuth token, cached in-process ~55 min (module-level cache keyed by client_id).
  - `create_meeting(token, *, host, topic, start_iso, duration_min, timezone="Pacific/Honolulu", opener=None) -> dict` returns `{"join_url", "meeting_id", "start_url"}`. `opener` is an injectable `urlopen`-like callable for tests; defaults to `urllib.request.urlopen`. Builds `POST https://api.zoom.us/v2/users/{host}/meetings` body `{"topic","type":2,"start_time","duration","timezone","settings":{"waiting_room":True,"join_before_host":False}}`.
- Consumes: nothing (stdlib `urllib`).

- [ ] **Step 1: Write the failing test (mock HTTP; no network)**

```python
# append to tests/test_consult_pure.py
from dashboard import zoom
import io, json as _json2

def test_create_meeting_builds_request_and_parses_response():
    captured = {}
    def fake_opener(req, timeout=None):
        captured["url"] = req.full_url
        captured["body"] = _json2.loads(req.data.decode())
        captured["auth"] = req.get_header("Authorization")
        return io.BytesIO(_json2.dumps({
            "id": 87654321, "join_url": "https://zoom.us/j/87654321",
            "start_url": "https://zoom.us/s/87654321"}).encode())
    out = zoom.create_meeting("tok123", host="me", topic="Biofield Consult",
                              start_iso="2026-07-06T13:00:00", duration_min=30,
                              opener=fake_opener)
    assert out == {"join_url": "https://zoom.us/j/87654321",
                   "meeting_id": "87654321", "start_url": "https://zoom.us/s/87654321"}
    assert captured["url"] == "https://api.zoom.us/v2/users/me/meetings"
    assert captured["auth"] == "Bearer tok123"
    assert captured["body"]["type"] == 2 and captured["body"]["duration"] == 30
    assert captured["body"]["settings"]["waiting_room"] is True
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_consult_pure.py -q`
Expected: FAIL (`ModuleNotFoundError: dashboard.zoom`).

- [ ] **Step 3: Create `dashboard/zoom.py`**

```python
"""Zoom Server-to-Server OAuth + scheduled-meeting creation. Stdlib-only."""
import json, base64, urllib.request, urllib.parse

_TOKEN_CACHE = {}  # client_id -> (token, expiry_epoch)


def get_token(account_id, client_id, client_secret, *, _now=None):
    import time
    now = _now if _now is not None else time.time()
    cached = _TOKEN_CACHE.get(client_id)
    if cached and cached[1] > now:
        return cached[0]
    url = "https://zoom.us/oauth/token?" + urllib.parse.urlencode(
        {"grant_type": "account_credentials", "account_id": account_id})
    basic = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    req = urllib.request.Request(url, data=b"", method="POST",
                                 headers={"Authorization": f"Basic {basic}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        d = json.load(r)
    tok = d["access_token"]
    _TOKEN_CACHE[client_id] = (tok, now + int(d.get("expires_in", 3600)) - 300)
    return tok


def create_meeting(token, *, host, topic, start_iso, duration_min,
                   timezone="Pacific/Honolulu", opener=None):
    opener = opener or urllib.request.urlopen
    body = {"topic": topic, "type": 2, "start_time": start_iso,
            "duration": int(duration_min), "timezone": timezone,
            "settings": {"waiting_room": True, "join_before_host": False}}
    req = urllib.request.Request(
        f"https://api.zoom.us/v2/users/{host}/meetings",
        data=json.dumps(body).encode(), method="POST",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    with opener(req, timeout=30) as r:
        d = json.load(r)
    return {"join_url": d.get("join_url"), "meeting_id": str(d.get("id") or ""),
            "start_url": d.get("start_url")}
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_consult_pure.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add dashboard/zoom.py tests/test_consult_pure.py
git commit -m "feat(consult): Zoom S2S OAuth + create_meeting module"
```

---

### Task 4: "Mark consult ready" console flip

**Files:**
- Modify: `app.py` (new route near the other console-biofield routes ~`app.py:15109`)
- Modify: `static/console-biofield-portal.html` (add a toggle using the page's `api()` helper)
- Test: `tests/test_consult_api.py`

**Interfaces:**
- Consumes: `dashboard.consult` (Task 2), `_portal_console_ok()` (`app.py:13728`), `_db_lock`, `LOG_DB`.
- Produces (HTTP): `POST /api/console/consult-ready {email, ready:true|false}` → `_portal_console_ok` gate; sets the flag via `consult.set_consult_ready`; returns `{ok:true, email, ready}`. 401 when unauthorized, 400 when email missing.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_consult_api.py
import os, pytest
if not os.environ.get("PINECONE_API_KEY"):
    pytest.skip("needs doppler env for import app", allow_module_level=True)
import app as appmod

@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    monkeypatch.setattr(appmod, "CONSOLE_SECRET", "test-secret")
    monkeypatch.setattr(appmod, "GLEN_CONSULT_HOURS", "1-7:09:00-17:00", raising=False)
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()

ADMIN = {"X-Console-Key": "test-secret"}

def test_consult_ready_flip_requires_auth(client):
    r = client.post("/api/console/consult-ready", json={"email": "c@x.com", "ready": True})
    assert r.status_code == 401

def test_consult_ready_flip_sets_flag(client):
    r = client.post("/api/console/consult-ready",
                    json={"email": "c@x.com", "ready": True}, headers=ADMIN)
    assert r.status_code == 200 and r.get_json()["ready"] is True
    import sqlite3
    from dashboard import consult
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert consult.consult_is_ready(cx, "c@x.com") is True
```

- [ ] **Step 2: Run to verify failure**

Run: `mkdir -p /tmp/dc-test && doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/dc-test python3 -m pytest tests/test_consult_api.py -q`
Expected: FAIL (404 on the route).

- [ ] **Step 3: Add the route to `app.py`**

```python
@app.route("/api/console/consult-ready", methods=["POST"])
def api_console_consult_ready():
    if not _portal_console_ok():
        return jsonify({"error": "unauthorized"}), 401
    from dashboard import consult as _consult
    body = request.get_json(silent=True) or {}
    email = (body.get("email") or "").strip().lower()
    if not email:
        return jsonify({"error": "email required"}), 400
    ready = bool(body.get("ready"))
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _consult.init_consult_tables(cx)
        new_state = _consult.set_consult_ready(cx, email, ready)
    return jsonify({"ok": True, "email": email, "ready": new_state})
```

- [ ] **Step 4: Run to verify pass**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/dc-test python3 -m pytest tests/test_consult_api.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Add the console toggle to `static/console-biofield-portal.html`**

Near the publish controls, add (reusing the page's existing `api()` helper + `key()`):

```html
<div style="margin-top:10px">
  <button onclick="markConsultReady(true)">Mark consult ready</button>
  <button onclick="markConsultReady(false)">Clear consult ready</button>
  <span id="consult-ready-msg"></span>
</div>
<script>
async function markConsultReady(ready){
  const email = (document.getElementById('email')?.value || '').trim();
  if(!email){ document.getElementById('consult-ready-msg').textContent = 'Enter an email first.'; return; }
  const r = await api('POST', '/api/console/consult-ready', {email, ready});
  document.getElementById('consult-ready-msg').textContent =
    r.ok ? (ready ? 'Consult unlocked ✓' : 'Consult locked') : ('Error ' + r.status);
}
</script>
```

(If the page's email input has a different id than `email`, use that id — inspect the page and match it. The `api()` helper is defined at `static/console-biofield-portal.html:215`.)

- [ ] **Step 6: Commit**

```bash
git add app.py static/console-biofield-portal.html tests/test_consult_api.py
git commit -m "feat(consult): console Mark-consult-ready flip"
```

---

### Task 5: Consult routes — state / availability / book (+ Zoom)

**Files:**
- Modify: `app.py` (config globals; routes near the EVOX routes ~`app.py:13874`; a `_hst_now`/`_evox_days` already exist from EVOX — reuse them)
- Test: `tests/test_consult_api.py`

**Interfaces:**
- Consumes: `dashboard.evox` (`available_slots`, `booked_starts`, `rae_busy_intervals`, `create_booking`, `SlotTaken`, `init_evox_tables`), `dashboard.consult` (`CONSULT`, `consult_is_ready`, `init_consult_tables`), `dashboard.zoom` (`get_token`, `create_meeting`), `resolve_identity`, `_evox_days(range_name)`, `_hst_now()`, `_db_lock`.
- Produces (HTTP), all portal-token authed via `resolve_identity`:
  - `GET /api/consult/state?token=` → `{ready, booked, stages:{member,test_paid,ready}}` (404 if identity missing).
  - `GET /api/consult/availability?token=&range=week` → `{slots}`; 403 `{error:"not_ready"}` if `consult_is_ready` is false.
  - `POST /api/consult/book?token= {start_ts}` → readiness gate; re-validate `start_ts` against live `available_slots` for `practitioner='glen'`; `create_booking(session_type='biofield-consult', practitioner='glen', duration_min=30, medium='video')`; then create the Zoom meeting (best-effort) and persist `zoom_join_url`/`zoom_meeting_id`; return `{ok, start_ts, join_url}`. `SlotTaken` → 409 `{error:"slot_taken"}`; unavailable slot → 409 `{error:"slot_unavailable"}`.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_consult_api.py
def _mk_portal(email="p@x.com"):
    import sqlite3
    from dashboard import client_portal as cp
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cp.init_client_portal_table(cx)
        token, _ = cp.upsert_portal(cx, email, "P", {"source": "test"})
    return token

def test_availability_blocked_until_ready(client, monkeypatch):
    monkeypatch.setattr(appmod, "send_evox_email", lambda *a, **k: ("console-log", None), raising=False)
    tok = _mk_portal("p1@x.com")
    r = client.get(f"/api/consult/availability?token={tok}&range=week")
    assert r.status_code == 403 and r.get_json()["error"] == "not_ready"

def test_full_consult_flow(client, monkeypatch):
    monkeypatch.setattr(appmod, "send_evox_email", lambda *a, **k: ("console-log", None), raising=False)
    monkeypatch.setattr("dashboard.zoom.get_token", lambda *a, **k: "tok")
    monkeypatch.setattr("dashboard.zoom.create_meeting",
                        lambda *a, **k: {"join_url": "https://zoom.us/j/1", "meeting_id": "1", "start_url": "x"})
    tok = _mk_portal("p2@x.com")
    client.post("/api/console/consult-ready", json={"email": "p2@x.com", "ready": True}, headers=ADMIN)
    slots = client.get(f"/api/consult/availability?token={tok}&range=week").get_json()["slots"]
    assert slots
    r = client.post(f"/api/consult/book?token={tok}", json={"start_ts": slots[0]})
    assert r.status_code == 200 and r.get_json()["join_url"] == "https://zoom.us/j/1"
    r2 = client.post(f"/api/consult/book?token={tok}", json={"start_ts": slots[0]})
    assert r2.status_code == 409
```

- [ ] **Step 2: Run to verify failure**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/dc-test python3 -m pytest tests/test_consult_api.py -q`
Expected: FAIL (404 on `/api/consult/availability`).

- [ ] **Step 3: Add config + routes to `app.py`**

Config (near `EVOX_HOURS`):

```python
GLEN_CONSULT_HOURS = os.environ.get("GLEN_CONSULT_HOURS", "1-7:09:00-17:00")
GLEN_ZOOM_USER = os.environ.get("GLEN_ZOOM_USER", "me")
```

Routes (near the EVOX routes; reuse `_evox_ident`, `_evox_days`, `_hst_now` from the EVOX build):

```python
def _consult_ident(cx, token):
    from dashboard import portal_identity as _pi
    return _pi.resolve_identity(cx, token=token,
                                session_token=request.cookies.get("rm_portal_session", ""),
                                client_login_enabled=_client_login_enabled())

@app.route("/api/consult/state")
def consult_state():
    from dashboard import consult as _consult
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _consult.init_consult_tables(cx)
        ident = _consult_ident(cx, request.args.get("token", ""))
        if ident is None:
            return jsonify({"error": "not_found"}), 404
        ready = _consult.consult_is_ready(cx, ident.email)
        booked = ident.email in _get_consult_booked(cx)
        stages = {"member": _is_paid_member(ident.email),
                  "test_paid": _consult.has_paid_purchase(cx, ident.email, _consult.CONSULT["test_slug"]),
                  "ready": ready}
        return jsonify({"ready": ready, "booked": booked, "stages": stages})

def _get_consult_booked(cx):
    try:
        rows = cx.execute("SELECT lower(email) FROM evox_bookings "
                          "WHERE session_type='biofield-consult' AND status='booked'").fetchall()
        return {r[0] for r in rows}
    except Exception:
        return set()

@app.route("/api/consult/availability")
def consult_availability():
    from dashboard import evox as _ev, consult as _consult
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _ev.init_evox_tables(cx); _consult.init_consult_tables(cx)
        ident = _consult_ident(cx, request.args.get("token", ""))
        if ident is None:
            return jsonify({"error": "not_found"}), 404
        if not _consult.consult_is_ready(cx, ident.email):
            return jsonify({"error": "not_ready"}), 403
        days = _evox_days(request.args.get("range", "week"))
        lo, hi = days[0].isoformat(), days[-1].isoformat()
        busy = _ev.rae_busy_intervals(cx, lo, hi, practitioner="glen")
        booked = _ev.booked_starts(cx, practitioner="glen")
        slots = _ev.available_slots(days, GLEN_CONSULT_HOURS, busy, booked, _hst_now(),
                                    duration_min=_consult.CONSULT["duration_min"])
        return jsonify({"slots": slots})

@app.route("/api/consult/book", methods=["POST"])
def consult_book():
    from dashboard import evox as _ev, consult as _consult, zoom as _zoom
    from datetime import date
    body = request.get_json(force=True) or {}
    start_ts = (body.get("start_ts") or "").strip()
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _ev.init_evox_tables(cx); _consult.init_consult_tables(cx)
        ident = _consult_ident(cx, request.args.get("token", ""))
        if ident is None:
            return jsonify({"error": "not_found"}), 404
        if not _consult.consult_is_ready(cx, ident.email):
            return jsonify({"error": "not_ready"}), 403
        try:
            d = date.fromisoformat(start_ts[:10])
        except ValueError:
            return jsonify({"error": "bad_start_ts"}), 400
        busy = _ev.rae_busy_intervals(cx, d.isoformat(), d.isoformat(), practitioner="glen")
        if start_ts not in _ev.available_slots([d], GLEN_CONSULT_HOURS, busy,
                                               _ev.booked_starts(cx, practitioner="glen"),
                                               _hst_now(), duration_min=30):
            return jsonify({"error": "slot_unavailable"}), 409
        try:
            b = _ev.create_booking(cx, ident.email, start_ts, duration_min=30,
                                   practitioner="glen", session_type="biofield-consult",
                                   medium="video")
        except _ev.SlotTaken:
            return jsonify({"error": "slot_taken"}), 409
        # Zoom (best-effort; never blocks the booking)
        join_url = None
        try:
            tok = _zoom.get_token(os.environ["ZOOM_ACCOUNT_ID"], os.environ["ZOOM_CLIENT_ID"],
                                  os.environ["ZOOM_CLIENT_SECRET"])
            m = _zoom.create_meeting(tok, host=GLEN_ZOOM_USER, topic="Biofield Consult with Dr. Glen",
                                     start_iso=start_ts, duration_min=30)
            join_url = m.get("join_url")
            cx.execute("UPDATE evox_bookings SET zoom_join_url=?, zoom_meeting_id=? WHERE id=?",
                       (join_url, m.get("meeting_id"), b["id"]))
            cx.commit()
        except Exception:
            app.logger.exception("consult Zoom meeting create failed")
        b["join_url"] = join_url
    _consult_send_confirmations(ident.email, b)   # Task 6
    return jsonify({"ok": True, "start_ts": start_ts, "join_url": join_url})
```

Add a stub so this task's tests pass before Task 6 lands (Task 6 replaces it):

```python
def _consult_send_confirmations(email, booking):
    pass  # replaced in Task 6
```

- [ ] **Step 4: Run to verify pass**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/dc-test python3 -m pytest tests/test_consult_api.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_consult_api.py
git commit -m "feat(consult): state/availability/book routes with Zoom meeting"
```

---

### Task 6: Consult confirmation emails (client + Glen) with Zoom link

**Files:**
- Modify: `app.py` (replace the `_consult_send_confirmations` stub)
- Test: `tests/test_consult_api.py`

**Interfaces:**
- Consumes: `send_evox_email` (`app.py:478`), `dashboard.evox.build_ics`, `GLEN_ZOOM_USER`, a new `GLEN_CONSULT_EMAIL` global (default Glen's address).
- Produces: `_consult_send_confirmations(email, booking)` — builds a client email (with the Zoom **join link**, or "Zoom link to follow" if `booking.get("join_url")` is falsy) + a Glen-notification email, each with the same `.ics` (location = the join URL or "Video"). Best-effort per-send try/except; never raises. The whole body wrapped so a `build_ics` raise cannot 500 the booking.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_consult_api.py
def test_consult_confirmations_include_join_link(monkeypatch):
    calls = []
    monkeypatch.setattr(appmod, "send_evox_email",
                        lambda to, name, subj, html, text, ics: calls.append((to, html, ics)), raising=False)
    monkeypatch.setattr(appmod, "GLEN_CONSULT_EMAIL", "glen@illtowell.com", raising=False)
    appmod._consult_send_confirmations("c@x.com", {
        "id": 1, "email": "c@x.com", "start_ts": "2026-07-06T13:00:00",
        "end_ts": "2026-07-06T13:30:00", "ics_uid": "u1@illtowell.com",
        "join_url": "https://zoom.us/j/9", "session_type": "biofield-consult", "medium": "video"})
    assert len(calls) == 2
    assert any("zoom.us/j/9" in c[1] for c in calls)          # client email carries the link
    assert all(b"BEGIN:VCALENDAR" in c[2] for c in calls)
```

- [ ] **Step 2: Run to verify failure**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/dc-test python3 -m pytest tests/test_consult_api.py::test_consult_confirmations_include_join_link -q`
Expected: FAIL (the stub sends nothing → `len(calls) == 0`).

- [ ] **Step 3: Replace the stub in `app.py`**

```python
GLEN_CONSULT_EMAIL = os.environ.get("GLEN_CONSULT_EMAIL", "drglenswartwout@gmail.com")

def _consult_send_confirmations(email, booking):
    """Best-effort: client + Glen Biofield Consult confirmations with the Zoom link + ICS.
    Never raises into the booking response."""
    try:
        from dashboard import evox as _ev
        start = booking["start_ts"]; nice = start.replace("T", " ")
        join = booking.get("join_url")
        join_line = (f"Join your Zoom consult here: {join}" if join
                     else "Your Zoom link will follow by email shortly.")
        ics = _ev.build_ics(uid=booking["ics_uid"], start_ts=start, end_ts=booking["end_ts"],
                            summary="Biofield Consult with Dr. Glen",
                            description=join_line, location=(join or "Video"))
        client_html = (f"<p>Your Biofield Consult with Dr. Glen is booked for "
                       f"<b>{nice} HST</b>.</p><p>{join_line}</p>"
                       "<p>The calendar invite is attached.</p>")
        client_text = f"Biofield Consult booked for {nice} HST. {join_line}"
        glen_html = (f"<p>New Biofield Consult: <b>{email}</b> on <b>{nice} HST</b>.</p>"
                     f"<p>{join_line}</p>")
        for to, nm, subj, html, text in [
            (email, "", "Your Biofield Consult is booked", client_html, client_text),
            (GLEN_CONSULT_EMAIL, "Glen", f"Biofield Consult booked: {email}", glen_html, glen_html)]:
            try:
                send_evox_email(to, nm, subj, html, text, ics)
            except Exception:
                app.logger.exception("consult confirmation send failed to %s", to)
    except Exception:
        app.logger.exception("consult confirmation build failed")
```

- [ ] **Step 4: Run to verify pass**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/dc-test python3 -m pytest tests/test_consult_api.py -q`
Expected: PASS (all, including the full-flow test which now sends real confirmations through the monkeypatched sender).

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_consult_api.py
git commit -m "feat(consult): confirmation emails with Zoom link + ICS"
```

---

### Task 7: Portal integration — consult block + card + slot-picker JS

**Files:**
- Modify: `dashboard/portal_view.py` (`get_portal_view` ~line 174-208)
- Modify: `static/client-portal.html` (add a consult card before the `app.innerHTML` commit ~line 921; add slot-picker JS)
- Test: `tests/test_consult_api.py` (assert the `/view` payload carries the `consult` block)

**Interfaces:**
- Consumes: `dashboard.consult` (`consult_is_ready`, `has_paid_purchase`, `CONSULT`), `_is_paid_member` (passed in or imported), the consult routes from Task 5.
- Produces: `get_portal_view(...)` return dict gains `"consult": {"ready": bool, "booked": bool, "stages": {...}}`. `client-portal.html` renders a "Biofield Consult" card: when `v.consult.ready` shows a "Schedule your consult" button that fetches `/api/consult/availability` and books via `/api/consult/book`; otherwise shows the journey status. Copy: no em dashes, no ALL CAPS.

- [ ] **Step 1: Write the failing test (the `/view` block)**

```python
# append to tests/test_consult_api.py
def test_portal_view_carries_consult_block(client):
    tok = _mk_portal("pv@x.com")
    client.post("/api/console/consult-ready", json={"email": "pv@x.com", "ready": True}, headers=ADMIN)
    d = client.get(f"/api/portal/{tok}/view").get_json()
    assert "consult" in d and d["consult"]["ready"] is True
    assert "stages" in d["consult"]
```

- [ ] **Step 2: Run to verify failure**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/dc-test python3 -m pytest tests/test_consult_api.py::test_portal_view_carries_consult_block -q`
Expected: FAIL (`'consult' not in d`).

- [ ] **Step 3: Add the consult block to `get_portal_view`**

In `dashboard/portal_view.py`, add a helper and include it in the return dict:

```python
def _consult_block(cx, email):
    from dashboard import consult as _consult
    try:
        _consult.init_consult_tables(cx)
        ready = _consult.consult_is_ready(cx, email)
        paid = _consult.has_paid_purchase(cx, email, _consult.CONSULT["test_slug"])
        booked = False
        try:
            row = cx.execute("SELECT 1 FROM evox_bookings WHERE lower(email)=? "
                             "AND session_type='biofield-consult' AND status='booked' LIMIT 1",
                             (email,)).fetchone()
            booked = row is not None
        except Exception:
            pass
        return {"ready": ready, "booked": booked,
                "stages": {"test_paid": paid, "ready": ready}}
    except Exception:
        return {"ready": False, "booked": False, "stages": {}}
```

Add to the `get_portal_view` return dict: `"consult": _consult_block(cx, email),`.

- [ ] **Step 4: Run to verify pass**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/dc-test python3 -m pytest tests/test_consult_api.py -q`
Expected: PASS (all).

- [ ] **Step 5: Add the consult card + slot-picker JS to `static/client-portal.html`**

Before the single `app.innerHTML = html;` commit (~line 921), append a card gated on `v.consult`:

```javascript
if (v && v.consult) {
  const c = v.consult;
  if (c.booked) {
    html += `<div class="card"><h2>Biofield Consult</h2>
      <p>Your consult with Dr. Glen is booked. Check your email for the Zoom link and calendar invite.</p></div>`;
  } else if (c.ready) {
    html += `<div class="card"><h2>Biofield Consult</h2>
      <p>Your report is ready. Schedule your 30-minute consult with Dr. Glen to review your program.</p>
      <button id="consult-sched" onclick="consultSchedule()">Schedule your consult</button>
      <div id="consult-slots"></div></div>`;
  } else {
    html += `<div class="card"><h2>Biofield Consult</h2>
      <p>Your Biofield Consult with Dr. Glen unlocks once your Causal report and program are posted.
      We will let you know when it is ready to schedule.</p></div>`;
  }
}
```

Add the JS functions (near the other portal helpers; `seg` is the portal token already in scope, and `fetchJson` exists):

```javascript
async function consultSchedule(){
  const d = await fetchJson(`/api/consult/availability?token=${encodeURIComponent(seg)}&range=week`);
  const box = document.getElementById('consult-slots');
  box.innerHTML = (d.slots||[]).map(s =>
    `<button class="slot" onclick="consultBook('${s}')">${s.replace('T',' ')}</button>`).join('')
    || '<p>No open times this week. Please check back soon.</p>';
}
async function consultBook(start_ts){
  const r = await fetch(`/api/consult/book?token=${encodeURIComponent(seg)}`,
    {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({start_ts})});
  const d = await r.json();
  if(d.ok){ document.getElementById('consult-slots').innerHTML =
    '<p>Booked. Check your email for the Zoom link and calendar invite.</p>'; }
  else { alert('That time is no longer available. Please pick another.'); consultSchedule(); }
}
```

(Verify `seg` and `fetchJson` are the in-scope names in `render()` — they are per the load() at `static/client-portal.html:371`; if the token var differs, use the real one.)

- [ ] **Step 6: Verify the page still serves + JS parses**

Run: extract the `<script>` and `node --check` it (as the EVOX Task 9 did), and run the full consult API suite once more:
`doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/dc-test python3 -m pytest tests/test_consult_api.py tests/test_consult_pure.py -q`
Expected: PASS. (Full headless render-verify of the portal card is a go-live step.)

- [ ] **Step 7: Commit**

```bash
git add dashboard/portal_view.py static/client-portal.html tests/test_consult_api.py
git commit -m "feat(consult): portal consult card + slot-picker + /view block"
```

---

## Post-implementation (not code tasks)

- **Set env in Render** (per-key API, like EVOX): `GLEN_CONSULT_HOURS` (default is fine), `GLEN_ZOOM_USER` (Glen's Zoom user/email if not `me`), `GLEN_CONSULT_EMAIL` (Glen's notification address). Zoom creds already present.
- **Verify Zoom `meeting:write` scope** on the S2S app against prod before enabling (one real create).
- **Go-live smoke:** flip one test client "consult ready" → book a consult end to end → confirm the Zoom link + ICS arrive and the booking shows on Glen's console calendar lane → headless render-verify the portal consult card.

---

## Self-Review

**Spec coverage:**
- Generalize evox.py (session_type + medium) → Task 1. ✓
- Session-type catalog entry (biofield-consult) → Task 2 (`CONSULT`). ✓
- Manual "consult ready" gate + console flip → Tasks 2 (flag) + 4 (endpoint/button). ✓
- Zoom (dashboard/zoom.py, S2S, per-booking, failure-tolerant) → Tasks 3 + 5. ✓
- Glen availability (GLEN_CONSULT_HOURS all-days-9-5) + glen-lane busy → Task 5. ✓
- Confirmations with Zoom link → Task 6. ✓
- Portal surface (card + checklist status + slot picker) + /view block → Task 7. ✓
- No payment at booking → Tasks 5/6 (none charged). ✓
- Objective stage status (member, test_paid) → Tasks 5 (state) + 7 (view block). ✓
- Out of scope (intake form, post-consult video, auto-detect Causal report, hard prereq enforcement) → not built; noted. ✓

**Placeholder scan:** No TBD/TODO. The Task 5 `_consult_send_confirmations` stub is explicitly replaced in Task 6 (documented). The console email-input id + the portal `seg`/`fetchJson` names carry an explicit "verify against the page" instruction with the real line cited.

**Type consistency:** `create_booking(..., session_type=, medium=)` returns `session_type`/`medium` consumed unchanged by `_consult_send_confirmations`; `consult_is_ready`/`set_consult_ready`/`has_paid_purchase`/`CONSULT` names identical across Tasks 2/4/5/7; `create_meeting(...)->{join_url,meeting_id,start_url}` matches the Task 5 consumer; slot times naive ISO everywhere.
