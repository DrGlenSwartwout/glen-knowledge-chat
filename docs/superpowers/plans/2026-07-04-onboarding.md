# New-Member Onboarding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give a paid member a free, self-serve 15-minute phone welcome call with Rae, booked from their portal, once per member.

**Architecture:** Reuse the EVOX booking engine (`dashboard/evox.py`) with a new `session_type='onboarding'`. A tiny new module holds the onboarding config + the once-per-member lookup. New portal-token-gated routes add a `_is_paid_member` gate; the portal shows a card. No Zoom, no Stripe, no new env.

**Tech Stack:** Python 3 / Flask (single `app.py` + `dashboard/*.py` modules), sqlite (`chat_log.db`, `?` placeholders, `_db_lock` for writes, `cx.row_factory = sqlite3.Row`), vanilla JS/HTML in `static/`.

## Global Constraints

- Client-facing copy: **no em dashes, no ALL CAPS.** Warm, welcoming. Rae calls the member; say so plainly.
- Money: **none** — onboarding is a free member benefit (`amount=0`, no Stripe).
- DB writes go under `with _db_lock, sqlite3.connect(LOG_DB)`; email sends run **after** the lock is released (best-effort, never unwind a durable booking).
- `create_booking`'s signature must not change; EVOX/consult/triage/masterclass behavior must not change except the additive calendar-label map.
- Emails are lowercased before storage and lookup.
- Onboarding is **once per member**: a member with an existing booked onboarding gets no new slots and cannot re-book.

**Repo facts the implementer needs:**
- `dashboard/evox.py:create_booking(cx, email, start_ts, *, duration_min=60, prepaid=False, practitioner="rae", session_type="evox", medium="phone", tag_fn=None) -> dict` — inserts an `evox_bookings` row (status `'booked'`) + a synthetic `calendar_events` row, commits, returns `{"id","email","start_ts","end_ts","ics_uid","prepaid","session_type","medium"}`. Raises `evox.SlotTaken(start_ts)` on the partial-unique-index collision.
- `dashboard/evox.py`: `rae_busy_intervals(cx, lo, hi, practitioner="rae")`, `booked_starts(cx, practitioner="rae")`, `available_slots(days, hours_spec, busy, booked, now, duration_min=60) -> [iso_str,...]`, `build_ics(*, uid, start_ts, end_ts, summary, description, location, organizer_email="rae@illtowell.com") -> bytes`, `init_evox_tables(cx)`.
- `app.py` helpers: `_evox_ident(cx, token)` → identity object with `.email` (or `None`); `_hst_now()` → tz-aware HST `datetime`; `_evox_days(range_name)` → list of `date`; `_is_paid_member(email) -> bool`; `_init_calendar_table()`; `send_evox_email(to, name, subject, html, text, ics_bytes)`; module constants `EVOX_HOURS`, `EVOX_RAE_PHONE`, `PUBLIC_BASE_URL`; `_db_lock`, `LOG_DB`.
- `dashboard/portal_view.py:get_portal_view(cx, person_id, *, ...)` returns a dict; add a key. `dashboard/consult.py:_consult_block` and the consult routes are the pattern to mirror.

**Testing note (READ FIRST):** the test suite imports `app`, which opens `LOG_DB = DATA_DIR/chat_log.db` at import. Locally `DATA_DIR` from Doppler `prd` points at a prod path that does not exist, so you MUST override it. Run every test in this plan as:

```bash
export DATA_DIR="$(mktemp -d)"
doppler run -p remedy-match -c prd -- env DATA_DIR="$DATA_DIR" python3 -m pytest <paths> -q
```

Pure-sqlite tests that do NOT `import app` (Task 1) can run without Doppler:

```bash
python3 -m pytest tests/test_onboarding_pure.py -q
```

---

### Task 1: Onboarding module (config + once-per-member lookup)

**Files:**
- Create: `dashboard/onboarding.py`
- Test: `tests/test_onboarding_pure.py`

**Interfaces:**
- Consumes: nothing (pure sqlite; operates on the `evox_bookings` table that `evox.init_evox_tables` creates).
- Produces:
  - `ONBOARDING = {"session_type": "onboarding", "practitioner": "rae", "medium": "phone", "duration_min": 15}`
  - `existing_onboarding(cx, email) -> dict | None` — the member's currently booked onboarding row as a dict, or `None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_onboarding_pure.py
import sqlite3
from dashboard import evox as _ev
from dashboard import onboarding as _ob


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    _ev.init_evox_tables(cx)
    return cx


def test_config_values():
    assert _ob.ONBOARDING["session_type"] == "onboarding"
    assert _ob.ONBOARDING["practitioner"] == "rae"
    assert _ob.ONBOARDING["medium"] == "phone"
    assert _ob.ONBOARDING["duration_min"] == 15


def test_existing_none_when_no_booking():
    cx = _cx()
    assert _ob.existing_onboarding(cx, "a@b.com") is None


def test_existing_returns_booked_row():
    cx = _cx()
    _ev.create_booking(cx, "A@B.com", "2026-07-10T09:00:00", duration_min=15,
                       practitioner="rae", session_type="onboarding", medium="phone")
    row = _ob.existing_onboarding(cx, "a@b.com")
    assert row is not None
    assert row["start_ts"] == "2026-07-10T09:00:00"


def test_existing_ignores_other_session_types():
    cx = _cx()
    _ev.create_booking(cx, "a@b.com", "2026-07-10T09:00:00", session_type="evox")
    assert _ob.existing_onboarding(cx, "a@b.com") is None


def test_existing_ignores_cancelled():
    cx = _cx()
    _ev.create_booking(cx, "a@b.com", "2026-07-10T09:00:00", duration_min=15,
                       practitioner="rae", session_type="onboarding", medium="phone")
    cx.execute("UPDATE evox_bookings SET status='cancelled' WHERE lower(email)='a@b.com'")
    cx.commit()
    assert _ob.existing_onboarding(cx, "a@b.com") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_onboarding_pure.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'dashboard.onboarding'`

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/onboarding.py
"""New-member onboarding: a free 15-minute phone welcome call with Rae.

Reuses the EVOX booking engine (session_type='onboarding'). This module holds
only the onboarding config and the once-per-member lookup; the membership gate
(_is_paid_member) lives in the route layer so this module stays free of
app-layer imports (same shape as dashboard/consult.py)."""

ONBOARDING = {
    "session_type": "onboarding",
    "practitioner": "rae",
    "medium": "phone",
    "duration_min": 15,
}


def existing_onboarding(cx, email):
    """Return the member's currently booked onboarding row as a dict, or None.

    Booked-only (ignores cancelled rows) and onboarding-only. Used to enforce
    the once-per-member rule and to show the confirmed time on the portal card."""
    email = (email or "").strip().lower()
    row = cx.execute(
        "SELECT * FROM evox_bookings WHERE lower(email)=? "
        "AND session_type='onboarding' AND status='booked' "
        "ORDER BY start_ts DESC LIMIT 1", (email,)).fetchone()
    return dict(row) if row else None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_onboarding_pure.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add dashboard/onboarding.py tests/test_onboarding_pure.py
git commit -m "feat(onboarding): config + once-per-member lookup module"
```

---

### Task 2: Calendar label map in create_booking

**Files:**
- Modify: `dashboard/evox.py` (the `label = ...` line inside `create_booking`, currently `dashboard/evox.py:185`)
- Test: `tests/test_onboarding_label.py`

**Interfaces:**
- Consumes: `evox.create_booking` (existing).
- Produces: no signature change. `calendar_events.summary` / `calendar_name` now use a session-type → label map.

**Context:** `create_booking` currently sets `label = "Biofield Consult" if session_type == "biofield-consult" else "EVOX"`, so triage and onboarding both mislabel as "EVOX". Replace with a map so onboarding shows "Welcome Call" and triage shows "Discovery Call". EVOX and consult labels are unchanged (regression-protected below).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_onboarding_label.py
import sqlite3
from dashboard import evox as _ev


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    _ev.init_evox_tables(cx)
    _ev._ensure_calendar_events(cx) if hasattr(_ev, "_ensure_calendar_events") else None
    cx.execute("""CREATE TABLE IF NOT EXISTS calendar_events (
        id INTEGER PRIMARY KEY, pushed_at TEXT, google_cal_id TEXT, google_event_id TEXT,
        calendar_name TEXT, summary TEXT, start TEXT, end TEXT, location TEXT,
        owner TEXT, status TEXT, cal_alert INTEGER)""")
    return cx


def _label_for(cx, session_type, medium="phone", practitioner="rae"):
    _ev.create_booking(cx, "a@b.com", "2026-07-10T09:00:00", duration_min=15,
                       practitioner=practitioner, session_type=session_type, medium=medium)
    row = cx.execute("SELECT calendar_name, summary FROM calendar_events "
                     "ORDER BY id DESC LIMIT 1").fetchone()
    return row["calendar_name"], row["summary"]


def test_onboarding_label():
    name, summary = _label_for(_cx(), "onboarding")
    assert name == "Welcome Call booking"
    assert summary.startswith("Welcome Call — ")


def test_triage_label():
    name, summary = _label_for(_cx(), "triage")
    assert name == "Discovery Call booking"


def test_consult_label_unchanged():
    name, _ = _label_for(_cx(), "biofield-consult", medium="video", practitioner="glen")
    assert name == "Biofield Consult booking"


def test_evox_label_unchanged():
    name, _ = _label_for(_cx(), "evox")
    assert name == "EVOX booking"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_onboarding_label.py -q`
Expected: FAIL — `test_onboarding_label` and `test_triage_label` fail (both currently produce "EVOX booking").

- [ ] **Step 3: Write minimal implementation**

In `dashboard/evox.py`, inside `create_booking`, replace this line:

```python
    label = "Biofield Consult" if session_type == "biofield-consult" else "EVOX"
```

with:

```python
    label = {"biofield-consult": "Biofield Consult",
             "triage": "Discovery Call",
             "onboarding": "Welcome Call"}.get(session_type, "EVOX")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_onboarding_label.py tests/test_onboarding_pure.py -q`
Expected: PASS (4 + 5 = 9 passed)

- [ ] **Step 5: Commit**

```bash
git add dashboard/evox.py tests/test_onboarding_label.py
git commit -m "feat(onboarding): calendar-label map (Welcome Call / Discovery Call)"
```

---

### Task 3: Booking routes + confirmation + reminder copy

**Files:**
- Modify: `app.py` (add three routes + one confirmation helper near the consult routes ~`app.py:15473`; add an `onboarding` branch to `evox_run_reminders` ~`app.py:14379`)
- Test: `tests/test_onboarding_api.py`

**Interfaces:**
- Consumes: `dashboard/onboarding.py` (`ONBOARDING`, `existing_onboarding`), `evox.create_booking/available_slots/rae_busy_intervals/booked_starts/build_ics/SlotTaken`, `_evox_ident`, `_hst_now`, `_evox_days`, `_is_paid_member`, `_init_calendar_table`, `send_evox_email`, `EVOX_HOURS`, `EVOX_RAE_PHONE`, `PUBLIC_BASE_URL`, `_db_lock`, `LOG_DB`.
- Produces: `GET /api/onboarding/state`, `GET /api/onboarding/availability`, `POST /api/onboarding/book`, `_onboarding_send_confirmations(email, booking)`.

**Route contract:**
- `GET /api/onboarding/state?token=…` → 404 `{"error":"not_found"}` if token invalid; else `{"eligible": bool, "booked": {"start_ts": str} | null}`. `eligible = _is_paid_member(email)`.
- `GET /api/onboarding/availability?token=…&range=week` → 404 if bad token; 403 `{"error":"not_member"}` if not a member; if already booked → `{"slots": []}`; else `{"slots": [...]}` (Rae's `EVOX_HOURS`, 15-min).
- `POST /api/onboarding/book {start_ts}` (token in query) → 404 bad token; 403 `not_member`; 409 `already_booked` if `existing_onboarding`; 400 `bad_start_ts`; 409 `slot_unavailable`; 409 `slot_taken` on race; else `{"ok": true, "start_ts": str}` and a confirmation is sent.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_onboarding_api.py
import sqlite3
from unittest import mock
import app as appmod


def _client():
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()


def _seed_member(email):
    """Mint a REAL portal token so _evox_ident/resolve_identity resolves it.
    The client_portal token is sha256-hashed at rest, so faking the column will
    not resolve — use the real minting path and return the plaintext token."""
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        from dashboard import evox as _ev, client_portal as _cp
        _ev.init_evox_tables(cx)
        _cp.init_client_portal_table(cx)
        token = _ev.ensure_portal_token(cx, email, "")
        cx.commit()
    return token


def test_state_member_no_booking():
    c = _client()
    tok = _seed_member("m1@x.com")
    with mock.patch.object(appmod, "_is_paid_member", return_value=True):
        r = c.get(f"/api/onboarding/state?token={tok}")
    assert r.status_code == 200
    d = r.get_json()
    assert d["eligible"] is True
    assert d["booked"] is None


def test_state_non_member():
    c = _client()
    tok = _seed_member("m2@x.com")
    with mock.patch.object(appmod, "_is_paid_member", return_value=False):
        r = c.get(f"/api/onboarding/state?token={tok}")
    assert r.get_json()["eligible"] is False


def test_state_bad_token():
    c = _client()
    r = c.get("/api/onboarding/state?token=nope")
    assert r.status_code == 404


def test_availability_non_member_403():
    c = _client()
    tok = _seed_member("m3@x.com")
    with mock.patch.object(appmod, "_is_paid_member", return_value=False):
        r = c.get(f"/api/onboarding/availability?token={tok}")
    assert r.status_code == 403


def test_book_free_then_second_is_blocked():
    c = _client()
    tok = _seed_member("m4@x.com")
    with mock.patch.object(appmod, "_is_paid_member", return_value=True), \
         mock.patch.object(appmod, "send_evox_email") as send:
        avail = c.get(f"/api/onboarding/availability?token={tok}").get_json()["slots"]
        assert avail, "expected at least one Rae slot"
        slot = avail[0]
        r1 = c.post(f"/api/onboarding/book?token={tok}", json={"start_ts": slot})
        assert r1.status_code == 200 and r1.get_json()["ok"] is True
        assert send.called  # confirmation attempted
        # once-per-member: a second booking attempt is refused
        r2 = c.post(f"/api/onboarding/book?token={tok}", json={"start_ts": slot})
        assert r2.status_code == 409
        assert r2.get_json()["error"] == "already_booked"
        # and availability now returns no slots
        assert c.get(f"/api/onboarding/availability?token={tok}").get_json()["slots"] == []


def test_book_non_member_403():
    c = _client()
    tok = _seed_member("m5@x.com")
    with mock.patch.object(appmod, "_is_paid_member", return_value=False):
        r = c.post(f"/api/onboarding/book?token={tok}", json={"start_ts": "2026-07-10T09:00:00"})
    assert r.status_code == 403
```

- [ ] **Step 2: Run test to verify it fails**

Run: `export DATA_DIR="$(mktemp -d)" && doppler run -p remedy-match -c prd -- env DATA_DIR="$DATA_DIR" python3 -m pytest tests/test_onboarding_api.py -q`
Expected: FAIL — routes 404 (not registered yet).

- [ ] **Step 3: Write minimal implementation**

In `app.py`, immediately after `_consult_send_confirmations` (ends ~`app.py:15470`, before `@app.route("/api/consult/state")`), add the confirmation helper and the three routes:

```python
def _onboarding_send_confirmations(email, booking):
    """Best-effort welcome-call confirmation to the member + Rae, with an ICS
    invite. Phone call: Rae calls the member. Never raises into the booking
    response."""
    try:
        from dashboard import evox as _ev
        start = booking["start_ts"]; nice = start.replace("T", " ")
        phone = EVOX_RAE_PHONE or "the number on file"
        line = ("This is a phone call. Rae will call you at your appointment time at "
                "the number on file. Questions before then? Reach out any time.")
        ics = _ev.build_ics(uid=booking["ics_uid"], start_ts=start, end_ts=booking["end_ts"],
                            summary="New-member welcome call with Rae",
                            description=line, location="Phone")
        client_html = (f"<p>Welcome to Healing Oasis. Your welcome call with Rae is booked for "
                       f"<b>{nice} HST</b>.</p><p>{line}</p><p>The calendar invite is attached.</p>")
        client_text = f"Welcome call with Rae booked for {nice} HST. {line}"
        rae_html = (f"<p>New welcome call: <b>{email}</b> on <b>{nice} HST</b>. "
                    f"Please call them at the number on file.</p>")
        for to, nm, subj, html, text in [
            (email, "", "Your welcome call with Rae is booked", client_html, client_text),
            (EVOX_RAE_EMAIL, "Rae", f"Welcome call booked: {email}", rae_html, rae_html)]:
            try:
                send_evox_email(to, nm, subj, html, text, ics)
            except Exception:
                app.logger.exception("onboarding confirmation send failed to %s", to)
    except Exception:
        app.logger.exception("onboarding confirmation build failed")


@app.route("/api/onboarding/state")
def onboarding_state():
    from dashboard import onboarding as _ob
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        from dashboard import evox as _ev
        _ev.init_evox_tables(cx)
        ident = _evox_ident(cx, request.args.get("token", ""))
        if ident is None:
            return jsonify({"error": "not_found"}), 404
        row = _ob.existing_onboarding(cx, ident.email)
        booked = {"start_ts": row["start_ts"]} if row else None
        return jsonify({"eligible": _is_paid_member(ident.email), "booked": booked})


@app.route("/api/onboarding/availability")
def onboarding_availability():
    from dashboard import evox as _ev, onboarding as _ob
    _init_calendar_table()
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _ev.init_evox_tables(cx)
        ident = _evox_ident(cx, request.args.get("token", ""))
        if ident is None:
            return jsonify({"error": "not_found"}), 404
        if not _is_paid_member(ident.email):
            return jsonify({"error": "not_member"}), 403
        if _ob.existing_onboarding(cx, ident.email):
            return jsonify({"slots": []})
        days = _evox_days(request.args.get("range", "week"))
        lo, hi = days[0].isoformat(), days[-1].isoformat()
        busy = _ev.rae_busy_intervals(cx, lo, hi)
        booked = _ev.booked_starts(cx)
        slots = _ev.available_slots(days, EVOX_HOURS, busy, booked, _hst_now(),
                                    duration_min=_ob.ONBOARDING["duration_min"])
        return jsonify({"slots": slots})


@app.route("/api/onboarding/book", methods=["POST"])
def onboarding_book():
    from dashboard import evox as _ev, onboarding as _ob
    body = request.get_json(force=True) or {}
    start_ts = (body.get("start_ts") or "").strip()
    _init_calendar_table()
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _ev.init_evox_tables(cx)
        ident = _evox_ident(cx, request.args.get("token", ""))
        if ident is None:
            return jsonify({"error": "not_found"}), 404
        if not _is_paid_member(ident.email):
            return jsonify({"error": "not_member"}), 403
        if _ob.existing_onboarding(cx, ident.email):
            return jsonify({"error": "already_booked"}), 409
        try:
            d = _evox_date.fromisoformat(start_ts[:10])
        except ValueError:
            return jsonify({"error": "bad_start_ts"}), 400
        busy = _ev.rae_busy_intervals(cx, d.isoformat(), d.isoformat())
        if start_ts not in _ev.available_slots([d], EVOX_HOURS, busy,
                                               _ev.booked_starts(cx), _hst_now(),
                                               duration_min=_ob.ONBOARDING["duration_min"]):
            return jsonify({"error": "slot_unavailable"}), 409
        try:
            b = _ev.create_booking(cx, ident.email, start_ts, duration_min=15,
                                   practitioner="rae", session_type="onboarding",
                                   medium="phone")
        except _ev.SlotTaken:
            return jsonify({"error": "slot_taken"}), 409
        email = ident.email
    # --- lock released ---
    _onboarding_send_confirmations(email, b)
    return jsonify({"ok": True, "start_ts": start_ts})
```

Note: `_evox_date` is the module alias the consult/EVOX book routes already use for `datetime.date`; if a `NameError` appears, use `from datetime import date as _bk_date` inside the function and swap the name. Verify by grepping `_evox_date` in `app.py` first.

Then add the onboarding reminder branch. In `evox_run_reminders` (~`app.py:14379`), the current code is:

```python
            if stype == "biofield-consult":
                join_line = "Join from your Healing Oasis portal at your appointment time."
                subject = "Reminder: your Biofield Consult tomorrow"
                html = (f"<p>Reminder: your Biofield Consult with Dr. Glen is tomorrow at "
                        f"<b>{nice} HST</b>. {join_line}</p>")
            else:
```

Insert an `onboarding` branch between them:

```python
            if stype == "biofield-consult":
                join_line = "Join from your Healing Oasis portal at your appointment time."
                subject = "Reminder: your Biofield Consult tomorrow"
                html = (f"<p>Reminder: your Biofield Consult with Dr. Glen is tomorrow at "
                        f"<b>{nice} HST</b>. {join_line}</p>")
            elif stype == "onboarding":
                phone = EVOX_RAE_PHONE or "the number on file"
                subject = "Reminder: your welcome call with Rae tomorrow"
                html = (f"<p>Reminder: your welcome call with Rae is tomorrow at "
                        f"<b>{nice} HST</b>. This is a phone call. Rae will call you at "
                        f"the number on file.</p>")
            else:
```

- [ ] **Step 4: Run test to verify it passes**

Run: `export DATA_DIR="$(mktemp -d)" && doppler run -p remedy-match -c prd -- env DATA_DIR="$DATA_DIR" python3 -m pytest tests/test_onboarding_api.py -q`
Expected: PASS (7 passed). If `_seed_member` can't set the token column, first grep the `client_portal` schema (`grep -n "CREATE TABLE.*client_portal" dashboard/client_portal.py`) and adjust the seed to match the real column names.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_onboarding_api.py
git commit -m "feat(onboarding): booking routes, confirmation, reminder copy"
```

---

### Task 4: Portal card

**Files:**
- Modify: `dashboard/portal_view.py` (add `_onboarding_block`, wire into `get_portal_view`'s returned dict)
- Modify: `static/client-portal.html` (add the welcome-call card + 15-min slot picker, mirroring the consult card)
- Test: `tests/test_onboarding_block.py`

**Interfaces:**
- Consumes: `dashboard/onboarding.py:existing_onboarding`; the `/api/onboarding/{state,availability,book}` routes from Task 3.
- Produces: `get_portal_view(...)["onboarding"] = {"eligible": bool, "booked_start": str | None}`. Note `_onboarding_block` cannot see membership (no app import) — it reports `booked_start` and leaves `eligible` for the client JS, which calls `/api/onboarding/state`. Set `eligible` to `True` here as a render hint only when a booking exists; otherwise `False`. The card's real gate is the `state` fetch.

**Design note:** the portal payload is built without knowing `_is_paid_member` (that lives in `app.py`). So `_onboarding_block` returns only what it can compute from the DB — whether a booking exists. The card in the HTML calls `/api/onboarding/state` on load to decide whether to render for this member (same fetch-on-load pattern the consult card uses). This keeps `portal_view.py` free of app-layer imports.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_onboarding_block.py
import sqlite3
from dashboard import evox as _ev
from dashboard import portal_view as _pv


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    _ev.init_evox_tables(cx)
    return cx


def test_block_no_booking():
    cx = _cx()
    b = _pv._onboarding_block(cx, "a@b.com")
    assert b == {"eligible": False, "booked_start": None}


def test_block_with_booking():
    cx = _cx()
    _ev.create_booking(cx, "a@b.com", "2026-07-10T09:00:00", duration_min=15,
                       practitioner="rae", session_type="onboarding", medium="phone")
    b = _pv._onboarding_block(cx, "a@b.com")
    assert b["booked_start"] == "2026-07-10T09:00:00"
    assert b["eligible"] is True


def test_block_never_raises():
    # a broken connection must fall back safely, not raise
    b = _pv._onboarding_block(None, "a@b.com")
    assert b == {"eligible": False, "booked_start": None}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_onboarding_block.py -q`
Expected: FAIL — `_onboarding_block` does not exist.

- [ ] **Step 3: Write minimal implementation**

In `dashboard/portal_view.py`, add near `_consult_block`:

```python
def _onboarding_block(cx, email):
    """Whether the member has a booked new-member welcome call. Membership
    eligibility is decided by the client JS via /api/onboarding/state (this
    layer has no app import); here `eligible` is a render hint that is True only
    when a booking already exists. Defensive: any failure falls back to a safe
    not-eligible/not-booked default so it never breaks the portal payload."""
    from dashboard import onboarding as _ob
    try:
        row = _ob.existing_onboarding(cx, email)
        start = row["start_ts"] if row else None
        return {"eligible": start is not None, "booked_start": start}
    except Exception:
        return {"eligible": False, "booked_start": None}
```

Then wire it into `get_portal_view`'s returned dict, alongside the existing `"consult": _consult_block(cx, email)` entry:

```python
        "onboarding": _onboarding_block(cx, email),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_onboarding_block.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Add the portal card (HTML/JS)**

In `static/client-portal.html`, find the Biofield Consult card block (search for `consult` — it fetches `/api/consult/state`, shows a slot picker, posts to `/api/consult/book`). Add a sibling "Welcome call" card following the same pattern. The card must:

- On load, `fetch('/api/onboarding/state?token='+TOKEN)`. If `!d.eligible`, hide the card (`return`). This is the member gate.
- If `d.booked`, show: "Your welcome call with Rae is booked for &lt;d.booked.start_ts, T→space&gt; HST. This is a phone call, and Rae will call you at your appointment time." No picker.
- Else render a heading "Book your welcome call", one line of copy ("A free 15-minute phone call with Rae to welcome you and answer your first questions. Rae will call you."), then fetch `/api/onboarding/availability?token='+TOKEN`, render each slot as a button; on click, `POST /api/onboarding/book?token='+TOKEN` with `{start_ts}`. On `d.ok`, re-render into the booked state. Copy must contain no em dashes and no ALL CAPS.

Reuse whatever token variable and `fetch` helper the consult card already uses in this file (grep for how the consult card gets its token). Keep the markup and classes consistent with the consult card so styling is inherited.

- [ ] **Step 6: Verify the page still parses / no console errors**

Run the app locally or, at minimum, confirm the HTML has balanced tags for the new block and the JS has no syntax error:

```bash
node --check <(sed -n '/BEGIN onboarding card script/,/END onboarding card script/p' static/client-portal.html) 2>/dev/null || echo "wrap the new JS in clearly-marked BEGIN/END comments and re-check"
```

(If a Playwright/headless render harness exists in this repo — grep `tests/` for `playwright` — prefer rendering `/portal/<token>` and asserting 0 console errors, per the project's render-verify rule.)

- [ ] **Step 7: Commit**

```bash
git add dashboard/portal_view.py static/client-portal.html tests/test_onboarding_block.py
git commit -m "feat(onboarding): portal welcome-call card + view block"
```

---

## Definition of Done

- `dashboard/onboarding.py` + the three routes + the portal card ship a member-gated, once-per-member, free 15-minute phone welcome call booked on Rae's availability.
- Calendar rows for onboarding read "Welcome Call" and triage rows read "Discovery Call"; EVOX and consult labels unchanged.
- The existing `glen-evox-reminders` cron sends onboarding-specific reminder copy.
- All new tests pass; EVOX/consult/triage/masterclass are untouched except the additive label map.
- No new env; no Stripe; no Zoom.

## Deferred (not in this plan)

- **Push discovery** — auto-emailing a "book your welcome call" invite when someone becomes a paid member (pull-first was chosen).
- Reschedule/cancel from the portal, post-call notes, onboarding checklist/content.
- Triage reminder copy still falls into the EVOX `else` branch (its own pre-existing fast-follow); this plan only touches the onboarding branch.
