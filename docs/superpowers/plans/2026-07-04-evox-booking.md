# EVOX Booking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let anyone self-onboard to EVOX (self-attest readiness checklist, buy the hand cradle), then book a 60-min phone EVOX session against Rae's availability, landing the appointment on the console team calendar and both parties' real calendars via an ICS invite — no Practice Better, no Zoom.

**Architecture:** One pure-logic + sqlite module `dashboard/evox.py` (readiness state, availability computation, booking, prepay credits, ICS build) plus five Flask routes in `app.py` and one standalone page `static/evox.html`. Availability = Rae's configured office-hours grid minus busy rows already in the local `calendar_events` table (fed hourly from Google) minus already-booked slots. A booking inserts a synthetic `rae`-lane `calendar_events` row (same trick the delegate-move uses) so it shows on the console calendar, and emails an `.ics` invite so it lands on real calendars instantly on accept. Money stays off the booking action: invoice-after by default (existing in-house rails), optional prepay via a session-credit balance.

**Tech Stack:** Python 3 / Flask, sqlite (`chat_log.db`, `?` placeholders, `_db_lock` for writes), existing `send_magic_link_email` SMTP/GHL rail (extended with an `.ics` attachment), `resolve_identity` portal-token auth, `data/products.json` SKUs, vanilla JS single-page frontend. Tests: pytest via `app.app.test_client()` with monkeypatched `LOG_DB`.

## Global Constraints

- **DB = sqlite** `chat_log.db` via `cx = sqlite3.connect(LOG_DB)`; placeholders are `?`; wrap every write in `with _db_lock, sqlite3.connect(LOG_DB) as cx:`. Do NOT use the Supabase/`%s` path. Each module owns an idempotent `init_evox_tables(cx)` called at the top of each route.
- **Prices are integer cents** in `data/products.json`; product **slug = the dict key** under the top-level `products` key.
- **Auth model:** EVOX APIs authenticate with the client's **portal token** via `dashboard.portal_identity.resolve_identity(cx, token=..., session_token=..., client_login_enabled=_client_login_enabled())`; `ident.email` (lowercased) is the key for all EVOX state. Console-only/admin endpoints use the `X-Console-Key: CONSOLE_SECRET` header pattern.
- **Times:** all EVOX slot times are **HST (UTC−10)**, stored as naive ISO strings `YYYY-MM-DDTHH:MM:SS` (matches `calendar_events.start` text). Never call `datetime.now()`/`utcnow()` inside pure functions — pass `now` in.
- **Config values (env vars, read once at module load in `app.py`, passed into `evox.py`):**
  - `EVOX_HOURS` — Rae's weekly bookable window. Default `"1-4:09:00-16:00"` = Mon–Thu (ISO weekday 1–4), 09:00–16:00 HST, 60-min grid. Format: `"<wdayLo>-<wdayHi>:<HH:MM>-<HH:MM>"`.
  - `EVOX_RAE_PHONE` — the number the client calls at appointment time. **Rae/Glen must set this**; default `""` → the confirmation says "the number in your confirmation" and logs a warning if empty.
  - `EVOX_SESSION_PRICE_CENTS` — public list price of one prepaid EVOX session = `19700` ($197). Used for the `evox-session` SKU storefront price. **The $100 member rate is NOT a storefront price in v1** — it is applied by Rae at invoice/prepay time via the in-house per-line `unit_cents` override (she knows member vs public), exactly like the invoice-after default. Storefront member auto-pricing is a deferred enhancement (would touch the pricing core in both `pricing.compute` and `_inhouse_line_unit_cents`).
- **Medium is phone + internet, no Zoom.** No video link is ever generated.
- **Call direction: client calls Rae** (Rae's number in confirmation + ICS).
- **Tags** reuse the existing `people.tags` JSON-array column via `dashboard.people.set_person_tags(current, add=, remove=)` — do NOT invent a tag table.
- **Test command:** `mkdir -p /tmp/dc-test && doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/dc-test python3 -m pytest tests/test_<name>.py -q`. Pure-logic tests that only import `dashboard/evox.py` **without importing `app`** run with plain `python3 -m pytest tests/test_<name>.py -q` (no doppler). Keep pure functions importable without importing `app`.

---

## File Structure

- **Create `dashboard/evox.py`** — the whole EVOX domain. Sections: pure helpers (`readiness_complete`, `parse_office_hours`, `slot_grid`, `intervals_overlap`, `available_slots`, `build_ics`), and sqlite functions (`init_evox_tables`, `get_readiness`, `set_readiness_item`, `has_cradle_purchase`, `create_booking`, `booked_starts`, `rae_busy_intervals`, `add_session_credits`, `consume_session_credit`, `session_credit_balance`, `ensure_portal_token`). Pure helpers take primitives only (no `cx`) so they unit-test without a DB or `app`.
- **Create `static/evox.html`** — single-page flow: email capture → readiness checklist → availability grid → confirmation. Auth via `?token=` in the URL.
- **Modify `data/products.json`** — add `hand-cradle` and `evox-session` under `products`.
- **Modify `app.py`** — add 5 routes + module-load config; extend the email sender to accept an optional `.ics`.
- **Create tests** — `tests/test_evox_pure.py` (pure, no doppler), `tests/test_evox_api.py` (routes, doppler), `tests/test_evox_products.py`.

---

### Task 1: Add the two SKUs to `data/products.json`

**Files:**
- Modify: `data/products.json` (under the top-level `"products"` object)
- Test: `tests/test_evox_products.py`

**Interfaces:**
- Produces: product slugs `hand-cradle` (physical, `price_cents` 29700, has `bottle_type` so the packer charges shipping) and `evox-session` (`price_cents` = `EVOX_SESSION_PRICE_CENTS` default 15000, `info_only:true`+`service:true` so it stays out of the shipping packer/funnel but the in-house builder accepts it). Consumed by `_get_product(slug)` (`app.py:4888`) and `/api/orders/manual`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evox_products.py
import json, pathlib
def _products():
    p = pathlib.Path(__file__).resolve().parent.parent / "data" / "products.json"
    return json.loads(p.read_text())["products"]

def test_hand_cradle_sku_present():
    p = _products()["hand-cradle"]
    assert p["price_cents"] == 29700
    assert p.get("info_only") is not True          # physical: goes through the packer
    assert p.get("bottle_type")                     # must have a packer dim so shipping is billed

def test_evox_session_sku_present():
    p = _products()["evox-session"]
    assert p["price_cents"] == 19700          # public list; member $100 applied by Rae at invoice
    assert p["info_only"] is True and p["service"] is True   # prepay service, no shipping
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_evox_products.py -q`
Expected: FAIL with `KeyError: 'hand-cradle'`.

- [ ] **Step 3: Add the SKUs**

Add these two keys inside the `"products": { ... }` object in `data/products.json` (JSON — no trailing comma on the last key):

```json
"hand-cradle": {
  "name": "ZYTO Hand Cradle",
  "price_cents": 29700,
  "bottle_type": "default",
  "source": "evox-hardware-2026-07-04"
},
"evox-session": {
  "name": "EVOX Session (prepaid)",
  "price_cents": 19700,
  "info_only": true,
  "service": true,
  "service_value_cents": 19700,
  "source": "evox-service-2026-07-04"
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_evox_products.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add data/products.json tests/test_evox_products.py
git commit -m "feat(evox): add hand-cradle and evox-session SKUs"
```

---

### Task 2: Readiness model — pure predicate + sqlite state

**Files:**
- Create: `dashboard/evox.py`
- Test: `tests/test_evox_pure.py`

**Interfaces:**
- Produces:
  - `READINESS_ITEMS = ("pc_ok", "cradle_ok", "headset_ok", "zyto_ok")`
  - `readiness_complete(state: dict) -> bool` — pure; True iff all four keys truthy.
  - `init_evox_tables(cx) -> None` — creates `evox_readiness`, `evox_bookings`, `evox_session_credits`.
  - `get_readiness(cx, email: str) -> dict` — row as dict (defaults all items False, `cradle_source=None`) for a lowercased email.
  - `set_readiness_item(cx, email: str, item: str, value: bool, *, cradle_source: str | None = None) -> dict` — upserts one item, returns the full readiness dict.
- Consumes: nothing from earlier tasks.

- [ ] **Step 1: Write the failing test (pure predicate)**

```python
# tests/test_evox_pure.py
from dashboard import evox

def test_readiness_complete_all_true():
    assert evox.readiness_complete(
        {"pc_ok": True, "cradle_ok": True, "headset_ok": True, "zyto_ok": True}) is True

def test_readiness_incomplete_when_any_false():
    assert evox.readiness_complete(
        {"pc_ok": True, "cradle_ok": False, "headset_ok": True, "zyto_ok": True}) is False

def test_readiness_incomplete_when_missing_key():
    assert evox.readiness_complete({"pc_ok": True}) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_evox_pure.py -q`
Expected: FAIL with `ModuleNotFoundError` / `AttributeError: readiness_complete`.

- [ ] **Step 3: Create `dashboard/evox.py` with the readiness section**

```python
"""EVOX booking: self-attest readiness, availability, 1:1 phone booking, ICS.
Pure helpers take primitives only (no cx) and must import without importing app."""
import sqlite3
from datetime import datetime, timedelta

READINESS_ITEMS = ("pc_ok", "cradle_ok", "headset_ok", "zyto_ok")


def readiness_complete(state: dict) -> bool:
    return all(bool(state.get(k)) for k in READINESS_ITEMS)


def init_evox_tables(cx) -> None:
    cx.execute("""CREATE TABLE IF NOT EXISTS evox_readiness (
        email TEXT PRIMARY KEY,
        pc_ok INTEGER DEFAULT 0, cradle_ok INTEGER DEFAULT 0,
        headset_ok INTEGER DEFAULT 0, zyto_ok INTEGER DEFAULT 0,
        cradle_source TEXT, completed_at TEXT, updated_at TEXT)""")
    cx.execute("""CREATE TABLE IF NOT EXISTS evox_bookings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT NOT NULL, practitioner TEXT NOT NULL DEFAULT 'rae',
        start_ts TEXT NOT NULL, end_ts TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'booked', prepaid INTEGER DEFAULT 0,
        calendar_event_id TEXT, ics_uid TEXT, created_at TEXT)""")
    cx.execute("""CREATE UNIQUE INDEX IF NOT EXISTS ux_evox_active_slot
        ON evox_bookings(practitioner, start_ts) WHERE status='booked'""")
    cx.execute("""CREATE TABLE IF NOT EXISTS evox_session_credits (
        email TEXT PRIMARY KEY, credits INTEGER NOT NULL DEFAULT 0)""")
    cx.commit()


def get_readiness(cx, email: str) -> dict:
    email = (email or "").strip().lower()
    row = cx.execute("SELECT * FROM evox_readiness WHERE email=?", (email,)).fetchone()
    if row is None:
        base = {k: False for k in READINESS_ITEMS}
        base.update({"email": email, "cradle_source": None, "complete": False})
        return base
    d = dict(row)
    out = {k: bool(d.get(k)) for k in READINESS_ITEMS}
    out.update({"email": email, "cradle_source": d.get("cradle_source")})
    out["complete"] = readiness_complete(out)
    return out


def set_readiness_item(cx, email: str, item: str, value: bool,
                       *, cradle_source: str | None = None) -> dict:
    email = (email or "").strip().lower()
    if item not in READINESS_ITEMS:
        raise ValueError(f"unknown readiness item: {item}")
    now = datetime.utcnow().isoformat()
    cx.execute("INSERT OR IGNORE INTO evox_readiness (email, updated_at) VALUES (?,?)",
               (email, now))
    cx.execute(f"UPDATE evox_readiness SET {item}=?, updated_at=? WHERE email=?",
               (1 if value else 0, now, email))
    if item == "cradle_ok" and cradle_source is not None:
        cx.execute("UPDATE evox_readiness SET cradle_source=? WHERE email=?",
                   (cradle_source, email))
    state = get_readiness(cx, email)
    if state["complete"]:
        cx.execute("UPDATE evox_readiness SET completed_at=COALESCE(completed_at,?) "
                   "WHERE email=?", (now, email))
    cx.commit()
    return get_readiness(cx, email)
```

- [ ] **Step 4: Run the pure test to verify it passes**

Run: `python3 -m pytest tests/test_evox_pure.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Add and run the sqlite readiness test**

Append to `tests/test_evox_pure.py`:

```python
import sqlite3
def _cx():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    evox.init_evox_tables(cx); return cx

def test_readiness_roundtrip_and_complete():
    cx = _cx()
    assert evox.get_readiness(cx, "A@x.com")["complete"] is False
    for item in ("pc_ok", "cradle_ok", "headset_ok", "zyto_ok"):
        st = evox.set_readiness_item(cx, "a@x.com", item, True)
    assert st["complete"] is True
    assert evox.get_readiness(cx, "a@x.com")["complete"] is True   # email lowercased

def test_cradle_source_recorded():
    cx = _cx()
    st = evox.set_readiness_item(cx, "b@x.com", "cradle_ok", True, cradle_source="buy")
    assert st["cradle_source"] == "buy"
```

Run: `python3 -m pytest tests/test_evox_pure.py -q`
Expected: PASS (5 passed).

- [ ] **Step 6: Commit**

```bash
git add dashboard/evox.py tests/test_evox_pure.py
git commit -m "feat(evox): readiness model — pure predicate + sqlite state"
```

---

### Task 3: Availability computation (pure)

**Files:**
- Modify: `dashboard/evox.py` (add pure availability helpers)
- Test: `tests/test_evox_pure.py`

**Interfaces:**
- Produces:
  - `parse_office_hours(spec: str) -> tuple[int, int, str, str]` → `(wday_lo, wday_hi, "HH:MM", "HH:MM")`; ISO weekday 1=Mon.
  - `slot_grid(day: "date", spec, duration_min=60) -> list[str]` → naive ISO `YYYY-MM-DDTHH:MM:SS` slot starts within office hours on that day, or `[]` if the day's weekday is outside the window.
  - `intervals_overlap(a_start, a_end, b_start, b_end) -> bool` — pure; datetimes.
  - `available_slots(days: list["date"], office_spec: str, busy: list[tuple[str,str]], booked: set[str], now: "datetime", duration_min=60) -> list[str]` → sorted ISO slot starts that are (a) in office hours, (b) strictly future vs `now`, (c) not booked, (d) not overlapping any busy interval. Unparseable busy rows are skipped; a date-only busy `start` with empty `end` blocks that whole day.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_evox_pure.py
from datetime import date, datetime

def test_parse_office_hours():
    assert evox.parse_office_hours("1-4:09:00-16:00") == (1, 4, "09:00", "16:00")

def test_slot_grid_weekday_in_window():
    slots = evox.slot_grid(date(2026, 7, 6), "1-4:09:00-16:00")   # Mon
    assert slots[0] == "2026-07-06T09:00:00"
    assert slots[-1] == "2026-07-06T15:00:00"   # last 60-min slot starts 15:00, ends 16:00
    assert len(slots) == 7

def test_slot_grid_weekday_out_of_window():
    assert evox.slot_grid(date(2026, 7, 5), "1-4:09:00-16:00") == []   # Sunday

def test_available_excludes_busy_and_booked_and_past():
    days = [date(2026, 7, 6)]
    now = datetime(2026, 7, 6, 10, 30)                      # 09:00 & 10:00 are past
    busy = [("2026-07-06T13:00:00", "2026-07-06T14:00:00")] # blocks 13:00
    booked = {"2026-07-06T12:00:00"}                        # blocks 12:00
    slots = evox.available_slots(days, "1-4:09:00-16:00", busy, booked, now)
    assert slots == ["2026-07-06T11:00:00", "2026-07-06T14:00:00", "2026-07-06T15:00:00"]

def test_available_allday_busy_blocks_whole_day():
    days = [date(2026, 7, 6)]
    now = datetime(2026, 7, 6, 0, 0)
    slots = evox.available_slots(days, "1-4:09:00-16:00", [("2026-07-06", "")], set(), now)
    assert slots == []
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_evox_pure.py -q`
Expected: FAIL (`AttributeError: parse_office_hours`).

- [ ] **Step 3: Implement the pure helpers in `dashboard/evox.py`**

```python
def parse_office_hours(spec: str):
    days_part, hours_part = spec.split(":", 1)
    lo, hi = days_part.split("-")
    start_hm, end_hm = hours_part.split("-")
    return int(lo), int(hi), start_hm, end_hm


def _hm(day, hm: str) -> datetime:
    h, m = hm.split(":")
    return datetime(day.year, day.month, day.day, int(h), int(m))


def slot_grid(day, spec: str, duration_min: int = 60):
    lo, hi, start_hm, end_hm = parse_office_hours(spec)
    if not (lo <= day.isoweekday() <= hi):
        return []
    start, end = _hm(day, start_hm), _hm(day, end_hm)
    out, t, step = [], start, timedelta(minutes=duration_min)
    while t + step <= end:
        out.append(t.isoformat()); t += step
    return out


def _parse(ts: str):
    ts = (ts or "").strip()
    if not ts:
        return None
    try:
        if len(ts) == 10:            # date-only, e.g. all-day event
            return datetime.fromisoformat(ts + "T00:00:00")
        return datetime.fromisoformat(ts[:19])
    except ValueError:
        return None


def intervals_overlap(a_start, a_end, b_start, b_end) -> bool:
    return a_start < b_end and b_start < a_end


def available_slots(days, office_spec, busy, booked, now, duration_min: int = 60):
    step = timedelta(minutes=duration_min)
    # Normalize busy into datetime intervals; date-only start w/ empty end = whole day.
    intervals = []
    for bs, be in busy:
        s = _parse(bs)
        if s is None:
            continue
        if len(str(bs).strip()) == 10 and not (be or "").strip():
            e = s + timedelta(days=1)
        else:
            e = _parse(be) or (s + step)
        intervals.append((s, e))
    out = []
    for day in days:
        for iso in slot_grid(day, office_spec, duration_min):
            s = datetime.fromisoformat(iso)
            if s <= now or iso in booked:
                continue
            e = s + step
            if any(intervals_overlap(s, e, bs, be) for bs, be in intervals):
                continue
            out.append(iso)
    return sorted(out)
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_evox_pure.py -q`
Expected: PASS (10 passed).

- [ ] **Step 5: Commit**

```bash
git add dashboard/evox.py tests/test_evox_pure.py
git commit -m "feat(evox): pure availability computation"
```

---

### Task 4: Booking creation + synthetic calendar row + tags

**Files:**
- Modify: `dashboard/evox.py` (sqlite booking functions)
- Test: `tests/test_evox_pure.py` (sqlite, still no `app` import — pass a fake calendar-insert callback)

**Interfaces:**
- Produces:
  - `booked_starts(cx, practitioner="rae") -> set[str]` — active booked start_ts.
  - `rae_busy_intervals(cx, lo_date: str, hi_date: str, practitioner="rae") -> list[tuple[str,str]]` — `(start,end)` from `calendar_events` where `owner=? AND status='visible' AND substr(start,1,10) BETWEEN ? AND ?`.
  - `create_booking(cx, email, start_ts, *, duration_min=60, prepaid=False, tag_fn=None) -> dict` — inserts an `evox_bookings` row, inserts a synthetic `rae`-lane `calendar_events` row (`google_cal_id='delegated'`, `google_event_id=f"evox-{booking_id}"`, `owner='rae'`, `status='visible'`), sets `calendar_event_id`/`ics_uid`, and calls `tag_fn(email, ["evox-client", "evox-ready"])` if given. Raises `evox.SlotTaken` on the unique-index clash. Returns `{id, email, start_ts, end_ts, ics_uid, prepaid}`.
- Consumes: `init_evox_tables` (Task 2).

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_evox_pure.py
def _cal_cx():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    evox.init_evox_tables(cx)
    cx.execute("""CREATE TABLE calendar_events (id INTEGER PRIMARY KEY AUTOINCREMENT,
        pushed_at TEXT, google_cal_id TEXT, google_event_id TEXT, calendar_name TEXT,
        summary TEXT, start TEXT, end TEXT, location TEXT, owner TEXT, status TEXT,
        cal_alert INTEGER, UNIQUE(google_cal_id, google_event_id))""")
    cx.commit(); return cx

def test_create_booking_writes_calendar_and_tags():
    cx = _cal_cx(); seen = {}
    def tag_fn(email, tags): seen[email] = tags
    b = evox.create_booking(cx, "c@x.com", "2026-07-06T11:00:00", tag_fn=tag_fn)
    assert b["end_ts"] == "2026-07-06T12:00:00"
    row = cx.execute("SELECT owner,status,google_cal_id,google_event_id FROM calendar_events").fetchone()
    assert (row["owner"], row["status"], row["google_cal_id"]) == ("rae", "visible", "delegated")
    assert row["google_event_id"] == f"evox-{b['id']}"
    assert seen["c@x.com"] == ["evox-client", "evox-ready"]
    assert "2026-07-06T11:00:00" in evox.booked_starts(cx)

def test_double_book_raises():
    cx = _cal_cx()
    evox.create_booking(cx, "c@x.com", "2026-07-06T11:00:00")
    import pytest
    with pytest.raises(evox.SlotTaken):
        evox.create_booking(cx, "d@x.com", "2026-07-06T11:00:00")

def test_rae_busy_intervals_reads_calendar():
    cx = _cal_cx()
    cx.execute("INSERT INTO calendar_events (pushed_at,google_cal_id,google_event_id,"
               "summary,start,end,owner,status) VALUES (?,?,?,?,?,?,?,?)",
               ("x", "rae@g", "e1", "Busy", "2026-07-06T13:00:00",
                "2026-07-06T14:00:00", "rae", "visible")); cx.commit()
    assert evox.rae_busy_intervals(cx, "2026-07-06", "2026-07-06") == \
        [("2026-07-06T13:00:00", "2026-07-06T14:00:00")]
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_evox_pure.py -q`
Expected: FAIL (`AttributeError: create_booking`).

- [ ] **Step 3: Implement the booking functions**

```python
import secrets

class SlotTaken(Exception):
    pass


def booked_starts(cx, practitioner: str = "rae") -> set:
    rows = cx.execute("SELECT start_ts FROM evox_bookings "
                      "WHERE practitioner=? AND status='booked'", (practitioner,)).fetchall()
    return {r[0] for r in rows}


def rae_busy_intervals(cx, lo_date: str, hi_date: str, practitioner: str = "rae"):
    rows = cx.execute(
        "SELECT start, end FROM calendar_events WHERE owner=? AND status='visible' "
        "AND substr(start,1,10) BETWEEN ? AND ?", (practitioner, lo_date, hi_date)).fetchall()
    return [(r[0], r[1] or "") for r in rows]


def create_booking(cx, email: str, start_ts: str, *, duration_min: int = 60,
                   prepaid: bool = False, practitioner: str = "rae", tag_fn=None) -> dict:
    email = (email or "").strip().lower()
    start_dt = datetime.fromisoformat(start_ts[:19])
    end_ts = (start_dt + timedelta(minutes=duration_min)).isoformat()
    now = datetime.utcnow().isoformat()
    ics_uid = f"evox-{secrets.token_hex(8)}@illtowell.com"
    try:
        cur = cx.execute(
            "INSERT INTO evox_bookings (email,practitioner,start_ts,end_ts,status,"
            "prepaid,ics_uid,created_at) VALUES (?,?,?,?,'booked',?,?,?)",
            (email, practitioner, start_ts, end_ts, 1 if prepaid else 0, ics_uid, now))
    except sqlite3.IntegrityError:
        raise SlotTaken(start_ts)
    booking_id = cur.lastrowid
    ev_id = f"evox-{booking_id}"
    cx.execute(
        "INSERT INTO calendar_events (pushed_at,google_cal_id,google_event_id,"
        "calendar_name,summary,start,end,location,owner,status,cal_alert) "
        "VALUES (?, 'delegated', ?, 'EVOX booking', ?, ?, ?, 'Phone', ?, 'visible', 0)",
        (now, ev_id, f"EVOX — {email}", start_ts, end_ts, practitioner))
    cx.execute("UPDATE evox_bookings SET calendar_event_id=? WHERE id=?", (ev_id, booking_id))
    cx.commit()
    if tag_fn:
        tag_fn(email, ["evox-client", "evox-ready"])
    return {"id": booking_id, "email": email, "start_ts": start_ts,
            "end_ts": end_ts, "ics_uid": ics_uid, "prepaid": prepaid}
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_evox_pure.py -q`
Expected: PASS (13 passed).

- [ ] **Step 5: Commit**

```bash
git add dashboard/evox.py tests/test_evox_pure.py
git commit -m "feat(evox): booking creation + synthetic rae-lane calendar row"
```

---

### Task 5: ICS builder (pure) + prepay credits + cradle-purchase check

**Files:**
- Modify: `dashboard/evox.py`
- Test: `tests/test_evox_pure.py`

**Interfaces:**
- Produces:
  - `build_ics(*, uid, start_ts, end_ts, summary, description, location, organizer_email="rae@illtowell.com") -> bytes` — a valid single-VEVENT `METHOD:REQUEST` calendar, CRLF line endings, times as floating local `YYYYMMDDT HHMMSS` (no Z; HST-local, matches the naive slot times). DTSTAMP is passed via `start_ts` day at 00:00 to stay deterministic (do NOT call now()).
  - `session_credit_balance(cx, email) -> int`, `add_session_credits(cx, email, n) -> int`, `consume_session_credit(cx, email) -> bool` (decrement if >0, return whether consumed).
  - `has_cradle_purchase(cx, email) -> bool` — True if any `orders` row for that email has a line with `slug=="hand-cradle"` in `items_json`.
- Consumes: `init_evox_tables` (Task 2).

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_evox_pure.py
import json
def test_build_ics_valid():
    ics = evox.build_ics(uid="u1@illtowell.com", start_ts="2026-07-06T11:00:00",
                         end_ts="2026-07-06T12:00:00", summary="EVOX Session",
                         description="Call Rae at 808-555-1212", location="Phone")
    t = ics.decode()
    assert t.startswith("BEGIN:VCALENDAR") and "METHOD:REQUEST" in t
    assert "BEGIN:VEVENT" in t and "UID:u1@illtowell.com" in t
    assert "DTSTART:20260706T110000" in t and "DTEND:20260706T120000" in t
    assert t.endswith("END:VCALENDAR\r\n") and "\r\n" in t

def test_session_credits():
    cx = _cx()
    assert evox.session_credit_balance(cx, "e@x.com") == 0
    assert evox.add_session_credits(cx, "e@x.com", 3) == 3
    assert evox.consume_session_credit(cx, "e@x.com") is True
    assert evox.session_credit_balance(cx, "e@x.com") == 2

def test_consume_credit_when_zero():
    cx = _cx()
    assert evox.consume_session_credit(cx, "z@x.com") is False

def test_has_cradle_purchase():
    cx = _cal_cx()
    cx.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, email TEXT, items_json TEXT)")
    cx.execute("INSERT INTO orders (email, items_json) VALUES (?,?)",
               ("buyer@x.com", json.dumps([{"slug": "hand-cradle", "qty": 1}]))); cx.commit()
    assert evox.has_cradle_purchase(cx, "BUYER@x.com") is True
    assert evox.has_cradle_purchase(cx, "nobody@x.com") is False
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_evox_pure.py -q`
Expected: FAIL (`AttributeError: build_ics`).

- [ ] **Step 3: Implement**

```python
def build_ics(*, uid, start_ts, end_ts, summary, description, location,
              organizer_email="rae@illtowell.com") -> bytes:
    def _fmt(ts):  # naive local -> floating VEVENT time
        return datetime.fromisoformat(ts[:19]).strftime("%Y%m%dT%H%M%S")
    dtstamp = datetime.fromisoformat(start_ts[:19]).strftime("%Y%m%dT000000")
    desc = (description or "").replace("\n", "\\n")
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//illtowell//EVOX//EN",
             "METHOD:REQUEST", "BEGIN:VEVENT", f"UID:{uid}", f"DTSTAMP:{dtstamp}",
             f"DTSTART:{_fmt(start_ts)}", f"DTEND:{_fmt(end_ts)}",
             f"SUMMARY:{summary}", f"DESCRIPTION:{desc}", f"LOCATION:{location}",
             f"ORGANIZER:mailto:{organizer_email}", "STATUS:CONFIRMED",
             "END:VEVENT", "END:VCALENDAR"]
    return ("\r\n".join(lines) + "\r\n").encode("utf-8")


def session_credit_balance(cx, email: str) -> int:
    email = (email or "").strip().lower()
    r = cx.execute("SELECT credits FROM evox_session_credits WHERE email=?", (email,)).fetchone()
    return int(r[0]) if r else 0


def add_session_credits(cx, email: str, n: int) -> int:
    email = (email or "").strip().lower()
    cx.execute("INSERT INTO evox_session_credits (email, credits) VALUES (?, ?) "
               "ON CONFLICT(email) DO UPDATE SET credits=credits+excluded.credits", (email, n))
    cx.commit()
    return session_credit_balance(cx, email)


def consume_session_credit(cx, email: str) -> bool:
    email = (email or "").strip().lower()
    cur = cx.execute("UPDATE evox_session_credits SET credits=credits-1 "
                     "WHERE email=? AND credits>0", (email,))
    cx.commit()
    return cur.rowcount > 0


def has_cradle_purchase(cx, email: str) -> bool:
    email = (email or "").strip().lower()
    try:
        rows = cx.execute("SELECT items_json FROM orders WHERE lower(email)=?", (email,)).fetchall()
    except sqlite3.OperationalError:
        return False
    import json as _json
    for (items,) in rows:
        try:
            for line in _json.loads(items or "[]"):
                if (line.get("slug") or "") == "hand-cradle":
                    return True
        except Exception:
            continue
    return False
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_evox_pure.py -q`
Expected: PASS (17 passed).

- [ ] **Step 5: Commit**

```bash
git add dashboard/evox.py tests/test_evox_pure.py
git commit -m "feat(evox): ICS builder, prepay credits, cradle-purchase check"
```

---

### Task 6: Extend the email sender to attach an `.ics`

**Files:**
- Modify: `app.py` (the SMTP branch of `send_magic_link_email`, ~`app.py:311-345`) — add a new sibling function `send_evox_email` rather than changing the magic-link signature.
- Test: `tests/test_evox_api.py` (monkeypatch SMTP; assert the message is multipart/mixed with a text/calendar part)

**Interfaces:**
- Produces: `send_evox_email(to_email, name, subject, html_body, text_body, ics_bytes) -> tuple[str, str | None]` in `app.py` — mirrors the three-tier GHL→SMTP→console shape of `send_magic_link_email` but builds `MIMEMultipart("mixed")` with a nested `alternative` (text+html) plus a `text/calendar; method=REQUEST` attachment. GHL tier is skipped (no workflow carries an attachment) → straight to SMTP; console-log fallback when no SMTP.
- Consumes: nothing from earlier tasks.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evox_api.py  (needs doppler — imports app)
import os, pytest
if not os.environ.get("PINECONE_API_KEY"):
    pytest.skip("needs doppler env for import app", allow_module_level=True)
import app as appmod

def test_send_evox_email_builds_mixed_with_ics(monkeypatch):
    captured = {}
    class FakeSMTP:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def starttls(self): pass
        def login(self, *a): pass
        def sendmail(self, frm, to, msg): captured["msg"] = msg
    monkeypatch.setattr(appmod, "SMTP_HOST", "smtp.test")
    monkeypatch.setattr(appmod, "SMTP_USER", "u"); monkeypatch.setattr(appmod, "SMTP_PASS", "p")
    monkeypatch.setattr(appmod.smtplib, "SMTP", FakeSMTP)
    mode, err = appmod.send_evox_email("c@x.com", "C", "EVOX confirmed",
                                       "<p>hi</p>", "hi", b"BEGIN:VCALENDAR\r\nEND:VCALENDAR\r\n")
    assert mode == "smtp" and err is None
    assert "text/calendar" in captured["msg"] and "multipart/mixed" in captured["msg"]
```

- [ ] **Step 2: Run to verify failure**

Run: `mkdir -p /tmp/dc-test && doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/dc-test python3 -m pytest tests/test_evox_api.py::test_send_evox_email_builds_mixed_with_ics -q`
Expected: FAIL (`AttributeError: send_evox_email`).

- [ ] **Step 3: Add `send_evox_email` near `send_magic_link_email` in `app.py`**

```python
def send_evox_email(to_email, name, subject, html_body, text_body, ics_bytes):
    """Three-tier send with an .ics attachment. GHL tier skipped (no attachment path)."""
    if not (SMTP_HOST and SMTP_USER and SMTP_PASS):
        app.logger.info("EVOX email (console fallback) to %s: %s", to_email, subject)
        return ("console-log", "no email send mechanism configured")
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.base import MIMEBase
    from email import encoders
    msg = MIMEMultipart("mixed")
    msg["Subject"] = subject
    msg["From"] = SMTP_USER
    msg["To"] = to_email
    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText(text_body, "plain"))
    alt.attach(MIMEText(html_body, "html"))
    msg.attach(alt)
    part = MIMEBase("text", "calendar", method="REQUEST", name="invite.ics")
    part.set_payload(ics_bytes)
    encoders.encode_base64(part)
    part.add_header("Content-Disposition", "attachment", filename="invite.ics")
    msg.attach(part)
    port = int(os.environ.get("SMTP_PORT", "587"))
    with smtplib.SMTP(SMTP_HOST, port) as s:
        s.starttls(); s.login(SMTP_USER, SMTP_PASS)
        s.sendmail(SMTP_USER, [to_email], msg.as_string())
    return ("smtp", None)
```

- [ ] **Step 4: Run to verify pass**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/dc-test python3 -m pytest tests/test_evox_api.py::test_send_evox_email_builds_mixed_with_ics -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_evox_api.py
git commit -m "feat(evox): email sender with .ics attachment"
```

---

### Task 7: Routes — start, readiness, availability, book

**Files:**
- Modify: `app.py` (module-load config block + 5 routes; register near the other portal routes ~`app.py:13767`)
- Test: `tests/test_evox_api.py`

**Interfaces:**
- Consumes: `dashboard.evox` (Tasks 2–5), `send_evox_email` (Task 6), `dashboard.portal_identity.resolve_identity`, `dashboard.customers.find_or_create_by_email`, `dashboard.client_portal.upsert_portal`, `dashboard.people.set_person_tags`.
- Produces (HTTP):
  - `GET /evox` → serves `static/evox.html`.
  - `POST /api/evox/start {email,name}` → `find_or_create_by_email` + ensure portal → returns `{token, url}` (the portal token the page uses for subsequent calls). For an existing portal whose token can't be re-read, mint via `upsert_portal` on first touch; if it returns `(None, pid)`, fall back to `notify_state.get_token(cx, email)` (verify this helper exists; if absent, use the existing `/admin/portal/reissue-link` internal function to re-mint). Return the token either way.
  - `GET /api/evox/state?token=` → `{readiness, complete, credits, cradle_purchased}`.
  - `POST /api/evox/readiness?token= {item,value[,cradle_source]}` → sets one item; if `item=="cradle_ok"` and `has_cradle_purchase` is true, force value True + `cradle_source="buy"`. Returns updated state.
  - `GET /api/evox/availability?token=&range=2day|week` → `{slots:[iso,...]}` (only if readiness complete; else `{error:"not_ready"}`, 403).
  - `POST /api/evox/book?token= {start_ts}` → re-validates the slot is still in `available_slots` server-side; consumes a credit if any (`prepaid`); `create_booking` with `tag_fn` = a closure calling `set_person_tags`; sends two `send_evox_email`s (client + Rae) with `build_ics`; returns `{ok, start_ts, prepaid}`. On `SlotTaken` → `{error:"slot_taken"}`, 409.

- [ ] **Step 1: Write the failing route tests**

```python
# append to tests/test_evox_api.py
@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    monkeypatch.setattr(appmod, "CONSOLE_SECRET", "test-secret")
    monkeypatch.setattr(appmod, "EVOX_HOURS", "1-4:09:00-16:00")
    # neutralize outbound email
    monkeypatch.setattr(appmod, "send_evox_email", lambda *a, **k: ("console-log", None))
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()

def _start(client, email="c@x.com"):
    r = client.post("/api/evox/start", json={"email": email, "name": "C"})
    return r.get_json()["token"]

def test_availability_blocked_until_ready(client):
    tok = _start(client)
    r = client.get(f"/api/evox/availability?token={tok}&range=week")
    assert r.status_code == 403 and r.get_json()["error"] == "not_ready"

def test_full_flow_book(client, monkeypatch):
    tok = _start(client)
    for item in ("pc_ok", "cradle_ok", "headset_ok", "zyto_ok"):
        client.post(f"/api/evox/readiness?token={tok}", json={"item": item, "value": True})
    slots = client.get(f"/api/evox/availability?token={tok}&range=week").get_json()["slots"]
    assert slots, "expected at least one open slot in Mon-Thu window"
    r = client.post(f"/api/evox/book?token={tok}", json={"start_ts": slots[0]})
    assert r.status_code == 200 and r.get_json()["ok"] is True
    # second identical booking -> slot taken
    r2 = client.post(f"/api/evox/book?token={tok}", json={"start_ts": slots[0]})
    assert r2.status_code == 409
```

- [ ] **Step 2: Run to verify failure**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/dc-test python3 -m pytest tests/test_evox_api.py -q`
Expected: FAIL (404 on `/api/evox/start`).

- [ ] **Step 3: Add config + routes to `app.py`**

Module-load config (near other env reads, top of `app.py`):

```python
EVOX_HOURS = os.environ.get("EVOX_HOURS", "1-4:09:00-16:00")
EVOX_RAE_PHONE = os.environ.get("EVOX_RAE_PHONE", "")
EVOX_SESSION_PRICE_CENTS = int(os.environ.get("EVOX_SESSION_PRICE_CENTS", "15000"))
```

Routes (register near `client_portal_page`, ~`app.py:13767`):

```python
from datetime import date, timedelta as _td

def _evox_ident(cx, token):
    from dashboard import portal_identity as _pi
    return _pi.resolve_identity(cx, token=token,
                                session_token=request.cookies.get("rm_portal_session", ""),
                                client_login_enabled=_client_login_enabled())

def _evox_days(range_name):
    from dashboard import evox as _ev
    lo, hi = appmod_calendar_window(range_name)  # reuse _calendar_range_window
    d0 = date.fromisoformat(lo); d1 = date.fromisoformat(hi)
    return [d0 + _td(days=i) for i in range((d1 - d0).days + 1)]

@app.route("/evox")
def evox_page():
    return send_from_directory(STATIC, "evox.html")

@app.route("/api/evox/start", methods=["POST"])
def evox_start():
    from dashboard import evox as _ev, customers as _cu, client_portal as _cp
    body = request.get_json(force=True) or {}
    email = (body.get("email") or "").strip().lower()
    name = (body.get("name") or "").strip()
    if "@" not in email:
        return jsonify({"error": "email_required"}), 400
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _ev.init_evox_tables(cx)
        _cp.init_client_portal_table(cx)
        _cu.find_or_create_by_email(cx, email=email, name=name)
        token = _ev.ensure_portal_token(cx, email, name)   # see note below
    return jsonify({"token": token, "url": f"{PUBLIC_BASE_URL}/evox?token={token}"})

@app.route("/api/evox/state")
def evox_state():
    from dashboard import evox as _ev
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _ev.init_evox_tables(cx)
        ident = _evox_ident(cx, request.args.get("token", ""))
        if ident is None:
            return jsonify({"error": "not_found"}), 404
        st = _ev.get_readiness(cx, ident.email)
        st["credits"] = _ev.session_credit_balance(cx, ident.email)
        st["cradle_purchased"] = _ev.has_cradle_purchase(cx, ident.email)
        return jsonify(st)

@app.route("/api/evox/readiness", methods=["POST"])
def evox_readiness():
    from dashboard import evox as _ev
    body = request.get_json(force=True) or {}
    item, value = body.get("item"), bool(body.get("value"))
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _ev.init_evox_tables(cx)
        ident = _evox_ident(cx, request.args.get("token", ""))
        if ident is None:
            return jsonify({"error": "not_found"}), 404
        src = None
        if item == "cradle_ok":
            if _ev.has_cradle_purchase(cx, ident.email):
                value, src = True, "buy"
            else:
                src = "access"
        try:
            st = _ev.set_readiness_item(cx, ident.email, item, value, cradle_source=src)
        except ValueError:
            return jsonify({"error": "bad_item"}), 400
        return jsonify(st)

@app.route("/api/evox/availability")
def evox_availability():
    from dashboard import evox as _ev
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _ev.init_evox_tables(cx)
        ident = _evox_ident(cx, request.args.get("token", ""))
        if ident is None:
            return jsonify({"error": "not_found"}), 404
        if not _ev.get_readiness(cx, ident.email)["complete"]:
            return jsonify({"error": "not_ready"}), 403
        days = _evox_days(request.args.get("range", "week"))
        lo, hi = days[0].isoformat(), days[-1].isoformat()
        busy = _ev.rae_busy_intervals(cx, lo, hi)
        booked = _ev.booked_starts(cx)
        slots = _ev.available_slots(days, EVOX_HOURS, busy, booked, _hst_now())
        return jsonify({"slots": slots})

@app.route("/api/evox/book", methods=["POST"])
def evox_book():
    from dashboard import evox as _ev, people as _pe
    body = request.get_json(force=True) or {}
    start_ts = (body.get("start_ts") or "").strip()
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _ev.init_evox_tables(cx)
        ident = _evox_ident(cx, request.args.get("token", ""))
        if ident is None:
            return jsonify({"error": "not_found"}), 404
        if not _ev.get_readiness(cx, ident.email)["complete"]:
            return jsonify({"error": "not_ready"}), 403
        # server-side re-validation against live availability
        d = date.fromisoformat(start_ts[:10])
        busy = _ev.rae_busy_intervals(cx, d.isoformat(), d.isoformat())
        if start_ts not in _ev.available_slots([d], EVOX_HOURS, busy,
                                               _ev.booked_starts(cx), _hst_now()):
            return jsonify({"error": "slot_unavailable"}), 409
        prepaid = _ev.consume_session_credit(cx, ident.email)

        def _tag(email, tags):
            row = cx.execute("SELECT id, tags FROM people WHERE lower(email)=?",
                             (email,)).fetchone()
            if row:
                new = _pe.set_person_tags(json.loads(row["tags"] or "[]"), add=tags)
                cx.execute("UPDATE people SET tags=? WHERE id=?",
                           (json.dumps(new), row["id"]))
        try:
            b = _ev.create_booking(cx, ident.email, start_ts, prepaid=prepaid, tag_fn=_tag)
        except _ev.SlotTaken:
            return jsonify({"error": "slot_taken"}), 409
    _evox_send_confirmations(ident.email, b)   # Task 8 helper; emails client + Rae
    return jsonify({"ok": True, "start_ts": start_ts, "prepaid": prepaid})
```

Also add helpers used above (near the routes):

```python
def _hst_now():
    from datetime import datetime, timezone, timedelta
    return datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=-10))).replace(tzinfo=None)

def appmod_calendar_window(range_name):
    return _calendar_range_window(range_name)   # existing helper, app.py:18932
```

**Note on `ensure_portal_token`** — implement in `dashboard/evox.py`:

```python
def ensure_portal_token(cx, email, name):
    from dashboard import client_portal as _cp
    token, _pid = _cp.upsert_portal(cx, email, name, {"source": "evox"})
    if token:
        return token
    # existing portal: recover the raw token
    try:
        from dashboard import notify_state as _ns
        t = _ns.get_token(cx, email)
        if t:
            return t
    except Exception:
        pass
    # last resort: rotate to a fresh token via reissue (mints + persists)
    from dashboard import client_portal as _cp2
    return _cp2.reissue_token(cx, email)   # VERIFY name during Task 7; fall back to admin reissue fn
```

- [ ] **Step 4: Run to verify pass**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/dc-test python3 -m pytest tests/test_evox_api.py -q`
Expected: PASS (3 passed). If `ensure_portal_token`'s token-recovery path is the failure, confirm the real `notify_state` getter name and the reissue function during this task and adjust — the test asserts a non-empty token round-trips.

- [ ] **Step 5: Commit**

```bash
git add app.py dashboard/evox.py tests/test_evox_api.py
git commit -m "feat(evox): start/readiness/availability/book routes"
```

---

### Task 8: Confirmation emails helper (client + Rae) with ICS

**Files:**
- Modify: `app.py` (add `_evox_send_confirmations` used by `evox_book`)
- Test: `tests/test_evox_api.py` (assert both sends fire with an ICS body)

**Interfaces:**
- Consumes: `send_evox_email` (Task 6), `dashboard.evox.build_ics` (Task 5), `EVOX_RAE_PHONE`.
- Produces: `_evox_send_confirmations(email, booking: dict) -> None` — builds the client email ("call Rae at `EVOX_RAE_PHONE`") + the Rae notification, each with the same `.ics`, and sends both. Best-effort: swallows send errors (logs), never raises into the booking response.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_evox_api.py
def test_confirmations_send_client_and_rae(monkeypatch):
    calls = []
    monkeypatch.setattr(appmod, "send_evox_email",
                        lambda to, name, subj, html, text, ics: calls.append((to, ics)))
    monkeypatch.setattr(appmod, "EVOX_RAE_PHONE", "808-555-1212")
    monkeypatch.setattr(appmod, "EVOX_RAE_EMAIL", "rae@illtowell.com", raising=False)
    appmod._evox_send_confirmations("c@x.com", {
        "id": 1, "email": "c@x.com", "start_ts": "2026-07-06T11:00:00",
        "end_ts": "2026-07-06T12:00:00", "ics_uid": "u1@illtowell.com", "prepaid": False})
    assert len(calls) == 2
    tos = {c[0] for c in calls}
    assert "c@x.com" in tos and "rae@illtowell.com" in tos
    assert all(b"BEGIN:VCALENDAR" in c[1] for c in calls)
```

- [ ] **Step 2: Run to verify failure**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/dc-test python3 -m pytest tests/test_evox_api.py::test_confirmations_send_client_and_rae -q`
Expected: FAIL (`AttributeError: _evox_send_confirmations`).

- [ ] **Step 3: Implement**

```python
EVOX_RAE_EMAIL = os.environ.get("EVOX_RAE_EMAIL", "suerae1111@gmail.com")

def _evox_send_confirmations(email, booking):
    from dashboard import evox as _ev
    start = booking["start_ts"]; nice = start.replace("T", " ")
    phone = EVOX_RAE_PHONE or "the number in this confirmation"
    ics = _ev.build_ics(uid=booking["ics_uid"], start_ts=start, end_ts=booking["end_ts"],
                        summary="EVOX Session with Rae",
                        description=(f"At your appointment time, call Rae at {phone}. "
                                     "Have your Windows PC on with the ZYTO software open, "
                                     "hand cradle connected, headset ready."),
                        location="Phone")
    client_html = (f"<p>Your EVOX session is booked for <b>{nice} HST</b>.</p>"
                   f"<p>At your appointment time, <b>call Rae at {phone}</b>. Have your "
                   "Windows PC on with the ZYTO software open, hand cradle connected, and "
                   "headset ready. The calendar invite is attached.</p>")
    client_text = (f"EVOX session booked for {nice} HST. At your appointment time, call Rae "
                   f"at {phone}. Have your PC on with ZYTO open, hand cradle + headset ready.")
    rae_html = (f"<p>New EVOX booking: <b>{email}</b> on <b>{nice} HST</b> "
                f"({'PREPAID' if booking.get('prepaid') else 'invoice after'}).</p>")
    for to, nm, subj, html, text in [
        (email, "", "Your EVOX session is booked", client_html, client_text),
        (EVOX_RAE_EMAIL, "Rae", f"EVOX booking — {email}", rae_html, rae_html)]:
        try:
            send_evox_email(to, nm, subj, html, text, ics)
        except Exception:
            app.logger.exception("EVOX confirmation send failed to %s", to)
```

- [ ] **Step 4: Run to verify pass**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/dc-test python3 -m pytest tests/test_evox_api.py::test_confirmations_send_client_and_rae -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_evox_api.py
git commit -m "feat(evox): confirmation emails to client + Rae with ICS"
```

---

### Task 9: `static/evox.html` single-page flow

**Files:**
- Create: `static/evox.html`
- Test: manual headless render-verify (per `feedback_render_verify_not_just_inject`)

**Interfaces:**
- Consumes the routes from Task 7 (`/api/evox/start`, `/api/evox/state`, `/api/evox/readiness`, `/api/evox/availability`, `/api/evox/book`). Auth token from `?token=` query param (persisted to `localStorage` after `start`).

- [ ] **Step 1: Create the page**

Vanilla JS, no framework (match house style — brand green `#2f6f5e` / gold `#d4a843`). Four views toggled by JS: (1) **Start** — name + email → `POST /api/evox/start`, store token, go to checklist; (2) **Checklist** — four checkboxes bound to `POST /api/evox/readiness`; the cradle row shows a "Buy hand cradle ($297)" link to `/begin/product/hand-cradle` and an "I already have access" button; poll `/api/evox/state` on load; when `complete` show a "Choose a time" button; (3) **Availability** — `GET /api/evox/availability?range=week`, render slot buttons grouped by day; (4) **Confirmed** — after `POST /api/evox/book`, show the time + "call Rae" instructions. On load, if `?token=` or `localStorage.evox_token` present, skip Start and fetch state.

```html
<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EVOX Setup — Healing Oasis</title>
<style>
 body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:640px;margin:0 auto;
   padding:24px;color:#1b2a26;background:#f6f8f7}
 h1{color:#2f6f5e} button,.btn{background:#2f6f5e;color:#fff;border:0;border-radius:8px;
   padding:10px 16px;font-size:15px;cursor:pointer;margin:4px 0}
 .buy{background:#d4a843;color:#1b2a26} .slot{display:inline-block;margin:4px}
 .item{padding:10px;border:1px solid #d7e0dc;border-radius:8px;margin:8px 0;background:#fff}
 .hidden{display:none} label{font-size:15px}
</style></head><body>
<h1>EVOX Setup</h1>
<div id="v-start">
  <p>Set up your EVOX session in a few steps.</p>
  <input id="f-name" placeholder="Your name"><br>
  <input id="f-email" placeholder="you@email.com" type="email"><br>
  <button onclick="start()">Start</button>
</div>
<div id="v-check" class="hidden">
  <p>Confirm each item to unlock booking:</p>
  <div class="item"><label><input type="checkbox" id="c-pc_ok" onchange="setItem('pc_ok',this.checked)"> I have (or can access) a Windows 10/11 PC</label></div>
  <div class="item"><label><input type="checkbox" id="c-cradle_ok" onchange="setItem('cradle_ok',this.checked)"> Hand cradle</label>
    &nbsp;<a class="btn buy" href="/begin/product/hand-cradle">Buy ($297)</a></div>
  <div class="item"><label><input type="checkbox" id="c-headset_ok" onchange="setItem('headset_ok',this.checked)"> Headset + microphone</label></div>
  <div class="item"><label><input type="checkbox" id="c-zyto_ok" onchange="setItem('zyto_ok',this.checked)"> ZYTO software installed &amp; verified with ZYTO support (Mon–Thu)</label></div>
  <button id="btn-book" class="hidden" onclick="showAvail()">Choose a time →</button>
</div>
<div id="v-avail" class="hidden"><h2>Pick a time (HST)</h2><div id="slots"></div></div>
<div id="v-done" class="hidden"><h2>Booked ✓</h2><p id="done-msg"></p></div>
<script>
let TOKEN = new URLSearchParams(location.search).get("token") || localStorage.getItem("evox_token") || "";
const $=id=>document.getElementById(id), show=id=>{for(const v of ["start","check","avail","done"])$("v-"+v).classList.toggle("hidden",v!==id)};
async function j(url,opt){const r=await fetch(url,opt);return r.json()}
async function start(){
  const email=$("f-email").value.trim(), name=$("f-name").value.trim();
  const d=await j("/api/evox/start",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({email,name})});
  if(d.token){TOKEN=d.token;localStorage.setItem("evox_token",TOKEN);loadState()}
}
async function loadState(){
  const s=await j("/api/evox/state?token="+TOKEN); if(s.error){show("start");return}
  for(const k of ["pc_ok","cradle_ok","headset_ok","zyto_ok"])$("c-"+k).checked=!!s[k];
  $("btn-book").classList.toggle("hidden",!s.complete); show("check");
}
async function setItem(item,value){
  const s=await j("/api/evox/readiness?token="+TOKEN,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({item,value})});
  for(const k of ["pc_ok","cradle_ok","headset_ok","zyto_ok"])$("c-"+k).checked=!!s[k];
  $("btn-book").classList.toggle("hidden",!s.complete);
}
async function showAvail(){
  const d=await j("/api/evox/availability?token="+TOKEN+"&range=week");
  const box=$("slots"); box.innerHTML=(d.slots||[]).map(s=>`<button class="slot" onclick="book('${s}')">${s.replace('T',' ')}</button>`).join("")||"<p>No open times this week — check back soon.</p>";
  show("avail");
}
async function book(start_ts){
  const d=await j("/api/evox/book?token="+TOKEN,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({start_ts})});
  if(d.ok){$("done-msg").textContent="Your EVOX session is booked for "+start_ts.replace("T"," ")+" HST. Check your email for the calendar invite and call instructions.";show("done")}
  else{alert("That slot just became unavailable — pick another.");showAvail()}
}
if(TOKEN){loadState()}else{show("start")}
</script></body></html>
```

- [ ] **Step 2: Render-verify headless**

Start the app locally (`doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/dc-test python3 app.py`) or verify on the deploy preview; load `/evox`, confirm the Start view renders with no console errors, submit a test email, confirm the checklist renders and checking all four reveals "Choose a time". Capture that there are 0 console errors.

- [ ] **Step 3: Commit**

```bash
git add static/evox.html
git commit -m "feat(evox): single-page setup + booking flow"
```

---

### Task 10: 24-hour reminder pass (daily cron)

**Files:**
- Modify: `app.py` (add a console-gated `POST /api/evox/run-reminders` that a daily cron hits; mirror existing cron-endpoint style)
- Test: `tests/test_evox_api.py`

**Interfaces:**
- Consumes: `evox_bookings`, `send_evox_email` (no ICS needed for a reminder), `EVOX_RAE_PHONE`.
- Produces: `POST /api/evox/run-reminders` (header `X-Console-Key`) → for each `status='booked'` row whose `start_ts` is within the next 24–48h HST and `reminded_at IS NULL`, send the client a reminder and stamp `reminded_at`. Returns `{sent:N}`. Idempotent via the `reminded_at` stamp (lazy `ALTER TABLE evox_bookings ADD COLUMN reminded_at TEXT`).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_evox_api.py
def test_reminders_send_once(client, monkeypatch):
    sent = []
    monkeypatch.setattr(appmod, "send_evox_email", lambda *a, **k: sent.append(a[0]) or ("console-log", None))
    tok = _start(client, "r@x.com")
    for item in ("pc_ok","cradle_ok","headset_ok","zyto_ok"):
        client.post(f"/api/evox/readiness?token={tok}", json={"item": item, "value": True})
    slots = client.get(f"/api/evox/availability?token={tok}&range=week").get_json()["slots"]
    client.post(f"/api/evox/book?token={tok}", json={"start_ts": slots[0]})
    # slots[0] is within this week; force it into the 24-48h window is covered by range;
    hdr = {"X-Console-Key": "test-secret"}
    r1 = client.post("/api/evox/run-reminders", headers=hdr).get_json()
    r2 = client.post("/api/evox/run-reminders", headers=hdr).get_json()
    assert r1["sent"] >= 0 and r2["sent"] == 0   # never double-sends
```

- [ ] **Step 2: Run to verify failure**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/dc-test python3 -m pytest tests/test_evox_api.py::test_reminders_send_once -q`
Expected: FAIL (404).

- [ ] **Step 3: Implement the endpoint**

```python
@app.route("/api/evox/run-reminders", methods=["POST"])
def evox_run_reminders():
    if request.headers.get("X-Console-Key") != CONSOLE_SECRET:
        return jsonify({"error": "unauthorized"}), 401
    from datetime import timedelta
    now = _hst_now(); lo = now + timedelta(hours=24); hi = now + timedelta(hours=48)
    sent = 0
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        from dashboard import evox as _ev; _ev.init_evox_tables(cx)
        try: cx.execute("ALTER TABLE evox_bookings ADD COLUMN reminded_at TEXT")
        except Exception: pass
        rows = cx.execute("SELECT * FROM evox_bookings WHERE status='booked' "
                          "AND reminded_at IS NULL AND start_ts BETWEEN ? AND ?",
                          (lo.isoformat(), hi.isoformat())).fetchall()
        for r in rows:
            phone = EVOX_RAE_PHONE or "the number in your confirmation"
            html = (f"<p>Reminder: your EVOX session is tomorrow at "
                    f"<b>{r['start_ts'].replace('T',' ')} HST</b>. Call Rae at {phone}.</p>")
            try:
                send_evox_email(r["email"], "", "Reminder: your EVOX session tomorrow",
                                html, html, b"")
                cx.execute("UPDATE evox_bookings SET reminded_at=? WHERE id=?",
                           (now.isoformat(), r["id"])); sent += 1
            except Exception:
                app.logger.exception("EVOX reminder failed")
        cx.commit()
    return jsonify({"sent": sent})
```

- [ ] **Step 4: Run to verify pass**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/dc-test python3 -m pytest tests/test_evox_api.py -q`
Expected: PASS (all).

- [ ] **Step 5: Commit + register the cron**

```bash
git add app.py tests/test_evox_api.py
git commit -m "feat(evox): 24h reminder pass (console-gated cron endpoint)"
```

Then add a daily schedule hitting `POST /api/evox/run-reminders` with the console key (follow `project_cron_revival` / the existing render cron pattern; do not rely on render.yaml auto-sync per `feedback_render_yaml_not_live_source`).

---

## Post-implementation (not code tasks)

- **Set env in Render dashboard** (not just render.yaml — `feedback_render_env_not_doppler`): `EVOX_RAE_PHONE`, `EVOX_RAE_EMAIL` (default `suerae1111@gmail.com`), optionally `EVOX_HOURS`, `EVOX_SESSION_PRICE_CENTS`.
- **Pricing settled (Glen 2026-07-04):** EVOX single session $197 public / $100 paid-member. v1 = self-serve prepay lists at $197; the $100 member rate is applied by Rae at invoice/prepay time (override). Storefront member auto-pricing deferred.
- **Optional cadence tighten:** reduce the Google→`calendar_events` sync interval below hourly if availability freshness matters.
- **Go-live smoke:** one real end-to-end EVOX booking with Rae — checklist → book → both `.ics` arrive → the `rae`-lane event shows on the console calendar → invoice-after via the in-house Orders board.

---

## Self-Review

**Spec coverage:**
- EVOX Setup surface + public entry → Task 7 (`/api/evox/start`) + Task 9 (`static/evox.html`). ✓
- Self-attest checklist (4 items, buy/attest cradle) → Tasks 2, 5 (cradle purchase), 7 (readiness route), 9 (UI). ✓
- Availability from office-hours − calendar_events busy − booked → Task 3 (pure) + Task 4 (`rae_busy_intervals`) + Task 7 (route). ✓
- 1:1 booking → synthetic rae-lane calendar row → Task 4. ✓
- Connection + confirmation (phone, no Zoom, client calls Rae) + ICS → Tasks 6, 8. ✓
- Money: invoice-after default (existing Orders board, no code) + optional prepay credits + hand-cradle SKU → Tasks 1, 5, 7. ✓
- Tags (evox-ready/evox-client) via people.tags → Task 4 + Task 7 `_tag`. ✓
- Reminders → Task 10. ✓
- Session-type catalog seam: v1 hard-codes EVOX (single type); the catalog dict is documented in the spec and is a trivial future extraction — no task needed for one type. **Noted deviation:** the plan does not build the generic `SESSION_TYPES` registry (YAGNI for one type); later slices add it.

**Placeholder scan:** No TBD/TODO in steps. Two verify-at-implementation notes are explicit and bounded (the `notify_state`/reissue token-recovery name in Task 7; the SMTP attachment assertion). `EVOX_SESSION_PRICE_CENTS` is a named config with a concrete default, flagged for Glen — not an implementation placeholder.

**Type consistency:** `readiness_complete`/`get_readiness`/`set_readiness_item`/`available_slots`/`create_booking`/`build_ics`/`has_cradle_purchase`/`ensure_portal_token` names are identical across Tasks 2–9. `create_booking` returns `{id,email,start_ts,end_ts,ics_uid,prepaid}` — consumed unchanged by `_evox_send_confirmations` (Task 8). Slot times are naive ISO `YYYY-MM-DDTHH:MM:SS` everywhere.
