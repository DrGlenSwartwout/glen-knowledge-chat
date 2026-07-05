# MasterClass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Let Glen create a one-to-many MasterClass event (system creates the Zoom Meeting), publish a public event page, and let people register with member-tiered pricing (members free, non-members pay via Stripe), delivering the Zoom link to registrants.

**Architecture:** A new `dashboard/masterclass.py` (events + registrations store), a console create endpoint that also creates the Zoom Meeting + a calendar row, a public event page + register endpoint (free/member immediate; non-member paid via the existing `dashboard/stripe_pay.py` checkout), a `_fulfill_masterclass` webhook handler, and `static/masterclass.html`. Reuses `create_meeting`, `_is_paid_member`, the Stripe fan-out webhook, and the email/ICS rail.

**Tech Stack:** Python 3 / Flask, sqlite, `dashboard/stripe_pay.py` (raw-HTTPS Stripe), `dashboard/zoom.py`, existing email/ICS rail, vanilla-JS page. Tests via `app.app.test_client()` with mocked Stripe/Zoom.

## Global Constraints

- sqlite `?` placeholders; writes under `with _db_lock, sqlite3.connect(LOG_DB) as cx:` + `cx.row_factory=Row`. Timestamps `datetime.now(timezone.utc).isoformat()`; `start_ts` naive ISO HST.
- Console create is `_portal_console_ok()`-gated. Public register/event routes are open (registration IS the flow).
- **Member-tiered pricing:** `is_member = _is_paid_member(email)`; `amount = member_price_cents if is_member else price_cents`. `amount == 0` → register immediately; `amount > 0` → Stripe checkout, registration completes on `checkout.session.completed`.
- **Stripe** via `dashboard.stripe_pay.create_checkout_session(amount_cents, *, customer_email, description, metadata, success_url, cancel_url) -> {id,url}`; gate paid path on `_STRIPE_ACTIVE`. Fulfillment: `get_session(id)` → `metadata`/`payment_intent`; `get_payment_intent(pi)` → `status`. Webhook fans out to every `_fulfill_*(session_id)`; each re-fetches + no-ops on non-matching `metadata["kind"]`.
- **Zoom:** `create_meeting(..., waiting_room=False)` for MasterClass (registration is the gate). Add a `waiting_room: bool = True` param to `create_meeting` (default preserves EVOX/consult behavior). Auto-create is **best-effort** — a failure does not block event creation (manual URL fallback).
- Raw Zoom link is delivered to registrants only (register response + confirmation email), not on the pre-register public page. Client-facing copy: **no em dashes, no ALL CAPS.**
- Test cmd: `mkdir -p /tmp/dc-test && doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/dc-test python3 -m pytest tests/<file>.py -q`. Pure tests (dashboard-only import) run with plain pytest.
- Regression: EVOX/consult/triage untouched; `create_meeting`'s new param defaults to old behavior.

---

### Task 1: `dashboard/masterclass.py` — events + registrations store

**Files:** Create `dashboard/masterclass.py`; Test `tests/test_masterclass_pure.py`

**Interfaces:**
- Produces: `init_masterclass_tables(cx)`; `create_event(cx, *, topic, description, start_ts, duration_min, price_cents, member_price_cents) -> int` (event id); `get_event(cx, event_id) -> dict|None`; `set_zoom(cx, event_id, join_url, meeting_id)`; `price_for(event, is_member) -> int`; `register(cx, event_id, email, name, is_member, amount_cents, *, paid) -> None` (upsert on `(event_id,email)`); `mark_paid(cx, event_id, email)`; `is_registered(cx, event_id, email) -> bool` (True only when a paid=1 row exists).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_masterclass_pure.py
import sqlite3
from dashboard import masterclass as mc

def _cx():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    mc.init_masterclass_tables(cx); return cx

def test_event_create_get_zoom_price():
    cx = _cx()
    eid = mc.create_event(cx, topic="Terrain 101", description="d",
                          start_ts="2026-07-10T18:00:00", duration_min=60,
                          price_cents=5000, member_price_cents=0)
    ev = mc.get_event(cx, eid)
    assert ev["topic"] == "Terrain 101" and ev["price_cents"] == 5000
    assert mc.price_for(ev, is_member=True) == 0
    assert mc.price_for(ev, is_member=False) == 5000
    mc.set_zoom(cx, eid, "https://zoom.us/j/123", "123")
    assert mc.get_event(cx, eid)["zoom_join_url"] == "https://zoom.us/j/123"

def test_register_upsert_and_mark_paid():
    cx = _cx()
    eid = mc.create_event(cx, topic="T", description="", start_ts="2026-07-10T18:00:00",
                          duration_min=60, price_cents=5000, member_price_cents=0)
    mc.register(cx, eid, "A@x.com", "A", is_member=False, amount_cents=5000, paid=False)
    assert mc.is_registered(cx, eid, "a@x.com") is False        # pending, not paid
    mc.mark_paid(cx, eid, "a@x.com")
    assert mc.is_registered(cx, eid, "A@x.com") is True          # lowercased
    # re-register (upsert) doesn't duplicate
    mc.register(cx, eid, "a@x.com", "A", is_member=True, amount_cents=0, paid=True)
    n = cx.execute("SELECT COUNT(*) FROM masterclass_registrations WHERE event_id=?", (eid,)).fetchone()[0]
    assert n == 1 and mc.is_registered(cx, eid, "a@x.com") is True
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_masterclass_pure.py -q`
Expected: FAIL (`ModuleNotFoundError: dashboard.masterclass`).

- [ ] **Step 3: Create `dashboard/masterclass.py`**

```python
"""MasterClass events + registrations. Stdlib-only; import without importing app."""
import sqlite3
from datetime import datetime, timezone

def _now():
    return datetime.now(timezone.utc).isoformat()

def init_masterclass_tables(cx) -> None:
    cx.execute("""CREATE TABLE IF NOT EXISTS masterclass_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT, topic TEXT, description TEXT,
        start_ts TEXT, duration_min INTEGER DEFAULT 60,
        price_cents INTEGER DEFAULT 0, member_price_cents INTEGER DEFAULT 0,
        zoom_join_url TEXT, zoom_meeting_id TEXT, created_at TEXT)""")
    cx.execute("""CREATE TABLE IF NOT EXISTS masterclass_registrations (
        id INTEGER PRIMARY KEY AUTOINCREMENT, event_id INTEGER, email TEXT, name TEXT,
        is_member INTEGER, amount_cents INTEGER, paid INTEGER DEFAULT 0, created_at TEXT,
        UNIQUE(event_id, email))""")
    cx.commit()

def create_event(cx, *, topic, description, start_ts, duration_min,
                 price_cents, member_price_cents) -> int:
    cur = cx.execute(
        "INSERT INTO masterclass_events (topic, description, start_ts, duration_min, "
        "price_cents, member_price_cents, created_at) VALUES (?,?,?,?,?,?,?)",
        ((topic or "").strip(), (description or "").strip(), start_ts, int(duration_min),
         int(price_cents), int(member_price_cents), _now()))
    cx.commit()
    return cur.lastrowid

def get_event(cx, event_id):
    cur = cx.execute("SELECT * FROM masterclass_events WHERE id=?", (event_id,))
    cols = [c[0] for c in cur.description]; r = cur.fetchone()
    return dict(zip(cols, r)) if r is not None else None

def set_zoom(cx, event_id, join_url, meeting_id) -> None:
    cx.execute("UPDATE masterclass_events SET zoom_join_url=?, zoom_meeting_id=? WHERE id=?",
               (join_url, meeting_id, event_id))
    cx.commit()

def price_for(event, is_member) -> int:
    return int(event["member_price_cents"] if is_member else event["price_cents"])

def register(cx, event_id, email, name, is_member, amount_cents, *, paid) -> None:
    email = (email or "").strip().lower()
    cx.execute("INSERT INTO masterclass_registrations (event_id, email, name, is_member, "
               "amount_cents, paid, created_at) VALUES (?,?,?,?,?,?,?) "
               "ON CONFLICT(event_id, email) DO UPDATE SET name=excluded.name, "
               "is_member=excluded.is_member, amount_cents=excluded.amount_cents, "
               "paid=excluded.paid",
               (event_id, email, (name or "").strip(), 1 if is_member else 0,
                int(amount_cents), 1 if paid else 0, _now()))
    cx.commit()

def mark_paid(cx, event_id, email) -> None:
    cx.execute("UPDATE masterclass_registrations SET paid=1 WHERE event_id=? AND lower(email)=?",
               (event_id, (email or "").strip().lower()))
    cx.commit()

def is_registered(cx, event_id, email) -> bool:
    r = cx.execute("SELECT 1 FROM masterclass_registrations WHERE event_id=? AND lower(email)=? "
                   "AND paid=1", (event_id, (email or "").strip().lower())).fetchone()
    return r is not None
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_masterclass_pure.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add dashboard/masterclass.py tests/test_masterclass_pure.py
git commit -m "feat(masterclass): events + registrations store"
```

---

### Task 2: Zoom `waiting_room` param + console create endpoint

**Files:** Modify `dashboard/zoom.py` (`create_meeting` line 25-38); Modify `app.py` (new routes + config); Modify `static/console-biofield-portal.html` (form); Test `tests/test_masterclass_api.py`

**Interfaces:**
- Produces: `create_meeting(token, *, host, topic, start_iso, duration_min, timezone="Pacific/Honolulu", waiting_room=True, opener=None)` (the settings use the param). `POST /api/console/masterclass {topic, description, start_ts, duration_min, price_cents, member_price_cents}` (`_portal_console_ok`) → `create_event`; best-effort Zoom Meeting (`get_token`+`create_meeting(waiting_room=False)`) → `set_zoom`; a synthetic `glen`-lane `calendar_events` row (`google_event_id=f"masterclass-{id}"`); returns `{ok, event_id, event_url, zoom_ok}`. `POST /api/console/masterclass/<id>/zoom-url {url}` → `set_zoom(url, "")`. Uses `GLEN_ZOOM_USER`, `PUBLIC_BASE_URL`, `_init_calendar_table`.

- [ ] **Step 1: Add the `waiting_room` param to `dashboard/zoom.py:create_meeting`**

Change the signature + the settings line:
```python
def create_meeting(token, *, host, topic, start_iso, duration_min,
                   timezone="Pacific/Honolulu", waiting_room=True, opener=None):
    opener = opener or urllib.request.urlopen
    body = {"topic": topic, "type": 2, "start_time": start_iso,
            "duration": int(duration_min), "timezone": timezone,
            "settings": {"waiting_room": waiting_room, "join_before_host": False}}
    ...
```
Run the existing consult/evox pure tests to confirm no regression:
`python3 -m pytest tests/test_consult_pure.py -q` → PASS (default `waiting_room=True` preserves behavior).

- [ ] **Step 2: Write the failing console-create test**

```python
# tests/test_masterclass_api.py
import os, pytest
if not os.environ.get("PINECONE_API_KEY"):
    pytest.skip("needs doppler env for import app", allow_module_level=True)
import app as appmod

@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    monkeypatch.setattr(appmod, "CONSOLE_SECRET", "test-secret")
    monkeypatch.setattr(appmod, "send_evox_email", lambda *a, **k: ("console-log", None), raising=False)
    monkeypatch.setattr("dashboard.zoom.get_token", lambda *a, **k: "tok")
    monkeypatch.setattr("dashboard.zoom.create_meeting",
                        lambda *a, **k: {"join_url": "https://zoom.us/j/mc", "meeting_id": "mc1", "start_url": "x"})
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()

ADMIN = {"X-Console-Key": "test-secret"}

def test_console_create_requires_auth(client):
    r = client.post("/api/console/masterclass", json={"topic": "T", "start_ts": "2026-07-10T18:00:00"})
    assert r.status_code == 401

def test_console_create_makes_event_and_zoom(client):
    r = client.post("/api/console/masterclass",
                    json={"topic": "Terrain 101", "description": "d", "start_ts": "2026-07-10T18:00:00",
                          "duration_min": 60, "price_cents": 5000, "member_price_cents": 0}, headers=ADMIN)
    assert r.status_code == 200
    d = r.get_json()
    assert d["ok"] is True and d["zoom_ok"] is True and "/masterclass/" in d["event_url"]
    import sqlite3
    from dashboard import masterclass as mc
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        ev = mc.get_event(cx, d["event_id"])
        assert ev["zoom_join_url"] == "https://zoom.us/j/mc"
```

- [ ] **Step 3: Run to verify failure**

Run: `mkdir -p /tmp/dc-test && doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/dc-test python3 -m pytest tests/test_masterclass_api.py -q`
Expected: FAIL (404).

- [ ] **Step 4: Add the routes to `app.py`** (near other console routes)

```python
@app.route("/api/console/masterclass", methods=["POST"])
def api_console_masterclass_create():
    if not _portal_console_ok():
        return jsonify({"error": "unauthorized"}), 401
    from dashboard import masterclass as _mc, zoom as _zoom
    body = request.get_json(silent=True) or {}
    topic = (body.get("topic") or "").strip()
    start_ts = (body.get("start_ts") or "").strip()
    if not topic or not start_ts:
        return jsonify({"error": "topic and start_ts required"}), 400
    duration = int(body.get("duration_min") or 60)
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _mc.init_masterclass_tables(cx); _init_calendar_table()
        eid = _mc.create_event(cx, topic=topic, description=body.get("description") or "",
                               start_ts=start_ts, duration_min=duration,
                               price_cents=int(body.get("price_cents") or 0),
                               member_price_cents=int(body.get("member_price_cents") or 0))
        # synthetic glen-lane calendar row
        from datetime import datetime as _dt, timedelta as _td
        try:
            end_ts = (_dt.fromisoformat(start_ts[:19]) + _td(minutes=duration)).isoformat()
        except Exception:
            end_ts = start_ts
        now = datetime.now(timezone.utc).isoformat()
        try:
            cx.execute(
                "INSERT INTO calendar_events (pushed_at,google_cal_id,google_event_id,"
                "calendar_name,summary,start,end,location,owner,status,cal_alert) "
                "VALUES (?, 'delegated', ?, 'MasterClass', ?, ?, ?, 'Zoom', 'glen', 'visible', 0)",
                (now, f"masterclass-{eid}", f"MasterClass: {topic}", start_ts, end_ts))
        except Exception:
            app.logger.exception("masterclass calendar insert failed")
        cx.commit()
    # best-effort Zoom (outside the lock)
    zoom_ok = False
    try:
        tok = _zoom.get_token(os.environ["ZOOM_ACCOUNT_ID"], os.environ["ZOOM_CLIENT_ID"],
                              os.environ["ZOOM_CLIENT_SECRET"])
        m = _zoom.create_meeting(tok, host=GLEN_ZOOM_USER, topic=f"MasterClass: {topic}",
                                 start_iso=start_ts, duration_min=duration, waiting_room=False)
        with _db_lock, sqlite3.connect(LOG_DB) as cx2:
            _mc.init_masterclass_tables(cx2)
            _mc.set_zoom(cx2, eid, m.get("join_url"), m.get("meeting_id"))
        zoom_ok = bool(m.get("join_url"))
    except Exception:
        app.logger.exception("masterclass zoom create failed")
    return jsonify({"ok": True, "event_id": eid,
                    "event_url": f"{PUBLIC_BASE_URL}/masterclass/{eid}", "zoom_ok": zoom_ok})

@app.route("/api/console/masterclass/<int:event_id>/zoom-url", methods=["POST"])
def api_console_masterclass_zoom_url(event_id):
    if not _portal_console_ok():
        return jsonify({"error": "unauthorized"}), 401
    from dashboard import masterclass as _mc
    url = ((request.get_json(silent=True) or {}).get("url") or "").strip()
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _mc.init_masterclass_tables(cx)
        _mc.set_zoom(cx, event_id, url, "")
    return jsonify({"ok": True})
```

- [ ] **Step 5: Run to verify pass**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/dc-test python3 -m pytest tests/test_masterclass_api.py -q`
Expected: PASS (2 passed).

- [ ] **Step 6: Add a "Create MasterClass" form to `static/console-biofield-portal.html`** (reuse `api()`):

```html
<div style="margin-top:12px">
  <h3>Create MasterClass</h3>
  <input id="mc-topic" placeholder="topic">
  <input id="mc-start" placeholder="2026-07-10T18:00:00 (HST)">
  <input id="mc-price" placeholder="non-member price (cents)" value="0">
  <input id="mc-mprice" placeholder="member price (cents)" value="0">
  <button onclick="createMasterclass()">Create</button>
  <span id="mc-msg"></span>
</div>
<script>
async function createMasterclass(){
  const topic=(document.getElementById('mc-topic').value||'').trim();
  const start_ts=(document.getElementById('mc-start').value||'').trim();
  const price_cents=parseInt(document.getElementById('mc-price').value||'0',10);
  const member_price_cents=parseInt(document.getElementById('mc-mprice').value||'0',10);
  const r=await api('POST','/api/console/masterclass',{topic,start_ts,duration_min:60,price_cents,member_price_cents});
  document.getElementById('mc-msg').textContent = r.ok
    ? (r.json.event_url + (r.json.zoom_ok?' (Zoom ready)':' (add Zoom link manually)')) : ('Error '+r.status);
}
</script>
```

- [ ] **Step 7: Commit**

```bash
git add dashboard/zoom.py app.py static/console-biofield-portal.html tests/test_masterclass_api.py
git commit -m "feat(masterclass): console create + Zoom meeting + calendar row"
```

---

### Task 3: Public event page route + GET event + register (free/member/paid)

**Files:** Modify `app.py` (routes + `_masterclass_send_confirmation` helper + config `MASTERCLASS_FROM`); Test `tests/test_masterclass_api.py`

**Interfaces:**
- Consumes: `dashboard.masterclass`, `_is_paid_member`, `dashboard.stripe_pay.create_checkout_session`, `_STRIPE_ACTIVE`, `send_evox_email`, `dashboard.evox.build_ics`, `PUBLIC_BASE_URL`, `STATIC`.
- Produces: `GET /masterclass/<id>` → `send_from_directory(STATIC, "masterclass.html")`. `GET /api/masterclass/<id>` → `{topic, description, start_ts, duration_min, price_cents, member_price_cents}` (404 if none). `POST /api/masterclass/<id>/register {email, name}` → `is_member=_is_paid_member(email)`, `amount=price_for`: `amount<=0` → `register(paid=True)` + `_masterclass_send_confirmation` + `{ok, registered:true, join_url}`; `amount>0` → (503 `payment_unavailable` if not `_STRIPE_ACTIVE`) else `create_checkout_session(metadata={kind:'masterclass',event_id,email,name})` + `register(paid=False)` + `{ok, checkout_url}`. `_masterclass_send_confirmation(event, email, name)` best-effort (join link + ICS).

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_masterclass_api.py
def _mk_event(client, price=0, mprice=0):
    r = client.post("/api/console/masterclass",
                    json={"topic": "T", "description": "d", "start_ts": "2026-07-10T18:00:00",
                          "duration_min": 60, "price_cents": price, "member_price_cents": mprice}, headers=ADMIN)
    return r.get_json()["event_id"]

def test_public_get_event(client):
    eid = _mk_event(client, price=5000)
    d = client.get(f"/api/masterclass/{eid}").get_json()
    assert d["topic"] == "T" and d["price_cents"] == 5000 and "zoom_join_url" not in d

def test_register_free_returns_join_link(client, monkeypatch):
    eid = _mk_event(client, price=0, mprice=0)
    r = client.post(f"/api/masterclass/{eid}/register", json={"email": "free@x.com", "name": "F"})
    d = r.get_json()
    assert r.status_code == 200 and d["registered"] is True and d["join_url"] == "https://zoom.us/j/mc"

def test_register_nonmember_paid_returns_checkout(client, monkeypatch):
    monkeypatch.setattr(appmod, "_STRIPE_ACTIVE", True, raising=False)
    import dashboard.stripe_pay as _sp
    cap = {}
    def fake_session(amount_cents, *, customer_email, description, metadata, success_url, cancel_url, save_card=False):
        cap["amount"] = amount_cents; cap["metadata"] = metadata
        return {"id": "cs_test", "url": "https://stripe/mc"}
    monkeypatch.setattr(_sp, "create_checkout_session", fake_session)
    monkeypatch.setattr(appmod.stripe_pay, "create_checkout_session", fake_session, raising=False)
    eid = _mk_event(client, price=5000, mprice=0)
    r = client.post(f"/api/masterclass/{eid}/register", json={"email": "nonmember@x.com", "name": "N"})
    d = r.get_json()
    assert r.status_code == 200 and d["checkout_url"] == "https://stripe/mc"
    assert cap["amount"] == 5000 and cap["metadata"]["kind"] == "masterclass" and cap["metadata"]["event_id"] == str(eid)
```

- [ ] **Step 2: Run to verify failure**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/dc-test python3 -m pytest tests/test_masterclass_api.py -q`
Expected: FAIL (404 on `/api/masterclass/<id>`).

- [ ] **Step 3: Add routes + helper to `app.py`**

```python
MASTERCLASS_FROM = os.environ.get("GLEN_CONSULT_EMAIL", "drglenswartwout@gmail.com")

@app.route("/masterclass/<int:event_id>")
def masterclass_page(event_id):
    return send_from_directory(STATIC, "masterclass.html")

@app.route("/api/masterclass/<int:event_id>")
def api_masterclass_get(event_id):
    from dashboard import masterclass as _mc
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _mc.init_masterclass_tables(cx)
        ev = _mc.get_event(cx, event_id)
    if not ev:
        return jsonify({"error": "not_found"}), 404
    return jsonify({"topic": ev["topic"], "description": ev["description"],
                    "start_ts": ev["start_ts"], "duration_min": ev["duration_min"],
                    "price_cents": ev["price_cents"], "member_price_cents": ev["member_price_cents"]})

def _masterclass_send_confirmation(event, email, name):
    try:
        from dashboard import evox as _ev
        start = event["start_ts"]; nice = start.replace("T", " ")
        join = event.get("zoom_join_url") or ""
        join_line = (f"Join here: {join}" if join else "Your join link will follow by email.")
        try:
            end = (datetime.fromisoformat(start[:19]) + timedelta(minutes=int(event["duration_min"]))).isoformat()
        except Exception:
            end = start
        ics = _ev.build_ics(uid=f"masterclass-{event['id']}-{(email or '').strip().lower()}@illtowell.com",
                            start_ts=start, end_ts=end, summary=f"MasterClass: {event['topic']}",
                            description=join_line, location=(join or "Zoom"))
        html = (f"<p>You are registered for <b>{event['topic']}</b> on <b>{nice} HST</b>.</p>"
                f"<p>{join_line}</p><p>The calendar invite is attached.</p>")
        try:
            send_evox_email(email, name or "", f"You are registered: {event['topic']}", html, html, ics)
        except Exception:
            app.logger.exception("masterclass confirmation send failed to %s", email)
    except Exception:
        app.logger.exception("masterclass confirmation build failed")

@app.route("/api/masterclass/<int:event_id>/register", methods=["POST"])
def api_masterclass_register(event_id):
    from dashboard import masterclass as _mc
    body = request.get_json(silent=True) or {}
    email = (body.get("email") or "").strip().lower()
    name = (body.get("name") or "").strip()
    if "@" not in email:
        return jsonify({"error": "email required"}), 400
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _mc.init_masterclass_tables(cx)
        ev = _mc.get_event(cx, event_id)
        if not ev:
            return jsonify({"error": "not_found"}), 404
        is_member = _is_paid_member(email)
        amount = _mc.price_for(ev, is_member)
        if amount <= 0:
            _mc.register(cx, event_id, email, name, is_member, 0, paid=True)
    if amount <= 0:
        _masterclass_send_confirmation(ev, email, name)
        return jsonify({"ok": True, "registered": True, "join_url": ev.get("zoom_join_url")})
    # paid non-member
    if not _STRIPE_ACTIVE:
        return jsonify({"error": "payment_unavailable"}), 503
    from dashboard import stripe_pay as _sp
    sess = _sp.create_checkout_session(
        amount, customer_email=email, description=f"MasterClass: {ev['topic']}",
        metadata={"kind": "masterclass", "event_id": str(event_id), "email": email, "name": name},
        success_url=f"{PUBLIC_BASE_URL}/masterclass/{event_id}?paid=1",
        cancel_url=f"{PUBLIC_BASE_URL}/masterclass/{event_id}")
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _mc.init_masterclass_tables(cx)
        _mc.register(cx, event_id, email, name, is_member, amount, paid=False)
    return jsonify({"ok": True, "checkout_url": sess.get("url")})
```

- [ ] **Step 4: Run to verify pass**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/dc-test python3 -m pytest tests/test_masterclass_api.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_masterclass_api.py
git commit -m "feat(masterclass): event page route + public get + member-tiered register"
```

---

### Task 4: Stripe fulfillment (`_fulfill_masterclass` + webhook wiring)

**Files:** Modify `app.py` (`_fulfill_masterclass` + the `/webhook/stripe` dispatch at ~app.py:19678); Test `tests/test_masterclass_api.py`

**Interfaces:**
- Consumes: `dashboard.stripe_pay.get_session`/`get_payment_intent`, `dashboard.masterclass.mark_paid`/`get_event`, `_masterclass_send_confirmation`.
- Produces: `_fulfill_masterclass(session_id) -> dict` — re-fetches the session; no-op unless `metadata["kind"]=="masterclass"`; reads `event_id`/`email`; verifies `get_payment_intent(pi).status=="succeeded"`; `mark_paid`; sends the confirmation. Best-effort (never raises). Wired into `/webhook/stripe` after the existing `_fulfill_*` calls.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_masterclass_api.py
def test_fulfill_masterclass_marks_paid_and_sends(client, monkeypatch):
    sent = []
    monkeypatch.setattr(appmod, "send_evox_email", lambda to, *a, **k: sent.append(to) or ("console-log", None), raising=False)
    eid = _mk_event(client, price=5000, mprice=0)
    import sqlite3
    from dashboard import masterclass as mc
    with sqlite3.connect(appmod.LOG_DB) as cx:
        mc.register(cx, eid, "buyer@x.com", "B", is_member=False, amount_cents=5000, paid=False)
        cx.commit()
    import dashboard.stripe_pay as _sp
    monkeypatch.setattr(_sp, "get_session",
                        lambda sid: {"metadata": {"kind": "masterclass", "event_id": str(eid), "email": "buyer@x.com", "name": "B"},
                                     "payment_intent": "pi_1"})
    monkeypatch.setattr(_sp, "get_payment_intent", lambda pi: {"status": "succeeded"})
    out = appmod._fulfill_masterclass("cs_test")
    assert out["ok"] is True
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert mc.is_registered(cx, eid, "buyer@x.com") is True
    assert "buyer@x.com" in sent
    # non-masterclass session is a no-op
    monkeypatch.setattr(_sp, "get_session", lambda sid: {"metadata": {"kind": "retail"}})
    assert appmod._fulfill_masterclass("cs_other")["ok"] is False
```

- [ ] **Step 2: Run to verify failure**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/dc-test python3 -m pytest tests/test_masterclass_api.py::test_fulfill_masterclass_marks_paid_and_sends -q`
Expected: FAIL (`AttributeError: _fulfill_masterclass`).

- [ ] **Step 3: Add `_fulfill_masterclass` + wire the webhook**

```python
def _fulfill_masterclass(session_id):
    try:
        from dashboard import stripe_pay as _sp, masterclass as _mc
        sess = _sp.get_session(session_id)
        md = sess.get("metadata") or {}
        if md.get("kind") != "masterclass":
            return {"ok": False, "reason": "not_masterclass"}
        email = (md.get("email") or "").strip().lower()
        try:
            event_id = int(md.get("event_id"))
        except Exception:
            return {"ok": False, "reason": "no_event"}
        pi_id = sess.get("payment_intent")
        if not (email and pi_id):
            return {"ok": False, "reason": "incomplete"}
        if _sp.get_payment_intent(pi_id).get("status") != "succeeded":
            return {"ok": False, "reason": "unpaid"}
        with _db_lock, sqlite3.connect(LOG_DB) as cx:
            cx.row_factory = sqlite3.Row
            _mc.init_masterclass_tables(cx)
            _mc.mark_paid(cx, event_id, email)
            ev = _mc.get_event(cx, event_id)
        _masterclass_send_confirmation(ev, email, md.get("name") or "")
        return {"ok": True, "email": email, "event_id": event_id}
    except Exception as e:
        print(f"[masterclass] fulfill failed: {e!r}", flush=True)
        return {"ok": False, "reason": "error"}
```

In `/webhook/stripe` (`app.py:~19678`), after `_fulfill_continuous_care_monthly(session_id)`, add:
```python
                _fulfill_masterclass(session_id)
```

- [ ] **Step 4: Run to verify pass**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/dc-test python3 -m pytest tests/test_masterclass_api.py -q`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_masterclass_api.py
git commit -m "feat(masterclass): Stripe fulfillment marks paid + sends join link"
```

---

### Task 5: `static/masterclass.html` event page

**Files:** Create `static/masterclass.html`; Test: `node --check` + route-serves assertion

**Interfaces:** Consumes `GET /api/masterclass/<id>`, `POST /api/masterclass/<id>/register`. Event id from `location.pathname` (`/masterclass/<id>`).

- [ ] **Step 1: Create the page**

Vanilla JS, inline CSS/JS, no external assets (brand green `#2f6f5e`). Views: Loading → Event (topic, date/time, price line, name+email register form) → Registered (free/member: show the join link; paid: "redirecting to secure checkout"). On load, parse the id from `location.pathname`, fetch `/api/masterclass/<id>`; render topic/time and a price line ("Free with membership, $50 otherwise" style from `price_cents`/`member_price_cents`). Register posts `{email, name}`: if `{registered, join_url}` show the join link; if `{checkout_url}` `window.location = checkout_url`. Copy: no em dashes, no ALL CAPS.

```html
<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MasterClass · Healing Oasis</title>
<style>
 body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:640px;margin:0 auto;padding:24px;color:#1b2a26;background:#f6f8f7}
 h1{color:#2f6f5e} button{background:#2f6f5e;color:#fff;border:0;border-radius:8px;padding:10px 16px;font-size:15px;cursor:pointer}
 input{display:block;width:100%;box-sizing:border-box;padding:10px;margin:6px 0;border:1px solid #d7e0dc;border-radius:8px;font-size:15px}
 .muted{color:#5a6b64} .hidden{display:none}
</style></head><body>
<div id="v-load"><p class="muted">Loading...</p></div>
<div id="v-event" class="hidden">
  <h1 id="topic"></h1><p id="when" class="muted"></p><p id="price"></p><p id="desc"></p>
  <input id="f-name" placeholder="Your name">
  <input id="f-email" placeholder="you@email.com" type="email">
  <p id="err" class="muted"></p>
  <button onclick="register()">Register</button>
</div>
<div id="v-done" class="hidden"><h1>You are registered</h1><p id="done-msg"></p>
  <p id="join-wrap" class="hidden">Join link: <a id="join-link" target="_blank" rel="noopener"></a></p></div>
<script>
const ID = (location.pathname.split("/masterclass/")[1]||"").split(/[?#]/)[0];
const $=id=>document.getElementById(id), show=v=>{for(const x of ["load","event","done"])$("v-"+x).classList.toggle("hidden",x!==v)};
function money(c){return "$"+(c/100).toFixed(2);}
async function load(){
  const r=await fetch(`/api/masterclass/${encodeURIComponent(ID)}`); if(!r.ok){$("v-load").innerHTML='<p class="muted">This class was not found.</p>';return;}
  const e=await r.json();
  $("topic").textContent=e.topic; $("when").textContent=(e.start_ts||"").replace("T"," ")+" HST · "+e.duration_min+" min";
  $("desc").textContent=e.description||"";
  $("price").textContent = e.price_cents>0
    ? (e.member_price_cents>0 ? (money(e.member_price_cents)+" for members, "+money(e.price_cents)+" otherwise")
                              : ("Free with membership, "+money(e.price_cents)+" otherwise"))
    : "Free";
  show("event");
}
async function register(){
  const email=($("f-email").value||"").trim(), name=($("f-name").value||"").trim();
  if(email.indexOf("@")<0){$("err").textContent="Please enter your email.";return;}
  const r=await fetch(`/api/masterclass/${encodeURIComponent(ID)}/register`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({email,name})});
  const d=await r.json();
  if(d.checkout_url){ window.location=d.checkout_url; return; }
  if(d.registered){
    $("done-msg").textContent="Check your email for the calendar invite and join link.";
    if(d.join_url){ $("join-link").href=d.join_url; $("join-link").textContent=d.join_url; $("join-wrap").classList.remove("hidden"); }
    show("done"); return;
  }
  $("err").textContent = (d.error==="payment_unavailable") ? "Registration is not open yet. Please try again soon." : "Something went wrong. Please try again.";
}
load();
</script></body></html>
```

- [ ] **Step 2: Verify JS parses + route serves**

Extract the `<script>` and `node --check` it (clean up temp file). Add to `tests/test_masterclass_api.py`:
```python
def test_masterclass_page_served(client):
    r = client.get("/masterclass/1")
    assert r.status_code == 200 and b"MasterClass" in r.data
```
Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/dc-test python3 -m pytest tests/test_masterclass_api.py -q` → PASS.

- [ ] **Step 3: Commit**

```bash
git add static/masterclass.html tests/test_masterclass_api.py
git commit -m "feat(masterclass): public event + registration page"
```

---

## Post-implementation
- **Enable the Zoom S2S app** (marketplace.zoom.us, `meeting:write`) for auto meeting creation; else use the manual zoom-url endpoint. Ensure `STRIPE_ACTIVE` + `STRIPE_SECRET_KEY` + `STRIPE_WEBHOOK_SECRET` are set (they are, for the existing paid flows). Render-verify a live `/masterclass/<id>` page.

## Self-Review
- Event + registration store → Task 1. ✓
- Console create + Zoom Meeting (waiting_room=False) + calendar row + manual fallback → Task 2. ✓
- Public event page route + GET + member-tiered register (free/member immediate, non-member Stripe) → Task 3. ✓
- Stripe fulfillment marks paid + sends join → Task 4. ✓
- Event page → Task 5. ✓
- Member-tiered pricing via `_is_paid_member` + `price_for` → Tasks 1/3. ✓
- Deferred (announcements, capacity, edit, recording) → not built. ✓
- No placeholders. Names consistent: `create_event`/`get_event`/`set_zoom`/`price_for`/`register`/`mark_paid`/`is_registered`/`_masterclass_send_confirmation`/`_fulfill_masterclass`; metadata `kind='masterclass'`; `create_meeting(..., waiting_room=)`.
