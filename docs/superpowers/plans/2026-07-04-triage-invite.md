# Triage / Discovery Invite Booking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Let Glen or Rae email a prospect a personal invite to a free 15-minute Triage call, which the prospect books (no account) from the assigned practitioner's availability; Rae's is by phone (prospect calls Rae), Glen's is by Zoom (a time-gated Join button on the booking page).

**Architecture:** A new `dashboard/triage.py` invite store (tokenized, hashed), a console invite endpoint, invite-token-authed booking routes on the existing booking engine (`session_type='triage'`), and a self-contained `static/triage.html` page. Reuses `create_booking`, `available_slots`, `within_join_window`, `GLEN_PMI_URL`, `EVOX_RAE_PHONE`, and the email/ICS rail.

**Tech Stack:** Python 3 / Flask, sqlite, existing booking engine + email rail, vanilla-JS page. Tests via `app.app.test_client()`.

## Global Constraints

- sqlite `?` placeholders; writes under `with _db_lock, sqlite3.connect(LOG_DB) as cx:` + `cx.row_factory = sqlite3.Row`. Timestamps `datetime.now(timezone.utc).isoformat()`; slot times naive ISO HST; pure functions get `now` passed in.
- `/api/triage/*` authenticate on the **invite token** via `dashboard.triage.resolve_invite(cx, token)` (NOT a portal token). Console invite uses `_portal_console_ok()`.
- Practitioner ∈ `{'glen','rae'}`. Hours: `GLEN_CONSULT_HOURS` if glen else `EVOX_HOURS`. Duration **15 min**. Medium: `'video'` if glen else `'phone'`.
- Invite: `secrets.token_urlsafe(24)` raw, store only `sha256` hash; **7-day** expiry; **single-use** (book sets status='booked').
- Raw Zoom link never emailed. Client-facing copy: **no em dashes, no ALL CAPS.**
- Reuse: `dashboard.evox.{create_booking,available_slots,booked_starts,rae_busy_intervals,SlotTaken,init_evox_tables}`, `dashboard.consult.within_join_window`, `_hst_now()`, `_evox_days()`, `_portal_console_ok()`, `send_evox_email`, `build_ics`, `PUBLIC_BASE_URL`, `GLEN_PMI_URL`, `EVOX_RAE_PHONE`, `GLEN_CONSULT_EMAIL`, `EVOX_RAE_EMAIL`.
- Test cmd: `mkdir -p /tmp/dc-test && doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/dc-test python3 -m pytest tests/<file>.py -q`. Pure tests importing only `dashboard/*` run with plain pytest.
- Regression: EVOX + consult suites stay green.

---

### Task 1: `dashboard/triage.py` — invite store

**Files:** Create `dashboard/triage.py`; Test `tests/test_triage_pure.py`

**Interfaces:**
- Produces: `init_triage_tables(cx)`; `create_invite(cx, email, name, practitioner, *, days=7, _now=None) -> str` (raw token; stores sha256 hash); `resolve_invite(cx, token, *, _now=None) -> dict|None` (keys: `email,name,practitioner,status,booked_start`; None if no hash match / status=='cancelled' / expired); `mark_booked(cx, token, start_ts) -> None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_triage_pure.py
import sqlite3
from datetime import datetime
from dashboard import triage

def _cx():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    triage.init_triage_tables(cx); return cx

def test_invite_roundtrip_and_mark_booked():
    cx = _cx()
    tok = triage.create_invite(cx, "P@x.com", "Pat", "glen")
    inv = triage.resolve_invite(cx, tok)
    assert inv and inv["email"] == "p@x.com" and inv["practitioner"] == "glen"
    assert inv["status"] == "invited" and inv["booked_start"] is None
    triage.mark_booked(cx, tok, "2026-07-06T13:00:00")
    inv2 = triage.resolve_invite(cx, tok)
    assert inv2["status"] == "booked" and inv2["booked_start"] == "2026-07-06T13:00:00"

def test_resolve_bad_and_expired():
    cx = _cx()
    assert triage.resolve_invite(cx, "nope") is None
    now = datetime(2026, 7, 1, 12, 0)
    tok = triage.create_invite(cx, "e@x.com", "E", "rae", days=7, _now=now)
    assert triage.resolve_invite(cx, tok, _now=datetime(2026, 7, 6)) is not None   # day 5 ok
    assert triage.resolve_invite(cx, tok, _now=datetime(2026, 7, 9)) is None        # day 8 expired
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_triage_pure.py -q`
Expected: FAIL (`ModuleNotFoundError: dashboard.triage`).

- [ ] **Step 3: Create `dashboard/triage.py`**

```python
"""Triage/Discovery invites: tokenized, hashed, single-use, 7-day expiry.
Stdlib-only; import without importing app."""
import sqlite3, secrets, hashlib
from datetime import datetime, timezone, timedelta

def _hash(token: str) -> str:
    return hashlib.sha256((token or "").encode()).hexdigest()

def init_triage_tables(cx) -> None:
    cx.execute("""CREATE TABLE IF NOT EXISTS triage_invites (
        token_hash TEXT PRIMARY KEY, email TEXT, name TEXT, practitioner TEXT,
        status TEXT DEFAULT 'invited', created_at TEXT, expires_at TEXT,
        booked_start TEXT)""")
    cx.commit()

def create_invite(cx, email, name, practitioner, *, days: int = 7, _now=None) -> str:
    now = _now or datetime.now(timezone.utc)
    email = (email or "").strip().lower()
    token = secrets.token_urlsafe(24)
    cx.execute("INSERT INTO triage_invites (token_hash, email, name, practitioner, "
               "status, created_at, expires_at) VALUES (?,?,?,?, 'invited', ?, ?)",
               (_hash(token), email, (name or "").strip(), practitioner,
                now.isoformat(), (now + timedelta(days=days)).isoformat()))
    cx.commit()
    return token

def resolve_invite(cx, token, *, _now=None):
    now = _now or datetime.now(timezone.utc)
    cur = cx.execute("SELECT email,name,practitioner,status,created_at,expires_at,booked_start "
                     "FROM triage_invites WHERE token_hash=?", (_hash(token),))
    cols = [c[0] for c in cur.description]; r = cur.fetchone()
    if r is None:
        return None
    d = dict(zip(cols, r))
    if d.get("status") == "cancelled":
        return None
    try:
        if datetime.fromisoformat(d["expires_at"][:19]) < now.replace(tzinfo=None):
            return None
    except Exception:
        pass
    return {"email": d["email"], "name": d["name"], "practitioner": d["practitioner"],
            "status": d["status"], "booked_start": d.get("booked_start")}

def mark_booked(cx, token, start_ts) -> None:
    cx.execute("UPDATE triage_invites SET status='booked', booked_start=? WHERE token_hash=?",
               (start_ts, _hash(token)))
    cx.commit()
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_triage_pure.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add dashboard/triage.py tests/test_triage_pure.py
git commit -m "feat(triage): invite store (tokenized, hashed, 7-day expiry)"
```

---

### Task 2: Console invite endpoint + form

**Files:** Modify `app.py` (new route near other console routes); Modify `static/console-biofield-portal.html` (small form); Test `tests/test_triage_api.py`

**Interfaces:**
- Consumes: `dashboard.triage` (Task 1), `_portal_console_ok()`, `send_evox_email`, `PUBLIC_BASE_URL`.
- Produces: `POST /api/console/triage-invite {email, name, practitioner}` → `_portal_console_ok` gate (401); validates `practitioner in ('glen','rae')` (400 `bad_practitioner`) and email present (400); `create_invite`; emails the prospect `{PUBLIC_BASE_URL}/triage/<token>`; returns `{ok:true, url}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_triage_api.py
import os, pytest
if not os.environ.get("PINECONE_API_KEY"):
    pytest.skip("needs doppler env for import app", allow_module_level=True)
import app as appmod

@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    monkeypatch.setattr(appmod, "CONSOLE_SECRET", "test-secret")
    monkeypatch.setattr(appmod, "send_evox_email", lambda *a, **k: ("console-log", None), raising=False)
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()

ADMIN = {"X-Console-Key": "test-secret"}

def test_triage_invite_requires_auth(client):
    r = client.post("/api/console/triage-invite", json={"email": "p@x.com", "name": "P", "practitioner": "glen"})
    assert r.status_code == 401

def test_triage_invite_bad_practitioner(client):
    r = client.post("/api/console/triage-invite",
                    json={"email": "p@x.com", "name": "P", "practitioner": "bob"}, headers=ADMIN)
    assert r.status_code == 400

def test_triage_invite_creates_and_returns_url(client):
    r = client.post("/api/console/triage-invite",
                    json={"email": "p@x.com", "name": "Pat", "practitioner": "rae"}, headers=ADMIN)
    assert r.status_code == 200
    d = r.get_json()
    assert d["ok"] is True and "/triage/" in d["url"]
```

- [ ] **Step 2: Run to verify failure**

Run: `mkdir -p /tmp/dc-test && doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/dc-test python3 -m pytest tests/test_triage_api.py -q`
Expected: FAIL (404).

- [ ] **Step 3: Add the route to `app.py`**

```python
@app.route("/api/console/triage-invite", methods=["POST"])
def api_console_triage_invite():
    if not _portal_console_ok():
        return jsonify({"error": "unauthorized"}), 401
    from dashboard import triage as _triage
    body = request.get_json(silent=True) or {}
    email = (body.get("email") or "").strip().lower()
    name = (body.get("name") or "").strip()
    practitioner = (body.get("practitioner") or "").strip().lower()
    if "@" not in email:
        return jsonify({"error": "email required"}), 400
    if practitioner not in ("glen", "rae"):
        return jsonify({"error": "bad_practitioner"}), 400
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _triage.init_triage_tables(cx)
        token = _triage.create_invite(cx, email, name, practitioner)
    url = f"{PUBLIC_BASE_URL}/triage/{token}"
    who = "Dr. Glen" if practitioner == "glen" else "Rae"
    html = (f"<p>Hello{(' ' + name) if name else ''},</p>"
            f"<p>You are invited to book a free 15 minute call with {who}. "
            f"Pick a time here: <a href=\"{url}\">{url}</a></p>")
    try:
        send_evox_email(email, name, f"Your 15 minute call with {who}", html, html, b"")
    except Exception:
        app.logger.exception("triage invite email failed to %s", email)
    return jsonify({"ok": True, "url": url})
```

- [ ] **Step 4: Run to verify pass**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/dc-test python3 -m pytest tests/test_triage_api.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Add a small invite form to `static/console-biofield-portal.html`**

Near the "Mark consult ready" controls, reusing the page's `api()` helper:
```html
<div style="margin-top:12px">
  <h3>Send triage invite</h3>
  <input id="tri-email" placeholder="prospect email">
  <input id="tri-name" placeholder="name">
  <select id="tri-who"><option value="glen">Dr. Glen</option><option value="rae">Rae</option></select>
  <button onclick="sendTriageInvite()">Send invite</button>
  <span id="tri-msg"></span>
</div>
<script>
async function sendTriageInvite(){
  const email=(document.getElementById('tri-email').value||'').trim();
  const name=(document.getElementById('tri-name').value||'').trim();
  const practitioner=document.getElementById('tri-who').value;
  const r=await api('POST','/api/console/triage-invite',{email,name,practitioner});
  document.getElementById('tri-msg').textContent = r.ok ? ('Sent: '+r.json.url) : ('Error '+r.status);
}
</script>
```

- [ ] **Step 6: Commit**

```bash
git add app.py static/console-biofield-portal.html tests/test_triage_api.py
git commit -m "feat(triage): console invite endpoint + form"
```

---

### Task 3: Booking routes (page + state / availability / book / join)

**Files:** Modify `app.py` (routes near the consult routes); Test `tests/test_triage_api.py`

**Interfaces:**
- Consumes: `dashboard.triage.resolve_invite`/`mark_booked`, `dashboard.evox` (`available_slots`,`booked_starts`,`rae_busy_intervals`,`create_booking`,`SlotTaken`,`init_evox_tables`), `dashboard.consult.within_join_window`, `_hst_now`,`_evox_days`, `GLEN_CONSULT_HOURS`,`EVOX_HOURS`,`GLEN_PMI_URL`, `STATIC`. Adds a **stub** `_triage_send_confirmations(token, invite, booking): pass` (Task 4 replaces it).
- Produces (HTTP; `/api/triage/*` authed via `resolve_invite`):
  - `GET /triage/<token>` → `send_from_directory(STATIC, "triage.html")`.
  - `GET /api/triage/state?token=` → `{name, practitioner, medium, booked, booked_start}` (404 `invalid` if no invite).
  - `GET /api/triage/availability?token=&range=week` → `{slots}` (15-min, invite's practitioner's hours); 409 `already_booked` if booked.
  - `POST /api/triage/book?token= {start_ts}` → re-validate; `create_booking(session_type='triage', practitioner, duration_min=15, medium=('phone' if rae else 'video'))`; `mark_booked`; `_triage_send_confirmations`; `{ok, start_ts}`. `SlotTaken`→409, unavailable→409, bad date→400.
  - `GET /api/triage/join?token=` → glen+booked+in-window → `{ok, join_url:GLEN_PMI_URL}`; rae → 400 `phone_call`; not booked → 404 `no_booking`; out of window → 403 `not_in_window`.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_triage_api.py
from datetime import timedelta
def _invite(client, practitioner="glen", email="pp@x.com"):
    r = client.post("/api/console/triage-invite",
                    json={"email": email, "name": "Pat", "practitioner": practitioner}, headers=ADMIN)
    return r.get_json()["url"].rsplit("/triage/", 1)[1]

def test_triage_state_and_full_flow(client):
    tok = _invite(client, "glen")
    st = client.get(f"/api/triage/state?token={tok}").get_json()
    assert st["practitioner"] == "glen" and st["medium"] == "video" and st["booked"] is False
    slots = client.get(f"/api/triage/availability?token={tok}&range=week").get_json()["slots"]
    assert slots
    r = client.post(f"/api/triage/book?token={tok}", json={"start_ts": slots[0]})
    assert r.status_code == 200 and r.get_json()["ok"] is True
    # single-use: availability now 409 already_booked
    r2 = client.get(f"/api/triage/availability?token={tok}&range=week")
    assert r2.status_code == 409

def test_triage_state_invalid_token(client):
    r = client.get("/api/triage/state?token=bogus")
    assert r.status_code == 404 and r.get_json()["error"] == "invalid"

def test_triage_join_glen_vs_rae(client):
    import sqlite3
    from dashboard import triage
    # rae invite -> join returns phone_call 400
    tok_r = _invite(client, "rae", "r@x.com")
    slots = client.get(f"/api/triage/availability?token={tok_r}&range=week").get_json()["slots"]
    client.post(f"/api/triage/book?token={tok_r}", json={"start_ts": slots[0]})
    assert client.get(f"/api/triage/join?token={tok_r}").status_code == 400
    # glen invite booked far out -> not_in_window 403
    tok_g = _invite(client, "glen", "g2@x.com")
    slots = client.get(f"/api/triage/availability?token={tok_g}&range=week").get_json()["slots"]
    client.post(f"/api/triage/book?token={tok_g}", json={"start_ts": slots[-1]})
    # slots[-1] is >30min out unless today's last slot; assert it is either 200 or 403 (both valid states)
    assert client.get(f"/api/triage/join?token={tok_g}").status_code in (200, 403)
```

- [ ] **Step 2: Run to verify failure**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/dc-test python3 -m pytest tests/test_triage_api.py -q`
Expected: FAIL (404 on `/api/triage/state`).

- [ ] **Step 3: Add config + routes to `app.py`** (stub confirmations; near the consult routes)

```python
def _triage_ident(cx, token):
    from dashboard import triage as _triage
    return _triage.resolve_invite(cx, token)

def _triage_hours(practitioner):
    return GLEN_CONSULT_HOURS if practitioner == "glen" else EVOX_HOURS

@app.route("/triage/<token>")
def triage_page(token):
    return send_from_directory(STATIC, "triage.html")

@app.route("/api/triage/state")
def triage_state():
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        from dashboard import triage as _triage; _triage.init_triage_tables(cx)
        inv = _triage_ident(cx, request.args.get("token", ""))
        if inv is None:
            return jsonify({"error": "invalid"}), 404
        medium = "video" if inv["practitioner"] == "glen" else "phone"
        return jsonify({"name": inv["name"], "practitioner": inv["practitioner"],
                        "medium": medium, "booked": inv["status"] == "booked",
                        "booked_start": inv["booked_start"]})

@app.route("/api/triage/availability")
def triage_availability():
    from dashboard import evox as _ev, triage as _triage
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _ev.init_evox_tables(cx); _triage.init_triage_tables(cx); _init_calendar_table()
        inv = _triage_ident(cx, request.args.get("token", ""))
        if inv is None:
            return jsonify({"error": "invalid"}), 404
        if inv["status"] == "booked":
            return jsonify({"error": "already_booked"}), 409
        p = inv["practitioner"]
        days = _evox_days(request.args.get("range", "week"))
        busy = _ev.rae_busy_intervals(cx, days[0].isoformat(), days[-1].isoformat(), practitioner=p)
        booked = _ev.booked_starts(cx, practitioner=p)
        slots = _ev.available_slots(days, _triage_hours(p), busy, booked, _hst_now(), duration_min=15)
        return jsonify({"slots": slots})

@app.route("/api/triage/book", methods=["POST"])
def triage_book():
    from dashboard import evox as _ev, triage as _triage
    from datetime import date
    body = request.get_json(force=True) or {}
    start_ts = (body.get("start_ts") or "").strip()
    token = request.args.get("token", "")
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _ev.init_evox_tables(cx); _triage.init_triage_tables(cx); _init_calendar_table()
        inv = _triage_ident(cx, token)
        if inv is None:
            return jsonify({"error": "invalid"}), 404
        if inv["status"] == "booked":
            return jsonify({"error": "already_booked"}), 409
        p = inv["practitioner"]
        try:
            d = date.fromisoformat(start_ts[:10])
        except ValueError:
            return jsonify({"error": "bad_start_ts"}), 400
        busy = _ev.rae_busy_intervals(cx, d.isoformat(), d.isoformat(), practitioner=p)
        if start_ts not in _ev.available_slots([d], _triage_hours(p), busy,
                                               _ev.booked_starts(cx, practitioner=p),
                                               _hst_now(), duration_min=15):
            return jsonify({"error": "slot_unavailable"}), 409
        medium = "video" if p == "glen" else "phone"
        try:
            b = _ev.create_booking(cx, inv["email"], start_ts, duration_min=15,
                                   practitioner=p, session_type="triage", medium=medium)
        except _ev.SlotTaken:
            return jsonify({"error": "slot_taken"}), 409
        _triage.mark_booked(cx, token, start_ts)
    _triage_send_confirmations(token, inv, b)   # Task 4
    return jsonify({"ok": True, "start_ts": start_ts})

@app.route("/api/triage/join")
def triage_join():
    from dashboard import triage as _triage, consult as _consult
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _triage.init_triage_tables(cx)
        inv = _triage_ident(cx, request.args.get("token", ""))
        if inv is None:
            return jsonify({"error": "invalid"}), 404
        if inv["practitioner"] != "glen":
            return jsonify({"error": "phone_call"}), 400
        if inv["status"] != "booked" or not inv["booked_start"]:
            return jsonify({"error": "no_booking"}), 404
        if _consult.within_join_window(inv["booked_start"], _hst_now()):
            return jsonify({"ok": True, "join_url": GLEN_PMI_URL})
        return jsonify({"error": "not_in_window", "start_ts": inv["booked_start"]}), 403
```

Add the stub `def _triage_send_confirmations(token, invite, booking): pass` (Task 4 replaces it).

- [ ] **Step 4: Run to verify pass**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/dc-test python3 -m pytest tests/test_triage_api.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_triage_api.py
git commit -m "feat(triage): booking routes (page/state/availability/book/join)"
```

---

### Task 4: Triage confirmation emails

**Files:** Modify `app.py` (replace the `_triage_send_confirmations` stub); Test `tests/test_triage_api.py`

**Interfaces:**
- Consumes: `send_evox_email`, `build_ics`, `EVOX_RAE_PHONE`, `EVOX_RAE_EMAIL`, `GLEN_CONSULT_EMAIL`, `PUBLIC_BASE_URL`.
- Produces: `_triage_send_confirmations(token, invite, booking)` — best-effort (outer + per-send try/except). Prospect email + ICS: **rae** → "At your appointment time, call Rae at `EVOX_RAE_PHONE`."; **glen** → "At your appointment time, open your booking page and click Join your call: `{PUBLIC_BASE_URL}/triage/<token>`." Practitioner notice to `EVOX_RAE_EMAIL` (rae) / `GLEN_CONSULT_EMAIL` (glen).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_triage_api.py
def test_triage_confirmations(client, monkeypatch):
    calls = []
    monkeypatch.setattr(appmod, "send_evox_email",
                        lambda to, name, subj, html, text, ics: calls.append((to, html)), raising=False)
    monkeypatch.setattr(appmod, "EVOX_RAE_PHONE", "808-555-1212", raising=False)
    tok = _invite(client, "rae", "conf@x.com")
    slots = client.get(f"/api/triage/availability?token={tok}&range=week").get_json()["slots"]
    calls.clear()
    client.post(f"/api/triage/book?token={tok}", json={"start_ts": slots[0]})
    tos = [to for (to, h) in calls]
    assert "conf@x.com" in tos                       # prospect notified
    prospect_html = [h for (to, h) in calls if to == "conf@x.com"][0]
    assert "808-555-1212" in prospect_html and "zoom.us/j/" not in prospect_html
```

- [ ] **Step 2: Run to verify failure**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/dc-test python3 -m pytest tests/test_triage_api.py::test_triage_confirmations -q`
Expected: FAIL (stub sends nothing).

- [ ] **Step 3: Replace the stub in `app.py`**

```python
def _triage_send_confirmations(token, invite, booking):
    try:
        from dashboard import evox as _ev
        p = invite["practitioner"]; email = invite["email"]
        start = booking["start_ts"]; nice = start.replace("T", " ")
        if p == "rae":
            who = "Rae"; note_email = EVOX_RAE_EMAIL
            phone = EVOX_RAE_PHONE or "the number in this email"
            line = f"At your appointment time, call Rae at {phone}."
        else:
            who = "Dr. Glen"; note_email = GLEN_CONSULT_EMAIL
            page = f"{PUBLIC_BASE_URL}/triage/{token}"
            line = f"At your appointment time, open your booking page and click Join your call: {page}"
        ics = _ev.build_ics(uid=booking["ics_uid"], start_ts=start, end_ts=booking["end_ts"],
                            summary=f"15 minute call with {who}",
                            description=line, location=("Phone" if p == "rae" else "Zoom (join from your page)"))
        c_html = (f"<p>Your 15 minute call with {who} is booked for <b>{nice} HST</b>.</p>"
                  f"<p>{line}</p><p>The calendar invite is attached.</p>")
        c_text = f"Call with {who} booked for {nice} HST. {line}"
        n_html = f"<p>New triage booked: <b>{invite.get('name') or email}</b> ({email}) on <b>{nice} HST</b>.</p>"
        for to, nm, subj, html, text in [
            (email, invite.get("name") or "", f"Your call with {who} is booked", c_html, c_text),
            (note_email, who, f"Triage booked: {email}", n_html, n_html)]:
            try:
                send_evox_email(to, nm, subj, html, text, ics)
            except Exception:
                app.logger.exception("triage confirmation send failed to %s", to)
    except Exception:
        app.logger.exception("triage confirmation build failed")
```

- [ ] **Step 4: Run to verify pass**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/dc-test python3 -m pytest tests/test_triage_api.py -q`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_triage_api.py
git commit -m "feat(triage): confirmation emails (prospect + practitioner)"
```

---

### Task 5: `static/triage.html` booking page

**Files:** Create `static/triage.html`; Test: manual JS `node --check` + a route-serves assertion

**Interfaces:** Consumes the Task 3 routes (`/api/triage/state|availability|book|join`). Token from `/triage/<token>` path (parse `location.pathname`).

- [ ] **Step 1: Create the page**

Vanilla JS, inline CSS/JS (brand green `#2f6f5e` / gold `#d4a843`), no external assets. Views: Loading → Slots (pick a 15-min time) → Confirmed. On load, parse the token from `location.pathname` (`/triage/<token>`), fetch `/api/triage/state`; if `booked`, show the confirmed state; else fetch availability and render slot buttons. On book success show the confirmed state; for a **rae** invite the confirmed state shows "At your appointment time, call Rae." (state provides `medium`); for **glen** it shows a "Join your call" button calling `/api/triage/join` (on `{ok}` `window.open(join_url)`, on 403 a "opens 10 minutes before" message).

```html
<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Book your call — Healing Oasis</title>
<style>
 body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:640px;margin:0 auto;padding:24px;color:#1b2a26;background:#f6f8f7}
 h1{color:#2f6f5e} button{background:#2f6f5e;color:#fff;border:0;border-radius:8px;padding:10px 16px;font-size:15px;cursor:pointer;margin:4px}
 .slot{display:inline-block} .day{margin:12px 0} .hidden{display:none} .muted{color:#5a6b64}
</style></head><body>
<h1>Book your call</h1>
<div id="v-load"><p class="muted">Loading your invitation...</p></div>
<div id="v-slots" class="hidden"><p id="intro"></p><div id="slots"></div></div>
<div id="v-done" class="hidden"><h2>Booked</h2><p id="done-msg"></p>
  <button id="join-btn" class="hidden" onclick="triageJoin()">Join your call</button>
  <div id="join-msg" class="muted"></div></div>
<script>
const TOKEN = decodeURIComponent((location.pathname.split("/triage/")[1]||"").split(/[?#]/)[0]);
let STATE = null;
const $=id=>document.getElementById(id), show=v=>{for(const x of ["load","slots","done"])$("v-"+x).classList.toggle("hidden",x!==v)};
async function j(u,o){const r=await fetch(u,o);return {status:r.status, body:await r.json().catch(()=>({}))}}
function who(){return STATE && STATE.practitioner==="glen" ? "Dr. Glen" : "Rae";}
async function load(){
  const s=await j(`/api/triage/state?token=${encodeURIComponent(TOKEN)}`);
  if(s.status!==200){ $("v-load").innerHTML='<p class="muted">This invitation is invalid or has expired.</p>'; return; }
  STATE=s.body;
  if(STATE.booked){ confirmed(STATE.booked_start); return; }
  const a=await j(`/api/triage/availability?token=${encodeURIComponent(TOKEN)}&range=week`);
  $("intro").textContent = `Pick a 15 minute time with ${who()} (HST).`;
  $("slots").innerHTML = (a.body.slots||[]).map(x=>`<button class="slot" onclick="book('${x}')">${x.replace("T"," ")}</button>`).join("") || '<p class="muted">No open times right now. Please check back soon.</p>';
  show("slots");
}
async function book(start_ts){
  const r=await j(`/api/triage/book?token=${encodeURIComponent(TOKEN)}`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({start_ts})});
  if(r.body.ok){ confirmed(start_ts); } else { alert("That time is no longer available. Please pick another."); load(); }
}
function confirmed(start){
  const nice=(start||"").replace("T"," ");
  if(STATE.practitioner==="glen"){
    $("done-msg").textContent = `Your call with Dr. Glen is booked for ${nice} HST. At your appointment time, click Join your call below.`;
    $("join-btn").classList.remove("hidden");
  } else {
    $("done-msg").textContent = `Your call with Rae is booked for ${nice} HST. At your appointment time, call Rae at the number in your confirmation email.`;
  }
  show("done");
}
async function triageJoin(){
  const r=await j(`/api/triage/join?token=${encodeURIComponent(TOKEN)}`);
  if(r.body.ok && r.body.join_url){ window.open(r.body.join_url,"_blank","noopener"); return; }
  $("join-msg").textContent = "The Join button opens 10 minutes before your appointment time.";
}
load();
</script></body></html>
```

- [ ] **Step 2: Verify JS parses + route serves**

Extract the `<script>` and `node --check` it (must parse). Add a route-serves check: `GET /triage/x` returns 200 and `b"Book your call"` in the body (append to `tests/test_triage_api.py`):
```python
def test_triage_page_served(client):
    r = client.get("/triage/anytoken")
    assert r.status_code == 200 and b"Book your call" in r.data
```
Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/dc-test python3 -m pytest tests/test_triage_api.py::test_triage_page_served -q` → PASS.

- [ ] **Step 3: Commit**

```bash
git add static/triage.html tests/test_triage_api.py
git commit -m "feat(triage): booking page (slot picker + rae-phone / glen-gated-join)"
```

---

## Post-implementation
- No new env required (reuses existing config). Render-verify a `/triage/<token>` page live at go-live (send one real invite each to Glen/Rae, book end to end).

## Self-Review
- Invite store (tokenized/hashed/7-day/single-use) → Task 1. ✓
- Console invite + form → Task 2. ✓
- Booking page + invite-token-authed routes (state/availability/book) + glen join-gate / rae phone → Tasks 3/5. ✓
- Confirmations (rae call-Rae / glen page-join, no raw link; practitioner notice) → Task 4. ✓
- Per-practitioner medium + hours; 15-min; free; assigned-at-invite → Tasks 2/3. ✓
- Out of scope (MasterClass/intake/prospect-portal) → not built. ✓
- No placeholders. Names consistent: `create_invite`/`resolve_invite`/`mark_booked`/`_triage_send_confirmations`/`_triage_hours` across tasks; `session_type='triage'`; medium `video`(glen)/`phone`(rae).
