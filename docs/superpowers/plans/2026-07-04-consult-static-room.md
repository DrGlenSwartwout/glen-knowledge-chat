# Consult Static Room + Portal-Gated Join — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Switch the Biofield Consult from a per-booking Zoom API meeting to Dr. Glen's static Personal Meeting Room, with access controlled by a server-gated, time-windowed "Join your consult" button in the client portal (not a raw emailed link). Keep the Zoom S2S app reserved for a later stage-9 recording fetch.

**Architecture (settled with Glen):** The consult uses one persistent room (`GLEN_PMI_URL`). The booking no longer calls the Zoom API. The confirmation email omits the raw Zoom link and points the client to their portal. A new `GET /api/consult/join` returns the room URL only if the authenticated client has a booked consult whose window contains "now" (10 min before to 30 after the start). Waiting Room + cloud auto-recording are already enabled on Glen's Zoom account. This also unblocks consults from the currently-disabled Zoom app, since booking no longer needs it.

**Tech Stack:** Python 3 / Flask, sqlite, existing `resolve_identity` portal auth + `send_evox_email` rail, vanilla-JS portal. Tests via `app.app.test_client()`.

## Global Constraints

- sqlite `?` placeholders; writes under `with _db_lock, sqlite3.connect(LOG_DB) as cx:` + `cx.row_factory = sqlite3.Row`. Timestamps `datetime.now(timezone.utc).isoformat()`; slot/window times are naive ISO HST; pure functions get `now` passed in.
- Consult APIs auth via portal token (`_evox_ident(cx, token)` → `ident.email`, 404 if None).
- **`GLEN_PMI_URL`** default `"https://zoom.us/j/9071793431"`. Join window: **10 min before to 30 min after** the start.
- **The raw Zoom link must NOT appear in the confirmation email** — the client reaches the room only through the portal-gated join endpoint.
- Client-facing copy: **no em dashes, no ALL CAPS.**
- Test command: `mkdir -p /tmp/dc-test && doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/dc-test python3 -m pytest tests/<file>.py -q`. Pure tests importing only `dashboard/*` run with plain `python3 -m pytest`.
- **Regression:** the consult booking/gate/portal must stay green; EVOX untouched.

---

### Task 1: `within_join_window` helper + `/api/consult/join` route

**Files:**
- Modify: `dashboard/consult.py` (add pure helper)
- Modify: `app.py` (add `GLEN_PMI_URL` config near `GLEN_ZOOM_USER` ~app.py:208; add the route near the other consult routes)
- Test: `tests/test_consult_pure.py` + `tests/test_consult_api.py`

**Interfaces:**
- Produces: `dashboard.consult.within_join_window(start_ts, now, before_min=10, after_min=30) -> bool` (pure; True iff `start - before_min <= now <= start + after_min`). Route `GET /api/consult/join?token=` → `{ok:true, join_url}` (200) if the client has a booked `biofield-consult` whose window contains `_hst_now()`, else `{error:"not_in_window", start_ts: <next booked consult start or null>}` (403), or `{error:"not_found"}` (404) if identity missing / `{error:"no_booking"}` (404) if no booked consult at all.

- [ ] **Step 1: Write the failing pure test**

```python
# append to tests/test_consult_pure.py
from datetime import datetime
def test_within_join_window():
    from dashboard import consult
    s = "2026-07-06T13:00:00"
    assert consult.within_join_window(s, datetime(2026,7,6,12,55)) is True   # 5 min before
    assert consult.within_join_window(s, datetime(2026,7,6,12,49)) is False  # 11 min before
    assert consult.within_join_window(s, datetime(2026,7,6,13,29)) is True   # 29 after
    assert consult.within_join_window(s, datetime(2026,7,6,13,31)) is False  # 31 after
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_consult_pure.py -q`
Expected: FAIL (`AttributeError: within_join_window`).

- [ ] **Step 3: Add the pure helper to `dashboard/consult.py`**

```python
from datetime import timedelta

def within_join_window(start_ts, now, before_min: int = 10, after_min: int = 30) -> bool:
    start = datetime.fromisoformat(str(start_ts)[:19])
    return (start - timedelta(minutes=before_min)) <= now <= (start + timedelta(minutes=after_min))
```
(`datetime` is already imported in consult.py; add `timedelta` to that import.)

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_consult_pure.py -q`
Expected: PASS.

- [ ] **Step 5: Add config + route to `app.py`**

Config near `GLEN_ZOOM_USER`:
```python
GLEN_PMI_URL = os.environ.get("GLEN_PMI_URL", "https://zoom.us/j/9071793431")
```

Route near the other `/api/consult/*` routes:
```python
@app.route("/api/consult/join")
def consult_join():
    from dashboard import consult as _consult
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        from dashboard import evox as _ev
        _ev.init_evox_tables(cx)
        ident = _evox_ident(cx, request.args.get("token", ""))
        if ident is None:
            return jsonify({"error": "not_found"}), 404
        rows = cx.execute("SELECT start_ts FROM evox_bookings WHERE lower(email)=? "
                          "AND session_type='biofield-consult' AND status='booked' "
                          "ORDER BY start_ts", (ident.email,)).fetchall()
        if not rows:
            return jsonify({"error": "no_booking"}), 404
        now = _hst_now()
        for r in rows:
            if _consult.within_join_window(r["start_ts"], now):
                return jsonify({"ok": True, "join_url": GLEN_PMI_URL})
        # not in any window: report the next upcoming consult start (if any)
        upcoming = [r["start_ts"] for r in rows if r["start_ts"] >= now.isoformat()]
        return jsonify({"error": "not_in_window",
                        "start_ts": (upcoming[0] if upcoming else rows[-1]["start_ts"])}), 403
```

- [ ] **Step 6: Write the failing API test**

```python
# append to tests/test_consult_api.py
def test_consult_join_gated_by_window(client, monkeypatch):
    import sqlite3
    from datetime import timedelta
    from dashboard import evox
    tok = _mk_portal("join@x.com")
    # a consult starting 5 min from now (inside the -10/+30 window)
    start = (appmod._hst_now() + timedelta(minutes=5)).replace(microsecond=0).isoformat()
    end = (appmod._hst_now() + timedelta(minutes=35)).replace(microsecond=0).isoformat()
    with sqlite3.connect(appmod.LOG_DB) as cx:
        evox.init_evox_tables(cx)
        cx.execute("INSERT INTO evox_bookings (email,practitioner,start_ts,end_ts,status,"
                   "session_type,medium) VALUES (?,?,?,?, 'booked', 'biofield-consult','video')",
                   ("join@x.com", "glen", start, end)); cx.commit()
    r = client.get(f"/api/consult/join?token={tok}")
    assert r.status_code == 200 and r.get_json()["join_url"] == appmod.GLEN_PMI_URL
    # move the booking 2 hours out -> outside the window -> 403
    with sqlite3.connect(appmod.LOG_DB) as cx:
        far = (appmod._hst_now() + timedelta(hours=2)).replace(microsecond=0).isoformat()
        cx.execute("UPDATE evox_bookings SET start_ts=? WHERE email='join@x.com'", (far,)); cx.commit()
    r2 = client.get(f"/api/consult/join?token={tok}")
    assert r2.status_code == 403 and r2.get_json()["error"] == "not_in_window"

def test_consult_join_no_booking(client):
    tok = _mk_portal("nobook@x.com")
    r = client.get(f"/api/consult/join?token={tok}")
    assert r.status_code == 404 and r.get_json()["error"] == "no_booking"
```

- [ ] **Step 7: Run both suites**

Run: `python3 -m pytest tests/test_consult_pure.py -q` (pass) and `doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/dc-test python3 -m pytest tests/test_consult_api.py -q` (pass).

- [ ] **Step 8: Commit**

```bash
git add dashboard/consult.py app.py tests/test_consult_pure.py tests/test_consult_api.py
git commit -m "feat(consult): static-room join gate — within_join_window + /api/consult/join"
```

---

### Task 2: Strip Zoom from booking + rework confirmation (no raw link)

**Files:**
- Modify: `app.py` (`consult_book` ~15440-15470 Zoom block; `_consult_send_confirmations` ~15355)
- Test: `tests/test_consult_api.py`

**Interfaces:**
- Consumes: `client_portal.ensure_token` (to build the client's portal URL), `PUBLIC_BASE_URL`.
- Produces: `consult_book` no longer calls `dashboard.zoom` (no `get_token`/`create_meeting`, no `zoom_join_url` write); returns `{ok:true, start_ts}`. `_consult_send_confirmations(email, booking)` sends confirmations with **no raw Zoom link**; the client email instructs "join from your portal at your appointment time" with a link to their portal (`booking["portal_url"]`); ICS location = "Zoom (join from your portal)".

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_consult_api.py
def test_booking_no_zoom_call_and_email_has_no_raw_link(client, monkeypatch):
    calls = []
    monkeypatch.setattr(appmod, "send_evox_email",
                        lambda to, name, subj, html, text, ics: calls.append((to, html)), raising=False)
    # if the code still calls Zoom, this makes it explode -> test would fail
    def _boom(*a, **k): raise AssertionError("Zoom must not be called at booking")
    monkeypatch.setattr("dashboard.zoom.get_token", _boom)
    monkeypatch.setattr("dashboard.zoom.create_meeting", _boom)
    tok = _mk_portal("nolink@x.com")
    client.post("/api/console/consult-ready", json={"email": "nolink@x.com", "ready": True}, headers=ADMIN)
    slots = client.get(f"/api/consult/availability?token={tok}&range=week").get_json()["slots"]
    r = client.post(f"/api/consult/book?token={tok}", json={"start_ts": slots[0]})
    assert r.status_code == 200 and r.get_json()["ok"] is True
    client_email = [h for (to, h) in calls if to == "nolink@x.com"][0]
    assert "zoom.us/j/" not in client_email          # no raw Zoom link in the email
    assert "portal" in client_email.lower()          # points them to the portal
```

- [ ] **Step 2: Run to verify failure**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/dc-test python3 -m pytest tests/test_consult_api.py::test_booking_no_zoom_call_and_email_has_no_raw_link -q`
Expected: FAIL (Zoom is still called → AssertionError, or the email still carries a link).

- [ ] **Step 3: Remove the Zoom block from `consult_book`**

Delete the best-effort Zoom section (the `join_url = None` / `try: tok = _zoom.get_token(...) ... create_meeting(...) ... UPDATE zoom_...` block that runs after the `with _db_lock` block). Replace the tail of `consult_book` so that, after the booking `with` block, it resolves the client's portal URL, attaches it to `b`, sends confirmations, and returns without a join_url:

```python
    # (after the `with _db_lock ...:` block that created the booking `b`)
    try:
        from dashboard import client_portal as _cp
        with sqlite3.connect(LOG_DB) as cx2:
            token = _cp.ensure_token(cx2, email, "") if hasattr(_cp, "ensure_token") else None
        b["portal_url"] = f"{PUBLIC_BASE_URL}/portal/{token}" if token else f"{PUBLIC_BASE_URL}/portal/login"
    except Exception:
        b["portal_url"] = f"{PUBLIC_BASE_URL}/portal/login"
    _consult_send_confirmations(email, b)
    return jsonify({"ok": True, "start_ts": start_ts})
```
Remove the now-unused `_zoom` import in `consult_book` if it is only used there. Do NOT change the readiness gate / re-validation / `create_booking` inside the lock.

- [ ] **Step 4: Rework `_consult_send_confirmations` (drop the raw link)**

```python
def _consult_send_confirmations(email, booking):
    """Best-effort: client + Glen Biofield Consult confirmations, ICS, and portal
    join instructions. No raw Zoom link (the client joins via the portal-gated
    button). Never raises into the booking response."""
    try:
        from dashboard import evox as _ev
        start = booking["start_ts"]; nice = start.replace("T", " ")
        portal = booking.get("portal_url") or ""
        join_line = ("At your appointment time, open your Healing Oasis portal and click "
                     "Join your consult" + (f": {portal}" if portal else "."))
        ics = _ev.build_ics(uid=booking["ics_uid"], start_ts=start, end_ts=booking["end_ts"],
                            summary="Biofield Consult with Dr. Glen",
                            description=join_line, location="Zoom (join from your portal)")
        client_html = (f"<p>Your Biofield Consult with Dr. Glen is booked for "
                       f"<b>{nice} HST</b>.</p><p>{join_line}</p>"
                       "<p>The calendar invite is attached.</p>")
        client_text = f"Biofield Consult booked for {nice} HST. {join_line}"
        glen_html = f"<p>New Biofield Consult: <b>{email}</b> on <b>{nice} HST</b>.</p>"
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

- [ ] **Step 5: Run to verify pass (+ full consult suite)**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/dc-test python3 -m pytest tests/test_consult_api.py -q`
Expected: PASS (all, including the updated full-flow test — note the older `test_full_consult_flow` asserted `join_url` in the response; if present, update that assertion to `r.get_json()["ok"] is True` since booking no longer returns a join_url).

- [ ] **Step 6: Commit**

```bash
git add app.py tests/test_consult_api.py
git commit -m "feat(consult): booking uses static room; drop Zoom API + raw link from email"
```

---

### Task 3: Portal card — appointment time + gated "Join your consult" button

**Files:**
- Modify: `dashboard/portal_view.py` (`_consult_block` — add `booked_start`)
- Modify: `static/client-portal.html` (booked-state card + join JS)
- Test: `tests/test_consult_api.py`

**Interfaces:**
- Produces: `_consult_block` return dict gains `"booked_start"` (the booked consult's `start_ts`, or `null`). The portal booked-state card shows the appointment time and a "Join your consult" button that calls `/api/consult/join`; on `{ok}` it opens `join_url` in a new tab, on 403 it shows "Your consult is at <time>. The Join button opens 10 minutes before."

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_consult_api.py
def test_view_consult_block_has_booked_start(client):
    import sqlite3
    from datetime import timedelta
    from dashboard import evox
    tok = _mk_portal("bs@x.com")
    start = (appmod._hst_now() + timedelta(days=1)).replace(microsecond=0).isoformat()
    with sqlite3.connect(appmod.LOG_DB) as cx:
        evox.init_evox_tables(cx)
        cx.execute("INSERT INTO evox_bookings (email,practitioner,start_ts,end_ts,status,"
                   "session_type,medium) VALUES (?,?,?,?, 'booked','biofield-consult','video')",
                   ("bs@x.com", "glen", start, start)); cx.commit()
    d = client.get(f"/api/portal/{tok}/view").get_json()
    assert d["consult"]["booked"] is True and d["consult"]["booked_start"] == start
```

- [ ] **Step 2: Run to verify failure**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/dc-test python3 -m pytest tests/test_consult_api.py::test_view_consult_block_has_booked_start -q`
Expected: FAIL (`KeyError: 'booked_start'`).

- [ ] **Step 3: Add `booked_start` to `_consult_block` in `portal_view.py`**

In the existing `_consult_block`, replace the booked lookup so it also returns the start:
```python
        booked_start = None
        try:
            row = cx.execute("SELECT start_ts FROM evox_bookings WHERE lower(email)=? "
                             "AND session_type='biofield-consult' AND status='booked' "
                             "ORDER BY start_ts DESC LIMIT 1", (email,)).fetchone()
            booked_start = row[0] if row else None
        except Exception:
            pass
        return {"ready": ready, "booked": booked_start is not None,
                "booked_start": booked_start,
                "stages": {"test_paid": paid, "ready": ready}}
```
(Keep the outer try/except → safe default `{"ready":False,"booked":False,"booked_start":None,"stages":{}}`.)

- [ ] **Step 4: Run to verify pass**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/dc-test python3 -m pytest tests/test_consult_api.py -q`
Expected: PASS.

- [ ] **Step 5: Update the portal booked-state card + add join JS in `static/client-portal.html`**

Replace the `if(c.booked){ ... }` branch (the "Check your email for the Zoom link" card) with:
```javascript
    if(c.booked){
      const when = (c.booked_start||"").replace("T"," ");
      html += `<div class="card"><h2>Biofield Consult</h2>
        <p class="muted">Your consult with Dr. Glen is booked for <b>${esc(when)} HST</b>.</p>
        <button class="btn full" onclick="consultJoin()">Join your consult</button>
        <div id="consult-join-msg" class="muted" style="margin-top:.5rem"></div></div>`;
    }
```
Add the join function (near `consultSchedule`/`consultBook`):
```javascript
async function consultJoin(){
  const r = await fetch(`/api/consult/join?token=${encodeURIComponent(seg)}`);
  const d = await r.json();
  if(d && d.ok && d.join_url){ window.open(d.join_url, "_blank", "noopener"); return; }
  const when = (d && d.start_ts ? d.start_ts.replace("T"," ")+" HST" : "your appointment time");
  document.getElementById("consult-join-msg").textContent =
    "Your consult is at " + when + ". The Join button opens 10 minutes before.";
}
```
(Use the real in-scope token var `seg` and the `esc()` helper, both already used in this file.)

- [ ] **Step 6: Verify the page still parses + suite green**

Extract the `<script>` and `node --check` it (as prior tasks did); run `doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/dc-test python3 -m pytest tests/test_consult_api.py -q`. Full headless render-verify is a go-live step.

- [ ] **Step 7: Commit**

```bash
git add dashboard/portal_view.py static/client-portal.html tests/test_consult_api.py
git commit -m "feat(consult): portal Join-your-consult button (time-gated) + booked_start"
```

---

## Post-implementation
- Set `GLEN_PMI_URL` in Render (default is Glen's room, so optional). No Zoom app dependency for booking/joining anymore.
- Zoom side already done: Waiting Room on, cloud auto-recording on.
- Stage 9 (later): fetch the cloud recording by consult date/time via the recording API and post it to the portal.

## Self-Review
- Static room via `GLEN_PMI_URL` + no Zoom API at booking → Tasks 1/2. ✓
- Server-gated timed join (10 before / 30 after) → Task 1 (`within_join_window` + `/api/consult/join`). ✓
- Raw link out of the email; portal instructions → Task 2. ✓
- Portal card: time + gated Join button → Task 3. ✓
- No placeholders; `within_join_window`/`GLEN_PMI_URL`/`/api/consult/join`/`booked_start` names consistent across tasks. The `ensure_token` availability is guarded (`hasattr` fallback to `/portal/login`).
