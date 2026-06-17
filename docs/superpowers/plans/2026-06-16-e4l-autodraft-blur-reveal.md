# E4L Auto-draft + Two-click Blur-reveal — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** A scan's AI analysis lands in the portal blurred; two clicks (understand → request) drive a server-enforced blur-reveal with Glen's confirm in between, each step emitting a GHL followup tag.

**Architecture:** A `biofield_status` on the portal content (`ai_draft → interested → requested → confirmed`); the blur is enforced server-side (unconfirmed remedies never sent); transition endpoints + a confirm path move the state and enqueue GHL tags; the portal page renders per state. Legacy/hand-built portals default to `confirmed` (no regression).

**Tech Stack:** Flask, sqlite, the existing `client_portals`/`ghl_queue` modules. Tests via `doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" ~/.venvs/deploy-chat311/bin/python -m pytest`.

**Spec:** `docs/superpowers/specs/2026-06-16-e4l-autodraft-blur-reveal-design.md`

---

## File Structure
- **Modify** `dashboard/client_portal.py` — `get/set_biofield_status`.
- **Modify** `dashboard/portal_view.py` — `_biofield_block` gates remedy/dosing by status.
- **Modify** `app.py` — `/biofield/interest`, `/biofield/request` (transitions+tags), `/api/console/biofield/review-queue`, confirm-on-publish.
- **Modify** `static/client-portal.html` — two-click rendering.
- **Modify** `02 Skills/e4l-email-trigger.sh` (vault) — auto-draft hook.
- **Tests:** `tests/test_client_portal.py` (new, for the status helpers + block), `tests/test_client_portal_routes.py` (routes).

---

## Task 1: status helpers — `client_portal.get/set_biofield_status`

**Files:** Modify `dashboard/client_portal.py`; Test `tests/test_client_portal.py` (new)

- [ ] **Step 1: Failing test**

```python
# tests/test_client_portal.py
import sqlite3
from dashboard import client_portal as cp


def _cx(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "t.db")); cp.init_client_portal_table(cx); return cx


def test_status_defaults_confirmed_for_legacy(tmp_path):
    cx = _cx(tmp_path)
    cp.upsert_portal(cx, "a@x.com", "A", {"layers": [{"n": 1, "title": "t", "remedy": "R"}]})
    assert cp.get_biofield_status(cx, "a@x.com") == "confirmed"   # no field -> confirmed


def test_set_and_get_status(tmp_path):
    cx = _cx(tmp_path)
    cp.upsert_portal(cx, "a@x.com", "A", {"biofield_status": "ai_draft", "layers": []})
    assert cp.get_biofield_status(cx, "a@x.com") == "ai_draft"
    assert cp.set_biofield_status(cx, "a@x.com", "interested") is True
    assert cp.get_biofield_status(cx, "a@x.com") == "interested"
    assert cp.set_biofield_status(cx, "nobody@x.com", "interested") is False
```

- [ ] **Step 2: Run — FAIL** (`... -m pytest tests/test_client_portal.py -q`)

- [ ] **Step 3: Implement** (append to `dashboard/client_portal.py`)

```python
def get_biofield_status(cx, email):
    """The biofield review status; legacy/hand-built portals (no field) = 'confirmed'."""
    rec = get_portal_content_by_email(cx, email)
    if not rec:
        return None
    return (rec.get("content") or {}).get("biofield_status") or "confirmed"


def set_biofield_status(cx, email, status):
    """Set content.biofield_status in place. Returns False if no portal for that email."""
    email = (email or "").strip().lower()
    row = cx.execute("SELECT content_json FROM client_portals WHERE email=?", (email,)).fetchone()
    if not row:
        return False
    try:
        content = json.loads(row[0] or "{}")
    except Exception:
        content = {}
    content["biofield_status"] = status
    cx.execute("UPDATE client_portals SET content_json=?, updated_at=? WHERE email=?",
               (json.dumps(content), _now_iso(), email))
    cx.commit()
    return True
```

- [ ] **Step 4: PASS** · **Step 5: Commit** (`-m "portal: biofield_status helpers (legacy=confirmed)"`)

---

## Task 2: blur enforcement — `_biofield_block` gates remedies by status

**Files:** Modify `dashboard/portal_view.py`; Test `tests/test_portal_view.py`

- [ ] **Step 1: Failing test** (add to `tests/test_portal_view.py`)

```python
def test_biofield_block_blurs_remedies_until_confirmed(tmp_path):
    from dashboard import portal_view as pv
    from dashboard import client_portal as cp
    cx = _conn(tmp_path)
    pid = _add_person(cx, "bf@example.com", "BF")
    cp.upsert_portal(cx, "bf@example.com", "BF", {"biofield_status": "interested", "greeting": "hi",
        "layers": [{"n": 1, "title": "Calm", "meaning": "m", "remedy": "Nous Energy", "dosing": "1/day"}]})
    bf = pv.get_portal_view(cx, pid)["biofield"]
    assert bf["status"] == "interested" and bf["blurred"] is True
    assert bf["layers"][0]["title"] == "Calm" and bf["layers"][0]["meaning"] == "m"   # shown
    assert "remedy" not in bf["layers"][0] and "dosing" not in bf["layers"][0]        # withheld


def test_biofield_block_reveals_remedies_when_confirmed(tmp_path):
    from dashboard import portal_view as pv
    from dashboard import client_portal as cp
    cx = _conn(tmp_path)
    pid = _add_person(cx, "cf@example.com", "CF")
    cp.upsert_portal(cx, "cf@example.com", "CF", {"biofield_status": "confirmed", "greeting": "hi",
        "layers": [{"n": 1, "title": "Calm", "meaning": "m", "remedy": "Nous Energy", "dosing": "1/day"}]})
    bf = pv.get_portal_view(cx, pid)["biofield"]
    assert bf["status"] == "confirmed" and bf["blurred"] is False
    assert bf["layers"][0]["remedy"] == "Nous Energy"
```

- [ ] **Step 2: Run — FAIL**

- [ ] **Step 3: Implement** — replace the `layers` handling in `dashboard/portal_view.py:_biofield_block`:

```python
def _biofield_block(cx, email):
    try:
        rec = _cp.get_portal_content_by_email(cx, email)
    except Exception:
        rec = None
    content = (rec or {}).get("content") or {}
    has = bool(content.get("greeting") or content.get("layers") or content.get("video"))
    if not has:
        return {"visible": False}
    status = content.get("biofield_status") or "confirmed"
    confirmed = status == "confirmed"
    layers = []
    for L in (content.get("layers") or []):
        item = {"n": L.get("n"), "title": L.get("title", ""), "meaning": L.get("meaning", "")}
        if confirmed:  # unconfirmed remedies NEVER leave the server
            item["remedy"] = L.get("remedy", "")
            item["dosing"] = L.get("dosing", "")
        layers.append(item)
    return {"visible": True, "status": status, "blurred": not confirmed,
            "greeting": content.get("greeting", ""), "video": content.get("video") or {},
            "layers": layers, "pricing_note": content.get("pricing_note", "") if confirmed else ""}
```

- [ ] **Step 4: PASS** · **Step 5: Commit** (`-m "portal: blur biofield remedies until confirmed (server-side)"`)

---

## Task 3: transition endpoints — interest + request (+ GHL tags)

**Files:** Modify `app.py`; Test `tests/test_client_portal_routes.py`

- [ ] **Step 1: Failing tests** (add)

```python
def test_biofield_interest_then_request(client, monkeypatch):
    c, appmod = client
    from dashboard import client_portal as cp, ghl_queue
    import sqlite3
    cx = sqlite3.connect(appmod.LOG_DB); cp.init_client_portal_table(cx)
    cp.upsert_portal(cx, "e@y.com", "E", {"biofield_status": "ai_draft", "greeting": "hi",
        "layers": [{"n": 1, "title": "Calm", "meaning": "m", "remedy": "R", "dosing": "d"}]})
    tok = cx.execute("SELECT token_hash FROM client_portals WHERE email='e@y.com'")  # need raw token
    cx.close()
    tk = _seed_portal(appmod, email="e@y.com", name="E")  # re-mint not needed; use existing helper
    # click 1
    r = c.post(f"/api/portal/{tk}/biofield/interest")
    assert r.status_code == 200 and r.get_json()["status"] == "interested"
    # click 2
    r = c.post(f"/api/portal/{tk}/biofield/request")
    assert r.status_code == 200 and r.get_json()["status"] == "requested"
    # tags enqueued
    cx = sqlite3.connect(appmod.LOG_DB)
    tags = [row[0] for row in cx.execute("SELECT payload_json FROM ghl_write_queue").fetchall()]
    assert any("e4l:interested" in t for t in tags) and any("e4l:requested" in t for t in tags)
```

(Note: `_seed_portal` re-upserts content; adjust the seed to include `biofield_status: ai_draft` so the test starts from draft. Use the existing `_seed_portal(appmod, email, name, content=...)` with that status.)

- [ ] **Step 2: Run — FAIL**

- [ ] **Step 3: Implement** (add near the other portal routes in `app.py`)

```python
def _biofield_transition(token, new_status, tag):
    from dashboard import client_portal as _cp, portal_identity as _pi, ghl_queue as _gq
    sess = request.cookies.get("rm_portal_session", "")
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        _cp.init_client_portal_table(cx)
        ident = _pi.resolve_identity(cx, token=token, session_token=sess,
                                     client_login_enabled=_client_login_enabled())
        if ident is None or not _cp.set_biofield_status(cx, ident.email, new_status):
            return jsonify({"error": "not found"}), 404
        try:
            _gq.init_ghl_queue_table(cx)
            _gq.enqueue(cx, op="tag_add", email=ident.email, payload={"tag": tag}, actor="portal")
        except Exception as e:
            print(f"[biofield-transition] ghl enqueue failed: {e!r}", flush=True)
    return jsonify({"ok": True, "status": new_status})


@app.route("/api/portal/<token>/biofield/interest", methods=["POST"])
def api_portal_biofield_interest(token):
    return _biofield_transition(token, "interested", "e4l:interested")


@app.route("/api/portal/<token>/biofield/request", methods=["POST"])
def api_portal_biofield_request(token):
    return _biofield_transition(token, "requested", "e4l:requested")
```

- [ ] **Step 4: PASS** · **Step 5: Commit** (`-m "portal: biofield interest/request transitions + GHL tags"`)
  - Verify during impl: `ghl_queue.enqueue` payload key for `tag_add` (the existing tag action's expected shape) — adjust `{"tag": tag}` to match.

---

## Task 4: review queue + confirm-on-publish

**Files:** Modify `app.py`; Test `tests/test_console_biofield_portal.py`

- [ ] **Step 1: Failing tests**

```python
def test_review_queue_lists_requested(client):
    c, appmod = client
    from dashboard import client_portal as cp
    import sqlite3
    cx = sqlite3.connect(appmod.LOG_DB); cp.init_client_portal_table(cx)
    cp.upsert_portal(cx, "req@y.com", "Req", {"biofield_status": "requested", "layers": []})
    cp.upsert_portal(cx, "drft@y.com", "Drft", {"biofield_status": "ai_draft", "layers": []})
    cx.close()
    j = c.get("/api/console/biofield/review-queue?key=test-secret").get_json()
    emails = [r["email"] for r in j["queue"]]
    assert "req@y.com" in emails and "drft@y.com" not in emails


def test_publish_confirms_status(client):
    c, _ = client
    r = c.post("/api/console/biofield-portal?key=test-secret",
               json={"email": "cf@y.com", "name": "CF", "content": {"layers": [
                   {"n": 1, "title": "Calm", "remedy": "R"}]}})
    assert r.status_code == 200
    # publishing through the editor confirms the analysis
    from dashboard import client_portal as cp
    import sqlite3, app as appmod
    cx = sqlite3.connect(appmod.LOG_DB)
    assert cp.get_biofield_status(cx, "cf@y.com") == "confirmed"
```

- [ ] **Step 2: Run — FAIL**

- [ ] **Step 3: Implement**
  - Add the review-queue route:
```python
@app.route("/api/console/biofield/review-queue", methods=["GET"])
def api_console_biofield_review_queue():
    if not _portal_console_ok():
        return jsonify({"error": "unauthorized"}), 401
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        rows = cx.execute("SELECT email, name, updated_at FROM client_portals").fetchall()
    from dashboard import client_portal as _cp
    queue = []
    with sqlite3.connect(LOG_DB) as cx:
        for r in rows:
            if _cp.get_biofield_status(cx, r["email"]) == "requested":
                queue.append({"email": r["email"], "name": r["name"], "requested_at": r["updated_at"]})
    return jsonify({"queue": queue})
```
  - In `api_console_biofield_publish` (the editor publish), force confirmation + tag: after `upsert_portal`, set status confirmed (publishing a portal IS Glen's confirmation):
```python
        _cp.upsert_portal(cx, email, name, content)
        _cp.set_biofield_status(cx, email, "confirmed")
        try:
            from dashboard import ghl_queue as _gq
            _gq.init_ghl_queue_table(cx); _gq.enqueue(cx, op="tag_add", email=email,
                payload={"tag": "e4l:confirmed"}, actor="console")
        except Exception as e:
            print(f"[biofield-publish] confirm tag failed: {e!r}", flush=True)
```
  (The existing `send:true` path already emails the client — that's the notify.)

- [ ] **Step 4: PASS** · **Step 5: Commit** (`-m "portal: biofield review queue + confirm-on-publish + tag"`)

---

## Task 5: portal rendering (two-click) — `static/client-portal.html`

**Files:** Modify `static/client-portal.html` (no unit test; manual smoke)

- [ ] **Step 1: Render the biofield block by status.** In the biofield section, branch on `d.status`/`v.biofield.status` (the content endpoint should also pass `biofield_status`; reuse the `/view` block which already carries `status`+`blurred`):
  - `ai_draft` → a card "Your scan analysis is ready — view it" + button calling `POST /api/portal/<seg>/biofield/interest` then re-`load()`.
  - `interested`/`requested` → render the layers' `title`+`meaning` (no remedies); a **blurred** placeholder card "Your remedy matches are being confirmed by Dr. Glen." If `interested`, a "Request my remedy matches" button → `POST …/biofield/request` → re-load. If `requested`, static "Requested — Dr. Glen is confirming your matches."
  - `confirmed` → the existing full layer rendering (titles/meanings/remedies/dosing).
  - Always show, until confirmed: a small "AI-generated, pending Dr. Glen's review" line.

- [ ] **Step 2: Manual smoke** — seed a portal at each status (via the editor / a quick upsert), confirm the page shows the right state + the buttons advance it.

- [ ] **Step 3: Commit** (`-m "portal page: two-click biofield blur-reveal rendering"`)

---

## Task 6: auto-draft on ingest — `02 Skills/e4l-email-trigger.sh` (vault)

**Files:** Modify `02 Skills/e4l-email-trigger.sh`

- [ ] **Step 1:** After the parse step (where new scans land), for each newly-ingested client run the importer and publish as a draft:
```bash
  # Auto-draft fresh analyses into the portal (AI-labeled, remedies blurred).
  for EMAIL in $NEW_SCAN_EMAILS; do
    $DOPPLER $PYTHON -u "$SKILLS/e4l-portal-import.py" --email "$EMAIL" --publish-draft 2>&1 | tail -3 >> "$LOG"
  done
```
  Add a `--publish-draft` flag to `e4l-portal-import.py` that, instead of only writing the seed file, POSTs to `/admin/portal/upsert` with `content.biofield_status="ai_draft"` (CONSOLE_SECRET from env). `$NEW_SCAN_EMAILS` = the emails whose scans were just parsed (derive from the parse output or query e4l.db for scans with today's ingest).

- [ ] **Step 2: Manual run** on a test client → confirm a draft portal appears with status `ai_draft` and blurred remedies.

- [ ] **Step 3: Commit** (vault auto-snapshots).

---

## Task 7: full suite, push, PR
- [ ] Run `... -m pytest -q` → all green.
- [ ] Push `sess/5326cc61`; open PR (base main).

---

## Self-Review notes
- **Spec coverage:** content model + server-side blur (T1, T2), interest/request transitions + GHL tags (T3), review queue + confirm + notify (T4), two-click rendering + AI-disclosure (T5), auto-draft on ingest (T6). Legacy=confirmed no-regression covered (T1/T2). Deferred items (access-gating, purchase state, offer) correctly absent; the GHL `e4l:*` tags are emitted so GHL can branch later.
- **Type consistency:** `biofield_status` ∈ {ai_draft,interested,requested,confirmed}; `get/set_biofield_status(cx,email[,status])`; `_biofield_block` returns `{visible,status,blurred,greeting,video,layers,pricing_note}` (layers omit remedy/dosing unless confirmed); transitions return `{ok,status}`; tags `e4l:interested|requested|confirmed`.
- **Verify during impl:** exact `ghl_queue.enqueue` payload shape for `tag_add`; the content endpoint (`/api/portal/<token>`) should also surface `biofield_status` (or have the page read it from `/view`); `$NEW_SCAN_EMAILS` derivation in the email-trigger.
