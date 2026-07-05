# Request Further Analysis (Sub-project B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a client turn an "Available" scan into a full report via a rate-gated (1/mo free, unlimited paid) "Request analysis" action fulfilled automatically by a local worker, plus a proactive new-scan email inviting analysis.

**Architecture:** A new `analysis_requests` queue (LOG_DB) + a portal request endpoint reusing the existing `analysis_quota` gate; the sync endpoint fires an anti-nag new-scan email (cc'd to caregivers); a local worker polls pending requests, runs the existing synthesis pipeline against `e4l.db`, publishes, and marks done — piggybacked on the 5-minute ingest.

**Tech Stack:** Python 3, Flask, SQLite (`LOG_DB`; read-only `e4l.db` locally), urllib, pytest, vanilla JS.

## Global Constraints

- **Reuse, don't rebuild:** rate gate = `dashboard/analysis_quota.py` (`try_claim`/`release`/`claimed_this_month`, paid bypass via `_is_paid_member`); fulfillment = `dashboard/biofield_reveal_import.synthesize_reveal_layers(email, scan_id, *, e4l_db, catalog, today, runner)` + `biofield_portal_publish.publish_to_portal(payload, *, base_url, console_key, send, http_post)`; scan source = A's `client_scans` + `available_scans`.
- **Do NOT touch** the existing published-report `status="requested"` flow (`_biofield_transition`, app.py ~16175) — B is a separate queue.
- **Quota keyed on the scan OWNER** (`email_for_reports`, which may be a household member when `?member=` is active).
- **Automated fulfillment**, piggybacked on the 5-minute `e4l-email-trigger`.
- **New-scan email anti-nag:** only when the owner can act (`_is_paid_member(email)` OR `not analysis_quota.claimed_this_month(email)`), once per scan (`client_scans.notified_at`), respecting `email_suppression` + `notify_state` opt-out; cc'd to caregivers via `household.cc_recipients_for` (private separate copy).
- **One-click email GET lands a page; the page's confirm button POSTs** the claim — so an email-scanner prefetch cannot consume a monthly slot.
- **Behind `SCAN_REQUEST_ENABLED` (default OFF):** request endpoint + button + new-scan email + one-click page all inert; A unchanged.
- Deploy-chat tests via `doppler run -p remedy-match -c dev -- python3 -m pytest ...` (use `python3`). Do NOT `git stash`. **Task 8 edits the VAULT `~/AI-Training/02 Skills/` directly** (not the worktree).

---

## File Structure

- **Create** `dashboard/analysis_requests.py` — the queue.
- **Modify** `dashboard/client_scans.py` — `notified_at` column + `unnotified`/`mark_notified`.
- **Modify** `app.py` — `_scan_request_enabled()`; `POST /api/portal/<token>/request-analysis`; `GET /portal/<token>/analyze`; `available_scans` gains `requested`; new-scan email in the sync endpoint; `GET/POST /api/console/analysis-requests...`.
- **Modify** `static/client-portal.html` — Request button + states.
- **Create** `static/portal-analyze.html` — one-click result page.
- **Create** `~/AI-Training/02 Skills/e4l-analysis-fulfill.py` (vault) + wire into `e4l-email-trigger.sh`.
- **Test** `tests/test_analysis_requests.py`, `tests/test_request_analysis_routes.py`.

---

### Task 1: `dashboard/analysis_requests.py` — the queue

**Files:** Create `dashboard/analysis_requests.py`; Test `tests/test_analysis_requests.py`

**Interfaces:** `init_analysis_requests_table(cx)`; `create_request(cx, email, scan_id, scan_date) -> {"created":bool,"status":str|None}` (idempotent per (email,scan_date)); `has_pending(cx, email, scan_date) -> bool`; `pending(cx, limit=50) -> [{id,email,scan_id,scan_date}]`; `statuses_for(cx, email) -> {scan_date: status}`; `mark(cx, req_id, status)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_analysis_requests.py
import sqlite3
from dashboard import analysis_requests as ar


def _cx():
    cx = sqlite3.connect(":memory:"); ar.init_analysis_requests_table(cx); return cx


def test_create_idempotent():
    cx = _cx()
    assert ar.create_request(cx, "K@x.com", 7, "2026-06-28") == {"created": True, "status": "pending"}
    assert ar.create_request(cx, "k@x.com", 7, "2026-06-28") == {"created": False, "status": "pending"}
    assert ar.has_pending(cx, "k@x.com", "2026-06-28") is True


def test_pending_and_mark_and_statuses():
    cx = _cx()
    ar.create_request(cx, "a@x.com", 1, "2026-06-01")
    ar.create_request(cx, "b@x.com", 2, "2026-06-02")
    p = ar.pending(cx)
    assert {r["email"] for r in p} == {"a@x.com", "b@x.com"}
    ar.mark(cx, p[0]["id"], "done")
    assert ar.has_pending(cx, p[0]["email"], p[0]["scan_date"]) is False
    assert ar.statuses_for(cx, p[0]["email"])[p[0]["scan_date"]] == "done"


def test_blank_skipped():
    cx = _cx()
    assert ar.create_request(cx, "", 1, "2026-06-01")["created"] is False
    assert ar.create_request(cx, "a@x.com", 1, "")["created"] is False
```

- [ ] **Step 2: Run to verify it fails** — `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_analysis_requests.py -q` → FAIL (module missing).

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/analysis_requests.py
"""Queue of client requests to analyze an as-yet-unprocessed E4L scan. A local worker
fulfills pending rows (synthesize + publish) and marks them done. One row per
(email, scan_date). LOG_DB (SQLite). Separate from the published-report 'requested' flow."""
import datetime


def _now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _norm(e):
    return (e or "").strip().lower()


def init_analysis_requests_table(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS analysis_requests (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            email        TEXT NOT NULL,
            scan_id      TEXT,
            scan_date    TEXT NOT NULL,
            requested_at TEXT,
            status       TEXT NOT NULL,
            fulfilled_at TEXT,
            UNIQUE(email, scan_date)
        )
    """)
    cx.execute("CREATE INDEX IF NOT EXISTS ix_ar_status ON analysis_requests(status)")
    cx.commit()


def create_request(cx, email, scan_id, scan_date):
    e, d = _norm(email), (scan_date or "").strip()
    if not e or not d:
        return {"created": False, "status": None}
    row = cx.execute("SELECT status FROM analysis_requests WHERE email=? AND scan_date=?",
                     (e, d)).fetchone()
    if row:
        return {"created": False, "status": row[0]}
    cx.execute("INSERT INTO analysis_requests (email, scan_id, scan_date, requested_at, status) "
               "VALUES (?,?,?,?, 'pending')", (e, str(scan_id or ""), d, _now()))
    cx.commit()
    return {"created": True, "status": "pending"}


def has_pending(cx, email, scan_date):
    return cx.execute("SELECT 1 FROM analysis_requests WHERE email=? AND scan_date=? "
                      "AND status='pending' LIMIT 1", (_norm(email), (scan_date or "").strip())
                      ).fetchone() is not None


def pending(cx, limit=50):
    rows = cx.execute("SELECT id, email, scan_id, scan_date FROM analysis_requests "
                      "WHERE status='pending' ORDER BY id LIMIT ?", (int(limit),)).fetchall()
    return [{"id": r[0], "email": r[1], "scan_id": r[2], "scan_date": r[3]} for r in rows]


def statuses_for(cx, email):
    rows = cx.execute("SELECT scan_date, status FROM analysis_requests WHERE email=?",
                      (_norm(email),)).fetchall()
    return {r[0]: r[1] for r in rows}


def mark(cx, req_id, status):
    cx.execute("UPDATE analysis_requests SET status=?, fulfilled_at=? WHERE id=?",
               (status, _now() if status in ("done", "failed") else None, req_id))
    cx.commit()
```

- [ ] **Step 4: Run to verify it passes** — GREEN.

- [ ] **Step 5: Commit**

```bash
git add dashboard/analysis_requests.py tests/test_analysis_requests.py
git commit -m "feat(request-analysis): analysis_requests queue"
```

---

### Task 2: `client_scans.notified_at` — new-scan email bookkeeping

**Files:** Modify `dashboard/client_scans.py`; Test `tests/test_client_scans.py` (append)

**Interfaces:** `init_client_scans_table` migrates a `notified_at TEXT` column (additive); `unnotified(cx, email=None, limit=500) -> [{email, scan_date, scan_id}]` (rows with `notified_at IS NULL`, optionally scoped to one email); `mark_notified(cx, email, scan_date)`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_client_scans.py
def test_notified_flow():
    from dashboard import client_scans as cs
    import sqlite3
    cx = sqlite3.connect(":memory:"); cs.init_client_scans_table(cx)
    cs.upsert_scans(cx, "k@x.com", [{"scan_date": "2026-06-28"}, {"scan_date": "2026-06-25"}])
    un = cs.unnotified(cx, "k@x.com")
    assert {u["scan_date"] for u in un} == {"2026-06-28", "2026-06-25"}
    cs.mark_notified(cx, "k@x.com", "2026-06-28")
    assert [u["scan_date"] for u in cs.unnotified(cx, "k@x.com")] == ["2026-06-25"]
```

- [ ] **Step 2: Run to verify it fails** — FAIL (`unnotified` missing).

- [ ] **Step 3: Write minimal implementation** (edit `dashboard/client_scans.py`)

In `init_client_scans_table`, after the CREATE + index, add the additive column:

```python
    try:
        cx.execute("ALTER TABLE client_scans ADD COLUMN notified_at TEXT")
    except Exception:
        pass
    cx.commit()
```

Add:

```python
def unnotified(cx, email=None, limit=500):
    if email:
        rows = cx.execute("SELECT email, scan_date, scan_id FROM client_scans "
                          "WHERE email=? AND notified_at IS NULL ORDER BY scan_date DESC LIMIT ?",
                          (_norm(email), int(limit))).fetchall()
    else:
        rows = cx.execute("SELECT email, scan_date, scan_id FROM client_scans "
                          "WHERE notified_at IS NULL ORDER BY id LIMIT ?", (int(limit),)).fetchall()
    return [{"email": r[0], "scan_date": r[1], "scan_id": r[2] or ""} for r in rows]


def mark_notified(cx, email, scan_date):
    cx.execute("UPDATE client_scans SET notified_at=? WHERE email=? AND scan_date=?",
               (_now(), _norm(email), (scan_date or "").strip()))
    cx.commit()
```

- [ ] **Step 4: Run to verify it passes** — `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_client_scans.py -q` GREEN.

- [ ] **Step 5: Commit**

```bash
git add dashboard/client_scans.py tests/test_client_scans.py
git commit -m "feat(request-analysis): client_scans.notified_at + unnotified/mark_notified"
```

---

### Task 3: request endpoint + `available_scans.requested`

**Files:** Modify `app.py`; Test `tests/test_request_analysis_routes.py`

**Interfaces:**
- Consumes: `analysis_requests.{create_request,has_pending,statuses_for}` (Task 1); `analysis_quota.{try_claim,claimed_this_month}` + `_is_paid_member`; `_pbr.list_report_dates`; the `?member=` household resolution.
- Produces: `_scan_request_enabled()`; `POST /api/portal/<token>/request-analysis`; `available_scans` items gain `requested` (a pending request exists).

**Context:** The `available_scans` payload block is at ~app.py:14807. The `?member=` resolution that yields `email_for_reports` (with `household.can_view`) is the same one used by the read-receipt `/open` endpoint (grep `def api_portal_open`) — replicate it in the new endpoint. `email_for_reports` in the payload path is already member-switched.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_request_analysis_routes.py
import importlib, sqlite3, sys
from pathlib import Path
import pytest


def _app(tmp_path, monkeypatch, *, flag="1"):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SCAN_REQUEST_ENABLED", flag)
    monkeypatch.setenv("SCAN_LIST_ENABLED", "1")
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path: sys.path.insert(0, str(repo))
    try:
        import app as appmod; importlib.reload(appmod)
        import dashboard as _d; monkeypatch.setattr(_d, "CONSOLE_SECRET", "", raising=False)
    except Exception as e:
        pytest.skip(f"app not importable: {e}")
    return appmod


def _mint(appmod, email, scan_date="2026-06-28"):
    from dashboard import client_portal as cp, client_scans as cs
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cp.init_client_portal_table(cx); cs.init_client_scans_table(cx)
        cs.upsert_scans(cx, email, [{"scan_date": scan_date, "scan_id": 9}])
        tok = cp.upsert_portal(cx, email, "N", {}); cx.commit()
    return tok[0] if isinstance(tok, (tuple, list)) else tok


def test_free_member_one_then_quota(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    monkeypatch.setattr(appmod, "_is_paid_member", lambda e: False)
    token = _mint(appmod, "k@x.com")
    if not token: pytest.skip("no mint")
    c = appmod.app.test_client()
    r = c.post(f"/api/portal/{token}/request-analysis", json={"scan_id": 9, "scan_date": "2026-06-28"})
    assert r.get_json()["status"] == "pending"
    # second scan same month → quota exceeded
    from dashboard import client_scans as cs
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cs.upsert_scans(cx, "k@x.com", [{"scan_date": "2026-05-01"}]); cx.commit()
    r2 = c.post(f"/api/portal/{token}/request-analysis", json={"scan_id": 1, "scan_date": "2026-05-01"})
    assert r2.get_json().get("reason") == "monthly_quota"


def test_paid_member_unlimited(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    monkeypatch.setattr(appmod, "_is_paid_member", lambda e: True)
    token = _mint(appmod, "p@x.com")
    if not token: pytest.skip("no mint")
    c = appmod.app.test_client()
    from dashboard import client_scans as cs
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cs.upsert_scans(cx, "p@x.com", [{"scan_date": "2026-05-01"}]); cx.commit()
    assert c.post(f"/api/portal/{token}/request-analysis", json={"scan_id": 9, "scan_date": "2026-06-28"}).get_json()["status"] == "pending"
    assert c.post(f"/api/portal/{token}/request-analysis", json={"scan_id": 1, "scan_date": "2026-05-01"}).get_json()["status"] == "pending"


def test_requested_flag_and_flag_off(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    monkeypatch.setattr(appmod, "_is_paid_member", lambda e: True)
    token = _mint(appmod, "k@x.com")
    if not token: pytest.skip("no mint")
    c = appmod.app.test_client()
    c.post(f"/api/portal/{token}/request-analysis", json={"scan_id": 9, "scan_date": "2026-06-28"})
    j = c.get(f"/api/portal/{token}").get_json()
    req = {s["scan_date"]: s.get("requested") for s in j.get("available_scans", [])}
    assert req.get("2026-06-28") is True
    # flag off → endpoint inert
    appmod2 = _app(tmp_path, monkeypatch, flag="0")
    c2 = appmod2.app.test_client()
    assert c2.post(f"/api/portal/{token}/request-analysis", json={"scan_id": 9, "scan_date": "2026-06-28"}).get_json()["status"] == "disabled"
```

- [ ] **Step 2: Run to verify it fails** — FAIL (404 / no `requested`).

- [ ] **Step 3: Write minimal implementation**

Flag helper near `_scan_list_enabled` (app.py):

```python
def _scan_request_enabled():
    return (os.environ.get("SCAN_REQUEST_ENABLED", "") or "").strip().lower() in ("1", "true", "yes")
```

Endpoint (near the other `/api/portal/<token>/*` POST routes) — a `_resolve_request(token, body)` inner does the shared resolution + gate, reused by Task 4's page:

```python
def _request_analysis_core(token, scan_id, scan_date):
    """Shared by the POST endpoint and the one-click page. Returns a (result_dict, http_status)."""
    from dashboard import client_portal as _cp
    from dashboard import analysis_requests as _ar
    from dashboard import analysis_quota as _aq
    scan_date = (scan_date or "").strip()
    with sqlite3.connect(LOG_DB) as cx:
        _cp.init_client_portal_table(cx)
        portal = _portal_record_for(cx, token)
    if not portal:
        return {"ok": False, "error": "not found"}, 404
    email_for_reports = (portal.get("email") or "").strip().lower()
    # same ?member= household resolution as api_portal_open (fail-closed)
    if _household_view_enabled() and email_for_reports:
        try:
            from dashboard import household as _hh
            with sqlite3.connect(LOG_DB) as _cxh:
                _hh.init_household_tables(_cxh)
                _m = (request.args.get("member") or (request.get_json(silent=True) or {}).get("member") or "").strip().lower()
                if _m and _hh.can_view(_cxh, email_for_reports, _m):
                    email_for_reports = _m
        except Exception as _e:
            print(f"[request-analysis] household {_e!r}", flush=True)
    if not scan_date or not email_for_reports:
        return {"ok": False, "error": "missing scan_date"}, 400
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        # already processed (published report) or already requested → no quota spent
        if scan_date in set(_pbr.list_report_dates(cx, email_for_reports)):
            return {"ok": True, "status": "already"}, 200
        _ar.init_analysis_requests_table(cx)
        _st = _ar.statuses_for(cx, email_for_reports).get(scan_date)
        if _st in ("pending", "done"):
            return {"ok": True, "status": _st}, 200
        if not _is_paid_member(email_for_reports):
            _aq.init_analysis_quota_table(cx)
            if not _aq.try_claim(cx, email_for_reports):
                return {"ok": False, "reason": "monthly_quota",
                        "upgrade_url": "https://illtowell.com/prepay"}, 200
        res = _ar.create_request(cx, email_for_reports, scan_id, scan_date)
    return {"ok": True, "status": res["status"]}, 200


@app.route("/api/portal/<token>/request-analysis", methods=["POST"])
def api_portal_request_analysis(token):
    if not _scan_request_enabled():
        return jsonify({"ok": True, "status": "disabled"})
    body = request.get_json(silent=True) or {}
    res, code = _request_analysis_core(token, body.get("scan_id"), body.get("scan_date"))
    return jsonify(res), code
```

`available_scans` — in the payload block (~app.py:14807), annotate each scan with `requested`:

```python
                from dashboard import analysis_requests as _ar
                _ar.init_analysis_requests_table(_cxs)
                _reqst = _ar.statuses_for(_cxs, email_for_reports)
                payload["available_scans"] = [
                    {"scan_date": s["scan_date"], "scan_id": s["scan_id"],
                     "processed": s["scan_date"] in _processed,
                     "requested": _reqst.get(s["scan_date"]) == "pending"} for s in _synced]
```

> Implementer note: fold this into the existing `if _scan_list_enabled():` block (read its current shape); reuse the same `_cxs` connection. The `requested` flag is only meaningful when `_scan_request_enabled()` — but including it always (defaulting False when no request row) is harmless and simpler; keep it in the `_scan_list_enabled()` block.

- [ ] **Step 4: Run to verify it passes** — GREEN.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_request_analysis_routes.py
git commit -m "feat(request-analysis): request endpoint + quota gate + requested flag"
```

---

### Task 4: one-click email page — `GET /portal/<token>/analyze`

**Files:** Modify `app.py`; Create `static/portal-analyze.html`

**Interfaces:** Consumes `_request_analysis_core` (Task 3). Serves a page whose confirm button POSTs `/request-analysis`.

- [ ] **Step 1: Add the route + page**

Route (mirrors `/portal/<token>` static-serve at app.py:14617):

```python
@app.route("/portal/<token>/analyze")
def portal_analyze_page(token):
    # Landing page for the new-scan email's one-click link. The page's confirm button
    # POSTs /request-analysis — so an email-scanner GET-prefetch can't consume a slot.
    resp = send_from_directory(STATIC, "portal-analyze.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp
```

`static/portal-analyze.html` — reads `?scan_id`/`?scan_date` and the token from the path, shows the scan date + a "Analyze this scan" confirm button that POSTs `/api/portal/<token>/request-analysis`, then renders the result (queued / upgrade / already). Vanilla JS, escape injected values.

```html
<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Analyze your scan · Remedy Match</title>
<style>body{font-family:-apple-system,sans-serif;max-width:520px;margin:60px auto;padding:0 18px;color:#1a1a1a}
.btn{background:#2f6f5e;color:#fff;border:0;border-radius:8px;padding:10px 18px;font-size:15px;cursor:pointer}
.msg{margin-top:16px;font-size:15px}.up{color:#1F5A4D}a{color:#1F5A4D}</style></head>
<body>
<h1>Analyze your biofield scan</h1>
<p id="lead">Scan date: <strong id="d"></strong></p>
<button class="btn" id="go">Analyze this scan</button>
<div class="msg" id="out"></div>
<script>
const esc = s => String(s==null?"":s).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const P = new URLSearchParams(location.search);
const TOKEN = location.pathname.split("/")[2];
const SCAN_ID = P.get("scan_id")||"", SCAN_DATE = P.get("scan_date")||"";
document.getElementById("d").textContent = SCAN_DATE;
document.getElementById("go").addEventListener("click", async () => {
  document.getElementById("go").disabled = true;
  let j = {};
  try {
    const r = await fetch(`/api/portal/${encodeURIComponent(TOKEN)}/request-analysis`,
      {method:"POST",headers:{"Content-Type":"application/json"},
       body:JSON.stringify({scan_id:SCAN_ID, scan_date:SCAN_DATE})});
    j = await r.json();
  } catch(e){ j = {ok:false}; }
  const out = document.getElementById("out");
  document.getElementById("go").style.display = "none";
  if (j.status === "pending") out.textContent = "Your analysis is being prepared. It will appear in your portal shortly.";
  else if (j.status === "already" || j.status === "done") out.textContent = "This scan has already been analyzed — check your portal.";
  else if (j.reason === "monthly_quota") out.innerHTML = `<span class="up">You've used your free analysis this month. <a href="${esc(j.upgrade_url||"https://illtowell.com/prepay")}">Upgrade for unlimited</a>.</span>`;
  else out.textContent = "Sorry, we couldn't process that right now.";
});
</script></body></html>
```

- [ ] **Step 2: Verify (static)** — `node --check` the inline script; `py_compile app.py`; grep-confirm the page POSTs on the button click (not on load), escapes values.

- [ ] **Step 3: Commit**

```bash
git add app.py static/portal-analyze.html
git commit -m "feat(request-analysis): one-click analyze landing page (confirm-to-claim)"
```

---

### Task 5: new-scan email in the sync endpoint

**Files:** Modify `app.py` (`api_console_client_scans_sync`)

**Interfaces:** Consumes `client_scans.{unnotified,mark_notified}` (Task 2); `analysis_quota.claimed_this_month` + `_is_paid_member`; `household.cc_recipients_for`; `_send_inquiry_email`; `email_suppression.is_suppressed`; `notify_state` opt-out.

- [ ] **Step 1: Write the failing test** (append to `tests/test_request_analysis_routes.py`)

```python
def test_new_scan_email_gated_and_once(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    monkeypatch.setattr(appmod, "_is_paid_member", lambda e: False)   # free member
    sent = []
    monkeypatch.setattr(appmod, "_send_inquiry_email", lambda to, s, b, **k: sent.append(to) or (True, ""))
    # need a portal token so the email can carry a one-click link
    from dashboard import client_portal as cp
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cp.init_client_portal_table(cx); cp.upsert_portal(cx, "k@x.com", "K", {}); cx.commit()
    c = appmod.app.test_client()
    # sync a new scan → email fires (free member, slot unused)
    c.post("/api/console/client-scans/sync", json={"email": "k@x.com", "scans": [{"scan_date": "2026-06-28", "scan_id": 9}]})
    assert "k@x.com" in sent
    sent.clear()
    # re-sync same scan → no re-email (notified_at set)
    c.post("/api/console/client-scans/sync", json={"email": "k@x.com", "scans": [{"scan_date": "2026-06-28", "scan_id": 9}]})
    assert sent == []
```

- [ ] **Step 2: Run to verify it fails** — FAIL (no email sent).

- [ ] **Step 3: Write minimal implementation**

In `api_console_client_scans_sync`, AFTER the upsert loop (still inside `_scan_request_enabled()` guard), notify newly-synced scans:

```python
    if _scan_request_enabled():
        try:
            from dashboard import client_scans as _cs
            from dashboard import analysis_quota as _aq
            from dashboard import client_portal as _cp2
            from dashboard import email_suppression as _es
            with _db_lock, sqlite3.connect(LOG_DB) as cx:
                _cs.init_client_scans_table(cx); _aq.init_analysis_quota_table(cx)
                for row in _cs.unnotified(cx):
                    em, sd, sid = row["email"], row["scan_date"], row["scan_id"]
                    # anti-nag: only when the owner can act
                    can_act = _is_paid_member(em) or not _aq.claimed_this_month(cx, em)
                    if can_act and not _es.is_suppressed(cx, em):
                        tok = _cp2.portal_token_for(cx, em)   # confirm the real helper name
                        if tok:
                            _send_new_scan_email(em, sd, sid, tok)
                    _cs.mark_notified(cx, em, sd)   # mark regardless, so we never re-nag
        except Exception as _e:
            print(f"[new-scan-email] {_e!r}", flush=True)
```

Add the sender:

```python
def _send_new_scan_email(email, scan_date, scan_id, token):
    """New-scan invite: a one-click analyze link + the client's limit + upgrade path. Best-effort.
    Cc'd (private separate copy) to consented+subscribed caregivers via household.cc_recipients_for."""
    base = "https://illtowell.com"
    link = f"{base}/portal/{token}/analyze?scan_id={scan_id}&scan_date={scan_date}"
    subj = "Your new biofield scan is ready to analyze"
    body = (f"A new biofield scan ({scan_date}) is on file for you.\n\n"
            f"Would you like it analyzed? Free members get one analysis per month; members get unlimited.\n\n"
            f"Analyze this scan: {link}\n\nUpgrade for unlimited: {base}/prepay")
    recips = [email]
    try:
        from dashboard import household as _hh
        with sqlite3.connect(LOG_DB) as _cxh:
            _hh.init_household_tables(_cxh)
            recips += _hh.cc_recipients_for(_cxh, email)   # caregivers get their own copy
    except Exception:
        pass
    for to in dict.fromkeys(recips):   # de-dup, private separate copies
        try:
            _send_inquiry_email(to, subj, body)
        except Exception as _e:
            print(f"[new-scan-email] to {to}: {_e!r}", flush=True)
```

> Implementer note: confirm the real "portal token for an email" helper (grep `def portal_token_for` / `client_portal` / `notify_state` — the same one the reveal/portal-link tools use) and the `_send_inquiry_email`/`email_suppression`/`notify_state` opt-out call. If no token exists for an email, skip the email (still `mark_notified` so we don't loop). Keep everything best-effort — the sync's upsert must never fail because of the email.

- [ ] **Step 4: Run to verify it passes** — GREEN.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_request_analysis_routes.py
git commit -m "feat(request-analysis): proactive new-scan email (anti-nag, cc caregivers, once)"
```

---

### Task 6: Request button in `static/client-portal.html`

**Files:** Modify `static/client-portal.html`

**Interfaces:** Consumes `d.available_scans` items now carrying `requested`.

- [ ] **Step 1: Extend the Scan history rows**

In the Scan history section (from A), extend each row: `processed` → "Analyzed"; else `requested` → "Requested"; else a **"Request analysis"** button (`data-scan`/`data-sid` + delegated listener). Only render the button when `d.available_scans` is present AND the payload indicates the request layer is usable — simplest: always render the button for a non-processed, non-requested row; the endpoint returns `disabled` when the flag is off (the click then shows nothing/no-op). On click POST `/api/portal/${token}/request-analysis {scan_id, scan_date}`; on `pending` swap the row to "Requested"; on `monthly_quota` show inline "1 free analysis/month — upgrade" with `j.upgrade_url`. Escape all values; `data-*` + `addEventListener` (no inline onclick with interpolated values).

```javascript
      const tag = s.processed
        ? `<span style="color:#2f6f5e;font-size:.8rem">Analyzed</span>`
        : s.requested
          ? `<span style="color:var(--muted);font-size:.8rem">Requested</span>`
          : `<button class="ra-btn" data-scan="${esc(s.scan_date)}" data-sid="${esc(s.scan_id||"")}"
               style="font-size:.8rem;padding:2px 10px;border-radius:8px;background:#2f6f5e;color:#fff;border:0;cursor:pointer">Request analysis</button>`;
```

Wire after render:

```javascript
  app.querySelectorAll(".ra-btn").forEach(b => b.addEventListener("click", async () => {
    b.disabled = true;
    let j = {};
    try {
      const r = await fetch(`/api/portal/${encodeURIComponent(token)}/request-analysis`,
        {method:"POST",headers:{"Content-Type":"application/json"},
         body:JSON.stringify({scan_date:b.dataset.scan, scan_id:b.dataset.sid})});
      j = await r.json();
    } catch(e){ j = {}; }
    if (j.status === "pending" || j.status === "already" || j.status === "done")
      b.replaceWith(Object.assign(document.createElement("span"),
        {style:"color:var(--muted);font-size:.8rem", textContent:"Requested"}));
    else if (j.reason === "monthly_quota")
      b.replaceWith(Object.assign(document.createElement("span"),
        {style:"color:var(--muted);font-size:.8rem", innerHTML:`1 free analysis/month — <a href="${esc(j.upgrade_url||"/prepay")}">upgrade</a>`}));
    else b.disabled = false;
  }));
```

- [ ] **Step 2: Verify (static)** — `node --check`; grep-confirm the button uses `data-*`+`addEventListener` (no inline onclick), escapes values, POSTs `/request-analysis`, and A's read-only rows still render when the request layer is off.

- [ ] **Step 3: Commit**

```bash
git add static/client-portal.html
git commit -m "feat(request-analysis): Request analysis button + states on Scan history"
```

---

### Task 7: worker console endpoints

**Files:** Modify `app.py`; Test `tests/test_request_analysis_routes.py` (append)

**Interfaces:** Consumes `analysis_requests.{pending,mark}`; `_portal_console_ok()`.

- [ ] **Step 1: Write the failing test**

```python
def test_worker_endpoints(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    from dashboard import analysis_requests as ar
    with sqlite3.connect(appmod.LOG_DB) as cx:
        ar.init_analysis_requests_table(cx); ar.create_request(cx, "k@x.com", 9, "2026-06-28"); cx.commit()
    c = appmod.app.test_client()
    g = c.get("/api/console/analysis-requests?status=pending").get_json()
    assert g["requests"] and g["requests"][0]["email"] == "k@x.com"
    rid = g["requests"][0]["id"]
    assert c.post(f"/api/console/analysis-requests/{rid}/complete", json={"status": "done"}).status_code == 200
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert ar.has_pending(cx, "k@x.com", "2026-06-28") is False
```

- [ ] **Step 2: Run to verify it fails** — FAIL (404).

- [ ] **Step 3: Write minimal implementation**

```python
@app.route("/api/console/analysis-requests", methods=["GET"])
def api_console_analysis_requests():
    if not _portal_console_ok():
        return jsonify({"error": "unauthorized"}), 401
    from dashboard import analysis_requests as _ar
    with sqlite3.connect(LOG_DB) as cx:
        _ar.init_analysis_requests_table(cx)
        reqs = _ar.pending(cx, int(request.args.get("limit", 50)))
    return jsonify({"ok": True, "requests": reqs})


@app.route("/api/console/analysis-requests/<int:req_id>/complete", methods=["POST"])
def api_console_analysis_request_complete(req_id):
    if not _portal_console_ok():
        return jsonify({"error": "unauthorized"}), 401
    from dashboard import analysis_requests as _ar
    status = ((request.get_json(silent=True) or {}).get("status") or "done").strip()
    if status not in ("done", "failed"):
        status = "done"
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        _ar.init_analysis_requests_table(cx)
        _ar.mark(cx, req_id, status)
    return jsonify({"ok": True, "status": status})
```

- [ ] **Step 4: Run to verify it passes** — GREEN.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_request_analysis_routes.py
git commit -m "feat(request-analysis): worker console endpoints (pending + complete)"
```

---

### Task 8: local worker `e4l-analysis-fulfill.py` (VAULT, not worktree)

**Files:** Create `~/AI-Training/02 Skills/e4l-analysis-fulfill.py` — **edit the vault directly; NOT in the deploy-chat worktree.** No deploy-chat commit.

**Interfaces:** Consumes `GET /api/console/analysis-requests?status=pending` + `POST .../<id>/complete` (Task 7); the local synthesis pipeline (mirror `e4l-reveal-push.py` — `synthesize_reveal_layers` + `publish_to_portal`).

- [ ] **Step 1: Write the script**

Mirror `~/AI-Training/02 Skills/e4l-reveal-push.py` for the synthesis+publish invocation and the `CONSOLE_SECRET`/urllib auth. A unit-testable `fulfill_one(req, *, synth, publish) -> str` (returns "done"/"failed") with injected `synth`/`publish` so tests need no real e4l.db/LLM.

```python
#!/usr/bin/env python3
"""Fulfill pending analysis requests: synthesize + publish each scan, then mark done.
Reads pending from prod, runs the local synthesis pipeline (needs e4l.db), publishes.
Piggyback the 5-min e4l-email-trigger. Usage: [--dry] [--limit N]. CONSOLE_SECRET from Doppler prd."""
import argparse, json, os, sys, urllib.request

BASE = os.environ.get("WEB_URL", "https://glen-knowledge-chat.onrender.com").rstrip("/")


def _get_pending(secret, limit):
    req = urllib.request.Request(f"{BASE}/api/console/analysis-requests?status=pending&limit={limit}",
                                 headers={"X-Console-Key": secret})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode()).get("requests", [])


def _complete(secret, req_id, status):
    req = urllib.request.Request(f"{BASE}/api/console/analysis-requests/{req_id}/complete",
                                 method="POST", data=json.dumps({"status": status}).encode(),
                                 headers={"Content-Type": "application/json", "X-Console-Key": secret})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def fulfill_one(req, *, synth, publish):
    """Synthesize + publish one request. synth(email, scan_id)->payload|None ; publish(payload)->bool."""
    try:
        payload = synth(req["email"], req["scan_id"])
        if not payload:
            return "failed"
        return "done" if publish(payload) else "failed"
    except Exception as e:
        print(f"  req {req.get('id')} error: {e!r}", flush=True)
        return "failed"


def _real_synth(email, scan_id):
    import datetime
    sys.path.insert(0, os.path.expanduser("~/deploy-chat"))
    from dashboard import biofield_reveal_import as _bri
    r = _bri.synthesize_reveal_layers(email, int(scan_id) if scan_id else None,
                                      today=datetime.date.today().isoformat())
    return r if r.get("found") else None


def _real_publish(payload):
    sys.path.insert(0, os.path.expanduser("~/deploy-chat"))
    from dashboard import biofield_portal_publish as _pub
    out = _pub.publish_to_portal(payload, base_url=BASE, console_key=os.environ["CONSOLE_SECRET"], send=False)
    return bool(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry", action="store_true"); ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()
    secret = os.environ.get("CONSOLE_SECRET", "")
    if not secret:
        print("CONSOLE_SECRET not set (run via doppler)", file=sys.stderr); sys.exit(2)
    reqs = _get_pending(secret, args.limit)
    print(f"pending={len(reqs)}")
    if args.dry:
        for r in reqs: print("  ", r); return
    for r in reqs:
        st = fulfill_one(r, synth=_real_synth, publish=_real_publish)
        _complete(secret, r["id"], st)
        print(f"  req {r['id']} ({r['email']} {r['scan_date']}) -> {st}")


if __name__ == "__main__":
    main()
```

> Implementer note: read `~/AI-Training/02 Skills/e4l-reveal-push.py` and confirm the EXACT `synthesize_reveal_layers` return → `publish_to_portal` payload wiring it already uses (the payload shape publish expects). Adapt `_real_synth`/`_real_publish` to match that working invocation verbatim — do not guess the payload contract.

- [ ] **Step 2: Verify** — unit test `fulfill_one` with injected `synth`/`publish` (done on success, failed when synth returns None / publish False / raises). Write at `~/AI-Training/02 Skills/test_e4l_analysis_fulfill.py`; run `python3 -m pytest`. Then a `--dry` run: `doppler run -p remedy-match -c prd -- python3 "$HOME/AI-Training/02 Skills/e4l-analysis-fulfill.py" --dry` (lists pending; makes no synthesis). Report counts.

- [ ] **Step 3: (No git commit — vault-local.)** Report the file path, the `fulfill_one` test result, and `--dry` output. Wiring into `e4l-email-trigger.sh` is a one-line go-live step the controller confirms with Glen — do NOT edit the trigger in this task.

---

## Self-Review

**Spec coverage:**
- `analysis_requests` queue → Task 1. ✓
- `notified_at` bookkeeping → Task 2. ✓
- Request endpoint (household-auth, idempotent, quota on scan owner, paid bypass) + `available_scans.requested` → Task 3. ✓
- One-click page (confirm-to-claim, no prefetch abuse) → Task 4. ✓
- Proactive new-scan email (anti-nag, once, cc caregivers, suppression) → Task 5. ✓
- Portal Request button + states → Task 6. ✓
- Worker console endpoints → Task 7. ✓
- Local worker (synthesize+publish, piggyback 5-min) → Task 8 (vault). ✓
- Behind `SCAN_REQUEST_ENABLED`, existing published-report flow untouched → Global Constraints + per-task flag guards. ✓

**Placeholder scan:** Tasks 5 & 8 carry implementer notes to confirm the real portal-token helper and the exact `synthesize_reveal_layers`→`publish_to_portal` payload wiring (from the working `e4l-reveal-push.py`) — "confirm the real name/contract," not TBDs. No hand-waves.

**Type consistency:** `create_request -> {created,status}` / `statuses_for -> {date:status}` / `pending -> [{id,email,scan_id,scan_date}]` (Task 1) consumed by Tasks 3 (endpoint + requested flag), 7 (worker endpoints), 8 (worker). `_request_analysis_core` (Task 3) reused by Task 4's page. `client_scans.unnotified/mark_notified` (Task 2) consumed by Task 5. `available_scans` items gain `requested` (Task 3) consumed by Task 6. `SCAN_REQUEST_ENABLED` gates Tasks 3-6; the quota reuse (`analysis_quota`) and `_is_paid_member` are the same as the existing flow.
