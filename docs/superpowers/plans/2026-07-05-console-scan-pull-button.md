# Console Scan-Pull Button (Feature 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "Pull scan" control to the Biofield Reveals console toolbar that, given a client email / E4L client-id / name, has the Mac scrape that client's latest scan directly from E4L (bypassing E4L's email notification), ingest it, and push a silent reveal draft for review — fixing the class of miss where E4L never emails a scan (the Sean Luscombe case).

**Architecture:** A new `scan_pull_requests` queue table + four console endpoints in the deploy-chat web app (Render) mirror the existing `analysis_requests` queue exactly. A new Mac-side worker `scan-pull-fulfill.py` (wired into the existing 5-min `e4l-email-trigger.sh`, reusing its lock) polls the queue, resolves each query to exactly one E4L client, runs the proven `scrape-e4l-http.py → parse-e4l-scans.py → bulk-vectorize-e4l-scans.py` pipeline, then pushes a silent (`notify=False`) reveal draft via the same `e4l-reveal-push` library the analysis worker uses. The web app never touches `e4l.db`; all E4L access stays on the Mac.

**Tech Stack:** Python 3 + Flask (deploy-chat, `~/.venvs/deploy-chat311` venv), SQLite (`LOG_DB` = `$DATA_DIR/chat_log.db`), vanilla JS (console static HTML), launchd + Doppler (`remedy-match/prd`) on the Mac.

## Global Constraints

- **Two work locations.** deploy-chat edits happen ONLY in the worktree `/tmp/wt-deploy-chat-f0f9fd5b` (branch `sess/f0f9fd5b`). Vault scripts (`~/AI-Training/02 Skills/`) are edited directly (the vault is exempt from worktrees; hourly auto-snapshot covers it).
- **Silent drafts only.** Both the push and the reveal are `notify=False` — never email the client from this feature. Client notification stays the separate approve→send / "Send all approved un-notified" step.
- **Reuse, do not rebuild.** Synthesis + push come from `02 Skills/e4l-reveal-push.py` (`_resolve_scan`, `E`, `build_payload`, `post_reveal_draft`, `E4L_DB`, `CATALOG`). Scrape/parse/vectorize are existing `02 Skills/` scripts. The queue mirrors `dashboard/analysis_requests.py` + its endpoints verbatim in shape.
- **Auth is split by surface.** Console endpoints use `_portal_console_ok()` (X-Console-Key header or `?key=`). The reveal-draft ingest (`POST /api/e4l/reveal-draft`) uses a DIFFERENT gate (`X-Cron-Secret`/`CRON_SECRET`, falls back to `CONSOLE_SECRET`) — already handled inside `post_reveal_draft`; do not re-implement it.
- **Uniqueness key = normalized query while `pending`/`working`.** NOT `(email, scan_date)` — the scan_date is unknown until the pull runs.
- **Identity resolution order (Sean-class invariant):** all-digits → E4L client-id (exact); `@` → email → `e4l.db` lookup → exact id; else name → `--client-name` (most-recent-scan auto-pick). Ambiguous name (multiple `e4l.db` emails) or unseen email → fail with candidates; never silently guess.
- **UI copy rules (Glen):** no em dashes, no ALL CAPS, no "Hook:" labels in any user-facing string. Use commas/periods.
- **Dark launch.** Everything ships behind `SCAN_PULL_ENABLED` (OFF). The worker poll/complete endpoints stay `_portal_console_ok()`-gated only (not behind the flag) so the worker can run while the button is dark. Flip the flag on the Render dashboard after verify (render.yaml is not the live source).
- **Python interpreters on the Mac:** the worker runs under `~/.venvs/deploy-chat311/bin/python` with `DATA_DIR=$HOME/deploy-chat` (it imports `e4l-reveal-push`). It shells out to scrape/parse/vectorize using the system python `/Library/Developer/CommandLineTools/usr/bin/python3`, inheriting the Doppler env.

## File Structure

**deploy-chat (worktree `/tmp/wt-deploy-chat-f0f9fd5b`):**
- Create `dashboard/scan_pull_requests.py` — queue store (table + `create_request`/`pending`/`mark`/`get`).
- Create `tests/test_scan_pull_requests.py` — store unit tests + endpoint tests.
- Modify `app.py` — `_scan_pull_enabled()` helper; four `/api/console/scan-pull-requests*` endpoints; add `scan_pull_enabled` to the existing biofield-reveals GET payload.
- Modify `static/console-biofield-reveals.html` — the "Pull scan" input+button (left of "Send all") + poll JS + small CSS.

**Vault (`~/AI-Training/02 Skills/`, edited directly):**
- Create `scan-pull-fulfill.py` — the Mac worker (`resolve_client` + orchestration).
- Create `tests/test_scan_pull_resolve.py` (under the vault or alongside) — `resolve_client` unit tests.
- Modify `e4l-email-trigger.sh` — add the scan-pull-fulfill step.

---

### Task 1: `scan_pull_requests` queue store

**Files:**
- Create: `/tmp/wt-deploy-chat-f0f9fd5b/dashboard/scan_pull_requests.py`
- Test: `/tmp/wt-deploy-chat-f0f9fd5b/tests/test_scan_pull_requests.py`

**Interfaces:**
- Produces: `init_scan_pull_requests_table(cx)`; `create_request(cx, query, requested_by=None) -> {"created": bool, "id": int|None, "status": str|None}`; `pending(cx, limit=50) -> [{"id","query"}]`; `mark(cx, req_id, status, scan_id=None, draft_id=None, message=None)`; `get(cx, req_id) -> dict|None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scan_pull_requests.py
import sqlite3
from dashboard import scan_pull_requests as spr


def _cx():
    cx = sqlite3.connect(":memory:")
    spr.init_scan_pull_requests_table(cx)
    return cx


def test_create_pending_mark_get_roundtrip():
    cx = _cx()
    res = spr.create_request(cx, "luscombesean@gmail.com", "glen")
    assert res["created"] is True and res["status"] == "pending"
    rid = res["id"]
    assert spr.pending(cx) == [{"id": rid, "query": "luscombesean@gmail.com"}]
    spr.mark(cx, rid, "working")
    assert spr.pending(cx) == []  # working is not pending
    spr.mark(cx, rid, "done", scan_id="1037956", draft_id=52)
    row = spr.get(cx, rid)
    assert row["status"] == "done" and row["scan_id"] == "1037956" and row["draft_id"] == 52


def test_create_dedups_while_pending_or_working():
    cx = _cx()
    a = spr.create_request(cx, "Sean Luscombe")
    b = spr.create_request(cx, "sean luscombe")  # normalized dup
    assert b["created"] is False and b["id"] == a["id"] and b["status"] == "pending"
    spr.mark(cx, a["id"], "done")
    c = spr.create_request(cx, "Sean Luscombe")  # prior is done → new one allowed
    assert c["created"] is True and c["id"] != a["id"]


def test_blank_query_and_missing_get():
    cx = _cx()
    assert spr.create_request(cx, "   ")["created"] is False
    assert spr.get(cx, 9999) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-f0f9fd5b && python -m pytest tests/test_scan_pull_requests.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'dashboard.scan_pull_requests'`.

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/scan_pull_requests.py
"""Queue of console-initiated 'pull this client's latest scan from E4L' requests.

A local worker (02 Skills/scan-pull-fulfill.py) resolves each query to exactly one
E4L client, scrapes the scan, ingests it, and pushes a SILENT reveal draft for
Glen's console review, then marks the row done/failed. Unlike analysis_requests
(keyed on a KNOWN email+scan_date), a pull targets a client whose scan may not be
in e4l.db yet — so the dedup key is the normalized query while pending/working."""
import datetime
import sqlite3


def _now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _norm(q):
    return (q or "").strip().lower()


def init_scan_pull_requests_table(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS scan_pull_requests (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            query        TEXT NOT NULL,
            query_norm   TEXT NOT NULL,
            status       TEXT NOT NULL,       -- pending | working | done | failed
            requested_by TEXT,
            scan_id      TEXT,
            draft_id     INTEGER,
            message      TEXT,
            created_at   TEXT,
            updated_at   TEXT
        )
    """)
    cx.execute("CREATE INDEX IF NOT EXISTS ix_spr_status ON scan_pull_requests(status)")
    cx.commit()


def create_request(cx, query, requested_by=None):
    qn = _norm(query)
    if not qn:
        return {"created": False, "id": None, "status": None}
    row = cx.execute(
        "SELECT id, status FROM scan_pull_requests "
        "WHERE query_norm=? AND status IN ('pending','working') ORDER BY id DESC LIMIT 1",
        (qn,)).fetchone()
    if row:
        return {"created": False, "id": row[0], "status": row[1]}
    now = _now()
    cur = cx.execute(
        "INSERT INTO scan_pull_requests (query, query_norm, status, requested_by, created_at, updated_at) "
        "VALUES (?,?, 'pending', ?, ?, ?)",
        (query.strip(), qn, (requested_by or None), now, now))
    cx.commit()
    return {"created": True, "id": cur.lastrowid, "status": "pending"}


def pending(cx, limit=50):
    rows = cx.execute(
        "SELECT id, query FROM scan_pull_requests WHERE status='pending' ORDER BY id LIMIT ?",
        (int(limit),)).fetchall()
    return [{"id": r[0], "query": r[1]} for r in rows]


def mark(cx, req_id, status, scan_id=None, draft_id=None, message=None):
    cx.execute(
        "UPDATE scan_pull_requests SET status=?, "
        "scan_id=COALESCE(?,scan_id), draft_id=COALESCE(?,draft_id), "
        "message=COALESCE(?,message), updated_at=? WHERE id=?",
        (status, (str(scan_id) if scan_id is not None else None),
         draft_id, message, _now(), req_id))
    cx.commit()


def get(cx, req_id):
    r = cx.execute(
        "SELECT id, query, status, requested_by, scan_id, draft_id, message, created_at, updated_at "
        "FROM scan_pull_requests WHERE id=?", (req_id,)).fetchone()
    if not r:
        return None
    return {"id": r[0], "query": r[1], "status": r[2], "requested_by": r[3],
            "scan_id": r[4], "draft_id": r[5], "message": r[6],
            "created_at": r[7], "updated_at": r[8]}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /tmp/wt-deploy-chat-f0f9fd5b && python -m pytest tests/test_scan_pull_requests.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-f0f9fd5b
git add dashboard/scan_pull_requests.py tests/test_scan_pull_requests.py
git commit -m "feat: scan_pull_requests queue store"
```

---

### Task 2: Console endpoints + flag + reveals-payload field

**Files:**
- Modify: `/tmp/wt-deploy-chat-f0f9fd5b/app.py` (add `_scan_pull_enabled()` near the other `_*_enabled()` helpers; add 4 routes near the analysis-requests routes ~app.py:10479-10503; add one field to `api_console_biofield_reveals()` ~app.py:12670-12695)
- Test: `/tmp/wt-deploy-chat-f0f9fd5b/tests/test_scan_pull_requests.py` (append endpoint tests)

**Interfaces:**
- Consumes: `dashboard.scan_pull_requests` (Task 1); existing `_portal_console_ok()` (app.py:14128), `_db_lock`, `LOG_DB`, `sqlite3`, `os`.
- Produces routes: `POST/GET /api/console/scan-pull-requests`, `POST /api/console/scan-pull-requests/<int:req_id>/complete`, `GET /api/console/scan-pull-requests/<int:req_id>`; helper `_scan_pull_enabled()`; payload key `scan_pull_enabled` on the biofield-reveals GET.

- [ ] **Step 1: Write the failing endpoint tests**

```python
# append to tests/test_scan_pull_requests.py
import importlib, sys
from pathlib import Path
import pytest


def _app(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)   # auth open in test
    monkeypatch.setenv("SCAN_PULL_ENABLED", "1")
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        import app as appmod
        importlib.reload(appmod)
    except Exception as e:
        pytest.skip(f"app not importable: {e}")
    return appmod


def test_endpoints_enqueue_list_complete_get(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    c = appmod.app.test_client()
    r = c.post("/api/console/scan-pull-requests", json={"query": "luscombesean@gmail.com"})
    assert r.status_code == 200
    rid = r.get_json()["id"]
    assert rid
    lst = c.get("/api/console/scan-pull-requests?limit=50").get_json()
    assert any(x["id"] == rid and x["query"] == "luscombesean@gmail.com" for x in lst["requests"])
    done = c.post(f"/api/console/scan-pull-requests/{rid}/complete",
                  json={"status": "done", "scan_id": "1037956", "draft_id": 52})
    assert done.status_code == 200
    got = c.get(f"/api/console/scan-pull-requests/{rid}").get_json()["request"]
    assert got["status"] == "done" and got["draft_id"] == 52
    # completed → no longer pending
    assert c.get("/api/console/scan-pull-requests").get_json()["requests"] == []


def test_enqueue_requires_query_and_flag(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    c = appmod.app.test_client()
    assert c.post("/api/console/scan-pull-requests", json={"query": ""}).status_code == 400
    # flag off → inert (no row created)
    monkeypatch.setenv("SCAN_PULL_ENABLED", "0")
    importlib.reload(appmod)
    c2 = appmod.app.test_client()
    r = c2.post("/api/console/scan-pull-requests", json={"query": "x@y.com"})
    assert r.get_json().get("status") == "disabled"


def test_reveals_payload_exposes_flag(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    c = appmod.app.test_client()
    body = c.get("/api/console/biofield-reveals").get_json()
    assert body.get("scan_pull_enabled") is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /tmp/wt-deploy-chat-f0f9fd5b && python -m pytest tests/test_scan_pull_requests.py -q`
Expected: the three endpoint tests FAIL with 404 (routes not registered) / missing `scan_pull_enabled` key.

- [ ] **Step 3: Add the flag helper**

Find where sibling flag helpers live (e.g. `_scan_request_enabled`, `_household_view_enabled`). Add:

```python
def _scan_pull_enabled():
    return (os.environ.get("SCAN_PULL_ENABLED", "") or "").strip().lower() in ("1", "true", "yes", "on")
```

- [ ] **Step 4: Add the four routes** (place next to `api_console_analysis_requests` ~app.py:10479)

```python
@app.route("/api/console/scan-pull-requests", methods=["POST"])
def api_console_scan_pull_create():
    """Owner console: enqueue a 'pull this client's latest scan from E4L' request.
    Behind SCAN_PULL_ENABLED — off means inert (no DB change)."""
    if not _portal_console_ok():
        return jsonify({"error": "unauthorized"}), 401
    if not _scan_pull_enabled():
        return jsonify({"ok": True, "status": "disabled"})
    from dashboard import scan_pull_requests as _spr
    query = ((request.get_json(silent=True) or {}).get("query") or "").strip()
    if not query:
        return jsonify({"ok": False, "error": "query required"}), 400
    requested_by = request.headers.get("X-Console-User", "") or None
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        _spr.init_scan_pull_requests_table(cx)
        res = _spr.create_request(cx, query, requested_by)
    return jsonify({"ok": True, "id": res["id"], "status": res["status"]})


@app.route("/api/console/scan-pull-requests", methods=["GET"])
def api_console_scan_pull_list():
    """Owner tool: list pending scan-pull requests for the local worker."""
    if not _portal_console_ok():
        return jsonify({"error": "unauthorized"}), 401
    from dashboard import scan_pull_requests as _spr
    with sqlite3.connect(LOG_DB) as cx:
        _spr.init_scan_pull_requests_table(cx)
        reqs = _spr.pending(cx, int(request.args.get("limit", 50)))
    return jsonify({"ok": True, "requests": reqs})


@app.route("/api/console/scan-pull-requests/<int:req_id>/complete", methods=["POST"])
def api_console_scan_pull_complete(req_id):
    """Owner tool: the local worker marks a request done (or failed) with result."""
    if not _portal_console_ok():
        return jsonify({"error": "unauthorized"}), 401
    from dashboard import scan_pull_requests as _spr
    body = request.get_json(silent=True) or {}
    status = (body.get("status") or "done").strip()
    if status not in ("done", "failed", "working"):
        status = "done"
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        _spr.init_scan_pull_requests_table(cx)
        _spr.mark(cx, req_id, status, scan_id=body.get("scan_id"),
                  draft_id=body.get("draft_id"), message=body.get("message"))
    return jsonify({"ok": True, "status": status})


@app.route("/api/console/scan-pull-requests/<int:req_id>", methods=["GET"])
def api_console_scan_pull_get(req_id):
    """Owner console: poll a single scan-pull request's status."""
    if not _portal_console_ok():
        return jsonify({"error": "unauthorized"}), 401
    from dashboard import scan_pull_requests as _spr
    with sqlite3.connect(LOG_DB) as cx:
        _spr.init_scan_pull_requests_table(cx)
        row = _spr.get(cx, req_id)
    if not row:
        return jsonify({"ok": False, "error": "not found"}), 404
    return jsonify({"ok": True, "request": row})
```

- [ ] **Step 5: Expose the flag in the biofield-reveals GET** (in `api_console_biofield_reveals()` ~app.py:12670, where it builds the `{drafts, approved}` response, add the key):

```python
    # existing return builds {"drafts": ..., "approved": ...}; add the flag so the
    # console only renders the Pull-scan control when the feature is on.
    return jsonify({"ok": True, "drafts": drafts, "approved": approved,
                    "scan_pull_enabled": _scan_pull_enabled()})
```

(Match the existing return's exact keys/shape — add only `scan_pull_enabled`.)

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd /tmp/wt-deploy-chat-f0f9fd5b && python -m pytest tests/test_scan_pull_requests.py -q`
Expected: PASS (all tests). If `app not importable`, run under Doppler: `doppler run -p remedy-match -c prd -- python -m pytest tests/test_scan_pull_requests.py -q`.

- [ ] **Step 7: Commit**

```bash
cd /tmp/wt-deploy-chat-f0f9fd5b
git add app.py tests/test_scan_pull_requests.py
git commit -m "feat: console scan-pull-requests endpoints + SCAN_PULL_ENABLED flag"
```

---

### Task 3: Console "Pull scan" control + poll

**Files:**
- Modify: `/tmp/wt-deploy-chat-f0f9fd5b/static/console-biofield-reveals.html` (CSS ~lines 60-67; toolbar block ~lines 234-241 inside `loadList`; add `doPullScan`/`pollPull` near `doSendAll` ~line 769)

**Interfaces:**
- Consumes: existing `api(method, path, body)` helper (line ~101), `loadList()`, the `r.json.scan_pull_enabled` field (Task 2), the `.btn.ghost`/`.toolbar` CSS.
- Produces: the input+button+status control (rendered left of "Send all approved un-notified") and the two JS functions.

- [ ] **Step 1: Add CSS** (after the `.btn.ghost` rule ~line 67):

```css
  .pullwrap{ display:inline-flex; gap:6px; align-items:center; }
  .pullinput{ background:transparent; color:var(--fg); border:1px solid var(--line);
              border-radius:8px; padding:8px 10px; font:inherit; min-width:220px; }
  .pullstatus{ font-size:13px; color:var(--fg); opacity:.8; }
```

- [ ] **Step 2: Insert the control before `sendAllBtn` is appended** (in the toolbar block ~line 239, so it renders to the left — flex order is insertion order). Guard on the flag from the list payload:

```javascript
  // --- Pull-scan control (left of Send all), only when the feature is on ---
  if (window.__SCAN_PULL_ENABLED__) {
    var pullWrap = document.createElement('span');
    pullWrap.className = 'pullwrap';
    var pullInput = document.createElement('input');
    pullInput.type = 'text';
    pullInput.className = 'pullinput';
    pullInput.placeholder = 'email, E4L client id, or name';
    var pullBtn = document.createElement('button');
    pullBtn.className = 'btn ghost';
    pullBtn.textContent = 'Pull scan';
    var pullStatus = document.createElement('span');
    pullStatus.className = 'pullstatus';
    pullBtn.onclick = function(){ doPullScan(pullInput, pullBtn, pullStatus); };
    pullInput.addEventListener('keydown', function(e){
      if (e.key === 'Enter') { doPullScan(pullInput, pullBtn, pullStatus); }
    });
    pullWrap.appendChild(pullInput);
    pullWrap.appendChild(pullBtn);
    pullWrap.appendChild(pullStatus);
    toolbar.appendChild(pullWrap);
  }
  toolbar.appendChild(sendAllBtn);   // <-- existing line 240 stays; ensure it is AFTER the block above
```

- [ ] **Step 3: Set the flag from the list payload.** In `loadList()`, right after the `var r = await api('GET', '/api/console/biofield-reveals');` call succeeds, record the flag before the toolbar is built:

```javascript
  window.__SCAN_PULL_ENABLED__ = !!(r.json && r.json.scan_pull_enabled);
```

- [ ] **Step 4: Add the pull + poll functions** (near `doSendAll`, ~line 769). Note: no em dashes in copy.

```javascript
async function doPullScan(input, btn, st){
  var q = (input.value || '').trim();
  if(!q){ input.focus(); return; }
  btn.disabled = true; input.disabled = true;
  st.textContent = 'Queued, pulling from E4L. Usually a few minutes.';
  var r = await api('POST', '/api/console/scan-pull-requests', { query: q });
  if(!r.ok || !r.json || !r.json.id){
    st.textContent = (r.json && r.json.status === 'disabled')
      ? 'Scan pull is currently off.'
      : 'Could not queue (HTTP ' + r.status + ').';
    btn.disabled = false; input.disabled = false;
    return;
  }
  pollPull(r.json.id, input, btn, st, 0);
}

async function pollPull(id, input, btn, st, tries){
  if(tries > 30){   // about 5 minutes at 10s
    st.textContent = 'Still working. Refresh shortly.';
    btn.disabled = false; input.disabled = false;
    return;
  }
  var r = await api('GET', '/api/console/scan-pull-requests/' + id);
  var row = r.ok && r.json && r.json.request;
  if(row && row.status === 'done'){
    st.textContent = 'Pulled. Draft ready.';
    input.value = '';
    btn.disabled = false; input.disabled = false;
    loadList();
    return;
  }
  if(row && row.status === 'failed'){
    st.textContent = row.message || 'Pull failed.';
    btn.disabled = false; input.disabled = false;
    return;
  }
  setTimeout(function(){ pollPull(id, input, btn, st, tries + 1); }, 10000);
}
```

- [ ] **Step 5: Render-verify (not just inject).** With `SCAN_PULL_ENABLED=1` set for the running app, headless-render the console page and confirm the control renders left of "Send all". Use the claude-in-chrome skill (or any headless browser) to load `/console/biofield-reveals?key=<CONSOLE_SECRET>` and assert:
  - an `input.pullinput` with the placeholder exists,
  - the `Pull scan` button precedes the `Send all approved un-notified` button in DOM order,
  - with the flag OFF, neither appears.

Record the result (screenshot or DOM assertion) in the task notes.

- [ ] **Step 6: Commit**

```bash
cd /tmp/wt-deploy-chat-f0f9fd5b
git add static/console-biofield-reveals.html
git commit -m "feat: Pull scan control + status poll in Biofield Reveals console"
```

---

### Task 4: Mac worker `scan-pull-fulfill.py`

**Files:**
- Create: `~/AI-Training/02 Skills/scan-pull-fulfill.py`
- Test: `~/AI-Training/02 Skills/tests/test_scan_pull_resolve.py`

**Interfaces:**
- Consumes: `02 Skills/e4l-reveal-push.py` (`E4L_DB`, `_resolve_scan`, `E`, `fetch_history`, `CATALOG`, `build_payload`, `post_reveal_draft`); the prod endpoints from Task 2 (`GET`/`.../complete`); existing `scrape-e4l-http.py` (`--client <id>` / `--client-name <name>`), `parse-e4l-scans.py`, `bulk-vectorize-e4l-scans.py`.
- Produces: `resolve_client(query, db_path=None) -> dict` (the testable core); `fulfill_one(req) -> dict`; `main()`.

- [ ] **Step 1: Write the failing resolver test**

```python
# 02 Skills/tests/test_scan_pull_resolve.py
import importlib.util, os, sqlite3, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SKILLS = os.path.dirname(HERE)


def _load():
    spec = importlib.util.spec_from_file_location(
        "scan_pull_fulfill", os.path.join(SKILLS, "scan-pull-fulfill.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _seed(path):
    cx = sqlite3.connect(path)
    cx.execute("CREATE TABLE e4l_clients (client_id INTEGER, name TEXT, email TEXT)")
    cx.execute("CREATE TABLE e4l_scans (scan_id INTEGER, client_id INTEGER, scan_date TEXT)")
    cx.executemany("INSERT INTO e4l_clients VALUES (?,?,?)", [
        (97515, "Sean Luscombe", "luscombesean@gmail.com"),
        (21168, "Sean Surkes", "ssurkes@yahoo.com"),
        (25305, "Rae Luscombe", "suerae1111@gmail.com"),
        (317035, "Rae Luscombe", "suerae1111@hotmail.com"),  # same name, different email → ambiguous
    ])
    cx.executemany("INSERT INTO e4l_scans VALUES (?,?,?)", [
        (1037956, 97515, "2026-07-05"), (1013838, 97515, "2026-04-19"),
        (900001, 25305, "2026-06-01"), (900002, 317035, "2026-06-20"),
    ])
    cx.commit(); cx.close()


def test_resolve_digit_email_name(tmp_path):
    db = str(tmp_path / "e4l.db"); _seed(db)
    m = _load()
    # digit → exact id
    r = m.resolve_client("97515", db)
    assert r["ok"] and r["client_arg"] == ["--client", "97515"] and r["email"] == "luscombesean@gmail.com"
    # email present → exact id
    r = m.resolve_client("luscombesean@gmail.com", db)
    assert r["ok"] and r["client_arg"] == ["--client", "97515"] and r["client_id"] == 97515
    # unique name → id
    r = m.resolve_client("Sean Luscombe", db)
    assert r["ok"] and r["client_arg"] == ["--client", "97515"]


def test_resolve_ambiguous_and_unseen(tmp_path):
    db = str(tmp_path / "e4l.db"); _seed(db)
    m = _load()
    # ambiguous name (two emails) → failed with candidates
    r = m.resolve_client("Rae Luscombe", db)
    assert r["ok"] is False and "ambiguous" in r["message"].lower()
    # unseen email → failed
    r = m.resolve_client("nobody@nowhere.com", db)
    assert r["ok"] is False and "not on file" in r["message"].lower()
    # unknown name, zero local matches → fall back to --client-name (brand-new on E4L)
    r = m.resolve_client("Brand New Person", db)
    assert r["ok"] and r["client_arg"] == ["--client-name", "Brand New Person"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/AI-Training && python3 -m pytest "02 Skills/tests/test_scan_pull_resolve.py" -q`
Expected: FAIL — file `scan-pull-fulfill.py` does not exist.

- [ ] **Step 3: Write the worker** (`02 Skills/scan-pull-fulfill.py`)

```python
#!/usr/bin/env python3
"""Fulfill pending scan-pull requests: resolve each query to ONE E4L client,
scrape that client's latest scan from E4L (bypassing the email trigger), parse +
vectorize into e4l.db, then push a SILENT reveal draft (no client email) for
Glen's console review. Marks each row done/failed on the prod console endpoints.

Mirrors e4l-analysis-fulfill.py. The new capability vs that worker: it SCRAPES
E4L first, so it handles scans that were never emailed/ingested (the Sean case).

Run under the deploy-chat311 venv with DATA_DIR set (it imports e4l-reveal-push);
it shells out to scrape/parse/vectorize with the system python. CONSOLE_SECRET
from Doppler prd. Usage: [--dry] [--limit N]."""
import argparse
import datetime
import importlib.util
import json
import os
import re
import sqlite3
import subprocess
import sys
import urllib.request

BASE = os.environ.get("WEB_URL", "https://glen-knowledge-chat.onrender.com").rstrip("/")
SKILLS = os.path.dirname(os.path.abspath(__file__))
E4L_DB = os.path.expanduser(os.environ.get("E4L_DB", "~/AI-Training/e4l.db"))
SYS_PY = os.environ.get("E4L_SYS_PY", "/Library/Developer/CommandLineTools/usr/bin/python3")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

_RP = None  # cached e4l-reveal-push module


# ---- identity resolution (the testable core) --------------------------------

def _most_recent_client_id(cx, client_ids):
    best, best_key = None, None
    for cid in client_ids:
        r = cx.execute("SELECT MAX(scan_date) FROM e4l_scans WHERE client_id=?", (cid,)).fetchone()
        key = (r[0] or "", cid)
        if best_key is None or key > best_key:
            best, best_key = cid, key
    return best


def _candidates_str(cx, rows):
    parts = []
    for cid, name, email in rows:
        last = cx.execute("SELECT MAX(scan_date) FROM e4l_scans WHERE client_id=?", (cid,)).fetchone()[0]
        parts.append(f"{name} <{email or 'no-email'}> id={cid} last={last or 'none'}")
    return "; ".join(parts)


def resolve_client(query, db_path=None):
    """Resolve a console query to exactly one E4L client.
    OK  -> {"ok": True, "client_arg": [...], "client_id": int|None, "email": str, "name": str}
    ERR -> {"ok": False, "message": str}"""
    q = (query or "").strip()
    if not q:
        return {"ok": False, "message": "empty query"}
    cx = sqlite3.connect(f"file:{db_path or E4L_DB}?mode=ro", uri=True)
    try:
        if q.isdigit():
            row = cx.execute("SELECT name, email FROM e4l_clients WHERE client_id=?", (int(q),)).fetchone()
            return {"ok": True, "client_arg": ["--client", q], "client_id": int(q),
                    "email": (row[1] if row else "") or "", "name": (row[0] if row else "") or f"Client_{q}"}
        if EMAIL_RE.match(q):
            rows = cx.execute("SELECT client_id, name, email FROM e4l_clients WHERE lower(email)=lower(?)",
                              (q,)).fetchall()
            if not rows:
                return {"ok": False, "message": f"email {q} not on file locally; enter the E4L client name or id"}
            cid = _most_recent_client_id(cx, [r[0] for r in rows])
            row = next(r for r in rows if r[0] == cid)
            return {"ok": True, "client_arg": ["--client", str(cid)], "client_id": cid,
                    "email": row[2] or q, "name": row[1] or f"Client_{cid}"}
        rows = cx.execute("SELECT client_id, name, email FROM e4l_clients WHERE lower(name)=lower(?)",
                          (q,)).fetchall()
        emails = {(r[2] or "").lower() for r in rows if r[2]}
        if len(emails) > 1:
            return {"ok": False, "message": "ambiguous name; resend with the exact email or id: "
                    + _candidates_str(cx, rows)}
        if rows:
            cid = _most_recent_client_id(cx, [r[0] for r in rows])
            row = next(r for r in rows if r[0] == cid)
            return {"ok": True, "client_arg": ["--client", str(cid)], "client_id": cid,
                    "email": row[2] or "", "name": row[1] or q}
        return {"ok": True, "client_arg": ["--client-name", q], "client_id": None, "email": "", "name": q}
    finally:
        cx.close()


# ---- prod endpoint I/O (mirror e4l-analysis-fulfill.py) ----------------------

def _get_pending(secret, limit):
    req = urllib.request.Request(f"{BASE}/api/console/scan-pull-requests?limit={limit}",
                                 headers={"X-Console-Key": secret})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode()).get("requests", [])


def _complete(secret, req_id, status, **fields):
    body = {"status": status, **{k: v for k, v in fields.items() if v is not None}}
    req = urllib.request.Request(f"{BASE}/api/console/scan-pull-requests/{req_id}/complete",
                                 method="POST", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "X-Console-Key": secret})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


# ---- pipeline ---------------------------------------------------------------

def _load_reveal_push():
    global _RP
    if _RP is None:
        spec = importlib.util.spec_from_file_location(
            "e4l_reveal_push", os.path.join(SKILLS, "e4l-reveal-push.py"))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _RP = m
    return _RP


def _run(argv):
    print("  $", " ".join(argv), flush=True)
    p = subprocess.run(argv, capture_output=True, text=True, env=os.environ)
    if p.stdout:
        print("   ", p.stdout.strip()[-500:], flush=True)
    return p


def _email_for_client(cx, client_id):
    if client_id is None:
        return ""
    r = cx.execute("SELECT email FROM e4l_clients WHERE client_id=? AND email IS NOT NULL AND email!='' LIMIT 1",
                   (client_id,)).fetchone()
    return (r[0] if r else "") or ""


def synth_and_push(email):
    """Mirror e4l-analysis-fulfill._real_synth but notify=False; return (draft_id, scan_id)."""
    rp = _load_reveal_push()
    cx = sqlite3.connect(rp.E4L_DB)
    scan = rp._resolve_scan(cx, email, None)   # latest scan for this email
    if not scan:
        return (None, None)
    E = rp.E
    patterns = E.pull_patterns(cx, scan["scan_id"], limit=12)
    label_map = {p["item_code"]: (p.get("full_name") or p.get("name") or p["item_code"])
                 for p in patterns if p.get("item_code")}
    catalog = E.load_catalog(rp.CATALOG)
    synth = E.synthesize(patterns, history=rp.fetch_history(email), rules=E.load_rules(),
                         ff_names=E.curated_ff_names(catalog), layer_count=6)
    synth["layers"] = E.order_layers_by_pattern_count(synth.get("layers") or [])
    today = datetime.date.today().isoformat()
    content = E.to_portal_content(synth, catalog, formulation_map=E.load_formulation_map(cx),
                                  member_age=E.member_age_for_email(cx, email, today),
                                  age_rules=E.load_age_rules(cx))
    payload = rp.build_payload(content, email, scan["scan_date"], label_map=label_map, notify=False)
    if not payload:
        return (None, None)
    r = rp.post_reveal_draft(payload)
    did = (r or {}).get("id")
    return (did, scan["scan_id"]) if did is not None else (None, None)


def fulfill_one(req):
    res = resolve_client(req["query"])
    if not res.get("ok"):
        return {"status": "failed", "message": res.get("message")}
    _run([SYS_PY, os.path.join(SKILLS, "scrape-e4l-http.py")] + res["client_arg"])
    for _ in range(3):   # small retry like the email trigger
        _run([SYS_PY, os.path.join(SKILLS, "parse-e4l-scans.py")])
        break
    _run([SYS_PY, os.path.join(SKILLS, "bulk-vectorize-e4l-scans.py"), "--batch-size", "30"])
    email = res.get("email")
    if not email:
        cx = sqlite3.connect(f"file:{E4L_DB}?mode=ro", uri=True)
        try:
            email = _email_for_client(cx, res.get("client_id"))
        finally:
            cx.close()
    if not email:
        return {"status": "failed", "message": "no email on file for client after scrape"}
    draft_id, scan_id = synth_and_push(email)
    if not draft_id:
        return {"status": "failed", "message": "no scan on file / synthesis produced no layers"}
    return {"status": "done", "scan_id": scan_id, "draft_id": draft_id}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--limit", type=int, default=10)
    args = ap.parse_args()
    secret = os.environ.get("CONSOLE_SECRET", "")
    if not secret:
        print("CONSOLE_SECRET not set (run via doppler)", file=sys.stderr)
        sys.exit(2)
    reqs = _get_pending(secret, args.limit)
    print(f"pending={len(reqs)}")
    if args.dry:
        for r in reqs:
            print("  ", r)
        return
    for r in reqs:
        try:
            _complete(secret, r["id"], "working")
            out = fulfill_one(r)
            _complete(secret, r["id"], out["status"],
                      scan_id=out.get("scan_id"), draft_id=out.get("draft_id"),
                      message=out.get("message"))
            print(f"  req {r['id']} ({r['query']}) -> {out['status']} {out.get('message') or ''}")
        except Exception as e:
            print(f"  req {r.get('id')} error: {e!r}", flush=True)
            try:
                _complete(secret, r["id"], "failed", message=f"worker error: {e!r}")
            except Exception:
                pass


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the resolver test to verify it passes**

Run: `cd ~/AI-Training && python3 -m pytest "02 Skills/tests/test_scan_pull_resolve.py" -q`
Expected: PASS (2 tests, 6 cases).

- [ ] **Step 5: Integration smoke (real E4L, one client).** With Doppler prd + the venv, dry-run then a single real fulfill against a known client whose latest scan is already ingested (should still push/refresh a silent draft). Do NOT enqueue via prod yet — call `fulfill_one` directly:

```bash
cd ~/AI-Training
doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" \
  ~/.venvs/deploy-chat311/bin/python -c \
  'import importlib.util,os; s=importlib.util.spec_from_file_location("w",os.path.expanduser("~/AI-Training/02 Skills/scan-pull-fulfill.py")); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print(m.fulfill_one({"query":"luscombesean@gmail.com"}))'
```

Expected: prints `{'status': 'done', 'scan_id': ..., 'draft_id': ...}`; confirm a PENDING draft exists via `GET /api/console/biofield-reveals`. (Sean already has draft id=52 from the manual fix; a re-run upserts/refreshes the same email+scan_date row — that is the expected idempotent behavior.)

- [ ] **Step 6: Commit (vault auto-snapshots, but commit the script + test explicitly)**

```bash
cd ~/AI-Training
git add "02 Skills/scan-pull-fulfill.py" "02 Skills/tests/test_scan_pull_resolve.py"
git commit -m "feat(e4l): scan-pull-fulfill worker (scrape-first, silent reveal draft)"
```

---

### Task 5: Wire the worker into the 5-min trigger + go-live

**Files:**
- Modify: `~/AI-Training/02 Skills/e4l-email-trigger.sh` (add a step after Step 0 analysis-fulfill, ~line 90)

**Interfaces:**
- Consumes: Task 4's `scan-pull-fulfill.py`; the existing `$DOPPLER`, `$SKILLS`, `$LOG`, lock from the script.

- [ ] **Step 1: Add the scan-pull step** (immediately after the existing "fulfilling pending analysis requests" block):

```bash
# ── Step 0b: fulfill pending console scan-pull requests ──────────────────────
# The owner console "Pull scan" button enqueues a request to pull a specific
# client's latest scan straight from E4L (bypassing the scan-notification email —
# the Sean Luscombe case). Same runtime as e4l-reveal-push.py (deploy-chat311 venv
# + DATA_DIR). Self-gating: the queue is empty unless SCAN_PULL_ENABLED is on prod.
echo "$(ts) fulfilling pending scan-pull requests" >> "$LOG"
$DOPPLER env DATA_DIR="$HOME/deploy-chat" ~/.venvs/deploy-chat311/bin/python \
  "$SKILLS/scan-pull-fulfill.py" --limit 10 2>&1 | tail -10 >> "$LOG"
```

- [ ] **Step 2: Syntax-check the script**

Run: `zsh -n "$HOME/AI-Training/02 Skills/e4l-email-trigger.sh" && echo OK`
Expected: `OK`.

- [ ] **Step 3: Commit the wiring**

```bash
cd ~/AI-Training
git add "02 Skills/e4l-email-trigger.sh"
git commit -m "feat(e4l): run scan-pull-fulfill each email-trigger cycle"
```

- [ ] **Step 4: Ship the deploy-chat branch.** Push `sess/f0f9fd5b` and open a PR (or merge per the finishing-a-development-branch skill). Do NOT flip the flag yet.

```bash
cd /tmp/wt-deploy-chat-f0f9fd5b
git push -u origin sess/f0f9fd5b
gh pr create --fill --base main
```

- [ ] **Step 5: Go-live (after deploy).** On the Render dashboard for the web service, set `SCAN_PULL_ENABLED=1` (render.yaml is not the live source). Then verify end-to-end:
  - Open `/console/biofield-reveals` → the "Pull scan" control renders left of "Send all".
  - Enter a test client (e.g. `luscombesean@gmail.com`), click Pull scan.
  - Within one trigger cycle (≤5 min) the status flips to "Pulled. Draft ready." and the draft appears in Needs Review. (For an immediate check, run `scan-pull-fulfill.py` once by hand as in Task 4 Step 5.)
  - Confirm no client email was sent (the draft is silent; `notified_at` is null).

- [ ] **Step 6: Note deferred work.** Feature 2 (email-independent recency poll) is specified in the design doc and intentionally not built here.

---

## Self-Review

**Spec coverage:**
- Console button left of "Send all" → Task 3. ✓
- Enqueue + queue + worker poll/complete endpoints → Tasks 1, 2. ✓
- Identity resolution (email / client-id / name, ambiguity → candidates) → Task 4 `resolve_client` + tests. ✓
- Scrape-first → parse → vectorize → silent reveal draft → Task 4 `fulfill_one`/`synth_and_push` (`notify=False`). ✓
- Wire into 5-min trigger, reuse lock → Task 5. ✓
- Dark flag `SCAN_PULL_ENABLED`, ship dark then flip on Render → Tasks 2, 3, 5. ✓
- Manifest refresh for portal scan-history (sub-project A): NOTE — the spec lists this as a best-effort step. It is intentionally omitted from this plan to keep Feature 1 focused; the existing ingest cron already runs `e4l-scan-manifest-push.py` on its own cadence, so a pulled scan reaches `client_scans` within the normal window. If Glen wants it immediate, add a `_run` of the manifest push in `fulfill_one` after vectorize.

**Placeholder scan:** none — every code step has complete code; every run step has an exact command + expected output.

**Type consistency:** store fns (`create_request`/`pending`/`mark`/`get`) match between Task 1 (def) and Task 2 (calls). `resolve_client` return keys (`ok`/`client_arg`/`client_id`/`email`/`name`/`message`) match between Task 4 def, its tests, and `fulfill_one`. `_complete(status, scan_id, draft_id, message)` matches the endpoint body keys in Task 2. The list payload key `scan_pull_enabled` matches between Task 2 (server) and Task 3 (`window.__SCAN_PULL_ENABLED__`).
