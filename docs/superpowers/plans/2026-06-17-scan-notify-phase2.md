# Scan Notification — Phase 2 (engagement-gated processing) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Stop synthesizing every scan. Pre-process for engaged clients (instant on open); for cold clients, a portal tap enqueues a request that a tight local watcher fulfills (~6s) and publishes, ending the silent "auto-draft everyone."

**Architecture:** A server `portal_process_requests` queue bridges the remote portal and the local Mac. The portal posts a request when a cold client opens a still-`pending` portal; a local watcher polls, runs the importer, and publishes. The scan trigger's blanket auto-draft becomes engaged-only (reads Phase 1's `engaged` flag).

**Tech Stack:** Flask + sqlite (server), JS (portal), zsh/launchd + Python (local watcher). Spec: `docs/superpowers/specs/2026-06-17-scan-notify-ondemand-unfold-design.md` (Phase 2 = its components 4 + the cold producer; the polished unfold is Phase 3). Builds on Phase 1 (`notify_state.engaged`, `ensure_token` pending portals).

---

## Phasing note
Phase 2 ships the **complete cold path with a basic "preparing…" + auto-poll** so cold clients are never stranded when blanket auto-draft is retired. **Phase 3** replaces that basic state with the staged live-unfold animation. (The cold producer moved from Phase 3 to Phase 2 for exactly this reason.)

## Two repos
- **DEPLOY-CHAT** worktree `/tmp/wt-deploy-chat-5326cc61` branch `sess/5326cc61-phase2` (Tasks 1–3, 6). Suite: `doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" ~/.venvs/deploy-chat311/bin/python -m pytest -q`
- **VAULT** `~/AI-Training/02 Skills/` (Tasks 4–5, auto-snapshot).

**Baseline:** full suite 1769 passed, 2 skipped.

---

## Task 1: `portal_process_requests` queue — `dashboard/process_queue.py`

**Files:** Create `dashboard/process_queue.py`; Test `tests/test_process_queue.py`

- [ ] **Step 1: Failing test**

```python
import sqlite3
from dashboard import process_queue as Q


def _cx(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "t.db")); Q.init_table(cx); return cx


def test_enqueue_idempotent_list_and_done(tmp_path):
    cx = _cx(tmp_path)
    Q.enqueue(cx, "a@x.com", "2026-06-05")
    Q.enqueue(cx, "a@x.com", "2026-06-05")            # idempotent: still one pending row
    Q.enqueue(cx, "b@x.com", "")
    pend = Q.list_pending(cx)
    emails = sorted(p["email"] for p in pend)
    assert emails == ["a@x.com", "b@x.com"] and len(pend) == 2
    assert Q.mark_done(cx, "a@x.com") is True
    assert [p["email"] for p in Q.list_pending(cx)] == ["b@x.com"]
    assert Q.mark_done(cx, "nobody@x.com") is False


def test_done_then_reenqueue(tmp_path):
    cx = _cx(tmp_path)
    Q.enqueue(cx, "a@x.com", "")
    Q.mark_done(cx, "a@x.com")
    Q.enqueue(cx, "a@x.com", "2026-07-01")            # a later scan re-requests
    assert len(Q.list_pending(cx)) == 1
```

- [ ] **Step 2: Run → FAIL** (`-m pytest tests/test_process_queue.py -q`)

- [ ] **Step 3: Implement** — `dashboard/process_queue.py`:

```python
"""Cold-client synthesis request queue (Phase 2). One pending row per email; the
local watcher claims pending requests, runs the importer, and marks done."""
import datetime
import sqlite3


def _now():
    return datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"


def init_table(cx):
    cx.execute("""CREATE TABLE IF NOT EXISTS portal_process_requests (
        email TEXT PRIMARY KEY, scan_date TEXT, status TEXT DEFAULT 'pending',
        requested_at TEXT, updated_at TEXT)""")
    cx.commit()


def _norm(email):
    return (email or "").strip().lower()


def enqueue(cx, email, scan_date=""):
    init_table(cx)
    email = _norm(email)
    now = _now()
    cx.execute("""INSERT INTO portal_process_requests (email, scan_date, status, requested_at, updated_at)
                  VALUES (?,?, 'pending', ?, ?)
                  ON CONFLICT(email) DO UPDATE SET scan_date=excluded.scan_date,
                      status='pending', requested_at=?, updated_at=?""",
               (email, scan_date or "", now, now, now, now))
    cx.commit()


def list_pending(cx):
    init_table(cx)
    rows = cx.execute("SELECT email, scan_date, requested_at FROM portal_process_requests "
                      "WHERE status='pending' ORDER BY requested_at ASC").fetchall()
    return [{"email": r[0], "scan_date": r[1], "requested_at": r[2]} for r in rows]


def mark_done(cx, email):
    init_table(cx)
    cur = cx.execute("UPDATE portal_process_requests SET status='done', updated_at=? "
                     "WHERE email=? AND status='pending'", (_now(), _norm(email)))
    cx.commit()
    return cur.rowcount > 0
```

- [ ] **Step 4: Run → PASS.** **Step 5: Commit** (`-m "notify P2: portal_process_requests queue"`)

---

## Task 2: process-request endpoints — portal enqueue + admin list/done

**Files:** Modify `app.py`; Test `tests/test_process_queue_routes.py`

- [ ] **Step 1: Failing tests** (reuse `client` + `_seed_portal` like other route tests)

```python
def test_portal_process_request_enqueues(client):
    c, appmod = client
    tok = _seed_portal(appmod, "pr@y.com", "PR", {"biofield_status": "pending"})
    assert c.post(f"/api/portal/{tok}/process-request").status_code == 200
    j = c.get("/api/admin/process-requests?key=test-secret").get_json()
    assert any(x["email"] == "pr@y.com" for x in j["pending"])


def test_admin_mark_done(client):
    c, appmod = client
    tok = _seed_portal(appmod, "pd@y.com", "PD", {"biofield_status": "pending"})
    c.post(f"/api/portal/{tok}/process-request")
    assert c.post("/api/admin/process-request/done?key=test-secret",
                  json={"email": "pd@y.com"}).status_code == 200
    j = c.get("/api/admin/process-requests?key=test-secret").get_json()
    assert all(x["email"] != "pd@y.com" for x in j["pending"])


def test_admin_process_requests_requires_key(client):
    c, _ = client
    assert c.get("/api/admin/process-requests").status_code == 401
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement** — add to `app.py` (near the other portal routes):

```python
@app.route("/api/portal/<token>/process-request", methods=["POST"])
def api_portal_process_request(token):
    from dashboard import client_portal as _cp, process_queue as _pq
    scan_date = ((request.get_json(silent=True) or {}).get("scan_date") or "").strip()
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        portal = _cp.get_portal_by_token(cx, token)
        if not portal:
            return jsonify({"error": "not found"}), 404
        _pq.enqueue(cx, portal["email"], scan_date)
    return jsonify({"ok": True})


@app.route("/api/admin/process-requests", methods=["GET"])
def api_admin_process_requests():
    if not _portal_console_ok():
        return jsonify({"error": "unauthorized"}), 401
    from dashboard import process_queue as _pq
    with sqlite3.connect(LOG_DB) as cx:
        pending = _pq.list_pending(cx)
    return jsonify({"pending": pending})


@app.route("/api/admin/process-request/done", methods=["POST"])
def api_admin_process_request_done():
    if not _portal_console_ok():
        return jsonify({"error": "unauthorized"}), 401
    from dashboard import process_queue as _pq
    email = ((request.get_json(silent=True) or {}).get("email") or "").strip().lower()
    if not email:
        return jsonify({"error": "email required"}), 400
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        _pq.mark_done(cx, email)
    return jsonify({"ok": True})
```

- [ ] **Step 4: Run → PASS.** **Step 5: FULL suite.** **Step 6: Commit** (`-m "notify P2: process-request endpoints (portal enqueue + admin list/done)"`)

---

## Task 3: portal cold producer — `static/client-portal.html`

**Files:** Modify `static/client-portal.html`; no unit test (node --check + suite + manual)

- [ ] **Step 1:** In the biofield render, when the block's status is `"pending"` (a cold portal that has no analysis yet — `ensure_token` created it with `biofield_status:"pending"`), render a card: **"Preparing your analysis…"** with a subtle spinner/animation and a line "This takes about a minute — it'll appear right here." Do NOT render layers.
- [ ] **Step 2:** On first render of the `pending` state, POST once to `/api/portal/<seg>/biofield`-area: `fetch(\`/api/portal/${seg}/process-request\`, {method:"POST", headers:{"Content-Type":"application/json"}, body:"{}"})` — guard with a module flag so it fires only once per load. Then start a **poll**: every 6s call the existing `load()`; when the status is no longer `"pending"` (the watcher published the `ai_draft`), stop polling — the normal biofield render (patterns shown, remedies blurred) takes over.
- [ ] **Step 3:** Stop the poll after a sane cap (e.g. 20 tries / 2 min) → show "Still working — we'll text you the moment it's ready" (graceful; the request stays queued).
- [ ] **Step 4: Verify.** `node --check` the extracted script; run the full Python suite (served-page test still 200). Manual for reviewer: open a `pending` portal → "Preparing…" + a process-request is posted + it polls; once an `ai_draft` lands, the patterns appear. No emojis; brand styling.
- [ ] **Step 5: Commit** (`-m "portal: cold-client 'preparing' + process-request + poll"`)

---

## Task 4: local watcher — `02 Skills/scan-process-watcher.sh` (VAULT)

**Files:** Create `02 Skills/scan-process-watcher.sh` + `02 Skills/launchagents/com.glen.scan-process-watcher.plist`; vault auto-snapshots

- [ ] **Step 1:** Create `scan-process-watcher.sh` — a loop that, every ~6s, GETs `https://illtowell.com/api/admin/process-requests` (X-Console-Key), and for each pending `email`: runs `doppler run … e4l-portal-import.py --email "$EMAIL" --publish-draft` (synthesize + publish the `ai_draft`), then POSTs `/api/admin/process-request/done {email}`. Use the same `$DOPPLER`/`$PYTHON`/venv invocation as the other E4L scripts. Gate the whole loop behind a dark flag `SCAN_PROCESS_ENABLED` (exit immediately if unset). Single-instance lock (like the email-trigger's `LOCK`) so two watchers don't double-process. Log to `/tmp/scan-process-watcher.log`.
- [ ] **Step 2:** Create the launchd plist `com.glen.scan-process-watcher.plist` — a `KeepAlive` long-running agent (not StartInterval; the script self-loops with a `sleep 6`), `EnvironmentVariables` with `PATH`/`HOME` (+ `SCAN_PROCESS_ENABLED` when Glen flips it live). Mirror the existing `com.glen.e4l-email-trigger.plist` structure.
- [ ] **Step 3:** Verify: `zsh -n "02 Skills/scan-process-watcher.sh"` → valid; `plutil -lint "02 Skills/launchagents/com.glen.scan-process-watcher.plist"` → OK. Do NOT load it (stays off until Glen installs + flips `SCAN_PROCESS_ENABLED`).
- [ ] **Step 4:** Vault auto-snapshots.

---

## Task 5: engaged-gated pre-process — `02 Skills/e4l-email-trigger.sh` (VAULT)

**Files:** Modify `02 Skills/e4l-email-trigger.sh`; vault auto-snapshots

- [ ] **Step 1:** The current autodraft loop (behind `E4L_AUTODRAFT_ENABLED`) drafts EVERY new-scan client. Change it to **engaged-only**: for each new-scan client, after resolving email, query the engagement flag — `GET /api/admin/notify-state` is a POST that returns `engaged`? No — use a tiny check: POST `/api/admin/notify-state {email}` returns the state used for notifications, but it does NOT return `engaged` directly. Add a lightweight read: extend the loop to call a one-liner that POSTs `/api/admin/notify-state` and reads a new `engaged` field. **Prerequisite:** add `"engaged": state["engaged"]` to the `api_admin_notify_state` response (one-line change in `app.py`, Task 2's file — include it in Task 2's endpoint OR here as a noted server tweak). For this vault task, assume `notify-state` returns `engaged`; gate the pre-synthesis on it:
```zsh
      ENGAGED=$($DOPPLER $PYTHON -c 'import os,json,sys,urllib.request; k=os.environ["CONSOLE_SECRET"]; req=urllib.request.Request("https://illtowell.com/api/admin/notify-state", data=json.dumps({"email":sys.argv[1]}).encode(), method="POST", headers={"Content-Type":"application/json","X-Console-Key":k}); print("1" if json.load(urllib.request.urlopen(req,timeout=20)).get("engaged") else "0")' "$EMAIL" 2>/dev/null)
      if [ "$ENGAGED" != "1" ]; then
        echo "$(ts) pre-process: '$NM' not engaged — skipping (will process on click)" >> "$LOG"
        continue
      fi
```
  Place this guard at the top of the existing autodraft `for NM` loop body (after resolving `$EMAIL`), so only engaged clients are pre-drafted. Keep the `E4L_AUTODRAFT_ENABLED` flag as the master switch but rename the log line to "pre-process".
- [ ] **Step 2:** `zsh -n "02 Skills/e4l-email-trigger.sh"` → valid.
- [ ] **Step 3:** Vault auto-snapshots.
- [ ] **NOTE for Task 2 implementer:** add `"engaged": _ns.get_state(...)["engaged"]` to the `api_admin_notify_state` JSON response (so this gate can read it). If Task 2 already shipped, make this a one-line follow-up commit in `app.py`.

---

## Task 6: validate + PR
- [ ] Full suite green. Push `sess/5326cc61-phase2`; open a PR (base main) "Scan notification Phase 2 (server+portal): process-request queue + cold producer". Note the vault watcher + engaged-gate ship via auto-snapshot, and go-live needs `SCAN_PROCESS_ENABLED` + installing the watcher launchd.
- [ ] **Manual end-to-end (after deploy):** create a `pending` portal (via `/api/admin/notify-state`), open it → "preparing" + a process-request appears in `/api/admin/process-requests`; run `scan-process-watcher.sh` once by hand with `SCAN_PROCESS_ENABLED=1` → it synthesizes, publishes, marks done; the portal poll then shows the analysis.

---

## Self-Review notes
- **Spec coverage (Phase 2):** process-request queue (T1); portal enqueue + admin list/done (T2); cold producer "preparing"+poll (T3); local watcher synthesize→publish→done (T4); engaged-gated pre-process replacing blanket auto-draft (T5). The polished unfold is Phase 3 (out of scope).
- **Type consistency:** queue row `{email, scan_date, status, requested_at}`; `enqueue/list_pending/mark_done`; admin list returns `{pending:[{email,scan_date,requested_at}]}`; the cold producer keys off `biofield_status == "pending"`; the engaged-gate reads `notify-state.engaged` (Task 2 adds that field).
- **Verify during impl:** the `api_admin_notify_state` `engaged` field add (T2/T5 dependency — do it in T2); the portal's `seg`/`load()` reuse (T3, same as the multi-scan tabs); confirm `ensure_token` placeholder content is `{"biofield_status":"pending"}` so the producer triggers; the watcher's lock + single-instance.
