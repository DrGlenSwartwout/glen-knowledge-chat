# BOS Phase 6: CRM GHL-write sync-queue

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make GHL-write CRM actions work despite the Cloudflare WAF blocking GHL writes from Render. The actions (add tag, log outreach note, create opportunity, enroll workflow) ENQUEUE the write locally (works from the server, audited + governed by the dispatch spine); a local Mac drain script pushes the queued writes to GHL via curl (the residential IP is not blocked), mirroring the proven `sync-ghl-leads.py` pattern.

**Architecture:** A new `dashboard/ghl_queue.py` owns the `ghl_write_queue` table + enqueue/list/mark helpers + the four CRM enqueue actions (registered on the dispatch spine). `app.py` adds two drain endpoints (mirroring `/leads/pending-ghl` + `/leads/mark-ghl-synced`). A new local script `sync-ghl-writes.py` (mirroring `sync-ghl-leads.py`) drains the queue from the Mac.

**Builds on:** the merged Business OS (full Home board + Orders + Money + CRM signal). New branch `sess/ec0e1f15` off main, worktree `/tmp/wt-deploy-chat-ec0e1f15`.

**Constants** (from app.py:3215-3217, used by the drain script): `GHL_PIPELINE_ID="A6LWJMBoIsOFBMeCa6NY"`, `GHL_STAGE_NEW="397c5fb2-1612-4b7a-aa14-f0dac42a7fda"`, `GHL_WORKFLOW_ID="0b02dd3e-b82a-4032-a575-f9269afbd3ac"`.

---

## File Structure

- `dashboard/ghl_queue.py` (new): the queue table + enqueue/list_pending/mark_result + the four `crm.*` enqueue actions.
- `tests/test_bos_ghl_queue.py` (new): unit tests.
- `app.py` (modify): import the module + init the table at startup + two drain endpoints.
- `sync-ghl-writes.py` (new): the local Mac drain script.

---

## Task 1: The queue + enqueue actions (`dashboard/ghl_queue.py`)

**Files:**
- Create: `dashboard/ghl_queue.py`
- Test: `tests/test_bos_ghl_queue.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_bos_ghl_queue.py`:

```python
import sqlite3
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def _db():
    from dashboard import ghl_queue as Q
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    Q.init_ghl_queue_table(cx)
    return Q, cx


def test_enqueue_validates_and_lists():
    Q, cx = _db()
    qid = Q.enqueue(cx, op="tag_add", email="a@b.com", payload={"tag": "vip"}, actor="glen")
    assert qid > 0
    pend = Q.list_pending(cx)
    assert len(pend) == 1 and pend[0]["op"] == "tag_add" and pend[0]["status"] == "pending"
    import json
    assert json.loads(pend[0]["payload_json"]) == {"tag": "vip"}


def test_enqueue_rejects_bad_op_and_blank_email():
    Q, cx = _db()
    for bad in (lambda: Q.enqueue(cx, op="nope", email="a@b.com", payload={}),
                lambda: Q.enqueue(cx, op="note", email="", payload={})):
        try:
            bad(); assert False, "expected ValueError"
        except ValueError:
            pass


def test_mark_result_removes_from_pending():
    Q, cx = _db()
    qid = Q.enqueue(cx, op="note", email="a@b.com", payload={"note": "called"})
    Q.mark_result(cx, qid, "done", "ok")
    assert Q.list_pending(cx) == []
    row = cx.execute("SELECT status, result FROM ghl_write_queue WHERE id=?", (qid,)).fetchone()
    assert row["status"] == "done" and row["result"] == "ok"


def test_crm_add_tag_action_enqueues():
    Q, cx = _db()
    from dashboard import dispatch as D, events as E, rbac as R, actions as A
    E.init_event_tables(cx)
    assert A.get_action("crm.add_tag") is not None
    res = D.dispatch_action(cx, "crm.add_tag", {"email": "a@b.com", "tag": "warm"},
                            R.Actor(role=R.OWNER, name="glen"))
    assert res["status"] == "done"
    pend = Q.list_pending(cx)
    assert len(pend) == 1 and pend[0]["op"] == "tag_add" and pend[0]["email"] == "a@b.com"
    ev = E.list_events(cx, module="crm")
    assert ev and ev[0]["action_key"] == "crm.add_tag"


def test_crm_actions_registered():
    from dashboard import ghl_queue as Q  # noqa: F401
    from dashboard import actions as A
    for k in ("crm.add_tag", "crm.log_outreach", "crm.create_opportunity", "crm.enroll_workflow"):
        assert A.get_action(k) is not None, k
    # opportunity/workflow are owner/ops only
    assert A.get_action("crm.create_opportunity").permission == ("owner", "ops")
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_bos_ghl_queue.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'dashboard.ghl_queue'`).

- [ ] **Step 3: Write the implementation**

Create `dashboard/ghl_queue.py`:

```python
"""GHL write-queue: GHL writes are blocked from Render by the Cloudflare WAF, so
CRM write actions enqueue here (a local DB insert that works from the server) and
a local Mac drain script (sync-ghl-writes.py) pushes them to GHL via curl. The
actions are audited + governed by the dispatch spine like any other."""
import json
from datetime import datetime, timezone

from dashboard.actions import action, LOW_WRITE
from dashboard.rbac import OWNER, OPS, VA

OP_TYPES = ("tag_add", "tag_remove", "note", "opportunity", "workflow")


def _now():
    return datetime.now(timezone.utc).isoformat()


def init_ghl_queue_table(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS ghl_write_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            op TEXT NOT NULL,
            email TEXT,
            payload_json TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            result TEXT,
            actor TEXT,
            processed_at TEXT
        )
    """)
    cx.execute("CREATE INDEX IF NOT EXISTS idx_ghl_queue_status ON ghl_write_queue(status)")
    cx.commit()


def enqueue(cx, *, op, email, payload, actor=""):
    if op not in OP_TYPES:
        raise ValueError(f"unknown op: {op}")
    if not (email or "").strip():
        raise ValueError("email required")
    cur = cx.execute(
        "INSERT INTO ghl_write_queue (created_at, op, email, payload_json, status, actor) "
        "VALUES (?,?,?,?, 'pending', ?)",
        (_now(), op, email.strip(), json.dumps(payload or {}), actor or ""))
    cx.commit()
    return cur.lastrowid


def list_pending(cx, limit=100):
    cur = cx.execute(
        "SELECT * FROM ghl_write_queue WHERE status='pending' ORDER BY id ASC LIMIT ?",
        (limit,))
    return [dict(r) for r in cur.fetchall()]


def mark_result(cx, qid, status, result=""):
    cx.execute("UPDATE ghl_write_queue SET status=?, result=?, processed_at=? WHERE id=?",
               (status, str(result)[:500], _now(), qid))
    cx.commit()
    return True


def _enqueue_action(op, label):
    def _exec(params, ctx):
        cx = (ctx or {}).get("cx") or (params or {}).get("cx")
        if cx is None:
            raise ValueError("no db connection")
        init_ghl_queue_table(cx)
        email = (params.get("email") or "").strip()
        if not email:
            raise ValueError("email required")
        payload = {k: v for k, v in (params or {}).items()
                   if k not in ("email", "cx", "confirmed")}
        actor = (ctx or {}).get("actor")
        qid = enqueue(cx, op=op, email=email, payload=payload,
                      actor=getattr(actor, "name", "") if actor else "")
        return {"queue_id": qid, "op": op, "email": email,
                "message": f"{label} for {email} queued for GHL sync (#{qid})."}
    return _exec


action(key="crm.add_tag", module="crm", title="Add GHL tag",
       description="Tag a contact in GHL (queued; pushes on the next sync).",
       risk_tier=LOW_WRITE, permission=(OWNER, OPS, VA))(_enqueue_action("tag_add", "Tag"))
action(key="crm.log_outreach", module="crm", title="Log outreach note",
       description="Add a note to a GHL contact (queued).", risk_tier=LOW_WRITE,
       permission=(OWNER, OPS, VA))(_enqueue_action("note", "Note"))
action(key="crm.create_opportunity", module="crm", title="Create opportunity",
       description="Create a pipeline opportunity in GHL (queued).", risk_tier=LOW_WRITE,
       permission=(OWNER, OPS))(_enqueue_action("opportunity", "Opportunity"))
action(key="crm.enroll_workflow", module="crm", title="Enroll in workflow",
       description="Enroll a contact in the onboarding workflow (queued).", risk_tier=LOW_WRITE,
       permission=(OWNER, OPS))(_enqueue_action("workflow", "Workflow enrollment"))
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_bos_ghl_queue.py -q`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add dashboard/ghl_queue.py tests/test_bos_ghl_queue.py
git commit -m "feat(bos): GHL write-queue + crm enqueue actions (tag/note/opportunity/workflow)"
```

---

## Task 2: app.py wiring + drain endpoints (verified under doppler)

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Register + init at startup** (in the BOS startup block, near `import dashboard.crm as _bos_crm`):

```python
import dashboard.ghl_queue as _bos_ghl_queue  # noqa: F401 (registers crm enqueue actions)


def _init_bos_ghl_queue():
    cx = _sqlite3.connect(LOG_DB)
    try:
        _bos_ghl_queue.init_ghl_queue_table(cx)
    finally:
        cx.close()


_init_bos_ghl_queue()
```

- [ ] **Step 2: Add the two drain endpoints** (near the existing `/leads/pending-ghl` route, ~app.py:5575, reusing its webhook-secret-or-console-key auth pattern):

```python
def _ghl_queue_auth():
    ws = os.environ.get("WEBHOOK_SECRET", "")
    cs = os.environ.get("CONSOLE_SECRET", "")
    given = request.headers.get("X-Webhook-Secret", "") or request.headers.get("X-Console-Key", "")
    return (ws and given == ws) or (cs and given == cs)


@app.route("/api/ghl/queue/pending", methods=["GET"])
def ghl_queue_pending():
    if not _ghl_queue_auth():
        return jsonify({"error": "unauthorized"}), 401
    cx = _sqlite3.connect(LOG_DB); cx.row_factory = _sqlite3.Row
    try:
        rows = _bos_ghl_queue.list_pending(cx, limit=int(request.args.get("limit", 100) or 100))
    except (TypeError, ValueError):
        rows = _bos_ghl_queue.list_pending(cx)
    finally:
        cx.close()
    return jsonify({"queue": rows, "count": len(rows)})


@app.route("/api/ghl/queue/result", methods=["POST"])
def ghl_queue_result():
    if not _ghl_queue_auth():
        return jsonify({"error": "unauthorized"}), 401
    data = request.get_json(force=True) or {}
    qid = data.get("id")
    status = data.get("status", "done")
    if not qid:
        return jsonify({"ok": False, "error": "id required"}), 400
    cx = _sqlite3.connect(LOG_DB)
    try:
        _bos_ghl_queue.mark_result(cx, int(qid), status, data.get("result", ""))
    finally:
        cx.close()
    return jsonify({"ok": True})
```

- [ ] **Step 3: Compile + verify under doppler**

Run: `python3 -m py_compile app.py` (OK).
Run:
```bash
doppler run -p remedy-match -c prd -- bash -c 'mkdir -p /tmp/bostest && DATA_DIR=/tmp/bostest python3 - <<PY
import app, sqlite3, json
from dashboard import actions as A, ghl_queue as Q, dispatch as D, rbac as R, events as E
for k in ("crm.add_tag","crm.log_outreach","crm.create_opportunity","crm.enroll_workflow"):
    assert A.get_action(k) is not None, "missing "+k
cx = sqlite3.connect(app.LOG_DB); cx.row_factory=sqlite3.Row
E.init_event_tables(cx)
res = D.dispatch_action(cx, "crm.add_tag", {"email":"verify@x.com","tag":"bos-verify"}, R.Actor(role=R.OWNER, name="glen"))
assert res["status"]=="done", res
# drain endpoint sees it
c = app.app.test_client(); key = app.dashboard.CONSOLE_SECRET or ""
r = c.get("/api/ghl/queue/pending", headers={"X-Console-Key": key})
assert r.status_code==200, r.status_code
assert any(q["email"]=="verify@x.com" for q in r.get_json()["queue"]), "not in pending"
print("GHL_QUEUE_OK", res["result"]["message"][:50])
PY'
rm -rf /tmp/bostest
```
Expected: `GHL_QUEUE_OK ...` no assertion error.

Run: `python3 -m pytest tests/test_bos_ghl_queue.py tests/test_bos_spine.py tests/test_bos_signals.py -q` (green).

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat(bos): GHL write-queue drain endpoints + startup init"
```

---

## Task 3: The local drain script (`sync-ghl-writes.py`)

**Files:**
- Create: `sync-ghl-writes.py`

This runs on Glen's Mac (not Render), like `sync-ghl-leads.py`. It is not run in CI; verify it imports/parses.

- [ ] **Step 1: Write the script**

Create `sync-ghl-writes.py`:

```python
#!/usr/bin/env python3
"""Run from LOCAL Mac (not Render) to drain the GHL write-queue.
Render's AWS IP is blocked by GHL's Cloudflare WAF; the Mac's residential IP is
not. Uses curl to also bypass JA3 TLS fingerprint blocking. Mirrors
sync-ghl-leads.py.

Usage:
  doppler run --project remedy-match --config prd -- python3 sync-ghl-writes.py [--dry-run]
"""
import json
import os
import subprocess
import sys
import urllib.parse

RENDER_URL     = os.environ.get("PUBLIC_BASE_URL", "https://illtowell.com").rstrip("/")
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "")
GHL_API_KEY    = os.environ.get("GHL_API_KEY", "")
GHL_BASE       = "https://rest.gohighlevel.com/v1"
GHL_PIPELINE_ID = "A6LWJMBoIsOFBMeCa6NY"
GHL_STAGE_NEW   = "397c5fb2-1612-4b7a-aa14-f0dac42a7fda"
GHL_WORKFLOW_ID = "0b02dd3e-b82a-4032-a575-f9269afbd3ac"
DRY_RUN = "--dry-run" in sys.argv


def _curl(method, url, headers=None, payload=None):
    cmd = ["curl", "-s", "-X", method, url]
    if payload is not None:
        cmd += ["-H", "Content-Type: application/json", "-d", json.dumps(payload)]
    for k, v in (headers or {}).items():
        cmd += ["-H", f"{k}: {v}"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    try:
        return json.loads(r.stdout), None
    except Exception:
        return None, (r.stdout or "")[:200]


def _ghl_headers():
    return {"Authorization": f"Bearer {GHL_API_KEY}"}


def _find_contact(email):
    data, _ = _curl("GET", f"{GHL_BASE}/contacts/lookup?email={urllib.parse.quote(email)}",
                    _ghl_headers())
    contacts = (data or {}).get("contacts") or []
    return contacts[0] if contacts else None


def _do_op(item):
    """Execute one queued GHL write. Returns (ok, message)."""
    op = item["op"]
    email = item["email"]
    payload = json.loads(item.get("payload_json") or "{}")
    contact = _find_contact(email)
    if not contact and op != "tag_add":
        return False, "contact not found"
    cid = contact["id"] if contact else None

    if op in ("tag_add", "tag_remove"):
        tag = payload.get("tag") or payload.get("tags")
        tags = set((contact or {}).get("tags") or [])
        for t in ([tag] if isinstance(tag, str) else (tag or [])):
            if op == "tag_add":
                tags.add(t)
            else:
                tags.discard(t)
        if not cid:  # create the contact with the tag
            data, err = _curl("POST", f"{GHL_BASE}/contacts/",
                              _ghl_headers(), {"email": email, "tags": sorted(tags)})
            return (err is None), (err or "created")
        _, err = _curl("PUT", f"{GHL_BASE}/contacts/{cid}", _ghl_headers(),
                       {"tags": sorted(tags)})
        return (err is None), (err or "tagged")

    if op == "note":
        _, err = _curl("POST", f"{GHL_BASE}/contacts/{cid}/notes", _ghl_headers(),
                       {"body": payload.get("note", "")})
        return (err is None), (err or "noted")

    if op == "opportunity":
        _, err = _curl("POST", f"{GHL_BASE}/pipelines/{GHL_PIPELINE_ID}/opportunities",
                       _ghl_headers(), {"stageId": GHL_STAGE_NEW, "contactId": cid,
                                        "title": payload.get("title", email), "status": "open"})
        return (err is None), (err or "opportunity created")

    if op == "workflow":
        _, err = _curl("POST", f"{GHL_BASE}/contacts/{cid}/workflow/{GHL_WORKFLOW_ID}",
                       _ghl_headers(), {})
        return (err is None), (err or "enrolled")

    return False, f"unknown op {op}"


def _render(method, path, payload=None):
    return _curl(method, f"{RENDER_URL}{path}",
                 {"X-Webhook-Secret": WEBHOOK_SECRET}, payload)


def main():
    if not (GHL_API_KEY and WEBHOOK_SECRET):
        print("Set GHL_API_KEY + WEBHOOK_SECRET (use doppler run ...)")
        sys.exit(1)
    data, err = _render("GET", "/api/ghl/queue/pending")
    items = (data or {}).get("queue") or []
    print(f"Found {len(items)} queued GHL writes")
    done = failed = 0
    for it in items:
        print(f"  [{it['id']}] {it['op']} {it['email']}")
        if DRY_RUN:
            continue
        ok, msg = _do_op(it)
        _render("POST", "/api/ghl/queue/result",
                {"id": it["id"], "status": "done" if ok else "failed", "result": msg})
        print(f"    -> {'done' if ok else 'FAILED'}: {msg}")
        done += ok
        failed += (not ok)
    if not DRY_RUN:
        print(f"\nDone: {done} synced, {failed} failed")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it parses**

Run: `python3 -m py_compile sync-ghl-writes.py`
Expected: OK.

- [ ] **Step 3: Commit**

```bash
git add sync-ghl-writes.py
git commit -m "feat(bos): local GHL write-queue drain script (mirrors sync-ghl-leads)"
```

---

## Self-Review

**Spec coverage:** GHL-write CRM actions now WORK end to end despite the WAF: enqueue from the server (audited + governed), drain from the Mac. `crm.add_tag`, `crm.log_outreach`, `crm.create_opportunity`, `crm.enroll_workflow` all register on the dispatch spine and queue; the drain endpoints + script push to GHL.

**Permissions:** tag/note are owner/ops/va; opportunity/workflow are owner/ops (pipeline/workflow changes are higher-stakes).

**Production-only:** the actual GHL push runs on the Mac (residential IP + real GHL key). The enqueue + drain endpoints are server-side and fully testable; the live GHL calls in `sync-ghl-writes.py` mirror the proven `sync-ghl-leads.py`.

**Follow-up (not in scope):** scheduling the drain (a launchd plist every N minutes, like the CNS watcher); a failed-queue alert on the CRM cell; board buttons to trigger these CRM actions (a CRM board UI).

**Placeholder scan:** none.

**Type consistency:** `enqueue`/`list_pending`/`mark_result`, the queue row keys, the action keys (`crm.add_tag`/`log_outreach`/`create_opportunity`/`enroll_workflow`), and the drain endpoint shapes (`{queue: [...]}`, `{id, status, result}`) are consistent across Tasks 1-3.
