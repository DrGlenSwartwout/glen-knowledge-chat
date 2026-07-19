# Portal-Reveal Slice 3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** A dry-run-default, idempotent console endpoint that provisions a bare portal (no System B report, no email) for every existing `biofield_reveals` email that lacks one.

**Architecture:** A testable `backfill_portals(cx, commit, limit)` in a new `dashboard/portal_backfill.py`, wrapped by a thin console endpoint. Provision-only — sends nothing.

**Tech Stack:** Python 3.11 Flask, sqlite (`chat_log.db`/`LOG_DB`), pytest.

## Global Constraints

- **No emails** anywhere in this slice.
- **No `portal_biofield_reports` write** — provision ONLY via `client_portal.ensure_token`.
- **Dry-run default**: `commit=False` reports counts and writes nothing.
- **Idempotent**: emails that already have a `client_portals` row are skipped and counted as `already`; safe to re-run.
- **Bounded**: `limit` caps NEW portals provisioned per run; the rest are `remaining`.
- Endpoint provisions under `_db_lock` and is console/owner gated.

---

### Task 1: `backfill_portals` core

**Files:**
- Create: `dashboard/portal_backfill.py`
- Test: `tests/test_portal_backfill.py`

**Interfaces:**
- Produces: `backfill_portals(cx, commit=False, limit=None) -> dict` with keys `reveal_emails, already, provisioned, remaining, committed`.

- [ ] **Step 1: Write the failing tests**

```python
import sqlite3
from dashboard import portal_backfill as pb
from dashboard import client_portal as cp
from dashboard import biofield_reveals as br
from dashboard import portal_biofield_reports as pbr

def _db():
    cx = sqlite3.connect(":memory:")
    cp.init_client_portal_table(cx); br.init_table(cx); pbr.init_table(cx)
    return cx

def _seed(cx, *emails):
    for i, e in enumerate(emails):
        br.upsert(cx, e, f"2026-07-{10+i:02d}", {}, [], "t")

def test_dry_run_reports_counts_and_writes_nothing():
    cx = _db(); _seed(cx, "a@x.com", "b@x.com")
    cp.ensure_token(cx, "a@x.com", "")                       # a already has a portal
    res = pb.backfill_portals(cx, commit=False)
    assert res == {"reveal_emails": 2, "already": 1, "provisioned": 0,
                   "remaining": 1, "committed": False}
    assert cx.execute("SELECT 1 FROM client_portals WHERE email='b@x.com'").fetchone() is None

def test_commit_provisions_bare_portals_and_no_report():
    cx = _db(); _seed(cx, "a@x.com", "b@x.com")
    res = pb.backfill_portals(cx, commit=True)
    assert res["provisioned"] == 2 and res["already"] == 0 and res["remaining"] == 0
    assert cx.execute("SELECT COUNT(*) FROM client_portals").fetchone()[0] == 2
    assert cx.execute("SELECT COUNT(*) FROM portal_biofield_reports").fetchone()[0] == 0

def test_rerun_is_idempotent():
    cx = _db(); _seed(cx, "a@x.com")
    pb.backfill_portals(cx, commit=True)
    res = pb.backfill_portals(cx, commit=True)
    assert res["provisioned"] == 0 and res["already"] == 1

def test_limit_caps_provisioning_and_reports_remaining():
    cx = _db(); _seed(cx, "a@x.com", "b@x.com", "c@x.com")
    res = pb.backfill_portals(cx, commit=True, limit=2)
    assert res["provisioned"] == 2 and res["remaining"] == 1
```

- [ ] **Step 2: Run — expect FAIL** (module missing)

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_portal_backfill.py -q`
Expected: import error / module not found.

- [ ] **Step 3: Implement**

Create `dashboard/portal_backfill.py`:

```python
"""Slice 3: provision-only backfill of client portals for existing reveal clients.
No emails. Idempotent. Dry-run by default."""
from dashboard import client_portal as _cp


def backfill_portals(cx, commit=False, limit=None):
    """Provision a bare portal (ensure_token — no System B report) for every
    biofield_reveals email that lacks one. Dry-run unless commit=True. `limit` caps
    NEW portals this run; the rest are counted in `remaining`. Never emails."""
    emails = [r[0] for r in cx.execute(
        "SELECT DISTINCT lower(email) FROM biofield_reveals "
        "WHERE email IS NOT NULL AND email <> '' ORDER BY lower(email)").fetchall()]
    already = provisioned = remaining = 0
    for email in emails:
        if cx.execute("SELECT 1 FROM client_portals WHERE email=?", (email,)).fetchone():
            already += 1
            continue
        if commit and (limit is None or provisioned < limit):
            _cp.ensure_token(cx, email, "")
            provisioned += 1
        else:
            remaining += 1
    return {"reveal_emails": len(emails), "already": already,
            "provisioned": provisioned, "remaining": remaining, "committed": bool(commit)}
```

- [ ] **Step 4: Run — expect PASS** (4 passed)

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_portal_backfill.py -q`

- [ ] **Step 5: Commit**

```bash
git add dashboard/portal_backfill.py tests/test_portal_backfill.py
git commit -m "feat(portal): backfill_portals — provision-only bare-portal migration (dry-run default)"
```

---

### Task 2: Console endpoint `POST /api/console/portal-backfill`

**Files:**
- Modify: `app.py` (add the route near other `/api/console/portal*` routes)

**Interfaces:**
- Consumes: `portal_backfill.backfill_portals` (Task 1), existing console auth, `_db_lock`, `LOG_DB`.

- [ ] **Step 1: Add the endpoint**

Find an existing console route to mirror auth (e.g. `api_console_biofield_review_queue` uses `if not _portal_console_ok(): return jsonify({"error": "unauthorized"}), 401`). Add:

```python
@app.route("/api/console/portal-backfill", methods=["POST"])
def api_console_portal_backfill():
    """Provision-only backfill: bare portals for reveal clients lacking one. No email.
    Dry-run unless commit is set. Owner/console gated. Idempotent."""
    if not _portal_console_ok():
        return jsonify({"error": "unauthorized"}), 401
    data = request.get_json(silent=True) or {}
    commit = str(data.get("commit") or request.args.get("commit") or "").strip().lower() in ("1", "true", "yes", "on")
    raw_limit = data.get("limit") or request.args.get("limit")
    try:
        limit = int(raw_limit) if raw_limit else None
    except (TypeError, ValueError):
        limit = None
    from dashboard import portal_backfill as _pb, client_portal as _cp, biofield_reveals as _br
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        _cp.init_client_portal_table(cx)
        _br.init_table(cx)
        res = _pb.backfill_portals(cx, commit=commit, limit=limit)
    return jsonify(res)
```

Confirm `_portal_console_ok` is the right auth helper by checking how `api_console_biofield_review_queue` gates; if the surrounding console routes use a different gate (e.g. `CONSOLE_SECRET` header check or `_owner_token_ok`), match that instead. Do NOT invent a new auth scheme.

- [ ] **Step 2: Compile**

Run: `~/.venvs/deploy-chat311/bin/python -m py_compile app.py` — clean.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat(portal): POST /api/console/portal-backfill — dry-run-default provision-only backfill"
```

---

## Verification notes
- Endpoint has no isolated unit test (needs app import); it's a thin wrapper over the unit-tested `backfill_portals` + a standard console gate. Post-merge, run it dry-run against prod first (reports counts), then `commit=true` (optionally with a `limit` to batch) to actually provision. It emails nothing, so it's safe to run and re-run.
