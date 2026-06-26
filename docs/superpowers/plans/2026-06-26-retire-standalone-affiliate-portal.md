# Retire Standalone Affiliate Portal (2b-3, Option A) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Redirect the standalone `/affiliate/portal` + its login into the personal-portal login, and guarantee every approved affiliate has a `people` row (so none are locked out).

**Architecture:** A `backfill_affiliate_people` helper (reuses `customers.find_or_create_by_email`), a console-gated endpoint to run it, on-approval coverage in the affiliate approve/apply paths, and 302 redirects from the standalone affiliate portal/login → `/portal/login`.

**Tech Stack:** Python 3.11, Flask, sqlite3, pytest.

## Global Constraints

- Affiliate→people coverage REUSES `dashboard/customers.py: find_or_create_by_email(cx, *, email, name="", phone="")` (creates a `people` row keyed by email if absent; sets `source='order-entry'` — acceptable provenance, the greeting/self-login only need email+name). The hourly GHL sync does NOT prune people, so created rows persist.
- `backfill_affiliate_people` is idempotent + none-raising per-affiliate.
- On-approval coverage is best-effort (try/except → never breaks approve/apply).
- Keep public surfaces: `/affiliate`, `/affiliate/hub/<slug>`, `/affiliate/apply(-form)`, `/affiliate/portal-data`. Only the dashboard page + the affiliate login redirect away.
- `app.py` can't import offline → Tasks 2-4 verified live. Task 1 offline-TDD.
- Offline test cmd: `~/.venvs/deploy-chat311/bin/python -m pytest tests/<file> -v`.

---

### Task 1: `backfill_affiliate_people` helper

**Files:**
- Modify: `dashboard/affiliate_dashboard.py` (append)
- Test: `tests/test_affiliate_people_backfill.py`

**Interfaces:**
- Consumes: `customers.find_or_create_by_email`.
- Produces: `backfill_affiliate_people(cx) -> int`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_affiliate_people_backfill.py
import sqlite3
from dashboard import affiliate_dashboard as ad

def _cx():
    cx = sqlite3.connect(":memory:")
    cx.executescript("""
      CREATE TABLE affiliate_signups (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT,
        name TEXT, email TEXT, slug TEXT, token TEXT, status TEXT DEFAULT 'approved');
      CREATE TABLE people (id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT UNIQUE NOT NULL,
        first_name TEXT DEFAULT '', last_name TEXT DEFAULT '', name TEXT DEFAULT '',
        phone TEXT DEFAULT '', source TEXT DEFAULT '', created_at TEXT, updated_at TEXT);
    """)
    return cx

def test_creates_people_only_for_approved_missing():
    cx = _cx()
    cx.execute("INSERT INTO affiliate_signups (created_at,name,email,slug,token,status) VALUES "
               "('t','Has People','has@x.com','s1','t1','approved')")
    cx.execute("INSERT INTO affiliate_signups (created_at,name,email,slug,token,status) VALUES "
               "('t','Needs People','need@x.com','s2','t2','approved')")
    cx.execute("INSERT INTO affiliate_signups (created_at,name,email,slug,token,status) VALUES "
               "('t','Pending Guy','pend@x.com','s3','t3','pending')")
    cx.execute("INSERT INTO people (email, name, created_at, updated_at) VALUES ('has@x.com','Has People','t','t')")
    n = ad.backfill_affiliate_people(cx)
    assert n == 1
    emails = {r[0] for r in cx.execute("SELECT email FROM people").fetchall()}
    assert "need@x.com" in emails          # approved + was missing -> created
    assert "pend@x.com" not in emails      # pending -> skipped
    created = cx.execute("SELECT name FROM people WHERE email='need@x.com'").fetchone()
    assert created[0] == "Needs People"    # name carried from the affiliate row

def test_idempotent():
    cx = _cx()
    cx.execute("INSERT INTO affiliate_signups (created_at,name,email,slug,token,status) VALUES "
               "('t','A','a@x.com','s1','t1','approved')")
    assert ad.backfill_affiliate_people(cx) == 1
    assert ad.backfill_affiliate_people(cx) == 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_affiliate_people_backfill.py -v`
Expected: FAIL — `backfill_affiliate_people` missing.

- [ ] **Step 3: Implement** — append to `dashboard/affiliate_dashboard.py` (add `from dashboard import customers as _customers` near the top):

```python
def backfill_affiliate_people(cx):
    """Ensure every APPROVED affiliate has a people row (so they can self-login to
    the personal portal). Reuses customers.find_or_create_by_email. Idempotent;
    returns the count of people rows created. None-raising per affiliate."""
    rows = cx.execute(
        "SELECT email, name FROM affiliate_signups WHERE status='approved'").fetchall()
    created = 0
    for email, name in rows:
        em = (email or "").strip().lower()
        if not em:
            continue
        try:
            existing = cx.execute("SELECT 1 FROM people WHERE lower(email)=?", (em,)).fetchone()
            if existing:
                continue
            _customers.find_or_create_by_email(cx, email=em, name=(name or "").strip())
            created += 1
        except Exception:
            continue
    return created
```

- [ ] **Step 4: Run to verify it passes**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_affiliate_people_backfill.py -v`
Expected: PASS (2 tests). Also run the affiliate_dashboard suite for no regression: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_affiliate_dashboard.py tests/test_affiliate_social_links.py -q`.

- [ ] **Step 5: Commit**

```bash
git add dashboard/affiliate_dashboard.py tests/test_affiliate_people_backfill.py
git commit -m "feat(2b3): backfill_affiliate_people (ensure approved affiliates have a people row)"
```

---

### Task 2: Console endpoint to run the backfill

**Files:**
- Modify: `app.py` — add a route near the existing `/api/console/backfill-trial-orders` (~line 24712).

- [ ] **Step 1: Add the route** (mirror the existing console backfill's gate/shape; it uses `_bos_actor()`/console-key — match that exactly):

```python
@app.route("/api/console/backfill-affiliate-people", methods=["POST"])
def api_console_backfill_affiliate_people():
    if _bos_actor() is None:
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    from dashboard import affiliate_dashboard as _ad
    dry = request.args.get("dry_run", "0") == "1"
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        missing = [r[0] for r in cx.execute(
            "SELECT a.email FROM affiliate_signups a WHERE a.status='approved' "
            "AND NOT EXISTS (SELECT 1 FROM people p WHERE lower(p.email)=lower(a.email))").fetchall()]
        if dry:
            return jsonify({"ok": True, "dry_run": True, "would_create": len(missing), "emails": missing})
        created = _ad.backfill_affiliate_people(cx)
    return jsonify({"ok": True, "created": created, "emails": missing})
```
(Confirm the auth helper used by `/api/console/backfill-trial-orders` — if it's not `_bos_actor()`, use whatever that route uses, e.g. an `X-Console-Key`/`require_console_key` pattern. Match it exactly.)

- [ ] **Step 2: Parse-check + commit**

```bash
~/.venvs/deploy-chat311/bin/python -c "import ast; ast.parse(open('app.py').read()); print('OK')"
git add app.py
git commit -m "feat(2b3): POST /api/console/backfill-affiliate-people (dry_run aware)"
```

- [ ] **Step 3: Live verification (post-deploy — record in report)**

`POST /api/console/backfill-affiliate-people?dry_run=1` (console key) → `{would_create:1, emails:["vicsantos336699@gmail.com"]}`; then without dry_run → `{created:1}`; re-run dry → `would_create:0`.

---

### Task 3: On-approval coverage (approve + apply)

**Files:**
- Modify: `app.py` — `patch_affiliate` (the `if status == "approved":` block) and `/affiliate/apply` (after the signup INSERT).

- [ ] **Step 1: Approve path** — in `patch_affiliate`, inside the `if status == "approved":` block (after the `UPDATE affiliate_signups SET status=?`), add (best-effort):

```python
            try:
                from dashboard import customers as _customers
                r = cx.execute("SELECT email, name FROM affiliate_signups WHERE id=?", (aff_id,)).fetchone()
                if r and (r[0] or "").strip():
                    _customers.find_or_create_by_email(cx, email=r[0], name=(r[1] or ""))
            except Exception as _e:
                print(f"[affiliate-approve] people upsert skipped: {_e!r}", flush=True)
```

- [ ] **Step 2: Apply path** — in `/affiliate/apply`, after the affiliate_signups INSERT (where `slug`/`email`/`name` are in scope; the route's signup default status is `approved`), add the same best-effort `find_or_create_by_email(cx, email=email, name=name)` call (use the route's existing connection + variables).

- [ ] **Step 3: Parse-check + commit**

```bash
~/.venvs/deploy-chat311/bin/python -c "import ast; ast.parse(open('app.py').read()); print('OK')"
git add app.py
git commit -m "feat(2b3): ensure a people row on affiliate approve + apply"
```

- [ ] **Step 4: Live verification (post-deploy — record in report)**

(Covered transitively: after deploy, a newly approved/applied affiliate has a people row. Spot-check via `/api/console/backfill-affiliate-people?dry_run=1` staying at 0 after a test apply, if feasible; otherwise rely on Task 2's backfill for existing + this for future.)

---

### Task 4: Retire/redirect the standalone portal + login

**Files:**
- Modify: `app.py` — `affiliate_portal_page` (`/affiliate/portal`), `affiliate_login_request` (`/affiliate/login-request`), `affiliate_login_verify` (`/affiliate/login-verify`), and the three `/affiliate/portal?token=…` redirect targets (~lines 8429, 8485, 8522 — and 8610 `portal_url`).

- [ ] **Step 1: Redirect the dashboard page + login routes**

- `affiliate_portal_page`: replace the `send_from_directory(...)` body with `from flask import redirect as _redir; return _redir("/portal/login")`.
- `affiliate_login_request`: replace its body with `from flask import redirect as _redir; return _redir("/portal/login")` (stop emailing the affiliate magic link; the personal-portal login is the one front door).
- `affiliate_login_verify`: replace its body with `from flask import redirect as _redir; return _redir("/portal/login")` (in-flight affiliate magic links land at the personal login).

- [ ] **Step 2: Repoint the apply/social redirects**

Change each `f"/affiliate/portal?token={...}"` redirect target (the `return _redir(...)`/`resp = _redirect(...)` at ~8429 [now inside the retired login-verify — already handled in Step 1, skip], ~8485, ~8522, and the `portal_url` at ~8610) to `"/portal/login"`. (8429 is inside `affiliate_login_verify`, already replaced in Step 1 — don't double-edit it. Focus on the apply-flow ones at 8485/8522 and `portal_url` 8610.)

- [ ] **Step 3: Parse-check + commit**

```bash
~/.venvs/deploy-chat311/bin/python -c "import ast; ast.parse(open('app.py').read()); print('OK')"
git add app.py
git commit -m "feat(2b3): redirect standalone affiliate portal + login -> /portal/login"
```

- [ ] **Step 4: Live verification (post-deploy — record in report)**

`/affiliate/portal`, `/affiliate/login-request` (POST), `/affiliate/login-verify` → 302 `/portal/login`. `GET /affiliate` and `/affiliate/hub/<a-real-slug>` still 200 (public surfaces intact).

---

## Self-Review

**1. Spec coverage:** backfill helper → Task 1; console run-endpoint → Task 2; on-approval coverage → Task 3; redirects + keep-public → Task 4. ✅
**2. Placeholder scan:** No TBD; Task 1 full code+tests; Tasks 2-4 give exact routes/anchors + "match the existing auth/redirect" with concrete line refs + live checks. ✅
**3. Type consistency:** `backfill_affiliate_people(cx) -> int` (Task 1 def, Task 2 call); `find_or_create_by_email(cx, *, email, name)` reused identically in Tasks 1/3; redirect target `/portal/login` consistent across Task 4. ✅
