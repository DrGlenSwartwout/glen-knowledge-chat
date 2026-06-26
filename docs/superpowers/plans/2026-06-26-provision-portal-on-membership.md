# Provision Portal on Membership Join (Step 3, core) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Guarantee every member (present + future) has a `people` row so their personal portal is reachable via self-login. Mirrors the 2b-3 affiliate coverage.

**Architecture:** `subscriptions.backfill_member_people` (reuses `customers.find_or_create_by_email`) + a console-gated run endpoint + best-effort `people`-ensure hooks in the two membership chokepoints (`create_membership`, `_grant_membership`).

**Tech Stack:** Python 3.11, Flask, sqlite3, pytest.

## Global Constraints

- Reuse `customers.find_or_create_by_email(cx, *, email, name="", phone="")` (creates a `people` row by email if absent). No welcome email (core only).
- "Member" = `subscriptions` with `kind='membership' AND status='active'` UNION `memberships` with `expires_at > now` (ISO-Z text compare).
- `backfill_member_people` idempotent + none-raising per email. Hooks best-effort (try/except → never break create_membership / _grant_membership).
- `dashboard/subscriptions.py` is offline-importable (stdlib only). `app.py` is NOT → Task 2 + the `_grant_membership` half of Task 3 verified live.
- Offline test cmd: `~/.venvs/deploy-chat311/bin/python -m pytest tests/<file> -v`.

---

### Task 1: `backfill_member_people` helper + create_membership hook

**Files:**
- Modify: `dashboard/subscriptions.py`
- Test: `tests/test_member_people_backfill.py`

**Interfaces:**
- Consumes: `customers.find_or_create_by_email`, `_now_iso()` (subscriptions.py:93).
- Produces: `backfill_member_people(cx) -> int`; create_membership now ensures a people row.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_member_people_backfill.py
import sqlite3
from dashboard import subscriptions as subs

def _cx():
    cx = sqlite3.connect(":memory:")
    cx.executescript("""
      CREATE TABLE subscriptions (id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT,
        kind TEXT, status TEXT);
      CREATE TABLE memberships (id TEXT PRIMARY KEY, email TEXT, expires_at TEXT);
      CREATE TABLE people (id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT UNIQUE NOT NULL,
        name TEXT DEFAULT '', phone TEXT DEFAULT '', source TEXT DEFAULT '',
        created_at TEXT, updated_at TEXT);
    """)
    return cx

def test_backfill_covers_active_members_missing_people():
    cx = _cx()
    cx.execute("INSERT INTO subscriptions (email,kind,status) VALUES ('paid@x.com','membership','active')")
    cx.execute("INSERT INTO subscriptions (email,kind,status) VALUES ('cancel@x.com','membership','cancelled')")
    cx.execute("INSERT INTO subscriptions (email,kind,status) VALUES ('remedy@x.com','subscription','active')")
    cx.execute("INSERT INTO memberships (id,email,expires_at) VALUES ('m1','grant@x.com','2999-01-01T00:00:00Z')")
    cx.execute("INSERT INTO memberships (id,email,expires_at) VALUES ('m2','expired@x.com','2000-01-01T00:00:00Z')")
    cx.execute("INSERT INTO people (email) VALUES ('paid@x.com')")  # already has people
    n = subs.backfill_member_people(cx)
    emails = {r[0] for r in cx.execute("SELECT email FROM people").fetchall()}
    assert "grant@x.com" in emails        # unexpired grant -> created
    assert "cancel@x.com" not in emails   # cancelled membership -> skipped
    assert "remedy@x.com" not in emails   # non-membership subscription -> skipped
    assert "expired@x.com" not in emails  # expired grant -> skipped
    assert n == 1                         # only grant@ was a member missing people (paid@ already had one)

def test_idempotent():
    cx = _cx()
    cx.execute("INSERT INTO subscriptions (email,kind,status) VALUES ('a@x.com','membership','active')")
    assert subs.backfill_member_people(cx) == 1
    assert subs.backfill_member_people(cx) == 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_member_people_backfill.py -v`
Expected: FAIL — `backfill_member_people` missing.

- [ ] **Step 3: Implement** — in `dashboard/subscriptions.py`, add `from dashboard import customers as _customers` near the top, and append:

```python
def backfill_member_people(cx):
    """Ensure every current member (active membership subscription OR unexpired access
    grant) has a people row, so their personal portal is reachable via self-login.
    Reuses customers.find_or_create_by_email. Idempotent; returns count created."""
    now = _now_iso()
    rows = cx.execute(
        "SELECT DISTINCT email FROM subscriptions WHERE kind='membership' AND status='active' "
        "UNION SELECT DISTINCT email FROM memberships WHERE expires_at > ?", (now,)).fetchall()
    created = 0
    for (email,) in rows:
        em = (email or "").strip().lower()
        if not em:
            continue
        try:
            if cx.execute("SELECT 1 FROM people WHERE lower(email)=?", (em,)).fetchone():
                continue
            _customers.find_or_create_by_email(cx, email=em)
            created += 1
        except Exception:
            continue
    return created
```

Then add the create_membership hook — in `create_membership`, after `cx.commit()` and before `return cur.lastrowid`, insert (best-effort):

```python
    try:
        _customers.find_or_create_by_email(cx, email=email)
    except Exception:
        pass
```

- [ ] **Step 4: Run to verify it passes**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_member_people_backfill.py -v`
Expected: PASS (2 tests). (If `_now_iso()` returns a non-Z ISO, the test's `2999/2000` sentinels still compare correctly lexically.)

- [ ] **Step 5: Commit**

```bash
git add dashboard/subscriptions.py tests/test_member_people_backfill.py
git commit -m "feat(step3): backfill_member_people + create_membership ensures a people row"
```

---

### Task 2: Console endpoint to run the backfill

**Files:**
- Modify: `app.py` — add near `/api/console/backfill-affiliate-people` (search for it).

- [ ] **Step 1: Add the route** (mirror the affiliate-people one exactly — same `_bos_actor()` gate + dry_run shape):

```python
@app.route("/api/console/backfill-member-people", methods=["POST"])
def api_console_backfill_member_people():
    if _bos_actor() is None:
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    from dashboard import subscriptions as _subs
    dry = request.args.get("dry_run", "0") == "1"
    now = _subs._now_iso()
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        missing = [r[0] for r in cx.execute(
            "SELECT DISTINCT m.email FROM ("
            "  SELECT email FROM subscriptions WHERE kind='membership' AND status='active' "
            "  UNION SELECT email FROM memberships WHERE expires_at > ?) m "
            "WHERE m.email IS NOT NULL AND TRIM(m.email)<>'' "
            "AND NOT EXISTS (SELECT 1 FROM people p WHERE lower(p.email)=lower(m.email))", (now,)).fetchall()]
        if dry:
            return jsonify({"ok": True, "dry_run": True, "would_create": len(missing), "emails": missing})
        created = _subs.backfill_member_people(cx)
    return jsonify({"ok": True, "created": created, "emails": missing})
```

- [ ] **Step 2: Parse-check + commit**

```bash
~/.venvs/deploy-chat311/bin/python -c "import ast; ast.parse(open('app.py').read()); print('OK')"
git add app.py
git commit -m "feat(step3): POST /api/console/backfill-member-people (dry_run aware)"
```

- [ ] **Step 3: Live verification (post-deploy — record in report)**

`POST /api/console/backfill-member-people?dry_run=1` (console key) → reports member emails missing a people row; then real → `created: N`; re-dry → 0.

---

### Task 3: `_grant_membership` hook

**Files:**
- Modify: `app.py` — `_grant_membership` (the access-grant helper; after its `INSERT INTO memberships`, before `return mid`).

- [ ] **Step 1: Add the best-effort hook**

In `_grant_membership(cx, email, days, source)`, after the `cx.execute("INSERT INTO memberships ...")` and before `return mid`, add:

```python
    try:
        from dashboard import customers as _customers
        _customers.find_or_create_by_email(cx, email=email)
    except Exception as _e:
        print(f"[grant-membership] people upsert skipped: {_e!r}", flush=True)
```

- [ ] **Step 2: Parse-check + commit**

```bash
~/.venvs/deploy-chat311/bin/python -c "import ast; ast.parse(open('app.py').read()); print('OK')"
git add app.py
git commit -m "feat(step3): _grant_membership ensures a people row (portal reachable)"
```

- [ ] **Step 3: Live verification (post-deploy — record in report)**

Covered transitively: a new grant/subscription creates a people row → the dry-run backfill stays at 0 after a test grant. (Hooks are tiny best-effort calls; primarily code-reviewed.)

---

## Self-Review

**1. Spec coverage:** backfill helper + create_membership hook → Task 1; console run-endpoint → Task 2; _grant_membership hook → Task 3. Member definition (active membership sub UNION unexpired grant) consistent across the helper + endpoint. ✅
**2. Placeholder scan:** No TBD; Task 1 full code+tests; Tasks 2-3 give exact routes/anchors + concrete code + live checks. ✅
**3. Type consistency:** `backfill_member_people(cx) -> int` (Task 1 def, Task 2 call); `find_or_create_by_email(cx, *, email)` reused identically in all three; the endpoint's `missing` query matches the helper's member definition. ✅
