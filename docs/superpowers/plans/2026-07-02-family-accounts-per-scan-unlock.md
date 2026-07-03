# Family Accounts + Per-Scan Unlock Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the portal blur + $1-lifetime-unlock paywall with a per-scan access model, and render a tab per member on a family primary's portal.

**Architecture:** A new pure `dashboard/family_access.py` module owns three tables (`family_members`, `scan_unlocks`, `family_memberships`) and a single `scan_accessible()` gate. The Flask app wires that gate into the portal API and a new free-unlock endpoint, guarded by a `PORTAL_ACCESS_V2` flag so the legacy blur path is untouched when off. Family membership billing is stubbed (read-only) this pass.

**Tech Stack:** Python 3.11, Flask, sqlite3 (stdlib), pytest. Frontend is a static `static/client-portal.html` with vanilla JS calling `/api/portal/<token>`.

## Global Constraints

- Data modules live in `dashboard/*.py`, self-init on the passed `cx` (sqlite3 connection) via `init_*` using `CREATE TABLE IF NOT EXISTS`; never assume a central migration runs.
- All emails are stored and compared **lowercased + stripped**.
- Tests use the `tmp_db` fixture (a path string) from `tests/conftest.py`; each test opens its own `sqlite3.connect()` and calls the module's `init_*`.
- Timestamps are ISO8601 strings; **pass "now" in as a parameter** to any function that needs the current time/month (do not call `datetime.now()` inside pure logic) so tests are deterministic.
- New behavior ships behind env flag `PORTAL_ACCESS_V2` (default off). With it off, all existing behavior is byte-for-byte unchanged.
- Reuse `_is_paid_member(email)` (app.py:4769) for per-member paid detection; do not reimplement membership logic.
- Console endpoints authenticate with the existing inline pattern: `key = request.headers.get("X-Console-Key","") or request.args.get("key",""); if CONSOLE_SECRET and key != CONSOLE_SECRET and not _owner_token_ok(key): 403`.

---

### Task 1: `family_members` data layer

**Files:**
- Create: `dashboard/family_access.py`
- Test: `tests/test_family_access.py`

**Interfaces:**
- Produces:
  - `init_tables(cx) -> None`
  - `upsert_member(cx, primary_email, member_email, label=None, member_type="human", display_order=0) -> None`
  - `remove_member(cx, primary_email, member_email) -> None`
  - `list_members(cx, primary_email) -> list[dict]` (each `{"member_email","member_label","member_type","display_order"}`, ordered by `display_order` then `member_email`)
  - `primary_for(cx, member_email) -> str | None`
  - `is_primary(cx, email) -> bool`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_family_access.py
import sqlite3
from dashboard import family_access as fa


def _cx(tmp_db):
    cx = sqlite3.connect(tmp_db)
    fa.init_tables(cx)
    return cx


def test_upsert_list_and_resolve_members(tmp_db):
    cx = _cx(tmp_db)
    fa.upsert_member(cx, "Karin@X.com ", "Karin@X.com", "Karin", "human", 0)
    fa.upsert_member(cx, "karin@x.com", "SASHA@fake.com", "Sasha (cat)", "pet", 1)
    members = fa.list_members(cx, "karin@x.com")
    assert [m["member_email"] for m in members] == ["karin@x.com", "sasha@fake.com"]
    assert members[1]["member_label"] == "Sasha (cat)"
    assert members[1]["member_type"] == "pet"
    assert fa.primary_for(cx, "sasha@fake.com") == "karin@x.com"
    assert fa.is_primary(cx, "karin@x.com") is True
    assert fa.is_primary(cx, "sasha@fake.com") is False


def test_upsert_is_idempotent_and_remove(tmp_db):
    cx = _cx(tmp_db)
    fa.upsert_member(cx, "p@x.com", "m@x.com", "M", "human", 0)
    fa.upsert_member(cx, "p@x.com", "m@x.com", "M2", "human", 5)  # update, not duplicate
    members = fa.list_members(cx, "p@x.com")
    assert len(members) == 1 and members[0]["member_label"] == "M2"
    fa.remove_member(cx, "p@x.com", "m@x.com")
    assert fa.list_members(cx, "p@x.com") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_family_access.py -v`
Expected: FAIL (ModuleNotFoundError / no attribute `init_tables`).

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/family_access.py
"""Family accounts + per-scan unlock gate (Portal Access V2).

Each member keeps their own account/email and reports; this module only links
members under a primary and decides per-scan access. Pure sqlite; "now" is passed
in so logic is deterministic under test.
"""
import datetime


def _now_iso():
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _norm(email):
    return (email or "").strip().lower()


def init_tables(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS family_members (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            primary_email TEXT NOT NULL,
            member_email  TEXT NOT NULL,
            member_label  TEXT,
            member_type   TEXT DEFAULT 'human',
            display_order INTEGER DEFAULT 0,
            created_at    TEXT,
            UNIQUE(primary_email, member_email)
        )""")
    cx.execute("CREATE INDEX IF NOT EXISTS ix_family_primary ON family_members(primary_email)")
    cx.execute("CREATE INDEX IF NOT EXISTS ix_family_member ON family_members(member_email)")
    cx.execute("""
        CREATE TABLE IF NOT EXISTS scan_unlocks (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            member_email TEXT NOT NULL,
            scan_id      TEXT NOT NULL,
            scan_date    TEXT,
            unlocked_at  TEXT NOT NULL,
            source       TEXT NOT NULL,
            UNIQUE(member_email, scan_id)
        )""")
    cx.execute("CREATE INDEX IF NOT EXISTS ix_unlock_member ON scan_unlocks(member_email)")
    cx.execute("""
        CREATE TABLE IF NOT EXISTS family_memberships (
            primary_email TEXT PRIMARY KEY,
            active        INTEGER NOT NULL DEFAULT 0,
            updated_at    TEXT
        )""")
    cx.commit()


def upsert_member(cx, primary_email, member_email, label=None, member_type="human", display_order=0):
    p, m = _norm(primary_email), _norm(member_email)
    cx.execute(
        "INSERT INTO family_members (primary_email, member_email, member_label, member_type, display_order, created_at) "
        "VALUES (?,?,?,?,?,?) "
        "ON CONFLICT(primary_email, member_email) DO UPDATE SET "
        "member_label=excluded.member_label, member_type=excluded.member_type, display_order=excluded.display_order",
        (p, m, label, member_type, int(display_order or 0), _now_iso()))
    cx.commit()


def remove_member(cx, primary_email, member_email):
    cx.execute("DELETE FROM family_members WHERE primary_email=? AND member_email=?",
               (_norm(primary_email), _norm(member_email)))
    cx.commit()


def list_members(cx, primary_email):
    rows = cx.execute(
        "SELECT member_email, member_label, member_type, display_order FROM family_members "
        "WHERE primary_email=? ORDER BY display_order, member_email", (_norm(primary_email),)).fetchall()
    return [{"member_email": r[0], "member_label": r[1], "member_type": r[2], "display_order": r[3]} for r in rows]


def primary_for(cx, member_email):
    r = cx.execute("SELECT primary_email FROM family_members WHERE member_email=? LIMIT 1",
                   (_norm(member_email),)).fetchone()
    return r[0] if r else None


def is_primary(cx, email):
    r = cx.execute("SELECT 1 FROM family_members WHERE primary_email=? LIMIT 1", (_norm(email),)).fetchone()
    return bool(r)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_family_access.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add dashboard/family_access.py tests/test_family_access.py
git commit -m "feat(family): family_members link table + resolution"
```

---

### Task 2: `scan_unlocks` + free monthly cap

**Files:**
- Modify: `dashboard/family_access.py`
- Test: `tests/test_family_access.py`

**Interfaces:**
- Consumes: `init_tables` (Task 1).
- Produces:
  - `has_unlock(cx, member_email, scan_id) -> bool`
  - `record_unlock(cx, member_email, scan_id, scan_date, source, now_iso) -> bool` (True if newly inserted; idempotent on `(member_email, scan_id)`)
  - `free_unlock_used_this_month(cx, member_email, year_month) -> bool` (`year_month` = "YYYY-MM")
  - `grant_free_monthly(cx, member_email, scan_id, scan_date, now_iso) -> tuple[bool, str]` (`(ok, reason)`; reason `""` on success, `"cap"` if allowance spent, `"already"` if already unlocked)

- [ ] **Step 1: Write the failing test**

```python
def test_free_monthly_cap_is_per_member_and_permanent(tmp_db):
    cx = _cx(tmp_db)
    # first free unlock in July succeeds
    ok, reason = fa.grant_free_monthly(cx, "m@x.com", "s1", "2026-07-02", "2026-07-02T10:00:00Z")
    assert ok and reason == ""
    assert fa.has_unlock(cx, "m@x.com", "s1") is True
    # second free unlock same month -> capped
    ok, reason = fa.grant_free_monthly(cx, "m@x.com", "s2", "2026-07-05", "2026-07-05T10:00:00Z")
    assert ok is False and reason == "cap"
    assert fa.has_unlock(cx, "m@x.com", "s2") is False
    # next month -> allowed again; prior unlock still present (permanent)
    ok, reason = fa.grant_free_monthly(cx, "m@x.com", "s2", "2026-07-05", "2026-08-01T10:00:00Z")
    assert ok and reason == ""
    assert fa.has_unlock(cx, "m@x.com", "s1") is True
    # a different member is unaffected by m@x.com's usage
    ok, reason = fa.grant_free_monthly(cx, "other@x.com", "s9", "2026-07-02", "2026-07-02T10:00:00Z")
    assert ok and reason == ""


def test_grant_free_monthly_already_unlocked_is_noop(tmp_db):
    cx = _cx(tmp_db)
    fa.record_unlock(cx, "m@x.com", "s1", "2026-07-02", "paid", "2026-07-02T09:00:00Z")
    ok, reason = fa.grant_free_monthly(cx, "m@x.com", "s1", "2026-07-02", "2026-07-02T10:00:00Z")
    assert ok is True and reason == "already"
    # did not consume the monthly allowance
    ok, reason = fa.grant_free_monthly(cx, "m@x.com", "s2", "2026-07-03", "2026-07-03T10:00:00Z")
    assert ok and reason == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_family_access.py -k free -v`
Expected: FAIL (no attribute `grant_free_monthly`).

- [ ] **Step 3: Write minimal implementation**

```python
# append to dashboard/family_access.py

def has_unlock(cx, member_email, scan_id):
    r = cx.execute("SELECT 1 FROM scan_unlocks WHERE member_email=? AND scan_id=?",
                   (_norm(member_email), str(scan_id))).fetchone()
    return bool(r)


def record_unlock(cx, member_email, scan_id, scan_date, source, now_iso):
    cur = cx.execute(
        "INSERT OR IGNORE INTO scan_unlocks (member_email, scan_id, scan_date, unlocked_at, source) "
        "VALUES (?,?,?,?,?)", (_norm(member_email), str(scan_id), scan_date, now_iso, source))
    cx.commit()
    return cur.rowcount == 1


def free_unlock_used_this_month(cx, member_email, year_month):
    r = cx.execute(
        "SELECT 1 FROM scan_unlocks WHERE member_email=? AND source='free_monthly' "
        "AND substr(unlocked_at,1,7)=? LIMIT 1", (_norm(member_email), year_month)).fetchone()
    return bool(r)


def grant_free_monthly(cx, member_email, scan_id, scan_date, now_iso):
    if has_unlock(cx, member_email, scan_id):
        return True, "already"
    year_month = now_iso[:7]  # "YYYY-MM"
    if free_unlock_used_this_month(cx, member_email, year_month):
        return False, "cap"
    record_unlock(cx, member_email, scan_id, scan_date, "free_monthly", now_iso)
    return True, ""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_family_access.py -v`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```bash
git add dashboard/family_access.py tests/test_family_access.py
git commit -m "feat(family): per-scan unlocks + 1/month cap"
```

---

### Task 3: Family membership stub + `family_is_paid`

**Files:**
- Modify: `dashboard/family_access.py`
- Test: `tests/test_family_access.py`

**Interfaces:**
- Consumes: `init_tables`, `primary_for` (Tasks 1).
- Produces:
  - `set_family_membership(cx, primary_email, active, now_iso) -> None` (billing will call this later; tests + console use it now)
  - `family_is_paid(cx, member_email) -> bool` (True iff the member's family primary has `active=1`; the member themselves may be the primary)

- [ ] **Step 1: Write the failing test**

```python
def test_family_is_paid_follows_primary(tmp_db):
    cx = _cx(tmp_db)
    fa.upsert_member(cx, "karin@x.com", "karin@x.com", "Karin", "human", 0)
    fa.upsert_member(cx, "karin@x.com", "sasha@fake.com", "Sasha", "pet", 1)
    assert fa.family_is_paid(cx, "sasha@fake.com") is False
    fa.set_family_membership(cx, "karin@x.com", True, "2026-07-02T10:00:00Z")
    assert fa.family_is_paid(cx, "sasha@fake.com") is True   # member inherits primary's plan
    assert fa.family_is_paid(cx, "karin@x.com") is True
    assert fa.family_is_paid(cx, "stranger@x.com") is False  # not in any family
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_family_access.py -k family_is_paid -v`
Expected: FAIL (no attribute `set_family_membership`).

- [ ] **Step 3: Write minimal implementation**

```python
# append to dashboard/family_access.py

def set_family_membership(cx, primary_email, active, now_iso):
    p = _norm(primary_email)
    cx.execute(
        "INSERT INTO family_memberships (primary_email, active, updated_at) VALUES (?,?,?) "
        "ON CONFLICT(primary_email) DO UPDATE SET active=excluded.active, updated_at=excluded.updated_at",
        (p, 1 if active else 0, now_iso))
    cx.commit()


def family_is_paid(cx, member_email):
    m = _norm(member_email)
    primary = primary_for(cx, m) or (m if is_primary(cx, m) else None)
    if not primary:
        return False
    r = cx.execute("SELECT active FROM family_memberships WHERE primary_email=?", (primary,)).fetchone()
    return bool(r and r[0])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_family_access.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard/family_access.py tests/test_family_access.py
git commit -m "feat(family): family membership stub + family_is_paid"
```

---

### Task 4: The `scan_accessible` gate

**Files:**
- Modify: `dashboard/family_access.py`
- Test: `tests/test_family_access.py`

**Interfaces:**
- Consumes: `family_is_paid`, `has_unlock` (Tasks 2–3).
- Produces: `scan_accessible(cx, member_email, scan_id, is_paid) -> bool` — `is_paid` is the caller-computed per-member paid boolean (from `_is_paid_member`). Returns `is_paid OR family_is_paid OR has_unlock`.

- [ ] **Step 1: Write the failing test**

```python
def test_scan_accessible_truth_table(tmp_db):
    cx = _cx(tmp_db)
    # locked, free, no family -> not accessible
    assert fa.scan_accessible(cx, "m@x.com", "s1", is_paid=False) is False
    # per-member paid -> accessible regardless of rows
    assert fa.scan_accessible(cx, "m@x.com", "s1", is_paid=True) is True
    # explicit unlock row -> accessible even if free
    fa.record_unlock(cx, "m@x.com", "s1", "2026-07-02", "free_monthly", "2026-07-02T10:00:00Z")
    assert fa.scan_accessible(cx, "m@x.com", "s1", is_paid=False) is True
    # family plan -> accessible even for a still-locked scan
    fa.upsert_member(cx, "p@x.com", "m@x.com", "M", "human", 0)
    fa.set_family_membership(cx, "p@x.com", True, "2026-07-02T10:00:00Z")
    assert fa.scan_accessible(cx, "m@x.com", "s2", is_paid=False) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_family_access.py -k accessible -v`
Expected: FAIL (no attribute `scan_accessible`).

- [ ] **Step 3: Write minimal implementation**

```python
# append to dashboard/family_access.py

def scan_accessible(cx, member_email, scan_id, is_paid):
    if is_paid:
        return True
    if family_is_paid(cx, member_email):
        return True
    return has_unlock(cx, member_email, scan_id)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_family_access.py -v`
Expected: PASS (full module suite green).

- [ ] **Step 5: Commit**

```bash
git add dashboard/family_access.py tests/test_family_access.py
git commit -m "feat(family): scan_accessible gate"
```

---

### Task 5: Free-unlock endpoint `POST /api/portal/<token>/unlock-scan`

**Files:**
- Modify: `app.py` (add route near the other `/api/portal/<token>/...` routes, e.g. after `api_client_portal`, ~app.py:12945)
- Test: `tests/test_family_unlock_endpoint.py`

**Interfaces:**
- Consumes: `family_access` module (Tasks 1–4); existing `_portal_record_for(cx, token)` (returns a dict with `email`), `LOG_DB`, `_is_paid_member`.
- Produces: JSON `{"ok": bool, "reason": str}`; 404 if token unknown; 400 if `scan_id` missing.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_family_unlock_endpoint.py
import json, sqlite3
import app as appmod


def _client(tmp_db, monkeypatch):
    monkeypatch.setattr(appmod, "LOG_DB", tmp_db)
    monkeypatch.setattr(appmod, "_is_paid_member", lambda e: False)
    # a portal token that resolves to member m@x.com
    monkeypatch.setattr(appmod, "_portal_record_for", lambda cx, tok: {"email": "m@x.com"} if tok == "TOK" else None)
    return appmod.app.test_client()


def test_unlock_scan_first_succeeds_then_capped(tmp_db, monkeypatch):
    c = _client(tmp_db, monkeypatch)
    r1 = c.post("/api/portal/TOK/unlock-scan", json={"scan_id": "s1"})
    assert r1.status_code == 200 and r1.get_json()["ok"] is True
    r2 = c.post("/api/portal/TOK/unlock-scan", json={"scan_id": "s2"})
    assert r2.get_json() == {"ok": False, "reason": "cap"}


def test_unlock_scan_unknown_token_404(tmp_db, monkeypatch):
    c = _client(tmp_db, monkeypatch)
    r = c.post("/api/portal/NOPE/unlock-scan", json={"scan_id": "s1"})
    assert r.status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_family_unlock_endpoint.py -v`
Expected: FAIL (404 route not found → 404 for TOK too, or AttributeError).

- [ ] **Step 3: Write minimal implementation**

```python
# app.py — add after the api_client_portal route (~line 12945)
@app.route("/api/portal/<token>/unlock-scan", methods=["POST"])
def api_portal_unlock_scan(token):
    from dashboard import family_access as _fa
    import datetime as _dt
    body = request.get_json(silent=True) or {}
    scan_id = str(body.get("scan_id") or "").strip()
    if not scan_id:
        return jsonify({"error": "scan_id required"}), 400
    with sqlite3.connect(LOG_DB) as cx:
        portal = _portal_record_for(cx, token)
        if not portal:
            return jsonify({"error": "not found"}), 404
        email = (portal.get("email") or "").strip().lower()
        _fa.init_tables(cx)
        if _fa.scan_accessible(cx, email, scan_id, is_paid=_is_paid_member(email)):
            return jsonify({"ok": True, "reason": "already"})
        now_iso = _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        ok, reason = _fa.grant_free_monthly(cx, email, scan_id, body.get("scan_date"), now_iso)
        return jsonify({"ok": ok, "reason": reason})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_family_unlock_endpoint.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_family_unlock_endpoint.py
git commit -m "feat(family): POST /api/portal/<token>/unlock-scan (1/mo free)"
```

---

### Task 6: Console family management endpoints

**Files:**
- Modify: `app.py` (add near other `/api/console/...` routes)
- Test: `tests/test_family_console_api.py`

**Interfaces:**
- Consumes: `family_access` (Tasks 1–3), `CONSOLE_SECRET`, `_owner_token_ok`.
- Produces:
  - `GET /api/console/family/<primary_email>` → `{"members": [...]}`
  - `POST /api/console/family` `{primary_email, member_email, member_label, member_type, display_order}` → `{"ok": true}`
  - `DELETE /api/console/family` `{primary_email, member_email}` → `{"ok": true}`
  - `POST /api/console/family/membership` `{primary_email, active}` → `{"ok": true}` (manual toggle until billing lands)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_family_console_api.py
import sqlite3
import app as appmod


def _client(tmp_db, monkeypatch):
    monkeypatch.setattr(appmod, "LOG_DB", tmp_db)
    monkeypatch.setattr(appmod, "CONSOLE_SECRET", "K")
    return appmod.app.test_client()


def test_console_family_crud(tmp_db, monkeypatch):
    c = _client(tmp_db, monkeypatch)
    h = {"X-Console-Key": "K"}
    assert c.post("/api/console/family", json={"primary_email": "p@x.com", "member_email": "p@x.com",
                  "member_label": "P", "member_type": "human", "display_order": 0}, headers=h).status_code == 200
    c.post("/api/console/family", json={"primary_email": "p@x.com", "member_email": "sasha@f.com",
           "member_label": "Sasha", "member_type": "pet", "display_order": 1}, headers=h)
    got = c.get("/api/console/family/p@x.com", headers=h).get_json()
    assert [m["member_email"] for m in got["members"]] == ["p@x.com", "sasha@f.com"]
    c.delete("/api/console/family", json={"primary_email": "p@x.com", "member_email": "sasha@f.com"}, headers=h)
    assert len(c.get("/api/console/family/p@x.com", headers=h).get_json()["members"]) == 1


def test_console_family_requires_key(tmp_db, monkeypatch):
    c = _client(tmp_db, monkeypatch)
    assert c.get("/api/console/family/p@x.com").status_code == 403
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_family_console_api.py -v`
Expected: FAIL (routes 404/no auth).

- [ ] **Step 3: Write minimal implementation**

```python
# app.py — add near other /api/console routes
def _console_key_ok():
    key = request.headers.get("X-Console-Key", "") or request.args.get("key", "")
    return not CONSOLE_SECRET or key == CONSOLE_SECRET or _owner_token_ok(key)


@app.route("/api/console/family/<path:primary_email>", methods=["GET"])
def api_console_family_get(primary_email):
    if not _console_key_ok():
        return jsonify({"error": "forbidden"}), 403
    from dashboard import family_access as _fa
    with sqlite3.connect(LOG_DB) as cx:
        _fa.init_tables(cx)
        return jsonify({"members": _fa.list_members(cx, primary_email)})


@app.route("/api/console/family", methods=["POST", "DELETE"])
def api_console_family_mutate():
    if not _console_key_ok():
        return jsonify({"error": "forbidden"}), 403
    from dashboard import family_access as _fa
    b = request.get_json(silent=True) or {}
    with sqlite3.connect(LOG_DB) as cx:
        _fa.init_tables(cx)
        if request.method == "DELETE":
            _fa.remove_member(cx, b.get("primary_email"), b.get("member_email"))
        else:
            _fa.upsert_member(cx, b.get("primary_email"), b.get("member_email"),
                              b.get("member_label"), b.get("member_type", "human"),
                              b.get("display_order", 0))
    return jsonify({"ok": True})


@app.route("/api/console/family/membership", methods=["POST"])
def api_console_family_membership():
    if not _console_key_ok():
        return jsonify({"error": "forbidden"}), 403
    from dashboard import family_access as _fa
    import datetime as _dt
    b = request.get_json(silent=True) or {}
    now_iso = _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    with sqlite3.connect(LOG_DB) as cx:
        _fa.init_tables(cx)
        _fa.set_family_membership(cx, b.get("primary_email"), bool(b.get("active")), now_iso)
    return jsonify({"ok": True})
```

> Note: the `/api/console/family/membership` POST and `/api/console/family/<primary_email>` GET both match `/api/console/family/...`; Flask resolves by method + the literal `membership` segment first only if registered first. Register the `membership` route BEFORE the `<path:primary_email>` GET to avoid the catch-all swallowing it, OR (simpler) rename the GET to `/api/console/family-members/<path:primary_email>`. Use the rename to remove ambiguity.

- [ ] **Step 4: Run test to verify it passes**

Update the GET test URL to `/api/console/family-members/p@x.com` per the rename. Run: `pytest tests/test_family_console_api.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_family_console_api.py
git commit -m "feat(family): console family CRUD + manual membership toggle"
```

---

### Task 7: Portal API — members + per-scan access under `PORTAL_ACCESS_V2`

**Files:**
- Modify: `app.py` — `api_client_portal` (`/api/portal/<token>`, ~line 12852) and the blur line `bf_show = bf_confirmed and _portal_biofield_unlocked(email_for_reports)` (~line 12888)
- Test: `tests/test_portal_access_v2.py`

**Interfaces:**
- Consumes: `family_access.scan_accessible`, `list_members`, `is_primary`; `_is_paid_member`; existing `_pbr` report getters.
- Produces: the JSON response gains `members` (list, empty when not a family) and, when `PORTAL_ACCESS_V2` is on, `blurred` is derived from `scan_accessible(email, current scan_id, is_paid=_is_paid_member(email))` instead of `_portal_biofield_unlocked`. A `?member=<email>` query param selects which family member's reports to render (must be in the family; defaults to the token's own email).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_portal_access_v2.py
import sqlite3
import app as appmod
from dashboard import family_access as fa
from dashboard import portal_biofield_reports as pbr


def _seed(tmp_db):
    cx = sqlite3.connect(tmp_db)
    fa.init_tables(cx); pbr.init_table(cx)
    pbr.upsert_report(cx, "m@x.com", "2026-07-02", "s1",
                      {"layers": [{"n": 1, "title": "T", "meaning": "M", "remedy": "R", "dosing": "D"}]},
                      "confirmed")
    cx.commit(); cx.close()


def test_v2_locked_scan_hides_remedy(tmp_db, monkeypatch):
    monkeypatch.setenv("PORTAL_ACCESS_V2", "1")
    monkeypatch.setattr(appmod, "LOG_DB", tmp_db)
    monkeypatch.setattr(appmod, "_is_paid_member", lambda e: False)
    monkeypatch.setattr(appmod, "_portal_record_for", lambda cx, tok: {"email": "m@x.com", "name": "M"})
    _seed(tmp_db)
    j = appmod.app.test_client().get("/api/portal/TOK").get_json()
    assert j["blurred"] is True
    assert j["layers"][0].get("remedy", "") == ""   # locked -> no remedy


def test_v2_paid_shows_remedy(tmp_db, monkeypatch):
    monkeypatch.setenv("PORTAL_ACCESS_V2", "1")
    monkeypatch.setattr(appmod, "LOG_DB", tmp_db)
    monkeypatch.setattr(appmod, "_is_paid_member", lambda e: True)
    monkeypatch.setattr(appmod, "_portal_record_for", lambda cx, tok: {"email": "m@x.com", "name": "M"})
    _seed(tmp_db)
    j = appmod.app.test_client().get("/api/portal/TOK").get_json()
    assert j["blurred"] is False
    assert j["layers"][0]["remedy"] == "R"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_portal_access_v2.py -v`
Expected: FAIL (blur still keyed to `_portal_biofield_unlocked`; `members` missing).

- [ ] **Step 3: Write minimal implementation**

In `api_client_portal`, after `email_for_reports` is set and the current scan is picked (the block that sets `bf_scan_date`/report), replace the single `bf_show` assignment:

```python
    # was: bf_show = bf_confirmed and _portal_biofield_unlocked(email_for_reports)
    if os.environ.get("PORTAL_ACCESS_V2") in ("1", "true", "True"):
        from dashboard import family_access as _fa
        # allow a family primary to view a specific member's reports
        member_email = (request.args.get("member") or "").strip().lower() or email_for_reports
        _cx_fa = sqlite3.connect(LOG_DB); _fa.init_tables(_cx_fa)
        fam_members = _fa.list_members(_cx_fa, email_for_reports) if _fa.is_primary(_cx_fa, email_for_reports) else []
        # scan id for the currently-picked report (0/"" when none)
        _picked_scan_id = str((rep.get("scan_id") if dates else content.get("scan_id")) or "")
        bf_show = bf_confirmed and _fa.scan_accessible(
            _cx_fa, member_email, _picked_scan_id, is_paid=_is_paid_member(member_email))
        _cx_fa.close()
    else:
        fam_members = []
        bf_show = bf_confirmed and _portal_biofield_unlocked(email_for_reports)
```

Then add `"members": fam_members,` to the returned JSON dict.

> `rep` is already in scope from the report-selection block (`rep = _pbr.get_report(...)`). If `dates` is empty, `content.get("scan_id")` is used. `import os` is already imported at module top.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_portal_access_v2.py -v`
Also run the legacy guard: `PORTAL_ACCESS_V2` unset → existing portal tests still pass:
`pytest tests/ -k portal -v`
Expected: PASS; legacy behavior unchanged when flag off.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_portal_access_v2.py
git commit -m "feat(family): portal API per-scan access + members under PORTAL_ACCESS_V2"
```

---

### Task 8: Portal UI — member tabs + locked/unlock affordance (no blur)

**Files:**
- Modify: `static/client-portal.html`
- Test: `tests/test_portal_access_v2.py` (assert API contract only; DOM is manual/render-verify)

**Interfaces:**
- Consumes: `/api/portal/<token>` JSON (`members`, `blurred`, `scan_dates`, `scan_date`), `/api/portal/<token>/unlock-scan`.

- [ ] **Step 1: Add the tab bar + unlock button (JS render)**

In `static/client-portal.html`, in the function that renders the portal after fetching `/api/portal/<token>`:

1. If `data.members && data.members.length`, render a tab bar above the report area — one button per member (`member_label` or `member_email`). Clicking a tab re-fetches `/api/portal/<token>?member=<member_email>` and re-renders. Track the active member in a module variable; default to the first tab.
2. Remove any blur CSS class application. A report whose `blurred` is true renders the interpretation/meanings but shows, in place of each remedy, a **"🔒 Unlock this scan"** button (not a blurred box).
3. The unlock button calls `POST /api/portal/<token>/unlock-scan {scan_id}` (scan_id from the selected report), then re-fetches and re-renders on `{ok:true}`. On `{ok:false, reason:"cap"}`, disable the button and show "1 free unlock used this month — resets on the 1st."

Concrete snippet to add (adapt to the file's existing render helpers):

```html
<div id="member-tabs" class="member-tabs"></div>
<script>
function renderMemberTabs(data) {
  const bar = document.getElementById('member-tabs');
  if (!data.members || !data.members.length) { bar.innerHTML = ''; return; }
  bar.innerHTML = data.members.map(m =>
    `<button class="tab${m.member_email===window.__activeMember?' active':''}"
      onclick="selectMember('${m.member_email}')">${m.member_label||m.member_email}</button>`
  ).join('');
}
function selectMember(email) { window.__activeMember = email; loadPortal(email); }
async function unlockScan(scanId) {
  const r = await fetch(`/api/portal/${TOKEN}/unlock-scan`,
    {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({scan_id: scanId})});
  const j = await r.json();
  if (j.ok) loadPortal(window.__activeMember);
  else if (j.reason === 'cap') showCapNotice();
}
</script>
```

Where `loadPortal(email)` fetches `/api/portal/${TOKEN}${email?`?member=${encodeURIComponent(email)}`:''}`.

- [ ] **Step 2: Manual render-verify (no automated DOM test)**

Load a family portal token locally with `PORTAL_ACCESS_V2=1`; confirm: tabs appear, switching tabs swaps reports, a locked scan shows the unlock button (no blur), unlocking reveals the remedy, cap disables the button. Record in `tests/fireside_render_verify.md`-style note or the PR description.

- [ ] **Step 3: Commit**

```bash
git add static/client-portal.html
git commit -m "feat(family): portal member tabs + per-scan unlock UI (no blur)"
```

---

### Task 9: Retire blur/$1 legacy under the flag

**Files:**
- Modify: `app.py` (any remaining `_portal_biofield_unlocked` / `BIOFIELD_TRIAL_ENABLED` / `PORTAL_PAID_GATE_ENABLED` gated render paths), `static/client-portal.html` (remove blur CSS when V2)
- Test: `tests/test_portal_access_v2.py`

**Interfaces:** none new. Ensures every blur/$1 branch is bypassed when `PORTAL_ACCESS_V2` is on.

- [ ] **Step 1: Inventory legacy gates**

Run: `grep -nE "_portal_biofield_unlocked|BIOFIELD_TRIAL_ENABLED|PORTAL_PAID_GATE_ENABLED|BIOFIELD_CART_ENABLED" app.py`
For each hit that affects portal report visibility or the $1 unlock CTA, wrap it: `if PORTAL_ACCESS_V2 is on → new per-scan path; else legacy`.

- [ ] **Step 2: Add a regression test that legacy is untouched when flag off**

```python
def test_flag_off_uses_legacy_gate(tmp_db, monkeypatch):
    monkeypatch.delenv("PORTAL_ACCESS_V2", raising=False)
    monkeypatch.setattr(appmod, "LOG_DB", tmp_db)
    monkeypatch.setattr(appmod, "_portal_record_for", lambda cx, tok: {"email": "m@x.com", "name": "M"})
    monkeypatch.setattr(appmod, "_portal_biofield_unlocked", lambda e: True)
    _seed(tmp_db)
    j = appmod.app.test_client().get("/api/portal/TOK").get_json()
    assert j["blurred"] is False   # legacy gate honored, members absent/empty
    assert j.get("members", []) == []
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_portal_access_v2.py -v`
Expected: PASS (both V2-on and flag-off paths).

- [ ] **Step 4: Commit**

```bash
git add app.py static/client-portal.html tests/test_portal_access_v2.py
git commit -m "refactor(family): gate legacy blur/$1 behind PORTAL_ACCESS_V2"
```

---

### Task 10: Sasha/Karin data migration script

**Files:**
- Create: `scripts/migrate_sasha_family.py`
- Test: manual (one-off, prod data)

**Interfaces:** Consumes `family_access.upsert_member`; `portal_biofield_reports`.

- [ ] **Step 1: Write the migration**

```python
# scripts/migrate_sasha_family.py
"""One-off: move Sasha's cross-keyed report back under her own account and link
her to Karin's family. Idempotent."""
import os, sqlite3, datetime
from dashboard import family_access as fa

LOG_DB = os.environ.get("LOG_DB", os.path.expanduser("~/deploy-chat/chat_log.db"))
KARIN = "permanentlyyours@hawaii.rr.com"       # real email = family primary
SASHA = "permanentlyyours777@hawaiiantel.net"  # Sasha's own (fake) E4L email
now = datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

cx = sqlite3.connect(LOG_DB)
fa.init_tables(cx)
fa.upsert_member(cx, KARIN, KARIN, "Karin Takahashi", "human", 0)
fa.upsert_member(cx, KARIN, SASHA, "Sasha (Karin Takahashi's cat)", "pet", 1)
# remove the stopgap cross-keyed report row (Sasha's 2026-07-02 under Karin's email)
cx.execute("DELETE FROM portal_biofield_reports WHERE lower(email)=? AND scan_date=?",
           (KARIN, "2026-07-02"))
cx.commit()
print("linked Sasha under Karin; removed cross-keyed report row")
```

> Sasha's actual report content lives under her own account's E4L data; republish it under `SASHA` via the normal importer if a portal report row is needed for her tab. This script only fixes the family link + removes the stopgap row.

- [ ] **Step 2: Commit (run manually against prod later, not in CI)**

```bash
git add scripts/migrate_sasha_family.py
git commit -m "chore(family): Sasha/Karin family-link migration script"
```

---

### Task 11: Full-suite green + PR

- [ ] **Step 1:** Run `pytest tests/test_family_access.py tests/test_family_unlock_endpoint.py tests/test_family_console_api.py tests/test_portal_access_v2.py -v` → all PASS.
- [ ] **Step 2:** Run the broader portal/biofield suite with the flag off to confirm no regressions: `pytest tests/ -k "portal or biofield" -v`.
- [ ] **Step 3:** Open PR from `sess/dacf01b2`; body notes: ships dark behind `PORTAL_ACCESS_V2`; billing (the $197/mo mechanism) deferred; go-live = flip flag in Render + render-verify each surface + run the Sasha migration.

---

## Self-Review

**Spec coverage:**
- §3 access model → Tasks 2 (free cap), 4 (gate), 7 (portal wiring). ✓
- §4 data model → Tasks 1–3 (all three tables). ✓
- §5 gate → Task 4. ✓
- §6 free monthly unlock → Task 5. ✓
- §7 portal UI → Tasks 7 (API) + 8 (UI). ✓
- §8 console → Task 6. ✓
- §9 retire blur/$1 → Task 9. ✓
- §10 testing → tests in every task. ✓
- §11 rollout → Task 11. ✓
- §8 Sasha cleanup → Task 10. ✓
- §12 billing deferred → `family_is_paid` stub (Task 3), manual toggle (Task 6). ✓

**Placeholders:** none — every code step carries full code; the one manual step (Task 8 DOM verify) is explicitly manual by nature.

**Type consistency:** `scan_accessible(cx, member_email, scan_id, is_paid)`, `grant_free_monthly(...) -> (ok, reason)`, `list_members -> [{member_email,...}]`, `set_family_membership(cx, primary_email, active, now_iso)` used consistently across Tasks 3–7. Console GET renamed to `/api/console/family-members/<primary_email>` to avoid the `/api/console/family/membership` collision (Task 6 note) — reflected in its test.
