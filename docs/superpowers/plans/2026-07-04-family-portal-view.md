# Family / Household Portal-View Accounts (v1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let one caregiver/primary account view any linked member's scans from its own portal (a member switcher), and let the owner reassign a mis-attributed scan among a household — reusing the entire live portal + paywall, adding zero new billing.

**Architecture:** A new `dashboard/household.py` owns two SQLite tables (caregiver→member links + a reassignment audit log) plus pure guard functions. `api_client_portal` gains a household payload and a guarded `?member=` override placed exactly where `email_for_reports` is first computed — so overriding it re-points the whole existing portal (reports/blur/brand) at the member with no other backend change. Viewing is behind `HOUSEHOLD_VIEW_ENABLED`; owner console CRUD + reassignment endpoints/page are always available (a data-correction tool).

**Tech Stack:** Python 3, Flask, SQLite (`LOG_DB`), pytest, vanilla JS.

## Global Constraints

- **Portal-view first, NOT billing.** The link grants viewing only; each member's own paywall (`_portal_biofield_unlocked(member_email)`, app.py:10255) governs remedy blur unchanged. No new billing/entitlement.
- **Fail-closed, no IDOR.** `?member=` is honored only if `household.can_view(cx, viewer, target)` is True; otherwise serve the primary's own view. Never error, never reveal whether an email exists.
- **Viewing behind `HOUSEHOLD_VIEW_ENABLED` (default OFF).** Flag-off = byte-identical portal (no `household` key, `?member=` ignored). Console CRUD/reassign tools are NOT behind this flag.
- **Reassignment is within-household only, never clobbers.** Refuse cross-household moves and refuse when the target already has a report for that `scan_date` (`portal_biofield_reports` has `UNIQUE(email, scan_date)`).
- All emails stored and compared **lowercased/stripped**.
- Console endpoints gated by `_portal_console_ok()` (app.py:13720; X-Console-Key / owner token).
- No change to `portal_biofield_reports` schema, the reveal/publish pipeline, or E4L.
- Tests import `app` → run via `doppler run -p remedy-match -c dev -- python3 -m pytest ...`.

---

## File Structure

- **Create** `dashboard/household.py` — link tables + reassignment audit table; `init_household_tables`, `add_member`, `remove_member`, `members_for`, `can_view`, `same_household`, `reassign_report`, `list_reassignments`.
- **Modify** `app.py` — `api_client_portal`: household payload + guarded `?member=` (behind flag); a `_household_view_enabled()` helper; console endpoints `GET/POST/DELETE /api/console/household`, `POST /api/console/household/reassign`, and the `/console/household` page route.
- **Modify** `static/client-portal.html` — member selector rendered from `d.household`.
- **Create** `static/console-household.html` — owner CRUD + reassignment UI.
- **Test** `tests/test_household.py`, `tests/test_household_portal_route.py`, `tests/test_household_console_api.py`.

---

### Task 1: `dashboard/household.py` — link tables + CRUD + view guards

**Files:** Create `dashboard/household.py`; Test `tests/test_household.py`

**Interfaces:**
- Produces:
  - `init_household_tables(cx)` — creates `household_members` + `scan_reassignments`.
  - `add_member(cx, primary_email, member_email, label="", relationship="") -> bool`
  - `remove_member(cx, primary_email, member_email) -> None`
  - `members_for(cx, primary_email) -> list[{"email","label","relationship"}]`
  - `can_view(cx, viewer_email, target_email) -> bool`
  - `same_household(cx, a, b) -> bool`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_household.py
import sqlite3
from dashboard import household as h


def _cx():
    cx = sqlite3.connect(":memory:")
    h.init_household_tables(cx)
    return cx


def test_add_members_and_list():
    cx = _cx()
    assert h.add_member(cx, "Karin@x.com", "mochi@x.com", "Mochi", "pet") is True
    h.add_member(cx, "karin@x.com", "kai@x.com", "Kai", "child")
    ms = h.members_for(cx, "karin@x.com")
    assert [m["email"] for m in ms] == ["mochi@x.com", "kai@x.com"]
    assert ms[0]["label"] == "Mochi" and ms[0]["relationship"] == "pet"


def test_add_member_rejects_self_and_blank():
    cx = _cx()
    assert h.add_member(cx, "a@x.com", "a@x.com") is False
    assert h.add_member(cx, "", "b@x.com") is False
    assert h.members_for(cx, "a@x.com") == []


def test_add_member_idempotent():
    cx = _cx()
    h.add_member(cx, "p@x.com", "m@x.com", "M", "pet")
    h.add_member(cx, "p@x.com", "m@x.com", "M2", "child")  # dup ignored
    assert len(h.members_for(cx, "p@x.com")) == 1


def test_remove_member():
    cx = _cx()
    h.add_member(cx, "p@x.com", "m@x.com")
    h.remove_member(cx, "P@x.com", "M@x.com")  # case-insensitive
    assert h.members_for(cx, "p@x.com") == []


def test_can_view():
    cx = _cx()
    h.add_member(cx, "p@x.com", "m@x.com")
    assert h.can_view(cx, "p@x.com", "p@x.com") is True     # self
    assert h.can_view(cx, "P@x.com", "M@x.com") is True     # linked (case-insensitive)
    assert h.can_view(cx, "p@x.com", "stranger@x.com") is False
    assert h.can_view(cx, "m@x.com", "p@x.com") is False    # reverse is NOT a view grant
    assert h.can_view(cx, "", "m@x.com") is False


def test_same_household():
    cx = _cx()
    h.add_member(cx, "p@x.com", "m1@x.com")
    h.add_member(cx, "p@x.com", "m2@x.com")
    assert h.same_household(cx, "p@x.com", "m1@x.com") is True   # primary↔member
    assert h.same_household(cx, "m1@x.com", "p@x.com") is True   # order-independent
    assert h.same_household(cx, "m1@x.com", "m2@x.com") is True  # siblings share primary
    assert h.same_household(cx, "m1@x.com", "stranger@x.com") is False
```

- [ ] **Step 2: Run to verify it fails**

Run: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_household.py -q`
Expected: FAIL (`ModuleNotFoundError: dashboard.household`).

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/household.py
"""Household / caregiver→member links + within-household scan reassignment.

A caregiver (primary_email) links to member scan accounts (member_email). The
primary may VIEW any linked member's portal; the owner may REASSIGN a mis-attributed
portal report among the members of one household. Lives in LOG_DB (SQLite). Every
scan-subject already has its own E4L email, so members are ordinary email accounts."""
import datetime


def _now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _norm(e):
    return (e or "").strip().lower()


def init_household_tables(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS household_members (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            primary_email TEXT NOT NULL,
            member_email  TEXT NOT NULL,
            label         TEXT,
            relationship  TEXT,
            created_at    TEXT,
            UNIQUE(primary_email, member_email)
        )
    """)
    cx.execute("CREATE INDEX IF NOT EXISTS ix_hm_primary ON household_members(primary_email)")
    cx.execute("CREATE INDEX IF NOT EXISTS ix_hm_member ON household_members(member_email)")
    cx.execute("""
        CREATE TABLE IF NOT EXISTS scan_reassignments (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_date  TEXT,
            from_email TEXT,
            to_email   TEXT,
            by         TEXT,
            at         TEXT
        )
    """)
    cx.commit()


def add_member(cx, primary_email, member_email, label="", relationship=""):
    p, m = _norm(primary_email), _norm(member_email)
    if not p or not m or p == m:
        return False
    cx.execute(
        "INSERT OR IGNORE INTO household_members "
        "(primary_email, member_email, label, relationship, created_at) VALUES (?,?,?,?,?)",
        (p, m, label or "", relationship or "", _now()))
    cx.commit()
    return True


def remove_member(cx, primary_email, member_email):
    cx.execute("DELETE FROM household_members WHERE primary_email=? AND member_email=?",
               (_norm(primary_email), _norm(member_email)))
    cx.commit()


def members_for(cx, primary_email):
    rows = cx.execute(
        "SELECT member_email, label, relationship FROM household_members "
        "WHERE primary_email=? ORDER BY created_at, id", (_norm(primary_email),)).fetchall()
    return [{"email": r[0], "label": r[1] or "", "relationship": r[2] or ""} for r in rows]


def can_view(cx, viewer_email, target_email):
    v, t = _norm(viewer_email), _norm(target_email)
    if not v or not t:
        return False
    if v == t:
        return True
    return cx.execute(
        "SELECT 1 FROM household_members WHERE primary_email=? AND member_email=? LIMIT 1",
        (v, t)).fetchone() is not None


def same_household(cx, a, b):
    a, b = _norm(a), _norm(b)
    if not a or not b:
        return False
    if a == b:
        return True
    if cx.execute(
        "SELECT 1 FROM household_members WHERE (primary_email=? AND member_email=?) "
        "OR (primary_email=? AND member_email=?) LIMIT 1", (a, b, b, a)).fetchone():
        return True
    return cx.execute(
        "SELECT 1 FROM household_members h1 JOIN household_members h2 "
        "ON h1.primary_email=h2.primary_email "
        "WHERE h1.member_email=? AND h2.member_email=? LIMIT 1", (a, b)).fetchone() is not None
```

- [ ] **Step 4: Run to verify it passes** — GREEN.

- [ ] **Step 5: Commit**

```bash
git add dashboard/household.py tests/test_household.py
git commit -m "feat(household): caregiver→member link tables + view guards"
```

---

### Task 2: `household.reassign_report` + `list_reassignments`

**Files:** Modify `dashboard/household.py`; Test `tests/test_household.py`

**Interfaces:**
- Consumes: `same_household` (Task 1); the `portal_biofield_reports` table (`dashboard/portal_biofield_reports.init_table(cx)`, columns `email, scan_date, ...`, `UNIQUE(email, scan_date)`).
- Produces:
  - `reassign_report(cx, scan_date, from_email, to_email, *, by="console") -> {"ok": bool, "error": str|None}`
  - `list_reassignments(cx, limit=100) -> list[{"scan_date","from_email","to_email","by","at"}]`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_household.py
from dashboard import portal_biofield_reports as pbr


def test_reassign_moves_report_and_logs():
    cx = sqlite3.connect(":memory:")
    h.init_household_tables(cx)
    pbr.init_table(cx)
    h.add_member(cx, "p@x.com", "wrong@x.com")
    h.add_member(cx, "p@x.com", "right@x.com")
    pbr.upsert_report(cx, "wrong@x.com", "2026-06-25", "s1", {"n": 1}, "confirmed"); cx.commit()
    r = h.reassign_report(cx, "2026-06-25", "wrong@x.com", "right@x.com")
    assert r["ok"] is True
    assert pbr.list_report_dates(cx, "right@x.com") == ["2026-06-25"]
    assert pbr.list_report_dates(cx, "wrong@x.com") == []
    logs = h.list_reassignments(cx)
    assert logs[0]["from_email"] == "wrong@x.com" and logs[0]["to_email"] == "right@x.com"


def test_reassign_rejects_cross_household():
    cx = sqlite3.connect(":memory:")
    h.init_household_tables(cx); pbr.init_table(cx)
    h.add_member(cx, "p@x.com", "a@x.com")
    pbr.upsert_report(cx, "a@x.com", "2026-06-25", "s1", {}, "confirmed"); cx.commit()
    r = h.reassign_report(cx, "2026-06-25", "a@x.com", "stranger@x.com")
    assert r["ok"] is False and "household" in r["error"]
    assert pbr.list_report_dates(cx, "a@x.com") == ["2026-06-25"]  # unchanged


def test_reassign_refuses_collision():
    cx = sqlite3.connect(":memory:")
    h.init_household_tables(cx); pbr.init_table(cx)
    h.add_member(cx, "p@x.com", "a@x.com"); h.add_member(cx, "p@x.com", "b@x.com")
    pbr.upsert_report(cx, "a@x.com", "2026-06-25", "s1", {}, "confirmed")
    pbr.upsert_report(cx, "b@x.com", "2026-06-25", "s2", {}, "confirmed"); cx.commit()
    r = h.reassign_report(cx, "2026-06-25", "a@x.com", "b@x.com")
    assert r["ok"] is False and "already has" in r["error"]
    assert pbr.list_report_dates(cx, "a@x.com") == ["2026-06-25"]  # unchanged


def test_reassign_missing_report():
    cx = sqlite3.connect(":memory:")
    h.init_household_tables(cx); pbr.init_table(cx)
    h.add_member(cx, "p@x.com", "a@x.com"); h.add_member(cx, "p@x.com", "b@x.com")
    r = h.reassign_report(cx, "2026-06-25", "a@x.com", "b@x.com")
    assert r["ok"] is False and "no report" in r["error"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_household.py -q`
Expected: FAIL (`AttributeError: reassign_report`).

- [ ] **Step 3: Write minimal implementation** (append to `dashboard/household.py`)

```python
def reassign_report(cx, scan_date, from_email, to_email, *, by="console"):
    """Move a portal_biofield_reports row from one household member to another.
    Refuses cross-household moves and refuses to overwrite an existing report on the
    target for that date. Logs to scan_reassignments. Returns {"ok", "error"}."""
    f, t = _norm(from_email), _norm(to_email)
    sd = (scan_date or "").strip()
    if not f or not t or not sd:
        return {"ok": False, "error": "missing scan_date/from/to"}
    if f == t:
        return {"ok": False, "error": "from and to are the same account"}
    if not same_household(cx, f, t):
        return {"ok": False, "error": "from and to are not in the same household"}
    if not cx.execute("SELECT 1 FROM portal_biofield_reports WHERE email=? AND scan_date=? LIMIT 1",
                      (f, sd)).fetchone():
        return {"ok": False, "error": "no report for that account/date"}
    if cx.execute("SELECT 1 FROM portal_biofield_reports WHERE email=? AND scan_date=? LIMIT 1",
                  (t, sd)).fetchone():
        return {"ok": False, "error": "target already has a report for that date"}
    cx.execute("UPDATE portal_biofield_reports SET email=?, updated_at=? WHERE email=? AND scan_date=?",
               (t, _now(), f, sd))
    cx.execute("INSERT INTO scan_reassignments (scan_date, from_email, to_email, by, at) "
               "VALUES (?,?,?,?,?)", (sd, f, t, by, _now()))
    cx.commit()
    return {"ok": True, "error": None}


def list_reassignments(cx, limit=100):
    rows = cx.execute(
        "SELECT scan_date, from_email, to_email, by, at FROM scan_reassignments "
        "ORDER BY id DESC LIMIT ?", (int(limit),)).fetchall()
    return [{"scan_date": r[0], "from_email": r[1], "to_email": r[2], "by": r[3], "at": r[4]}
            for r in rows]
```

- [ ] **Step 4: Run to verify it passes** — GREEN.

- [ ] **Step 5: Commit**

```bash
git add dashboard/household.py tests/test_household.py
git commit -m "feat(household): within-household scan reassignment + audit log"
```

---

### Task 3: Guarded `?member=` switcher + household payload in `api_client_portal`

**Files:** Modify `app.py`; Test `tests/test_household_portal_route.py`

**Interfaces:**
- Consumes: `household.members_for`, `household.can_view`, `household.init_household_tables` (Tasks 1–2).
- Produces: `_household_view_enabled() -> bool`; `api_client_portal` payload gains `"household": [...]`; `?member=<email>` overrides `email_for_reports` when authorized.

**Context:** In `api_client_portal` (app.py:14287), `email_for_reports` is first computed at **app.py:14303** as `(portal.get("email") or "").strip().lower()`. Overriding it there re-points every downstream read (reports, dates, blur, D-brand) at the member — that is the whole switcher. The payload is returned at the `return jsonify(payload)` right after the `practitioner_brand` line (~app.py:14426).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_household_portal_route.py
import importlib, sqlite3, sys
from pathlib import Path
import pytest


def _app(tmp_path, monkeypatch, *, flag="1"):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("HOUSEHOLD_VIEW_ENABLED", flag)
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path: sys.path.insert(0, str(repo))
    try:
        import app as appmod; importlib.reload(appmod)
    except Exception as e:
        pytest.skip(f"app not importable: {e}")
    return appmod


def _seed(appmod, primary, member):
    from dashboard import client_portal as cp, household as h, portal_biofield_reports as pbr
    db = appmod.LOG_DB
    with sqlite3.connect(db) as cx:
        cp.init_client_portal_table(cx); h.init_household_tables(cx); pbr.init_table(cx)
        # a stable portal token for the primary
        token = cp.upsert_portal(cx, primary, "Karin", {}) if hasattr(cp, "upsert_portal") else None
        h.add_member(cx, primary, member, "Mochi", "pet")
        pbr.upsert_report(cx, primary, "2026-06-20", "s0", {"who": "primary"}, "confirmed")
        pbr.upsert_report(cx, member, "2026-06-25", "s1", {"who": "member"}, "confirmed")
        cx.commit()
    return token


def test_household_payload_and_member_switch(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    token = _seed(appmod, "karin@x.com", "mochi@x.com")
    if not token: pytest.skip("no portal upsert helper; wire token via _portal_record_for")
    c = appmod.app.test_client()
    # primary view carries the household list
    j = c.get(f"/api/portal/{token}").get_json()
    assert any(m["email"] == "mochi@x.com" for m in j.get("household", []))
    # ?member= to a LINKED member serves the member's scan dates
    jm = c.get(f"/api/portal/{token}?member=mochi@x.com").get_json()
    assert "2026-06-25" in (jm.get("bf_scan_dates") or jm.get("scan_dates") or [])
    # ?member= to an UNLINKED email falls back to the primary (no leak)
    js = c.get(f"/api/portal/{token}?member=stranger@x.com").get_json()
    assert "2026-06-20" in (js.get("bf_scan_dates") or js.get("scan_dates") or [])


def test_flag_off_no_household(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch, flag="0")
    token = _seed(appmod, "karin@x.com", "mochi@x.com")
    if not token: pytest.skip("no portal upsert helper")
    c = appmod.app.test_client()
    j = c.get(f"/api/portal/{token}?member=mochi@x.com").get_json()
    assert "household" not in j                       # no household key when flag off
    assert "2026-06-20" in (j.get("bf_scan_dates") or j.get("scan_dates") or [])  # served primary
```

> Implementer note: confirm the portal payload's scan-date key name by reading `api_client_portal` (it is `bf_scan_dates` in current code) and assert on the real key; confirm the helper used to mint a portal token for a test email (`_portal_record_for` / `client_portal.upsert_portal`) and seed accordingly.

- [ ] **Step 2: Run to verify it fails**

Run: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_household_portal_route.py -q`
Expected: FAIL (no `household` key / `?member=` ignored).

- [ ] **Step 3: Write minimal implementation**

Add the flag helper near the other portal helpers in `app.py`:

```python
def _household_view_enabled():
    return (os.environ.get("HOUSEHOLD_VIEW_ENABLED", "") or "").strip().lower() in ("1", "true", "yes")
```

In `api_client_portal`, immediately AFTER `email_for_reports = (portal.get("email") or "").strip().lower()` (app.py:14303), insert:

```python
    primary_email = email_for_reports
    household = []
    if _household_view_enabled() and primary_email:
        try:
            from dashboard import household as _hh
            with sqlite3.connect(LOG_DB) as _cxh:
                _hh.init_household_tables(_cxh)
                household = _hh.members_for(_cxh, primary_email)
                _req_member = (request.args.get("member") or "").strip().lower()
                if _req_member and _hh.can_view(_cxh, primary_email, _req_member):
                    email_for_reports = _req_member   # re-point the whole portal at the member
        except Exception as _e:
            print(f"[household] {_e!r}", flush=True)
            household = []
```

Immediately BEFORE `return jsonify(payload)` (the line after `payload["practitioner_brand"] = ...`, ~app.py:14426), insert:

```python
    if _household_view_enabled():
        payload["household"] = household
```

- [ ] **Step 4: Run to verify it passes** — GREEN.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_household_portal_route.py
git commit -m "feat(household): guarded ?member= switcher + household payload (flag-gated)"
```

---

### Task 4: Member selector in `static/client-portal.html`

**Files:** Modify `static/client-portal.html`

**Interfaces:**
- Consumes: the portal payload's `d.household` (`[{email,label,relationship}]` or absent).

This is UI-only (no pytest cycle).

- [ ] **Step 1: Add the selector markup + render**

In `static/client-portal.html`'s main `render(d, v)` (grep `function render`), near the TOP of the content (before the practitioner band / first card), when `d.household && d.household.length`, render a selector. Determine the current member from the URL (`?member=`), default = "you". Each option reloads the page with the chosen `?member=` (or removes it for "you"), preserving other query params. Escape all injected strings with the file's existing `esc()`. When `d.household` is absent/empty, render nothing.

```javascript
  // Household member switcher — the caregiver picks whose scans to view.
  if (d.household && d.household.length) {
    const params = new URLSearchParams(location.search);
    const cur = (params.get("member") || "").toLowerCase();
    const go = (email) => {
      const p = new URLSearchParams(location.search);
      if (email) p.set("member", email); else p.delete("member");
      p.delete("scan_date");                 // reset date when switching people
      location.search = p.toString();
    };
    window.__hhGo = go;                        // referenced by the inline handlers below
    let opts = `<option value="">You</option>`;
    for (const m of d.household) {
      const label = esc(m.label || m.email);
      const sel = (cur && cur === (m.email || "").toLowerCase()) ? " selected" : "";
      opts += `<option value="${esc(m.email)}"${sel}>${label}</option>`;
    }
    html += `<div class="card" style="display:flex;align-items:center;gap:10px">
      <span style="color:var(--muted);font-size:.9rem">Viewing scans for</span>
      <select onchange="window.__hhGo(this.value)"
        style="padding:6px 10px;border-radius:8px;background:var(--card);color:inherit;border:1px solid var(--border)">
        ${opts}
      </select></div>`;
  }
```

- [ ] **Step 2: Verify (static)**

Extract the inline `<script>` and `node --check` it; confirm the block reads `d.household`, reloads with `?member=`, and escapes labels/emails. Report that live browser render is pending (controller render-verifies).

- [ ] **Step 3: Commit**

```bash
git add static/client-portal.html
git commit -m "feat(household): member switcher on the client portal"
```

---

### Task 5: Console endpoints — household CRUD + reassignment

**Files:** Modify `app.py`; Test `tests/test_household_console_api.py`

**Interfaces:**
- Consumes: `household.{init_household_tables,add_member,remove_member,members_for,reassign_report}`; `portal_biofield_reports.{init_table,list_report_dates}`; `_portal_console_ok()` (app.py:13720).
- Produces (all JSON, all gated by `_portal_console_ok()`):
  - `GET /api/console/household?primary_email=` → `{"ok":True,"members":[{email,label,relationship,scan_dates:[...]}]}`
  - `POST /api/console/household` `{primary_email, member_email, label, relationship}` → `{"ok":True}`
  - `DELETE /api/console/household` `{primary_email, member_email}` → `{"ok":True}`
  - `POST /api/console/household/reassign` `{scan_date, from_email, to_email}` → `{"ok":bool,"error":?}` (400 when `ok` is False)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_household_console_api.py
import importlib, sqlite3, sys
from pathlib import Path
import pytest


def _app(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)   # auth open in test
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path: sys.path.insert(0, str(repo))
    try:
        import app as appmod; importlib.reload(appmod)
        import dashboard as _d; monkeypatch.setattr(_d, "CONSOLE_SECRET", "", raising=False)
    except Exception as e:
        pytest.skip(f"app not importable: {e}")
    return appmod


def test_console_household_crud_and_reassign(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    from dashboard import portal_biofield_reports as pbr
    with sqlite3.connect(appmod.LOG_DB) as cx:
        pbr.init_table(cx)
        pbr.upsert_report(cx, "wrong@x.com", "2026-06-25", "s1", {"n": 1}, "confirmed"); cx.commit()
    c = appmod.app.test_client()
    assert c.post("/api/console/household",
                  json={"primary_email": "p@x.com", "member_email": "wrong@x.com", "label": "W"}).status_code == 200
    assert c.post("/api/console/household",
                  json={"primary_email": "p@x.com", "member_email": "right@x.com", "label": "R"}).status_code == 200
    g = c.get("/api/console/household?primary_email=p@x.com").get_json()
    emails = {m["email"] for m in g["members"]}
    assert emails == {"wrong@x.com", "right@x.com"}
    assert "2026-06-25" in next(m for m in g["members"] if m["email"] == "wrong@x.com")["scan_dates"]
    # reassign wrong→right
    r = c.post("/api/console/household/reassign",
               json={"scan_date": "2026-06-25", "from_email": "wrong@x.com", "to_email": "right@x.com"})
    assert r.status_code == 200 and r.get_json()["ok"] is True
    assert pbr.list_report_dates(sqlite3.connect(appmod.LOG_DB), "right@x.com") == ["2026-06-25"]
    # cross-household reassign refused (400)
    bad = c.post("/api/console/household/reassign",
                 json={"scan_date": "2026-06-25", "from_email": "right@x.com", "to_email": "stranger@x.com"})
    assert bad.status_code == 400 and bad.get_json()["ok"] is False
    # delete a link
    assert c.delete("/api/console/household",
                    json={"primary_email": "p@x.com", "member_email": "wrong@x.com"}).status_code == 200
```

- [ ] **Step 2: Run to verify it fails**

Run: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_household_console_api.py -q`
Expected: FAIL (404 — routes not defined).

- [ ] **Step 3: Write minimal implementation** (add near the other `/api/console/*` routes in `app.py`)

```python
@app.route("/api/console/household", methods=["GET", "POST", "DELETE"])
def api_console_household():
    if not _portal_console_ok():
        return jsonify({"error": "unauthorized"}), 401
    from dashboard import household as _hh
    from dashboard import portal_biofield_reports as _pbr
    if request.method == "GET":
        primary = (request.args.get("primary_email") or "").strip().lower()
        with sqlite3.connect(LOG_DB) as cx:
            _hh.init_household_tables(cx); _pbr.init_table(cx)
            members = _hh.members_for(cx, primary)
            for m in members:
                m["scan_dates"] = _pbr.list_report_dates(cx, m["email"])
        return jsonify({"ok": True, "members": members})
    data = request.get_json(silent=True) or {}
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        _hh.init_household_tables(cx)
        if request.method == "POST":
            _hh.add_member(cx, data.get("primary_email"), data.get("member_email"),
                           data.get("label", ""), data.get("relationship", ""))
        else:  # DELETE
            _hh.remove_member(cx, data.get("primary_email"), data.get("member_email"))
    return jsonify({"ok": True})


@app.route("/api/console/household/reassign", methods=["POST"])
def api_console_household_reassign():
    if not _portal_console_ok():
        return jsonify({"error": "unauthorized"}), 401
    from dashboard import household as _hh
    from dashboard import portal_biofield_reports as _pbr
    data = request.get_json(silent=True) or {}
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        _hh.init_household_tables(cx); _pbr.init_table(cx)
        res = _hh.reassign_report(cx, data.get("scan_date"), data.get("from_email"),
                                  data.get("to_email"), by="console")
    return jsonify(res), (200 if res.get("ok") else 400)
```

- [ ] **Step 4: Run to verify it passes** — GREEN.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_household_console_api.py
git commit -m "feat(household): console CRUD + reassignment endpoints"
```

---

### Task 6: `/console/household` owner page

**Files:** Create `static/console-household.html`; Modify `app.py` (page route)

**Interfaces:**
- Consumes: `GET/POST/DELETE /api/console/household`, `POST /api/console/household/reassign` (Task 5).

This is UI + one route (no pytest cycle; verified by serving + node --check).

- [ ] **Step 1: Add the page route** in `app.py` (mirror `/console/portal-links` at app.py:12212):

```python
@app.route("/console/household")
def console_household_page():
    return send_from_directory("static", "console-household.html")
```

> Read app.py:12212 (`console_portal_links_page`) and match however it serves its static page (e.g. `send_from_directory("static", ...)` vs a render helper) exactly.

- [ ] **Step 2: Build `static/console-household.html`**

A single-page owner tool that: (a) takes a `primary_email`, (b) `GET`s the household + each member's scan dates, (c) shows a table with add/remove link controls, and (d) a reassignment form (scan_date + from + to → `POST /reassign`, surfacing `error` on 400). Include the console key from `?key=` in an `X-Console-Key` header (mirror how `static/console-portal-links.html` — or whatever the portal-links page file is named — sends its key). Keep it vanilla JS, escape injected strings.

Complete minimal page:

```html
<!doctype html><html><head><meta charset="utf-8"><title>Household</title>
<style>body{font-family:-apple-system,sans-serif;max-width:820px;margin:24px auto;padding:0 16px}
input,select,button{padding:6px 8px;margin:2px}table{border-collapse:collapse;width:100%}
td,th{border:1px solid #ccc;padding:6px;text-align:left;font-size:14px}.err{color:#b00}</style></head>
<body>
<h1>Household accounts</h1>
<p>Primary: <input id="primary" placeholder="caregiver@email"><button onclick="load()">Load</button></p>
<div id="out"></div>
<h3>Reassign a scan</h3>
<p>Date <input id="rd" placeholder="2026-06-25"> from <input id="rf" placeholder="wrong@email">
to <input id="rt" placeholder="right@email"> <button onclick="reassign()">Reassign</button>
<span id="rerr" class="err"></span></p>
<script>
const KEY = new URLSearchParams(location.search).get("key") || "";
const H = {"Content-Type":"application/json","X-Console-Key":KEY};
const esc = s => String(s==null?"":s).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
async function load(){
  const p = document.getElementById("primary").value.trim().toLowerCase();
  const r = await fetch("/api/console/household?primary_email="+encodeURIComponent(p),{headers:H});
  const j = await r.json();
  let h = `<table><tr><th>Member</th><th>Label</th><th>Rel</th><th>Scan dates</th><th></th></tr>`;
  for(const m of (j.members||[])){
    h += `<tr><td>${esc(m.email)}</td><td>${esc(m.label)}</td><td>${esc(m.relationship)}</td>
      <td>${(m.scan_dates||[]).map(esc).join(", ")}</td>
      <td><button onclick="rm('${esc(m.email)}')">remove</button></td></tr>`;
  }
  h += `</table><p>Add member: <input id="me" placeholder="member@email">
    <input id="ml" placeholder="label"> <input id="mrel" placeholder="pet/child/…">
    <button onclick="add()">Add</button></p>`;
  document.getElementById("out").innerHTML = h;
}
async function add(){
  const p=document.getElementById("primary").value.trim().toLowerCase();
  await fetch("/api/console/household",{method:"POST",headers:H,body:JSON.stringify({
    primary_email:p, member_email:document.getElementById("me").value.trim(),
    label:document.getElementById("ml").value, relationship:document.getElementById("mrel").value})});
  load();
}
async function rm(email){
  const p=document.getElementById("primary").value.trim().toLowerCase();
  await fetch("/api/console/household",{method:"DELETE",headers:H,
    body:JSON.stringify({primary_email:p, member_email:email})});
  load();
}
async function reassign(){
  document.getElementById("rerr").textContent="";
  const r = await fetch("/api/console/household/reassign",{method:"POST",headers:H,body:JSON.stringify({
    scan_date:document.getElementById("rd").value.trim(),
    from_email:document.getElementById("rf").value.trim(),
    to_email:document.getElementById("rt").value.trim()})});
  const j = await r.json();
  if(!j.ok){ document.getElementById("rerr").textContent = j.error || "failed"; return; }
  load();
}
</script></body></html>
```

- [ ] **Step 3: Verify (static)** — `node --check` the inline script; serve `static/` and load `/console-household.html?key=` to confirm it renders. Report live render-verify pending.

- [ ] **Step 4: Commit**

```bash
git add app.py static/console-household.html
git commit -m "feat(household): /console/household owner page"
```

---

## Self-Review

**Spec coverage:**
- Link data model (`household_members`, `scan_reassignments`) → Task 1 + Task 2. ✓
- `members_for` / `can_view` / `same_household` guards → Task 1. ✓
- Guarded `?member=` switcher + `household` payload + fail-closed + flag → Task 3. ✓
- Member's own paywall unchanged → Task 3 (overriding `email_for_reports` re-uses `_portal_biofield_unlocked(email_for_reports)` downstream; no blur change). ✓
- Member selector UI → Task 4. ✓
- Reassignment (within-household, no-clobber, audit) → Task 2 + console wiring Task 5. ✓
- Console CRUD + reassign endpoints, `_portal_console_ok()` gated, ungated by feature flag → Task 5. ✓
- `/console/household` page → Task 6. ✓
- No `portal_biofield_reports` schema / pipeline / E4L change → confirmed (only re-keys `email` on an existing row). ✓

**Placeholder scan:** Task 3 + Task 6 carry implementer notes to confirm the real payload date-key (`bf_scan_dates`) and the portal-links page's serve pattern rather than guessing. No TBD/TODO, no dead scaffold left.

**Type consistency:** `reassign_report -> {"ok","error"}` (Task 2) is what Task 5's endpoint returns and 400-maps. `members_for -> [{email,label,relationship}]` (Task 1) is what Task 3's payload and Task 5's GET (which adds `scan_dates`) and Task 4's `d.household` consume. `can_view(cx, viewer, target)` used identically in Task 3. `HOUSEHOLD_VIEW_ENABLED` gates payload + `?member=` in Task 3 only; console tasks never read it.
