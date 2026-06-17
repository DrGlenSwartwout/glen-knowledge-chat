# Portal Multi-Scan Biofield History — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Keep every biofield scan as its own dated, independently-statused report; let clients navigate by date tabs (newest default); recent scans actionable, older read-only; capture edits for AI training.

**Architecture:** A new `portal_biofield_reports` table (one evolving row per email+scan_date) becomes the source of truth for the biofield analysis; `client_portals.content_json` remains a no-migration legacy fallback. Reads/transitions/editor become scan-date-aware; a 30-day window gates actionability; publish logs a training correction.

**Tech Stack:** Flask, sqlite, the existing `client_portal`/`portal_view`/`ghl_queue` modules. Tests: `doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" ~/.venvs/deploy-chat311/bin/python -m pytest`.

**Spec:** `docs/superpowers/specs/2026-06-17-portal-multiscan-history-design.md`

---

## File Structure
- **Create** `dashboard/portal_biofield_reports.py` — table + accessors + `is_actionable` window helper.
- **Modify** `dashboard/portal_view.py` — `_biofield_block` scan-date-aware (+ `get_portal_view` threads `scan_date`).
- **Modify** `app.py` — content endpoint, transitions, editor publish + review-queue, `/admin/portal/upsert`, corrections endpoint.
- **Modify** `static/client-portal.html` — date tabs.
- **Modify** `02 Skills/e4l-portal-import.py` (vault) — send `scan_date`/`scan_id`.
- **Tests:** `tests/test_portal_biofield_reports.py` (new), extend `tests/test_portal_view.py`, `tests/test_client_portal_routes.py`, `tests/test_console_biofield_portal.py`.

**Convention:** every accessor takes `cx` first; emails normalized `(email or "").strip().lower()`; mirror `dashboard/client_portal.py`. Full-suite baseline before this plan: **1739 passed, 2 skipped, 0 failed**.

---

## Task 1: reports table + accessors + window helper

**Files:** Create `dashboard/portal_biofield_reports.py`; Test `tests/test_portal_biofield_reports.py`

- [ ] **Step 1: Failing test** — `tests/test_portal_biofield_reports.py`

```python
import sqlite3, datetime
from dashboard import portal_biofield_reports as R


def _cx(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "t.db")); R.init_table(cx); return cx


def test_upsert_get_and_overwrite(tmp_path):
    cx = _cx(tmp_path)
    R.upsert_report(cx, "a@x.com", "2026-06-05", "s1", {"layers": [{"n": 1, "title": "t"}]}, "ai_draft")
    rep = R.get_report(cx, "a@x.com", "2026-06-05")
    assert rep["status"] == "ai_draft" and rep["scan_id"] == "s1"
    assert rep["content"]["layers"][0]["title"] == "t"
    # same (email, scan_date) overwrites content+status, not a 2nd row
    R.upsert_report(cx, "a@x.com", "2026-06-05", "s1", {"layers": []}, "confirmed")
    assert R.get_report(cx, "a@x.com", "2026-06-05")["status"] == "confirmed"
    assert R.list_report_dates(cx, "a@x.com") == ["2026-06-05"]


def test_list_dates_newest_first_and_latest(tmp_path):
    cx = _cx(tmp_path)
    for d in ["2026-04-01", "2026-06-05", "2026-05-02"]:
        R.upsert_report(cx, "a@x.com", d, "s", {"layers": []}, "ai_draft")
    assert R.list_report_dates(cx, "a@x.com") == ["2026-06-05", "2026-05-02", "2026-04-01"]
    assert R.latest_report(cx, "a@x.com")["scan_date"] == "2026-06-05"
    assert R.get_report(cx, "a@x.com", "nope") is None
    assert R.latest_report(cx, "nobody@x.com") is None


def test_set_status(tmp_path):
    cx = _cx(tmp_path)
    R.upsert_report(cx, "a@x.com", "2026-06-05", "s", {"layers": []}, "ai_draft")
    assert R.set_report_status(cx, "a@x.com", "2026-06-05", "requested") is True
    assert R.get_report(cx, "a@x.com", "2026-06-05")["status"] == "requested"
    assert R.set_report_status(cx, "a@x.com", "missing", "requested") is False


def test_is_actionable_window():
    today = "2026-06-17"
    assert R.is_actionable("2026-06-05", today) is True     # within 30 days
    assert R.is_actionable("2026-06-17", today) is True     # today
    assert R.is_actionable("2026-05-17", today) is True     # exactly 31 days? -> see impl (30-day inclusive)
    assert R.is_actionable("2026-04-01", today) is False    # > 30 days
    assert R.is_actionable("", today) is False              # no/garbage date
```

- [ ] **Step 2: Run → FAIL** (`-m pytest tests/test_portal_biofield_reports.py -q`)

- [ ] **Step 3: Implement** — `dashboard/portal_biofield_reports.py`

```python
"""Per-scan biofield reports: one evolving row per (email, scan_date).
Source of truth for the biofield analysis; client_portals.content_json is the
legacy fallback when a client has no rows here. See the 2026-06-17 spec."""
import datetime
import json
import sqlite3


def _now_iso() -> str:
    return datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"


def init_table(cx) -> None:
    cx.execute("""
        CREATE TABLE IF NOT EXISTS portal_biofield_reports (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            email        TEXT,
            scan_date    TEXT,
            scan_id      TEXT,
            content_json TEXT,
            status       TEXT,
            created_at   TEXT,
            updated_at   TEXT,
            UNIQUE(email, scan_date)
        )
    """)
    cx.execute("CREATE INDEX IF NOT EXISTS ix_pbr_email ON portal_biofield_reports(email)")
    cx.commit()


def upsert_report(cx, email, scan_date, scan_id, content, status):
    email = (email or "").strip().lower()
    now = _now_iso()
    row = cx.execute(
        "SELECT id FROM portal_biofield_reports WHERE email=? AND scan_date=?",
        (email, scan_date)).fetchone()
    cj = json.dumps(content or {})
    if row:
        cx.execute("UPDATE portal_biofield_reports SET scan_id=?, content_json=?, "
                   "status=?, updated_at=? WHERE id=?",
                   (scan_id, cj, status, now, row[0]))
    else:
        cx.execute("INSERT INTO portal_biofield_reports "
                   "(email, scan_date, scan_id, content_json, status, created_at, updated_at) "
                   "VALUES (?,?,?,?,?,?,?)",
                   (email, scan_date, scan_id, cj, status, now, now))
    cx.commit()


def _row_to_dict(row):
    try:
        content = json.loads(row[3] or "{}")
    except Exception:
        content = {}
    return {"scan_date": row[1], "scan_id": row[2], "content": content, "status": row[4]}


def get_report(cx, email, scan_date):
    email = (email or "").strip().lower()
    row = cx.execute("SELECT id, scan_date, scan_id, content_json, status "
                     "FROM portal_biofield_reports WHERE email=? AND scan_date=?",
                     (email, scan_date)).fetchone()
    return _row_to_dict(row) if row else None


def list_report_dates(cx, email):
    email = (email or "").strip().lower()
    rows = cx.execute("SELECT scan_date FROM portal_biofield_reports WHERE email=? "
                      "ORDER BY scan_date DESC", (email,)).fetchall()
    return [r[0] for r in rows]


def latest_report(cx, email):
    email = (email or "").strip().lower()
    row = cx.execute("SELECT id, scan_date, scan_id, content_json, status "
                     "FROM portal_biofield_reports WHERE email=? "
                     "ORDER BY scan_date DESC LIMIT 1", (email,)).fetchone()
    return _row_to_dict(row) if row else None


def set_report_status(cx, email, scan_date, status):
    email = (email or "").strip().lower()
    cur = cx.execute("UPDATE portal_biofield_reports SET status=?, updated_at=? "
                     "WHERE email=? AND scan_date=?", (status, _now_iso(), email, scan_date))
    cx.commit()
    return cur.rowcount > 0


def is_actionable(scan_date, today):
    """A scan is actionable (CTAs/transitions allowed) within 30 days of `today`.
    Both args are 'YYYY-MM-DD'. Bad/empty scan_date -> False."""
    try:
        sd = datetime.date.fromisoformat(scan_date)
        td = datetime.date.fromisoformat(today)
    except (ValueError, TypeError):
        return False
    return 0 <= (td - sd).days <= 30
```

- [ ] **Step 4: Run → PASS.** Note `2026-05-17` is exactly 31 days before `2026-06-17` → `is_actionable` returns **False**; fix that test line to `assert R.is_actionable("2026-05-18", today) is True` (30 days) before running.

- [ ] **Step 5: Commit** (`-m "portal: portal_biofield_reports table + accessors + 30-day window"`)

---

## Task 2: `_biofield_block` scan-date-aware (/view path)

**Files:** Modify `dashboard/portal_view.py`; Test `tests/test_portal_view.py`

- [ ] **Step 1: Failing test** (add to `tests/test_portal_view.py`; reuse the file's `_conn`/`_add_person` helpers and `import datetime`)

```python
def test_biofield_uses_reports_newest_default_with_tabs(tmp_path):
    from dashboard import portal_view as pv, portal_biofield_reports as R
    import datetime
    cx = _conn(tmp_path); R.init_table(cx)
    pid = _add_person(cx, "m@example.com", "M")
    today = datetime.date.today()
    new_d = today.isoformat()
    old_d = (today - datetime.timedelta(days=60)).isoformat()
    R.upsert_report(cx, "m@example.com", old_d, "s0",
                    {"layers": [{"n": 1, "title": "Old", "meaning": "o", "remedy": "X", "dosing": "1"}]}, "ai_draft")
    R.upsert_report(cx, "m@example.com", new_d, "s1",
                    {"layers": [{"n": 1, "title": "New", "meaning": "n", "remedy": "Y", "dosing": "2"}]}, "interested")
    bf = pv.get_portal_view(cx, pid)["biofield"]            # default newest
    assert bf["scan_date"] == new_d and bf["scan_dates"] == [new_d, old_d]
    assert bf["status"] == "interested" and bf["blurred"] is True and bf["actionable"] is True
    assert "remedy" not in bf["layers"][0]                  # blurred
    # explicit older scan -> read-only, not actionable
    bf_old = pv.get_portal_view(cx, pid, scan_date=old_d)["biofield"]
    assert bf_old["scan_date"] == old_d and bf_old["actionable"] is False
    assert "remedy" not in bf_old["layers"][0]              # old + unconfirmed -> still blurred, no CTA


def test_biofield_legacy_fallback_when_no_reports(tmp_path):
    from dashboard import portal_view as pv
    from dashboard import client_portal as cp
    cx = _conn(tmp_path)
    pid = _add_person(cx, "leg@example.com", "Leg")
    cp.upsert_portal(cx, "leg@example.com", "Leg",
                     {"layers": [{"n": 1, "title": "C", "meaning": "m", "remedy": "R", "dosing": "d"}]})
    bf = pv.get_portal_view(cx, pid)["biofield"]
    assert bf["scan_dates"] == [] and bf["blurred"] is False        # legacy = confirmed, no tabs
    assert bf["layers"][0]["remedy"] == "R"
```

- [ ] **Step 2: Run → FAIL** (`-m pytest tests/test_portal_view.py -q`)

- [ ] **Step 3: Implement** — in `dashboard/portal_view.py`:
  1. Add at top: `import datetime` and `from dashboard import portal_biofield_reports as _pbr` (use the module's existing import style; if it imports `client_portal as _cp`, add `_pbr` alongside).
  2. Change the signature `def get_portal_view(cx, person_id, *, offers_enabled_keys=None):` → add `scan_date=None`: `def get_portal_view(cx, person_id, *, offers_enabled_keys=None, scan_date=None):` and change the call at line ~134 to `"biofield": _biofield_block(cx, email, scan_date=scan_date),`.
  3. Replace `_biofield_block` with:

```python
def _biofield_block(cx, email, scan_date=None):
    try:
        _pbr.init_table(cx)
        dates = _pbr.list_report_dates(cx, email)
    except Exception:
        dates = []
    if dates:
        picked = scan_date if (scan_date in dates) else dates[0]
        rep = _pbr.get_report(cx, email, picked) or {}
        content = rep.get("content") or {}
        status = rep.get("status") or "confirmed"
        today = datetime.date.today().isoformat()
        actionable = (status != "confirmed") and _pbr.is_actionable(picked, today)
        return _assemble_biofield(content, status, scan_date=picked,
                                  scan_dates=dates, actionable=actionable)
    # Legacy fallback: single confirmed report, no tabs.
    try:
        rec = _cp.get_portal_content_by_email(cx, email)
    except Exception:
        rec = None
    content = (rec or {}).get("content") or {}
    if not (content.get("greeting") or content.get("layers") or content.get("video")):
        return {"visible": False}
    return _assemble_biofield(content, "confirmed", scan_date=None,
                              scan_dates=[], actionable=False)


def _assemble_biofield(content, status, *, scan_date, scan_dates, actionable):
    confirmed = status == "confirmed"
    layers = []
    for L in (content.get("layers") or []):
        item = {"n": L.get("n"), "title": L.get("title", ""), "meaning": L.get("meaning", "")}
        if confirmed:  # unconfirmed remedies NEVER leave the server
            item["remedy"] = L.get("remedy", "")
            item["dosing"] = L.get("dosing", "")
        layers.append(item)
    return {"visible": True, "status": status, "blurred": not confirmed,
            "actionable": actionable, "scan_date": scan_date, "scan_dates": scan_dates,
            "greeting": content.get("greeting", ""), "video": content.get("video") or {},
            "layers": layers, "pricing_note": content.get("pricing_note", "") if confirmed else ""}
```

(`_cp` is the existing `client_portal` alias in `portal_view.py`.)

- [ ] **Step 4: Run → PASS** (the existing `test_view_shows_biofield_when_portal_content_present` and PR #156 blur tests still pass — they have no reports rows → legacy fallback, confirmed.)

- [ ] **Step 5: Run FULL suite → no regressions.** **Step 6: Commit** (`-m "portal: scan-date-aware biofield block (reports + legacy fallback + window)"`)

---

## Task 3: content endpoint scan-date-aware

**Files:** Modify `app.py` (`api_client_portal`, line ~7238); Test `tests/test_client_portal_routes.py`

- [ ] **Step 1: Failing test** (reuse `_seed_portal` + the `client` fixture; add `import datetime`)

```python
def test_content_endpoint_reports_newest_and_scan_date_param(client):
    c, appmod = client
    from dashboard import portal_biofield_reports as R
    import sqlite3, datetime
    tok = _seed_portal(appmod, "ms@y.com", "MS", {"layers": []})  # ensures token row
    cx = sqlite3.connect(appmod.LOG_DB); R.init_table(cx)
    today = datetime.date.today().isoformat()
    old = (datetime.date.today() - datetime.timedelta(days=60)).isoformat()
    R.upsert_report(cx, "ms@y.com", today, "s1",
                    {"layers": [{"n": 1, "title": "New", "remedy": "Y", "dosing": "2"}]}, "interested")
    R.upsert_report(cx, "ms@y.com", old, "s0",
                    {"layers": [{"n": 1, "title": "Old", "remedy": "X", "dosing": "1"}]}, "confirmed")
    cx.close()
    j = c.get(f"/api/portal/{tok}").get_json()             # newest default
    assert j["scan_date"] == today and j["scan_dates"] == [today, old]
    assert j["blurred"] is True and "remedy" not in j["layers"][0]
    j2 = c.get(f"/api/portal/{tok}?scan_date={old}").get_json()   # older confirmed -> revealed
    assert j2["scan_date"] == old and j2["blurred"] is False and j2["layers"][0]["remedy"] == "X"
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement** — in `api_client_portal` (`app.py` ~7238). After loading `portal` and `content = dict(portal.get("content") or {})`, build the biofield view from reports with the same blur/window. Replace the block that currently sets `bf_status`/`bf_confirmed`/`bf_layers` (added in PR #156) and the `return jsonify({...})` with:

```python
    from dashboard import portal_biofield_reports as _pbr
    import datetime as _dt
    _pbr.init_table(cx_for_reports := sqlite3.connect(LOG_DB))
    email_for_reports = (portal.get("email") or "").strip().lower()
    dates = _pbr.list_report_dates(cx_for_reports, email_for_reports) if email_for_reports else []
    req_date = (request.args.get("scan_date") or "").strip()
    if dates:
        picked = req_date if req_date in dates else dates[0]
        rep = _pbr.get_report(cx_for_reports, email_for_reports, picked) or {}
        bf_content = rep.get("content") or {}
        bf_status = rep.get("status") or "confirmed"
        bf_scan_date, bf_scan_dates = picked, dates
        bf_actionable = (bf_status != "confirmed") and _pbr.is_actionable(
            picked, _dt.date.today().isoformat())
    else:
        bf_content, bf_status = content, content.get("biofield_status") or "confirmed"
        bf_scan_date, bf_scan_dates, bf_actionable = None, [], False
    cx_for_reports.close()
    bf_confirmed = bf_status == "confirmed"
    bf_layers = []
    for L in (bf_content.get("layers") or []):
        item = {"n": L.get("n"), "title": L.get("title", ""), "meaning": L.get("meaning", "")}
        if bf_confirmed:
            item["remedy"] = L.get("remedy", "")
            item["dosing"] = L.get("dosing", "")
        bf_layers.append(item)
    # reorder items come from the SELECTED report's content (falls back to legacy content)
    reorder_src = bf_content.get("reorder_items") if dates else content.get("reorder_items")
    display = []
    for it in (reorder_src or []):
        slug = (it.get("slug") or "").strip()
        p = _get_product(slug) if slug else None
        regular = (p or {}).get("price_cents")
        override = it.get("price_cents")
        special = int(override) if override is not None else regular
        display.append({
            "slug": slug, "qty": int(it.get("qty", 1) or 1),
            "name": (p or {}).get("name", slug), "price_cents": special,
            "regular_price_cents": regular,
            "is_special": bool(override is not None and regular is not None and int(override) < int(regular)),
            "available": bool(p)})
    return jsonify({
        "name": portal.get("name"),
        "biofield_status": bf_status, "blurred": not bf_confirmed,
        "actionable": bf_actionable, "scan_date": bf_scan_date, "scan_dates": bf_scan_dates,
        "greeting": bf_content.get("greeting", ""),
        "video": bf_content.get("video") or {},
        "layers": bf_layers,
        "pricing_note": bf_content.get("pricing_note", "") if bf_confirmed else "",
        "reorder_items": display,
    })
```

Note: the existing handler opens `with sqlite3.connect(LOG_DB) as cx:` only around `_portal_record_for`. The reorder enrichment runs outside that. Use a fresh `sqlite3.connect(LOG_DB)` for reports as shown (closed after). Verify `portal` dict exposes `email` (from `_portal_record_for`); if it does not, fetch it: the record is keyed by token — add `email` to what `_portal_record_for` returns, or look it up. CHECK during impl and adapt.

- [ ] **Step 4: Run → PASS.** **Step 5: FULL suite (PR #156 content-endpoint blur tests still pass via legacy fallback).** **Step 6: Commit** (`-m "portal: content endpoint scan-date-aware (reports + tabs + window)"`)

---

## Task 4: transitions carry scan_date + window enforcement

**Files:** Modify `app.py` (`_biofield_transition` ~7469 + the two routes); Test `tests/test_client_portal_routes.py`

- [ ] **Step 1: Failing test**

```python
def test_transition_targets_scan_date_and_rejects_out_of_window(client):
    c, appmod = client
    from dashboard import portal_biofield_reports as R
    import sqlite3, datetime
    tok = _seed_portal(appmod, "tw@y.com", "TW", {"layers": []})
    cx = sqlite3.connect(appmod.LOG_DB); R.init_table(cx)
    today = datetime.date.today().isoformat()
    old = (datetime.date.today() - datetime.timedelta(days=60)).isoformat()
    R.upsert_report(cx, "tw@y.com", today, "s1", {"layers": []}, "ai_draft")
    R.upsert_report(cx, "tw@y.com", old, "s0", {"layers": []}, "ai_draft")
    cx.close()
    # recent scan: actionable
    r = c.post(f"/api/portal/{tok}/biofield/request", json={"scan_date": today})
    assert r.status_code == 200 and r.get_json()["status"] == "requested"
    cx = sqlite3.connect(appmod.LOG_DB)
    assert R.get_report(cx, "tw@y.com", today)["status"] == "requested"
    # old scan: rejected, unchanged
    r2 = c.post(f"/api/portal/{tok}/biofield/request", json={"scan_date": old})
    assert r2.status_code == 409
    assert R.get_report(cx, "tw@y.com", old)["status"] == "ai_draft"
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement** — replace `_biofield_transition` (`app.py` ~7469) with a reports-aware version:

```python
def _biofield_transition(token, new_status, tag):
    from dashboard import client_portal as _cp
    from dashboard import portal_identity as _pi
    from dashboard import portal_biofield_reports as _pbr
    from dashboard import ghl_queue as _gq
    import datetime as _dt
    sess = request.cookies.get("rm_portal_session", "")
    body = request.get_json(silent=True) or {}
    req_date = (body.get("scan_date") or request.args.get("scan_date") or "").strip()
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        _cp.init_client_portal_table(cx)
        _pi._ensure_people_table(cx)
        _pbr.init_table(cx)
        ident = _pi.resolve_identity(cx, token=token, session_token=sess,
                                     client_login_enabled=_client_login_enabled())
        if ident is None:
            return jsonify({"error": "not found"}), 404
        dates = _pbr.list_report_dates(cx, ident.email)
        if dates:
            picked = req_date if req_date in dates else dates[0]
            if not _pbr.is_actionable(picked, _dt.date.today().isoformat()):
                return jsonify({"error": "scan no longer actionable"}), 409
            if not _pbr.set_report_status(cx, ident.email, picked, new_status):
                return jsonify({"error": "not found"}), 404
        else:
            # legacy single-content portal: keep PR #156 behavior on client_portals
            if not _cp.set_biofield_status(cx, ident.email, new_status):
                return jsonify({"error": "not found"}), 404
        try:
            _gq.init_ghl_queue_table(cx)
            _gq.enqueue(cx, op="tag_add", email=ident.email,
                        payload={"tags": [tag]}, actor="portal")
        except Exception as e:
            print(f"[biofield-transition] ghl enqueue failed: {e!r}", flush=True)
    return jsonify({"ok": True, "status": new_status})
```

The two route functions `api_portal_biofield_interest` / `_request` are unchanged (they call `_biofield_transition`).

- [ ] **Step 4: Run → PASS.** (The PR #156 transition tests use legacy portals → exercise the `else` branch → still pass.) **Step 5: FULL suite.** **Step 6: Commit** (`-m "portal: transitions target scan_date + enforce 30-day window"`)

---

## Task 5: training-correction capture

**Files:** Modify `app.py` (new helper + table + console endpoint); Test `tests/test_console_biofield_portal.py`

- [ ] **Step 1: Failing test**

```python
def test_corrections_logged_and_listable(client):
    c, appmod = client
    import app as appmod2  # the module under test
    import sqlite3
    with sqlite3.connect(appmod.LOG_DB) as cx:
        appmod2._log_biofield_correction(cx, "corr@y.com", "2026-06-05",
                                         {"layers": [{"n": 1, "title": "T", "remedy": "Real FF"}]})
    j = c.get("/api/console/biofield/corrections?key=test-secret&since=2000-01-01").get_json()
    hit = [x for x in j["corrections"] if x["email"] == "corr@y.com" and x["scan_date"] == "2026-06-05"]
    assert hit and hit[0]["content"]["layers"][0]["remedy"] == "Real FF"


def test_corrections_requires_key(client):
    c, _ = client
    assert c.get("/api/console/biofield/corrections").status_code == 401
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement** — add to `app.py` (near the other biofield routes):

```python
def _init_biofield_corrections(cx):
    cx.execute("""CREATE TABLE IF NOT EXISTS biofield_corrections (
        id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT, scan_date TEXT,
        content_json TEXT, created_at TEXT)""")
    cx.commit()


def _log_biofield_correction(cx, email, scan_date, content):
    _init_biofield_corrections(cx)
    cx.execute("INSERT INTO biofield_corrections (email, scan_date, content_json, created_at) "
               "VALUES (?,?,?,?)",
               ((email or "").strip().lower(), scan_date or "",
                json.dumps(content or {}), _now_iso_utc()))
    cx.commit()


@app.route("/api/console/biofield/corrections", methods=["GET"])
def api_console_biofield_corrections():
    if not _portal_console_ok():
        return jsonify({"error": "unauthorized"}), 401
    since = (request.args.get("since") or "").strip()
    out = []
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _init_biofield_corrections(cx)
        q = "SELECT email, scan_date, content_json, created_at FROM biofield_corrections"
        args = ()
        if since:
            q += " WHERE created_at >= ?"; args = (since,)
        q += " ORDER BY created_at ASC"
        for r in cx.execute(q, args).fetchall():
            try:
                content = json.loads(r["content_json"] or "{}")
            except Exception:
                content = {}
            out.append({"email": r["email"], "scan_date": r["scan_date"],
                        "content": content, "created_at": r["created_at"]})
    return jsonify({"corrections": out})
```

Verify a timestamp helper exists in `app.py` for `_now_iso_utc()`; if the app uses a different name (e.g. `_now_iso`/`datetime.utcnow().isoformat()`), use that. Confirm `json` is imported at module top (it is).

- [ ] **Step 4: Run → PASS.** **Step 5: FULL suite.** **Step 6: Commit** (`-m "portal: biofield training-correction log + console read endpoint"`)

---

## Task 6: editor publish + review-queue scan-date-aware

**Files:** Modify `app.py` (`api_console_biofield_publish` ~7344, `api_console_biofield_review_queue` ~7400); Test `tests/test_console_biofield_portal.py`

- [ ] **Step 1: Failing test**

```python
def test_publish_writes_dated_report_and_logs_correction(client):
    c, appmod = client
    from dashboard import portal_biofield_reports as R
    import sqlite3
    r = c.post("/api/console/biofield-portal?key=test-secret", json={
        "email": "pub@y.com", "name": "Pub", "scan_date": "2026-06-05",
        "content": {"layers": [{"n": 1, "title": "Calm", "remedy": "Real FF"}]}})
    assert r.status_code == 200
    cx = sqlite3.connect(appmod.LOG_DB); R.init_table(cx)
    rep = R.get_report(cx, "pub@y.com", "2026-06-05")
    assert rep is not None and rep["status"] == "confirmed"
    # correction logged
    j = c.get("/api/console/biofield/corrections?key=test-secret&since=2000-01-01").get_json()
    assert any(x["email"] == "pub@y.com" and x["scan_date"] == "2026-06-05" for x in j["corrections"])


def test_review_queue_lists_requested_reports_with_dates(client):
    c, appmod = client
    from dashboard import portal_biofield_reports as R
    import sqlite3, datetime
    cx = sqlite3.connect(appmod.LOG_DB); R.init_table(cx)
    today = datetime.date.today().isoformat()
    R.upsert_report(cx, "rq@y.com", today, "s1", {"layers": []}, "requested")
    R.upsert_report(cx, "rq2@y.com", today, "s2", {"layers": []}, "ai_draft")
    cx.close()
    j = c.get("/api/console/biofield/review-queue?key=test-secret").get_json()
    hits = [(x["email"], x.get("scan_date")) for x in j["queue"]]
    assert ("rq@y.com", today) in hits and all(e != "rq2@y.com" for e, _ in hits)
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3a: publish** — in `api_console_biofield_publish` (~7344): read `scan_date = (body.get("scan_date") or "").strip()`. Inside the `with _db_lock, sqlite3.connect(LOG_DB) as cx:` block, after the existing `token, pid = _cp.upsert_portal(cx, email, name, content)` and the PR #156 `set_biofield_status`/tag lines, add:

```python
        if scan_date:
            from dashboard import portal_biofield_reports as _pbr
            _pbr.init_table(cx)
            _pbr.upsert_report(cx, email, scan_date,
                               (body.get("scan_id") or ""), content, "confirmed")
        _log_biofield_correction(cx, email, scan_date, content)
```

(Publishing always confirms; when `scan_date` is supplied it writes the dated report, otherwise it keeps the legacy single-content path from PR #156. The correction is logged either way.)

- [ ] **Step 3b: review-queue** — replace the body of `api_console_biofield_review_queue` (~7400) to scan the reports table for `requested`:

```python
@app.route("/api/console/biofield/review-queue", methods=["GET"])
def api_console_biofield_review_queue():
    if not _portal_console_ok():
        return jsonify({"error": "unauthorized"}), 401
    from dashboard import client_portal as _cp
    from dashboard import portal_biofield_reports as _pbr
    queue = []
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _cp.init_client_portal_table(cx)
        _pbr.init_table(cx)
        names = {r["email"]: r["name"] for r in
                 cx.execute("SELECT email, name FROM client_portals").fetchall()}
        rows = cx.execute("SELECT email, scan_date, updated_at FROM portal_biofield_reports "
                          "WHERE status='requested' ORDER BY updated_at DESC").fetchall()
        for r in rows:
            queue.append({"email": r["email"], "name": names.get(r["email"], ""),
                          "scan_date": r["scan_date"], "requested_at": r["updated_at"]})
        # legacy single-content portals still at 'requested'
        for r in cx.execute("SELECT email, name, updated_at FROM client_portals").fetchall():
            if _cp.get_biofield_status(cx, r["email"]) == "requested" \
               and not _pbr.list_report_dates(cx, r["email"]):
                queue.append({"email": r["email"], "name": r["name"],
                              "scan_date": None, "requested_at": r["updated_at"]})
    return jsonify({"queue": queue})
```

- [ ] **Step 4: Run → PASS** (PR #156 `test_review_queue_lists_only_requested` / `test_publish_confirms_status` still pass via the legacy branch). **Step 5: FULL suite.** **Step 6: Commit** (`-m "portal: editor publish writes dated report + correction; review-queue by scan_date"`)

---

## Task 7: `/admin/portal/upsert` routes to reports; importer sends scan_date

**Files:** Modify `app.py` (`admin_client_portal_upsert` ~7626); Modify `02 Skills/e4l-portal-import.py` (vault); Test `tests/test_client_portal_routes.py`

- [ ] **Step 1: Failing test** (admin route)

```python
def test_admin_upsert_with_scan_date_writes_report(client):
    c, appmod = client
    from dashboard import portal_biofield_reports as R
    import sqlite3
    r = c.post("/admin/portal/upsert?key=test-secret", json={
        "email": "ad@y.com", "name": "Ad", "scan_date": "2026-06-05", "scan_id": "s9",
        "content": {"biofield_status": "ai_draft", "layers": [{"n": 1, "title": "T", "remedy": "R"}]}})
    assert r.status_code == 200
    cx = sqlite3.connect(appmod.LOG_DB); R.init_table(cx)
    rep = R.get_report(cx, "ad@y.com", "2026-06-05")
    assert rep is not None and rep["status"] == "ai_draft" and rep["scan_id"] == "s9"
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3a: admin route** — in `admin_client_portal_upsert` (~7626), after `token, pid = _cp.upsert_portal(cx, email, name, content)` inside its `with` block, add:

```python
        scan_date = (body.get("scan_date") or "").strip()
        if scan_date:
            from dashboard import portal_biofield_reports as _pbr
            _pbr.init_table(cx)
            _pbr.upsert_report(cx, email, scan_date, (body.get("scan_id") or ""),
                               content, content.get("biofield_status") or "ai_draft")
```

(Without `scan_date` it behaves exactly as today. The `client_portals` token row is still ensured so the portal link exists.)

- [ ] **Step 3b: importer** — in `02 Skills/e4l-portal-import.py`, the importer already has `scan` (with `scan["scan_id"]`, `scan["scan_date"]`). In `publish_ai_draft`, include them in the POST body. Change the function to accept and send them:

```python
def publish_ai_draft(email, name, content, scan_date, scan_id):
    content = dict(content)
    content["biofield_status"] = "ai_draft"
    key = os.environ["CONSOLE_SECRET"]
    body = json.dumps({"email": email, "name": name, "content": content,
                       "scan_date": scan_date, "scan_id": scan_id}).encode("utf-8")
    req = urllib.request.Request(
        "https://illtowell.com/admin/portal/upsert", data=body, method="POST",
        headers={"Content-Type": "application/json", "X-Console-Key": key})
    return json.load(urllib.request.urlopen(req, timeout=30))
```

and update the call site in `main()`:
```python
    if a.publish_draft:
        try:
            r = publish_ai_draft(a.email, name, content, scan["scan_date"], str(scan["scan_id"]))
            print(f"published ai_draft for scan {scan['scan_date']} → "
                  f"{r.get('url') or 'updated existing portal (link unchanged)'}")
        except Exception as ex:
            print(f"[publish-draft] FAILED: {ex!r}")
            sys.exit(2)
```

- [ ] **Step 4: Run admin test → PASS.** Then `python3 -m py_compile "02 Skills/e4l-portal-import.py"` → OK. (Do NOT run the importer's live POST.) **Step 5: FULL suite.** **Step 6: Commit app.py + the vault file is auto-snapshotted** (`-m "portal: /admin/portal/upsert routes scan_date to reports; importer sends date"`)

---

## Task 8: page UI — date tabs

**Files:** Modify `static/client-portal.html` (no unit test; manual smoke)

- [ ] **Step 1:** Read the biofield section (built in PR #156 from `v.biofield`). The block now carries `scan_dates` (newest-first), `scan_date` (selected), and `actionable`. Add, above the biofield render:
  - If `bf.scan_dates.length > 1`: render a tab row — the **first 3** dates as buttons (the selected one marked active) + a **"More dates ▾"** `<select>`/dropdown containing all dates. If `≤ 1` date (or `scan_dates` empty), render no tabs.
  - Each tab/dropdown change sets a module-level `selectedScanDate` and calls the existing `load()`; `load()` must append `?scan_date=<selectedScanDate>` to BOTH the `/api/portal/<seg>` and `/api/portal/<seg>/view` fetches when `selectedScanDate` is set.
  - Date display: format `scan_date` (YYYY-MM-DD) to a short human label (e.g. `Jun 5`); keep the raw value as the option value.
- [ ] **Step 2:** Gate the CTAs on `bf.actionable`: the "View your scan analysis" / "Request my remedy matches" buttons render only when `bf.actionable === true`. When `!actionable` and not confirmed, show the blurred patterns + the AI-disclosure line but no button (read-only history). When `confirmed`, full render (unchanged).
- [ ] **Step 3:** The transition POSTs (`advanceBiofield`) must include the selected scan_date: `fetch(.../biofield/${step}, {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({scan_date: selectedScanDate || bf.scan_date})})`.
- [ ] **Step 4: Verify** — `node --check` on the extracted `<script>`; run the FULL Python suite (no served-page test should break). Brand rules: no emojis, "Order" not "Reorder", dark-green/gold, AI-disclosure until confirmed.
- [ ] **Step 5: Commit** (`-m "portal page: biofield date tabs + per-scan CTAs"`)

---

## Task 9: Othon's 6/5 scan — first real report, FF names corrected

**Files:** local (vault data + a one-off publish); no app code

- [ ] **Step 1:** Open `~/AI-Training/05 Clients/Othon Molina e4l-portal-seed 2026-06-16.json`. For each layer, verify the `remedy` is a REAL catalog product (cross-check against `~/deploy-chat/data/products.json`). Fix invented names — known: L2 `Atlas Balance Formula`, L6 `Vision Support`; also re-check `Liver Support`, `Connective Tissue Support`, `Immune Modulation`, `Nerve Repair`. Replace each non-matching name with the closest real product (or blank the remedy if none fits — leave it for review).
- [ ] **Step 2:** Determine Othon's scan_date from `e4l.db` (his latest scan; expected `2026-06-05`): `sqlite3 ~/AI-Training/e4l.db "SELECT scan_id, scan_date FROM e4l_scans s JOIN e4l_clients c ON c.client_id=s.client_id WHERE lower(c.email)='backdoc.molina@gmail.com' ORDER BY scan_date DESC LIMIT 1"` (adapt join columns to the real schema).
- [ ] **Step 3:** Publish as his first dated report via the console editor at `illtowell.com/console/biofield-portal` (load his email, paste the corrected content, set the scan date, publish) — OR a one-off authenticated POST to `/admin/portal/upsert` with `scan_date`, `scan_id`, and the corrected content (status will become `confirmed` only via the editor publish path; for an `ai_draft` use `/admin/portal/upsert`). Confirm via `GET /api/portal/<his-token>?scan_date=2026-06-05` that the corrected remedies render.
- [ ] **Step 4:** No commit (data only; vault auto-snapshots the corrected seed file).

---

## Task 10: final suite + PR
- [ ] Run the FULL suite → all green (baseline 1739 + new tests).
- [ ] Push `sess/5326cc61-multiscan`; open PR (base main) summarizing the multi-scan history feature + the live smoke-test checklist (drive a 2-scan portal: newest actionable, older read-only; tab switch; publish older → reveal).

---

## Self-Review notes
- **Spec coverage:** data model (T1), scan-date-aware reads + tabs + window on both endpoints (T2 /view, T3 content), per-scan transitions + window (T4), training capture (T5), editor publish + review-queue by date (T6), importer/admin scan_date routing (T7), date-tab UI (T8), Othon fix (T9), legacy fallback covered in T2/T3/T4/T6. Deferred items (access-gating/offer/purchase) correctly absent; per-scan GHL tags still fire.
- **Type consistency:** report dict `{scan_date, scan_id, content, status}`; biofield block adds `scan_date`, `scan_dates`, `actionable` (plus PR #156's `status`, `blurred`); accessors `upsert_report/get_report/list_report_dates/latest_report/set_report_status/is_actionable`; `is_actionable(scan_date, today)` 0–30 days inclusive; correction `{email, scan_date, content, created_at}`.
- **Verify during impl:** `_portal_record_for`/`portal` dict exposes `email` (T3) — adapt if not; the app's UTC timestamp helper name (T5); `e4l_scans`↔`e4l_clients` join columns (T9); confirm PR #156 tests pass through the new legacy branches at each task.
