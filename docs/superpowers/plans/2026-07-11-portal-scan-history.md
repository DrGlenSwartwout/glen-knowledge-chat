# Portal Scan History + Orders Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the client portal into three tabs (Current Analysis · Scan History · Orders & Invoices) surfacing all past scans and invoices, with a managed `current_scan_date` pointer (auto-advance with client opt-out, console set-current, confirm-to-send notify on the non-Gmail bulk channel).

**Architecture:** Extend the existing single-page portal (`static/client-portal.html`) with a client-side tab bar over the current cards — no new renderer or server pages. The "current scan" is a single JSON key `current_scan_date` in `client_portals.content_json`, written by three paths (auto-advance on ingest, console set-current, client pin). Client notification opt-in reuses the existing `notify_state` system. Everything ships behind two OFF flags.

**Tech Stack:** Python 3.11 / Flask (`app.py`), SQLite (`chat_log.db` / `LOG_DB`), vanilla-JS SPA (`static/client-portal.html`), pytest.

## Global Constraints

- **Additive-key payload rule:** any new `api_client_portal` payload key is added AFTER the base `payload` dict, guarded by its `_xxx_enabled()` flag, best-effort in try/except, so flag-off ⇒ byte-identical response. (`app.py:16386-16500` set the precedent.)
- **Flag idiom (copy verbatim, 4-tuple):** `os.environ.get("NAME","").strip().lower() in ("1","true","yes","on")`.
- **`content_json` has no merge helper** — read-modify-write the whole dict, mirroring `client_portal.set_biofield_status` (`dashboard/client_portal.py:203-217`). `upsert_portal` REPLACES `content_json`.
- **Copy rules (client-facing text):** no em dashes, no ALL CAPS.
- **Durable flag flip:** `doppler secrets set <FLAG>=1 -p remedy-match -c prd` (Doppler is source; Render is pruned).
- **Test command (app-importing tests need env):** `doppler run -p remedy-match -c dev -- python3 -m pytest tests/<file> -v` (venv alt: `~/.venvs/deploy-chat311/bin/python -m pytest tests/<file> -v`).
- **Never send scan notifications via primary consumer Gmail** — use `inbox.send_bulk` (GHL-v2/Mailgun domain when `BULK_VIA_GHL` set); keep `PORTAL_SCAN_NOTIFY_ENABLED` off until that is configured.
- **Two new flags:** `PORTAL_SCAN_HISTORY_ENABLED` (whole UI + prefs endpoints), `PORTAL_SCAN_NOTIFY_ENABLED` (confirm-to-send emails). Both OFF by default.

---

### Task 1: Portal content helpers — auto_advance + current_scan_date

**Files:**
- Modify: `dashboard/client_portal.py` (add helpers after `set_biofield_status`, ~`:217`)
- Test: `tests/test_client_portal_scan_prefs.py` (create)

**Interfaces:**
- Consumes: existing `upsert_portal`, `init_client_portal_table`, `_now_iso` in the module.
- Produces:
  - `get_auto_advance(cx, email) -> bool` (default `True` when key absent)
  - `set_auto_advance(cx, email, on: bool) -> bool` (rowcount>0)
  - `get_current_scan(cx, email) -> str | None`
  - `set_current_scan(cx, email, scan_date: str) -> bool` (writes `content["current_scan_date"]`)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_client_portal_scan_prefs.py
import sqlite3
from dashboard import client_portal as cp

def _cx(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "chat_log.db"))
    cp.init_client_portal_table(cx)
    return cx

def test_auto_advance_defaults_true_and_toggles(tmp_path):
    cx = _cx(tmp_path)
    cp.upsert_portal(cx, "a@x.com", "A", {"biofield_status": "confirmed"})
    assert cp.get_auto_advance(cx, "a@x.com") is True          # absent = default on
    assert cp.set_auto_advance(cx, "a@x.com", False) is True
    assert cp.get_auto_advance(cx, "a@x.com") is False
    # unrelated content preserved
    assert cp.get_portal_content_by_email(cx, "a@x.com")["content"]["biofield_status"] == "confirmed"

def test_current_scan_set_and_get(tmp_path):
    cx = _cx(tmp_path)
    cp.upsert_portal(cx, "a@x.com", "A", {})
    assert cp.get_current_scan(cx, "a@x.com") is None
    assert cp.set_current_scan(cx, "a@x.com", "2026-07-09") is True
    assert cp.get_current_scan(cx, "a@x.com") == "2026-07-09"

def test_set_on_missing_portal_returns_false(tmp_path):
    cx = _cx(tmp_path)
    assert cp.set_auto_advance(cx, "nobody@x.com", False) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_client_portal_scan_prefs.py -v`
Expected: FAIL — `AttributeError: module 'dashboard.client_portal' has no attribute 'get_auto_advance'`

- [ ] **Step 3: Write minimal implementation** (append to `dashboard/client_portal.py`, mirroring `set_biofield_status`)

```python
def _read_content(cx, email):
    row = cx.execute("SELECT content_json FROM client_portals WHERE email=?",
                     (email.strip().lower(),)).fetchone()
    if not row:
        return None
    try:
        return json.loads(row[0] or "{}")
    except Exception:
        return {}

def _write_content(cx, email, content) -> bool:
    cur = cx.execute("UPDATE client_portals SET content_json=?, updated_at=? WHERE email=?",
                     (json.dumps(content), _now_iso(), email.strip().lower()))
    cx.commit()
    return cur.rowcount > 0

def get_auto_advance(cx, email) -> bool:
    content = _read_content(cx, email)
    if content is None:
        return True
    return bool(content.get("auto_advance", True))

def set_auto_advance(cx, email, on: bool) -> bool:
    content = _read_content(cx, email)
    if content is None:
        return False
    content["auto_advance"] = bool(on)
    return _write_content(cx, email, content)

def get_current_scan(cx, email):
    content = _read_content(cx, email)
    return (content or {}).get("current_scan_date")

def set_current_scan(cx, email, scan_date: str) -> bool:
    content = _read_content(cx, email)
    if content is None:
        return False
    content["current_scan_date"] = scan_date
    return _write_content(cx, email, content)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_client_portal_scan_prefs.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add dashboard/client_portal.py tests/test_client_portal_scan_prefs.py
git commit -m "feat(portal): content helpers for auto_advance + current_scan_date"
```

---

### Task 2: Flag helpers + payload keys + dangling-pointer guard

**Files:**
- Modify: `app.py` (add two flag helpers near `_ff_matches_enabled` ~`:15045`; add payload keys after base dict ~`:16380`; guard in selection block ~`:16256`)
- Test: `tests/test_portal_scan_history_payload.py` (create)

**Interfaces:**
- Consumes: Task 1 helpers; existing `_portal_record_for`, `_pbr.list_report_dates`, `_pbr.get_report`.
- Produces: `_portal_scan_history_enabled() -> bool`, `_portal_scan_notify_enabled() -> bool`; payload keys `scan_history_enabled` (bool), `auto_advance` (bool), `current_scan_date` (str|None) present only when the flag is on.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_portal_scan_history_payload.py
import importlib, sqlite3, sys
from pathlib import Path
import pytest

def _app(monkeypatch, tmp_db):
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        app = importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")
    monkeypatch.setattr(app, "LOG_DB", str(tmp_db))
    app._init_workspace_schema()
    return app

def _seed(app, tmp_db):
    from dashboard import client_portal as cp, portal_biofield_reports as pbr
    with sqlite3.connect(str(tmp_db)) as cx:
        cp.init_client_portal_table(cx)
        pbr.init_table(cx)
        token, _ = cp.upsert_portal(cx, "a@x.com", "A", {"biofield_status": "confirmed"})
        pbr.upsert_report(cx, "a@x.com", "2026-07-02", "111", {"greeting": "old"}, "confirmed")
        pbr.upsert_report(cx, "a@x.com", "2026-07-09", "222", {"greeting": "new"}, "confirmed")
    return token

def test_flag_off_no_new_keys(monkeypatch, tmp_db):
    monkeypatch.delenv("PORTAL_SCAN_HISTORY_ENABLED", raising=False)
    app = _app(monkeypatch, tmp_db)
    token = _seed(app, tmp_db)
    j = app.app.test_client().get(f"/api/portal/{token}").get_json()
    assert "scan_history_enabled" not in j
    assert "auto_advance" not in j

def test_flag_on_exposes_prefs_and_current(monkeypatch, tmp_db):
    monkeypatch.setenv("PORTAL_SCAN_HISTORY_ENABLED", "1")
    app = _app(monkeypatch, tmp_db)
    token = _seed(app, tmp_db)
    j = app.app.test_client().get(f"/api/portal/{token}").get_json()
    assert j["scan_history_enabled"] is True
    assert j["auto_advance"] is True
    assert j["scan_date"] == "2026-07-09"          # newest when no pointer
    assert j["scan_dates"] == ["2026-07-09", "2026-07-02"]

def test_dangling_pointer_falls_to_newest(monkeypatch, tmp_db):
    monkeypatch.setenv("PORTAL_SCAN_HISTORY_ENABLED", "1")
    app = _app(monkeypatch, tmp_db)
    token = _seed(app, tmp_db)
    from dashboard import client_portal as cp
    with sqlite3.connect(str(tmp_db)) as cx:
        cp.set_current_scan(cx, "a@x.com", "2099-01-01")   # points nowhere
    j = app.app.test_client().get(f"/api/portal/{token}").get_json()
    assert j["scan_date"] == "2026-07-09"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_portal_scan_history_payload.py -v`
Expected: FAIL — `KeyError: 'scan_history_enabled'` / `auto_advance` in the flag-on tests.

- [ ] **Step 3a: Add flag helpers** (`app.py`, near `_ff_matches_enabled`)

```python
def _portal_scan_history_enabled() -> bool:
    """Three-tab portal history UI + prefs endpoints. Default OFF — payload byte-identical when off."""
    return os.environ.get("PORTAL_SCAN_HISTORY_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")

def _portal_scan_notify_enabled() -> bool:
    """Confirm-to-send new-analysis emails. Default OFF until the bulk (non-Gmail) channel is configured."""
    return os.environ.get("PORTAL_SCAN_NOTIFY_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")
```

- [ ] **Step 3b: Add the dangling-pointer guard** in the selection block (`app.py:~16254`), changing the `elif cur_ptr ...` branch:

```python
    elif cur_ptr and cur_ptr in dates:
        picked = cur_ptr
    else:
        if cur_ptr and cur_ptr not in dates:
            app.logger.warning("portal current_scan_date %r not in reports for %s; using newest",
                               cur_ptr, email_for_reports)
        picked = dates[0]
```

- [ ] **Step 3c: Add payload keys** after the base `payload` dict (`app.py:~16380`), following the additive-key rule:

```python
    if _portal_scan_history_enabled():
        try:
            from dashboard import client_portal as _cp_sh
            with sqlite3.connect(LOG_DB) as _cx_sh:
                _aa = _cp_sh.get_auto_advance(_cx_sh, email_for_reports) if email_for_reports else True
            payload["scan_history_enabled"] = True
            payload["auto_advance"] = _aa
            payload["current_scan_date"] = bf_scan_date
        except Exception:
            pass
```

- [ ] **Step 4: Run test to verify it passes**

Run: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_portal_scan_history_payload.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_portal_scan_history_payload.py
git commit -m "feat(portal): scan-history flags, payload prefs, dangling-pointer guard"
```

---

### Task 3: Client prefs endpoint — POST /api/portal/<token>/scan-prefs

**Files:**
- Modify: `app.py` (add route near `api_portal_scene_pref` ~`:16679`)
- Test: `tests/test_portal_scan_prefs_endpoint.py` (create)

**Interfaces:**
- Consumes: `_portal_record_for`, `_portal_scan_history_enabled`, Task 1 `set_auto_advance`/`set_current_scan`; existing `notify_state.set_opt` (reused for the notification opt-in).
- Produces: route `POST /api/portal/<token>/scan-prefs` accepting `{auto_advance?: bool, pin_scan_date?: str, notify?: "in"|"out"}`; returns `{"ok": True, ...}`; 404 bad token; 403 when flag off.

Note: `pin_scan_date` sets `current_scan_date` AND `auto_advance=False` ("keep showing me this one"). Notification opt reuses the existing `notify_state` system (do not add a second concept).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_portal_scan_prefs_endpoint.py
import importlib, sqlite3, sys
from pathlib import Path
import pytest

def _app(monkeypatch, tmp_db):
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        app = importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")
    monkeypatch.setattr(app, "LOG_DB", str(tmp_db))
    app._init_workspace_schema()
    return app

def _seed(tmp_db):
    from dashboard import client_portal as cp
    with sqlite3.connect(str(tmp_db)) as cx:
        cp.init_client_portal_table(cx)
        token, _ = cp.upsert_portal(cx, "a@x.com", "A", {})
    return token

def test_flag_off_403(monkeypatch, tmp_db):
    monkeypatch.delenv("PORTAL_SCAN_HISTORY_ENABLED", raising=False)
    app = _app(monkeypatch, tmp_db)
    token = _seed(tmp_db)
    r = app.app.test_client().post(f"/api/portal/{token}/scan-prefs", json={"auto_advance": False})
    assert r.status_code == 403

def test_pin_sets_current_and_disables_autoadvance(monkeypatch, tmp_db):
    monkeypatch.setenv("PORTAL_SCAN_HISTORY_ENABLED", "1")
    app = _app(monkeypatch, tmp_db)
    token = _seed(tmp_db)
    r = app.app.test_client().post(f"/api/portal/{token}/scan-prefs", json={"pin_scan_date": "2026-07-02"})
    assert r.status_code == 200
    from dashboard import client_portal as cp
    with sqlite3.connect(str(tmp_db)) as cx:
        assert cp.get_current_scan(cx, "a@x.com") == "2026-07-02"
        assert cp.get_auto_advance(cx, "a@x.com") is False

def test_bad_token_404(monkeypatch, tmp_db):
    monkeypatch.setenv("PORTAL_SCAN_HISTORY_ENABLED", "1")
    app = _app(monkeypatch, tmp_db)
    r = app.app.test_client().post("/api/portal/BADTOKEN/scan-prefs", json={"auto_advance": False})
    assert r.status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_portal_scan_prefs_endpoint.py -v`
Expected: FAIL — 404 for all (route not registered) / 403 test fails.

- [ ] **Step 3: Write the route** (`app.py`, mirroring `api_portal_scene_pref`)

```python
@app.route("/api/portal/<token>/scan-prefs", methods=["POST"])
def api_portal_scan_prefs(token):
    """Client-set scan preferences: auto_advance on/off, pin a scan as current
    (pin also turns auto_advance off), and notification opt in/out (reuses notify_state)."""
    if not _portal_scan_history_enabled():
        return jsonify({"error": "not found"}), 403
    body = request.get_json(silent=True) or {}
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        from dashboard import client_portal as _cp, notify_state as _ns
        _cp.init_client_portal_table(cx)
        portal = _portal_record_for(cx, token)
        if not portal:
            return jsonify({"error": "not found"}), 404
        email = (portal.get("email") or "").strip().lower()
        pin = (body.get("pin_scan_date") or "").strip()
        if pin:
            _cp.set_current_scan(cx, email, pin)
            _cp.set_auto_advance(cx, email, False)
        elif "auto_advance" in body:
            _cp.set_auto_advance(cx, email, bool(body.get("auto_advance")))
        notify = (body.get("notify") or "").strip().lower()
        if notify in ("in", "out"):
            _ns.set_opt(cx, email, notify)
    return jsonify({"ok": True})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_portal_scan_prefs_endpoint.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_portal_scan_prefs_endpoint.py
git commit -m "feat(portal): client scan-prefs endpoint (auto_advance, pin, notify opt)"
```

---

### Task 4: Console set-current — POST /api/console/portal/set-current

**Files:**
- Modify: `app.py` (add route near `api_console_biofield_publish` ~`:18085`)
- Test: `tests/test_console_set_current.py` (create)

**Interfaces:**
- Consumes: `_portal_console_ok`, Task 1 `set_current_scan`, `_pbr.list_report_dates`.
- Produces: `POST /api/console/portal/set-current` `{email, scan_date}` → `{"ok": True}`; 401 unauth; 400 if the scan_date has no report row (guards against dangling pointer).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_console_set_current.py
import pytest, sqlite3

@pytest.fixture
def client(monkeypatch, tmp_path):
    import app as appmod
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    appmod._init_auth_tables()
    monkeypatch.setattr(appmod, "CONSOLE_SECRET", "test-secret")
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client(), appmod

def _seed(appmod):
    from dashboard import client_portal as cp, portal_biofield_reports as pbr
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cp.init_client_portal_table(cx); pbr.init_table(cx)
        cp.upsert_portal(cx, "a@x.com", "A", {})
        pbr.upsert_report(cx, "a@x.com", "2026-07-02", "1", {}, "confirmed")

def _auth(c):
    return {"X-Console-Secret": "test-secret"}

def test_set_current_ok(client):
    c, appmod = client; _seed(appmod)
    r = c.post("/api/console/portal/set-current", json={"email": "a@x.com", "scan_date": "2026-07-02"}, headers=_auth(c))
    assert r.status_code == 200
    from dashboard import client_portal as cp
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert cp.get_current_scan(cx, "a@x.com") == "2026-07-02"

def test_set_current_unknown_date_400(client):
    c, appmod = client; _seed(appmod)
    r = c.post("/api/console/portal/set-current", json={"email": "a@x.com", "scan_date": "2030-01-01"}, headers=_auth(c))
    assert r.status_code == 400

def test_set_current_requires_auth(client):
    c, appmod = client; _seed(appmod)
    r = c.post("/api/console/portal/set-current", json={"email": "a@x.com", "scan_date": "2026-07-02"})
    assert r.status_code == 401
```

Note: confirm the console-auth header/mechanism by reading `_portal_console_ok` (`app.py`); adjust `_auth()` to match how `tests/test_console_biofield_portal.py` authenticates (it sets `CONSOLE_SECRET`) if the header name differs.

- [ ] **Step 2: Run test to verify it fails**

Run: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_console_set_current.py -v`
Expected: FAIL — 404/405 (route missing).

- [ ] **Step 3: Write the route**

```python
@app.route("/api/console/portal/set-current", methods=["POST"])
def api_console_portal_set_current():
    """Operator: point a portal at an existing scan_date (authoritative, independent of auto_advance)."""
    if not _portal_console_ok():
        return jsonify({"error": "unauthorized"}), 401
    body = request.get_json(silent=True) or {}
    email = (body.get("email") or "").strip().lower()
    scan_date = (body.get("scan_date") or "").strip()
    if not email or not scan_date:
        return jsonify({"error": "email and scan_date required"}), 400
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        from dashboard import client_portal as _cp, portal_biofield_reports as _pbr
        _cp.init_client_portal_table(cx); _pbr.init_table(cx)
        if scan_date not in _pbr.list_report_dates(cx, email):
            return jsonify({"error": "no report for that scan_date"}), 400
        if not _cp.set_current_scan(cx, email, scan_date):
            return jsonify({"error": "portal not found"}), 404
    return jsonify({"ok": True})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_console_set_current.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_console_set_current.py
git commit -m "feat(portal): console set-current endpoint with dangling-date guard"
```

---

### Task 5: Auto-advance honored on ingest

**Files:**
- Modify: `app.py` `api_console_biofield_publish` (the `content["current_scan_date"] = scan_date` write ~`:18038`)
- Test: `tests/test_biofield_publish_autoadvance.py` (create)

**Interfaces:**
- Consumes: Task 1 `get_auto_advance`; existing publish flow.
- Produces: on publish, `current_scan_date` moves to the new scan only when the portal's `auto_advance` is on; the dated report row is always written regardless.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_biofield_publish_autoadvance.py
import pytest, sqlite3

@pytest.fixture
def client(monkeypatch, tmp_path):
    import app as appmod
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    appmod._init_auth_tables()
    monkeypatch.setattr(appmod, "CONSOLE_SECRET", "test-secret")
    # never actually send during this test
    monkeypatch.setattr(appmod, "_send_full_report_email", lambda *a, **k: ("console-log", None))
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client(), appmod

def _pub(c, appmod, scan_date):
    return c.post("/api/console/biofield-portal",
                  json={"email": "a@x.com", "name": "A", "scan_date": scan_date,
                        "scan_id": scan_date, "content": {"greeting": scan_date}},
                  headers={"X-Console-Secret": "test-secret"})

def test_optout_keeps_pointer_but_still_stores_report(client):
    c, appmod = client
    _pub(c, appmod, "2026-07-02")
    from dashboard import client_portal as cp, portal_biofield_reports as pbr
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cp.set_auto_advance(cx, "a@x.com", False)
    _pub(c, appmod, "2026-07-09")
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert cp.get_current_scan(cx, "a@x.com") == "2026-07-02"        # pointer unmoved
        assert "2026-07-09" in pbr.list_report_dates(cx, "a@x.com")      # but report stored
```

- [ ] **Step 2: Run test to verify it fails**

Run: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_biofield_publish_autoadvance.py -v`
Expected: FAIL — pointer is `2026-07-09` (auto-advance not yet conditional).

- [ ] **Step 3: Make the pointer write conditional** (`app.py:~18036-18039`)

Replace:
```python
    if scan_date:
        content["current_scan_date"] = scan_date
```
with (read the existing pref BEFORE the write; default on for new/absent portals):
```python
    if scan_date:
        _aa_on = True
        try:
            with sqlite3.connect(LOG_DB) as _cx_aa:
                from dashboard import client_portal as _cp_aa
                _cp_aa.init_client_portal_table(_cx_aa)
                _aa_on = _cp_aa.get_auto_advance(_cx_aa, email)
        except Exception:
            _aa_on = True
        if _aa_on:
            content["current_scan_date"] = scan_date
        else:
            _existing = None
            try:
                with sqlite3.connect(LOG_DB) as _cx_cur:
                    from dashboard import client_portal as _cp_cur
                    _existing = _cp_cur.get_current_scan(_cx_cur, email)
            except Exception:
                _existing = None
            if _existing:
                content["current_scan_date"] = _existing   # preserve the client's pin through the content replace
```

Rationale: `upsert_portal` REPLACES `content_json`, so when opted out we must carry the existing `current_scan_date` forward or the pin is lost.

- [ ] **Step 4: Run test to verify it passes**

Run: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_biofield_publish_autoadvance.py -v`
Expected: PASS. Also re-run the existing publish test to confirm no regression:
Run: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_console_biofield_portal.py -v`
Expected: PASS (unchanged).

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_biofield_publish_autoadvance.py
git commit -m "feat(portal): honor auto_advance opt-out on scan ingest"
```

---

### Task 6: Confirm-to-send notify — POST /api/console/portal/notify-scan

**Files:**
- Modify: `app.py` (add route near Task 4's)
- Test: `tests/test_console_notify_scan.py` (create)

**Interfaces:**
- Consumes: `_portal_console_ok`, `_portal_scan_notify_enabled`, `notify_state` opt lookup, `inbox.send_bulk`, `client_portal.portal_link_for`.
- Produces: `POST /api/console/portal/notify-scan` `{email}` → sends the "new analysis ready" email via `inbox.send_bulk` only when: flag on AND client notify-opt is not "out". Returns `{"ok": True, "sent": bool, "reason": str}`. Never calls the Gmail-first `_send_full_report_email`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_console_notify_scan.py
import pytest, sqlite3

@pytest.fixture
def client(monkeypatch, tmp_path):
    import app as appmod
    from dashboard import inbox as _inbox
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    appmod._init_auth_tables()
    monkeypatch.setattr(appmod, "CONSOLE_SECRET", "test-secret")
    sent = {}
    def _fake_bulk(to_email, subject, body, from_name=None, html=None):
        sent.update(to=to_email, subject=subject); return {"id": "x", "via": "ghl"}
    monkeypatch.setattr(_inbox, "send_bulk", _fake_bulk)
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client(), appmod, sent

def _seed(appmod):
    from dashboard import client_portal as cp
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cp.init_client_portal_table(cx)
        cp.upsert_portal(cx, "a@x.com", "A", {})

def _hdr(): return {"X-Console-Secret": "test-secret"}

def test_flag_off_does_not_send(client, monkeypatch):
    c, appmod, sent = client; _seed(appmod)
    monkeypatch.delenv("PORTAL_SCAN_NOTIFY_ENABLED", raising=False)
    r = c.post("/api/console/portal/notify-scan", json={"email": "a@x.com"}, headers=_hdr())
    assert r.get_json()["sent"] is False and not sent

def test_flag_on_sends_via_bulk(client, monkeypatch):
    c, appmod, sent = client; _seed(appmod)
    monkeypatch.setenv("PORTAL_SCAN_NOTIFY_ENABLED", "1")
    r = c.post("/api/console/portal/notify-scan", json={"email": "a@x.com"}, headers=_hdr())
    assert r.get_json()["sent"] is True and sent.get("to") == "a@x.com"

def test_opted_out_does_not_send(client, monkeypatch):
    c, appmod, sent = client; _seed(appmod)
    monkeypatch.setenv("PORTAL_SCAN_NOTIFY_ENABLED", "1")
    from dashboard import notify_state as ns
    with sqlite3.connect(appmod.LOG_DB) as cx:
        ns.set_opt(cx, "a@x.com", "out")
    r = c.post("/api/console/portal/notify-scan", json={"email": "a@x.com"}, headers=_hdr())
    assert r.get_json()["sent"] is False and not sent
```

Note: verify the `notify_state` opt getter name (e.g. `get_opt`/`is_opted_in`) by reading `dashboard/notify_state.py`, and the `send_bulk` signature, before finalizing; adjust the opt-check line accordingly.

- [ ] **Step 2: Run test to verify it fails**

Run: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_console_notify_scan.py -v`
Expected: FAIL — route missing.

- [ ] **Step 3: Write the route**

```python
@app.route("/api/console/portal/notify-scan", methods=["POST"])
def api_console_portal_notify_scan():
    """Operator confirm-to-send: email the client that a new analysis is ready.
    Sends via inbox.send_bulk (GHL-v2/Mailgun domain), never the Gmail-first path.
    Gated by PORTAL_SCAN_NOTIFY_ENABLED and the client's notify opt state."""
    if not _portal_console_ok():
        return jsonify({"error": "unauthorized"}), 401
    email = ((request.get_json(silent=True) or {}).get("email") or "").strip().lower()
    if not email:
        return jsonify({"error": "email required"}), 400
    if not _portal_scan_notify_enabled():
        return jsonify({"ok": True, "sent": False, "reason": "flag off"})
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        from dashboard import client_portal as _cp, notify_state as _ns, inbox as _inbox
        _cp.init_client_portal_table(cx)
        if _ns.get_opt(cx, email) == "out":
            return jsonify({"ok": True, "sent": False, "reason": "opted out"})
        link = _cp.portal_link_for(cx, email)   # canonical myhealingoasis.com portal URL
    subject = "Your new analysis is ready"
    body = ("Aloha,\n\nYour newest analysis is ready in your portal.\n\n"
            f"{link}\n\nIn wellness,\nDr. Glen & Rae")
    _inbox.send_bulk(email, subject, body, from_name="Dr. Glen & Rae")
    return jsonify({"ok": True, "sent": True})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_console_notify_scan.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_console_notify_scan.py
git commit -m "feat(portal): confirm-to-send scan notify via bulk channel, opt-gated"
```

---

### Task 7: Front-end three tabs + family grouping (`static/client-portal.html`)

**Files:**
- Modify: `static/client-portal.html` (tab bar markup; `showTab()` JS; Scan History render from `scan_dates` + `household`; Orders tab from `invoices`/`past_invoices`; client toggles POSTing to `/scan-prefs`)
- Verify: headless-Chrome render (no JS test framework in repo)

**Interfaces:**
- Consumes payload keys: `scan_history_enabled`, `auto_advance`, `current_scan_date`, `scan_date`, `scan_dates`, `household` (already emitted when household view on), `invoices`, `past_invoices`.
- Produces: no server interface; UI only.

This task is UI, so its gate is a render-verification checklist rather than pytest. Before editing, read the current card layout and the existing `.scantabs` switcher + "History & receipts" section so the new markup matches house style.

- [ ] **Step 1: Add the tab bar + panels.** Only render tabs when `payload.scan_history_enabled` is true; otherwise leave the page exactly as today. Representative markup to insert above the card stack:

```html
<div class="tabbar" id="portalTabs" hidden>
  <button class="tab is-active" data-tab="current" onclick="showTab('current')">Current Analysis</button>
  <button class="tab" data-tab="history" onclick="showTab('history')">Scan History</button>
  <button class="tab" data-tab="orders" onclick="showTab('orders')">Orders &amp; Invoices</button>
</div>
<section data-panel="current"></section>   <!-- existing analysis cards move inside -->
<section data-panel="history" hidden></section>
<section data-panel="orders" hidden></section>
```

CSS note: guard visibility with `[data-panel][hidden]{display:none!important}` so a later `display:` rule can't override the `hidden` attribute (see memory `hidden-attr-vs-display-css`).

- [ ] **Step 2: `showTab()` JS**

```javascript
function showTab(name){
  document.querySelectorAll('#portalTabs .tab').forEach(b =>
    b.classList.toggle('is-active', b.dataset.tab === name));
  document.querySelectorAll('[data-panel]').forEach(p =>
    p.hidden = (p.dataset.panel !== name));
}
```

- [ ] **Step 3: Render Scan History** from `data.scan_dates`, grouped by household member when `data.household` is present; badge `data.current_scan_date`; each row reloads via `?scan_date=` (and `?member=` for a member), reusing the existing scan-select reload path (`selectScan()` / the `?scan_date=` request builder). Show the existing blur/unlock treatment for locked scans unchanged.

- [ ] **Step 4: Render Orders & Invoices** by moving the existing "History & receipts" rendering (`invoices` = pay cards, `past_invoices` = receipts, each linking `/invoice/<token>`) into the orders panel. Empty states: "No past analyses yet" / "No invoices yet."

- [ ] **Step 5: Client toggles** in the Current panel footer, POSTing to `/api/portal/<token>/scan-prefs`:

```javascript
async function setAutoAdvance(on){
  await fetch(`/api/portal/${PORTAL_TOKEN}/scan-prefs`,
    {method:'POST', headers:{'Content-Type':'application/json'},
     body: JSON.stringify({auto_advance: on})});
}
```
"Email me when a new analysis is ready" sends `{notify:'in'|'out'}` to the same endpoint. Copy: no em dashes, no ALL CAPS.

- [ ] **Step 6: Render-verify (the gate).** Seed a local portal with two scans (Task 2's `_seed`), run the app locally with `PORTAL_SCAN_HISTORY_ENABLED=1`, and drive it in headless Chrome:
  - Current tab shows the current scan; switching tabs shows history/orders without a page reload.
  - History lists both dates, current badged; clicking the older one loads it.
  - Family portal (seed a `household.can_view` member with their own scans) shows a labeled group per subject and never a non-member's scan.
  - Toggling auto-advance off then publishing a newer scan leaves the current view pinned (ties to Task 5).
  - `PORTAL_SCAN_HISTORY_ENABLED` unset ⇒ page renders exactly as before (no tab bar).

- [ ] **Step 7: Commit**

```bash
git add static/client-portal.html
git commit -m "feat(portal): three-tab UI (current/history/orders) + family grouping + toggles"
```

---

### Task 8: Backfill script

**Files:**
- Create: `scripts/backfill_portal_scan_history.py`
- Test: `tests/test_backfill_portal_scan_history.py` (create)

**Interfaces:**
- Consumes: `portal_biofield_reports`, `client_portal`.
- Produces: `backfill(cx) -> dict` (counts). Idempotent: sets `auto_advance` default only where absent, `current_scan_date` to newest only where absent; never resets an opted-out pointer.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_backfill_portal_scan_history.py
import sqlite3, importlib.util, sys
from pathlib import Path
from dashboard import client_portal as cp, portal_biofield_reports as pbr

def _load():
    p = Path(__file__).resolve().parent.parent / "scripts" / "backfill_portal_scan_history.py"
    spec = importlib.util.spec_from_file_location("bf", p); m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m); return m

def test_backfill_sets_defaults_but_preserves_optout(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "chat_log.db"))
    cp.init_client_portal_table(cx); pbr.init_table(cx)
    cp.upsert_portal(cx, "a@x.com", "A", {})                       # fresh
    cp.upsert_portal(cx, "b@x.com", "B", {"auto_advance": False, "current_scan_date": "2026-07-02"})
    for e in ("a@x.com", "b@x.com"):
        pbr.upsert_report(cx, e, "2026-07-02", "1", {}, "confirmed")
        pbr.upsert_report(cx, e, "2026-07-09", "2", {}, "confirmed")
    bf = _load()
    bf.backfill(cx)
    assert cp.get_current_scan(cx, "a@x.com") == "2026-07-09"      # newest filled in
    assert cp.get_auto_advance(cx, "a@x.com") is True
    assert cp.get_current_scan(cx, "b@x.com") == "2026-07-02"      # opt-out preserved
    # idempotent: second run changes nothing
    bf.backfill(cx)
    assert cp.get_current_scan(cx, "b@x.com") == "2026-07-02"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_backfill_portal_scan_history.py -v`
Expected: FAIL — script file does not exist.

- [ ] **Step 3: Write the script**

```python
# scripts/backfill_portal_scan_history.py
"""Backfill: give every portal a current_scan_date (newest) + auto_advance default,
without disturbing clients who already pinned/opted out. Idempotent."""
import sqlite3, sys
from dashboard import client_portal as cp, portal_biofield_reports as pbr

def backfill(cx) -> dict:
    cp.init_client_portal_table(cx); pbr.init_table(cx)
    emails = [r[0] for r in cx.execute("SELECT DISTINCT email FROM client_portals WHERE email IS NOT NULL")]
    filled = 0
    for email in emails:
        content = cp._read_content(cx, email) or {}
        changed = False
        if "auto_advance" not in content:
            content["auto_advance"] = True; changed = True
        if not content.get("current_scan_date"):
            dates = pbr.list_report_dates(cx, email)
            if dates:
                content["current_scan_date"] = dates[0]; changed = True
        if changed:
            cp._write_content(cx, email, content); filled += 1
    return {"portals": len(emails), "updated": filled}

if __name__ == "__main__":
    import os
    with sqlite3.connect(os.environ.get("LOG_DB", "chat_log.db")) as cx:
        print(backfill(cx))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_backfill_portal_scan_history.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/backfill_portal_scan_history.py tests/test_backfill_portal_scan_history.py
git commit -m "feat(portal): idempotent scan-history backfill preserving opt-outs"
```

---

### Task 9: Phase 0 runbook — Sasha portal repoint (operational, no code)

**Files:**
- Create: `docs/runbooks/2026-07-11-sasha-portal-repoint.md`

This is the immediate ask that motivated the feature. It runs against **prod data**, gated on the identity question. No app code changes; it is a checklist that uses Task 4's console set-current once the feature branch is deployed (or a direct pointer write if repointing before merge).

- [ ] **Step 1: Resolve identity.** Determine which email the portal token `HwyyJEum…` resolves to, and which `(email, scan_date)` the newer infoceutical scan (scan_id 1035975) lives under in prod `portal_biofield_reports` — cat login `permanentlyyours777@hawaiiantel.net` vs Karin's mailbox `permanentlyyours@hawaii.rr.com`. If the subjects differ, the portal reaches the scan via household `?member=` rather than a same-email pointer.
- [ ] **Step 2: Ensure ingested.** If that scan has no prod `portal_biofield_reports` row, push it first (E4L scan push scripts — `e4l-scan-manifest-push.py` + `e4l-scan-recommendations-push.py`, doppler prd), watching for the deploy-window 502.
- [ ] **Step 3: Repoint.** Set `current_scan_date` to that scan's date via `POST /api/console/portal/set-current` (or a guarded direct write) for the resolved email.
- [ ] **Step 4: Render-verify** the live portal in headless Chrome (payload correctness is not enough — see memory `render-the-page-not-the-payload`): the current view shows the infoceutical scan, history lists both.
- [ ] **Step 5: Notify only later.** Do NOT auto-send. Karin's "new analysis" email is a separate confirm-to-send once `PORTAL_SCAN_NOTIFY_ENABLED` + the bulk channel are live.

- [ ] **Step 6: Commit the runbook**

```bash
git add docs/runbooks/2026-07-11-sasha-portal-repoint.md
git commit -m "docs: Sasha portal repoint runbook (Phase 0)"
```

---

## Rollout sequence

1. Tasks 1-6 (backend) behind `PORTAL_SCAN_HISTORY_ENABLED` (off) — safe to merge.
2. Task 7 (UI) — merge; still dark.
3. Task 8 backfill — run in prd once, before flipping the flag.
4. Flip `PORTAL_SCAN_HISTORY_ENABLED=1` in Doppler prd; render-verify a real portal.
5. Task 9 Sasha repoint (can run as soon as Task 4 is deployed).
6. `PORTAL_SCAN_NOTIFY_ENABLED` stays off until `inbox.send_bulk`'s GHL-v2/Mailgun path (`BULK_VIA_GHL`) is confirmed in prd.

## Self-review notes

- **Spec coverage:** data-model prefs (T1), selection + payload + dangling guard (T2), client prefs/pin/notify-opt (T3), console set-current (T4), auto-advance on ingest (T5), confirm-to-send notify (T6), three tabs + family grouping + invoices (T7), backfill (T8), Phase 0 (T9). All spec sections mapped.
- **Divergences from spec, intentional:** (a) client `notify_opt_in` reuses the existing `notify_state`/`notify-pref` system instead of a new key; (b) "Mailgun channel" is realized as `inbox.send_bulk` (GHL-v2/Mailgun domain) because no direct Mailgun client exists.
- **Verify-before-finalize flags for the implementer:** exact console-auth mechanism in `_portal_console_ok` (Task 4 test header), `notify_state` getter name and `send_bulk` signature (Task 6), and `portal_link_for` returning the canonical myhealingoasis URL.
