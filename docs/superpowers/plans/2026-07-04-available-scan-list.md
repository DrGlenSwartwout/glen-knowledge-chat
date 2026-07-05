# Available-Scan List (Sub-project A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surface every client's E4L scan dates in their portal as a read-only "Scan history" (processed vs. available), by syncing scan-date manifests from the local `e4l.db` to a prod table the portal reads.

**Architecture:** A local script reads `e4l_scans ⋈ e4l_clients` from `~/AI-Training/e4l.db` and POSTs per-client manifests to a console endpoint that upserts a new `client_scans` table in `LOG_DB`. `api_client_portal` reads that table and annotates each date `processed` from `portal_biofield_reports`. Behind `SCAN_LIST_ENABLED`.

**Tech Stack:** Python 3, Flask, SQLite (`LOG_DB` on prod; read-only `e4l.db` locally), urllib, pytest, vanilla JS.

## Global Constraints

- **`e4l.db` is local-only** (`~/AI-Training/e4l.db`); prod can't read it. The portal reads the synced `client_scans` table, NEVER `e4l.db`.
- **`e4l_scans` is keyed by `client_id`**, not email — resolve email via `e4l_clients` (`JOIN ... ON s.client_id=cl.client_id WHERE cl.email!=''`). `scan_id` is an integer; store as TEXT.
- **Read-only surfacing (A):** the list has NO request action and NO rate gate (that's sub-project B). Unprocessed rows are informational.
- **Behind `SCAN_LIST_ENABLED` (default OFF):** no `available_scans` payload key, no Scan history section → portal byte-identical. The sync endpoint is NOT behind this flag (it populates data regardless).
- **Emails lowercased/stripped.** Sync endpoint `_portal_console_ok()`-gated. Best-effort payload — a `client_scans` failure omits the section, never breaks the portal.
- **Task 4 (local sync script) edits the VAULT `~/AI-Training/02 Skills/` DIRECTLY, not the deploy-chat worktree** (the vault is exempt from worktree isolation; it has its own hourly snapshot). Tasks 1-3 edit the worktree `/tmp/wt-scanlist`.
- Deploy-chat tests via `doppler run -p remedy-match -c dev -- python3 -m pytest ...` (use `python3`). Do NOT `git stash`.

---

## File Structure

- **Create** `dashboard/client_scans.py` — `client_scans` table + `upsert_scans`/`scans_for`.
- **Modify** `app.py` — `_scan_list_enabled()`; `POST /api/console/client-scans/sync`; `available_scans` in `api_client_portal` payload (behind flag).
- **Modify** `static/client-portal.html` — the Scan history section.
- **Create** `~/AI-Training/02 Skills/e4l-scan-manifest-push.py` — local sync (reads `e4l.db`, POSTs to prod). Vault, not worktree.
- **Test** `tests/test_client_scans.py` (deploy-chat).

---

### Task 1: `dashboard/client_scans.py` — table + upsert/list

**Files:** Create `dashboard/client_scans.py`; Test `tests/test_client_scans.py`

**Interfaces:**
- Produces: `init_client_scans_table(cx)`; `upsert_scans(cx, email, scans) -> int` where `scans=[{"scan_date","scan_id"}]` (idempotent per `(email, scan_date)`, updates scan_id/synced_at); `scans_for(cx, email) -> [{"scan_date","scan_id"}]` (most-recent first).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_client_scans.py
import sqlite3
from dashboard import client_scans as cs


def _cx():
    cx = sqlite3.connect(":memory:"); cs.init_client_scans_table(cx); return cx


def test_upsert_and_list():
    cx = _cx()
    n = cs.upsert_scans(cx, "Karin@X.com", [{"scan_date": "2026-06-28", "scan_id": 1037676},
                                            {"scan_date": "2026-06-25", "scan_id": 1037001}])
    assert n == 2
    got = cs.scans_for(cx, "karin@x.com")
    assert [g["scan_date"] for g in got] == ["2026-06-28", "2026-06-25"]   # most-recent first
    assert got[0]["scan_id"] == "1037676"                                  # stored as str


def test_upsert_idempotent():
    cx = _cx()
    cs.upsert_scans(cx, "k@x.com", [{"scan_date": "2026-06-28", "scan_id": 1}])
    cs.upsert_scans(cx, "k@x.com", [{"scan_date": "2026-06-28", "scan_id": 2}])   # same date, new id
    got = cs.scans_for(cx, "k@x.com")
    assert len(got) == 1 and got[0]["scan_id"] == "2"                     # no dup; scan_id updated


def test_blank_email_and_date_skipped():
    cx = _cx()
    assert cs.upsert_scans(cx, "", [{"scan_date": "2026-06-28"}]) == 0
    assert cs.upsert_scans(cx, "k@x.com", [{"scan_date": ""}]) == 0
    assert cs.scans_for(cx, "k@x.com") == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_client_scans.py -q`
Expected: FAIL (`ModuleNotFoundError: dashboard.client_scans`).

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/client_scans.py
"""Synced manifest of every client's E4L scan dates, for the portal Scan-history list.
Populated from the local e4l.db by the e4l-scan-manifest-push sync (prod can't read e4l.db).
One row per (email, scan_date). LOG_DB (SQLite)."""
import datetime


def _now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _norm(e):
    return (e or "").strip().lower()


def init_client_scans_table(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS client_scans (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            email      TEXT NOT NULL,
            scan_date  TEXT NOT NULL,
            scan_id    TEXT,
            synced_at  TEXT,
            UNIQUE(email, scan_date)
        )
    """)
    cx.execute("CREATE INDEX IF NOT EXISTS ix_cs_email ON client_scans(email)")
    cx.commit()


def upsert_scans(cx, email, scans):
    e = _norm(email)
    if not e:
        return 0
    n = 0
    for s in scans or []:
        d = (s.get("scan_date") or "").strip()
        if not d:
            continue
        sid = str(s.get("scan_id") or "")
        cur = cx.execute("UPDATE client_scans SET scan_id=?, synced_at=? WHERE email=? AND scan_date=?",
                         (sid, _now(), e, d))
        if cur.rowcount == 0:
            cx.execute("INSERT OR IGNORE INTO client_scans (email, scan_date, scan_id, synced_at) "
                       "VALUES (?,?,?,?)", (e, d, sid, _now()))
        n += 1
    cx.commit()
    return n


def scans_for(cx, email):
    rows = cx.execute(
        "SELECT scan_date, scan_id FROM client_scans WHERE email=? ORDER BY scan_date DESC, id DESC",
        (_norm(email),)).fetchall()
    return [{"scan_date": r[0], "scan_id": r[1] or ""} for r in rows]
```

- [ ] **Step 4: Run to verify it passes** — GREEN.

- [ ] **Step 5: Commit**

```bash
git add dashboard/client_scans.py tests/test_client_scans.py
git commit -m "feat(scan-list): client_scans table + upsert/list"
```

---

### Task 2: sync endpoint + `available_scans` payload

**Files:** Modify `app.py`; Test `tests/test_client_scans.py`

**Interfaces:**
- Consumes: `client_scans.{init_client_scans_table,upsert_scans,scans_for}` (Task 1); `_portal_console_ok()`; `_pbr.list_report_dates(cx, email)` (published report dates).
- Produces: `_scan_list_enabled()`; `POST /api/console/client-scans/sync`; `api_client_portal` payload gains `available_scans` (behind flag).

**Context:** `api_client_portal` returns at `return jsonify(payload)` (~app.py:14591, right after the read-receipts `opens` block). The published report dates for `email_for_reports` are computed at ~app.py:14452 (`dates = _pbr.list_report_dates(cx_r, email_for_reports)`; also surfaced as `bf_scan_dates`). Reuse that list as the `processed` set — read the function to get the exact var name.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_client_scans.py
import importlib, sys
from pathlib import Path
import pytest


def _app(tmp_path, monkeypatch, *, flag="1"):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SCAN_LIST_ENABLED", flag)
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path: sys.path.insert(0, str(repo))
    try:
        import app as appmod; importlib.reload(appmod)
        import dashboard as _d; monkeypatch.setattr(_d, "CONSOLE_SECRET", "", raising=False)
    except Exception as e:
        pytest.skip(f"app not importable: {e}")
    return appmod


def test_sync_endpoint_upserts(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    c = appmod.app.test_client()
    r = c.post("/api/console/client-scans/sync",
               json={"email": "k@x.com", "scans": [{"scan_date": "2026-06-28", "scan_id": 5}]})
    assert r.status_code == 200 and r.get_json()["ok"] is True
    import sqlite3
    from dashboard import client_scans as cs
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert [s["scan_date"] for s in cs.scans_for(cx, "k@x.com")] == ["2026-06-28"]
    # batch form
    r2 = c.post("/api/console/client-scans/sync",
                json={"batch": [{"email": "a@x.com", "scans": [{"scan_date": "2026-07-01"}]}]})
    assert r2.status_code == 200


def test_available_scans_payload(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    from dashboard import client_portal as cp, client_scans as cs, portal_biofield_reports as pbr
    import sqlite3
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cp.init_client_portal_table(cx); cs.init_client_scans_table(cx); pbr.init_table(cx)
        cs.upsert_scans(cx, "k@x.com", [{"scan_date": "2026-06-28"}, {"scan_date": "2026-06-25"}])
        pbr.upsert_report(cx, "k@x.com", "2026-06-25", "s1", {"n": 1}, "confirmed")   # 06-25 processed
        tok = cp.upsert_portal(cx, "k@x.com", "K", {}); cx.commit()
    token = tok[0] if isinstance(tok, (tuple, list)) else tok
    if not token: pytest.skip("no mint helper")
    j = appmod.app.test_client().get(f"/api/portal/{token}").get_json()
    av = {s["scan_date"]: s["processed"] for s in j.get("available_scans", [])}
    assert av == {"2026-06-28": False, "2026-06-25": True}


def test_scan_list_flag_off(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch, flag="0")
    from dashboard import client_portal as cp
    import sqlite3
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cp.init_client_portal_table(cx); tok = cp.upsert_portal(cx, "k@x.com", "K", {}); cx.commit()
    token = tok[0] if isinstance(tok, (tuple, list)) else tok
    if not token: pytest.skip("no mint helper")
    j = appmod.app.test_client().get(f"/api/portal/{token}").get_json()
    assert "available_scans" not in j
```

- [ ] **Step 2: Run to verify it fails** — FAIL (404 sync route / no `available_scans`).

- [ ] **Step 3: Write minimal implementation**

Flag helper near `_read_receipts_enabled` (app.py):

```python
def _scan_list_enabled():
    return (os.environ.get("SCAN_LIST_ENABLED", "") or "").strip().lower() in ("1", "true", "yes")
```

Sync endpoint (near other `/api/console/*`):

```python
@app.route("/api/console/client-scans/sync", methods=["POST"])
def api_console_client_scans_sync():
    """Owner sync: upsert a client's (or a batch of clients') E4L scan-date manifest into
    client_scans (populated by the local e4l-scan-manifest-push, since prod can't read e4l.db)."""
    if not _portal_console_ok():
        return jsonify({"error": "unauthorized"}), 401
    from dashboard import client_scans as _cs
    data = request.get_json(silent=True) or {}
    items = data.get("batch") or [{"email": data.get("email"), "scans": data.get("scans")}]
    total = 0
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        _cs.init_client_scans_table(cx)
        for it in items:
            total += _cs.upsert_scans(cx, it.get("email"), it.get("scans") or [])
    return jsonify({"ok": True, "upserted": total})
```

Payload — immediately before `return jsonify(payload)` (~app.py:14591):

```python
    if _scan_list_enabled():
        try:
            from dashboard import client_scans as _cs
            with sqlite3.connect(LOG_DB) as _cxs:
                _cs.init_client_scans_table(_cxs)
                _synced = _cs.scans_for(_cxs, email_for_reports)
            _processed = set(bf_scan_dates or [])   # published report dates for this email
            payload["available_scans"] = [
                {"scan_date": s["scan_date"], "scan_id": s["scan_id"],
                 "processed": s["scan_date"] in _processed} for s in _synced]
        except Exception as _e:
            print(f"[scan-list] {_e!r}", flush=True)
```

> Implementer note: confirm `bf_scan_dates` is the in-scope list of published report dates for `email_for_reports` at that point (it is — set from `_pbr.list_report_dates`); if the var name differs, recompute `_processed = set(_pbr.list_report_dates(_cxs, email_for_reports))`.

- [ ] **Step 4: Run to verify it passes** — GREEN.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_client_scans.py
git commit -m "feat(scan-list): sync endpoint + available_scans payload (flag-gated)"
```

---

### Task 3: Scan history section in `static/client-portal.html`

**Files:** Modify `static/client-portal.html`

**Interfaces:**
- Consumes: `d.available_scans` (`[{scan_date, scan_id, processed}]` or absent).

UI-only (no pytest). Absent (flag off) → render nothing.

- [ ] **Step 1: Add the Scan history section**

In `render(d, v)`, when `d.available_scans && d.available_scans.length`, render a "Scan history" card listing every date, newest first. A `processed` date shows with an "Analyzed" tag (the existing report is reachable via the report card / scan-date selector already on the page — do NOT duplicate the report body here; just mark it analyzed). An unprocessed date shows the date + a muted "Available — not yet analyzed" label (no action in A). Escape everything with `esc()`.

```javascript
  // Scan history — every E4L scan date (A: read-only; request action is sub-project B).
  if (d.available_scans && d.available_scans.length) {
    let rows = "";
    for (const s of d.available_scans) {
      const tag = s.processed
        ? `<span style="color:#2f6f5e;font-size:.8rem">Analyzed</span>`
        : `<span style="color:var(--muted);font-size:.8rem">Available — not yet analyzed</span>`;
      rows += `<div style="display:flex;align-items:center;gap:10px;padding:6px 0;border-bottom:1px solid var(--border)">
        <strong style="font-size:.95rem">${esc(s.scan_date)}</strong><span style="flex:1"></span>${tag}</div>`;
    }
    html += `<div class="card"><h2 style="font-size:1rem">Scan history</h2>
      <p style="color:var(--muted);font-size:.85rem;margin:.2rem 0 .6rem">Every scan on file. Analyzed scans have a full report; others are available to analyze.</p>
      ${rows}</div>`;
  }
```

- [ ] **Step 2: Verify (static)** — extract the inline `<script>`, `node --check`; grep-confirm it reads `d.available_scans`, escapes `scan_date`, renders nothing when the key is absent, and adds no action button (A is read-only). Report live render-verify pending.

- [ ] **Step 3: Commit**

```bash
git add static/client-portal.html
git commit -m "feat(scan-list): Scan history section (read-only, processed vs available)"
```

---

### Task 4: local sync script `e4l-scan-manifest-push.py` (VAULT, not worktree)

**Files:** Create `~/AI-Training/02 Skills/e4l-scan-manifest-push.py` — **edit the vault directly; this file is NOT in the deploy-chat worktree.** (No deploy-chat commit for this file; the vault has its own hourly snapshot.)

**Interfaces:**
- Consumes: `~/AI-Training/e4l.db` (read-only); `POST /api/console/client-scans/sync` (Task 2).
- Produces: a runnable script + a unit-testable `build_manifests(e4l_db_path) -> {email: [{scan_date, scan_id}]}`.

- [ ] **Step 1: Write the script**

Mirror `~/AI-Training/02 Skills/console-push.py` (urllib POST to `https://glen-knowledge-chat.onrender.com`, `CONSOLE_SECRET` from Doppler `remedy-match/prd`, `X-Console-Key` header). Read `e4l.db` read-only, build per-email manifests, POST in batches.

```python
#!/usr/bin/env python3
"""Push every client's E4L scan-date manifest from the local e4l.db to prod client_scans
(prod can't read e4l.db). Idempotent. Run manually for backfill; piggyback the e4l ingest cron.
Auth: CONSOLE_SECRET from Doppler remedy-match/prd. Usage: [--dry] [--db PATH]."""
import argparse, json, os, sqlite3, sys, urllib.request, urllib.error

BASE = os.environ.get("WEB_URL", "https://glen-knowledge-chat.onrender.com").rstrip("/")
DEFAULT_DB = os.path.expanduser("~/AI-Training/e4l.db")


def build_manifests(e4l_db_path):
    """{lower(email): [{scan_date, scan_id}]} from e4l_scans JOIN e4l_clients."""
    cx = sqlite3.connect(f"file:{e4l_db_path}?mode=ro", uri=True)
    try:
        rows = cx.execute(
            "SELECT lower(trim(cl.email)) AS email, s.scan_date, s.scan_id "
            "FROM e4l_scans s JOIN e4l_clients cl ON s.client_id=cl.client_id "
            "WHERE cl.email IS NOT NULL AND trim(cl.email) != '' AND s.scan_date IS NOT NULL "
            "ORDER BY email, s.scan_date DESC").fetchall()
    finally:
        cx.close()
    out = {}
    for email, scan_date, scan_id in rows:
        out.setdefault(email, []).append({"scan_date": scan_date, "scan_id": str(scan_id or "")})
    return out


def _post_batch(secret, batch):
    req = urllib.request.Request(
        f"{BASE}/api/console/client-scans/sync", method="POST",
        data=json.dumps({"batch": batch}).encode(),
        headers={"Content-Type": "application/json", "X-Console-Key": secret})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--chunk", type=int, default=200)
    args = ap.parse_args()
    manifests = build_manifests(args.db)
    total = sum(len(v) for v in manifests.values())
    print(f"clients={len(manifests)} scans={total}")
    if args.dry:
        for e, s in list(manifests.items())[:5]:
            print(f"  {e}: {len(s)} scans (latest {s[0]['scan_date'] if s else '-'})")
        return
    secret = os.environ.get("CONSOLE_SECRET", "")
    if not secret:
        print("CONSOLE_SECRET not set (run via `doppler run -p remedy-match -c prd -- python3 ...`)", file=sys.stderr)
        sys.exit(2)
    items = [{"email": e, "scans": s} for e, s in manifests.items()]
    up = 0
    for i in range(0, len(items), args.chunk):
        res = _post_batch(secret, items[i:i + args.chunk])
        up += res.get("upserted", 0)
        print(f"  batch {i//args.chunk + 1}: upserted +{res.get('upserted', 0)}")
    print(f"done: upserted={up}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify** — unit-test `build_manifests` against a temp sqlite (create `e4l_scans`+`e4l_clients`, assert grouping/lowercasing/most-recent-first). Write this test at `~/AI-Training/02 Skills/test_e4l_scan_manifest_push.py` (or a scratch file) and run `python3 -m pytest`. Then a real dry run: `doppler run -p remedy-match -c prd -- python3 "$HOME/AI-Training/02 Skills/e4l-scan-manifest-push.py" --dry` — expect `clients=N scans=~3099`.

```python
# test for build_manifests (vault-local)
import sqlite3
import importlib.util, os
_p = os.path.expanduser("~/AI-Training/02 Skills/e4l-scan-manifest-push.py")
_spec = importlib.util.spec_from_file_location("e4lpush", _p); _m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)

def test_build_manifests(tmp_path):
    db = str(tmp_path / "e4l.db")
    cx = sqlite3.connect(db)
    cx.execute("CREATE TABLE e4l_clients (client_id INTEGER, email TEXT)")
    cx.execute("CREATE TABLE e4l_scans (scan_id INTEGER, client_id INTEGER, scan_date TEXT)")
    cx.execute("INSERT INTO e4l_clients VALUES (1,'Karin@X.com'),(2,'')")
    cx.executemany("INSERT INTO e4l_scans VALUES (?,?,?)",
                   [(10,1,'2026-06-25'),(11,1,'2026-06-28'),(12,2,'2026-06-01')])
    cx.commit(); cx.close()
    m = _m.build_manifests(db)
    assert set(m) == {"karin@x.com"}                       # blank email dropped
    assert [s["scan_date"] for s in m["karin@x.com"]] == ["2026-06-28","2026-06-25"]  # newest first
    assert m["karin@x.com"][0]["scan_id"] == "11"
```

- [ ] **Step 3: (No git commit for the vault file.)** Report the file path, the `build_manifests` test result, and the `--dry` counts in the task report. Wiring the cron trigger (piggyback `e4l-daily-watch.sh`) is a one-line addition the controller confirms with Glen at go-live — do NOT edit the cron in this task.

---

## Self-Review

**Spec coverage:**
- `client_scans` table + `upsert_scans`/`scans_for` → Task 1. ✓
- Sync endpoint (single + batch, console-gated) → Task 2. ✓
- `available_scans` payload annotated `processed` via published dates, flag-gated → Task 2. ✓
- Scan history portal section (processed vs available, read-only, household-switcher via `email_for_reports`) → Task 3. ✓
- Local sync script (e4l_scans⋈e4l_clients, mirrors console-push, dry/backfill) → Task 4 (vault). ✓
- Behind `SCAN_LIST_ENABLED`, flag-off byte-identical → Tasks 2 (payload) & 3 (section). ✓
- Read-only (no request action/rate gate = B) → Task 3 renders no button. ✓

**Placeholder scan:** Task 2 note = confirm the real published-dates var (`bf_scan_dates`), a name-check not a TBD. Task 4 is vault-local (no worktree commit) and its cron wiring is deferred to a Glen-confirmed go-live step, explicitly. No hand-waves.

**Type consistency:** `scans_for -> [{scan_date, scan_id}]` (Task 1) consumed by Task 2's payload, annotated with `processed` → `d.available_scans=[{scan_date,scan_id,processed}]` (Task 3). `upsert_scans(cx, email, scans)` (Task 1) called by the sync endpoint (Task 2) and mirrors the manifest shape `build_manifests` produces (Task 4). `SCAN_LIST_ENABLED` gates Tasks 2 & 3 only; the sync endpoint is ungated.
