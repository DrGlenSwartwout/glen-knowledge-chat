# Biofield Gate — Phase 2b: Scan Freshness Auto-Verify — Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Auto-verify the Biofield readiness gate's "fresh voice scan (≤7 days)" item instead of self-confirm only. The scan data lives in a LOCAL `e4l.db` on Glen's Mac (unreachable from Render), so the local ingestion pushes a compact `email → latest scan date` index to a deployed endpoint; the gate reads it. Self-confirm stays as a fallback.

**Architecture:** Server: a `scan_freshness` sqlite table (persistent disk) + a cron-secret-gated `POST /api/e4l/scan-freshness` ingest endpoint; the Biofield gate gains an injected `has_fresh_scan(email)` (≤7-day lookup) alongside the existing `has_intake`, so the scan item is green if a fresh scan is detected OR the patient self-confirmed. Local: a `02 Skills/push-e4l-scan-freshness.py` reads `email → MAX(scan_date)` from `e4l.db` and POSTs it after each ingestion run. No Supabase, no E4L API.

**Tech Stack:** Python 3.11, Flask, sqlite, pytest; a local Python push script.

**Spec:** `docs/superpowers/specs/2026-06-15-biofield-checkout-readiness-gate-design.md` (the 2a→2b note: "auto-verify becomes 2b once scan-freshness is mirrored to the server"). See [[project_e4l_scan_ingestion]] (the local cron + DB). Scope = scan only; intake already auto-detects (`inbound_leads`), photo is upload, payment is checkout/PB-receipt. A Practice Better API stays further-deferred.

**Reuse:** the console-push local→server pattern (`02 Skills/console-push.py` — cron-secret POST to the deployed app); the cron-secret auth (`X-Cron-Secret` == `CRON_SECRET`/`CONSOLE_SECRET`); `dashboard/biofield_gate.gate_state` (add `has_fresh_scan`); `_biofield_has_intake` route wiring (app.py ~6934); `e4l.db` join `e4l_clients.email` ⋈ `e4l_scans.scan_date`.

**Test invocation:** pure → `~/.venvs/deploy-chat311/bin/python -m pytest <path> -q`. App → `doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" ~/.venvs/deploy-chat311/bin/python -m pytest <path> -q` (worktree; ignore the 2 known pre-existing failures).

---

### Task 1: `scan_freshness` store + gate_state `has_fresh_scan`

**Files:** Create `dashboard/scan_freshness.py`; Modify `dashboard/biofield_gate.py`; Test `tests/test_scan_freshness.py`, `tests/test_biofield_gate.py` (append)

- [ ] **Step 1: Failing tests**

`tests/test_scan_freshness.py`:
```python
import sqlite3
from dashboard import scan_freshness as sf

def _cx():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    sf.init_table(cx); return cx

def test_upsert_and_latest():
    cx = _cx()
    sf.upsert(cx, [{"email": "p@x.com", "last_scan_date": "2026-06-10"},
                   {"email": "q@x.com", "last_scan_date": "2026-01-01"}])
    assert sf.latest_scan_date(cx, "P@X.COM") == "2026-06-10"   # case-insensitive
    assert sf.latest_scan_date(cx, "q@x.com") == "2026-01-01"
    assert sf.latest_scan_date(cx, "none@x.com") is None

def test_upsert_keeps_newest():
    cx = _cx()
    sf.upsert(cx, [{"email": "p@x.com", "last_scan_date": "2026-01-01"}])
    sf.upsert(cx, [{"email": "p@x.com", "last_scan_date": "2026-06-10"}])
    assert sf.latest_scan_date(cx, "p@x.com") == "2026-06-10"
    # an older push does not regress a newer stored date
    sf.upsert(cx, [{"email": "p@x.com", "last_scan_date": "2026-03-01"}])
    assert sf.latest_scan_date(cx, "p@x.com") == "2026-06-10"

def test_is_fresh():
    cx = _cx()
    sf.upsert(cx, [{"email": "p@x.com", "last_scan_date": "2026-06-12"}])
    assert sf.is_fresh(cx, "p@x.com", today="2026-06-15", window_days=7) is True
    assert sf.is_fresh(cx, "p@x.com", today="2026-06-25", window_days=7) is False
    assert sf.is_fresh(cx, "none@x.com", today="2026-06-15", window_days=7) is False
```

`tests/test_biofield_gate.py` (append) — scan green via fresh-scan auto, not just self-confirm:
```python
def test_gate_scan_green_via_fresh_scan_auto():
    import sqlite3
    from dashboard import biofield_store as bs, biofield_gate as bg
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    bs.init_table(cx)
    bs.seed_paid(cx, "p@x.com", via="stripe", order_ref="INV1")
    bs.set_photo_on_file(cx, "p@x.com", "x.jpg")
    st = bg.gate_state(cx, "p@x.com", has_intake=lambda e: True, has_fresh_scan=lambda e: True)
    assert st["items"]["scan"]["status"] == "green"   # auto, no self-confirm
    assert st["booking_unlocked"] is True

def test_gate_scan_defaults_self_confirm_only_when_no_fresh_checker():
    import sqlite3
    from dashboard import biofield_store as bs, biofield_gate as bg
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    bs.init_table(cx); bs.seed_paid(cx, "p@x.com", via="stripe", order_ref="INV1")
    st = bg.gate_state(cx, "p@x.com", has_intake=lambda e: True)  # no has_fresh_scan
    assert st["items"]["scan"]["status"] == "needed"
```

- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement.**

`dashboard/scan_freshness.py` (pure):
```python
"""Server-side mirror of e4l scan freshness (email -> latest scan date), pushed from
the local e4l ingestion. Lets the Biofield gate auto-verify a fresh voice scan."""

def init_table(cx):
    cx.execute("""CREATE TABLE IF NOT EXISTS scan_freshness (
        email TEXT PRIMARY KEY, last_scan_date TEXT, updated_at TEXT)""")
    cx.commit()

def _now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def upsert(cx, rows):
    """rows: [{email, last_scan_date}]. Keeps the NEWEST date per email."""
    for r in rows or []:
        email = (r.get("email") or "").strip().lower()
        d = (r.get("last_scan_date") or "").strip()
        if not email or not d:
            continue
        cx.execute("""INSERT INTO scan_freshness (email, last_scan_date, updated_at)
                      VALUES (?,?,?)
                      ON CONFLICT(email) DO UPDATE SET
                        last_scan_date=MAX(scan_freshness.last_scan_date, excluded.last_scan_date),
                        updated_at=excluded.updated_at""",
                   (email, d, _now()))
    cx.commit()

def latest_scan_date(cx, email):
    row = cx.execute("SELECT last_scan_date FROM scan_freshness WHERE email=lower(?)",
                     (str(email or "").strip(),)).fetchone()
    return row[0] if row else None

def is_fresh(cx, email, *, today, window_days=7):
    d = latest_scan_date(cx, email)
    if not d:
        return False
    from datetime import date
    try:
        sd = date.fromisoformat(d); td = date.fromisoformat(today)
    except ValueError:
        return False
    return 0 <= (td - sd).days <= window_days
```
(MAX on ISO `YYYY-MM-DD` strings is correct lexicographically.)

`dashboard/biofield_gate.py` — add `has_fresh_scan=None` param:
```python
def gate_state(cx, email, *, has_intake, has_fresh_scan=None, scan_window_days=7):
    ...
    scan = bool(row.get("scan_confirmed")) or bool(has_fresh_scan(email)) if has_fresh_scan else bool(row.get("scan_confirmed"))
    ...
```
(Write it clearly: `scan = bool(row.get("scan_confirmed")); if has_fresh_scan: scan = scan or bool(has_fresh_scan(email))`. Keep the existing 2 gate tests green — `has_fresh_scan` defaults None.)

- [ ] **Step 4: Run → pass** (new + the existing `test_biofield_gate.py`).
- [ ] **Step 5: Commit** — `feat(scan-freshness): store + gate_state has_fresh_scan`

---

### Task 2: ingest endpoint + wire the gate routes

**Files:** Modify `app.py`; Test `tests/test_scan_freshness_routes.py`

- [ ] **Step 1: Failing test** — `POST /api/e4l/scan-freshness` with `X-Cron-Secret` (wrong/none → 401) and body `{"rows":[{"email":"p@x.com","last_scan_date":"<2 days ago>"}]}` → 200 `{ok, upserted: N}`; `scan_freshness.latest_scan_date` reflects it. Then the Biofield gate: with `BIOFIELD_CHECKOUT_ENABLED=1`, an authed biofield email `p@x.com` who is paid + photo on file (seed via `biofield_store`) + a fresh scan pushed → `GET /api/biofield/ready` shows `scan` green and `booking_unlocked` true WITHOUT a self-confirm. (Reuse the biofield-gate-routes auth harness.)

- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement.**
  - `POST /api/e4l/scan-freshness`: `X-Cron-Secret` auth (== `CRON_SECRET` or `CONSOLE_SECRET`); body `{rows:[...]}`; `scan_freshness.init_table(cx)`; `scan_freshness.upsert(cx, rows)`; return `{ok, upserted: len(rows)}`. (Accept a reasonable batch; cap to e.g. 5000 rows.)
  - Add a `_biofield_has_fresh_scan(email)` helper: open LOG_DB, `scan_freshness.is_fresh(cx, email, today=<today ISO>, window_days=7)` (best-effort, False on error).
  - In the three Biofield gate routes that call `_gate.gate_state(cx, email, has_intake=_biofield_has_intake)` (GET ready, the confirm handler's return, the book recompute), add `has_fresh_scan=_biofield_has_fresh_scan`. Also the `/api/biofield/book` gate recompute (app.py — wherever `gate_state` is recomputed) must include it so a fresh-scan user can book without self-confirming.

- [ ] **Step 4: Run → pass.** Regression: `… -m pytest tests/test_scan_freshness_routes.py tests/test_biofield_gate_routes.py -q`.
- [ ] **Step 5: Commit** — `feat(scan-freshness): ingest endpoint + gate auto-verify wiring`

---

### Task 3: local push script

**Files:** Create `02 Skills/push-e4l-scan-freshness.py` (this is in the VAULT `~/AI-Training/02 Skills/`, NOT the deploy-chat repo)

> NOTE: this file lives in the vault, not the worktree. Create it at `/Users/remedymatch/AI-Training/02 Skills/push-e4l-scan-freshness.py`. The vault auto-snapshots; do NOT git-commit it in the deploy-chat repo.

- [ ] **Step 1:** Write `push-e4l-scan-freshness.py`:
  - Reads `~/AI-Training/e4l.db`: `SELECT lower(c.email), MAX(s.scan_date) FROM e4l_clients c JOIN e4l_scans s ON s.client_id=c.client_id WHERE c.email IS NOT NULL AND c.email!='' GROUP BY lower(c.email)`.
  - Builds `rows=[{email, last_scan_date}]`, batches (e.g. 1000), and POSTs each batch to `{PUBLIC_BASE_URL}/api/e4l/scan-freshness` with header `X-Cron-Secret: <CRON_SECRET or CONSOLE_SECRET>` (read from env via doppler, like the other push scripts).
  - `--dry-run` prints counts without POSTing. Robust: skip malformed dates; log a summary line.
  - Mirror `02 Skills/console-push.py`'s structure (env, PUBLIC_BASE_URL/illtowell.com, requests, secret header).
- [ ] **Step 2:** Verify dry-run locally: `~/.venvs/deploy-chat311/bin/python "02 Skills/push-e4l-scan-freshness.py" --dry-run` (or plain python3) — prints a count (~hundreds of emails), no POST. Confirm it reads e4l.db without error.
- [ ] **Step 3:** (No deploy-chat commit — vault file. Note in the doc that it should be wired into the e4l ingestion: add a line to `02 Skills/e4l-daily-watch.sh` (after vectorize) and/or a small launchd to run it after each ingestion, once the endpoint is live.) Report the dry-run count.

---

### Task 4: doc + suite

**Files:** Modify `docs/biofield-gate.md`

- [ ] **Step 1:** Append a "Phase 2b — scan auto-verify" section to `docs/biofield-gate.md`: the local `e4l.db` → `push-e4l-scan-freshness.py` → `POST /api/e4l/scan-freshness` → `scan_freshness` table → gate `has_fresh_scan` (≤7d) flow; self-confirm remains a fallback; wire the push into the e4l ingestion cron once the endpoint is live; intake/photo/payment unchanged; a full Practice Better API is still deferred.
- [ ] **Step 2:** Suite green: `… -m pytest tests/test_scan_freshness.py tests/test_scan_freshness_routes.py tests/test_biofield_gate.py tests/test_biofield_gate_routes.py -q`.
- [ ] **Step 3:** Commit (deploy-chat) — `docs(scan-freshness): Biofield scan auto-verify (Phase 2b)`

---

## Self-review
- **Spec coverage:** scan auto-verify via a server-side freshness mirror pushed from the local ingestion (Task 1 store, Task 2 endpoint + wiring, Task 3 push); hybrid (auto OR self-confirm) preserved; ≤7-day window.
- **Type consistency:** `scan_freshness` (init_table/upsert/latest_scan_date/is_fresh); `gate_state(..., has_fresh_scan=None)`; endpoint `/api/e4l/scan-freshness`; helper `_biofield_has_fresh_scan`.
- **Deferred:** Practice Better intake/photo/payment API; auto-photo verification; pushing freshness more granularly than per-email-latest.
- **Risk:** low — read-only freshness data, no money/PHI in the index (just email + a date); cron-secret gated; gate stays hybrid so a missing/ stale index just falls back to self-confirm; MAX keeps the newest date.

## Done
The Biofield gate auto-verifies a fresh (≤7-day) voice scan from a server-side index the local e4l ingestion pushes, with self-confirm as a fallback.
