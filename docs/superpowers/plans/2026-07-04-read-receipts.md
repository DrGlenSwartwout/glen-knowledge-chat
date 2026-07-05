# Read Receipts for Reports & Invoices (v1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A real read-receipt: reports and invoices render collapsed with a "New" badge until the client deliberately clicks to expand, and that click (a POST) records a true "opened" event surfaced to the owner — bots and owner previews excluded.

**Architecture:** A new `dashboard/opens.py` owns a `portal_opens` table + pure record/get functions with a 5-second debounce. Two token-scoped POST endpoints record an open on expand-click, skipping owner previews and (for reports) resolving `?member=` the same way `api_client_portal` does. The portal/invoice payloads carry open-status so the client renders New-vs-date; the reveals console and orders board surface it to the owner. All client-facing render is behind `READ_RECEIPTS_ENABLED`.

**Tech Stack:** Python 3, Flask, SQLite (`LOG_DB`), pytest, vanilla JS.

## Global Constraints

- **Real read-receipt:** an open is recorded ONLY by an explicit expand-click issuing a POST to an `/open` endpoint. No GET/page-load ever records (excludes link-prefetching scanners). Copy this property forward to every task.
- **False-open filtering:** the `/open` endpoints return 200 but do NOT record when the caller is the owner (`_portal_console_ok()` true). A re-open within **5 seconds** of `last_opened` updates `last_opened` but does NOT increment `open_count`. `first_opened` is set once, never changes.
- **Legit opens that DO count:** the client on their own token, and a household caregiver expanding a linked member's report (`?member=`).
- **Header badge:** the collapsed header ALWAYS shows the item's generation date (report = its `scan_date`; invoice = the date it was generated) + a **New** badge until the first open, then just the date.
- **Behind `READ_RECEIPTS_ENABLED` (default OFF):** flag-off = today's behavior (content shown immediately, no `opens`/`opened` payload keys, `/open` inert). Flag gates the client-facing render + payload; owner-side surfacing just shows whatever data exists.
- All emails lowercased/stripped; console endpoints `_portal_console_ok()`-gated.
- **Reassign resets to New:** when a scan is reassigned among a household, the old report's open record is cleared so the new member sees it as New.
- Tests import `app` → run via `doppler run -p remedy-match -c dev -- python3 -m pytest ...` (use `python3`). Do NOT `git stash` (shared repo).

---

## File Structure

- **Create** `dashboard/opens.py` — `portal_opens` table + `init_opens_table`/`record_open`/`get_open`/`opens_for`/`clear_open` + `report_key`/`invoice_key`.
- **Modify** `app.py` — `_read_receipts_enabled()` helper; `POST /api/portal/<token>/open`; `POST /api/invoice/<token>/open`; `opens` in `api_client_portal` payload; `opened` in `api_invoice_get`; clear-open in the reassign endpoint; open-status join in reveals + orders console payloads.
- **Modify** `static/client-portal.html` — collapsed report card + expand-tracks + selector New badges.
- **Modify** `static/invoice.html` — collapsed invoice + expand-tracks.
- **Test** `tests/test_opens.py`, `tests/test_open_routes.py`.

---

### Task 1: `dashboard/opens.py` — table + record/get with debounce

**Files:** Create `dashboard/opens.py`; Test `tests/test_opens.py`

**Interfaces:**
- Produces: `init_opens_table(cx)`; `report_key(email, scan_date) -> str`; `invoice_key(token) -> str`; `record_open(cx, kind, key, *, now=None) -> {first_opened,last_opened,open_count}`; `get_open(cx, kind, key) -> {...}|None`; `opens_for(cx, kind, keys) -> {key: {...}}`; `clear_open(cx, kind, key) -> None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_opens.py
import sqlite3
from dashboard import opens as o


def _cx():
    cx = sqlite3.connect(":memory:"); o.init_opens_table(cx); return cx


def test_keys():
    assert o.report_key(" Karin@X.com ", "2026-06-25") == "karin@x.com|2026-06-25"
    assert o.invoice_key(" tok123 ") == "tok123"


def test_first_open_sets_all():
    cx = _cx()
    r = o.record_open(cx, "report", "k", now="2026-07-04 10:00:00")
    assert r == {"first_opened": "2026-07-04 10:00:00", "last_opened": "2026-07-04 10:00:00", "open_count": 1}


def test_reopen_after_window_bumps_count():
    cx = _cx()
    o.record_open(cx, "report", "k", now="2026-07-04 10:00:00")
    r = o.record_open(cx, "report", "k", now="2026-07-04 10:00:10")   # +10s
    assert r["open_count"] == 2 and r["last_opened"] == "2026-07-04 10:00:10"
    assert r["first_opened"] == "2026-07-04 10:00:00"                 # unchanged


def test_reopen_within_debounce_does_not_bump():
    cx = _cx()
    o.record_open(cx, "report", "k", now="2026-07-04 10:00:00")
    r = o.record_open(cx, "report", "k", now="2026-07-04 10:00:03")   # +3s < 5s
    assert r["open_count"] == 1 and r["last_opened"] == "2026-07-04 10:00:03"


def test_get_and_opens_for_and_clear():
    cx = _cx()
    o.record_open(cx, "invoice", "t1", now="2026-07-04 10:00:00")
    assert o.get_open(cx, "invoice", "t1")["open_count"] == 1
    assert o.get_open(cx, "invoice", "missing") is None
    assert set(o.opens_for(cx, "invoice", ["t1", "missing"]).keys()) == {"t1"}
    o.clear_open(cx, "invoice", "t1")
    assert o.get_open(cx, "invoice", "t1") is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_opens.py -q`
Expected: FAIL (`ModuleNotFoundError: dashboard.opens`).

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/opens.py
"""Read-receipt opens: record when a client explicitly opens (expand-clicks) a
report or an invoice. One row per (kind, key). LOG_DB (SQLite). The 5s debounce
keeps a double-click from inflating open_count; first_opened is set once."""
import datetime

_DEBOUNCE_SECONDS = 5
_FMT = "%Y-%m-%d %H:%M:%S"


def _now():
    return datetime.datetime.now(datetime.timezone.utc).strftime(_FMT)


def report_key(email, scan_date):
    return f"{(email or '').strip().lower()}|{(scan_date or '').strip()}"


def invoice_key(token):
    return (token or "").strip()


def init_opens_table(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS portal_opens (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            kind         TEXT NOT NULL,
            key          TEXT NOT NULL,
            first_opened TEXT,
            last_opened  TEXT,
            open_count   INTEGER NOT NULL DEFAULT 0,
            UNIQUE(kind, key)
        )
    """)
    cx.commit()


def _secs(a, b):
    try:
        return abs((datetime.datetime.strptime(b, _FMT) - datetime.datetime.strptime(a, _FMT)).total_seconds())
    except Exception:
        return 10 ** 9  # unparseable → treat as far apart (bump)


def record_open(cx, kind, key, *, now=None):
    now = now or _now()
    row = cx.execute("SELECT first_opened, last_opened, open_count FROM portal_opens WHERE kind=? AND key=?",
                     (kind, key)).fetchone()
    if row is None:
        cx.execute("INSERT INTO portal_opens (kind, key, first_opened, last_opened, open_count) VALUES (?,?,?,?,1)",
                   (kind, key, now, now))
        cx.commit()
        return {"first_opened": now, "last_opened": now, "open_count": 1}
    first, last, count = row
    new_count = count + 1 if _secs(last, now) >= _DEBOUNCE_SECONDS else count
    cx.execute("UPDATE portal_opens SET last_opened=?, open_count=? WHERE kind=? AND key=?",
               (now, new_count, kind, key))
    cx.commit()
    return {"first_opened": first, "last_opened": now, "open_count": new_count}


def get_open(cx, kind, key):
    row = cx.execute("SELECT first_opened, last_opened, open_count FROM portal_opens WHERE kind=? AND key=?",
                     (kind, key)).fetchone()
    return None if row is None else {"first_opened": row[0], "last_opened": row[1], "open_count": row[2]}


def opens_for(cx, kind, keys):
    out = {}
    for k in keys:
        r = get_open(cx, kind, k)
        if r:
            out[k] = r
    return out


def clear_open(cx, kind, key):
    cx.execute("DELETE FROM portal_opens WHERE kind=? AND key=?", (kind, key))
    cx.commit()
```

- [ ] **Step 4: Run to verify it passes** — GREEN.

- [ ] **Step 5: Commit**

```bash
git add dashboard/opens.py tests/test_opens.py
git commit -m "feat(opens): read-receipt opens table + record/get with 5s debounce"
```

---

### Task 2: Track-open endpoints + payload keys + reassign-clears-open

**Files:** Modify `app.py`; Test `tests/test_open_routes.py`

**Interfaces:**
- Consumes: `opens.{init_opens_table,record_open,get_open,opens_for,clear_open,report_key,invoice_key}` (Task 1); `_portal_record_for`, `_portal_console_ok`, `_household_view_enabled`, `dashboard.household.{init_household_tables,can_view}`, `_invoice_order_for_token`/`_pp.order_id_from_invoice_token`, `_pbr.list_report_dates`.
- Produces: `_read_receipts_enabled() -> bool`; `POST /api/portal/<token>/open`; `POST /api/invoice/<token>/open`; `api_client_portal` payload gains `opens` (map by scan_date) when enabled; `api_invoice_get` gains `opened`; `api_console_household_reassign` clears the moved report's old open record.

**Context / exact anchors (read before editing — line numbers drift):**
- `api_client_portal` returns at `return jsonify(payload)` (~app.py:14463, right after the `payload["household"]` block). The `?member=` resolution that yields `email_for_reports` is at ~app.py:14320-14335 — REPLICATE that resolution in `POST /api/portal/<token>/open` so a caregiver's open records against the member's key.
- `api_invoice_get(token)` ~app.py:14210/30210 returns `{"ok":True,"order": summary}`; add `summary["opened"]`.
- `api_console_household_reassign` (from the shipped household feature) calls `household.reassign_report(...)`; after a successful reassign, clear the OLD report's open (`opens.clear_open(cx,'report', report_key(from_email, scan_date))`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_open_routes.py
import importlib, sqlite3, sys
from pathlib import Path
import pytest


def _app(tmp_path, monkeypatch, *, rr="1"):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("READ_RECEIPTS_ENABLED", rr)
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path: sys.path.insert(0, str(repo))
    try:
        import app as appmod; importlib.reload(appmod)
        import dashboard as _d; monkeypatch.setattr(_d, "CONSOLE_SECRET", "", raising=False)
    except Exception as e:
        pytest.skip(f"app not importable: {e}")
    return appmod


def _seed_portal(appmod, email):
    from dashboard import client_portal as cp, portal_biofield_reports as pbr, opens
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cp.init_client_portal_table(cx); pbr.init_table(cx); opens.init_opens_table(cx)
        pbr.upsert_report(cx, email, "2026-06-25", "s1", {"n": 1}, "confirmed"); cx.commit()
        token = cp.upsert_portal(cx, email, "Client", {})   # confirm the real mint helper
        cx.commit()
    return token[0] if isinstance(token, (tuple, list)) else token


def test_report_open_records_and_payload(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    token = _seed_portal(appmod, "c@x.com")
    if not token: pytest.skip("no portal mint helper")
    c = appmod.app.test_client()
    # before open: payload opens map has no entry for the date
    j = c.get(f"/api/portal/{token}").get_json()
    assert (j.get("opens") or {}).get("2026-06-25") in (None,)
    # explicit open records it
    r = c.post(f"/api/portal/{token}/open", json={"scan_date": "2026-06-25"})
    assert r.status_code == 200 and r.get_json()["ok"] is True
    from dashboard import opens
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert opens.get_open(cx, "report", "c@x.com|2026-06-25")["open_count"] == 1
    # now the payload reflects the open
    j2 = c.get(f"/api/portal/{token}").get_json()
    assert (j2.get("opens") or {}).get("2026-06-25", {}).get("open_count") == 1


def test_owner_open_does_not_record(tmp_path, monkeypatch):
    # with CONSOLE_SECRET set, an owner-keyed call must NOT record
    monkeypatch.setenv("CONSOLE_SECRET", "sek")
    appmod = _app(tmp_path, monkeypatch)
    import dashboard as _d; monkeypatch.setattr(_d, "CONSOLE_SECRET", "sek", raising=False)
    token = _seed_portal(appmod, "c@x.com")
    if not token: pytest.skip("no portal mint helper")
    c = appmod.app.test_client()
    r = c.post(f"/api/portal/{token}/open?key=sek", json={"scan_date": "2026-06-25"})
    assert r.status_code == 200
    from dashboard import opens
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert opens.get_open(cx, "report", "c@x.com|2026-06-25") is None   # skipped


def test_flag_off_inert(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch, rr="0")
    token = _seed_portal(appmod, "c@x.com")
    if not token: pytest.skip("no portal mint helper")
    c = appmod.app.test_client()
    j = c.get(f"/api/portal/{token}").get_json()
    assert "opens" not in j                       # no payload key when flag off
    r = c.post(f"/api/portal/{token}/open", json={"scan_date": "2026-06-25"})
    from dashboard import opens
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert opens.get_open(cx, "report", "c@x.com|2026-06-25") is None   # inert
```

> Implementer notes: confirm the real portal-mint helper (`client_portal.upsert_portal` returns a tuple in the household tests — reuse that exact call) and the payload's `opens` key name; if a helper differs, adapt the seed to the real one rather than leaving the test skipped.

- [ ] **Step 2: Run to verify it fails**

Run: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_open_routes.py -q`
Expected: FAIL (404 on `/open`, no `opens` key).

- [ ] **Step 3: Write minimal implementation**

Add the flag helper near `_household_view_enabled` (app.py):

```python
def _read_receipts_enabled():
    return (os.environ.get("READ_RECEIPTS_ENABLED", "") or "").strip().lower() in ("1", "true", "yes")
```

Add the two endpoints (near the other `/api/portal/<token>/*` POST routes, ~app.py:14489):

```python
@app.route("/api/portal/<token>/open", methods=["POST"])
def api_portal_open(token):
    """Record a real 'client opened this report' event (fired by the expand-click).
    Owner previews (console key) return ok but don't record. Resolves ?member= like
    api_client_portal so a caregiver's open records against the member's key."""
    if not _read_receipts_enabled():
        return jsonify({"ok": True, "recorded": False, "reason": "disabled"})
    from dashboard import client_portal as _cp
    from dashboard import opens as _op
    with sqlite3.connect(LOG_DB) as cx:
        _cp.init_client_portal_table(cx)
        portal = _portal_record_for(cx, token)
    if not portal:
        return jsonify({"ok": False, "error": "not found"}), 404
    email_for_reports = (portal.get("email") or "").strip().lower()
    # same household ?member= resolution as api_client_portal (fail-closed)
    if _household_view_enabled() and email_for_reports:
        try:
            from dashboard import household as _hh
            with sqlite3.connect(LOG_DB) as _cxh:
                _hh.init_household_tables(_cxh)
                _m = (request.args.get("member") or (request.get_json(silent=True) or {}).get("member") or "").strip().lower()
                if _m and _hh.can_view(_cxh, email_for_reports, _m):
                    email_for_reports = _m
        except Exception as _e:
            print(f"[open] household {_e!r}", flush=True)
    scan_date = ((request.get_json(silent=True) or {}).get("scan_date") or "").strip()
    if not scan_date or not email_for_reports:
        return jsonify({"ok": False, "error": "missing scan_date"}), 400
    if _portal_console_ok():   # owner preview → don't record
        return jsonify({"ok": True, "recorded": False, "reason": "owner"})
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        _op.init_opens_table(cx)
        st = _op.record_open(cx, "report", _op.report_key(email_for_reports, scan_date))
    return jsonify({"ok": True, "recorded": True, "open": st})


@app.route("/api/invoice/<token>/open", methods=["POST"])
def api_invoice_open(token):
    """Record a real 'client opened this invoice' event (fired by the expand-click)."""
    if not _read_receipts_enabled():
        return jsonify({"ok": True, "recorded": False, "reason": "disabled"})
    if not _pp.order_id_from_invoice_token(token):
        return jsonify({"ok": False, "error": "invalid or expired invoice"}), 404
    from dashboard import opens as _op
    if _portal_console_ok():
        return jsonify({"ok": True, "recorded": False, "reason": "owner"})
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        _op.init_opens_table(cx)
        st = _op.record_open(cx, "invoice", _op.invoice_key(token))
    return jsonify({"ok": True, "recorded": True, "open": st})
```

Payload — in `api_client_portal`, immediately before `return jsonify(payload)` (~app.py:14463):

```python
    if _read_receipts_enabled():
        try:
            from dashboard import opens as _op
            with sqlite3.connect(LOG_DB) as _cxo:
                _op.init_opens_table(_cxo)
                _keys = [_op.report_key(email_for_reports, d) for d in (bf_scan_dates or [])]
                _byk = _op.opens_for(_cxo, "report", _keys)
                payload["opens"] = {d: _byk.get(_op.report_key(email_for_reports, d))
                                    for d in (bf_scan_dates or []) if _byk.get(_op.report_key(email_for_reports, d))}
        except Exception as _e:
            print(f"[opens] payload {_e!r}", flush=True)
            payload["opens"] = {}
```

Payload — in `api_invoice_get`, after `summary = _invoice_summary(order)` and before returning:

```python
    if _read_receipts_enabled():
        try:
            from dashboard import opens as _op
            with sqlite3.connect(LOG_DB) as _cxo:
                _op.init_opens_table(_cxo)
                summary["opened"] = _op.get_open(_cxo, "invoice", _op.invoice_key(token))
        except Exception as _e:
            print(f"[opens] invoice {_e!r}", flush=True)
```

Reassign-clears-open — in `api_console_household_reassign`, after `res = _hh.reassign_report(...)` when `res["ok"]`:

```python
        if res.get("ok"):
            try:
                from dashboard import opens as _op
                _op.init_opens_table(cx)
                _op.clear_open(cx, "report", _op.report_key(data.get("from_email"), data.get("scan_date")))
            except Exception as _e:
                print(f"[opens] reassign-clear {_e!r}", flush=True)
```

- [ ] **Step 4: Run to verify it passes** — GREEN.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_open_routes.py
git commit -m "feat(opens): track-open endpoints + payload keys + reassign clears open (flag-gated)"
```

---

### Task 3: Collapsed report card in `static/client-portal.html`

**Files:** Modify `static/client-portal.html`

**Interfaces:**
- Consumes: payload `d.opens` (`{scan_date: {first_opened,last_opened,open_count}}`), `d.scan_date` (picked), `d.scan_dates`, and `TOKEN` + any `?member=` already in the URL.

UI-only (no pytest). Behind the payload: when `d.opens` is ABSENT (flag off), render the report exactly as today (expanded, no card). When present, wrap the report body in a collapsed card.

- [ ] **Step 1: Add the collapsed card + expand-tracks**

In `render(d, v)` (grep `function render`), wrap the biofield report body section so that, when `d.opens !== undefined`:
- Render a header row: the picked `d.scan_date` (the generation/test date) + a **New** badge when `!d.opens[d.scan_date]` (no recorded open). The body is hidden (`display:none`) until expanded.
- Expanding calls `openReport(d.scan_date)` which POSTs `/api/portal/${TOKEN}/open` (carry `?member=` from the URL if present) with `{scan_date}`, then reveals the body and removes the New badge.
- The scan-date selector options show a New badge next to any date with no `d.opens[date]` entry.
- Escape all injected strings via the file's `esc()`.

```javascript
  // Read-receipt: collapse the report behind a New/date header; expand = tracked open.
  if (d.opens !== undefined) {
    const sd = d.scan_date || "";
    const isNew = sd && !(d.opens && d.opens[sd]);
    const memberQS = new URLSearchParams(location.search).get("member");
    window.__openReport = async (scanDate) => {
      try {
        const u = `/api/portal/${encodeURIComponent(TOKEN)}/open` + (memberQS ? `?member=${encodeURIComponent(memberQS)}` : "");
        await fetch(u, {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({scan_date: scanDate})});
      } catch(e) {}
      const body = document.getElementById("rr-body");
      if (body) body.style.display = "";
      const badge = document.getElementById("rr-new");
      if (badge) badge.remove();
    };
    html += `<div class="card">
      <div style="display:flex;align-items:center;gap:10px;cursor:pointer" onclick="window.__openReport('${esc(sd)}')">
        <strong>Biofield report — ${esc(sd || "")}</strong>
        ${isNew ? `<span id="rr-new" style="background:#d4a843;color:#000;font-size:.72rem;font-weight:700;padding:1px 7px;border-radius:10px">New</span>` : ""}
        <span style="flex:1"></span><span style="color:var(--muted)">▾</span>
      </div>
      <div id="rr-body" style="display:${isNew ? "none" : ""};margin-top:12px">`;
    // ... existing report body markup goes INSIDE #rr-body ...
    html += `</div></div>`;
  } else {
    // flag off: existing report body markup, unchanged
  }
```

> Implementer: locate the current report-body markup block in `render` and move it inside `#rr-body` for the `d.opens !== undefined` branch, leaving the else-branch as today's markup verbatim (DRY: factor the body string once and place it in both branches). Confirm the payload uses `d.scan_date`/`d.scan_dates` (it does) and that `TOKEN` is the page's token var (grep it).

- [ ] **Step 2: Verify (static)** — extract the inline `<script>`, `node --check`; grep-confirm it POSTs `/open`, reads `d.opens`, escapes strings, and the flag-off branch renders the body expanded. Report live render-verify pending.

- [ ] **Step 3: Commit**

```bash
git add static/client-portal.html
git commit -m "feat(opens): collapsed report card with New badge + expand-tracks open"
```

---

### Task 4: Collapsed invoice in `static/invoice.html`

**Files:** Modify `static/invoice.html`

**Interfaces:**
- Consumes: the `/api/invoice/<token>` response `order.opened` (`{...}|null`) and `order` fields (date + line items); `TOKEN` (the page's token var, from `const API = '/api/invoice/' + TOKEN`).

UI-only. When `order.opened !== undefined` (flag on): render a collapsed summary header (invoice generation date + New when `!order.opened`); clicking reveals the line items and POSTs `/api/invoice/${TOKEN}/open`. When `opened` is absent (flag off), render as today.

- [ ] **Step 1: Add collapse + expand-tracks**

In `invoice.html`'s render (the function that consumes the fetched `order`), when `order.opened !== undefined`, wrap the line-items/detail block in a hidden container behind a header showing the invoice date + a New badge (`!order.opened`). Expanding calls a `openInvoice()` that POSTs `/api/invoice/${TOKEN}/open`, reveals the detail, and drops the New badge. Escape injected strings (mirror the file's existing escaping).

```javascript
  // Read-receipt: collapse invoice detail behind a New/date header; expand = tracked open.
  if (order.opened !== undefined) {
    const isNew = !order.opened;
    window.__openInvoice = async () => {
      try { await fetch('/api/invoice/' + encodeURIComponent(TOKEN) + '/open', {method:'POST'}); } catch(e) {}
      const det = document.getElementById('inv-detail'); if (det) det.style.display = '';
      const b = document.getElementById('inv-new'); if (b) b.remove();
    };
    // header (place ABOVE the detail container); invoiceDate = the order's generated date field
    headerHtml = `<div style="display:flex;align-items:center;gap:10px;cursor:pointer" onclick="window.__openInvoice()">
      <strong>Invoice — ${esc(invoiceDate)}</strong>
      ${isNew ? `<span id="inv-new" style="background:#d4a843;color:#000;font-size:12px;font-weight:700;padding:1px 7px;border-radius:10px">New</span>` : ``}
      <span style="flex:1"></span><span style="color:var(--muted)">▾</span></div>`;
    // wrap the existing detail markup in: <div id="inv-detail" style="display:${isNew?'none':''}">…</div>
  }
```

> Implementer: `invoice.html` may not have an `esc()` — if not, add a tiny one (same shape as other pages) and use it for the date. Read the file to find the invoice's generation-date field in `order`/`summary` (e.g. `order.created` / a date shown in the header) and use it for `invoiceDate`. Keep the flag-off path (when `order.opened === undefined`) rendering exactly as today.

- [ ] **Step 2: Verify (static)** — `node --check` the inline script; grep-confirm it POSTs `/api/invoice/<token>/open`, reads `order.opened`, and the flag-off path is unchanged. Report live render-verify pending.

- [ ] **Step 3: Commit**

```bash
git add static/invoice.html
git commit -m "feat(opens): collapsed invoice with New badge + expand-tracks open"
```

---

### Task 5: Owner surfacing — reveals console + orders board + read endpoint

**Files:** Modify `app.py`; (optionally `static/*` for the reveals/orders pages if they render server-injected data)

**Interfaces:**
- Consumes: `opens.opens_for` (Task 1); the reveals console payload (`api_console_biofield_reveals`, ~app.py:12496) and the orders board payload; `_portal_console_ok()`.
- Produces: report open-status in the reveals payload (per `email|scan_date`); invoice open-status in the orders payload (per invoice token); `GET /api/console/opens?kind=&keys=` (`_portal_console_ok`-gated) as a generic read.

- [ ] **Step 1: Write the failing test** (append to `tests/test_open_routes.py`)

```python
def test_console_opens_read(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    from dashboard import opens
    with sqlite3.connect(appmod.LOG_DB) as cx:
        opens.init_opens_table(cx)
        opens.record_open(cx, "invoice", "tokA", now="2026-07-04 10:00:00"); cx.commit()
    c = appmod.app.test_client()
    r = c.get("/api/console/opens?kind=invoice&keys=tokA,tokB")
    assert r.status_code == 200
    j = r.get_json()
    assert j["opens"]["tokA"]["open_count"] == 1 and "tokB" not in j["opens"]
```

- [ ] **Step 2: Run to verify it fails** — FAIL (404).

- [ ] **Step 3: Write minimal implementation**

Add the read endpoint (near other `/api/console/*`):

```python
@app.route("/api/console/opens", methods=["GET"])
def api_console_opens():
    if not _portal_console_ok():
        return jsonify({"error": "unauthorized"}), 401
    from dashboard import opens as _op
    kind = (request.args.get("kind") or "").strip()
    keys = [k for k in (request.args.get("keys") or "").split(",") if k]
    with sqlite3.connect(LOG_DB) as cx:
        _op.init_opens_table(cx)
        data = _op.opens_for(cx, kind, keys)
    return jsonify({"ok": True, "opens": data})
```

Reveals payload — in `api_console_biofield_reveals`, after the reveals list is built, annotate each with its open status:

```python
    from dashboard import opens as _op
    with sqlite3.connect(LOG_DB) as _cxo:
        _op.init_opens_table(_cxo)
        for r in reveals:   # use the actual list var name in this function
            r["opened"] = _op.get_open(_cxo, "report", _op.report_key(r.get("email", ""), r.get("scan_date", "")))
```

Orders board — in the orders payload builder, for each order carrying an invoice token, add `order["invoice_opened"] = opens.get_open(cx, "invoice", token)`. (Read the orders payload function; annotate with the invoice token it already computes.)

> Implementer: use the ACTUAL list/var names in `api_console_biofield_reveals` and the orders payload builder (read them). If the reveals rows don't carry `email`/`scan_date`, derive the key from whatever identifies the report. Keep the annotations additive; the console HTML can show them in a later pass — this task wires the DATA. If a console page renders server-side and a small "Opened …" label is trivial to add, add it; otherwise leave the payload for the page to consume.

- [ ] **Step 4: Run to verify it passes** — GREEN.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_open_routes.py
git commit -m "feat(opens): owner surfacing — reveals + orders open status + console read endpoint"
```

---

## Self-Review

**Spec coverage:**
- `portal_opens` table + record/get/opens_for/clear + keys + 5s debounce → Task 1. ✓
- Track-open POST endpoints, owner-skip, `?member=` resolution, flag-gated → Task 2. ✓
- `opens`/`opened` payload keys → Task 2. ✓
- Reassign resets to New (clear old open) → Task 2. ✓
- Collapsed report card (generation-date + New badge, expand-tracks) → Task 3. ✓
- Collapsed invoice → Task 4. ✓
- Owner surfacing (reveals + orders + read endpoint) → Task 5. ✓
- False-open filtering (POST-only, owner-skip, debounce) → Global Constraints + Tasks 1 (debounce) & 2 (POST + owner-skip). ✓
- Flag-off byte-identical → Tasks 2 (no payload keys, inert endpoints), 3 & 4 (else-branch renders as today). ✓

**Placeholder scan:** Tasks 3/4/5 carry implementer notes to read the real report-body markup block, invoice date field, and reveals/orders var names rather than guessing — these are "confirm the real name," not TBDs. No missing code. No "handle edge cases" hand-waves.

**Type consistency:** `record_open`/`get_open` return `{first_opened,last_opened,open_count}` everywhere (Tasks 1,2,5). `report_key(email,scan_date)` / `invoice_key(token)` used identically in Tasks 2 & 5. Payload keys: `opens` (report, map by scan_date) + `opened` (invoice, single) — consumed by Tasks 3 & 4 respectively with those exact names. `_read_receipts_enabled()` gates payload + endpoints (Task 2) and the render branches key on the payload's presence (Tasks 3,4), so the flag lives in one place.
