# Orders Board Biofield PDF Link — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show a "🖨 Biofield report (PDF)" print link on each console Orders-board card whose client has a published (confirmed) biofield report, so Rae can print it at pack time.

**Architecture:** A pure lookup helper `report_pdf_urls(cx, emails)` in `dashboard/portal_biofield_reports.py` returns `{email: pdf_url}` for clients with a latest-confirmed report; `GET /api/orders` annotates each order with `biofield_pdf_url` via one grouped call (mirroring the existing name-backfill block); `static/console-orders.html` renders the link when present.

**Tech Stack:** Python 3.11, Flask, sqlite3, pytest. Reuses `portal_biofield_reports` (the `report_pdf.url` stored by the publish flow).

## Global Constraints

- Only `status == "confirmed"` reports surface (never a draft/blurred report).
- The PDF link is the opaque `/portal-asset` URL already stored in `content.report_pdf.url` — same trust model; no new gating.
- Emails are lowercased on input and in returned keys.
- The `/api/orders` annotation must be ONE grouped lookup (no per-order round-trips) and wrapped in try/except that skips on error — exactly like the adjacent fulfillment/name-backfill annotations.
- `app.py` cannot be imported offline (Pinecone validates at import). Task 2's app.py + HTML changes are verified LIVE post-deploy (curl + board load); their steps document the exact checks. Task 1's helper is fully offline-TDD.
- Offline test cmd: `~/.venvs/deploy-chat311/bin/python -m pytest tests/<file> -v`.

---

### Task 1: `report_pdf_urls` lookup helper

**Files:**
- Modify: `dashboard/portal_biofield_reports.py`
- Test: `tests/test_portal_biofield_reports_pdf_urls.py`

**Interfaces:**
- Consumes: existing `init_table(cx)`, `upsert_report(cx, email, scan_date, scan_id, content, status)` (for test setup).
- Produces: `report_pdf_urls(cx, emails) -> dict[str, str]` — `{email_lower: url}` for emails whose LATEST confirmed report has a non-empty `content.report_pdf.url`; others omitted; empty input → `{}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_portal_biofield_reports_pdf_urls.py
import sqlite3
from dashboard import portal_biofield_reports as pbr

def _cx():
    cx = sqlite3.connect(":memory:")
    pbr.init_table(cx)
    return cx

def test_returns_url_for_confirmed_report_with_pdf():
    cx = _cx()
    pbr.upsert_report(cx, "k@example.com", "2026-06-25", "",
                      {"report_pdf": {"url": "https://h/portal-asset/r.pdf"}}, "confirmed")
    assert pbr.report_pdf_urls(cx, ["k@example.com"]) == {"k@example.com": "https://h/portal-asset/r.pdf"}

def test_picks_latest_confirmed_when_multiple():
    cx = _cx()
    pbr.upsert_report(cx, "k@example.com", "2026-06-20", "",
                      {"report_pdf": {"url": "https://h/old.pdf"}}, "confirmed")
    pbr.upsert_report(cx, "k@example.com", "2026-06-25", "",
                      {"report_pdf": {"url": "https://h/new.pdf"}}, "confirmed")
    assert pbr.report_pdf_urls(cx, ["k@example.com"]) == {"k@example.com": "https://h/new.pdf"}

def test_omits_non_confirmed_and_missing_pdf():
    cx = _cx()
    pbr.upsert_report(cx, "draft@example.com", "2026-06-25", "",
                      {"report_pdf": {"url": "https://h/d.pdf"}}, "ai_draft")
    pbr.upsert_report(cx, "nopdf@example.com", "2026-06-25", "",
                      {"greeting": "hi"}, "confirmed")
    assert pbr.report_pdf_urls(cx, ["draft@example.com", "nopdf@example.com"]) == {}

def test_lowercases_and_empty_input():
    cx = _cx()
    pbr.upsert_report(cx, "k@example.com", "2026-06-25", "",
                      {"report_pdf": {"url": "https://h/r.pdf"}}, "confirmed")
    assert pbr.report_pdf_urls(cx, ["K@Example.COM"]) == {"k@example.com": "https://h/r.pdf"}
    assert pbr.report_pdf_urls(cx, []) == {}
```

- [ ] **Step 2: Run to verify it fails**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_portal_biofield_reports_pdf_urls.py -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'report_pdf_urls'`.

- [ ] **Step 3: Implement**

Append to `dashboard/portal_biofield_reports.py` (the module already imports `json`):

```python
def report_pdf_urls(cx, emails):
    """{email_lower: url} for each email whose LATEST confirmed report carries a
    non-empty content.report_pdf.url. Emails without one are omitted. None-raising."""
    wanted = sorted({(e or "").strip().lower() for e in (emails or []) if (e or "").strip()})
    if not wanted:
        return {}
    ph = ",".join("?" * len(wanted))
    rows = cx.execute(
        f"SELECT lower(email), content_json FROM portal_biofield_reports "
        f"WHERE lower(email) IN ({ph}) AND status='confirmed' "
        f"ORDER BY scan_date DESC", wanted).fetchall()
    out = {}
    for em, content_json in rows:
        if em in out:
            continue                      # rows are newest-first; keep the first per email
        try:
            url = ((json.loads(content_json or "{}").get("report_pdf") or {}).get("url") or "").strip()
        except Exception:
            url = ""
        if url:
            out[em] = url
    return out
```

- [ ] **Step 4: Run to verify it passes**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_portal_biofield_reports_pdf_urls.py -v`
Expected: PASS (4 tests).

Note on `test_omits_non_confirmed_and_missing_pdf`: the `nopdf@` row IS confirmed but has no `report_pdf`, so it's skipped (url empty); the `draft@` row is filtered by the `status='confirmed'` clause. Both correctly omitted.

- [ ] **Step 5: Commit**

```bash
git add dashboard/portal_biofield_reports.py tests/test_portal_biofield_reports_pdf_urls.py
git commit -m "feat(orders-pdf): report_pdf_urls lookup for latest-confirmed reports"
```

---

### Task 2: `/api/orders` annotation + Orders-board link

**Files:**
- Modify: `app.py` (the GET branch of `bos_orders_create`, ~line 24820, after the name-backfill `try/except` block)
- Modify: `static/console-orders.html` (the `cardHtml(o)` meta block, ~line 122)

**Interfaces:**
- Consumes: Task 1's `report_pdf_urls(cx, emails)`; `from dashboard import portal_biofield_reports as _pbr` (the alias used throughout app.py).
- Produces: each order dict gains `biofield_pdf_url` (string, `""` when none); the board renders a print link when non-empty.

**Why no offline test:** `app.py` can't import offline (Pinecone at import); the board is JS. Verified live post-deploy (Steps 3-4). Keep the annotation a faithful mirror of the adjacent name-backfill block.

- [ ] **Step 1: Add the `/api/orders` annotation**

In `app.py`, in the GET branch of `bos_orders_create`, immediately AFTER the existing name-backfill `try/except` block (the one that ends `print(f"[orders] name backfill skipped: ...")`) and BEFORE `finally: cx.close()`, insert:

```python
            # Annotate each order with the client's published biofield report PDF
            # (latest CONFIRMED report only), so Rae can print it at pack time.
            try:
                from dashboard import portal_biofield_reports as _pbr
                _pbr.init_table(cx)
                emails = {(o.get("email") or "").strip().lower()
                          for o in rows if (o.get("email") or "").strip()}
                pdf_urls = _pbr.report_pdf_urls(cx, emails)
                for o in rows:
                    o["biofield_pdf_url"] = pdf_urls.get((o.get("email") or "").strip().lower(), "")
            except Exception as _e:
                print(f"[orders] biofield pdf annotate skipped: {_e!r}", flush=True)
```

- [ ] **Step 2: Add the board link**

In `static/console-orders.html`, in `cardHtml(o)`, inside the `meta` div string concatenation — add the link right after the tracking-number line and before the `+'</div>'` that closes the meta div. Change:

```javascript
      + (o.tracking_number?'<br>tracking '+esc(o.tracking_number):'')+'</div>'
```
to:
```javascript
      + (o.tracking_number?'<br>tracking '+esc(o.tracking_number):'')
      + (o.biofield_pdf_url?'<br><a href="'+esc(o.biofield_pdf_url)+'" target="_blank" rel="noopener">&#128424; Biofield report (PDF)</a>':'')+'</div>'
```

- [ ] **Step 3: Commit**

```bash
git add app.py static/console-orders.html
git commit -m "feat(orders-pdf): annotate /api/orders + render biofield PDF print link"
```

- [ ] **Step 4: Live verification (post-deploy, at go-live — record commands in the report)**

```bash
# /api/orders includes biofield_pdf_url; for a client with a published confirmed report it is non-empty:
doppler run -p remedy-match -c prd -- sh -c 'curl -s "https://glen-knowledge-chat.onrender.com/api/orders?limit=300" -H "X-Console-Key: $CONSOLE_SECRET"' | python3 -c "import sys,json; d=json.load(sys.stdin); print([{k:o.get(k) for k in (\"email\",\"biofield_pdf_url\")} for o in d.get(\"data\",[]) if o.get(\"biofield_pdf_url\")][:5])"
```
Then load `/console/orders` in a browser and confirm the "🖨 Biofield report (PDF)" link renders on the relevant order and opens the PDF (200) in a new tab. Also confirm `app.py` parses: `~/.venvs/deploy-chat311/bin/python -c "import ast; ast.parse(open('app.py').read()); print('OK')"`.

---

## Self-Review

**1. Spec coverage:**
- `report_pdf_urls` helper (latest-confirmed, omit-none, lowercase, empty→{}) → Task 1. ✅
- `/api/orders` grouped annotation `biofield_pdf_url`, try/except-skip → Task 2 Step 1. ✅
- Board print link, confirmed-only (enforced by the helper), opaque URL → Task 2 Step 2. ✅
- Live curl + board verify → Task 2 Step 4. ✅

**2. Placeholder scan:** No TBD/TODO. Task 2's no-offline-test rationale is explicit with concrete live commands. ✅

**3. Type consistency:** `report_pdf_urls(cx, emails) -> dict` used identically in Task 1 (producer) and Task 2 (consumer); the order field name `biofield_pdf_url` matches between Task 2 Step 1 (api) and Step 2 (html). ✅
