# Portal-Reveal Slice 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render a client's biofield **reveal** (`biofield_reveals`, System A) in the portal as their scan — most-recent + Scan History — for portal users who have no System B report, without changing the funnel or System B.

**Architecture:** Add a System-A branch to the portal's biofield resolution: System B report (if any) → **else** the reveal → legacy → nothing. A reveal is normalized into the existing portal report-content shape and rendered through the existing assemblers, so blur is **binary** (remedies show only when the client has paid — identical to the report model, strictly revenue-safe). Two independent readers exist (`portal_view._biofield_block` and the inline reader in the live `/api/portal` route); both get the same branch via one shared normalizer.

**Tech Stack:** Python 3.11 (Flask app), sqlite (`chat_log.db` / `LOG_DB`), pytest.

## Global Constraints

- The portal reveal MUST NOT un-blur remedies beyond the funnel: remedy/dosing strings appear only when the client has PAID (`_portal_biofield_unlocked(email)` / `unlocked=True`). When blurred, remedy/dosing strings are never assembled into the payload.
- System B (`portal_biofield_reports`) behavior is UNCHANGED — a client who has a System B report sees exactly what they see today.
- All new reads are best-effort / None-raising — a reveal read failure must never break portal load.
- Blur is BINARY in this slice (paid → all remedies; else all blurred). `top_unlocked` (top-only) is deferred to a later slice.
- `biofield_reveals` rows are keyed unique on `(email, scan_date)`.

---

### Task 1: `list_for_email` — read a client's reveals

**Files:**
- Modify: `dashboard/biofield_reveals.py` (add function near `list_pending`)
- Test: `tests/test_biofield_reveals_list_for_email.py`

**Interfaces:**
- Produces: `list_for_email(cx, email) -> list[dict]` — reveal row dicts (via `_row`) for the email, newest `scan_date` first. Empty list on none.

- [ ] **Step 1: Write the failing test**

```python
import sqlite3
from dashboard import biofield_reveals as br

def _db():
    cx = sqlite3.connect(":memory:"); br.init_table(cx); return cx

def test_list_for_email_returns_rows_newest_first():
    cx = _db()
    br.upsert(cx, "a@x.com", "2026-07-10", {"greeting": "hi"}, [], "t")
    br.upsert(cx, "a@x.com", "2026-07-18", {"greeting": "yo"}, [], "t")
    br.upsert(cx, "b@x.com", "2026-07-18", {}, [], "t")
    rows = br.list_for_email(cx, "A@x.com")            # case-insensitive
    assert [r["scan_date"] for r in rows] == ["2026-07-18", "2026-07-10"]
    assert rows[0]["interpretation"] == {"greeting": "yo"}

def test_list_for_email_empty_when_none():
    assert br.list_for_email(_db(), "none@x.com") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_reveals_list_for_email.py -q`
Expected: FAIL — `module 'dashboard.biofield_reveals' has no attribute 'list_for_email'`

- [ ] **Step 3: Write minimal implementation**

Add to `dashboard/biofield_reveals.py`:

```python
def list_for_email(cx, email):
    """All reveals for an email, newest scan_date first (row dicts via _row)."""
    email = (email or "").strip().lower()
    rows = _rows_cursor(cx).execute(
        "SELECT * FROM biofield_reveals WHERE email=? ORDER BY scan_date DESC, id DESC",
        (email,)).fetchall()
    return [_row(r) for r in rows]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_reveals_list_for_email.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add dashboard/biofield_reveals.py tests/test_biofield_reveals_list_for_email.py
git commit -m "feat(reveals): list_for_email — a client's reveals newest-first"
```

---

### Task 2: `_reveal_as_report_content` — normalize a reveal into portal report-content shape

**Files:**
- Modify: `dashboard/portal_view.py` (add near `_biofield_block`)
- Test: `tests/test_portal_reveal_normalize.py`

**Interfaces:**
- Consumes: a reveal row dict from `biofield_reveals._row` (`interpretation`, `layers`, `remedies`).
- Produces: `_reveal_as_report_content(reveal) -> dict` in the SAME shape `portal_biofield_reports` content uses: `{"greeting": str, "layers": [{"n","title","meaning","remedy","dosing"}], "video": {}}`. `remedy`/`dosing` are plain strings (the report shape); `_assemble_biofield`'s `show` gate blurs them when unpaid.

- [ ] **Step 1: Write the failing test**

```python
from dashboard.portal_view import _reveal_as_report_content

def test_layers_map_greeting_title_meaning_and_remedy_strings():
    reveal = {
        "interpretation": {"greeting": "Aloha", "body": "..."},
        "layers": [{"n": 1, "title": "Layer One", "meaning": "m1",
                    "remedy": {"name": "Calm Formula", "dosing": "2/day"}}],
        "remedies": [],
    }
    c = _reveal_as_report_content(reveal)
    assert c["greeting"] == "Aloha"
    assert c["layers"] == [{"n": 1, "title": "Layer One", "meaning": "m1",
                            "remedy": "Calm Formula", "dosing": "2/day"}]

def test_flat_reveal_without_layers_maps_remedies_to_layers():
    reveal = {"interpretation": {"greeting": "Hi"}, "layers": [],
              "remedies": [{"name": "Top Match", "meaning": "why", "dosing": "1/day"}]}
    c = _reveal_as_report_content(reveal)
    assert c["layers"] == [{"n": 1, "title": "", "meaning": "why",
                            "remedy": "Top Match", "dosing": "1/day"}]

def test_empty_reveal_yields_no_layers():
    assert _reveal_as_report_content({"interpretation": {}, "layers": [], "remedies": []})["layers"] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_portal_reveal_normalize.py -q`
Expected: FAIL — `cannot import name '_reveal_as_report_content'`

- [ ] **Step 3: Write minimal implementation**

Add to `dashboard/portal_view.py`:

```python
def _reveal_as_report_content(reveal):
    """Normalize a biofield_reveals row into the portal report-content shape so the
    existing assemblers render it identically to a System B report. Remedy/dosing
    are strings; the caller's blur gate decides whether they leave the server."""
    greeting = ((reveal.get("interpretation") or {}).get("greeting") or "").strip()
    layers = []
    raw = reveal.get("layers") or []
    if raw:
        for L in raw:
            rem = L.get("remedy") if isinstance(L.get("remedy"), dict) else {}
            layers.append({
                "n": L.get("n"),
                "title": L.get("title", "") or "",
                "meaning": (L.get("meaning") or L.get("summary") or ""),
                "remedy": (rem.get("name") or ""),
                "dosing": (rem.get("dosing") or ""),
            })
    else:
        for i, r in enumerate(reveal.get("remedies") or []):
            if not isinstance(r, dict):
                continue
            layers.append({"n": i + 1, "title": "", "meaning": (r.get("meaning") or ""),
                           "remedy": (r.get("name") or ""), "dosing": (r.get("dosing") or "")})
    return {"greeting": greeting, "layers": layers, "video": {}}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_portal_reveal_normalize.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add dashboard/portal_view.py tests/test_portal_reveal_normalize.py
git commit -m "feat(portal): normalize a reveal into report-content shape"
```

---

### Task 3: Add the reveal branch to `portal_view._biofield_block`

**Files:**
- Modify: `dashboard/portal_view.py:86-120` (`_biofield_block`)
- Test: `tests/test_portal_biofield_block_reveal.py`

**Interfaces:**
- Consumes: `biofield_reveals.list_for_email` (Task 1), `_reveal_as_report_content` (Task 2), existing `_assemble_biofield`.
- Produces: `_biofield_block` now returns a reveal-sourced block when no System B report/legacy content exists but reveals do.

- [ ] **Step 1: Write the failing test**

```python
import sqlite3
from dashboard import portal_view as pv
from dashboard import biofield_reveals as br

def _db():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    br.init_table(cx); return cx

def test_reveal_shows_blurred_when_unpaid():
    cx = _db()
    br.upsert(cx, "c@x.com", "2026-07-18", {"greeting": "Aloha"},
              [{"name": "Calm"}], "t",
              layers=[{"n": 1, "title": "T", "meaning": "m", "remedy": {"name": "Calm"}}])
    blk = pv._biofield_block(cx, "c@x.com", unlocked=False)
    assert blk["visible"] is True
    assert blk["blurred"] is True
    assert blk["scan_dates"] == ["2026-07-18"]
    assert "remedy" not in blk["layers"][0]          # remedy never leaves server when blurred

def test_reveal_shows_remedy_when_paid():
    cx = _db()
    br.upsert(cx, "c@x.com", "2026-07-18", {"greeting": "Aloha"},
              [{"name": "Calm"}], "t",
              layers=[{"n": 1, "title": "T", "meaning": "m", "remedy": {"name": "Calm"}}])
    blk = pv._biofield_block(cx, "c@x.com", unlocked=True)
    assert blk["blurred"] is False
    assert blk["layers"][0]["remedy"] == "Calm"

def test_no_biofield_data_is_not_visible():
    assert pv._biofield_block(_db(), "nobody@x.com", unlocked=True) == {"visible": False}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_portal_biofield_block_reveal.py -q`
Expected: FAIL — `test_reveal_shows_blurred_when_unpaid` gets `{"visible": False}` (reveal branch not wired).

- [ ] **Step 3: Write minimal implementation**

In `dashboard/portal_view.py`, inside `_biofield_block`, replace the legacy-fallback tail (currently the `# Legacy fallback` block that returns `{"visible": False}`) so the reveal is tried BEFORE giving up. After the `if dates:` System-B block, insert:

```python
    # System A: the funnel reveal (biofield_reveals). Rendered as the portal scan
    # when the client has no System B report. Blur is binary (paid -> remedies).
    try:
        from dashboard import biofield_reveals as _br
        _reveals = _br.list_for_email(cx, email)
    except Exception:
        _reveals = []
    if _reveals:
        _rev_dates = [r["scan_date"] for r in _reveals]
        _picked = scan_date if (scan_date in _rev_dates) else _rev_dates[0]
        _row = next((r for r in _reveals if r["scan_date"] == _picked), _reveals[0])
        _content = _reveal_as_report_content(_row)
        if not _content["layers"] and not _content["greeting"]:
            return {"visible": False}
        # status 'confirmed' so the assembler's show-gate depends only on `unlocked`
        # (paid) — binary blur, identical to a paid System B report.
        return _assemble_biofield(cx, _content, "confirmed", scan_date=_picked,
                                  scan_dates=_rev_dates, actionable=False, unlocked=unlocked)
```

Keep the existing legacy `client_portals` fallback AFTER this (a client with neither a report, a reveal, nor legacy content still returns `{"visible": False}`).

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_portal_biofield_block_reveal.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Run the existing portal_view tests (no regression)**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_portal_view_ambassador.py tests/test_portal_biofield_reports.py -q`
Expected: PASS (System B path unchanged)

- [ ] **Step 6: Commit**

```bash
git add dashboard/portal_view.py tests/test_portal_biofield_block_reveal.py
git commit -m "feat(portal): render the reveal in the /view biofield block (System B still wins)"
```

---

### Task 4: Add the reveal branch to the live `/api/portal` route

**Files:**
- Modify: `app.py:18420-18424` (the `else:` where `dates` is empty — no System B report)
- Test: covered by Task 3 unit tests for the shared normalizer + a render-verify in Task 5 (the live route is exercised end-to-end there; it has no isolated unit harness without the full app/doppler).

**Interfaces:**
- Consumes: `biofield_reveals.list_for_email`, `portal_view._reveal_as_report_content`.
- Produces: the live portal payload's `bf_content`/`bf_status`/`bf_scan_date`/`bf_scan_dates` are reveal-sourced when there is no System B report.

- [ ] **Step 1: Modify the `else` branch**

Current (`app.py:18420-18423`):

```python
    else:
        bf_content = content
        bf_status = content.get("biofield_status") or "confirmed"
        bf_scan_date, bf_scan_dates, bf_actionable = None, [], False
```

Replace with (reveal tried before falling back to legacy `content`):

```python
    else:
        _revs = []
        try:
            from dashboard import biofield_reveals as _brv
            _brv.init_table(cx_r)
            _revs = _brv.list_for_email(cx_r, email_for_reports) if email_for_reports else []
        except Exception as _re:
            print(f"[portal-reveal] read skipped: {_re!r}", flush=True)
        if _revs:
            _rev_dates = [r["scan_date"] for r in _revs]
            _picked = req_date if req_date in _rev_dates else _rev_dates[0]
            _row = next((r for r in _revs if r["scan_date"] == _picked), _revs[0])
            bf_content = _pv._reveal_as_report_content(_row)  # _pv = dashboard.portal_view (already imported)
            bf_status = "confirmed"
            bf_scan_date, bf_scan_dates, bf_actionable = _picked, _rev_dates, False
        else:
            bf_content = content
            bf_status = content.get("biofield_status") or "confirmed"
            bf_scan_date, bf_scan_dates, bf_actionable = None, [], False
```

Confirm `portal_view` is imported as `_pv` in this scope (it is used at `app.py:23731` as `_pv`); if the alias differs here, use the local import name. The paid gate downstream (`bf_show = bf_confirmed and _portal_biofield_unlocked(email_for_reports)`, `app.py:18425-18429`) already blurs remedies when unpaid — no change needed there.

- [ ] **Step 2: Compile check**

Run: `~/.venvs/deploy-chat311/bin/python -m py_compile app.py`
Expected: no output (success)

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat(portal): live /api/portal renders the reveal when no System B report"
```

---

### Task 5: Render-verify on the live app

**Files:** none (verification only)

- [ ] **Step 1: Start the app against a scratch LOG_DB with one reveal-only client**

Seed a scratch `chat_log.db`: a `client_portals`/people identity for `demo@x.com` with a portal token, a `biofield_reveals` row for `demo@x.com` (unpaid), and NO `portal_biofield_reports` row. Start the app with `LOG_DB=<scratch>` and a test `CONSOLE_SECRET`.

- [ ] **Step 2: Headless-render the portal**

Open `/<portal-token>` in headless Chrome. Confirm: the "most recent scan" card + Scan History show the reveal's layers, remedies **blurred**, greeting present.

- [ ] **Step 3: Verify no un-blur without pay**

Confirm the rendered HTML / payload contains NO remedy name for the unpaid client (grep the payload for a known remedy string — must be absent).

- [ ] **Step 4: Verify a System B client is unchanged**

Seed a second client with a `portal_biofield_reports` confirmed report; render; confirm the block is identical to `main` (System B still wins).

- [ ] **Step 5: Open the PR**

```bash
git push -u origin feat/portal-reveal-slice1
gh pr create --title "Portal reveal slice 1: portal reads the reveal (System A)" --body "<summary + verification>"
```

---

## Notes for the reviewer

- **Blur is binary this slice** (paid → all; else all blurred). The spec's 3-state (top-only via `top_unlocked`) is deferred — the portal shows *less* than the funnel for an approved-not-paid client, never more, so there's no revenue leak. Revisit if the top-only parity matters.
- **B-wins** is intentional for slice 1 so curated reports are untouched; folding/retiring System B is slice 4.
- The reveal read is added in two places (Task 3 + Task 4) sharing one normalizer (`_reveal_as_report_content`). Full consolidation of the two payload builders is a larger refactor deliberately NOT taken here to keep slice 1 low-risk; the shared normalizer prevents the reveal-mapping from drifting.
