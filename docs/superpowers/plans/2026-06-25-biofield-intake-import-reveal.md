# Biofield Intake — Import Reveal → Causal Chain — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a per-session button to the local Biofield Intake authoring page that runs the local reveal-synthesis pipeline for a client's recent E4L scan (< 7 days) and imports its layers + remedies as needs-review causal-chain rows.

**Architecture:** A new pure-ish module `dashboard/biofield_reveal_import.py` wraps the vault `e4l_synthesis` pipeline (the same one `02 Skills/e4l-reveal-push.py` uses) behind an injectable runner so tests stay offline, and writes the synthesized layers as `biofield_auth_chain` rows via the existing authoring store. One new Flask route in `biofield_local_app.py` gates on the 7-day rule and the append-confirm, and one button + JS function is added to the existing E4L panel / author page.

**Tech Stack:** Python 3.11, Flask, sqlite3, pytest. The vault modules (`e4l_synthesis`, `e4l_reveal_lib`) live in `~/AI-Training/02 Skills/`.

## Global Constraints

- Local-only tool (`biofield_local_app.py`, `127.0.0.1:8011`) — PHI stays on Glen's Mac; no prod deploy, no feature flag.
- Synthesis runs in-process via the vault pipeline; the app already runs under `doppler run -p remedy-match -c prd` so keys are present at runtime.
- Imported rows arrive **unconfirmed** (`confirmed=0`) — a review flag, not a schedule gate.
- Freshness gate is strict **`days_ago < 7`**.
- Tests must NOT touch the real `e4l.db` or the live pipeline — inject the runner / monkeypatch the synth function.
- Run tests with: `cd ~/deploy-chat && ~/.venvs/deploy-chat311/bin/python -m pytest <path> -v` (pure-module tests need no Doppler).
- Follow the existing authoring patterns: `add_chain_row(...)` already takes `confirmed=`; `remedy_dosing(cx, name)` returns `{"dosage","frequency","timing"}`.

---

### Task 1: `synthesize_reveal_layers` — run the pipeline and map layers

**Files:**
- Create: `dashboard/biofield_reveal_import.py`
- Test: `tests/test_biofield_reveal_import.py`

**Interfaces:**
- Produces:
  - `synthesize_reveal_layers(email, scan_id=None, *, e4l_db=DEFAULT_E4L_DB, catalog=DEFAULT_CATALOG, today, runner=None) -> dict` returning
    `{"found": bool, "scan_id": int|None, "scan_date": str|None, "days_ago": int|None, "fresh": bool, "layers": [ {"n": int|None, "title": str, "summary": str, "most_affected": str, "remedy_name": str} ... ]}`.
    `fresh = found and days_ago is not None and days_ago < 7`. `runner(email, scan_id, e4l_db, catalog, today) -> (scan|None, raw_layers)` is injectable; defaults to `_run_synthesis` (the real vault pipeline).
  - `_run_synthesis(email, scan_id, e4l_db, catalog, today) -> (scan, raw_layers)` — lazily imports the vault pipeline.
  - Module constants `DEFAULT_E4L_DB`, `DEFAULT_CATALOG`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_biofield_reveal_import.py
import datetime

from dashboard.biofield_reveal_import import synthesize_reveal_layers

_RAW = [
    {"n": 1, "title": "Oxidative load", "summary": "free-radical stress",
     "patterns": ["ED1", "ED2"], "pattern_labels": ["Cell membrane", "Mitochondria"],
     "remedy": {"name": "Neuro Magnesium"}},
    {"n": 2, "title": "Terrain", "summary": "", "patterns": ["ES3"],
     "pattern_labels": ["Lymphatics"], "remedy": None},
]


def _runner(found_date):
    def run(email, scan_id, e4l_db, catalog, today):
        if not found_date:
            return None, []
        return {"scan_id": 900, "scan_date": found_date}, _RAW
    return run


def test_maps_layers_and_marks_fresh_under_7_days():
    res = synthesize_reveal_layers("jane@x.com", today="2026-06-25",
                                   runner=_runner("2026-06-22"))
    assert res["found"] is True and res["fresh"] is True and res["days_ago"] == 3
    assert res["scan_id"] == 900 and res["scan_date"] == "2026-06-22"
    L0 = res["layers"][0]
    assert L0["n"] == 1 and L0["title"] == "Oxidative load"
    assert L0["most_affected"] == "Cell membrane, Mitochondria"
    assert L0["remedy_name"] == "Neuro Magnesium"
    # layer with remedy=None -> empty remedy_name, no crash
    assert res["layers"][1]["remedy_name"] == ""


def test_stale_scan_is_not_fresh_at_7_days():
    res = synthesize_reveal_layers("jane@x.com", today="2026-06-25",
                                   runner=_runner("2026-06-18"))  # 7 days
    assert res["found"] is True and res["days_ago"] == 7 and res["fresh"] is False


def test_no_scan_returns_not_found():
    res = synthesize_reveal_layers("nobody@x.com", today="2026-06-25",
                                   runner=_runner(None))
    assert res == {"found": False, "scan_id": None, "scan_date": None,
                   "days_ago": None, "fresh": False, "layers": []}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_reveal_import.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dashboard.biofield_reveal_import'`

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/biofield_reveal_import.py
"""Import a client's E4L reveal (synthesized layers + remedies) into a local
Biofield Intake authoring test as needs-review causal-chain rows.

Runs the SAME synthesis pipeline as `02 Skills/e4l-reveal-push.py`, in-process on
Glen's Mac (PHI stays local). The vault pipeline is imported lazily through an
injectable `runner` so unit tests never touch the real e4l.db or the live matcher.
"""
import datetime
import os
import sqlite3

VAULT = os.path.expanduser("~/AI-Training")
SKILLS = os.path.join(VAULT, "02 Skills")
DEFAULT_E4L_DB = os.path.join(VAULT, "e4l.db")
DEFAULT_CATALOG = os.path.expanduser("~/deploy-chat/data/products.json")


def _days_ago(scan_date, today):
    try:
        s = datetime.date.fromisoformat((scan_date or "").strip())
        t = datetime.date.fromisoformat((today or "").strip())
    except ValueError:
        return None
    return max(0, (t - s).days)


def _run_synthesis(email, scan_id, e4l_db, catalog, today):
    """Real pipeline: resolve the scan, synthesize, normalize to reveal layers.
    Returns ({scan_id, scan_date} | None, raw_layers). Mirrors e4l-reveal-push.py."""
    import sys
    if SKILLS not in sys.path:
        sys.path.insert(0, SKILLS)
    import e4l_synthesis as E  # noqa: E402
    from e4l_reveal_lib import build_payload  # noqa: E402
    cx = sqlite3.connect(e4l_db)
    try:
        if scan_id:
            row = cx.execute("SELECT scan_id, scan_date FROM e4l_scans WHERE scan_id=?",
                             (scan_id,)).fetchone()
            scan = {"scan_id": row[0], "scan_date": row[1]} if row else None
        else:
            scan = E.latest_scan(cx, email)
        if not scan:
            return None, []
        patterns = E.pull_patterns(cx, scan["scan_id"], limit=12)
        label_map = {p["item_code"]: (p.get("full_name") or p.get("name") or p["item_code"])
                     for p in patterns if p.get("item_code")}
        cat = E.load_catalog(catalog)
        synth = E.synthesize(patterns, history="", rules=E.load_rules(),
                             ff_names=E.curated_ff_names(cat), layer_count=6)
        synth["layers"] = E.order_layers_by_pattern_count(synth.get("layers") or [])
        content = E.to_portal_content(
            synth, cat, formulation_map=E.load_formulation_map(cx),
            member_age=E.member_age_for_email(cx, email, today),
            age_rules=E.load_age_rules(cx))
        payload = build_payload(content, email, scan["scan_date"],
                                label_map=label_map, notify=False)
        return scan, ((payload or {}).get("layers") or [])
    finally:
        cx.close()


def synthesize_reveal_layers(email, scan_id=None, *, e4l_db=DEFAULT_E4L_DB,
                             catalog=DEFAULT_CATALOG, today, runner=None):
    runner = runner or _run_synthesis
    scan, raw = runner(email, scan_id, e4l_db, catalog, today)
    if not scan or not raw:
        return {"found": False, "scan_id": None, "scan_date": None,
                "days_ago": None, "fresh": False, "layers": []}
    days = _days_ago(scan["scan_date"], today)
    layers = []
    for L in raw:
        rem = L.get("remedy") or {}
        name = (rem.get("name") or "").strip() if isinstance(rem, dict) else ""
        layers.append({"n": L.get("n"),
                       "title": (L.get("title") or "").strip(),
                       "summary": (L.get("summary") or "").strip(),
                       "most_affected": ", ".join(L.get("pattern_labels") or []),
                       "remedy_name": name})
    return {"found": True, "scan_id": scan["scan_id"], "scan_date": scan["scan_date"],
            "days_ago": days, "fresh": days is not None and days < 7, "layers": layers}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_reveal_import.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add dashboard/biofield_reveal_import.py tests/test_biofield_reveal_import.py
git commit -m "feat(biofield): synthesize_reveal_layers — map reveal layers + 7-day gate"
```

---

### Task 2: `import_layers_to_test` — write layers as needs-review chain rows

**Files:**
- Modify: `dashboard/biofield_reveal_import.py`
- Test: `tests/test_biofield_reveal_import.py`

**Interfaces:**
- Consumes: `synthesize_reveal_layers(...)["layers"]` shape from Task 1; `add_chain_row(cx, tid, layer, head, most_affected, remedy, dosage, frequency, timing, confirmed)` and `remedy_dosing(cx, name) -> {"dosage","frequency","timing"}` from `dashboard.biofield_authoring`.
- Produces: `import_layers_to_test(cx, tid, layers) -> int` (rows created). Each row is created with `confirmed=0`; dosing auto-filled from the catalog when the remedy name resolves.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_biofield_reveal_import.py
import sqlite3
from dashboard.biofield_authoring import (
    authored_report, create_test, init_auth_tables)
from dashboard.biofield_reveal_import import import_layers_to_test

_LAYERS = [
    {"n": 1, "title": "Oxidative load", "most_affected": "Cell membrane, Mitochondria",
     "remedy_name": "Neuro Magnesium"},
    {"n": 2, "title": "Terrain", "most_affected": "Lymphatics", "remedy_name": ""},
]


def test_import_creates_unconfirmed_rows(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "chat_log.db"))
    init_auth_tables(cx)
    tid = create_test(cx, "Jane", "jane@x.com", "2026-06-25")
    n = import_layers_to_test(cx, tid, _LAYERS)
    assert n == 2
    rep = authored_report(cx, tid)
    rows = {r["layer"]: r for r in rep["layers"]}
    assert rows[1]["remedy"] == "Neuro Magnesium" and rows[1]["confirmed"] == 0
    assert rows[1]["most_affected"] == "Cell membrane, Mitochondria"
    assert rows[2]["head"] == "Terrain"
    # Note: authored_report only returns rows with a non-empty remedy; the empty-remedy
    # layer 2 still appears here because head/most_affected carry it — but its remedy is "".
    cx.close()
```

NOTE: `authored_report` filters `WHERE TRIM(COALESCE(remedy,''))<>''`. The layer-2 row has an empty remedy, so it will NOT appear in `authored_report`. Adjust the test to assert layer 2 is absent from the report yet was counted by the importer:

```python
def test_import_creates_unconfirmed_rows(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "chat_log.db"))
    init_auth_tables(cx)
    tid = create_test(cx, "Jane", "jane@x.com", "2026-06-25")
    n = import_layers_to_test(cx, tid, _LAYERS)
    assert n == 2                       # both layers written
    rep = authored_report(cx, tid)
    rows = {r["layer"]: r for r in rep["layers"]}
    assert set(rows) == {1}            # only the remedy-bearing layer surfaces
    assert rows[1]["remedy"] == "Neuro Magnesium" and rows[1]["confirmed"] == 0
    assert rows[1]["most_affected"] == "Cell membrane, Mitochondria"
    cx.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_reveal_import.py::test_import_creates_unconfirmed_rows -v`
Expected: FAIL — `ImportError: cannot import name 'import_layers_to_test'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to dashboard/biofield_reveal_import.py
def import_layers_to_test(cx, tid, layers):
    """Create one needs-review (confirmed=0) chain row per reveal layer. Dosing is
    auto-filled from the product catalog when the remedy name resolves. Returns the
    number of rows created."""
    from dashboard.biofield_authoring import add_chain_row, remedy_dosing
    n = 0
    for L in layers or []:
        name = (L.get("remedy_name") or "").strip()
        d = remedy_dosing(cx, name) if name else {"dosage": "", "frequency": "", "timing": ""}
        add_chain_row(cx, tid, L.get("n"), L.get("title") or "",
                      L.get("most_affected") or "", name,
                      dosage=d.get("dosage", ""), frequency=d.get("frequency", ""),
                      timing=d.get("timing", ""), confirmed=0)
        n += 1
    return n
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_reveal_import.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add dashboard/biofield_reveal_import.py tests/test_biofield_reveal_import.py
git commit -m "feat(biofield): import_layers_to_test — write reveal layers as needs-review rows"
```

---

### Task 3: Route `POST /author/<test_id>/e4l/import-reveal` — gate + append-confirm

**Files:**
- Modify: `biofield_local_app.py` (add route near the other `/author/<test_id>/e4l/*` routes, ~line 257-290)
- Test: `tests/test_biofield_import_reveal_routes.py`

**Interfaces:**
- Consumes: `synthesize_reveal_layers(...)` and `import_layers_to_test(...)` from `dashboard.biofield_reveal_import`; the local app's `_report_for(cx, test_id)` (returns `{"client": {"email": ...}, "layers": [...]}`), `authored_report`, and `db_path`.
- Produces: route returning JSON — one of:
  - `{"ok": False, "reason": str}` (no client / no scan / stale)
  - `{"ok": False, "needs_confirm": True, "existing": int}` (rows exist, `force` not set)
  - `{"ok": True, "imported": int}` (rows written)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_biofield_import_reveal_routes.py
"""POST /author/<id>/e4l/import-reveal imports synthesized reveal layers as
needs-review chain rows. synthesize_reveal_layers is monkeypatched so the test
never runs the real vault pipeline; import_layers_to_test runs for real on a tmp db."""
import sqlite3
import pytest

from biofield_local_app import create_app
import dashboard.biofield_reveal_import as RI


@pytest.fixture(autouse=True)
def _no_gate(monkeypatch):
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)


_FRESH = {"found": True, "scan_id": 900, "scan_date": "2026-06-22", "days_ago": 3,
          "fresh": True, "layers": [
              {"n": 1, "title": "Oxidative load", "summary": "",
               "most_affected": "Cell membrane", "remedy_name": "Neuro Magnesium"}]}
_STALE = {"found": True, "scan_id": 900, "scan_date": "2026-06-01", "days_ago": 24,
          "fresh": False, "layers": []}
_NONE = {"found": False, "scan_id": None, "scan_date": None, "days_ago": None,
         "fresh": False, "layers": []}


def _new_test_with_email(client, email):
    tid = client.post("/author/new").headers["Location"].rstrip("/").split("/")[-1]
    client.post(f"/author/{tid}/header", json={"name": "Jane", "email": email,
                                               "date": "2026-06-25"})
    return tid


def test_import_writes_rows_when_fresh(tmp_path, monkeypatch):
    monkeypatch.setattr(RI, "synthesize_reveal_layers", lambda *a, **k: _FRESH)
    db = str(tmp_path / "chat_log.db")
    client = create_app(db, scan_lookup=lambda e: _NONE).test_client()
    tid = _new_test_with_email(client, "jane@x.com")
    j = client.post(f"/author/{tid}/e4l/import-reveal", json={}).get_json()
    assert j["ok"] is True and j["imported"] == 1
    # row landed, unconfirmed
    cx = sqlite3.connect(db)
    row = cx.execute("SELECT remedy, confirmed FROM biofield_auth_chain").fetchone()
    assert row[0] == "Neuro Magnesium" and row[1] == 0


def test_import_rejects_stale_scan(tmp_path, monkeypatch):
    monkeypatch.setattr(RI, "synthesize_reveal_layers", lambda *a, **k: _STALE)
    db = str(tmp_path / "chat_log.db")
    client = create_app(db, scan_lookup=lambda e: _NONE).test_client()
    tid = _new_test_with_email(client, "jane@x.com")
    j = client.post(f"/author/{tid}/e4l/import-reveal", json={}).get_json()
    assert j["ok"] is False and "24" in j["reason"]


def test_import_needs_confirm_then_appends_with_force(tmp_path, monkeypatch):
    monkeypatch.setattr(RI, "synthesize_reveal_layers", lambda *a, **k: _FRESH)
    db = str(tmp_path / "chat_log.db")
    client = create_app(db, scan_lookup=lambda e: _NONE).test_client()
    tid = _new_test_with_email(client, "jane@x.com")
    client.post(f"/author/{tid}/e4l/import-reveal", json={})          # first import (1 row)
    j = client.post(f"/author/{tid}/e4l/import-reveal", json={}).get_json()
    assert j == {"ok": False, "needs_confirm": True, "existing": 1}
    j2 = client.post(f"/author/{tid}/e4l/import-reveal", json={"force": True}).get_json()
    assert j2["ok"] is True and j2["imported"] == 1
    cx = sqlite3.connect(db)
    assert cx.execute("SELECT COUNT(*) FROM biofield_auth_chain").fetchone()[0] == 2


def test_import_no_client_email(tmp_path, monkeypatch):
    monkeypatch.setattr(RI, "synthesize_reveal_layers", lambda *a, **k: _FRESH)
    db = str(tmp_path / "chat_log.db")
    client = create_app(db, scan_lookup=lambda e: _NONE).test_client()
    tid = client.post("/author/new").headers["Location"].rstrip("/").split("/")[-1]
    j = client.post(f"/author/{tid}/e4l/import-reveal", json={}).get_json()
    assert j["ok"] is False and "client" in j["reason"].lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_import_reveal_routes.py -v`
Expected: FAIL — 404 on the new route (KeyError on `j["ok"]` / `j["imported"]`)

- [ ] **Step 3: Write minimal implementation**

Add this route in `biofield_local_app.py` immediately after the `author_e4l_refresh` route (the block ending around line 290). It imports the helpers lazily so the test's `monkeypatch.setattr(RI, "synthesize_reveal_layers", ...)` on the module is honored:

```python
    @app.route("/author/<test_id>/e4l/import-reveal", methods=["POST"])
    def author_import_reveal(test_id):
        """Import the client's recent (<7d) E4L reveal layers + remedies as
        needs-review causal-chain rows. Appends only after an explicit force when the
        session already has rows. Synthesis runs in-process (PHI stays local)."""
        import datetime as _dt
        from dashboard import biofield_reveal_import as _ri
        force = bool((request.get_json(silent=True) or {}).get("force"))
        with sqlite3.connect(db_path) as cx:
            rep = _report_for(cx, test_id)
            email = ((rep.get("client") or {}).get("email") or "").strip()
            if not email:
                return {"ok": False, "reason": "No client selected yet"}
            res = _ri.synthesize_reveal_layers(email, today=_dt.date.today().isoformat())
            if not res.get("found"):
                return {"ok": False, "reason": "No E4L scan on file"}
            if not res.get("fresh"):
                return {"ok": False,
                        "reason": f"Latest scan is {res.get('days_ago')} days old "
                                  "— refresh to import"}
            existing = len(rep.get("layers") or [])
            if existing and not force:
                return {"ok": False, "needs_confirm": True, "existing": existing}
            imported = _ri.import_layers_to_test(cx, test_id, res.get("layers") or [])
        return {"ok": True, "imported": imported}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_import_reveal_routes.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add biofield_local_app.py tests/test_biofield_import_reveal_routes.py
git commit -m "feat(biofield): /author/<id>/e4l/import-reveal route — 7-day gate + append-confirm"
```

---

### Task 4: Panel button + `importReveal()` JS

**Files:**
- Modify: `dashboard/biofield_report_html.py` — `render_e4l_panel(ctx)` (~line 351-398) to add the button; `render_author_html(...)` script block (~line 200-330) to add the `importReveal()` function.
- Test: `tests/test_biofield_import_reveal_button.py`

**Interfaces:**
- Consumes: `ctx` from `scan_context` (has `found`, `days_ago`); the author page's `__TID__`-templated `post()` helper + `location.reload()` convention.
- Produces: an "Import Reveal → Causal Chain" button in the E4L panel, active only when `ctx.found and ctx.days_ago is not None and ctx.days_ago < 7`; otherwise a disabled button with the reason (or nothing when no scan). The button calls `importReveal()`, which POSTs to `/author/__TID__/e4l/import-reveal`, handles `needs_confirm` with a JS `confirm()` re-POST (`force:true`), then reloads.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_biofield_import_reveal_button.py
"""The E4L panel offers an Import-Reveal button only when a scan is < 7 days old;
the author page defines the importReveal() handler."""
from dashboard.biofield_report_html import render_author_html, render_e4l_panel


def _ctx(found, days):
    return {"status": "fresh" if found else "none", "found": found,
            "scan_id": 900 if found else None, "scan_date": "2026-06-22",
            "days_ago": days, "fresh": True, "window_days": 14,
            "message": "Recent E4L scan", "findings": [],
            "infoceuticals": [], "stresses": []}


def test_button_active_when_scan_under_7_days():
    html = render_e4l_panel(_ctx(True, 3))
    assert "Import Reveal" in html
    assert "onclick=importReveal()" in html
    assert "disabled" not in html.split("Import Reveal")[0][-40:]  # button not disabled


def test_button_disabled_when_scan_stale():
    html = render_e4l_panel(_ctx(True, 12))
    assert "Import Reveal" in html
    assert "12 days old" in html
    assert "disabled" in html


def test_no_button_when_no_scan():
    html = render_e4l_panel(_ctx(False, None))
    assert "Import Reveal" not in html


def test_author_page_defines_import_reveal_handler():
    rep = {"test_id": "a7", "client": {"name": "Jane", "email": "jane@x.com"},
           "date": "2026-06-25", "layers": [], "schedule": []}
    html = render_author_html(rep, [], "")
    assert "function importReveal()" in html
    assert "/author/a7/e4l/import-reveal" in html
    assert "needs_confirm" in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_import_reveal_button.py -v`
Expected: FAIL — "Import Reveal" / "function importReveal()" not found

- [ ] **Step 3: Write minimal implementation**

In `render_e4l_panel(ctx)`, replace the `check = (...)` button-row block (the one with `Check E4L now`) so it also renders the import button per the gate. Locate:

```python
    check = ("<div class=btnrow style='margin-top:8px'>"
             "<button class='btn ghost' onclick=checkE4L()>Check E4L now</button>"
             "<span id=e4lchk class=food></span></div>")
```

Replace with:

```python
    days = ctx.get("days_ago")
    if ctx.get("found") and days is not None and days < 7:
        imp = "<button class='btn' onclick=importReveal()>Import Reveal &rarr; Causal Chain</button>"
    elif ctx.get("found"):
        imp = (f"<button class='btn' disabled title='Refresh to a scan under 7 days old'>"
               f"Import Reveal &rarr; Causal Chain</button>"
               f"<span class=food>scan is {_e(str(days))} days old</span>")
    else:
        imp = ""
    check = ("<div class=btnrow style='margin-top:8px'>"
             "<button class='btn ghost' onclick=checkE4L()>Check E4L now</button>"
             f"{imp}"
             "<span id=e4lchk class=food></span></div>")
```

NOTE: `test_button_disabled_when_scan_stale` asserts `"12 days old"` appears — the disabled branch emits `scan is 12 days old`. Good. The active branch has no `disabled` attribute.

Then in `render_author_html(...)`, add the `importReveal()` function alongside the existing `checkE4L`/`confirmRow` JS (these use the `__TID__` placeholder and a `post()` helper that returns parsed JSON, plus `location.reload()`). Insert after the `checkE4L` function definition:

```javascript
async function importReveal(){
  var j=await post('/author/__TID__/e4l/import-reveal',{});
  if(j && j.needs_confirm){
    if(!confirm('This session already has '+j.existing+' rows — add the reveal layers anyway?')) return;
    j=await post('/author/__TID__/e4l/import-reveal',{force:true});
  }
  if(j && j.ok){ location.reload(); }
  else { astat((j&&j.reason)||'Import failed.'); }
}
```

(`astat(...)` is the existing author-status helper used by `saveRow`; `__TID__` is replaced with the real test id when the page renders.)

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_import_reveal_button.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Run the whole biofield suite + commit**

```bash
~/.venvs/deploy-chat311/bin/python -m pytest tests/ -k biofield -q
git add dashboard/biofield_report_html.py tests/test_biofield_import_reveal_button.py
git commit -m "feat(biofield): Import-Reveal panel button + importReveal() handler"
```

---

## Self-Review

**Spec coverage:**
- Fresh local synthesis source → Task 1 (`_run_synthesis` mirrors `e4l-reveal-push.py`). ✓
- 7-day gate → Task 1 (`fresh = days_ago < 7`), Task 3 (route rejects stale), Task 4 (button gate). ✓
- Layer → chain-row mapping (layer/head/most_affected/remedy + dosing, confirmed=0) → Task 2. ✓
- Append-after-confirm on existing rows → Task 3 (`needs_confirm` / `force`). ✓
- Button placement in the E4L panel + disabled-with-reason → Task 4. ✓
- Tests run offline (runner injection / monkeypatch) → all tasks. ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code. ✓

**Type consistency:** `synthesize_reveal_layers` returns the same `layers` dict shape consumed by `import_layers_to_test` and the route; `most_affected`/`remedy_name`/`n`/`title` names are consistent across Tasks 1–4. The `authored_report` remedy-filter caveat is called out in Task 2's test. ✓

## Verification (manual, after all tasks)

Run the local app under Doppler and confirm end-to-end against the real pipeline:

```bash
cd ~/deploy-chat && doppler run -p remedy-match -c prd -- python3 biofield_local_app.py
```

Open `http://127.0.0.1:8011`, open a test for a client with a scan < 7 days old, confirm the "Import Reveal → Causal Chain" button appears, click it, and verify needs-review (unconfirmed) chain rows appear with remedies + dosing.
