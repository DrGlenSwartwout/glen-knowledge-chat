# Biofield Balancing B4 — Minimal-Remedy Consolidation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Suggest the fewest remedies (greedy set-cover over the B1 coverage map) that cover the most active, required, scan-code stresses — read-only.

**Architecture:** A pure `minimal_remedies(active_codes, coverage)` greedy set-cover; a `suggest_minimal_remedies(cx, tid, chain_rows)` store helper that selects the active/required/scan stresses, reads the coverage table, runs the algorithm, and maps codes→labels; a read-only GET route; a suggestion panel + button.

**Tech Stack:** Python 3.11, Flask, sqlite3, pytest. Local-only tool (`biofield_local_app.py`).

## Global Constraints

- Local-only; no feature flag, no prod deploy. Purely additive; no existing behavior changes.
- Suggest-only: NO chain writes. The route is read-only (GET).
- Cover target = stresses that are active AND `balance=='required'` AND `source=='scan'` AND have a non-empty `code`. (Voice/tag/optional/ER-MR have no coverage map and are excluded.)
- Greedy set-cover with a deterministic tie-break: pick the remedy covering the most still-uncovered codes; ties broken by remedy name ascending.
- Run tests: `cd /tmp/wt-deploy-chat-82bd74c2 && ~/.venvs/deploy-chat311/bin/python -m pytest <path> -v` (pure-module; no Doppler). B1/B2/B3a biofield tests must stay green.

---

### Task 1: `minimal_remedies` — greedy set-cover (pure)

**Files:**
- Create: `dashboard/biofield_setcover.py`
- Test: `tests/test_biofield_setcover.py`

**Interfaces:**
- Produces: `minimal_remedies(active_codes, coverage) -> {"picks": [{"remedy": str, "covers": [code, ...]}], "uncovered": [code, ...]}`. `active_codes` iterable of codes; `coverage` = `{remedy: set(codes)}`. Greedy max-coverage, tie-break remedy-name ascending; `covers` = codes that pick newly covers (sorted); `uncovered` = codes no candidate reaches (sorted). Empty active → `{"picks": [], "uncovered": []}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_biofield_setcover.py
from dashboard.biofield_setcover import minimal_remedies


def test_greedy_picks_max_then_alpha_tiebreak():
    cov = {"Zeta": {"a", "b"}, "Alpha": {"a", "b"}, "Beta": {"c"}}
    res = minimal_remedies({"a", "b", "c"}, cov)
    # Alpha and Zeta both cover 2 -> alphabetical tie-break picks Alpha first
    assert res["picks"][0] == {"remedy": "Alpha", "covers": ["a", "b"]}
    assert res["picks"][1] == {"remedy": "Beta", "covers": ["c"]}
    assert res["uncovered"] == []


def test_uncovered_codes_reported():
    res = minimal_remedies({"a", "x"}, {"R": {"a"}})
    assert res["picks"] == [{"remedy": "R", "covers": ["a"]}]
    assert res["uncovered"] == ["x"]


def test_subsumed_remedy_not_picked():
    res = minimal_remedies({"a", "b"}, {"Big": {"a", "b"}, "Small": {"a"}})
    assert [p["remedy"] for p in res["picks"]] == ["Big"]
    assert res["uncovered"] == []


def test_coverage_restricted_to_active():
    # remedy covers extra codes not in active -> covers only the active ones
    res = minimal_remedies({"a"}, {"R": {"a", "b", "c"}})
    assert res["picks"] == [{"remedy": "R", "covers": ["a"]}]


def test_empty_active():
    assert minimal_remedies(set(), {"R": {"a"}}) == {"picks": [], "uncovered": []}
```

- [ ] **Step 2: Run** → FAIL (module missing).

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_setcover.py -v`

- [ ] **Step 3: Implement**

```python
# dashboard/biofield_setcover.py
"""Greedy set-cover for the Biofield Intake balancing loop (B4): the fewest
remedies that cover the most active stress codes. Pure; deterministic."""


def minimal_remedies(active_codes, coverage):
    """active_codes: iterable of codes to cover. coverage: {remedy: set(codes)}.
    Greedy: repeatedly pick the remedy covering the most still-uncovered codes,
    tie broken by remedy name ascending. Returns picks + the uncovered remainder."""
    remaining = set(active_codes or [])
    # Restrict each remedy to the active codes; drop remedies that cover nothing.
    cov = {}
    for remedy, codes in (coverage or {}).items():
        c = set(codes) & remaining
        if c:
            cov[remedy] = c
    picks = []
    while remaining:
        best, best_n = None, 0
        for remedy in sorted(cov):            # alphabetical -> deterministic tie-break
            n = len(cov[remedy] & remaining)
            if n > best_n:
                best, best_n = remedy, n
        if not best:                          # nothing left covers a remaining code
            break
        covered = sorted(cov[best] & remaining)
        picks.append({"remedy": best, "covers": covered})
        remaining -= cov[best]
    return {"picks": picks, "uncovered": sorted(remaining)}
```

- [ ] **Step 4: Run** → PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add dashboard/biofield_setcover.py tests/test_biofield_setcover.py
git commit -m "feat(biofield-b4): minimal_remedies greedy set-cover"
```

---

### Task 2: `suggest_minimal_remedies` store helper

**Files:**
- Modify: `dashboard/biofield_stress.py`
- Test: `tests/test_biofield_suggest_remedies.py`

**Interfaces:**
- Consumes: `minimal_remedies` (Task 1); `list_stresses` (B1/B2); `_num` (B1); the `biofield_auth_remedy_coverage` table.
- Produces: `suggest_minimal_remedies(cx, tid, chain_rows) -> {"picks": [{"remedy": str, "covers": [label, ...]}], "uncovered": [label, ...]}` — selects active+required+scan stress codes from `list_stresses(cx, tid, chain_rows)`, builds coverage from the table, runs `minimal_remedies`, maps codes→labels.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_biofield_suggest_remedies.py
import sqlite3
from dashboard.biofield_stress import init_stress_tables, seed_from_scan, suggest_minimal_remedies

_FIND = [{"code": "ED1", "name": "Membrane"}, {"code": "ES3", "name": "Lymph"},
         {"code": "MR2", "name": "Calm Mind"}]            # MR2 not covered -> optional
_COV = {"neuro magnesium": {"ED1", "ES3"}}


def _seeded(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "c.db"))
    init_stress_tables(cx)
    seed_from_scan(cx, "a5", _FIND, _COV)                 # ED1/ES3 required, MR2 optional
    return cx


def test_suggests_minimal_set_with_labels(tmp_path):
    cx = _seeded(tmp_path)
    res = suggest_minimal_remedies(cx, "a5", [])          # nothing on the chain yet
    assert res["picks"] == [{"remedy": "neuro magnesium", "covers": ["Membrane", "Lymph"]}]
    assert res["uncovered"] == []                         # MR2 is optional -> not targeted


def test_excludes_already_balanced(tmp_path):
    cx = _seeded(tmp_path)
    # a chain row whose remedy covers ED1+ES3 -> both balanced -> nothing left to suggest
    res = suggest_minimal_remedies(cx, "a5", [{"head": "x", "remedy": "Neuro Magnesium"}])
    assert res["picks"] == [] and res["uncovered"] == []


def test_uncovered_required_with_no_remedy(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "c.db"))
    init_stress_tables(cx)
    # ED9 is required (in coverage union) but its only remedy is removed from the map:
    seed_from_scan(cx, "a5", [{"code": "ED1", "name": "Membrane"}], {"tonic": {"ED1"}})
    # break coverage so ED1 has no remedy (simulate a dropped remedy)
    cx.execute("DELETE FROM biofield_auth_remedy_coverage WHERE test_id=5")
    # re-mark ED1 required by re-seeding label only is unnecessary; ED1 stays required from first seed
    res = suggest_minimal_remedies(cx, "a5", [])
    assert res["picks"] == [] and res["uncovered"] == ["Membrane"]
```

NOTE on the third test: `seed_from_scan` makes a code `required` iff it is in the coverage union AT SEED TIME. ED1 is required from the first seed (coverage had `tonic:{ED1}`); deleting the coverage rows afterward leaves ED1 required but with no covering remedy, so it lands in `uncovered`. This exercises the required-but-uncoverable path.

- [ ] **Step 2: Run** → FAIL (`suggest_minimal_remedies` undefined).

- [ ] **Step 3: Implement** — append to `dashboard/biofield_stress.py`:

```python
def suggest_minimal_remedies(cx, tid, chain_rows):
    """Fewest remedies covering the active+required+scan stresses. Returns picks
    (remedy + covered stress LABELS) and the uncovered labels. Read-only."""
    from dashboard.biofield_setcover import minimal_remedies
    data = list_stresses(cx, tid, chain_rows)
    code_label, active_codes = {}, set()
    for s in data["active"]:
        code = s.get("code") or ""
        if code and s.get("balance") == "required" and s.get("source") == "scan":
            active_codes.add(code)
            code_label[code] = s.get("label") or code
    coverage = {}
    for remedy, code in cx.execute(
            "SELECT remedy, code FROM biofield_auth_remedy_coverage WHERE test_id=?",
            (_num(tid),)).fetchall():
        coverage.setdefault(remedy, set()).add(code)
    res = minimal_remedies(active_codes, coverage)
    picks = [{"remedy": p["remedy"], "covers": [code_label.get(c, c) for c in p["covers"]]}
             for p in res["picks"]]
    uncovered = [code_label.get(c, c) for c in res["uncovered"]]
    return {"picks": picks, "uncovered": uncovered}
```

- [ ] **Step 4: Run** → `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_suggest_remedies.py tests/test_biofield_stress_seed.py tests/test_biofield_stress_derive.py -v` → PASS (B1 seed/derive green).

- [ ] **Step 5: Commit**

```bash
git add dashboard/biofield_stress.py tests/test_biofield_suggest_remedies.py
git commit -m "feat(biofield-b4): suggest_minimal_remedies store helper"
```

---

### Task 3: `GET /author/<id>/suggest-remedies` route

**Files:**
- Modify: `biofield_local_app.py`
- Test: `tests/test_biofield_suggest_remedies_route.py`

**Interfaces:**
- Consumes: `suggest_minimal_remedies` (Task 2); `_report_for`, `db_path`; `render_suggest_panel` (Task 4 — add a minimal stub now).
- Produces: `GET /author/<test_id>/suggest-remedies` → `{"picks": [...], "uncovered": [...], "html": ...}` computed from the live chain.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_biofield_suggest_remedies_route.py
import sqlite3
import pytest
from biofield_local_app import create_app
from dashboard.biofield_stress import init_stress_tables, seed_from_scan


@pytest.fixture(autouse=True)
def _no_gate(monkeypatch):
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)


_NONE = {"status": "none", "found": False, "findings": [], "days_ago": None, "fresh": False}


def _seed(db, tid_num):
    cx = sqlite3.connect(db)
    init_stress_tables(cx)
    seed_from_scan(cx, "a" + str(tid_num),
                   [{"code": "ED1", "name": "Membrane"}, {"code": "ES3", "name": "Lymph"}],
                   {"neuro magnesium": {"ED1", "ES3"}})
    cx.close()


def test_route_returns_suggestion(tmp_path):
    db = str(tmp_path / "c.db")
    client = create_app(db, scan_lookup=lambda e: _NONE).test_client()
    tid = client.post("/author/new").headers["Location"].rstrip("/").split("/")[-1]
    _seed(db, int(tid.lstrip("a")))
    j = client.get(f"/author/{tid}/suggest-remedies").get_json()
    assert j["picks"] == [{"remedy": "neuro magnesium", "covers": ["Membrane", "Lymph"]}]
    assert j["uncovered"] == [] and "html" in j


def test_route_reflects_live_chain(tmp_path):
    db = str(tmp_path / "c.db")
    client = create_app(db, scan_lookup=lambda e: _NONE).test_client()
    tid = client.post("/author/new").headers["Location"].rstrip("/").split("/")[-1]
    _seed(db, int(tid.lstrip("a")))
    # put the covering remedy on the chain -> stresses balanced -> nothing to suggest
    client.post(f"/author/{tid}/row", json={"layer": 1, "head": "x", "remedy": "Neuro Magnesium"})
    j = client.get(f"/author/{tid}/suggest-remedies").get_json()
    assert j["picks"] == [] and j["uncovered"] == []
```

- [ ] **Step 2: Run** → FAIL (route 404 / `render_suggest_panel` import error).

- [ ] **Step 3: Implement** — in `biofield_local_app.py`:

(a) Add `render_suggest_panel` to the existing `from dashboard.biofield_report_html import (...)` import list. To keep this task importable before Task 4, add a minimal stub in `dashboard/biofield_report_html.py` now (Task 4 replaces it):

```python
def render_suggest_panel(data):
    return ""
```

(b) Add the route near `author_stresses`:

```python
    @app.route("/author/<test_id>/suggest-remedies")
    def author_suggest_remedies(test_id):
        from dashboard import biofield_stress as _st
        with sqlite3.connect(db_path) as cx:
            rep = _report_for(cx, test_id)
            chain_rows = [{"head": l.get("head"), "remedy": l.get("remedy")}
                          for l in (rep.get("layers") or [])]
            data = _st.suggest_minimal_remedies(cx, test_id, chain_rows)
        return {"picks": data["picks"], "uncovered": data["uncovered"],
                "html": render_suggest_panel(data)}
```

- [ ] **Step 4: Run** → `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_suggest_remedies_route.py tests/test_biofield_stress_routes.py -v` → PASS (B1 stress routes green).

- [ ] **Step 5: Commit**

```bash
git add biofield_local_app.py dashboard/biofield_report_html.py tests/test_biofield_suggest_remedies_route.py
git commit -m "feat(biofield-b4): suggest-remedies route (+ render_suggest_panel stub)"
```

---

### Task 4: UI — `render_suggest_panel` + button

**Files:**
- Modify: `dashboard/biofield_report_html.py`
- Test: `tests/test_biofield_suggest_panel.py`

**Interfaces:**
- Consumes: the `suggest_minimal_remedies` output shape (`picks`/`uncovered` of labels); existing `__TID__`/`_e()` conventions.
- Produces: full `render_suggest_panel(data)` (replaces the Task-3 stub); a "Suggest minimal remedies" button + `<div id=suggestpanel></div>` in the author page; `suggestRemedies()` JS that GETs the route and swaps the panel HTML.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_biofield_suggest_panel.py
from dashboard.biofield_report_html import render_suggest_panel, render_author_html


def test_panel_renders_picks_and_count():
    data = {"picks": [{"remedy": "Neuro Magnesium", "covers": ["Membrane", "Mitochondria"]}],
            "uncovered": ["Lymph"]}
    h = render_suggest_panel(data)
    assert "Neuro Magnesium" in h and "Membrane, Mitochondria" in h
    assert ">2<" in h or "(2)" in h           # coverage count shown
    assert "Lymph" in h                        # uncovered listed


def test_panel_empty_state():
    h = render_suggest_panel({"picks": [], "uncovered": []})
    assert "No active required stresses" in h


def test_author_page_has_button_and_handler():
    rep = {"test_id": "a7", "client": {"name": "J", "email": "j@x.com"}, "date": "",
           "layers": [], "schedule": []}
    h = render_author_html(rep, [], "")
    assert "Suggest minimal remedies" in h
    assert "function suggestRemedies" in h
    assert "/author/a7/suggest-remedies" in h
    assert "id=suggestpanel" in h
```

- [ ] **Step 2: Run** → FAIL (stub returns "", no button/handler).

- [ ] **Step 3: Implement** — in `dashboard/biofield_report_html.py`:

(a) Replace the `render_suggest_panel` stub with:

```python
def render_suggest_panel(data):
    data = data or {}
    picks = data.get("picks") or []
    unc = data.get("uncovered") or []
    if not picks and not unc:
        return "<div class=card><div class=food>No active required stresses to consolidate.</div></div>"
    items = ""
    for p in picks:
        covers = p.get("covers") or []
        items += (f"<li><b>{_e(p.get('remedy') or '')}</b> &rarr; covers "
                  f"{_e(', '.join(covers))} <span class=pill>{len(covers)}</span></li>")
    body = f"<ol style='margin:4px 0 0;padding-left:20px'>{items}</ol>" if items else ""
    unc_html = (f"<div class=food style='margin-top:6px'>No scan remedy for: "
                f"{_e(', '.join(unc))}</div>" if unc else "")
    return ("<div class=card><div class=food style='text-transform:uppercase;font-size:11px;"
            f"letter-spacing:.08em'>Minimal remedy set</div>{body}{unc_html}</div>")
```

(b) In `_AUTHOR_JS`, add (next to `mineProfile`):

```javascript
async function suggestRemedies(){
 try{var j=await (await fetch('/author/__TID__/suggest-remedies')).json();
  document.getElementById('suggestpanel').innerHTML=j.html}
 catch(e){document.getElementById('suggestpanel').innerHTML=''}}
```

(c) Add the button + panel container next to the stress panel. Find the existing "Mine profile → stresses" button row (added in B3a) above `<div id=stresspanel></div>` and add a suggest button + container after the stress panel. Concretely, replace the existing `<div id=stresspanel></div>` occurrence with:

```python
        "<div id=stresspanel></div>"
        "<div class=btnrow style='margin:6px 0'>"
        "<button class='btn ghost' onclick=suggestRemedies()>Suggest minimal remedies</button>"
        "</div>"
        "<div id=suggestpanel></div>"
```
(Preserve everything else around it; only append the button row + `#suggestpanel` after the existing stress panel div.)

- [ ] **Step 4: Run** → `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_suggest_panel.py tests/test_biofield_author_html.py tests/test_biofield_stress_panel.py tests/test_biofield_mine_profile_button.py -v` → PASS.

- [ ] **Step 5: Run the B4 + adjacent suite + commit**

```bash
~/.venvs/deploy-chat311/bin/python -m pytest \
  tests/test_biofield_setcover.py tests/test_biofield_suggest_remedies.py \
  tests/test_biofield_suggest_remedies_route.py tests/test_biofield_suggest_panel.py \
  tests/test_biofield_stress_seed.py tests/test_biofield_stress_derive.py \
  tests/test_biofield_stress_routes.py tests/test_biofield_author_html.py \
  tests/test_biofield_stress_panel.py -q
git add dashboard/biofield_report_html.py tests/test_biofield_suggest_panel.py
git commit -m "feat(biofield-b4): suggest-remedies panel + button"
```

---

## Self-Review

**Spec coverage:**
- Greedy set-cover w/ deterministic tie-break → Task 1. ✓
- Active+required+scan target + coverage from table + code→label → Task 2. ✓
- Read-only GET route reflecting the live chain → Task 3. ✓
- Suggest panel (picks + counts + uncovered + empty state) + button → Task 4. ✓
- Suggest-only / no chain writes → Tasks 3 & 4 (GET, no writes). ✓

**Placeholder scan:** No TBDs; every code step complete. Task 3 explicitly stubs `render_suggest_panel` so the route is importable before Task 4 fleshes it out (same pattern as B1/B2).

**Type consistency:** `minimal_remedies(active_codes, coverage) -> {picks:[{remedy,covers}],uncovered:[]}` (T1) consumed by T2; `suggest_minimal_remedies(cx,tid,chain_rows) -> {picks:[{remedy,covers:labels}],uncovered:labels}` (T2) consumed by T3/T4; `render_suggest_panel(data)` (T3 stub → T4) consumed by T3 route. Consistent.

## Verification (manual, after all tasks)

```bash
cd ~/deploy-chat && doppler run -p remedy-match -c prd -- python3 biofield_local_app.py
```
Open a test with a recent scan, click "Suggest minimal remedies": the panel lists the fewest remedies covering the active required stresses (with counts) + any uncovered ones. Put one suggested remedy on the chain and re-click: its stresses drop out of the suggestion.
