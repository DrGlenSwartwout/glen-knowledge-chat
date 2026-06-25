# Biofield Balancing B1 — Stress Engine + UI — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give each local Biofield Intake session a scan-seeded master stress list, a remedy↔stress coverage map, recompute-on-read auto-balancing, two-zone chain ordering (live on top / unbalanced scan trailing), and insert-at-N layer reorder.

**Architecture:** A new pure module `dashboard/biofield_stress.py` holds the stress store + derived-balance logic. SP-A's `dashboard/biofield_reveal_import.py` is extended to expose each layer's `codes` and a `build_coverage` helper. `dashboard/biofield_authoring.py` gains an `origin` column, two-zone `ordered_chain`, and `reorder_chain`. `biofield_local_app.py` wires a seed hook + stress routes + reorder. `dashboard/biofield_report_html.py` adds the stress panel, two-zone chain rendering, and a report listing.

**Tech Stack:** Python 3.11, Flask, sqlite3, pytest. Local-only tool (`biofield_local_app.py`, `127.0.0.1:8011`).

## Global Constraints

- Local-only; PHI stays on Glen's Mac. No feature flag, no prod deploy.
- Synthesis runs in-process via the vault pipeline, imported lazily; tests stub it and stay offline.
- Auto-balance is **derived, never stored**: `balanced(stress) = manual_balanced OR code ∈ covered_codes(test)`, where `covered_codes = ⋃ coverage[remedy_lower]` over remedies on the test's chain rows. Off-scan remedies clear nothing.
- Unbalanced scan layer = `origin='scan' AND confirmed=0`. Two zones: top = everything else (display 1..n, manual insert-at-N reorder); bottom = unbalanced scan (display n+1..k, trailing).
- `required` = stress code present in the coverage map (synthesis is balancing it); `optional` = all other scan findings.
- Re-seed is idempotent and **preserves `manual_balanced`**; coverage is fully rebuilt.
- Run tests: `cd /tmp/wt-deploy-chat-82bd74c2 && ~/.venvs/deploy-chat311/bin/python -m pytest <path> -v` (pure-module tests; no Doppler). SP-A's existing tests (`tests/test_biofield_reveal_import.py`, `tests/test_biofield_import_reveal_routes.py`, `tests/test_biofield_import_reveal_button.py`) must stay green.

---

### Task 1: `origin` column + `add_chain_row(origin=)` + import sets `origin='scan'`

**Files:**
- Modify: `dashboard/biofield_authoring.py` (`init_auth_tables` ~line 28, `add_chain_row` ~line 71)
- Modify: `dashboard/biofield_reveal_import.py` (`import_layers_to_test`)
- Test: `tests/test_biofield_chain_origin.py`

**Interfaces:**
- Produces: `add_chain_row(cx, tid, layer, head, most_affected, remedy, dosage="", frequency="", timing="", confirmed=1, origin="live") -> rowid`; `biofield_auth_chain` has an `origin TEXT NOT NULL DEFAULT 'live'` column; `import_layers_to_test` creates rows with `origin='scan'`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_biofield_chain_origin.py
import sqlite3
from dashboard.biofield_authoring import add_chain_row, init_auth_tables, create_test
from dashboard.biofield_reveal_import import import_layers_to_test


def _col_names(cx, table):
    return {r[1] for r in cx.execute(f"PRAGMA table_info({table})").fetchall()}


def test_chain_has_origin_column_default_live(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "c.db"))
    init_auth_tables(cx)
    assert "origin" in _col_names(cx, "biofield_auth_chain")
    tid = create_test(cx, "J", "j@x.com", "2026-06-25")
    rid = add_chain_row(cx, tid, 1, "Head", "Most", "Remedy")
    row = cx.execute("SELECT origin FROM biofield_auth_chain WHERE id=?", (rid,)).fetchone()
    assert row[0] == "live"


def test_add_chain_row_accepts_origin(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "c.db"))
    init_auth_tables(cx)
    tid = create_test(cx, "J", "j@x.com", "2026-06-25")
    rid = add_chain_row(cx, tid, 1, "H", "M", "R", origin="scan")
    assert cx.execute("SELECT origin FROM biofield_auth_chain WHERE id=?", (rid,)).fetchone()[0] == "scan"


def test_import_marks_rows_scan(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "c.db"))
    init_auth_tables(cx)
    tid = create_test(cx, "J", "j@x.com", "2026-06-25")
    import_layers_to_test(cx, tid, [{"n": 1, "title": "Ox", "most_affected": "A", "remedy_name": "Neuro Magnesium"}])
    origins = [r[0] for r in cx.execute("SELECT origin FROM biofield_auth_chain WHERE test_id=?", (int(str(tid).lstrip('a')),)).fetchall()]
    assert origins == ["scan"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_chain_origin.py -v`
Expected: FAIL — no `origin` column / `add_chain_row` has no `origin` kwarg.

- [ ] **Step 3: Write minimal implementation**

In `dashboard/biofield_authoring.py` `init_auth_tables`, add the column to the CREATE and a guarded migration next to the existing `confirmed` one:

```python
    cx.execute("""CREATE TABLE IF NOT EXISTS biofield_auth_chain(
        id INTEGER PRIMARY KEY AUTOINCREMENT, test_id INTEGER, layer INTEGER,
        head TEXT, most_affected TEXT, remedy TEXT, dosage TEXT, frequency TEXT,
        timing TEXT, sort_seq INTEGER, created_at TEXT, confirmed INTEGER DEFAULT 1,
        origin TEXT NOT NULL DEFAULT 'live')""")
    try:
        cx.execute("ALTER TABLE biofield_auth_chain ADD COLUMN confirmed INTEGER DEFAULT 1")
    except Exception:
        pass
    try:
        cx.execute("ALTER TABLE biofield_auth_chain ADD COLUMN origin TEXT NOT NULL DEFAULT 'live'")
    except Exception:
        pass
    cx.commit()
```

Update `add_chain_row` signature + INSERT:

```python
def add_chain_row(cx, tid, layer, head, most_affected, remedy,
                  dosage="", frequency="", timing="", confirmed=1, origin="live"):
    init_auth_tables(cx)
    cur = cx.execute(
        "INSERT INTO biofield_auth_chain(test_id,layer,head,most_affected,remedy,"
        "dosage,frequency,timing,sort_seq,created_at,confirmed,origin) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
        (_num(tid), layer, (head or "").strip(), (most_affected or "").strip(),
         (remedy or "").strip(), dosage or "", frequency or "", timing or "", 0, _now(),
         1 if confirmed else 0, (origin or "live")))
    cx.commit()
    return cur.lastrowid
```

In `dashboard/biofield_reveal_import.py` `import_layers_to_test`, add `origin="scan"` to the `add_chain_row(...)` call.

- [ ] **Step 4: Run tests** — `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_chain_origin.py tests/test_biofield_reveal_import.py -v` → PASS (SP-A import tests still green).

- [ ] **Step 5: Commit**

```bash
git add dashboard/biofield_authoring.py dashboard/biofield_reveal_import.py tests/test_biofield_chain_origin.py
git commit -m "feat(biofield-b1): chain origin column; import marks rows scan"
```

---

### Task 2: `synthesize_reveal_layers` returns `codes` + `build_coverage`

**Files:**
- Modify: `dashboard/biofield_reveal_import.py`
- Test: `tests/test_biofield_coverage.py`

**Interfaces:**
- Produces: each mapped layer dict gains `"codes": list[str]` (the layer's `patterns`); `build_coverage(layers) -> dict[str, set[str]]` maps `remedy_lower -> {codes}` (empty/falsy remedy skipped; codes unioned across repeated remedies).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_biofield_coverage.py
from dashboard.biofield_reveal_import import synthesize_reveal_layers, build_coverage

_RAW = [
    {"n": 1, "title": "Ox", "summary": "", "patterns": ["ED1", "ED2"],
     "pattern_labels": ["Mem", "Mito"], "remedy": {"name": "Neuro Magnesium"}},
    {"n": 2, "title": "Terr", "summary": "", "patterns": ["ES3"],
     "pattern_labels": ["Lymph"], "remedy": {"name": "neuro magnesium"}},
    {"n": 3, "title": "X", "summary": "", "patterns": ["MB1"],
     "pattern_labels": ["B"], "remedy": None},
]


def test_layers_carry_codes():
    res = synthesize_reveal_layers("j@x.com", today="2026-06-25",
                                   runner=lambda *a, **k: ({"scan_id": 1, "scan_date": "2026-06-24"}, _RAW))
    assert res["layers"][0]["codes"] == ["ED1", "ED2"]
    assert res["layers"][2]["codes"] == ["MB1"]


def test_build_coverage_unions_and_lowercases():
    layers = [
        {"codes": ["ED1", "ED2"], "remedy_name": "Neuro Magnesium"},
        {"codes": ["ES3"], "remedy_name": "neuro magnesium"},
        {"codes": ["MB1"], "remedy_name": ""},
    ]
    cov = build_coverage(layers)
    assert cov == {"neuro magnesium": {"ED1", "ED2", "ES3"}}  # unioned, lowercased, empty-remedy skipped
```

- [ ] **Step 2: Run** → FAIL (no `codes` key / no `build_coverage`).

- [ ] **Step 3: Implement** — in `synthesize_reveal_layers`, add `"codes": list(L.get("patterns") or [])` to each appended layer dict. Append:

```python
def build_coverage(layers):
    """Map each remedy (lowercased) to the set of scan stress codes it covers,
    derived from the synthesized layers. Empty-remedy layers are skipped."""
    cov = {}
    for L in layers or []:
        name = (L.get("remedy_name") or "").strip().lower()
        if not name:
            continue
        cov.setdefault(name, set()).update(L.get("codes") or [])
    return cov
```

- [ ] **Step 4: Run** — `pytest tests/test_biofield_coverage.py tests/test_biofield_reveal_import.py -v` → PASS (SP-A tests still green: adding a `codes` key doesn't break their key-specific asserts).

- [ ] **Step 5: Commit** — `git commit -am "feat(biofield-b1): synthesize layers expose codes; build_coverage"` (after `git add`).

---

### Task 3: `ordered_chain` two-zone ordering + `authored_report` uses it

**Files:**
- Modify: `dashboard/biofield_authoring.py` (`authored_report` ~line 303)
- Test: `tests/test_biofield_ordered_chain.py`

**Interfaces:**
- Produces: `ordered_chain(cx, tid) -> list[dict]` — remedy-bearing rows in display order, each with keys `id, layer (DISPLAY 1..k), head, most_affected, remedy, dosage, frequency, timing, confirmed, origin, zone ("top"|"bottom")`. Top zone = NOT (`origin='scan'` and `confirmed=0`), sorted by stored layer then id; bottom zone = unbalanced scan, trailing. `authored_report` returns layers via `ordered_chain` (so `layer` is the display number) and includes `origin`/`zone` per layer.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_biofield_ordered_chain.py
import sqlite3
from dashboard.biofield_authoring import (
    add_chain_row, authored_report, confirm_row, create_test, init_auth_tables, ordered_chain)


def _seed(cx):
    init_auth_tables(cx)
    tid = create_test(cx, "J", "j@x.com", "2026-06-25")
    # two unbalanced scan layers + one live layer added between them
    add_chain_row(cx, tid, 1, "ScanA", "a", "R1", confirmed=0, origin="scan")
    add_chain_row(cx, tid, 2, "ScanB", "b", "R2", confirmed=0, origin="scan")
    live = add_chain_row(cx, tid, 1, "Live", "c", "R3", confirmed=1, origin="live")
    return tid, live


def test_live_on_top_unbalanced_scan_trails_renumbered(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "c.db"))
    tid, _ = _seed(cx)
    rows = ordered_chain(cx, tid)
    assert [r["head"] for r in rows] == ["Live", "ScanA", "ScanB"]
    assert [r["layer"] for r in rows] == [1, 2, 3]          # contiguous display
    assert [r["zone"] for r in rows] == ["top", "bottom", "bottom"]


def test_confirming_scan_layer_promotes_it(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "c.db"))
    tid, _ = _seed(cx)
    sa = cx.execute("SELECT id FROM biofield_auth_chain WHERE head='ScanA'").fetchone()[0]
    confirm_row(cx, sa)
    rows = ordered_chain(cx, tid)
    # ScanA now top-zone (confirmed); ordered by stored layer (Live layer=1, ScanA layer=1 -> tie broken by id: Live first)
    assert [r["zone"] for r in rows] == ["top", "top", "bottom"]
    assert "ScanB" == rows[-1]["head"]


def test_authored_report_uses_display_numbering(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "c.db"))
    tid, _ = _seed(cx)
    rep = authored_report(cx, tid)
    assert [l["layer"] for l in rep["layers"]] == [1, 2, 3]
    assert rep["layers"][0]["head"] == "Live"
```

- [ ] **Step 2: Run** → FAIL (`ordered_chain` undefined).

- [ ] **Step 3: Implement** — add `ordered_chain`, and refactor `authored_report` to build its `layers` from it. Insert before `authored_report`:

```python
def ordered_chain(cx, tid):
    """Remedy-bearing chain rows in display order with two-zone numbering.
    Top zone = live + confirmed rows (manual order); bottom zone = unbalanced
    scan rows (origin='scan' AND confirmed=0), trailing. Display `layer` = 1..k."""
    cx.row_factory = sqlite3.Row
    rows = cx.execute(
        "SELECT id, layer, head, most_affected, remedy, dosage, frequency, timing, "
        "confirmed, origin FROM biofield_auth_chain "
        "WHERE test_id=? AND TRIM(COALESCE(remedy,''))<>''", (_num(tid),)).fetchall()
    def unbalanced_scan(r):
        return (r["origin"] == "scan") and (r["confirmed"] == 0)
    key = lambda r: (r["layer"] is None, r["layer"] if r["layer"] is not None else 0, r["id"])
    top = sorted([r for r in rows if not unbalanced_scan(r)], key=key)
    bottom = sorted([r for r in rows if unbalanced_scan(r)], key=key)
    out = []
    for i, r in enumerate(top + bottom, 1):
        out.append({"id": r["id"], "layer": i, "head": r["head"] or "",
                    "most_affected": r["most_affected"] or "", "remedy": r["remedy"] or "",
                    "dosage": r["dosage"] or "", "frequency": r["frequency"] or "",
                    "timing": r["timing"] or "",
                    "confirmed": 0 if r["confirmed"] == 0 else 1,
                    "origin": r["origin"] or "live",
                    "zone": "bottom" if unbalanced_scan(r) else "top"})
    return out
```

Refactor `authored_report` to use it (replace the inline SELECT + `layers = [...]` comprehension):

```python
def authored_report(cx, tid):
    init_auth_tables(cx)
    cx.row_factory = sqlite3.Row
    t = cx.execute("SELECT * FROM biofield_auth_tests WHERE id=?", (_num(tid),)).fetchone()
    layers = [{**l, "rid": l["id"]} for l in ordered_chain(cx, tid)]
    # Depth-of-penetration tags + reach match-check per layer (Increment 4b)
    for l in layers:
        sd = get_tag(cx, "auth_stress", l["rid"], DEPTH_KEY)
        rd = get_tag(cx, "auth_remedy", l["rid"], DEPTH_KEY)
        l["stress_depth"] = sd
        l["remedy_depth"] = rd
        l["depth_status"] = depth_match(sd, rd)
        l["depth_need"] = depth_label(cx, sd)
    schedule = build_schedule([
        {"name": l["remedy"], "dosage": l["dosage"],
         "frequency": l["frequency"], "timing": l["timing"]} for l in layers])
    return {"test_id": str(tid),
            "client": {"name": (t["name"] if t else "") or "",
                       "email": (t["email"] if t else "") or ""},
            "date": (t["date_test"] if t else "") or "",
            "layers": layers, "schedule": schedule}
```

- [ ] **Step 4: Run** — `pytest tests/test_biofield_ordered_chain.py tests/test_biofield_authoring.py tests/test_biofield_report.py -v` → PASS (existing authoring/report tests still green).

- [ ] **Step 5: Commit** — `git commit -am "feat(biofield-b1): two-zone ordered_chain; authored_report uses it"`.

---

### Task 4: `reorder_chain` (insert-at-N within top zone)

**Files:**
- Modify: `dashboard/biofield_authoring.py`
- Test: `tests/test_biofield_reorder_chain.py`

**Interfaces:**
- Produces: `reorder_chain(cx, tid, rid, new_layer) -> None` — moves the top-zone row `rid` to position `new_layer` (clamped to `[1, len(top)]`) and rewrites stored `layer = 1..n` for top-zone rows in the new order. Bottom-zone (unbalanced scan) rows are untouched. No-op if `rid` is not a top-zone remedy-bearing row.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_biofield_reorder_chain.py
import sqlite3
from dashboard.biofield_authoring import (
    add_chain_row, create_test, init_auth_tables, ordered_chain, reorder_chain)


def test_insert_at_n_renumbers_top_zone(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "c.db"))
    init_auth_tables(cx)
    tid = create_test(cx, "J", "j@x.com", "2026-06-25")
    a = add_chain_row(cx, tid, 1, "A", "", "R1")
    b = add_chain_row(cx, tid, 2, "B", "", "R2")
    c = add_chain_row(cx, tid, 3, "C", "", "R3")
    reorder_chain(cx, tid, c, 1)                 # move C to the top
    assert [r["head"] for r in ordered_chain(cx, tid)] == ["C", "A", "B"]
    assert [r["layer"] for r in ordered_chain(cx, tid)] == [1, 2, 3]


def test_reorder_leaves_unbalanced_scan_trailing(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "c.db"))
    init_auth_tables(cx)
    tid = create_test(cx, "J", "j@x.com", "2026-06-25")
    a = add_chain_row(cx, tid, 1, "A", "", "R1")
    b = add_chain_row(cx, tid, 2, "B", "", "R2")
    s = add_chain_row(cx, tid, 1, "Scan", "", "R3", confirmed=0, origin="scan")
    reorder_chain(cx, tid, b, 1)                 # B to top among the live rows
    rows = ordered_chain(cx, tid)
    assert [r["head"] for r in rows] == ["B", "A", "Scan"]
    assert rows[-1]["zone"] == "bottom"
```

- [ ] **Step 2: Run** → FAIL (`reorder_chain` undefined).

- [ ] **Step 3: Implement**

```python
def reorder_chain(cx, tid, rid, new_layer):
    """Move top-zone row `rid` to position `new_layer` and renumber the top zone
    contiguously. Unbalanced scan rows (bottom zone) are left untouched."""
    top = [l for l in ordered_chain(cx, tid) if l["zone"] == "top"]
    ids = [l["id"] for l in top]
    if rid not in ids:
        return
    ids.remove(rid)
    pos = max(1, min(int(new_layer or 1), len(ids) + 1)) - 1
    ids.insert(pos, rid)
    for i, _id in enumerate(ids, 1):
        cx.execute("UPDATE biofield_auth_chain SET layer=? WHERE id=?", (i, _id))
    cx.commit()
```

- [ ] **Step 4: Run** — `pytest tests/test_biofield_reorder_chain.py -v` → PASS.

- [ ] **Step 5: Commit** — `git commit -am "feat(biofield-b1): reorder_chain insert-at-N within top zone"`.

---

### Task 5: `biofield_stress.py` — tables, seeding, coverage persistence

**Files:**
- Create: `dashboard/biofield_stress.py`
- Test: `tests/test_biofield_stress_seed.py`

**Interfaces:**
- Produces:
  - `init_stress_tables(cx)`
  - `seed_from_scan(cx, tid, findings, coverage) -> dict` — upsert `source='scan'` stress rows (`balance='required'` if code in `coverage`'s value-union, else `'optional'`), preserving `manual_balanced`; rebuild `biofield_auth_remedy_coverage` for the test. `findings` = list of `{"code","name",...}` (from `scan_context`); `coverage` = `{remedy_lower: set(codes)}`. Returns `{"stresses": n, "required": r, "coverage": c}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_biofield_stress_seed.py
import sqlite3
from dashboard.biofield_stress import init_stress_tables, seed_from_scan

_FIND = [{"code": "ED1", "name": "Membrane"}, {"code": "ES3", "name": "Lymph"},
         {"code": "MR2", "name": "Calm Mind"}]
_COV = {"neuro magnesium": {"ED1", "ES3"}}   # MR2 not covered -> optional


def test_seed_assigns_required_optional_and_coverage(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "c.db"))
    init_stress_tables(cx)
    res = seed_from_scan(cx, "a5", _FIND, _COV)
    assert res["stresses"] == 3 and res["required"] == 2
    rows = {r[0]: r[1] for r in cx.execute(
        "SELECT code, balance FROM biofield_auth_stress WHERE test_id=5").fetchall()}
    assert rows == {"ED1": "required", "ES3": "required", "MR2": "optional"}
    cov = cx.execute("SELECT remedy, code FROM biofield_auth_remedy_coverage WHERE test_id=5 ORDER BY code").fetchall()
    assert ("neuro magnesium", "ED1") in cov and ("neuro magnesium", "ES3") in cov


def test_reseed_preserves_manual_balanced_and_rebuilds_coverage(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "c.db"))
    init_stress_tables(cx)
    seed_from_scan(cx, "a5", _FIND, _COV)
    cx.execute("UPDATE biofield_auth_stress SET manual_balanced=1 WHERE code='MR2' AND test_id=5")
    cx.commit()
    seed_from_scan(cx, "a5", _FIND, {"neuro magnesium": {"ED1"}})   # ES3 now optional, coverage shrank
    rows = {r[0]: (r[1], r[2]) for r in cx.execute(
        "SELECT code, balance, manual_balanced FROM biofield_auth_stress WHERE test_id=5").fetchall()}
    assert rows["MR2"][1] == 1                      # manual flag preserved
    assert rows["ES3"][0] == "optional"            # reclassified on re-seed
    assert cx.execute("SELECT COUNT(*) FROM biofield_auth_remedy_coverage WHERE test_id=5").fetchone()[0] == 1
```

- [ ] **Step 2: Run** → FAIL (module missing).

- [ ] **Step 3: Implement**

```python
# dashboard/biofield_stress.py
"""Per-test master stress list + remedy<->stress coverage map for the local
Biofield Intake balancing loop (B1). Pure sqlite; the caller passes a connection.
Balanced state is DERIVED at read time, never stored (see list_stresses)."""
from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _num(tid):
    return int(str(tid).lstrip("a") or 0)


def init_stress_tables(cx):
    cx.execute("""CREATE TABLE IF NOT EXISTS biofield_auth_stress(
        id INTEGER PRIMARY KEY AUTOINCREMENT, test_id INTEGER, code TEXT, label TEXT,
        source TEXT NOT NULL DEFAULT 'scan', balance TEXT NOT NULL DEFAULT 'optional',
        manual_balanced INTEGER NOT NULL DEFAULT 0, created_at TEXT, updated_at TEXT,
        UNIQUE(test_id, source, code))""")
    cx.execute("""CREATE TABLE IF NOT EXISTS biofield_auth_remedy_coverage(
        id INTEGER PRIMARY KEY AUTOINCREMENT, test_id INTEGER, remedy TEXT, code TEXT,
        UNIQUE(test_id, remedy, code))""")
    cx.commit()


def seed_from_scan(cx, tid, findings, coverage):
    init_stress_tables(cx)
    t = _num(tid)
    covered = set()
    for codes in (coverage or {}).values():
        covered |= set(codes)
    now = _now()
    req = 0
    for f in findings or []:
        code = (f.get("code") or "").strip()
        if not code:
            continue
        balance = "required" if code in covered else "optional"
        if balance == "required":
            req += 1
        cx.execute(
            "INSERT INTO biofield_auth_stress(test_id,code,label,source,balance,"
            "manual_balanced,created_at,updated_at) VALUES(?,?,?,'scan',?,0,?,?) "
            "ON CONFLICT(test_id,source,code) DO UPDATE SET "
            "label=excluded.label, balance=excluded.balance, updated_at=excluded.updated_at",
            (t, code, (f.get("name") or code).strip(), balance, now, now))
    cx.execute("DELETE FROM biofield_auth_remedy_coverage WHERE test_id=?", (t,))
    for remedy, codes in (coverage or {}).items():
        for code in codes:
            cx.execute("INSERT OR IGNORE INTO biofield_auth_remedy_coverage(test_id,remedy,code) "
                       "VALUES(?,?,?)", (t, (remedy or "").strip().lower(), code))
    cx.commit()
    n = cx.execute("SELECT COUNT(*) FROM biofield_auth_stress WHERE test_id=? AND source='scan'", (t,)).fetchone()[0]
    c = cx.execute("SELECT COUNT(*) FROM biofield_auth_remedy_coverage WHERE test_id=?", (t,)).fetchone()[0]
    return {"stresses": n, "required": req, "coverage": c}
```

NOTE: the `ON CONFLICT ... DO UPDATE` deliberately omits `manual_balanced`, so a re-seed never resets it.

- [ ] **Step 4: Run** — `pytest tests/test_biofield_stress_seed.py -v` → PASS.

- [ ] **Step 5: Commit** — `git add dashboard/biofield_stress.py tests/test_biofield_stress_seed.py && git commit -m "feat(biofield-b1): stress tables + seed_from_scan"`.

---

### Task 6: `biofield_stress.py` — derived balance read side

**Files:**
- Modify: `dashboard/biofield_stress.py`
- Test: `tests/test_biofield_stress_derive.py`

**Interfaces:**
- Produces:
  - `covered_codes(cx, tid, remedy_names) -> set[str]` — union of coverage codes for the given remedy names (case-insensitive).
  - `list_stresses(cx, tid, chain_remedy_names) -> dict` — `{"active":[...], "balanced":[...]}`; each item `{"id","code","label","source","balance","balanced","balanced_by"}`. `balanced = manual_balanced or code in covered_codes`. `balanced_by` = `"manual"` when manual and not covered, else the first chain remedy whose coverage includes the code (else `""`).
  - `set_manual_balanced(cx, tid, stress_id, value) -> None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_biofield_stress_derive.py
import sqlite3
from dashboard.biofield_stress import (
    init_stress_tables, list_stresses, seed_from_scan, set_manual_balanced)

_FIND = [{"code": "ED1", "name": "Membrane"}, {"code": "ES3", "name": "Lymph"},
         {"code": "MR2", "name": "Calm Mind"}]
_COV = {"neuro magnesium": {"ED1", "ES3"}}


def _seeded(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "c.db"))
    init_stress_tables(cx)
    seed_from_scan(cx, "a5", _FIND, _COV)
    return cx


def test_chain_remedy_balances_its_codes(tmp_path):
    cx = _seeded(tmp_path)
    res = list_stresses(cx, "a5", ["Neuro Magnesium"])    # case-insensitive
    bal = {s["code"]: s["balanced_by"] for s in res["balanced"]}
    assert set(bal) == {"ED1", "ES3"} and bal["ED1"] == "neuro magnesium"
    assert {s["code"] for s in res["active"]} == {"MR2"}


def test_off_scan_remedy_clears_nothing(tmp_path):
    cx = _seeded(tmp_path)
    res = list_stresses(cx, "a5", ["Some Tincture"])
    assert res["balanced"] == [] and len(res["active"]) == 3


def test_manual_overrides_regardless_of_chain(tmp_path):
    cx = _seeded(tmp_path)
    sid = cx.execute("SELECT id FROM biofield_auth_stress WHERE code='MR2' AND test_id=5").fetchone()[0]
    set_manual_balanced(cx, "a5", sid, True)
    res = list_stresses(cx, "a5", [])
    bal = {s["code"]: s["balanced_by"] for s in res["balanced"]}
    assert bal == {"MR2": "manual"}
    set_manual_balanced(cx, "a5", sid, False)
    assert list_stresses(cx, "a5", [])["balanced"] == []
```

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement** — append to `dashboard/biofield_stress.py`:

```python
def covered_codes(cx, tid, remedy_names):
    t = _num(tid)
    names = [(n or "").strip().lower() for n in (remedy_names or []) if (n or "").strip()]
    if not names:
        return set()
    ph = ",".join("?" for _ in names)
    rows = cx.execute(
        f"SELECT DISTINCT code FROM biofield_auth_remedy_coverage "
        f"WHERE test_id=? AND remedy IN ({ph})", (t, *names)).fetchall()
    return {r[0] for r in rows}


def _coverers(cx, tid, code, remedy_names):
    t = _num(tid)
    names = [(n or "").strip().lower() for n in (remedy_names or []) if (n or "").strip()]
    if not names:
        return []
    ph = ",".join("?" for _ in names)
    rows = cx.execute(
        f"SELECT remedy FROM biofield_auth_remedy_coverage "
        f"WHERE test_id=? AND code=? AND remedy IN ({ph})", (t, code, *names)).fetchall()
    return [r[0] for r in rows]


def list_stresses(cx, tid, chain_remedy_names):
    init_stress_tables(cx)
    cx.row_factory = sqlite3.Row
    t = _num(tid)
    covered = covered_codes(cx, tid, chain_remedy_names)
    rows = cx.execute(
        "SELECT id, code, label, source, balance, manual_balanced "
        "FROM biofield_auth_stress WHERE test_id=? ORDER BY "
        "CASE balance WHEN 'required' THEN 0 ELSE 1 END, id", (t,)).fetchall()
    active, balanced = [], []
    for r in rows:
        is_cov = r["code"] in covered
        is_bal = bool(r["manual_balanced"]) or is_cov
        by = ""
        if is_cov:
            cvs = _coverers(cx, tid, r["code"], chain_remedy_names)
            by = cvs[0] if cvs else ""
        elif r["manual_balanced"]:
            by = "manual"
        item = {"id": r["id"], "code": r["code"], "label": r["label"],
                "source": r["source"], "balance": r["balance"],
                "balanced": is_bal, "balanced_by": by}
        (balanced if is_bal else active).append(item)
    return {"active": active, "balanced": balanced}


def set_manual_balanced(cx, tid, stress_id, value):
    cx.execute("UPDATE biofield_auth_stress SET manual_balanced=?, updated_at=? "
               "WHERE id=? AND test_id=?",
               (1 if value else 0, _now(), stress_id, _num(tid)))
    cx.commit()
```

- [ ] **Step 4: Run** — `pytest tests/test_biofield_stress_derive.py tests/test_biofield_stress_seed.py -v` → PASS.

- [ ] **Step 5: Commit** — `git commit -am "feat(biofield-b1): derived balance — covered_codes/list_stresses/set_manual_balanced"`.

---

### Task 7: Routes + seed hook + reorder wiring (`biofield_local_app.py`)

**Files:**
- Modify: `biofield_local_app.py`
- Test: `tests/test_biofield_stress_routes.py`

**Interfaces:**
- Consumes: `biofield_stress.*`, `biofield_reveal_import.synthesize_reveal_layers`/`build_coverage`, `biofield_authoring.reorder_chain`, the in-scope `_report_for`, `scan_lookup`, `db_path`.
- Produces:
  - helper `_seed_stresses(cx, test_id, *, force=False)` — when the test's client has a found scan, run synthesis → coverage + `scan_context`-style findings (from the already-loaded `scan_lookup` ctx `findings`), then `seed_from_scan`. Skips (no re-synthesis) when `force=False` and scan stresses already exist for the test.
  - `GET /author/<test_id>/stresses` -> `{"html": ..., "data": {...}}` (uses current chain remedies from `_report_for`).
  - `POST /author/<test_id>/stress/<int:sid>/balance` body `{value}` -> `{"ok": True}` then caller reloads.
  - header-save and import-reveal routes call `_seed_stresses`.
  - row-save: when `layer` is in the posted fields, call `reorder_chain(cx, tid, rid, layer)` (and drop `layer` from the `update_chain_row` field set).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_biofield_stress_routes.py
import sqlite3
import pytest
from biofield_local_app import create_app
import dashboard.biofield_reveal_import as RI


@pytest.fixture(autouse=True)
def _no_gate(monkeypatch):
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)


_FRESH = {"status": "fresh", "found": True, "scan_id": 900, "scan_date": "2026-06-24",
          "days_ago": 1, "fresh": True, "window_days": 14, "message": "ok",
          "findings": [{"code": "ED1", "name": "Membrane", "group": "infoceutical"},
                       {"code": "MR2", "name": "Calm", "group": "stress"}],
          "infoceuticals": [], "stresses": []}
_NONE = {"status": "none", "found": False, "findings": [], "days_ago": None, "fresh": False}

# synthesis stub: ED1 covered by Neuro Magnesium -> required; MR2 optional
_SYNTH = {"found": True, "scan_id": 900, "scan_date": "2026-06-24", "days_ago": 1,
          "fresh": True, "layers": [{"n": 1, "title": "Ox", "summary": "",
          "most_affected": "Membrane", "remedy_name": "Neuro Magnesium", "codes": ["ED1"]}]}


def _app(db):
    return create_app(db, scan_lookup=lambda e: _FRESH if e == "j@x.com" else _NONE)


def _new(client, email):
    tid = client.post("/author/new").headers["Location"].rstrip("/").split("/")[-1]
    client.post(f"/author/{tid}/header", json={"name": "J", "email": email, "date": "2026-06-25"})
    return tid


def test_header_save_seeds_stresses(tmp_path, monkeypatch):
    monkeypatch.setattr(RI, "synthesize_reveal_layers", lambda *a, **k: _SYNTH)
    db = str(tmp_path / "c.db")
    client = _app(db)
    tid = _new(client, "j@x.com")
    j = client.get(f"/author/{tid}/stresses").get_json()
    codes = {s["code"] for s in j["data"]["active"]} | {s["code"] for s in j["data"]["balanced"]}
    assert codes == {"ED1", "MR2"}
    bal = {s["balance"] for s in j["data"]["active"] + j["data"]["balanced"]}
    assert bal == {"required", "optional"}


def test_stress_lists_balance_follows_chain(tmp_path, monkeypatch):
    monkeypatch.setattr(RI, "synthesize_reveal_layers", lambda *a, **k: _SYNTH)
    db = str(tmp_path / "c.db")
    client = _app(db)
    tid = _new(client, "j@x.com")
    client.post(f"/author/{tid}/row", json={"layer": 1, "head": "Ox", "most_affected": "Membrane",
                                            "remedy": "Neuro Magnesium"})
    j = client.get(f"/author/{tid}/stresses").get_json()
    assert {s["code"] for s in j["data"]["balanced"]} == {"ED1"}
    assert {s["code"] for s in j["data"]["active"]} == {"MR2"}


def test_manual_balance_toggle(tmp_path, monkeypatch):
    monkeypatch.setattr(RI, "synthesize_reveal_layers", lambda *a, **k: _SYNTH)
    db = str(tmp_path / "c.db")
    client = _app(db)
    tid = _new(client, "j@x.com")
    sid = sqlite3.connect(db).execute("SELECT id FROM biofield_auth_stress WHERE code='MR2'").fetchone()[0]
    client.post(f"/author/{tid}/stress/{sid}/balance", json={"value": True})
    j = client.get(f"/author/{tid}/stresses").get_json()
    assert "MR2" in {s["code"] for s in j["data"]["balanced"]}


def test_row_save_layer_change_reorders(tmp_path, monkeypatch):
    monkeypatch.setattr(RI, "synthesize_reveal_layers", lambda *a, **k: _NONE)  # no seed needed
    db = str(tmp_path / "c.db")
    client = create_app(db, scan_lookup=lambda e: _NONE)
    tid = client.post("/author/new").headers["Location"].rstrip("/").split("/")[-1]
    r1 = client.post(f"/author/{tid}/row", json={"layer": 1, "head": "A", "remedy": "R1"}).get_json()["rid"]
    r2 = client.post(f"/author/{tid}/row", json={"layer": 2, "head": "B", "remedy": "R2"}).get_json()["rid"]
    client.post(f"/author/{tid}/row/{r2}", json={"layer": 1})   # move B to top
    from dashboard.biofield_authoring import ordered_chain
    assert [l["head"] for l in ordered_chain(sqlite3.connect(db), tid)] == ["B", "A"]
```

- [ ] **Step 2: Run** → FAIL (routes 404 / no seeding).

- [ ] **Step 3: Implement** — in `biofield_local_app.py`:

(a) Add the seed helper inside `create_app` (near `_e4l`):

```python
    def _seed_stresses(cx, test_id, *, force=False):
        from dashboard import biofield_reveal_import as _ri
        from dashboard import biofield_stress as _st
        import datetime as _dt
        rep = _report_for(cx, test_id)
        email = ((rep.get("client") or {}).get("email") or "").strip()
        if not email:
            return
        _st.init_stress_tables(cx)
        if not force and cx.execute(
                "SELECT 1 FROM biofield_auth_stress WHERE test_id=? AND source='scan' LIMIT 1",
                (int(str(test_id).lstrip("a") or 0),)).fetchone():
            return
        ctx = scan_lookup(email)
        if not ctx.get("found"):
            return
        try:
            res = _ri.synthesize_reveal_layers(email, today=_dt.date.today().isoformat())
        except Exception:
            return
        coverage = _ri.build_coverage(res.get("layers") or [])
        _st.seed_from_scan(cx, test_id, ctx.get("findings") or [], coverage)
```

(b) In the header-save route, after the existing `_e4l(cx, test_id)` call, add `_seed_stresses(cx, test_id)`.

(c) In the import-reveal route, after a successful `import_layers_to_test`, add `_seed_stresses(cx, test_id, force=True)`.

(d) Add the two new routes (near the other `/author/<test_id>/...` routes):

```python
    @app.route("/author/<test_id>/stresses")
    def author_stresses(test_id):
        from dashboard import biofield_stress as _st
        with sqlite3.connect(db_path) as cx:
            rep = _report_for(cx, test_id)
            remedies = [l.get("remedy") for l in (rep.get("layers") or []) if l.get("remedy")]
            data = _st.list_stresses(cx, test_id, remedies)
        return {"data": data, "html": render_stress_panel(data)}

    @app.route("/author/<test_id>/stress/<int:sid>/balance", methods=["POST"])
    def author_stress_balance(test_id, sid):
        from dashboard import biofield_stress as _st
        value = bool((request.get_json(silent=True) or {}).get("value"))
        with sqlite3.connect(db_path) as cx:
            _st.set_manual_balanced(cx, test_id, sid, value)
        return {"ok": True}
```

(e) In the existing `author_row_save` route, branch on `layer`:

```python
        if "layer" in fields:
            new_layer = fields.pop("layer")
            from dashboard.biofield_authoring import reorder_chain
            with sqlite3.connect(db_path) as cx:
                if fields:
                    update_chain_row(cx, rid, **fields)
                reorder_chain(cx, test_id, rid, new_layer)
            return {"ok": True}
```
(Place this before the existing `update_chain_row` call so a layer change reorders; keep the original path for non-layer edits.)

(f) Import `render_stress_panel` from `dashboard.biofield_report_html` at the top alongside `render_e4l_panel` (it is created in Task 8 — this task's tests assert only the `data` field, so a temporary `render_stress_panel` stub returning `""` may be added in Task 8; to keep Task 7 importable, add `render_stress_panel` to the import list and ensure Task 8 defines it. If running Task 7 before Task 8, define a minimal `render_stress_panel(data): return ""` in `biofield_report_html.py` now and flesh it out in Task 8.)

- [ ] **Step 4: Run** — `pytest tests/test_biofield_stress_routes.py tests/test_biofield_import_reveal_routes.py tests/test_biofield_e4l_routes.py -v` → PASS (SP-A + e4l route tests still green).

- [ ] **Step 5: Commit** — `git add biofield_local_app.py dashboard/biofield_report_html.py tests/test_biofield_stress_routes.py && git commit -m "feat(biofield-b1): seed hook + stress routes + reorder wiring"`.

---

### Task 8: UI — stress panel, two-zone chain rendering, report listing

**Files:**
- Modify: `dashboard/biofield_report_html.py`
- Test: `tests/test_biofield_stress_panel.py`

**Interfaces:**
- Consumes: `list_stresses` output shape (`{"active":[...],"balanced":[...]}`); the author page's `__TID__`/`post()`/`location.reload()` JS conventions; the `#e4lpanel` embedding pattern; `ordered_chain` layer/zone fields surfaced via `authored_report`.
- Produces: `render_stress_panel(data) -> str`; a `#stresspanel` div + `loadStress()`/`setStress()`/`balanceStress(sid,val)` JS; the chain table renders a "Unbalanced from scan" group divider before bottom-zone rows; `render_report_html` gains a "Stresses balanced" section (best-effort: reads stresses for the report's test).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_biofield_stress_panel.py
from dashboard.biofield_report_html import render_stress_panel, render_author_html


def test_panel_shows_active_and_balanced_with_tags():
    data = {"active": [{"id": 1, "code": "MR2", "label": "Calm", "source": "scan",
                        "balance": "optional", "balanced": False, "balanced_by": ""}],
            "balanced": [{"id": 2, "code": "ED1", "label": "Membrane", "source": "scan",
                          "balance": "required", "balanced": True, "balanced_by": "neuro magnesium"}]}
    html = render_stress_panel(data)
    assert "Calm" in html and "Membrane" in html
    assert "neuro magnesium" in html            # balanced_by shown
    assert "optional" in html and "required" in html
    assert "balanceStress(1" in html            # toggle on the active item


def test_author_page_has_stress_panel_and_loader():
    rep = {"test_id": "a7", "client": {"name": "J", "email": "j@x.com"}, "date": "",
           "layers": [], "schedule": []}
    html = render_author_html(rep, [], "")
    assert "id=stresspanel" in html and "loadStress()" in html
    assert "function balanceStress" in html


def test_author_page_marks_unbalanced_scan_group():
    rep = {"test_id": "a7", "client": {"name": "J", "email": "j@x.com"}, "date": "",
           "layers": [
               {"layer": 1, "head": "Live", "most_affected": "", "remedy": "R1", "rid": 1,
                "confirmed": 1, "origin": "live", "zone": "top",
                "stress_depth": None, "remedy_depth": None, "depth_status": None, "depth_need": None},
               {"layer": 2, "head": "Scan", "most_affected": "", "remedy": "R2", "rid": 2,
                "confirmed": 0, "origin": "scan", "zone": "bottom",
                "stress_depth": None, "remedy_depth": None, "depth_status": None, "depth_need": None}],
           "schedule": []}
    html = render_author_html(rep, [], "")
    assert "Unbalanced from scan" in html
```

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement**

Add `render_stress_panel` (replace the Task-7 stub if present):

```python
def render_stress_panel(data):
    data = data or {}
    def _row(s, active):
        tag = _e(s.get("balance") or "")
        by = _e(s.get("balanced_by") or "")
        bytxt = f" <span class=food>&middot; {by}</span>" if (not active and by) else ""
        btn = (f"<button class='btn ghost' style='font-size:11px' "
               f"onclick=\"balanceStress({int(s.get('id') or 0)},{'true' if active else 'false'})\">"
               f"{'Balance' if active else 'Reactivate'}</button>")
        return (f"<li><b>{_e(s.get('code') or '')}</b> {_e(s.get('label') or '')} "
                f"<span class=pill>{tag}</span>{bytxt} {btn}</li>")
    act = "".join(_row(s, True) for s in data.get("active") or [])
    bal = "".join(_row(s, False) for s in data.get("balanced") or [])
    act_html = f"<div class=food style='font-weight:600;margin-top:6px'>Active &mdash; to balance</div><ul style='margin:4px 0;padding-left:18px'>{act}</ul>" if act else "<div class=food style='margin-top:6px'>No active stresses.</div>"
    bal_html = f"<div class=food style='font-weight:600;margin-top:6px'>Balanced</div><ul style='margin:4px 0;padding-left:18px'>{bal}</ul>" if bal else ""
    return ("<div class=card><div class=food style='text-transform:uppercase;font-size:11px;"
            "letter-spacing:.08em'>Stress balancing</div>" + act_html + bal_html + "</div>")
```

In the `_AUTHOR_JS` block, add (next to `loadE4L`):

```javascript
function setStress(j){if(j&&j.html!==undefined)document.getElementById('stresspanel').innerHTML=j.html}
async function loadStress(){try{setStress(await (await fetch('/author/__TID__/stresses')).json())}catch(e){}}
async function balanceStress(sid,val){await post('/author/__TID__/stress/'+sid+'/balance',{value:val});loadStress()}
```

Add `loadStress();` next to the existing `loadE4L();` call (line ~344). Add the `<div id=stresspanel></div>` next to `<div id=e4lpanel></div>` (line ~502).

In the chain table builder of `render_author_html`, when iterating `report["layers"]`, emit a divider row `"<tr><td colspan=...><b>Unbalanced from scan</b></td></tr>"` immediately before the first layer whose `zone == "bottom"` (track a `shown_divider` flag).

In `render_report_html`, after the chain, add a best-effort "Stresses balanced" section. Since `render_report_html` doesn't take stress data, accept it does not have DB access — instead surface the section from the already-rendered `report["layers"]` is insufficient; keep B1 minimal by having the caller (report route) pass stress data. Simplify: add an optional param `render_report_html(report, notes, narrative, vscript, stresses=None)`; when `stresses` is provided, append a "Stresses balanced" list. The report route (`/test/<id>` / report view) computes `list_stresses(...)` and passes it. (Only wire the param + section here; the route already exists — update its call.)

- [ ] **Step 4: Run** — `pytest tests/test_biofield_stress_panel.py tests/test_biofield_report_html.py tests/test_biofield_author_html.py -v` → PASS (existing author/report html tests green).

- [ ] **Step 5: Run the biofield suite + commit**

```bash
~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_stress_panel.py tests/test_biofield_stress_routes.py tests/test_biofield_stress_seed.py tests/test_biofield_stress_derive.py tests/test_biofield_ordered_chain.py tests/test_biofield_reorder_chain.py tests/test_biofield_chain_origin.py tests/test_biofield_coverage.py tests/test_biofield_reveal_import.py tests/test_biofield_import_reveal_routes.py tests/test_biofield_import_reveal_button.py tests/test_biofield_authoring.py tests/test_biofield_report.py tests/test_biofield_report_html.py tests/test_biofield_author_html.py tests/test_biofield_e4l_routes.py -q
git add dashboard/biofield_report_html.py tests/test_biofield_stress_panel.py
git commit -m "feat(biofield-b1): stress panel + two-zone chain render + report listing"
```

---

## Self-Review

**Spec coverage:**
- Master stress list (source/required-optional/manual) → Task 5. ✓
- Coverage map → Tasks 2 (build) + 5 (persist). ✓
- Recompute-on-read auto-balance → Task 6. ✓
- Seeding fold-in (header + import, idempotent, preserve manual) → Tasks 5 + 7. ✓
- Two-zone ordering → Task 3; layer reorder insert-at-N → Task 4 + wiring Task 7. ✓
- `origin` to distinguish scan vs live → Task 1. ✓
- UI panel + two-zone render + report listing → Task 8. ✓

**Placeholder scan:** Task 7(f)/Task 8 note the `render_stress_panel` stub-then-flesh ordering explicitly (so Task 7 stays importable if run first); no TBDs. ✓

**Type consistency:** `list_stresses` shape (`active`/`balanced` with `id/code/label/source/balance/balanced/balanced_by`) is identical across Tasks 6, 7, 8. `ordered_chain` keys (`layer/zone/origin/confirmed/...`) consistent across Tasks 3, 4, 8. `build_coverage` output (`{remedy_lower:set}`) consumed by Tasks 5/7. ✓

## Verification (manual, after all tasks)

```bash
cd ~/deploy-chat && doppler run -p remedy-match -c prd -- python3 biofield_local_app.py
```
Open a test for a client with a recent scan: confirm the Stress Balancing panel seeds (required vs optional), adding a remedy to a chain layer moves its scan stresses to Balanced, the manual toggle works, unbalanced scan layers trail under "Unbalanced from scan," and changing a top-zone layer number reorders.
