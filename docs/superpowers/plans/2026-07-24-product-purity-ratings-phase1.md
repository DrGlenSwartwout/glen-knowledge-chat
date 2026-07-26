# Product Purity Ratings — Phase 1 (the engine) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the reusable core of the purity-rating system — the avoid-list data asset, the role-aware excipient screen, and the `product_ratings` table with its state machine — all unit-tested with manual excipient entry, no UI, no data acquisition, no readers.

**Architecture:** Three pure modules with no Flask and no `app.py` involvement. `dashboard/purity_avoidlist.py` loads the versioned avoid-list from a repo data file. `dashboard/purity_screen.py` screens a product's Other Ingredients against it into a red/yellow/green/unrated color. `dashboard/product_ratings.py` owns the product-keyed cache table and the never-downgrade state machine. Everything takes its data by argument (the connection, the ingredient list, the avoid-list dict), so each unit is testable in isolation.

**Tech Stack:** Python 3, sqlite3 (Postgres in prod via the db adapter), pytest.

**Spec:** `docs/superpowers/specs/2026-07-24-product-purity-ratings-design.md`

## Global Constraints

Every task's requirements implicitly include this section.

- **Pure modules only.** No Flask imports, no `app.py` imports. The caller passes the connection (`cx`), the ingredient list, and the avoid-list dict. This mirrors `dashboard/fullscript.py` and `dashboard/prl_supplement.py`.
- **`product_ratings.product_key` is the natural PRIMARY KEY (TEXT).** There is no surrogate id and no `cur.lastrowid` anywhere — `lastrowid` raises on the Postgres adapter, and a product's key is its identity, so the table needs no autoincrement.
- **The avoid-list loads from the repo, not `DATA_DIR`.** `$DATA_DIR` strips repo data files in the full suite and in prod (that is why `products.json` has a repo fallback). The avoid-list is code-like config that ships with the repo, so its loader resolves a **module-relative path**, never `DATA_DIR`. Tests read the committed file directly.
- **Color vocabulary is exactly `red` / `yellow` / `green` / `unrated`.** No other values.
- **`unrated` is never `green`.** Absence of excipient data resolves to `unrated`, never to clean-by-default. This is the non-negotiable safety rule from the spec and it must be mutation-verified.
- **The screen reads ONLY Other Ingredients.** Actives never affect the result — silica in the Supplement Facts is a nutrient, silica in Other Ingredients is a filler.
- **Red beats yellow.** A product with both a red and a yellow item is red.
- **State transitions never downgrade.** A confirmed row is never walked back or overwritten (mirrors `dashboard/supplement_reviews.py`).
- **Every guard test must bite under mutation.** For each safety property, the implementer breaks the implementation, runs the test, confirms the relevant test fails, then reverts. A test that passes against the broken code is a defect.
- **Never run the bare full test suite — it sends real email.** Run only the named test files.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `data/excipient_avoidlist.json` (create) | the versioned avoid-list: red list + yellow list, each entry `{canonical, aliases, rationale}` |
| `dashboard/purity_avoidlist.py` (create) | load + validate the avoid-list from the repo path |
| `dashboard/purity_screen.py` (create) | role-aware screen: Other Ingredients + avoid-list → color + hits |
| `dashboard/product_ratings.py` (create) | the product-keyed table + never-downgrade state machine |
| `tests/test_purity_avoidlist.py` (create) | invariants of the committed avoid-list file |
| `tests/test_purity_screen.py` (create) | screen logic and its safety guards |
| `tests/test_product_ratings.py` (create) | schema, one-row-per-product, state machine |

---

### Task 1: The avoid-list asset and its loader

**Files:**
- Create: `data/excipient_avoidlist.json`
- Create: `dashboard/purity_avoidlist.py`
- Create: `tests/test_purity_avoidlist.py`

**Interfaces:**
- Produces: `load_avoidlist(path=None) -> dict` returning `{"version": str, "red": [entry], "yellow": [entry]}` where each `entry` is `{"canonical": str, "aliases": [str], "rationale": str}`; and `validate(avoidlist) -> None` which raises `ValueError` on a malformed avoid-list.

- [ ] **Step 1: Write the avoid-list data file**

Create `data/excipient_avoidlist.json`. These entries and rationales are Glen's documented positions from the Pam Schreur analysis; the aliases catch the same substance under any label wording.

```json
{
  "version": "2026-07-24",
  "red": [
    {"canonical": "magnesium stearate",
     "aliases": ["magnesium stearate", "vegetable stearate", "stearic acid", "vegetable magnesium stearate"],
     "rationale": "Stearates coat particles and reduce bioavailability."},
    {"canonical": "gelatin",
     "aliases": ["gelatin", "bovine gelatin", "porcine gelatin", "gelatin capsule"],
     "rationale": "Animal gelatin carries a documented glyphosate-contamination concern (Seneff) and is non-vegetarian."},
    {"canonical": "dicalcium phosphate",
     "aliases": ["dicalcium phosphate", "dcp", "calcium phosphate dibasic"],
     "rationale": "Dicalcium phosphate is a poorly-chosen filler that Glen avoids."},
    {"canonical": "titanium dioxide",
     "aliases": ["titanium dioxide"],
     "rationale": "Titanium dioxide is a whitening agent with no nutritional purpose and safety concerns."},
    {"canonical": "artificial color",
     "aliases": ["fd&c", "fd & c", "lake", "artificial color", "artificial colour", "tartrazine", "yellow 5", "red 40", "blue 1"],
     "rationale": "Synthetic dyes have no nutritional purpose."},
    {"canonical": "hydrogenated oil",
     "aliases": ["hydrogenated oil", "partially hydrogenated", "hydrogenated soybean oil", "hydrogenated palm oil"],
     "rationale": "Hydrogenated oils introduce trans fats."},
    {"canonical": "carrageenan",
     "aliases": ["carrageenan"],
     "rationale": "Carrageenan is a pro-inflammatory thickener."}
  ],
  "yellow": [
    {"canonical": "silicon dioxide",
     "aliases": ["silicon dioxide", "silica", "silicon dioxide (as a flow agent)", "colloidal silicon dioxide"],
     "rationale": "Silica as a flow-agent filler is inert and tolerated, but not ideal; it counts against a product only as a filler, never when present as an intentional nutrient."}
  ]
}
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_purity_avoidlist.py`:

```python
"""Invariants of the COMMITTED data/excipient_avoidlist.json plus its loader.
Reads the real repo file (no DATA_DIR), the way the app will."""
from dashboard import purity_avoidlist as pa


def test_loads_committed_file():
    al = pa.load_avoidlist()
    assert al["version"], "must carry a version stamp"
    assert al["red"] and al["yellow"], "both lists present and non-empty"


def test_committed_file_is_valid():
    pa.validate(pa.load_avoidlist())  # must not raise


def test_every_entry_has_canonical_aliases_rationale():
    al = pa.load_avoidlist()
    for bucket in ("red", "yellow"):
        for e in al[bucket]:
            assert e["canonical"], bucket
            assert e["aliases"] and all(a.strip() for a in e["aliases"]), e["canonical"]
            assert e["rationale"].strip(), e["canonical"]


def test_red_and_yellow_are_disjoint():
    al = pa.load_avoidlist()
    reds = {e["canonical"] for e in al["red"]}
    yellows = {e["canonical"] for e in al["yellow"]}
    assert reds.isdisjoint(yellows), "a canonical can't be both red and yellow"


def test_stearate_is_red_and_silica_is_yellow():
    al = pa.load_avoidlist()
    assert any(a == "vegetable stearate" for e in al["red"] for a in e["aliases"])
    assert any(a == "silica" for e in al["yellow"] for a in e["aliases"])


def test_validate_rejects_missing_version():
    import pytest
    with pytest.raises(ValueError):
        pa.validate({"red": [], "yellow": []})


def test_validate_rejects_entry_without_aliases():
    import pytest
    bad = {"version": "x", "red": [{"canonical": "c", "aliases": [], "rationale": "r"}], "yellow": []}
    with pytest.raises(ValueError):
        pa.validate(bad)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd ~/deploy-chat && python -m pytest tests/test_purity_avoidlist.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dashboard.purity_avoidlist'`

- [ ] **Step 4: Write minimal implementation**

Create `dashboard/purity_avoidlist.py`:

```python
"""Loads and validates the versioned excipient avoid-list. Pure: no Flask, no
app import. The avoid-list is code-like config shipped in the repo, so it is
read from a MODULE-RELATIVE path -- never DATA_DIR, which strips repo data files
in prod and in the full test suite."""
import json
import os

_REPO_AVOIDLIST = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "excipient_avoidlist.json")


def load_avoidlist(path=None):
    with open(path or _REPO_AVOIDLIST, "r", encoding="utf-8") as f:
        al = json.load(f)
    validate(al)
    return al


def validate(al):
    if not isinstance(al, dict) or not al.get("version"):
        raise ValueError("avoid-list needs a non-empty 'version'")
    for bucket in ("red", "yellow"):
        entries = al.get(bucket)
        if not isinstance(entries, list):
            raise ValueError(f"avoid-list '{bucket}' must be a list")
        for e in entries:
            if not e.get("canonical"):
                raise ValueError(f"{bucket} entry missing 'canonical'")
            aliases = e.get("aliases")
            if not aliases or not all(isinstance(a, str) and a.strip() for a in aliases):
                raise ValueError(f"{bucket} entry {e.get('canonical')!r} needs non-empty aliases")
            if not (e.get("rationale") or "").strip():
                raise ValueError(f"{bucket} entry {e.get('canonical')!r} needs a rationale")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd ~/deploy-chat && python -m pytest tests/test_purity_avoidlist.py -v`
Expected: PASS, 7 tests

- [ ] **Step 6: Commit**

```bash
git add data/excipient_avoidlist.json dashboard/purity_avoidlist.py tests/test_purity_avoidlist.py
git commit -m "feat(purity): versioned excipient avoid-list asset + validating loader"
```

---

### Task 2: The role-aware screen

**Files:**
- Create: `dashboard/purity_screen.py`
- Create: `tests/test_purity_screen.py`

**Interfaces:**
- Consumes: an avoid-list dict from Task 1 (`load_avoidlist()`).
- Produces: `screen_label(actives, other_ingredients, avoidlist) -> dict` returning `{"color": str, "red_hits": [str], "yellow_hits": [str], "avoidlist_version": str}`. `color` is one of `red` / `yellow` / `green` / `unrated`. `actives` is a list (or None) and is **never** consulted. `other_ingredients` is `None` (no data → `unrated`), `[]` (known to have none → `green`), or a list of strings (screened).

- [ ] **Step 1: Write the failing test**

Create `tests/test_purity_screen.py`:

```python
"""The role-aware excipient screen and its safety guards."""
from dashboard import purity_avoidlist as pa
from dashboard import purity_screen as ps

AL = pa.load_avoidlist()


def test_stearate_alias_matches_red_under_any_wording():
    for wording in ["Magnesium Stearate", "vegetable stearate", "Stearic Acid"]:
        out = ps.screen_label(["Vitamin C"], ["Cellulose", wording], AL)
        assert out["color"] == "red", wording
        assert out["red_hits"], wording


def test_silica_in_other_ingredients_is_yellow():
    out = ps.screen_label(["Vitamin C"], ["Silicon Dioxide"], AL)
    assert out["color"] == "yellow"
    assert out["yellow_hits"] == ["Silicon Dioxide"]


def test_silica_as_an_ACTIVE_is_ignored():
    # Same substance, nutritional role: must not count. This is the whole point
    # of screening only Other Ingredients.
    out = ps.screen_label(["Silica 10 mg", "Vitamin C"], ["Cellulose"], AL)
    assert out["color"] == "green"
    assert out["yellow_hits"] == []


def test_red_beats_yellow():
    out = ps.screen_label(None, ["Silicon Dioxide", "Magnesium Stearate"], AL)
    assert out["color"] == "red"


def test_clean_other_ingredients_are_green():
    out = ps.screen_label(["Vitamin C"], ["Hypromellose", "Microcrystalline Cellulose"], AL)
    assert out["color"] == "green"
    assert out["red_hits"] == [] and out["yellow_hits"] == []


def test_empty_list_is_green_but_none_is_unrated():
    # A product genuinely listing no other ingredients is pristine (green).
    assert ps.screen_label(["Taurine"], [], AL)["color"] == "green"
    # Absence of DATA is never green.
    assert ps.screen_label(["Taurine"], None, AL)["color"] == "unrated"


def test_version_is_echoed():
    out = ps.screen_label(None, ["Cellulose"], AL)
    assert out["avoidlist_version"] == AL["version"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/deploy-chat && python -m pytest tests/test_purity_screen.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dashboard.purity_screen'`

- [ ] **Step 3: Write minimal implementation**

Create `dashboard/purity_screen.py`:

```python
"""Role-aware excipient screen. Screens ONLY the Other Ingredients against the
avoid-list; the actives list is accepted for interface symmetry but is never
consulted (a substance's role decides -- silica as a nutrient is not a filler).
Pure: no Flask, no app import.

Contract for `other_ingredients`:
  None -> no excipient data was obtained -> color 'unrated' (NEVER green).
  []   -> the product is known to list no other ingredients -> 'green'.
  list -> screened item by item.
"""


def _normalize(name):
    """Lowercase and strip common descriptors so aliases match real labels."""
    s = (name or "").lower()
    for cut in ("(vegetable source)", "(vegetable)", "(as a flow agent)", "(from rice)"):
        s = s.replace(cut, "")
    return " ".join(s.split()).strip()


def _hits(normalized_item, entries):
    for e in entries:
        for alias in e["aliases"]:
            if alias in normalized_item:
                return True
    return False


def screen_label(actives, other_ingredients, avoidlist):
    version = avoidlist.get("version", "")
    if other_ingredients is None:                       # no data -> unrated, never green
        return {"color": "unrated", "red_hits": [], "yellow_hits": [],
                "avoidlist_version": version}
    red_hits, yellow_hits = [], []
    for raw in other_ingredients:
        norm = _normalize(raw)
        if _hits(norm, avoidlist["red"]):
            red_hits.append(raw)
        elif _hits(norm, avoidlist["yellow"]):
            yellow_hits.append(raw)
    if red_hits:
        color = "red"
    elif yellow_hits:
        color = "yellow"
    else:
        color = "green"
    return {"color": color, "red_hits": red_hits, "yellow_hits": yellow_hits,
            "avoidlist_version": version}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/deploy-chat && python -m pytest tests/test_purity_screen.py -v`
Expected: PASS, 7 tests

- [ ] **Step 5: Mutation-verify the two safety guards**

These two properties are the reason the screen exists; prove their tests bite.

1. Change `if other_ingredients is None:` to `return {"color": "green", ...}` (make no-data resolve to green). Run the file. `test_empty_list_is_green_but_none_is_unrated` MUST fail. Revert.
2. Change `screen_label` to also scan `actives` (e.g. iterate `(actives or []) + other_ingredients`). Run the file. `test_silica_as_an_ACTIVE_is_ignored` MUST fail. Revert.

Run: `cd ~/deploy-chat && python -m pytest tests/test_purity_screen.py -v`
Expected: FAIL on the named test while mutated, PASS after revert.

- [ ] **Step 6: Commit**

```bash
git add dashboard/purity_screen.py tests/test_purity_screen.py
git commit -m "feat(purity): role-aware excipient screen (red/yellow/green/unrated)"
```

---

### Task 3: The `product_ratings` table and state machine

**Files:**
- Create: `dashboard/product_ratings.py`
- Create: `tests/test_product_ratings.py`

**Interfaces:**
- Consumes: a screen result dict from Task 2.
- Produces:
  - `init_tables(cx)`
  - `record_screen(cx, product_key, *, brand, product_name, other_ingredients_raw, other_ingredients_parsed, screen)` — upserts the row from a screen result; sets `status='screened'` (or `'unrated'` when `screen["color"] == "unrated"`); never downgrades a row already at `ai_draft`/`confirmed`.
  - `set_tier2(cx, product_key, score, detail)` — advances a screened non-red row to `ai_draft`.
  - `confirm(cx, product_key)` — advances a red `screened` row or a yellow/green `ai_draft` row to `confirmed`.
  - `get(cx, product_key) -> dict | None`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_product_ratings.py`:

```python
"""product_ratings: one row per product, never-downgrade state machine,
unrated can't advance to a color."""
import sqlite3
import pytest
from dashboard import product_ratings as pr

GREEN = {"color": "green", "red_hits": [], "yellow_hits": [], "avoidlist_version": "v1"}
RED = {"color": "red", "red_hits": ["Magnesium Stearate"], "yellow_hits": [], "avoidlist_version": "v1"}
UNRATED = {"color": "unrated", "red_hits": [], "yellow_hits": [], "avoidlist_version": "v1"}


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    pr.init_tables(cx)
    return cx


def _rec(cx, key, screen):
    pr.record_screen(cx, key, brand="B", product_name="N",
                     other_ingredients_raw="...", other_ingredients_parsed=["..."],
                     screen=screen)


def test_one_row_per_product():
    cx = _cx()
    _rec(cx, "brand-x", GREEN)
    _rec(cx, "brand-x", GREEN)  # same key again
    assert cx.execute("SELECT COUNT(*) FROM product_ratings").fetchone()[0] == 1


def test_screen_sets_color_and_hits():
    cx = _cx()
    _rec(cx, "k", RED)
    row = pr.get(cx, "k")
    assert row["color"] == "red" and row["status"] == "screened"
    assert row["red_hits"] == ["Magnesium Stearate"]
    assert row["avoidlist_version"] == "v1"


def test_unrated_screen_lands_unrated_not_a_color():
    cx = _cx()
    _rec(cx, "k", UNRATED)
    row = pr.get(cx, "k")
    assert row["status"] == "unrated"
    assert row["color"] is None, "unrated must not be stored as a color"


def test_green_advances_screened_to_ai_draft_to_confirmed():
    cx = _cx()
    _rec(cx, "k", GREEN)
    pr.set_tier2(cx, "k", 8.5, '{"note":"good"}')
    assert pr.get(cx, "k")["status"] == "ai_draft"
    pr.confirm(cx, "k")
    assert pr.get(cx, "k")["status"] == "confirmed"


def test_red_confirms_without_tier2():
    cx = _cx()
    _rec(cx, "k", RED)
    pr.confirm(cx, "k")  # red skips ai_draft
    assert pr.get(cx, "k")["status"] == "confirmed"


def test_never_downgrades_a_confirmed_row():
    cx = _cx()
    _rec(cx, "k", GREEN)
    pr.set_tier2(cx, "k", 8.5, "{}")
    pr.confirm(cx, "k")
    _rec(cx, "k", RED)  # a later re-screen must not walk it back
    row = pr.get(cx, "k")
    assert row["status"] == "confirmed"


def test_unrated_cannot_advance_to_tier2():
    cx = _cx()
    _rec(cx, "k", UNRATED)
    with pytest.raises(ValueError):
        pr.set_tier2(cx, "k", 8.5, "{}")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/deploy-chat && python -m pytest tests/test_product_ratings.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dashboard.product_ratings'`

- [ ] **Step 3: Write minimal implementation**

Create `dashboard/product_ratings.py`:

```python
"""The product-keyed purity-ratings cache and its never-downgrade state machine.
Pure sqlite; caller passes cx. product_key is the natural PRIMARY KEY, so there
is no surrogate id and no cur.lastrowid (which raises on the Postgres adapter).

status order: requested/unrated (0) -> screened (1) -> ai_draft (2) -> confirmed (3).
A row never moves to a lower rank. 'unrated' means a screen ran but no excipient
data was available; it holds no color and cannot advance.

Writes follow the SELECT-then-INSERT/UPDATE pattern of dashboard/supplement_reviews.py
(no `ON CONFLICT ... excluded` upsert) and stamp timestamps in Python (no SQL
`datetime('now')`), because both of those forms vary by backend and this table
must run on the Postgres adapter in prod.
"""
import json
from datetime import datetime, timezone

_RANK = {"requested": 0, "unrated": 0, "screened": 1, "ai_draft": 2, "confirmed": 3}


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def init_tables(cx):
    cx.execute("""CREATE TABLE IF NOT EXISTS product_ratings (
        product_key TEXT PRIMARY KEY,
        brand TEXT, product_name TEXT,
        fullscript_slug TEXT, fullscript_external_id TEXT,
        other_ingredients_raw TEXT, other_ingredients_parsed TEXT,
        color TEXT, red_hits TEXT, yellow_hits TEXT, avoidlist_version TEXT,
        tier2_score REAL, tier2_json TEXT, best_ff TEXT,
        status TEXT NOT NULL,
        requested_at TEXT, screened_at TEXT, drafted_at TEXT, confirmed_at TEXT, updated_at TEXT)""")
    cx.execute("CREATE INDEX IF NOT EXISTS ix_prat_color ON product_ratings(color)")
    cx.execute("CREATE INDEX IF NOT EXISTS ix_prat_status ON product_ratings(status)")
    cx.commit()


def get(cx, product_key):
    r = cx.execute("SELECT * FROM product_ratings WHERE product_key=?", (product_key,)).fetchone()
    return dict(r) if r else None


def record_screen(cx, product_key, *, brand, product_name,
                  other_ingredients_raw, other_ingredients_parsed, screen):
    """Insert or update a row from a screen result. Never downgrades a row already
    at ai_draft/confirmed. An 'unrated' screen (no data) lands status 'unrated',
    color NULL -- never green."""
    existing = get(cx, product_key)
    if existing is not None and _RANK[existing["status"]] >= _RANK["ai_draft"]:
        return  # confirmed/ai_draft rows are not walked back by a re-screen
    color = screen["color"]
    if color == "unrated":
        new_status, stored_color = "unrated", None
    else:
        new_status, stored_color = "screened", color
    now = _now()
    parsed = json.dumps(other_ingredients_parsed or [])
    reds = json.dumps(screen.get("red_hits") or [])
    yellows = json.dumps(screen.get("yellow_hits") or [])
    version = screen.get("avoidlist_version")
    if existing is None:
        cx.execute("""INSERT INTO product_ratings
            (product_key, brand, product_name, other_ingredients_raw, other_ingredients_parsed,
             color, red_hits, yellow_hits, avoidlist_version, status, screened_at, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (product_key, brand, product_name, other_ingredients_raw, parsed,
             stored_color, reds, yellows, version, new_status, now, now))
    else:
        cx.execute("""UPDATE product_ratings SET
             brand=?, product_name=?, other_ingredients_raw=?, other_ingredients_parsed=?,
             color=?, red_hits=?, yellow_hits=?, avoidlist_version=?, status=?,
             screened_at=?, updated_at=? WHERE product_key=?""",
            (brand, product_name, other_ingredients_raw, parsed,
             stored_color, reds, yellows, version, new_status, now, now, product_key))
    cx.commit()


def set_tier2(cx, product_key, score, detail_json):
    """Advance a screened, non-red row to ai_draft. Reds never run tier-2, and an
    unrated row has no color to rank -- both raise."""
    row = get(cx, product_key)
    if not row:
        raise ValueError("no such product_rating")
    if row["status"] != "screened" or row["color"] not in ("yellow", "green"):
        raise ValueError(f"cannot run tier-2 on status={row['status']} color={row['color']}")
    now = _now()
    cx.execute("UPDATE product_ratings SET tier2_score=?, tier2_json=?, status='ai_draft', "
               "drafted_at=?, updated_at=? WHERE product_key=?",
               (score, detail_json, now, now, product_key))
    cx.commit()


def confirm(cx, product_key):
    """Confirm a red screened row (reds skip tier-2) or a yellow/green ai_draft row."""
    row = get(cx, product_key)
    if not row:
        raise ValueError("no such product_rating")
    ok = (row["status"] == "screened" and row["color"] == "red") or row["status"] == "ai_draft"
    if not ok:
        raise ValueError(f"cannot confirm status={row['status']} color={row['color']}")
    now = _now()
    cx.execute("UPDATE product_ratings SET status='confirmed', confirmed_at=?, "
               "updated_at=? WHERE product_key=?", (now, now, product_key))
    cx.commit()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/deploy-chat && python -m pytest tests/test_product_ratings.py -v`
Expected: PASS, 7 tests

- [ ] **Step 5: Mutation-verify the two state-machine guards**

1. In `record_screen`, delete the `if cur_status is not None and _RANK[cur_status] >= _RANK["ai_draft"]: return` guard. Run the file. `test_never_downgrades_a_confirmed_row` MUST fail. Revert.
2. In `set_tier2`, delete the `if row["status"] != "screened" ...: raise` guard. Run the file. `test_unrated_cannot_advance_to_tier2` MUST fail. Revert.

Run: `cd ~/deploy-chat && python -m pytest tests/test_product_ratings.py -v`
Expected: FAIL on the named test while mutated, PASS after revert.

- [ ] **Step 6: Run the whole Phase 1 suite, both orders**

Run:
```bash
cd ~/deploy-chat && python -m pytest \
  tests/test_purity_avoidlist.py tests/test_purity_screen.py tests/test_product_ratings.py -v
```
Expected: PASS, 21 tests. Then run the three files in reverse order; the total must match (guards against import-time contamination). Do NOT run the bare full suite — it sends real email.

- [ ] **Step 7: Commit**

```bash
git add dashboard/product_ratings.py tests/test_product_ratings.py
git commit -m "feat(purity): product_ratings table + never-downgrade state machine"
```

---

## Out of scope for Phase 1 (later phases)

- **Phase 2** — the on-request flow: gating (paid membership / explicit request), excipient acquisition (label photo / page scrape / staff entry), the formulation-analyzer tier-2 hand-off, and the confirm console. Reuses the product-review infrastructure.
- **Phase 3** — the two readers: the Fullscript seed-gate integration (red suppressed, yellow/green shown with color + `best_ff`) and the public aggregate `% fail` stat.

Nothing in Phase 1 touches `app.py`, the portal, or Fullscript. It is a self-contained engine with its own tests.
