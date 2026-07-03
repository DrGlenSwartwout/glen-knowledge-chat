# Dispensary Ranking + Practice-Type Recommendations — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development or superpowers:executing-plans to implement task-by-task. Steps use checkbox (`- [ ]`) syntax. Build in the `deploy-chat` worktree (`sess/547d1328`).

**Goal:** Turn the practitioner portal's Clients tab into a dispense-intelligence surface: a channel-split, unit-ranked table of products the practitioner moves, plus a curated, practice-type list of the next Functional Formulations to add.

**Architecture:** One new pure/defensive module `dashboard/dispensary_stats.py` (a pure ranker + a DB collector + a curated-recommendations resolver), a curated data file, two new payload keys wired into `portal_data()`, and two render functions in the existing portal page. Per-product units come from the main `orders.items_json`, reached by joining the practitioner's aggregate `wholesale_orders` / `dispensary_orders` rows (keyed by `invoice_id`) to `orders.external_ref`.

**Tech stack:** Python (sqlite3, `chat_log.db` via `DATA_DIR`), `dashboard/orders.py`, `data/products.json`, vanilla-JS `static/practitioner-portal.html`, pytest.

## Global constraints (verbatim)

- **Spec:** `docs/superpowers/specs/2026-07-03-dispensary-ranking-recommendations-design.md`.
- **Channels:** Dispensed = practitioner's own `wholesale_orders`; Drop-shipped = `dispensary_orders`; Patient portal = **0 placeholder** (deferred, rendered but de-emphasized "coming soon").
- **Product name → link:** `/begin/product/<slug>`, always `target="_blank" rel="noopener"`.
- **Recommendations:** curated per practice type (practitioner `credentials`), `default` fallback, exclude already-dispensed slugs. Blurbs = **structure/function language only** (supports / promotes / helps maintain), never disease claims; one-line disclaimer under the section.
- **DB:** `wholesale_orders`, `dispensary_orders`, and the main `orders` table all live in `chat_log.db` (`DATA_DIR`). Product names from `data/products.json` (`["products"][slug]["name"]`); unknown slug falls back to the slug string.
- **Defensive:** every DB read is best-effort and never raises (a failure yields an empty block, matching `portal_view.py` conventions).
- **Isolation:** all edits/commits in the `deploy-chat` worktree.

## File map

| File | Responsibility |
|------|----------------|
| Create `dashboard/dispensary_stats.py` | `rank_dispense_rows` (pure), `dispense_stats` (collector), `recommended_ffs` (resolver) |
| Create `data/practice_recommendations.json` | Curated practice-type → `[{slug, blurb}]` |
| Create `tests/test_dispensary_stats.py` | Unit tests for all three functions |
| Modify `dashboard/practitioner_portal.py:portal_data()` | Add `data["dispense_stats"]` + `data["recommended_ffs"]` |
| Modify `static/practitioner-portal.html` (Clients pane) | Render the ranked table + recommendations |

---

### Task 1: Pure ranker `rank_dispense_rows`

**Files:** Create `dashboard/dispensary_stats.py`; Test `tests/test_dispensary_stats.py`.

**Interfaces produced:**
`rank_dispense_rows(dispensed, dropshipped, patient_portal, *, catalog=None) -> list[dict]` — the three args are `{slug: units}` maps; returns rows `{"slug","name","url","dispensed","dropshipped","patient_portal","total"}` sorted by `total` desc then `name`. `catalog` is the products map (injectable; `None` loads `data/products.json`).

- [ ] **Step 1 — failing test:**
```python
from dashboard import dispensary_stats as ds

CAT = {"bone-builder": {"name": "Bone Builder"}, "nous-energy": {"name": "Nous Energy"}}

def test_rank_merges_channels_and_sorts_by_total():
    rows = ds.rank_dispense_rows({"bone-builder": 10, "nous-energy": 2},
                                 {"bone-builder": 5}, {}, catalog=CAT)
    assert [r["slug"] for r in rows] == ["bone-builder", "nous-energy"]  # 15 vs 2
    top = rows[0]
    assert top["name"] == "Bone Builder"
    assert top["url"] == "/begin/product/bone-builder"
    assert top["dispensed"] == 10 and top["dropshipped"] == 5
    assert top["patient_portal"] == 0 and top["total"] == 15

def test_rank_unknown_slug_falls_back_to_slug_name():
    rows = ds.rank_dispense_rows({"mystery-x": 3}, {}, {}, catalog=CAT)
    assert rows[0]["name"] == "mystery-x"
```

- [ ] **Step 2 — run, expect FAIL** (`AttributeError: rank_dispense_rows`). `python3 -m pytest tests/test_dispensary_stats.py -q`
- [ ] **Step 3 — implement:**
```python
"""Dispensary product-dispense ranking + practice-type FF recommendations.
Pure ranker + defensive DB collector + curated-recommendations resolver.
Self-contained; DB reads never raise (a failure yields an empty result)."""
import json
from pathlib import Path

_DATA = Path(__file__).resolve().parent.parent / "data"


def _catalog(catalog=None) -> dict:
    if catalog is not None:
        return catalog
    try:
        return json.loads((_DATA / "products.json").read_text()).get("products", {})
    except Exception:
        return {}


def _name(slug, cat) -> str:
    return (cat.get(slug) or {}).get("name") or slug


def _url(slug) -> str:
    return f"/begin/product/{slug}"


def rank_dispense_rows(dispensed, dropshipped, patient_portal, *, catalog=None):
    cat = _catalog(catalog)
    slugs = set(dispensed) | set(dropshipped) | set(patient_portal)
    rows = []
    for s in slugs:
        d, ds_, pp = int(dispensed.get(s, 0)), int(dropshipped.get(s, 0)), int(patient_portal.get(s, 0))
        rows.append({"slug": s, "name": _name(s, cat), "url": _url(s),
                     "dispensed": d, "dropshipped": ds_, "patient_portal": pp,
                     "total": d + ds_ + pp})
    rows.sort(key=lambda r: (-r["total"], r["name"].lower()))
    return rows
```
- [ ] **Step 4 — run, expect PASS.**
- [ ] **Step 5 — commit:** `feat(dispensary): pure rank_dispense_rows`

### Task 2: Collector `dispense_stats` (join to orders.items_json)

**Files:** Modify `dashboard/dispensary_stats.py`; Test `tests/test_dispensary_stats.py`.

**Interfaces produced:** `dispense_stats(practitioner_id, *, db_path=None, catalog=None) -> list[dict]` — same row shape as Task 1; reads the practitioner's invoice_ids and their `orders.items_json`. Never raises.

**Consumes:** `wholesale_orders(invoice_id, practitioner_id)` and `dispensary_orders(invoice_id, practitioner_id)` (both `chat_log.db`); `orders(external_ref, items_json)` (`chat_log.db`, `dashboard/orders.py`). `items_json` = `[{"slug","qty",...}]`.

- [ ] **Step 1 — verify the join key** (documented investigation, not a guess):
  Run `grep -nE "record_order\(|record_dispensary_order\(|external_ref|out.get\(.invoice_id" app.py | head` and confirm the checkout writes the main `orders` row with `external_ref == invoice_id`. If the key differs (e.g. `orders.id` or a `doc_number`), use that column in the query below. Confirm all three tables are in `chat_log.db` (`DATA_DIR`).
- [ ] **Step 2 — failing test** (seed a temp `chat_log.db` with the three tables + one order per channel):
```python
import sqlite3, json as _json

def _seed(tmp_path):
    p = tmp_path / "chat_log.db"
    cx = sqlite3.connect(p)
    cx.executescript(
        "CREATE TABLE wholesale_orders(invoice_id TEXT PRIMARY KEY, practitioner_id TEXT);"
        "CREATE TABLE dispensary_orders(invoice_id TEXT PRIMARY KEY, practitioner_id TEXT, bottles INT);"
        "CREATE TABLE orders(id INTEGER PRIMARY KEY, source TEXT, external_ref TEXT, items_json TEXT);")
    cx.execute("INSERT INTO wholesale_orders VALUES('INV1','p1')")
    cx.execute("INSERT INTO dispensary_orders VALUES('INV2','p1',3)")
    cx.execute("INSERT INTO orders(source,external_ref,items_json) VALUES('a','INV1',?)",
               (_json.dumps([{"slug":"bone-builder","qty":10}]),))
    cx.execute("INSERT INTO orders(source,external_ref,items_json) VALUES('b','INV2',?)",
               (_json.dumps([{"slug":"bone-builder","qty":3},{"slug":"nous-energy","qty":1}]),))
    cx.commit(); cx.close()
    return str(p)

def test_dispense_stats_buckets_by_channel(tmp_path):
    from dashboard import dispensary_stats as ds
    rows = ds.dispense_stats("p1", db_path=_seed(tmp_path), catalog=CAT)
    by = {r["slug"]: r for r in rows}
    assert by["bone-builder"]["dispensed"] == 10
    assert by["bone-builder"]["dropshipped"] == 3
    assert by["bone-builder"]["total"] == 13 and rows[0]["slug"] == "bone-builder"
    assert by["nous-energy"]["dropshipped"] == 1 and by["nous-energy"]["patient_portal"] == 0

def test_dispense_stats_never_raises_on_bad_db():
    from dashboard import dispensary_stats as ds
    assert ds.dispense_stats("p1", db_path="/nonexistent/x.db") == []
```
- [ ] **Step 3 — run, expect FAIL.**
- [ ] **Step 4 — implement** (uses the join column confirmed in Step 1; shown with `external_ref`):
```python
import sqlite3
import os

def _log_db(db_path=None) -> str:
    if db_path:
        return db_path
    base = os.environ.get("DATA_DIR") or str(Path(__file__).resolve().parent.parent)
    return str(Path(base) / "chat_log.db")


def _items_for_invoices(cx, invoice_ids):
    """{slug: units} summed across the given orders' items_json (by external_ref)."""
    out = {}
    for inv in invoice_ids:
        row = cx.execute("SELECT items_json FROM orders WHERE external_ref=? LIMIT 1", (inv,)).fetchone()
        if not row or not row[0]:
            continue
        try:
            for it in json.loads(row[0]):
                s = it.get("slug")
                if s:
                    out[s] = out.get(s, 0) + int(it.get("qty") or 0)
        except Exception:
            continue
    return out


def dispense_stats(practitioner_id, *, db_path=None, catalog=None):
    try:
        with sqlite3.connect(_log_db(db_path)) as cx:
            w = [r[0] for r in cx.execute(
                "SELECT invoice_id FROM wholesale_orders WHERE practitioner_id=?", (str(practitioner_id),))]
            d = [r[0] for r in cx.execute(
                "SELECT invoice_id FROM dispensary_orders WHERE practitioner_id=?", (str(practitioner_id),))]
            dispensed = _items_for_invoices(cx, w)
            dropshipped = _items_for_invoices(cx, d)
    except Exception:
        return []
    return rank_dispense_rows(dispensed, dropshipped, {}, catalog=catalog)
```
- [ ] **Step 5 — run, expect PASS.**
- [ ] **Step 6 — commit:** `feat(dispensary): dispense_stats collector over orders.items_json`

### Task 3: Curated recommendations `recommended_ffs` + data file

**Files:** Create `data/practice_recommendations.json`; Modify `dashboard/dispensary_stats.py`; Test `tests/test_dispensary_stats.py`.

**Interfaces produced:** `recommended_ffs(practice_type, *, exclude_slugs=(), recs_path=None, catalog=None) -> list[dict]` → `[{"slug","name","url","blurb"}]` in curated order, `practice_type` resolved case-insensitively else `default`, minus `exclude_slugs`.

- [ ] **Step 1 — create `data/practice_recommendations.json`** (seed: a `default` list + Health Coach + OD; blurbs in structure/function language, seeded from formulation data, Glen-editable):
```json
{
  "default": [
    {"slug": "bone-builder", "blurb": "Supports healthy bone density and mineral balance."},
    {"slug": "nous-energy", "blurb": "Promotes mental clarity and steady cellular energy."}
  ],
  "Health Coach": [
    {"slug": "vitality", "blurb": "Supports daily energy and whole-body resilience."}
  ],
  "OD": [
    {"slug": "refreshing-vision", "blurb": "Helps maintain macular pigment and visual comfort."}
  ]
}
```
  (Slugs above are placeholders for real catalog slugs — pick real ones from `data/products.json` when authoring; the resolver is agnostic to which slugs appear.)
- [ ] **Step 2 — failing test:**
```python
def test_recommended_resolves_type_then_default_and_excludes():
    from dashboard import dispensary_stats as ds
    recs = {"default": [{"slug": "a", "blurb": "A."}, {"slug": "b", "blurb": "B."}],
            "OD": [{"slug": "c", "blurb": "C."}]}
    import json as j, tempfile, os
    p = os.path.join(tempfile.mkdtemp(), "r.json"); open(p, "w").write(j.dumps(recs))
    cat = {"a": {"name": "A"}, "b": {"name": "B"}, "c": {"name": "C"}}
    od = ds.recommended_ffs("od", recs_path=p, catalog=cat)      # case-insensitive
    assert [r["slug"] for r in od] == ["c"] and od[0]["url"] == "/begin/product/c"
    dflt = ds.recommended_ffs("Unknown", exclude_slugs=["a"], recs_path=p, catalog=cat)
    assert [r["slug"] for r in dflt] == ["b"]                    # default minus excluded
```
- [ ] **Step 3 — run FAIL → implement:**
```python
def recommended_ffs(practice_type, *, exclude_slugs=(), recs_path=None, catalog=None):
    try:
        path = recs_path or str(_DATA / "practice_recommendations.json")
        recs = json.loads(Path(path).read_text())
    except Exception:
        return []
    key = next((k for k in recs if k.lower() == (practice_type or "").strip().lower()), "default")
    cat = _catalog(catalog)
    ex = {s for s in (exclude_slugs or ())}
    out = []
    for r in recs.get(key, []):
        s = r.get("slug")
        if not s or s in ex:
            continue
        out.append({"slug": s, "name": _name(s, cat), "url": _url(s), "blurb": r.get("blurb", "")})
    return out
```
- [ ] **Step 4 — run, expect PASS.**
- [ ] **Step 5 — commit:** `feat(dispensary): curated practice-type FF recommendations`

### Task 4: Wire both into `portal_data`

**Files:** Modify `dashboard/practitioner_portal.py`; Test `tests/test_dispensary_stats.py` (or `test_practitioner_portal.py`).

**Consumes:** `dispense_stats`, `recommended_ffs`. The practitioner's `credentials` is already selected in `portal_data`'s query — add it if absent.

- [ ] **Step 1 — failing test:** assert `portal_data(...)` (with a seeded practitioner + orders, `include_orders=True`) returns `data["dispense_stats"]` (a list) and `data["recommended_ffs"]` (a list), and that recommendations exclude any slug present in `dispense_stats`.
- [ ] **Step 2 — implement** (after the `data["training"] = ...` line):
```python
    from dashboard import dispensary_stats as _dstats
    try:
        stats = _dstats.dispense_stats(practitioner_id, db_path=db_path)
        data["dispense_stats"] = stats
        data["recommended_ffs"] = _dstats.recommended_ffs(
            row.get("credentials", ""), exclude_slugs=[r["slug"] for r in stats])
    except Exception:
        data["dispense_stats"] = []
        data["recommended_ffs"] = []
```
  (Ensure `credentials` is in the SELECT column list; add it if missing.)
- [ ] **Step 3 — run, expect PASS.** Then `python3 -m pytest tests/test_dispensary_stats.py tests/test_practitioner_portal.py -q`.
- [ ] **Step 4 — commit:** `feat(dispensary): dispense_stats + recommendations in portal-data`

### Task 5: Clients-tab UI

**Files:** Modify `static/practitioner-portal.html` (Clients pane + scoped CSS + `render()`).

- [ ] **Step 1 — markup:** in the Clients pane, below the existing dispensary share-link/credit block, add `<div class="panel" id="dispense-rank"></div>` (Section 1) and `<div class="panel" id="reco-ffs"></div>` (Section 2, lighter: muted border, smaller text).
- [ ] **Step 2 — `renderDispenseStats(d)`:** build a table (Product · Dispensed · Drop-shipped · Patient portal · Total) from `d.dispense_stats`; Product is `<a href="url" target="_blank" rel="noopener">name</a>`; the Patient-portal header carries a small "coming soon" note; empty `dispense_stats` → a friendly empty-state line instead of a table.
- [ ] **Step 3 — `renderRecommended(d)`:** from `d.recommended_ffs`, render a lighter list, each `<a target="_blank">name</a>` followed by its `blurb`; add the one-line structure/function disclaimer; hide the panel when the list is empty.
- [ ] **Step 4 — call both** at the end of `render(d)`.
- [ ] **Step 5 — verify (headless render):** inject a mock `d` with `dispense_stats` + `recommended_ffs` (and the empty cases); confirm the ranked table renders sorted, names open `/begin/product/<slug>` in a new tab, recommendations are visibly lighter and exclude dispensed items, Patient-portal shows the placeholder, both empty-states behave, zero console errors.
- [ ] **Step 6 — commit:** `feat(dispensary): Clients-tab ranked table + practice-type recommendations`

---

## Self-review

- **Coverage:** Section 1 ranked table (Tasks 1–2, UI Task 5), Section 2 recommendations (Task 3, UI Task 5), payload wiring (Task 4), channels + `/begin/product` links + structure/function blurbs + patient-portal placeholder all covered.
- **Type consistency:** row shape `{slug,name,url,dispensed,dropshipped,patient_portal,total}` is identical in Tasks 1, 2, 5; recommendation shape `{slug,name,url,blurb}` identical in Tasks 3–5.
- **The one live-schema dependency** is the `invoice_id → orders` join column (Task 2, Step 1) — a documented verification step, not a placeholder; the query swaps one column name if it differs from `external_ref`.
- **Out of scope:** real patient-portal numbers (placeholder), data-driven recommendations.
