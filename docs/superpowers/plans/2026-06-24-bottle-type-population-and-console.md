# Bottle-Type Population + Console Editor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Populate each storefront product's `bottle_type` from FileMaker + family rules, make assignments and the bottle catalog editable in the console, and add the `30ml` dropper type / rename `100cos`→`30g`.

**Architecture:** Builds on the geometric packer (same branch). `dashboard/shipping.py` gains dimension-aware bottle CRUD, a runtime per-product override table, and a resolver. A new `scripts/populate_bottle_types.py` writes the committed baseline into `products.json`. The existing static `/admin/shipping` page + `/api/shipping/*` endpoints are extended. Checkout resolves bottle type as: override table → `products.json` → `"default"`.

**Tech Stack:** Python 3, Flask, sqlite3 (`chat_log.db`), pytest, vanilla-JS static HTML. Stdlib only on the server side.

## Global Constraints

- No new dependencies. Pricing read-only — never write `usps_rates` or alter rate logic.
- Checkout never hard-fails: a `default`/unknown bottle type routes to the existing qty fallback.
- **Final bottle types (keys = fill content):** `5ml` (30×80), `15ml` (30×100), `30ml` (40×110, NEW), `50ml` (40×140), `100ml` (50×160), `30roll` (40×100), `30g` (70×70, renamed from `100cos`), `30cap` (50×90), `120cap` (80×100). Dims in mm.
- Box interiors (mm): S (50,150,230), M (130,220,270), L (140,290,300).
- DB access via `shipping._connect`/`sqlite3.Row`, optional `db_path` kwarg. Migrations idempotent, run from `init_shipping_schema(cx)`.
- Console endpoints under `/api/shipping/*`, `@require_console_key`, return `ok(...)`/`fail(...)`. Page is the static file `static/admin-shipping.html`.
- Family/packaging → bottle rules (priority order, write only where unset): infoceuticals (`source=="infoceutical-catalog"` or name `^(EI|ED|ES|ET|MB|MR)\d`) → `30ml`; eye drops → `5ml`; FMP join by normalized name → liquid ml→matching dropper, caps count ≤40→`30cap` / 41–140→`120cap`, `g` powder with FMP `type=="Pure Powders"`→`120cap` else→`30g`; everything else → unset + review list.

---

### Task 1: Rename `100cos`→`30g`, add `30ml` type, idempotent migration

**Files:**
- Modify: `dashboard/shipping.py` (`_STANDARD_BOTTLES`, `init_shipping_schema`)
- Modify: `tests/test_packing.py` (key rename + new entries)
- Modify: `scripts/infer_bottle_types.py` + `tests/test_infer_bottle_types.py` (rename `100cos`→`30g` references)
- Test: `tests/test_shipping.py`

**Interfaces:**
- Produces: catalog seeded with 9 types incl. `30ml` (40×110) and `30g` (renamed). Migration renames an existing `100cos` row to `30g` and inserts `30ml` if absent.

- [ ] **Step 1: Write the failing test** (in `tests/test_shipping.py`)

```python
def test_migration_renames_100cos_and_adds_30ml(tmp_path):
    import sqlite3
    from dashboard.shipping import init_shipping_schema, get_bottle_dims
    db = str(tmp_path / "chat_log.db")
    # Simulate an already-seeded older DB: insert a 100cos row, no 30ml
    with sqlite3.connect(db) as cx:
        cx.execute("CREATE TABLE bottle_types (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                   "name TEXT NOT NULL UNIQUE, notes TEXT, created_at TEXT NOT NULL "
                   "DEFAULT (datetime('now')))")
        cx.execute("ALTER TABLE bottle_types ADD COLUMN diameter_mm INTEGER")
        cx.execute("ALTER TABLE bottle_types ADD COLUMN height_mm INTEGER")
        cx.execute("INSERT INTO bottle_types (name, diameter_mm, height_mm) "
                   "VALUES ('100cos', 70, 70)")
        cx.commit()
        init_shipping_schema(cx)
    dims = get_bottle_dims(db_path=db)
    assert "100cos" not in dims
    assert dims["30g"] == (70, 70)
    assert dims["30ml"] == (40, 110)

def test_fresh_seed_has_30g_and_30ml_not_100cos(tmp_path):
    import sqlite3
    from dashboard.shipping import init_shipping_schema, get_bottle_dims
    db = str(tmp_path / "chat_log.db")
    with sqlite3.connect(db) as cx:
        init_shipping_schema(cx)
    dims = get_bottle_dims(db_path=db)
    assert dims["30g"] == (70, 70)
    assert dims["30ml"] == (40, 110)
    assert "100cos" not in dims
    assert len(dims) == 9
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /tmp/wt-deploy-chat-59a2725d && python3 -m pytest tests/test_shipping.py -k "migration_renames or fresh_seed_has" -q`
Expected: FAIL (`30g`/`30ml` absent; `100cos` present).

- [ ] **Step 3: Implement** — update `_STANDARD_BOTTLES` in `dashboard/shipping.py`:

```python
_STANDARD_BOTTLES = [
    ("120cap", "250 ml wide-mouth (120 caps / pure powder)", 80, 100),
    ("100ml", "100 ml dropper", 50, 160),
    ("30roll", "30 ml roll-on", 40, 100),
    ("50ml", "50 ml dropper", 40, 140),
    ("30ml", "30 ml dropper (infoceutical)", 40, 110),
    ("15ml", "15 ml dropper", 30, 100),
    ("5ml", "5 ml dropper (eye drops)", 30, 80),
    ("30g", "100 ml cosmetic jar (30 g powder)", 70, 70),
    ("30cap", "100 ml wide-mouth (30 caps)", 50, 90),
]
```

In `init_shipping_schema`, **after** the column-add migration and **before** the seed-on-empty block, add the idempotent rename + ensure-30ml migration:

```python
    # Rename legacy 100cos -> 30g if present and 30g not already there
    have = {r[0] for r in cx.execute("SELECT name FROM bottle_types")}
    if "100cos" in have and "30g" not in have:
        cx.execute("UPDATE bottle_types SET name='30g' WHERE name='100cos'")
        have.discard("100cos"); have.add("30g")
    # Ensure 30ml exists with dims (insert if missing)
    if "30ml" not in have:
        cx.execute("INSERT INTO bottle_types (name, notes, diameter_mm, height_mm) "
                   "VALUES ('30ml', '30 ml dropper (infoceutical)', 40, 110)")
```

(The seed-on-empty block already inserts all 9 from `_STANDARD_BOTTLES` for a brand-new DB; this migration covers DBs seeded before the rename/addition.)

- [ ] **Step 4: Update the packer tests** — in `tests/test_packing.py`, rename the `100cos` key to `30g` in `BOTTLES_MM` and `EXPECTED`, and add `30ml`:

```python
# in BOTTLES_MM:
    "30g": (70, 70),      # was "100cos"
    "30ml": (40, 110),
# in EXPECTED:
    "30g": (0, 9, 32),    # unchanged geometry, renamed key
    "30ml": (6, 36, 49),
```

- [ ] **Step 5: Update the old infer script for the rename** — in `scripts/infer_bottle_types.py` change `TYPES` and the powder branch return `"100cos"` → `"30g"`; in `tests/test_infer_bottle_types.py` change `test_powder_cosmetic` assertion `"100cos"` → `"30g"`.

- [ ] **Step 6: Run the full affected suites**

Run: `python3 -m pytest tests/test_shipping.py tests/test_packing.py tests/test_infer_bottle_types.py -q`
Expected: PASS (incl. the two new migration tests and the renamed/added packer counts).

- [ ] **Step 7: Commit**

```bash
git add dashboard/shipping.py tests/test_packing.py tests/test_shipping.py scripts/infer_bottle_types.py tests/test_infer_bottle_types.py
git commit -m "feat(shipping): rename 100cos->30g, add 30ml dropper type + migration"
```

---

### Task 2: Dimension-aware bottle CRUD

**Files:**
- Modify: `dashboard/shipping.py` (`add_bottle_type`, `update_bottle_type`)
- Test: `tests/test_shipping.py`

**Interfaces:**
- Produces: `add_bottle_type(name, diameter_mm=None, height_mm=None, notes=None, db_path=None) -> int`; `update_bottle_type(bottle_type_id, name, diameter_mm=None, height_mm=None, notes=None, db_path=None) -> None`.

- [ ] **Step 1: Write the failing test**

```python
def test_add_and_update_bottle_with_dims(tmp_path):
    import sqlite3
    from dashboard.shipping import (init_shipping_schema, add_bottle_type,
                                    update_bottle_type, get_bottle_dims)
    db = str(tmp_path / "chat_log.db")
    with sqlite3.connect(db) as cx:
        init_shipping_schema(cx)
    bid = add_bottle_type("250ml-spray", diameter_mm=55, height_mm=180, db_path=db)
    assert get_bottle_dims(db_path=db)["250ml-spray"] == (55, 180)
    update_bottle_type(bid, "250ml-spray", diameter_mm=60, height_mm=185, db_path=db)
    assert get_bottle_dims(db_path=db)["250ml-spray"] == (60, 185)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_shipping.py -k add_and_update_bottle_with_dims -q`
Expected: FAIL (TypeError — no `diameter_mm` kwarg).

- [ ] **Step 3: Implement** — replace `add_bottle_type` and `update_bottle_type`:

```python
def add_bottle_type(name, diameter_mm=None, height_mm=None, notes=None, db_path=None):
    with _connect(db_path) as cx:
        cur = cx.execute(
            "INSERT INTO bottle_types (name, notes, diameter_mm, height_mm) "
            "VALUES (?, ?, ?, ?)",
            (name, notes, diameter_mm, height_mm),
        )
        cx.commit()
        return int(cur.lastrowid)


def update_bottle_type(bottle_type_id, name, diameter_mm=None, height_mm=None,
                       notes=None, db_path=None):
    with _connect(db_path) as cx:
        cx.execute(
            "UPDATE bottle_types SET name=?, notes=?, diameter_mm=?, height_mm=? "
            "WHERE id=?",
            (name, notes, diameter_mm, height_mm, bottle_type_id),
        )
        cx.commit()
```

- [ ] **Step 4: Run to verify pass + no regressions**

Run: `python3 -m pytest tests/test_shipping.py -q`
Expected: PASS. (Existing callers pass `name`/`notes` positionally/by-keyword; new params default to None — verify the existing `add_bottle_type`/`update_bottle_type` tests still pass.)

- [ ] **Step 5: Commit**

```bash
git add dashboard/shipping.py tests/test_shipping.py
git commit -m "feat(shipping): bottle CRUD carries diameter/height"
```

---

### Task 3: Per-product override table + resolver

**Files:**
- Modify: `dashboard/shipping.py` (schema + override functions + resolver)
- Test: `tests/test_shipping.py`

**Interfaces:**
- Produces table `product_bottle_types(slug TEXT PK, bottle_type TEXT, updated_at TEXT)`.
- `list_product_bottle_overrides(db_path=None) -> dict[str,str]`
- `set_product_bottle_override(slug, bottle_type, db_path=None) -> None`
- `clear_product_bottle_override(slug, db_path=None) -> None`
- `resolve_bottle_type(slug, product, db_path=None) -> str` — override → `product.get("bottle_type")` → `"default"`. `product` is the product dict from `products.json`.

- [ ] **Step 1: Write the failing test**

```python
def test_product_override_crud_and_resolution(tmp_path):
    import sqlite3
    from dashboard.shipping import (init_shipping_schema, set_product_bottle_override,
        clear_product_bottle_override, list_product_bottle_overrides, resolve_bottle_type)
    db = str(tmp_path / "chat_log.db")
    with sqlite3.connect(db) as cx:
        init_shipping_schema(cx)
    # resolution with no override falls to products.json value, then default
    assert resolve_bottle_type("x", {"bottle_type": "15ml"}, db_path=db) == "15ml"
    assert resolve_bottle_type("y", {}, db_path=db) == "default"
    # override wins
    set_product_bottle_override("x", "30ml", db_path=db)
    assert resolve_bottle_type("x", {"bottle_type": "15ml"}, db_path=db) == "30ml"
    assert list_product_bottle_overrides(db_path=db)["x"] == "30ml"
    clear_product_bottle_override("x", db_path=db)
    assert resolve_bottle_type("x", {"bottle_type": "15ml"}, db_path=db) == "15ml"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_shipping.py -k product_override -q`
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement** — add the table in `init_shipping_schema` (with the other `CREATE TABLE`s):

```python
    cx.execute("""
        CREATE TABLE IF NOT EXISTS product_bottle_types (
            slug         TEXT PRIMARY KEY,
            bottle_type  TEXT NOT NULL,
            updated_at   TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
```

Add the functions (after `get_bottle_dims`):

```python
def list_product_bottle_overrides(db_path=None):
    with _connect(db_path) as cx:
        rows = cx.execute("SELECT slug, bottle_type FROM product_bottle_types").fetchall()
    return {r["slug"]: r["bottle_type"] for r in rows}


def set_product_bottle_override(slug, bottle_type, db_path=None):
    with _connect(db_path) as cx:
        cx.execute(
            "INSERT INTO product_bottle_types (slug, bottle_type, updated_at) "
            "VALUES (?, ?, datetime('now')) "
            "ON CONFLICT (slug) DO UPDATE SET bottle_type=excluded.bottle_type, "
            "updated_at=datetime('now')",
            (slug, bottle_type),
        )
        cx.commit()


def clear_product_bottle_override(slug, db_path=None):
    with _connect(db_path) as cx:
        cx.execute("DELETE FROM product_bottle_types WHERE slug=?", (slug,))
        cx.commit()


def resolve_bottle_type(slug, product, db_path=None):
    with _connect(db_path) as cx:
        row = cx.execute(
            "SELECT bottle_type FROM product_bottle_types WHERE slug=?", (slug,)
        ).fetchone()
    if row:
        return row["bottle_type"]
    return (product or {}).get("bottle_type") or "default"
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_shipping.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard/shipping.py tests/test_shipping.py
git commit -m "feat(shipping): per-product bottle-type override table + resolver"
```

---

### Task 4: Wire the resolver into checkout

**Files:**
- Modify: `app.py` (`_price_cart`, ~line 3412)
- Test: `tests/test_packing_integration.py`

**Interfaces:**
- Consumes: `shipping.resolve_bottle_type` (Task 3). The cart loop passes the product slug and dict.

- [ ] **Step 1: Write the failing test** (resolver precedence at the cart layer)

```python
def test_price_cart_uses_override(tmp_path, monkeypatch):
    """An override in product_bottle_types beats the products.json bottle_type."""
    import sqlite3
    from dashboard.shipping import init_shipping_schema, set_product_bottle_override, resolve_bottle_type
    db = str(tmp_path / "chat_log.db")
    with sqlite3.connect(db) as cx:
        init_shipping_schema(cx)
    set_product_bottle_override("foo", "30ml", db_path=db)
    # Direct resolver check (the function _price_cart will call)
    assert resolve_bottle_type("foo", {"bottle_type": "15ml"}, db_path=db) == "30ml"
    assert resolve_bottle_type("bar", {"bottle_type": "50ml"}, db_path=db) == "50ml"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_packing_integration.py -k uses_override -q`
Expected: FAIL until Task 3 is present (it is) — this test passes on Task 3 alone; its purpose is to lock the contract `_price_cart` depends on. Then wire `_price_cart`.

- [ ] **Step 3: Implement** — in `app.py` `_price_cart`, replace the bottle-type line:

Current:
```python
        bt = p.get("bottle_type") or "default"
```
New (the cart item `c` carries the slug; `p` is the product dict):
```python
        slug = (c.get("slug") or "").strip()
        bt = _shipping.resolve_bottle_type(slug, p)
        box_counts[bt] = box_counts.get(bt, 0) + qty
```
(Remove the now-duplicated `box_counts[bt] = ...` line that followed the old `bt =`.)

- [ ] **Step 4: Run to verify pass + no regressions**

Run: `python3 -m pytest tests/test_packing_integration.py tests/test_shipping.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_packing_integration.py
git commit -m "feat(shipping): checkout resolves bottle type via override->products->default"
```

---

### Task 5: FMP-join + family-rules populator

**Files:**
- Create: `scripts/populate_bottle_types.py`
- Test: `tests/test_populate_bottle_types.py`

**Interfaces:**
- `classify_from_fmp(fmp_row) -> str | None` — FMP packaging row → bottle key (or None if not classifiable).
- `family_rule(slug, product) -> str | None` — infoceutical/eye-drop family → key (or None).
- `build_assignments(products, fmp_by_name) -> {"assignments": {slug:key}, "review": [{slug,name,reason}]}` — priority: family_rule → classify_from_fmp(name match) → review. Never overwrites an existing `bottle_type`.
- CLI: dry-run prints counts + review; `--write` patches `data/products.json` only where unset.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_populate_bottle_types.py
from scripts.populate_bottle_types import classify_from_fmp, family_rule, build_assignments

def test_classify_liquid_and_caps_and_powder():
    assert classify_from_fmp({"zc_sold_display": "50ml", "sold_measurement": "ml", "type": ""}) == "50ml"
    assert classify_from_fmp({"zc_sold_display": "30pullulan", "sold_measurement": "pullulan", "type": ""}) == "30cap"
    assert classify_from_fmp({"zc_sold_display": "120vegicaps", "sold_measurement": "vegicaps", "type": ""}) == "120cap"
    assert classify_from_fmp({"zc_sold_display": "30g", "sold_measurement": "g", "type": "Pure Powders"}) == "120cap"
    assert classify_from_fmp({"zc_sold_display": "30g", "sold_measurement": "g", "type": "Functional Formulation"}) == "30g"
    assert classify_from_fmp({"zc_sold_display": "1000ml", "sold_measurement": "ml", "type": ""}) is None  # bulk -> review

def test_family_rule_infoceutical_and_eyedrops():
    assert family_rule("ei8-x", {"name": "EI8 Microbes", "source": "infoceutical-catalog"}) == "30ml"
    assert family_rule("mb1-x", {"name": "MB1 Brain Stem Hologram"}) == "30ml"
    assert family_rule("drops", {"name": "ACES Eyedrops"}) == "5ml"
    assert family_rule("z", {"name": "Quercetin"}) is None

def test_build_assignments_priority_and_review():
    products = {
        "ei8": {"name": "EI8 Microbes", "source": "infoceutical-catalog"},
        "cap": {"name": "Brain Boost"},
        "mystery": {"name": "Mystery Tonic"},
        "already": {"name": "Foo", "bottle_type": "15ml"},
    }
    fmp = {"brain boost": {"zc_sold_display": "30pullulan", "sold_measurement": "pullulan", "type": ""}}
    m = build_assignments(products, fmp)
    assert m["assignments"]["ei8"] == "30ml"     # family rule
    assert m["assignments"]["cap"] == "30cap"    # fmp join
    assert "already" not in m["assignments"]      # never overwrite
    assert any(r["slug"] == "mystery" for r in m["review"])  # unmatched -> review
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_populate_bottle_types.py -q`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement**

```python
# scripts/populate_bottle_types.py
"""Populate each storefront product's bottle_type from the FileMaker packaging
export + family rules. Re-runnable; never overwrites an existing assignment.
Dry-run by default; --write patches data/products.json (committed baseline)."""
from __future__ import annotations
import argparse, csv, json, os, re, sys

FMP_EXPORT = os.environ.get("FMP_PRODUCTS_CSV", "/tmp/fmp-export/newapp/products.csv")
_INFO_RE = re.compile(r'^(ei|ed|es|et|mb|mr)\s*\d', re.I)


def _norm(s):
    return re.sub(r'[^a-z0-9]+', ' ', (s or '').lower()).strip()


def family_rule(slug, product):
    name = product.get("name", "")
    src = product.get("source", "")
    if src == "infoceutical-catalog" or _INFO_RE.match(name.strip()):
        return "30ml"
    text = f"{name} {product.get('description','')}".lower()
    if "eye drop" in text or "eyedrop" in text:
        return "5ml"
    return None


def classify_from_fmp(row):
    disp = (row.get("zc_sold_display") or "").lower().replace(" ", "")
    meas = (row.get("sold_measurement") or "").lower().strip()
    ftype = (row.get("type") or "").strip()
    mml = re.match(r'^(\d+(?:\.\d+)?)ml$', disp)
    if mml or meas == "ml":
        ml = float(mml.group(1)) if mml else None
        return {5.0: "5ml", 15.0: "15ml", 50.0: "50ml", 100.0: "100ml"}.get(ml)  # 30/bulk -> None
    if any(x in disp for x in ("pullulan", "enteric", "vegicap", "gelcap", "capsule")) \
       or meas in ("pullulan", "enteric", "vegicaps", "gelcaps", "00 capsules"):
        mc = re.match(r'^(\d+)', disp)
        n = int(mc.group(1)) if mc else None
        if n is None:
            return None
        if n <= 40:
            return "30cap"
        if n <= 140:
            return "120cap"
        return None
    if disp.endswith("g") or meas == "g":
        return "120cap" if ftype == "Pure Powders" else "30g"
    return None


def build_assignments(products, fmp_by_name):
    assignments, review = {}, []
    for slug, p in products.items():
        if p.get("bottle_type"):
            continue
        key = family_rule(slug, p)
        if not key:
            row = fmp_by_name.get(_norm(p.get("name")))
            key = classify_from_fmp(row) if row else None
        if key:
            assignments[slug] = key
        else:
            review.append({"slug": slug, "name": p.get("name", ""),
                           "reason": "no family rule + no FMP packaging match"})
    return {"assignments": assignments, "review": review}


def _load_fmp(path):
    by_name = {}
    if not os.path.exists(path):
        return by_name
    for r in csv.DictReader(open(path)):
        by_name.setdefault(_norm(r.get("product_name")), r)
    return by_name


def _products_path():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = os.environ.get("DATA_DIR")
    if d and os.path.exists(os.path.join(d, "products.json")):
        return os.path.join(d, "products.json")
    return os.path.join(here, "data", "products.json")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)
    path = _products_path()
    with open(path) as f:
        doc = json.load(f)
    products = doc.get("products", {})
    fmp = _load_fmp(FMP_EXPORT)
    if not fmp:
        print(f"WARNING: no FMP export at {FMP_EXPORT} — only family rules will apply.")
    m = build_assignments(products, fmp)
    print(f"{len(m['assignments'])} products assigned; {len(m['review'])} need review.")
    for r in m["review"]:
        print(f"  REVIEW {r['slug']}: {r['name']!r} ({r['reason']})")
    if args.write:
        for slug, key in m["assignments"].items():
            products[slug]["bottle_type"] = key
        with open(path, "w") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
        print(f"Wrote {len(m['assignments'])} assignments to {path}")
    else:
        print("(dry run — pass --write to patch products.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests + a dry-run over the real catalog**

Run: `python3 -m pytest tests/test_populate_bottle_types.py -q` → PASS.
Run: `python3 scripts/populate_bottle_types.py` → prints assigned/review counts, no file change. Capture counts in the report. (Do NOT `--write`; that's operator-run under Glen's review.)

- [ ] **Step 5: Commit**

```bash
git add scripts/populate_bottle_types.py tests/test_populate_bottle_types.py
git commit -m "feat(shipping): FMP-join + family-rules bottle_type populator"
```

---

### Task 6: Console extensions on `/admin/shipping`

**Files:**
- Modify: `app.py` (extend `/api/shipping/bottles` POST+PATCH for dims; add `/api/shipping/packing-settings` GET+POST; add `/api/shipping/product-bottles` GET + POST + DELETE)
- Modify: `static/admin-shipping.html` (catalog dims + add-type; padding section; products section)
- Test: `tests/test_admin_shipping_api.py` (create — endpoint-level)

**Interfaces:**
- Consumes Tasks 2/3: `add_bottle_type`/`update_bottle_type` (dims), `get_packing_settings`/`set_packing_setting`, `list_product_bottle_overrides`/`set_product_bottle_override`/`clear_product_bottle_override`, `get_bottle_dims`, `dashboard.products.load_products`.

- [ ] **Step 1: Read the existing page + endpoints** to match the pattern: `app.py` `/api/shipping/*` handlers (≈19970–20075) and `static/admin-shipping.html`. Follow the same `@require_console_key`, `ok`/`fail`, and JS `fetch` conventions. Do not restyle existing sections.

- [ ] **Step 2: Write the failing API test**

```python
# tests/test_admin_shipping_api.py
import json, sqlite3, importlib

def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    db = str(tmp_path / "chat_log.db")
    from dashboard.shipping import init_shipping_schema
    with sqlite3.connect(db) as cx:
        init_shipping_schema(cx)
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)  # no auth in test
    import app as appmod
    importlib.reload(appmod)
    return appmod.app.test_client()

def test_set_bottle_dims_via_patch(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    bottles = c.get("/api/shipping/bottles").get_json()["data"]
    bid = next(b["id"] for b in bottles if b["name"] == "30ml")
    r = c.patch(f"/api/shipping/bottles/{bid}",
                json={"name": "30ml", "diameter_mm": 41, "height_mm": 112})
    assert r.status_code == 200
    from dashboard.shipping import get_bottle_dims
    assert get_bottle_dims(db_path=str(tmp_path / "chat_log.db"))["30ml"] == (41, 112)

def test_packing_settings_get_and_post(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    assert c.get("/api/shipping/packing-settings").get_json()["data"]["wrap_mm"] == 6
    assert c.post("/api/shipping/packing-settings", json={"wrap_mm": 8}).status_code == 200
    assert c.get("/api/shipping/packing-settings").get_json()["data"]["wrap_mm"] == 8

def test_product_bottle_override_endpoints(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    assert c.post("/api/shipping/product-bottles",
                  json={"slug": "foo", "bottle_type": "30ml"}).status_code == 200
    listing = c.get("/api/shipping/product-bottles").get_json()["data"]
    assert listing["overrides"]["foo"] == "30ml"
    assert c.delete("/api/shipping/product-bottles/foo").status_code == 200
```

(If `app` import has heavy import-time side effects that make `reload` impractical, the implementer may instead test the handlers via the existing app test fixture/conftest pattern — match whatever `tests/` already uses for Flask routes; keep the asserted behavior identical.)

- [ ] **Step 3: Run to verify it fails**

Run: `python3 -m pytest tests/test_admin_shipping_api.py -q`
Expected: FAIL (endpoints missing / dims not applied).

- [ ] **Step 4: Implement endpoints** — extend the bottles POST/PATCH to read dims, and add the new routes (place beside the existing `/api/shipping/*` handlers):

```python
# In api_shipping_add_bottle: after reading name/notes
        diameter_mm = body.get("diameter_mm")
        height_mm = body.get("height_mm")
        new_id = _shipping.add_bottle_type(
            name, diameter_mm=diameter_mm, height_mm=height_mm, notes=notes)

# In api_shipping_update_bottle: after reading name/notes
        diameter_mm = body.get("diameter_mm")
        height_mm = body.get("height_mm")
        _shipping.update_bottle_type(
            bid, name, diameter_mm=diameter_mm, height_mm=height_mm, notes=notes)
```

(Update `list_bottle_types` consumers in the UI to show dims — `list_bottle_types` already returns all columns; if it doesn't include the new dim columns, have it `SELECT` them. Confirm `list_bottle_types` returns `diameter_mm`/`height_mm`; if not, add them to its SELECT.)

```python
@app.route("/api/shipping/packing-settings", methods=["GET"])
@require_console_key
def api_shipping_packing_settings_get():
    try: return ok(_shipping.get_packing_settings())
    except Exception as e: return fail(e)


@app.route("/api/shipping/packing-settings", methods=["POST"])
@require_console_key
def api_shipping_packing_settings_set():
    try:
        body = request.get_json(silent=True) or {}
        for k in ("wrap_mm", "box_margin_mm"):
            if k in body:
                _shipping.set_packing_setting(k, int(body[k]))
        return ok(_shipping.get_packing_settings())
    except (TypeError, ValueError) as e: return fail(e, status=400)
    except Exception as e: return fail(e)


@app.route("/api/shipping/product-bottles", methods=["GET"])
@require_console_key
def api_shipping_product_bottles_get():
    try:
        from dashboard import products as _products
        prods = _products.load_products()
        overrides = _shipping.list_product_bottle_overrides()
        items = [{
            "slug": slug,
            "name": p.get("name"),
            "resolved": _shipping.resolve_bottle_type(slug, p),
            "override": overrides.get(slug),
            "baseline": p.get("bottle_type"),
        } for slug, p in prods.items()]
        return ok({"items": items, "overrides": overrides})
    except Exception as e: return fail(e)


@app.route("/api/shipping/product-bottles", methods=["POST"])
@require_console_key
def api_shipping_product_bottles_set():
    try:
        body = request.get_json(silent=True) or {}
        slug = (body.get("slug") or "").strip()
        bt = (body.get("bottle_type") or "").strip()
        if not slug or not bt:
            return fail("slug and bottle_type required", status=400)
        _shipping.set_product_bottle_override(slug, bt)
        return ok({"slug": slug, "bottle_type": bt})
    except Exception as e: return fail(e)


@app.route("/api/shipping/product-bottles/<path:slug>", methods=["DELETE"])
@require_console_key
def api_shipping_product_bottles_clear(slug):
    try:
        _shipping.clear_product_bottle_override(slug)
        return ok({"cleared": slug})
    except Exception as e: return fail(e)
```

- [ ] **Step 5: Implement the UI** — extend `static/admin-shipping.html`, matching the existing page's markup/JS style:
  - **Bottle catalog section:** add editable `diameter_mm` / `height_mm` inputs per row (PATCH on blur/save), and an "Add bottle type" row (key + dims → POST). Show current dims from the bottles GET.
  - **Padding section:** two number inputs `wrap_mm` / `box_margin_mm` wired to GET/POST `/api/shipping/packing-settings`.
  - **Products section:** fetch `/api/shipping/product-bottles`; render a table of products with a `<select>` (options = catalog bottle keys) bound to `resolved`; products whose `override` is null AND `baseline` is null (i.e. resolve to `default`) sorted to the top with a badge. Changing the select → POST; a "clear override" control → DELETE.

- [ ] **Step 6: Run the API tests + full affected suites**

Run: `python3 -m pytest tests/test_admin_shipping_api.py tests/test_shipping.py -q`
Expected: PASS. Manually load `/admin/shipping?key=...` is out of scope for automated tests; the endpoint tests cover the data path.

- [ ] **Step 7: Commit**

```bash
git add app.py static/admin-shipping.html tests/test_admin_shipping_api.py
git commit -m "feat(shipping): console dims editor, padding knobs, per-product bottle assignment"
```

---

## Self-Review

**Spec coverage:** rename+add type (T1), dims CRUD (T2), override table+resolver (T3), checkout wiring (T4), populator (T5), console UI (T6). ✓
**Placeholder scan:** complete code for backend; UI task gives endpoint code + concrete section contracts referencing the existing page. ✓
**Type consistency:** `resolve_bottle_type(slug, product, db_path)` signature consistent T3↔T4↔T6; `add_bottle_type`/`update_bottle_type` dim params consistent T2↔T6; populator `build_assignments` shape consistent with its tests. ✓
**Deferred:** `30ml` real dims now known (40×110, in seed); operator runs `populate_bottle_types.py --write` under Glen's review; essences without FMP match fall to review.
