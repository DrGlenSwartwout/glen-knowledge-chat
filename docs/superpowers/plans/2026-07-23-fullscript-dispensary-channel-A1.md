# Fullscript Dispensary Channel — Phase A1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the Fullscript dispensary as a separately-listed portal channel parallel to E4L and PRL, dark behind a flag, with the pinned and scan drivers working end to end.

**Architecture:** `dashboard/fullscript.py` is a direct sibling of `dashboard/prl_supplement.py`: schema and queries only, pure sqlite, caller passes `cx`, no Flask imports. A payload builder `_fullscript_for(email, scan_date)` in `app.py` sits beside `_prl_supplement_for`. A `/fs/<token>/<product_slug>` route records the click and 302s outbound to Fullscript. The client portal renders one card. Catalog data is a committed seed file; the running app never contacts Fullscript.

**Tech Stack:** Python 3, Flask, sqlite3 (Postgres in prod via the db adapter), pytest, vanilla JS in `static/client-portal.html`.

**Spec:** `docs/superpowers/specs/2026-07-23-fullscript-dispensary-channel-design.md`

## Global Constraints

Every task's requirements implicitly include this section.

- **Prod runs Postgres.** `cur.lastrowid` **raises** on the PG adapter. Use `RETURNING id` when an insert's new id is needed. Prefer `dashboard.dbwrite.insert_or_replace` over hand-rolled upserts, exactly as `prl_supplement.sync_from_seed` does.
- **Flag off means the key is absent.** With `FULLSCRIPT_ENABLED` unset, the portal payload must never gain a `fullscript` key. Responses stay byte-identical to today.
- **Relation vocabulary is `complement` or `substitute`,** defaulting to `consider` when null. Copied verbatim from PRL (`_prl_ff_view`, `app.py:21632`) so both cards render consistently.
- **Identity comes from the portal token only.** Never from a request field, query param, or body. This is guarded by a cross-client isolation test.
- **Redirect destinations are never taken from the request.** Built from a hardcoded `https://us.fullscript.com` base plus config plus a database row.
- **Attribution-safe default:** phase A routes every client to `https://us.fullscript.com/welcome/{FULLSCRIPT_DISPENSARY_SLUG}/store-start`, **not** the product deep link. Whether a new signup from a deep link attaches to the Remedy Match dispensary is unconfirmed; guessing wrong loses the margin. Deep-link mode is written but gated off.
- **No runtime calls to Fullscript.** The seed generator is an offline script under `scripts/`, never invoked by the app.
- **Never run the bare full test suite** — it sends real email. Run named test files.
- **`$DATA_DIR` strips `products.json` in the full suite.** Any test touching the catalog must pin `load_products` to the repo file.
- **All tables live in `LOG_DB`,** same as PRL and `recommendation_events`.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `dashboard/fullscript.py` (create) | schema, seed sync, driver queries, resolver. No Flask. |
| `scripts/build_fullscript_seed.py` (create) | offline seed generator. Run by hand, never by the app. |
| `data/fullscript_seed.json` (create) | committed catalog. Source of truth. |
| `app.py` (modify) | `_fullscript_enabled()`, `_fullscript_ff_view()`, `_fullscript_for()`, `/fs/` route, payload wiring |
| `static/client-portal.html` (modify) | the card |
| `tests/test_fullscript_module.py` (create) | schema, seed sync, driver queries |
| `tests/test_fullscript_resolver.py` (create) | dedupe and origin priority |
| `tests/test_fullscript_builder.py` (create) | payload builder + flag |
| `tests/test_fullscript_redirect.py` (create) | route, destination safety, isolation |

---

### Task 1: Module schema and seed sync

**Files:**
- Create: `dashboard/fullscript.py`
- Create: `tests/test_fullscript_module.py`

**Interfaces:**
- Consumes: `dashboard.dbwrite.insert_or_replace`
- Produces: `init_tables(cx)`, `sync_from_seed(cx, seed) -> dict` with keys `products`, `focus_area_products`, `focus_area_items`

- [ ] **Step 1: Write the failing test**

Create `tests/test_fullscript_module.py`:

```python
import sqlite3
from dashboard import fullscript as fs

SEED = {
  "products": [
    {"name": "Magnesium Taurate", "external_id": "U3ByZWU6OlByb2R1Y3QtMTA3Njc2",
     "product_slug": "magnesium-taurate", "brand": "Jarrow Formulas", "url": None,
     "focus_tags": ["Nervous System"], "product_type": "supplement",
     "best_ff": "Neuro Magnesium", "relation": "substitute", "ff_alts": [],
     "source": "seed", "active": 1},
    {"name": "Pure Taurine 500mg", "external_id": "U3ByZWU6OlByb2R1Y3QtNjc1NjE",
     "product_slug": "pure-taurine-500-mg-100-caps", "brand": "Montiff", "url": None,
     "focus_tags": [], "product_type": "supplement",
     "best_ff": None, "relation": None, "ff_alts": [], "source": "seed", "active": 1},
  ],
  "focus_area_products": [
    {"focus_area_id": 9, "focus_area_name": "Nervous System",
     "fs_product_name": "Magnesium Taurate", "rank": 0},
    {"focus_area_id": 9, "focus_area_name": "Nervous System",
     "fs_product_name": "Pure Taurine 500mg", "rank": 1},
  ],
  "focus_area_items": [
    {"focus_area_id": 9, "item_code": "ED4"},
    {"focus_area_id": 9, "item_code": "EI1"},
    {"focus_area_id": 14, "item_code": "ED8"},
  ],
}


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    fs.init_tables(cx)
    fs.sync_from_seed(cx, SEED)
    return cx


def test_sync_counts_and_idempotent():
    cx = _cx()
    c = fs.sync_from_seed(cx, SEED)  # second run
    assert c["products"] == 2 and c["focus_area_products"] == 2
    assert cx.execute("SELECT COUNT(*) FROM fullscript_products").fetchone()[0] == 2


def test_all_seven_tables_exist():
    cx = _cx()
    names = {r[0] for r in cx.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert {"fullscript_products", "fullscript_focus_area_products",
            "fullscript_focus_area_items", "fullscript_condition_products",
            "fullscript_client_pins", "fullscript_review_links",
            "fullscript_clicks"} <= names


def test_product_columns_roundtrip():
    cx = _cx()
    r = cx.execute("SELECT * FROM fullscript_products WHERE name=?",
                   ("Magnesium Taurate",)).fetchone()
    assert r["external_id"] == "U3ByZWU6OlByb2R1Y3QtMTA3Njc2"
    assert r["brand"] == "Jarrow Formulas"
    assert r["product_slug"] == "magnesium-taurate"
    assert r["best_ff"] == "Neuro Magnesium" and r["relation"] == "substitute"
    assert r["source"] == "seed" and r["active"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/deploy-chat && python -m pytest tests/test_fullscript_module.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dashboard.fullscript'`

- [ ] **Step 3: Write minimal implementation**

Create `dashboard/fullscript.py`:

```python
"""Fullscript dispensary channel data (pure sqlite; caller passes cx).
Sibling of dashboard/prl_supplement.py. Owns schema + queries only.

Fullscript is a SEPARATELY LISTED channel, like E4L and PRL. It deliberately
does NOT write to recommendation_events: that table's product_key is a
storefront slug, and both the portal recommendations block and the console 360
hub resolve keys against the storefront catalog. Fullscript products have no
storefront slug, so they would render broken there. Clicks live in
fullscript_clicks instead.
"""
import json


def init_tables(cx):
    cx.execute("""CREATE TABLE IF NOT EXISTS fullscript_products (
        name TEXT PRIMARY KEY, brand TEXT, external_id TEXT, product_slug TEXT,
        url TEXT, focus_tags TEXT, product_type TEXT, best_ff TEXT, relation TEXT,
        ff_alts TEXT, source TEXT, active INTEGER DEFAULT 1)""")
    cx.execute("""CREATE TABLE IF NOT EXISTS fullscript_focus_area_products (
        focus_area_id INTEGER, focus_area_name TEXT, fs_product_name TEXT, rank INTEGER)""")
    cx.execute("""CREATE TABLE IF NOT EXISTS fullscript_focus_area_items (
        focus_area_id INTEGER, item_code TEXT)""")
    cx.execute("""CREATE TABLE IF NOT EXISTS fullscript_condition_products (
        condition_key TEXT, fs_product_name TEXT, rank INTEGER)""")
    cx.execute("""CREATE TABLE IF NOT EXISTS fullscript_client_pins (
        email TEXT, fs_product_name TEXT, note TEXT, pinned_by TEXT, pinned_at TEXT)""")
    cx.execute("""CREATE TABLE IF NOT EXISTS fullscript_review_links (
        review_id INTEGER, fs_product_name TEXT, rank INTEGER, created_at TEXT)""")
    cx.execute("""CREATE TABLE IF NOT EXISTS fullscript_clicks (
        email TEXT, fs_product_name TEXT, origin TEXT, clicked_at TEXT)""")
    cx.execute("CREATE INDEX IF NOT EXISTS ix_fsfai_code "
               "ON fullscript_focus_area_products(focus_area_id, rank)")
    cx.execute("CREATE INDEX IF NOT EXISTS ix_fsfa_item "
               "ON fullscript_focus_area_items(item_code)")
    cx.execute("CREATE INDEX IF NOT EXISTS ix_fspins_email "
               "ON fullscript_client_pins(email)")
    cx.execute("CREATE INDEX IF NOT EXISTS ix_fsprod_slug "
               "ON fullscript_products(product_slug)")
    cx.commit()


def sync_from_seed(cx, seed):
    """Idempotent full replace of the three reference tables. Pins, review links
    and clicks are client data and are never touched."""
    cx.execute("DELETE FROM fullscript_products")
    cx.execute("DELETE FROM fullscript_focus_area_products")
    cx.execute("DELETE FROM fullscript_focus_area_items")
    cx.execute("DELETE FROM fullscript_condition_products")
    from dashboard import dbwrite
    for p in seed.get("products", []):
        dbwrite.insert_or_replace(
            cx, "fullscript_products",
            ("name", "brand", "external_id", "product_slug", "url", "focus_tags",
             "product_type", "best_ff", "relation", "ff_alts", "source", "active"),
            (p["name"], p.get("brand"), p.get("external_id"), p.get("product_slug"),
             p.get("url"), json.dumps(p.get("focus_tags") or []), p.get("product_type"),
             p.get("best_ff"), p.get("relation"), json.dumps(p.get("ff_alts") or []),
             p.get("source") or "seed", 1 if p.get("active", 1) else 0),
            conflict_cols=("name",))
    for fp in seed.get("focus_area_products", []):
        cx.execute("""INSERT INTO fullscript_focus_area_products
            (focus_area_id, focus_area_name, fs_product_name, rank) VALUES (?,?,?,?)""",
            (fp["focus_area_id"], fp.get("focus_area_name"), fp["fs_product_name"],
             fp.get("rank", 0)))
    for fi in seed.get("focus_area_items", []):
        cx.execute("INSERT INTO fullscript_focus_area_items "
                   "(focus_area_id, item_code) VALUES (?,?)",
                   (fi["focus_area_id"], fi["item_code"]))
    cx.commit()
    return {"products": len(seed.get("products", [])),
            "focus_area_products": len(seed.get("focus_area_products", [])),
            "focus_area_items": len(seed.get("focus_area_items", []))}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/deploy-chat && python -m pytest tests/test_fullscript_module.py -v`
Expected: PASS, 3 tests

- [ ] **Step 5: Commit**

```bash
git add dashboard/fullscript.py tests/test_fullscript_module.py
git commit -m "feat(fullscript): channel schema + idempotent seed sync"
```

---

### Task 2: Driver queries

**Files:**
- Modify: `dashboard/fullscript.py`
- Modify: `tests/test_fullscript_module.py`

**Interfaces:**
- Consumes: Task 1's tables
- Produces:
  - `focus_areas_for_items(cx, item_codes) -> [{"focus_area_id": int, "focus_area_name": str, "hits": int}]`
  - `products_for_focus_area(cx, focus_area_id) -> [product_dict]`
  - `pins_for_client(cx, email) -> [product_dict]` where `product_dict` has keys `name`, `brand`, `product_slug`, `external_id`, `best_ff`, `relation`, `note`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_fullscript_module.py`:

```python
def test_focus_areas_for_items_ranked():
    cx = _cx()
    fas = fs.focus_areas_for_items(cx, ["ED4", "EI1", "ED8"])
    assert fas[0]["focus_area_id"] == 9 and fas[0]["hits"] == 2
    assert any(f["focus_area_id"] == 14 for f in fas)


def test_focus_areas_for_items_empty_input():
    cx = _cx()
    assert fs.focus_areas_for_items(cx, []) == []
    assert fs.focus_areas_for_items(cx, None) == []


def test_products_for_focus_area_joined_and_ordered():
    cx = _cx()
    ps = fs.products_for_focus_area(cx, 9)
    assert [p["name"] for p in ps] == ["Magnesium Taurate", "Pure Taurine 500mg"]
    assert ps[0]["best_ff"] == "Neuro Magnesium"
    assert ps[0]["external_id"] == "U3ByZWU6OlByb2R1Y3QtMTA3Njc2"


def test_products_for_focus_area_skips_inactive():
    cx = _cx()
    cx.execute("UPDATE fullscript_products SET active=0 WHERE name=?",
               ("Pure Taurine 500mg",))
    ps = fs.products_for_focus_area(cx, 9)
    assert [p["name"] for p in ps] == ["Magnesium Taurate"]


def test_pins_for_client():
    cx = _cx()
    assert fs.pins_for_client(cx, "a@b.com") == []
    cx.execute("INSERT INTO fullscript_client_pins "
               "(email, fs_product_name, note, pinned_by, pinned_at) VALUES (?,?,?,?,?)",
               ("a@b.com", "Magnesium Taurate", "start here", "glen", "2026-07-23"))
    pins = fs.pins_for_client(cx, "A@B.com")  # case-insensitive
    assert len(pins) == 1
    assert pins[0]["name"] == "Magnesium Taurate" and pins[0]["note"] == "start here"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/deploy-chat && python -m pytest tests/test_fullscript_module.py -v`
Expected: FAIL — `AttributeError: module 'dashboard.fullscript' has no attribute 'focus_areas_for_items'`

- [ ] **Step 3: Write minimal implementation**

Append to `dashboard/fullscript.py`:

```python
_PRODUCT_COLS = ("p.name AS name, p.brand AS brand, p.product_slug AS product_slug, "
                 "p.external_id AS external_id, p.best_ff AS best_ff, "
                 "p.relation AS relation")


def focus_areas_for_items(cx, item_codes):
    """Focus areas whose infoceuticals include any of item_codes, ranked by hit count."""
    codes = [c for c in (item_codes or []) if c]
    if not codes:
        return []
    q = ("SELECT i.focus_area_id, COALESCE(n.focus_area_name, '') AS focus_area_name, "
         "COUNT(*) AS hits FROM fullscript_focus_area_items i "
         "LEFT JOIN (SELECT DISTINCT focus_area_id, focus_area_name "
         "           FROM fullscript_focus_area_products) n "
         "  ON n.focus_area_id = i.focus_area_id "
         f"WHERE i.item_code IN ({','.join('?' * len(codes))}) "
         "GROUP BY i.focus_area_id ORDER BY hits DESC, i.focus_area_id")
    return [dict(r) for r in cx.execute(q, codes).fetchall()]


def products_for_focus_area(cx, focus_area_id):
    rows = cx.execute(f"""
        SELECT {_PRODUCT_COLS}
        FROM fullscript_focus_area_products fap
        JOIN fullscript_products p ON p.name = fap.fs_product_name
        WHERE fap.focus_area_id = ? AND p.active = 1
        ORDER BY fap.rank""", (focus_area_id,)).fetchall()
    return [dict(r) for r in rows]


def pins_for_client(cx, email):
    e = (email or "").strip().lower()
    if not e:
        return []
    rows = cx.execute(f"""
        SELECT {_PRODUCT_COLS}, pin.note AS note
        FROM fullscript_client_pins pin
        JOIN fullscript_products p ON p.name = pin.fs_product_name
        WHERE LOWER(pin.email) = ? AND p.active = 1
        ORDER BY pin.pinned_at""", (e,)).fetchall()
    return [dict(r) for r in rows]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/deploy-chat && python -m pytest tests/test_fullscript_module.py -v`
Expected: PASS, 8 tests

- [ ] **Step 5: Commit**

```bash
git add dashboard/fullscript.py tests/test_fullscript_module.py
git commit -m "feat(fullscript): scan focus-area and client-pin driver queries"
```

---

### Task 3: The resolver

**Files:**
- Modify: `dashboard/fullscript.py`
- Create: `tests/test_fullscript_resolver.py`

**Interfaces:**
- Consumes: Task 2's `focus_areas_for_items`, `products_for_focus_area`, `pins_for_client`
- Produces: `candidates_for(cx, email, item_codes=None) -> [candidate]` where a candidate is a product dict plus `origin` (one of `pinned`, `review`, `scan`, `condition`), `reason` (str), and `focus_area_name` (str or None)

Origin priority is `pinned` > `review` > `scan` > `condition`. A product appearing from two drivers keeps the higher-priority origin and appears exactly once.

- [ ] **Step 1: Write the failing test**

Create `tests/test_fullscript_resolver.py`:

```python
"""candidates_for: unions the drivers, dedupes by product, keeps the
highest-priority origin. Pins are an explicit clinical decision by Glen and
therefore outrank anything derived."""
import sqlite3
from dashboard import fullscript as fs

SEED = {
  "products": [
    {"name": "Mag Taurate", "brand": "Jarrow", "product_slug": "mag-taurate",
     "external_id": "P1", "best_ff": "Neuro Magnesium", "relation": "substitute",
     "focus_tags": [], "ff_alts": [], "product_type": "supplement", "url": None,
     "source": "seed", "active": 1},
    {"name": "Taurine", "brand": "Montiff", "product_slug": "taurine",
     "external_id": "P2", "best_ff": None, "relation": None,
     "focus_tags": [], "ff_alts": [], "product_type": "supplement", "url": None,
     "source": "seed", "active": 1},
  ],
  "focus_area_products": [
    {"focus_area_id": 9, "focus_area_name": "Nervous System",
     "fs_product_name": "Mag Taurate", "rank": 0},
    {"focus_area_id": 9, "focus_area_name": "Nervous System",
     "fs_product_name": "Taurine", "rank": 1},
  ],
  "focus_area_items": [{"focus_area_id": 9, "item_code": "ED4"}],
}


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    fs.init_tables(cx)
    fs.sync_from_seed(cx, SEED)
    return cx


def test_scan_only():
    cx = _cx()
    out = fs.candidates_for(cx, "a@b.com", item_codes=["ED4"])
    assert [c["name"] for c in out] == ["Mag Taurate", "Taurine"]
    assert all(c["origin"] == "scan" for c in out)
    assert out[0]["focus_area_name"] == "Nervous System"


def test_pin_outranks_scan_and_dedupes():
    cx = _cx()
    cx.execute("INSERT INTO fullscript_client_pins "
               "(email, fs_product_name, note, pinned_by, pinned_at) VALUES (?,?,?,?,?)",
               ("a@b.com", "Taurine", "for sleep", "glen", "2026-07-23"))
    out = fs.candidates_for(cx, "a@b.com", item_codes=["ED4"])
    names = [c["name"] for c in out]
    assert names.count("Taurine") == 1, "deduped, not listed twice"
    assert names[0] == "Taurine", "pinned sorts first"
    taurine = out[0]
    assert taurine["origin"] == "pinned" and taurine["reason"] == "for sleep"


def test_no_drivers_yields_nothing():
    cx = _cx()
    assert fs.candidates_for(cx, "a@b.com", item_codes=[]) == []


def test_unknown_client_yields_nothing():
    cx = _cx()
    assert fs.candidates_for(cx, "nobody@nowhere.com", item_codes=None) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/deploy-chat && python -m pytest tests/test_fullscript_resolver.py -v`
Expected: FAIL — `AttributeError: module 'dashboard.fullscript' has no attribute 'candidates_for'`

- [ ] **Step 3: Write minimal implementation**

Append to `dashboard/fullscript.py`:

```python
# Lower number wins. A pin is an explicit clinical decision, so it outranks
# anything derived; after that, the more client-specific the evidence the higher.
ORIGIN_PRIORITY = {"pinned": 0, "review": 1, "scan": 2, "condition": 3}


def candidates_for(cx, email, item_codes=None):
    """Union the drivers, dedupe by product name keeping the highest-priority
    origin, then sort by that priority. Phase A1 wires the pinned and scan
    drivers; review and condition land in later phases and slot in here."""
    found = {}

    def offer(prod, origin, reason, focus_area_name=None):
        name = prod.get("name")
        if not name:
            return
        prior = found.get(name)
        if prior and ORIGIN_PRIORITY[prior["origin"]] <= ORIGIN_PRIORITY[origin]:
            return
        c = dict(prod)
        c["origin"] = origin
        c["reason"] = reason or ""
        c["focus_area_name"] = focus_area_name
        c.pop("note", None)
        found[name] = c

    for p in pins_for_client(cx, email):
        offer(p, "pinned", p.get("note") or "")

    for fa in focus_areas_for_items(cx, item_codes):
        fa_name = fa.get("focus_area_name") or ""
        for p in products_for_focus_area(cx, fa["focus_area_id"]):
            offer(p, "scan", fa_name, fa_name)

    return sorted(found.values(), key=lambda c: ORIGIN_PRIORITY[c["origin"]])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/deploy-chat && python -m pytest tests/test_fullscript_resolver.py -v`
Expected: PASS, 4 tests

- [ ] **Step 5: Commit**

```bash
git add dashboard/fullscript.py tests/test_fullscript_resolver.py
git commit -m "feat(fullscript): candidate resolver with origin priority + dedupe"
```

---

### Task 4: Offline seed generator and starter seed

**Files:**
- Create: `scripts/build_fullscript_seed.py`
- Create: `data/fullscript_seed.json`

**Interfaces:**
- Consumes: nothing in the app
- Produces: `data/fullscript_seed.json` in the three-part shape Task 1's `sync_from_seed` reads

This script is run **by hand**. The app must never import it. It queries the public unauthenticated Fullscript catalog at browsing volume, as documented in the spec.

- [ ] **Step 1: Write the generator**

Create `scripts/build_fullscript_seed.py`:

```python
#!/usr/bin/env python3
"""Build data/fullscript_seed.json from Fullscript's PUBLIC open catalog.

Run BY HAND, never by the app. The app makes no runtime calls to Fullscript.

The endpoint backing fullscript.com/catalog needs no authentication, but it is
undocumented and only permits allowlisted operations (arbitrary GraphQL returns
HTTP 400). It expects `variables` as a JSON-ENCODED STRING, not an object.

Usage:
    python3 scripts/build_fullscript_seed.py > data/fullscript_seed.json

Then review the output by hand: `best_ff` mappings are guesses and must be
corrected by Glen before the seed is committed.
"""
import json
import sys
import time
import urllib.request

ENDPOINT = "https://fullscript.com/api/fs-graphql"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")

TYPEAHEAD = """
  query TypeaheadSearchV2_Shared_Query($query: String, $filters: SearchFilterObject) {
    viewer {
      typeaheadSearchV2(query: $query, filters: $filters, useSkuIdentifier: true) {
        entityType
        data { id name entityType brandName productSlug }
      }
    }
  }
"""

# focus_area_id -> (focus area name, [search terms], {product name: best Functional
# Formulations equivalent}). Mirrors the focus areas PRL already covers so the two
# channels reach parity. Extend as Glen confirms mappings.
FOCUS_AREAS = {
    9: ("Nervous System", ["magnesium taurate", "l-theanine"], {}),
}


def search(term):
    body = json.dumps({
        "query": TYPEAHEAD,
        "variables": json.dumps({
            "query": term,
            "filters": {"list": ["PRODUCTS", "BRANDS", "INGREDIENTS"]},
        }),
    }).encode()
    req = urllib.request.Request(
        ENDPOINT, data=body,
        headers={"Content-Type": "application/json", "User-Agent": UA,
                 "Origin": "https://fullscript.com",
                 "Referer": "https://fullscript.com/catalog"})
    with urllib.request.urlopen(req, timeout=30) as r:
        payload = json.load(r)
    if payload.get("errors"):
        raise SystemExit(f"catalog error for {term!r}: {payload['errors']}")
    groups = payload["data"]["viewer"]["typeaheadSearchV2"]
    for g in groups:
        if g.get("entityType") == "Product":
            return g.get("data") or []
    return []


def main():
    products, fa_products = {}, []
    for fa_id, (fa_name, terms, ff_map) in sorted(FOCUS_AREAS.items()):
        rank = 0
        for term in terms:
            for hit in search(term):
                name = hit.get("name")
                if not name or name in products:
                    continue
                products[name] = {
                    "name": name,
                    "brand": hit.get("brandName"),
                    "external_id": hit.get("id"),
                    "product_slug": hit.get("productSlug"),
                    "url": None,
                    "focus_tags": [fa_name],
                    "product_type": "supplement",
                    "best_ff": ff_map.get(name),
                    "relation": "substitute" if ff_map.get(name) else None,
                    "ff_alts": [],
                    "source": "seed",
                    "active": 1,
                }
                fa_products.append({"focus_area_id": fa_id, "focus_area_name": fa_name,
                                    "fs_product_name": name, "rank": rank})
                rank += 1
            time.sleep(1.0)  # browsing volume, deliberately unhurried
    json.dump({"products": sorted(products.values(), key=lambda p: p["name"]),
               "focus_area_products": fa_products,
               "focus_area_items": []},
              sys.stdout, indent=1, ensure_ascii=False)
    sys.stdout.write("\n")
    print(f"{len(products)} products across {len(FOCUS_AREAS)} focus areas",
          file=sys.stderr)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it and verify it produces valid seed shape**

Run:
```bash
cd ~/deploy-chat && python3 scripts/build_fullscript_seed.py > /tmp/fs_seed_check.json && \
python3 -c "
import json
d = json.load(open('/tmp/fs_seed_check.json'))
assert set(d) == {'products','focus_area_products','focus_area_items'}, set(d)
assert d['products'], 'no products returned'
p = d['products'][0]
assert p['external_id'] and p['name'], p
print('OK', len(d['products']), 'products; first:', p['name'], '|', p['brand'])
"
```
Expected: `OK <n> products; first: ... | ...`

If the catalog endpoint has changed shape and this fails, stop and report. Do not fall back to scraping or to any authenticated route.

- [ ] **Step 3: Commit the generator and the generated seed**

Copy the verified output into place, then commit:

```bash
cd ~/deploy-chat && cp /tmp/fs_seed_check.json data/fullscript_seed.json
git add scripts/build_fullscript_seed.py data/fullscript_seed.json
git commit -m "feat(fullscript): offline seed generator + starter catalog seed"
```

Flag to Glen in the task report: `best_ff` mappings are empty or guessed and need his review before the card goes live. The channel works without them; the FF chip simply does not render.

---

### Task 5: Payload builder and flag

**Files:**
- Modify: `app.py` — add beside `_prl_supplement_for` (currently at `app.py:21642`) and beside `_prl_supplement_enabled` (currently at `app.py:18262`)
- Modify: `app.py:20011` region — the portal payload assembly that sets `prl_supplement_enabled`
- Create: `tests/test_fullscript_builder.py`

**Interfaces:**
- Consumes: `dashboard.fullscript.candidates_for`
- Produces: `app._fullscript_enabled() -> bool`, `app._fullscript_for(email, scan_date) -> dict | None`

The returned dict has keys `dispensary_url` (str) and `groups` (list of `{"origin": str, "heading": str, "products": [...]}`), each product carrying `name`, `brand`, `product_slug`, `reason`, and `ff` (the `_prl_ff_view`-shaped dict or None).

- [ ] **Step 1: Write the failing test**

Create `tests/test_fullscript_builder.py`:

```python
"""_fullscript_for: flag-gated Fullscript channel card builder. Mirrors
_prl_supplement_for. Covers derive-from-scan, pin priority, the default-OFF
flag, and the byte-identical guarantee that the payload key is absent when off.
"""
import sqlite3

import app as app_mod
from dashboard import fullscript as fs

SEED = {
  "products": [
    {"name": "Mag Taurate", "brand": "Jarrow", "product_slug": "mag-taurate",
     "external_id": "P1", "best_ff": "Neuro Magnesium", "relation": "substitute",
     "focus_tags": [], "ff_alts": [], "product_type": "supplement", "url": None,
     "source": "seed", "active": 1},
  ],
  "focus_area_products": [
    {"focus_area_id": 9, "focus_area_name": "Nervous System",
     "fs_product_name": "Mag Taurate", "rank": 0},
  ],
  "focus_area_items": [{"focus_area_id": 9, "item_code": "ED4"}],
}


def _seed(cx):
    fs.init_tables(cx)
    fs.sync_from_seed(cx, SEED)
    cx.execute("""CREATE TABLE IF NOT EXISTS scan_recommendations
        (email TEXT, scan_id TEXT, scan_date TEXT, item_code TEXT,
         priority_rank INTEGER, label TEXT)""")
    cx.execute("INSERT INTO scan_recommendations "
               "VALUES ('a@b.com','s1','2026-07-01','ED4',1,'ED4 - Nerve')")
    cx.commit()


def _db(tmp_path):
    db = str(tmp_path / "c.db")
    cx = sqlite3.connect(db)
    cx.row_factory = sqlite3.Row
    _seed(cx)
    cx.close()
    return db


def test_flag_off_returns_none(monkeypatch, tmp_path):
    monkeypatch.setattr(app_mod, "LOG_DB", _db(tmp_path))
    monkeypatch.delenv("FULLSCRIPT_ENABLED", raising=False)
    assert app_mod._fullscript_for("a@b.com", "2026-07-01") is None


def test_derive_builds_card(monkeypatch, tmp_path):
    monkeypatch.setattr(app_mod, "LOG_DB", _db(tmp_path))
    monkeypatch.setenv("FULLSCRIPT_ENABLED", "1")
    monkeypatch.setenv("FULLSCRIPT_DISPENSARY_SLUG", "remedymatch")
    out = app_mod._fullscript_for("a@b.com", "2026-07-01")
    assert out["dispensary_url"] == \
        "https://us.fullscript.com/welcome/remedymatch/store-start"
    g = out["groups"][0]
    assert g["origin"] == "scan" and g["heading"] == "Matched from your scan"
    p = g["products"][0]
    assert p["name"] == "Mag Taurate" and p["brand"] == "Jarrow"
    assert p["ff"]["name"] == "Neuro Magnesium"
    assert p["ff"]["relation"] == "substitute"


def test_no_candidates_returns_none(monkeypatch, tmp_path):
    monkeypatch.setattr(app_mod, "LOG_DB", _db(tmp_path))
    monkeypatch.setenv("FULLSCRIPT_ENABLED", "1")
    assert app_mod._fullscript_for("nobody@nowhere.com", None) is None


def test_never_raises_on_bad_db(monkeypatch):
    monkeypatch.setattr(app_mod, "LOG_DB", "/nonexistent/dir/nope.db")
    monkeypatch.setenv("FULLSCRIPT_ENABLED", "1")
    assert app_mod._fullscript_for("a@b.com", None) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/deploy-chat && python -m pytest tests/test_fullscript_builder.py -v`
Expected: FAIL — `AttributeError: module 'app' has no attribute '_fullscript_for'`

- [ ] **Step 3: Write minimal implementation**

Add near `_prl_supplement_enabled` (`app.py:18262`):

```python
def _fullscript_enabled():
    """Default OFF. When off the portal payload never gains a `fullscript` key,
    so responses stay byte-identical."""
    return (os.environ.get("FULLSCRIPT_ENABLED", "") or "").strip().lower() in (
        "1", "true", "yes", "on")


def _fullscript_dispensary_url():
    """Attribution-safe entry point. Phase A routes every client here rather than
    to a product deep link: whether a NEW signup from /u/catalog/product/... is
    attached to the Remedy Match dispensary is unconfirmed, and guessing wrong
    loses the margin. Built from a hardcoded base + config, never from a request."""
    slug = (os.environ.get("FULLSCRIPT_DISPENSARY_SLUG", "") or "remedymatch").strip()
    return f"https://us.fullscript.com/welcome/{slug}/store-start"
```

Add beside `_prl_supplement_for` (after it, around `app.py:21730`):

```python
_FULLSCRIPT_HEADINGS = {
    "pinned": "Chosen for you",
    "review": "Replaces something you're taking",
    "scan": "Matched from your scan",
    "condition": "For what you're working on",
}


def _fullscript_ff_view(best_ff, relation):
    """Same shape as _prl_ff_view so both channel cards render identically."""
    if not best_ff:
        return None
    try:
        slug = _resolve_remedy_slug({"name": best_ff})
    except Exception:
        slug = None
    return {"name": best_ff, "relation": relation or "consider", "slug": slug}


def _fullscript_for(email, scan_date):
    """The client's Fullscript channel card, or None (flag off / no candidates).
    Best-effort: any error returns None, never raises."""
    if not _fullscript_enabled():
        return None
    try:
        from dashboard import fullscript as _fs
        with db.connect(LOG_DB) as cx:
            cx.row_factory = sqlite3.Row
            _fs.init_tables(cx)
            email_norm = (email or "").strip().lower()
            sd = (scan_date or "").strip()
            if sd:
                rows = cx.execute(
                    "SELECT item_code FROM scan_recommendations "
                    "WHERE email=? AND scan_date=? ORDER BY priority_rank",
                    (email_norm, sd)).fetchall()
            else:
                rows = cx.execute(
                    "SELECT item_code FROM scan_recommendations "
                    "WHERE email=? ORDER BY scan_date DESC, priority_rank",
                    (email_norm,)).fetchall()
            codes = [r["item_code"] for r in rows]
            cands = _fs.candidates_for(cx, email_norm, item_codes=codes)
        if not cands:
            return None
        groups, seen = [], {}
        for c in cands:
            g = seen.get(c["origin"])
            if g is None:
                g = {"origin": c["origin"],
                     "heading": _FULLSCRIPT_HEADINGS.get(c["origin"], "Recommended"),
                     "products": []}
                seen[c["origin"]] = g
                groups.append(g)
            g["products"].append({
                "name": c.get("name"),
                "brand": c.get("brand"),
                "product_slug": c.get("product_slug"),
                "reason": c.get("reason") or "",
                "ff": _fullscript_ff_view(c.get("best_ff"), c.get("relation")),
            })
        return {"dispensary_url": _fullscript_dispensary_url(), "groups": groups}
    except Exception:
        return None
```

Wire into the portal payload beside the PRL lines (`app.py:20011`):

```python
    payload["fullscript_enabled"] = _fullscript_enabled()
    if _fullscript_enabled():
        try:
            _fs_block = _fullscript_for(email_for_reports, req_date or None)
            if _fs_block:
                payload["fullscript"] = _fs_block
        except Exception:
            pass
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/deploy-chat && python -m pytest tests/test_fullscript_builder.py -v`
Expected: PASS, 4 tests

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_fullscript_builder.py
git commit -m "feat(fullscript): flag-gated portal payload builder"
```

---

### Task 6: Tracked outbound redirect

**Files:**
- Modify: `app.py` — add near `email_click_redirect` (`app.py:7664`)
- Modify: `dashboard/fullscript.py` — add `record_click`
- Create: `tests/test_fullscript_redirect.py`

**Interfaces:**
- Consumes: `app._portal_record_for(cx, token)`, `dashboard.fullscript`
- Produces: route `GET /fs/<token>/<product_slug>`; `dashboard.fullscript.record_click(cx, email, fs_product_name, origin)`

- [ ] **Step 1: Write the failing test**

Create `tests/test_fullscript_redirect.py`:

```python
"""GET /fs/<token>/<product_slug>: records a click then 302s OUTBOUND to
Fullscript. Identity comes from the portal token only. The destination is built
from a hardcoded base + config + the DB row, never from the request, so the
route is structurally incapable of becoming an open redirect."""
import sqlite3
import pytest

import app as app_mod
from dashboard import fullscript as fs

SEED = {
  "products": [
    {"name": "Mag Taurate", "brand": "Jarrow", "product_slug": "mag-taurate",
     "external_id": "P1", "best_ff": None, "relation": None, "focus_tags": [],
     "ff_alts": [], "product_type": "supplement", "url": None,
     "source": "seed", "active": 1},
    {"name": "Retired Thing", "brand": "X", "product_slug": "retired-thing",
     "external_id": "P9", "best_ff": None, "relation": None, "focus_tags": [],
     "ff_alts": [], "product_type": "supplement", "url": None,
     "source": "seed", "active": 0},
  ],
  "focus_area_products": [], "focus_area_items": [],
}


@pytest.fixture
def client(monkeypatch, tmp_path):
    db = str(tmp_path / "c.db")
    cx = sqlite3.connect(db)
    cx.row_factory = sqlite3.Row
    fs.init_tables(cx)
    fs.sync_from_seed(cx, SEED)
    cx.commit()
    cx.close()
    monkeypatch.setattr(app_mod, "LOG_DB", db)
    monkeypatch.setenv("FULLSCRIPT_DISPENSARY_SLUG", "remedymatch")
    monkeypatch.setattr(app_mod, "_portal_record_for",
                        lambda cx, token: {"email": "a@b.com"} if token == "TOKA"
                        else ({"email": "b@b.com"} if token == "TOKB" else None))
    app_mod.app.config["TESTING"] = True
    return app_mod.app.test_client()


def _clicks(db):
    cx = sqlite3.connect(db)
    cx.row_factory = sqlite3.Row
    rows = [dict(r) for r in cx.execute("SELECT * FROM fullscript_clicks").fetchall()]
    cx.close()
    return rows


def test_redirects_to_dispensary_and_records(client, tmp_path):
    r = client.get("/fs/TOKA/mag-taurate")
    assert r.status_code == 302
    assert r.headers["Location"] == \
        "https://us.fullscript.com/welcome/remedymatch/store-start"
    rows = _clicks(app_mod.LOG_DB)
    assert len(rows) == 1
    assert rows[0]["email"] == "a@b.com"
    assert rows[0]["fs_product_name"] == "Mag Taurate"


def test_unknown_token_records_nothing_and_goes_home(client):
    r = client.get("/fs/NOPE/mag-taurate")
    assert r.status_code == 302 and r.headers["Location"] == "/"
    assert _clicks(app_mod.LOG_DB) == []


def test_unknown_slug_goes_home(client):
    r = client.get("/fs/TOKA/not-a-real-slug")
    assert r.status_code == 302 and r.headers["Location"] == "/"
    assert _clicks(app_mod.LOG_DB) == []


def test_inactive_product_goes_home(client):
    r = client.get("/fs/TOKA/retired-thing")
    assert r.status_code == 302 and r.headers["Location"] == "/"
    assert _clicks(app_mod.LOG_DB) == []


def test_destination_never_comes_from_the_request(client):
    """An attacker-supplied absolute URL in the slug position must not be honored."""
    r = client.get("/fs/TOKA/https:%2F%2Fevil.example.com")
    assert r.status_code == 302
    assert "evil.example.com" not in r.headers["Location"]
    assert r.headers["Location"] == "/"


def test_cross_client_isolation(client):
    """Client B's token records against B, never against A."""
    client.get("/fs/TOKB/mag-taurate")
    rows = _clicks(app_mod.LOG_DB)
    assert [r["email"] for r in rows] == ["b@b.com"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/deploy-chat && python -m pytest tests/test_fullscript_redirect.py -v`
Expected: FAIL — all six 404, because the route does not exist

- [ ] **Step 3: Write minimal implementation**

Append to `dashboard/fullscript.py`:

```python
def product_by_slug(cx, product_slug):
    """Active product row for a Fullscript product slug, or None."""
    s = (product_slug or "").strip().lower()
    if not s:
        return None
    r = cx.execute("SELECT * FROM fullscript_products "
                   "WHERE LOWER(product_slug) = ? AND active = 1", (s,)).fetchone()
    return dict(r) if r else None


def record_click(cx, email, fs_product_name, origin=""):
    from datetime import datetime, timezone
    cx.execute("INSERT INTO fullscript_clicks "
               "(email, fs_product_name, origin, clicked_at) VALUES (?,?,?,?)",
               ((email or "").strip().lower(), fs_product_name, origin or "",
                datetime.now(timezone.utc).isoformat(timespec="seconds")))
    cx.commit()
```

Add to `app.py` after `email_click_redirect`:

```python
@app.route("/fs/<token>/<product_slug>", methods=["GET"])
def fullscript_click_redirect(token, product_slug):
    """Tracked OUTBOUND redirect into Glen's Fullscript dispensary. Identity is
    server-resolved from the portal token only. The destination is built from a
    hardcoded base + config + the DB row and NEVER from the request, so this
    route cannot be turned into an open redirect. A recording failure must not
    block the redirect."""
    dest = "/"
    try:
        from dashboard import fullscript as _fs
        with _db_lock, db.connect(LOG_DB) as cx:
            cx.row_factory = sqlite3.Row
            _fs.init_tables(cx)
            portal = _portal_record_for(cx, token)
            if portal:
                email = (portal.get("email") or "").strip().lower()
                row = _fs.product_by_slug(cx, product_slug)
                if email and row:
                    dest = _fullscript_dispensary_url()
                    try:
                        _fs.record_click(cx, email, row["name"], "")
                    except Exception:
                        pass
    except Exception:
        dest = "/"
    return redirect(dest, code=302)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/deploy-chat && python -m pytest tests/test_fullscript_redirect.py -v`
Expected: PASS, 6 tests

- [ ] **Step 5: Mutation-test the safety guard**

Temporarily change the route's `dest = _fullscript_dispensary_url()` to
`dest = product_slug` and re-run. `test_destination_never_comes_from_the_request`
MUST go red. Then revert. A guard that never fails is not a guard.

Run: `cd ~/deploy-chat && python -m pytest tests/test_fullscript_redirect.py::test_destination_never_comes_from_the_request -v`
Expected: FAIL while mutated, PASS after revert.

- [ ] **Step 6: Commit**

```bash
git add app.py dashboard/fullscript.py tests/test_fullscript_redirect.py
git commit -m "feat(fullscript): tracked outbound redirect with token-only identity"
```

---

### Task 7: Portal card

**Files:**
- Modify: `static/client-portal.html` — add beside the PRL card block (`static/client-portal.html:2680`) and the PRL body helper (`:712`)

**Interfaces:**
- Consumes: `d.fullscript_enabled` and `d.fullscript` from Task 5's payload

**Critical pin:** `render(d, v)`'s signature is pinned by `tests/test_portal_reorder_ui.py`. Do NOT add a parameter to `render()`. The Fullscript payload arrives on `d`, exactly as `d.prl_supplement` does, so no signature change is needed.

- [ ] **Step 1: Add the card body helper**

Add near `prlSupplementBodyHtml` (around `static/client-portal.html:712`):

```javascript
// Fullscript dispensary card body: third-party products Glen recommends but does
// not formulate, grouped by why they surfaced. Present only when
// FULLSCRIPT_ENABLED -- see d.fullscript_enabled at the card block below.
// Every buy link goes through /fs/<token>/<slug> so the click is recorded and
// the destination is chosen server-side.
function fullscriptBodyHtml(fsData){
  const groups = (fsData && fsData.groups) || [];
  let blocks = "";
  groups.forEach(function(g){
    const items = (g.products || []).map(function(p){
      const ff = p.ff
        ? `<span class="small muted"> · you also have <strong>${esc(ff_name(p))}</strong></span>`
        : "";
      const brand = p.brand ? `<span class="small muted"> — ${esc(p.brand)}</span>` : "";
      const why = p.reason ? `<div class="small muted">${esc(p.reason)}</div>` : "";
      return `<li><a href="/fs/${esc(fsData.token)}/${esc(p.product_slug)}"
                 target="_blank" rel="noopener">${esc(p.name)}</a>${brand}${ff}${why}</li>`;
    }).join("");
    if(items) blocks += `<h4 class="small">${esc(g.heading)}</h4><ul>${items}</ul>`;
  });
  if(!blocks) blocks = `<p class="muted">No Fullscript matches yet.</p>`;
  const btnHref = safeUrl(fsData.dispensary_url) || "https://us.fullscript.com/";
  const micro = `<p class="small muted" style="margin-top:.4rem">Products I recommend but don't make. Ordering through my dispensary keeps everything in one place.</p>`;
  return blocks + `<p><a class="btn" href="${btnHref}" target="_blank" rel="noopener">Open my Fullscript dispensary</a></p>` + micro;
}

function ff_name(p){ return (p.ff && p.ff.name) || ""; }
```

- [ ] **Step 2: Add the card block**

Add immediately after the PRL card block (around `static/client-portal.html:2685`):

```javascript
  // Fullscript: separately-listed third-party channel, present only when
  // FULLSCRIPT_ENABLED and the client has at least one candidate.
  if (d.fullscript_enabled && d.fullscript) {
    html += `<div class="card fullscript-card">
      <h3>Fullscript</h3>
      ${fullscriptBodyHtml(Object.assign({token: d.token || ""}, d.fullscript))}
    </div>`;
  }
```

- [ ] **Step 3: Verify the payload carries the token**

The card builds `/fs/<token>/...` links, so the portal payload must expose the token
the client loaded with. Check whether `payload["token"]` is already set in the portal
view assembly near `app.py:20011`. If it is not, add it in Task 5's wiring block:

```python
    payload["token"] = token
```

Run: `cd ~/deploy-chat && grep -n 'payload\["token"\]' app.py`
Expected: at least one match. If none, add the line above and re-run.

- [ ] **Step 4: Render-verify with the synthetic-payload harness**

A true flag does not prove a rendered card. Drive the portal headless with a
synthetic payload containing a `fullscript` block and confirm the card, the group
headings, and a `/fs/` link all appear in the rendered DOM. Confirm on mobile width
via CSSOM rather than resizing the window.

Expected: card visible, heading "Matched from your scan" present, at least one
anchor whose href starts with `/fs/`.

- [ ] **Step 5: Commit**

```bash
git add static/client-portal.html app.py
git commit -m "feat(fullscript): client portal channel card"
```

---

### Task 8: Flag-off regression guard

**Files:**
- Modify: `tests/test_fullscript_builder.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_fullscript_builder.py`:

```python
def test_flag_off_payload_has_no_fullscript_key(monkeypatch, tmp_path):
    """The byte-identical guarantee: with the flag unset, the portal payload must
    not gain a `fullscript` key at all. A present-but-null key is a regression."""
    monkeypatch.setattr(app_mod, "LOG_DB", _db(tmp_path))
    monkeypatch.delenv("FULLSCRIPT_ENABLED", raising=False)
    payload = {}
    payload["fullscript_enabled"] = app_mod._fullscript_enabled()
    if app_mod._fullscript_enabled():
        blk = app_mod._fullscript_for("a@b.com", None)
        if blk:
            payload["fullscript"] = blk
    assert payload["fullscript_enabled"] is False
    assert "fullscript" not in payload
```

- [ ] **Step 2: Run it**

Run: `cd ~/deploy-chat && python -m pytest tests/test_fullscript_builder.py -v`
Expected: PASS, 5 tests

- [ ] **Step 3: Run the whole Fullscript suite together**

Run:
```bash
cd ~/deploy-chat && python -m pytest \
  tests/test_fullscript_module.py \
  tests/test_fullscript_resolver.py \
  tests/test_fullscript_builder.py \
  tests/test_fullscript_redirect.py -v
```
Expected: PASS, 23 tests. Do NOT run the bare full suite — it sends real email.

- [ ] **Step 4: Check for order-dependent contamination**

Run the four files in reverse order. Import-time state from one file can contaminate another.

Run:
```bash
cd ~/deploy-chat && python -m pytest \
  tests/test_fullscript_redirect.py \
  tests/test_fullscript_builder.py \
  tests/test_fullscript_resolver.py \
  tests/test_fullscript_module.py -v
```
Expected: PASS, same 23 tests.

- [ ] **Step 5: Commit**

```bash
git add tests/test_fullscript_builder.py
git commit -m "test(fullscript): flag-off payload byte-identical guard"
```

---

## Deployment

Ships dark. `FULLSCRIPT_ENABLED` stays unset until Glen reviews the seed's `best_ff` mappings and answers the attribution question with Fullscript.

Flipping the flag is a **second deploy**: flags are read at startup, so setting `FULLSCRIPT_ENABLED` in the Doppler `prd` config of the `remedy-match` project requires a Render restart to take effect. Merging the PR and flipping the flag are two separate deploys.

Verify live via `/api/portal/<token>/view` for a client with a scan, confirming the `fullscript` key is present and its groups populated.

## Out of scope for A1

- Condition driver and console pin UI (A2)
- Review driver and console attach-on-confirm (B)
- Folding Fullscript into the unified recommendations view (A3, deferred; requires teaching two display surfaces to resolve non-storefront keys)
- Deep-link mode, which stays written-but-gated until Fullscript confirms new-signup attribution
