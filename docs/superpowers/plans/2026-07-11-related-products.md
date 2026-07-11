# Related Products Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a bottom-of-page "related products" section to every `/begin/product/<slug>` page, combining Glen's manual "Dr. Glen recommends" picks with an auto list (harvested remedymatch merchandising + Pinecone semantic neighbors), curated from a console editor.

**Architecture:** Pure merge/guardrail logic lives in `dashboard/related_products.py` (unit-tested without importing `app.py`). A one-off harvest script scrapes remedymatch "Related Products:" lists into a versioned `data/related-harvested.json`. Manual picks live in a console-editable `related-manual.json` on the `/data` disk. `app.py`'s `begin_product_page_data` merges the three sources behind a flag and the frontend renders an inline-expandable section.

**Tech Stack:** Python 3 / Flask (app.py), SQLite (LOG_DB cache), Pinecone (semantic neighbors), vanilla JS in `static/begin-product.html`, pytest.

## Global Constraints

- Feature ships dark behind `RELATED_PRODUCTS_ENABLED` (Doppler); default off.
- `products.json` is a **read-only repo file**; manual state persists on the `/data` disk (pattern: `dashboard/products.py` `_fixed_path()`).
- `app.py` is NOT importable in the test env (needs `pinecone`); all unit-tested logic goes in importable `dashboard/` modules.
- Copy rules: no em dashes; use commas (matches existing generated copy).
- Auto list cap: **12** total. Guardrails apply to auto only; manual picks always show.
- Do-not-recommend slugs (verified against catalog 2026-07-11): `electrolyte-mineral-manna`, `water-ionizer-5plate`, `water-ionizer-9plate`, `water-ionizer-15plate`, `fungifuge`.

---

### Task 1: Pure guardrail + resolver module

**Files:**
- Create: `dashboard/related_products.py`
- Test: `tests/test_related_products.py`

**Interfaces:**
- Produces:
  - `DO_NOT_RECOMMEND: frozenset[str]`
  - `guardrail_ok(slug, base_slug, products) -> bool` — `products` is the `{slug: dict}` catalog map; returns False for self, `inactive`, do-not-recommend, and any slug whose product is missing.
  - `resolve_related(base_slug, *, manual, harvested, semantic, products, cap=12) -> dict` returning `{"featured": [slug,...], "more": [slug,...]}`. `manual`, `harvested`, `semantic` are ordered slug lists. Order: `featured` = manual (guardrails bypassed, deduped, dropped only if missing/self) + top 1 auto; `more` = remaining auto; auto = (harvested then semantic), guardrail-filtered, deduped against manual and each other, total auto capped at `cap`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_related_products.py
from dashboard import related_products as rp

_PRODUCTS = {
    "iop-syntropy": {"name": "IOP Syntropy"},
    "immune-modulation": {"name": "Immune Modulation"},
    "wholomega": {"name": "WholOmega"},
    "neuroprotect": {"name": "Neuroprotect"},
    "book-healing-glaucoma": {"name": "Healing Glaucoma (book)"},
    "denas-scenar": {"name": "DENAS PCM Pro"},
    "old-thing": {"name": "Old", "inactive": True},
    "water-ionizer-9plate": {"name": "9-Plate Water Ionizer (Living Water)"},
}

def test_manual_first_then_one_auto_in_featured():
    out = rp.resolve_related(
        "iop-syntropy",
        manual=["immune-modulation"],
        harvested=["wholomega", "neuroprotect"],
        semantic=["book-healing-glaucoma"],
        products=_PRODUCTS)
    assert out["featured"] == ["immune-modulation", "wholomega"]
    assert out["more"] == ["neuroprotect", "book-healing-glaucoma"]

def test_auto_drops_self_inactive_and_do_not_recommend():
    out = rp.resolve_related(
        "iop-syntropy",
        manual=[],
        harvested=["iop-syntropy", "old-thing", "water-ionizer-9plate", "wholomega"],
        semantic=[],
        products=_PRODUCTS)
    assert out["featured"] == ["wholomega"]
    assert out["more"] == []

def test_manual_bypasses_guardrail_but_dedups_from_auto():
    out = rp.resolve_related(
        "iop-syntropy",
        manual=["water-ionizer-9plate", "wholomega"],
        harvested=["wholomega", "neuroprotect"],
        semantic=[],
        products=_PRODUCTS)
    # manual keeps the do-not-recommend pick; wholomega not repeated in auto
    assert out["featured"] == ["water-ionizer-9plate", "wholomega", "neuroprotect"]
    assert out["more"] == []

def test_auto_capped(monkeypatch):
    prods = {f"p{i}": {"name": str(i)} for i in range(20)}
    prods["base"] = {"name": "base"}
    out = rp.resolve_related("base", manual=[], harvested=[f"p{i}" for i in range(20)],
                             semantic=[], products=prods, cap=12)
    assert len(out["featured"]) + len(out["more"]) == 12

def test_empty_when_nothing_related():
    out = rp.resolve_related("iop-syntropy", manual=[], harvested=[], semantic=[], products=_PRODUCTS)
    assert out["featured"] == [] and out["more"] == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_related_products.py -q`
Expected: FAIL (`ModuleNotFoundError` / `AttributeError: resolve_related`).

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/related_products.py
"""Pure logic for the product-page related-products section. No I/O here so it is
unit-testable without importing app.py (which needs pinecone)."""

DO_NOT_RECOMMEND = frozenset({
    "electrolyte-mineral-manna",
    "water-ionizer-5plate", "water-ionizer-9plate", "water-ionizer-15plate",
    "fungifuge",
})


def guardrail_ok(slug, base_slug, products):
    """Auto-list gate: real, sellable, not the product itself, not blocked."""
    if not slug or slug == base_slug:
        return False
    p = products.get(slug)
    if p is None or p.get("inactive"):
        return False
    if slug in DO_NOT_RECOMMEND:
        return False
    return True


def resolve_related(base_slug, *, manual, harvested, semantic, products, cap=12):
    """Merge the three sources into {featured, more}. Manual picks bypass the
    guardrail (Glen's explicit choice) and lead; auto = harvested then semantic,
    guardrail-filtered, deduped, capped at `cap`."""
    seen = set()
    featured_manual = []
    for s in manual:
        if s and s != base_slug and s not in seen and s in products:
            seen.add(s)
            featured_manual.append(s)

    auto = []
    for s in list(harvested) + list(semantic):
        if s in seen or not guardrail_ok(s, base_slug, products):
            continue
        seen.add(s)
        auto.append(s)
        if len(auto) >= cap:
            break

    if not featured_manual and not auto:
        return {"featured": [], "more": []}
    featured = featured_manual + auto[:1]
    more = auto[1:]
    return {"featured": featured, "more": more}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_related_products.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add dashboard/related_products.py tests/test_related_products.py
git commit -m "related-products: pure merge + guardrail logic"
```

---

### Task 2: Storefront slug mapper

**Files:**
- Modify: `dashboard/related_products.py`
- Test: `tests/test_related_products.py`

**Interfaces:**
- Produces: `map_storefront_slug(url, catalog_slugs, aliases) -> str | None`. `catalog_slugs` is a set; `aliases` maps storefront-slug -> catalog-slug. Handles `/remedies/<cat>/<id>-<slug>` and `/resources/<id>-<slug>`; returns the catalog slug or None (logged by caller).

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_related_products.py
_SLUGS = {"immune-modulation", "book-healing-glaucoma", "denas-scenar", "water-ionizer-9plate"}
_ALIASES = {
    "healing-glaucoma-book": "book-healing-glaucoma",
    "denas-microcurrent-system-for-eye-healing": "denas-scenar",
    "living-water-ionizer-9-plate": "water-ionizer-9plate",
}

def test_map_exact_remedies_url():
    assert rp.map_storefront_slug(
        "https://remedymatch.com/remedies/syntropy/56-immune-modulation",
        _SLUGS, _ALIASES) == "immune-modulation"

def test_map_resources_via_alias():
    assert rp.map_storefront_slug(
        "https://remedymatch.com/resources/50-healing-glaucoma-book",
        _SLUGS, _ALIASES) == "book-healing-glaucoma"

def test_map_unknown_returns_none():
    assert rp.map_storefront_slug(
        "https://remedymatch.com/resources/999-mystery-widget",
        _SLUGS, _ALIASES) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_related_products.py -k map -q`
Expected: FAIL (`AttributeError: map_storefront_slug`).

- [ ] **Step 3: Write minimal implementation**

```python
# append to dashboard/related_products.py
import re as _re

_URL_TAIL = _re.compile(r"/(?:remedies/[^/]+|resources)/\d+-([a-z0-9-]+)/?$", _re.I)


def map_storefront_slug(url, catalog_slugs, aliases):
    """remedymatch storefront URL -> catalog slug, or None if unresolvable."""
    if not url:
        return None
    m = _URL_TAIL.search(url.strip())
    if not m:
        return None
    sf = m.group(1).lower()
    if sf in aliases:
        return aliases[sf]
    if sf in catalog_slugs:
        return sf
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_related_products.py -q`
Expected: PASS (8 passed).

- [ ] **Step 5: Commit**

```bash
git add dashboard/related_products.py tests/test_related_products.py
git commit -m "related-products: storefront->catalog slug mapper"
```

---

### Task 3: Manual store (data-disk read/write)

**Files:**
- Create: `dashboard/related_store.py`
- Test: `tests/test_related_store.py`

**Interfaces:**
- Produces:
  - `manual_path() -> str` — `$DATA_DIR/related-manual.json` (falls back to repo `data/`).
  - `load_manual(slug=None) -> dict | list` — full map, or one product's list.
  - `save_manual(slug, related_slugs) -> None` — atomic write of one product's list.
  - `load_harvested(slug=None)` — reads repo `data/related-harvested.json`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_related_store.py
import json, os
from dashboard import related_store as rs

def test_save_then_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    rs.save_manual("iop-syntropy", ["immune-modulation", "wholomega"])
    assert rs.load_manual("iop-syntropy") == ["immune-modulation", "wholomega"]
    assert rs.load_manual()["iop-syntropy"] == ["immune-modulation", "wholomega"]

def test_load_missing_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    assert rs.load_manual("nope") == []
    assert rs.load_manual() == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_related_store.py -q`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/related_store.py
"""Disk persistence for related-products state. Manual picks are writable and live
on the /data disk (products.json is a read-only repo file). Harvested data is a
read-only versioned repo file."""
import json
import os

_REPO_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def manual_path():
    d = os.environ.get("DATA_DIR") or _REPO_DATA
    return os.path.join(d, "related-manual.json")


def _read(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, ValueError):
        return {}


def load_manual(slug=None):
    data = _read(manual_path())
    if slug is None:
        return data
    return list(data.get(slug, []))


def save_manual(slug, related_slugs):
    path = manual_path()
    data = _read(path)
    data[slug] = list(related_slugs)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def load_harvested(slug=None):
    data = _read(os.path.join(_REPO_DATA, "related-harvested.json"))
    if slug is None:
        return data
    return list(data.get(slug, []))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_related_store.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add dashboard/related_store.py tests/test_related_store.py
git commit -m "related-products: manual/harvested disk store"
```

---

### Task 4: Harvest parser + script

**Files:**
- Create: `scripts/harvest_related_products.py`
- Create: `tests/fixtures/remedymatch-related.html`
- Test: `tests/test_harvest_related_products.py`

**Interfaces:**
- Produces: `parse_related(html) -> list[str]` (returns the related-product URLs found in a "Related Products:" block). The script `main()` iterates catalog products with a `url`, fetches each, calls `parse_related` + `map_storefront_slug`, writes `data/related-harvested.json` and prints an unmapped report.

- [ ] **Step 1: Create the fixture**

Save a minimal real sample (trimmed) to `tests/fixtures/remedymatch-related.html`:

```html
<div class="product-related"><h3>Related Products:</h3><ul>
<li><a href="https://remedymatch.com/remedies/syntropy/56-immune-modulation">Immune Modulation</a></li>
<li><a href="https://remedymatch.com/resources/50-healing-glaucoma-book">Healing Glaucoma Book</a></li>
</ul></div>
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_harvest_related_products.py
import os
from scripts.harvest_related_products import parse_related

def test_parse_related_extracts_urls():
    html = open(os.path.join(os.path.dirname(__file__), "fixtures", "remedymatch-related.html")).read()
    urls = parse_related(html)
    assert "https://remedymatch.com/remedies/syntropy/56-immune-modulation" in urls
    assert "https://remedymatch.com/resources/50-healing-glaucoma-book" in urls
    assert len(urls) == 2
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python3 -m pytest tests/test_harvest_related_products.py -q`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 4: Write the parser + script**

```python
# scripts/harvest_related_products.py
"""Scrape remedymatch "Related Products:" lists into data/related-harvested.json.
Run ad hoc:  python3 scripts/harvest_related_products.py
Politeness: 1 request/sec. Not on any request path."""
import json
import os
import re
import sys
import time
import urllib.request

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

# Grab the "Related Products:" block, then every href inside it.
_BLOCK = re.compile(r"Related Products:.*?(?=</section>|</div>\s*</div>|$)", re.I | re.S)
_HREF = re.compile(r'href="(https?://[^"]*remedymatch\.com/(?:remedies/[^"]+|resources/[^"]+))"', re.I)


def parse_related(html):
    block = _BLOCK.search(html or "")
    if not block:
        return []
    seen, out = set(), []
    for m in _HREF.finditer(block.group(0)):
        u = m.group(1)
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


# Storefront-slug -> catalog-slug overrides for hand-added device/book SKUs whose
# catalog slug differs from the storefront slug. Extend as the unmapped report shows.
ALIASES = {
    "healing-glaucoma-book": "book-healing-glaucoma",
    "denas-microcurrent-system-for-eye-healing": "denas-scenar",
    "living-water-ionizer-9-plate": "water-ionizer-9plate",
    "kloud-mini-pemf-mat": "kloud-pemf-mini",
}


def _fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "healing-oasis-harvest/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", "replace")


def main():
    from dashboard.related_products import map_storefront_slug
    products = json.load(open(os.path.join(_REPO, "data", "products.json")))["products"]
    catalog_slugs = set(products)
    out, unmapped = {}, []
    targets = [(s, v["url"]) for s, v in products.items()
               if v.get("url") and "remedymatch.com" in v["url"]]
    for i, (slug, url) in enumerate(targets):
        try:
            urls = parse_related(_fetch(url))
        except Exception as e:  # noqa: BLE001
            print(f"[harvest] {slug}: fetch failed: {e}", file=sys.stderr)
            continue
        related = []
        for u in urls:
            mapped = map_storefront_slug(u, catalog_slugs, ALIASES)
            if mapped and mapped != slug:
                related.append(mapped)
            elif not mapped:
                unmapped.append(u)
        if related:
            out[slug] = related
        print(f"[harvest] {i+1}/{len(targets)} {slug}: {len(related)} related", file=sys.stderr)
        time.sleep(1)
    with open(os.path.join(_REPO, "data", "related-harvested.json"), "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(f"[harvest] wrote {len(out)} products; {len(set(unmapped))} unmapped urls", file=sys.stderr)
    for u in sorted(set(unmapped)):
        print(f"[unmapped] {u}", file=sys.stderr)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python3 -m pytest tests/test_harvest_related_products.py -q`
Expected: PASS (1 passed).

- [ ] **Step 6: Commit** (do NOT run the live scrape here; that is a manual step in Task 9)

```bash
git add scripts/harvest_related_products.py tests/test_harvest_related_products.py tests/fixtures/remedymatch-related.html
git commit -m "related-products: harvest parser + script"
```

---

### Task 5: Semantic-neighbor helper + cache

**Files:**
- Modify: `app.py` (near the Pinecone client + `begin_product_page_data`)
- Test: manual (needs Pinecone); covered by render-verify in Task 9.

**Interfaces:**
- Produces: `_related_semantic(slug, k=12) -> list[str]` — returns up to `k` catalog slugs nearest to `slug`'s vector, excluding self. Cached in LOG_DB table `related_semantic(slug TEXT PRIMARY KEY, slugs_json TEXT, generated_at TEXT)`.

- [ ] **Step 1: Add the cache table + helper**

Find the Pinecone index handle used by the remedy matcher (grep `index.query(` / `PINECONE`). Add near `begin_product_page_data`:

```python
def _related_semantic(slug, k=12):
    """Up to k catalog slugs nearest to `slug`'s Pinecone vector (cached)."""
    import sqlite3 as _sq, json as _json
    try:
        with _sq.connect(LOG_DB) as cx:
            cx.execute("CREATE TABLE IF NOT EXISTS related_semantic ("
                       "slug TEXT PRIMARY KEY, slugs_json TEXT, generated_at TEXT)")
            row = cx.execute("SELECT slugs_json FROM related_semantic WHERE slug=?", (slug,)).fetchone()
            if row:
                return _json.loads(row[0])
    except Exception as e:
        print(f"[related-sem] cache read failed: {e}", flush=True)
    try:
        # Fetch this product's vector by id, then query nearest neighbours.
        fetched = _PINECONE_INDEX.fetch(ids=[slug])
        vec = fetched.vectors.get(slug)
        if not vec:
            return []
        res = _PINECONE_INDEX.query(vector=vec.values, top_k=k + 1, include_metadata=False)
        slugs = [m.id for m in res.matches if m.id != slug][:k]
    except Exception as e:
        print(f"[related-sem] query failed: {e}", flush=True)
        return []
    try:
        with _sq.connect(LOG_DB) as cx:
            cx.execute("INSERT OR REPLACE INTO related_semantic(slug,slugs_json,generated_at) "
                       "VALUES (?,?,datetime('now'))", (slug, _json.dumps(slugs)))
    except Exception as e:
        print(f"[related-sem] cache write failed: {e}", flush=True)
    return slugs
```

Note: confirm the actual Pinecone index variable name and that vectors are keyed by catalog slug (grep how the matcher upserts ids). If ids are not slugs, map via `pinecone_title` -> slug before returning. Adjust the `fetch`/`query` calls to the installed pinecone client version.

- [ ] **Step 2: Commit**

```bash
git add app.py
git commit -m "related-products: semantic neighbour helper + cache"
```

---

### Task 6: Wire the related section into page data (behind flag)

**Files:**
- Modify: `app.py` — flag read near other `*_ENABLED` flags; section build in `begin_product_page_data`.

**Interfaces:**
- Consumes: `related_products.resolve_related`, `related_store.load_manual/load_harvested`, `_related_semantic`.
- Produces: a `related` section in the page-data `sections` list + a `related_cards` map in the response.

- [ ] **Step 1: Add the flag**

Near the other flags (grep `_SALES_AI_COPY_ENABLED =`):

```python
_RELATED_PRODUCTS_ENABLED = os.environ.get("RELATED_PRODUCTS_ENABLED", "").lower() in ("1", "true", "yes")
```

- [ ] **Step 2: Build the section**

In `begin_product_page_data`, after the `filter_sections(...)` line from #804 and before `if _REVIEWS_ENABLED:`:

```python
    if _RELATED_PRODUCTS_ENABLED:
        try:
            from dashboard import related_products as _rp, related_store as _rstore
            _prods = {s: v for s, v in _cached_products_map().items()}
            _res = _rp.resolve_related(
                slug,
                manual=_rstore.load_manual(slug),
                harvested=_rstore.load_harvested(slug),
                semantic=_related_semantic(slug),
                products=_prods)
            if _res["featured"]:
                def _card(rs):
                    rp_ = _prods.get(rs, {})
                    imgs = rp_.get("page_images") or []
                    return {"slug": rs, "name": rp_.get("name", rs),
                            "price": f"${rp_.get('price_cents',0)/100:.2f}",
                            "url": f"/begin/product/{rs}",
                            "image": (imgs[0] if imgs else "")}
                sections.append({
                    "id": "related", "title": "Dr. Glen recommends", "default_open": True,
                    "body": {"featured": [_card(s) for s in _res["featured"]],
                             "more": [_card(s) for s in _res["more"]]}})
        except Exception as _e:
            print(f"[related] page-data skipped: {_e}", flush=True)
```

Note: `_cached_products_map()` = the existing catalog map loader (grep `_get_product`'s source; reuse `dashboard.products.catalog(...)` or the in-memory products dict). Use whatever `begin_product_page_data` already has in scope for `p`'s siblings; if none, `json`-load once via `dashboard.products`.

- [ ] **Step 3: Verify app.py compiles**

Run: `python3 -m py_compile app.py`
Expected: no output (success).

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "related-products: append related section to page data (flagged)"
```

---

### Task 7: Frontend render + inline expand

**Files:**
- Modify: `static/begin-product.html`

- [ ] **Step 1: Add the renderer**

Add a `renderRelatedBody(body)` mirroring `renderIngredientsBody`, and route `case 'related':` in `buildSectionBody`. Cards show image (if present), name, price, linking to `card.url`. The "See more like this" button reveals `body.more`:

```javascript
function renderRelatedBody(body){
  var wrap = document.createElement('div');
  function card(c){
    var a = document.createElement('a');
    a.className = 'sp-related-card'; a.href = c.url;
    a.innerHTML = (c.image ? '<img src="'+c.image+'" alt="">' : '') +
      '<span class="sp-related-name">'+c.name+'</span>' +
      '<span class="sp-related-price">'+c.price+'</span>';
    return a;
  }
  (body.featured||[]).forEach(function(c){ wrap.appendChild(card(c)); });
  var more = body.more || [];
  if (more.length){
    var moreWrap = document.createElement('div');
    moreWrap.className = 'sp-related-more'; moreWrap.style.display = 'none';
    more.forEach(function(c){ moreWrap.appendChild(card(c)); });
    var btn = document.createElement('button');
    btn.className = 'sp-related-morebtn'; btn.textContent = 'See more like this';
    btn.onclick = function(){ moreWrap.style.display = 'flex'; btn.style.display = 'none'; };
    wrap.appendChild(moreWrap); wrap.appendChild(btn);
  }
  return wrap;
}
```

Add `case 'related': return renderRelatedBody(sec.body);` to `buildSectionBody`, a teaser (`case 'related': teaser.textContent = 'Explore what pairs well'; break;`), and CSS for `.sp-related-card` / `.sp-related-more` (flex, wrap) matching the page's card style.

- [ ] **Step 2: Commit**

```bash
git add static/begin-product.html
git commit -m "related-products: frontend related section + inline expand"
```

---

### Task 8: Console editor (action + route + page)

**Files:**
- Create: `dashboard/related_products_actions.py`
- Modify: `app.py` (register action; `/console/related-products` route)
- Create: `static/console-related-products.html`
- Test: `tests/test_related_products_actions.py`

**Interfaces:**
- Produces: console action `related_products.set` (LOW_WRITE, OWNER/OPS) with params `{slug, related:[...]}` -> writes `related-manual.json` via `related_store.save_manual`.

- [ ] **Step 1: Write the failing action test**

```python
# tests/test_related_products_actions.py
from dashboard import related_products_actions as rpa, related_store as rs

def test_set_action_saves_manual(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    res = rpa._exec_set({"slug": "iop-syntropy", "related": ["immune-modulation"]}, {"cx": None})
    assert res["slug"] == "iop-syntropy"
    assert rs.load_manual("iop-syntropy") == ["immune-modulation"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_related_products_actions.py -q`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement the action** (mirror `dashboard/sales_pages_actions.py`)

```python
# dashboard/related_products_actions.py
"""Console action: save Glen's manual "Dr. Glen recommends" picks for a product."""
from dashboard.actions import register_action, Action, LOW_WRITE, get_action
from dashboard.rbac import OWNER, OPS
from dashboard import related_store as _rs


def _exec_set(params, ctx):
    slug = (params.get("slug") or "").strip()
    if not slug:
        raise ValueError("slug required")
    related = [s for s in (params.get("related") or []) if isinstance(s, str) and s.strip()]
    _rs.save_manual(slug, related)
    return {"slug": slug, "related": related, "saved": True}


def register():
    if get_action("related_products.set"):
        return
    register_action(Action(
        key="related_products.set", module="related_products",
        title="Set related products",
        description="Save Dr. Glen's manual related-product picks for a product.",
        risk_tier=LOW_WRITE, permission=(OWNER, OPS), executor=_exec_set))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_related_products_actions.py -q`
Expected: PASS.

- [ ] **Step 5: Register + route in app.py**

Grep how `sales_pages_actions.register()` is called at startup and how `/console/support-programs` serves its page; add the analogous `related_products_actions.register()` call and:

```python
@app.route("/console/related-products")
def console_related_products():
    return app.send_static_file("console-related-products.html")

@app.route("/api/console/related-products/<slug>", methods=["GET"])
@require_console_key
def api_related_products_get(slug):
    from dashboard import related_store as _rs
    return ok({"manual": _rs.load_manual(slug), "harvested": _rs.load_harvested(slug)})
```

- [ ] **Step 6: Build the console page** `static/console-related-products.html`: a product search box, the harvested list as checkboxes, current manual picks (removable), a free catalog-search adder, and a Save button POSTing `related_products.set` to `/api/action/related_products.set` with `X-Console-Key`. Mirror `static/console-support-programs.html` structure and styling.

- [ ] **Step 7: Verify + commit**

Run: `python3 -m py_compile app.py`
```bash
git add dashboard/related_products_actions.py tests/test_related_products_actions.py app.py static/console-related-products.html
git commit -m "related-products: console editor action + route + page"
```

---

### Task 9: Harvest, verify, flip flag (manual / ops)

**Files:** none (operational).

- [ ] **Step 1: Run the harvest**

Run: `python3 scripts/harvest_related_products.py 2> harvest-report.txt`
Review `harvest-report.txt` for `[unmapped]` lines; add any device/book misses to `ALIASES` (Task 4) and re-run. Commit `data/related-harvested.json`.

```bash
git add data/related-harvested.json
git commit -m "related-products: harvested remedymatch related lists"
```

- [ ] **Step 2: Full test sweep**

Run: `python3 -m pytest tests/test_related_products.py tests/test_related_store.py tests/test_harvest_related_products.py tests/test_related_products_actions.py -q`
Expected: all PASS.

- [ ] **Step 3: Open PR, merge (Glen), deploy.**

- [ ] **Step 4: Set `RELATED_PRODUCTS_ENABLED=true` in Doppler** (project `remedy-match`, config `prd`).

- [ ] **Step 5: Render-verify** in headless Chrome: a formula (`/begin/product/iop-syntropy` — expect manual/harvested picks + "See more" inline expand) and a device (`/begin/product/therapeutic-nightlight` — expect semantic neighbours, no do-not-recommend items). Confirm no self/superseded/inactive/do-not-recommend leak into auto.

---

## Self-Review

**Spec coverage:** harvest (Task 4/9), harvested+manual+semantic model (Tasks 1,3,5), guardrails incl. do-not-recommend (Task 1), slug mapping + aliases (Tasks 2,4), console editor (Task 8), page render bottom + inline see-more (Tasks 6,7), flag + render-verify (Task 9), module boundaries/testability (Tasks 1-3,8). Wishlist correctly excluded.

**Placeholder scan:** integration Tasks 5/6/8 carry "grep the existing pattern" notes (Pinecone index name, catalog-map loader, startup registration) because those exact identifiers must be read from `app.py` at implementation time; each names the concrete symbol to grep and the shape to match, with full code for everything determinable now. No TBD/TODO left in logic tasks.

**Type consistency:** `resolve_related` returns `{featured, more}` (slug lists) in Task 1; Task 6 maps those slugs to card dicts and wraps as `{featured:[card], more:[card]}` for the frontend, which Task 7 consumes. `map_storefront_slug(url, catalog_slugs, aliases)` signature identical in Tasks 2 and 4. `save_manual/load_manual/load_harvested` identical across Tasks 3, 6, 8.
