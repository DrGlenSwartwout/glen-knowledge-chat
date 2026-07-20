# Cello-Pack Packing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Price the "Cellophane refill pack" (cello) format as a tighter-packing unit so shipping is correct, and carry a separate bottles-vs-cello-packs count to the fulfillment surfaces — without changing reorder demand.

**Architecture:** A cello pack reuses the existing geometric packer by resolving cello-format lines to a distinct, smaller `bottle_type` (so `box_counts` reflects the tighter unit and the quote drops). A pure helper `packing_bottle_type(product, fmt)` centralizes the format→type decision. The per-line `format` is persisted on each stored line, and a pure `pack_breakdown(items)` derives `{bottle_units, cello_pack_units}` for display. `physical_units` is left unchanged (reorder demand is packaging-agnostic).

**Tech Stack:** Python (Flask monolith `app.py` + `dashboard/*.py`), SQLite (`chat_log.db`, self-healing `init_*` migrations), vanilla JS in `static/*.html` (no build step), pytest.

## Global Constraints

- **No new dependencies.** Reuse the existing packer (`dashboard/packing.py`) and rate table.
- **SQLite migrations are self-healing** (`CREATE TABLE IF NOT EXISTS` / idempotent `ALTER`), invoked lazily — no SQL migration files (those are Postgres-only, out of scope).
- **App-importing tests run under a fake-cred env** (they otherwise skip): prefix with
  `env -u DOPPLER_TOKEN PINECONE_API_KEY=pcsk_fake OPENAI_API_KEY=sk-fake ANTHROPIC_API_KEY=sk-ant-fake SECRET_KEY=ci CONSOLE_SECRET=ci-fake-console-secret`. Pure `dashboard/` tests run under bare `python3 -m pytest`.
- **Cello format id is `refill`** (existing `_FORMATS`, `app.py:6125`); the cello bottle_type is `"cello-refill"`.
- **Do NOT change `physical_units` semantics** — it feeds reorder demand.
- **Prod's `bottle_types` table is hand-built and authoritative at runtime**; the code seed only affects fresh/test catalogs. Rae must add the `cello-refill` row (dims or capacity) via `/admin/shipping` for prod to rate it — code ships a safe seed so it's testable and degrades to today's behavior if absent.

---

### Task 1: Cello bottle_type + `packing_bottle_type` helper

**Files:**
- Modify: `dashboard/shipping.py` (`PROD_BOTTLE_NAMES` ~:80, `_STANDARD_BOTTLES` ~:118, near `resolve_bottle_type` ~:750)
- Test: `tests/test_cello_packing_type.py`

**Interfaces:**
- Produces: `CELLO_BOTTLE_TYPE = "cello-refill"`; `CELLO_FORMATS = frozenset({"refill"})`; `packing_bottle_type(product: dict, fmt: str|None) -> str` — returns `CELLO_BOTTLE_TYPE` when `fmt` is a cello format, else `resolve_bottle_type(product.get("slug"), product)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cello_packing_type.py
import sqlite3, sys
from pathlib import Path
repo = Path(__file__).resolve().parent.parent
if str(repo) not in sys.path: sys.path.insert(0, str(repo))
from dashboard import shipping as S

def test_packing_bottle_type_maps_cello_and_passes_through():
    prod = {"slug": "mag", "bottle_type": "default"}
    assert S.packing_bottle_type(prod, "refill") == S.CELLO_BOTTLE_TYPE
    assert S.packing_bottle_type(prod, "bottle") == "default"
    assert S.packing_bottle_type(prod, None) == "default"

def test_cello_type_is_registered_and_packs_smaller_than_default():
    cx = sqlite3.connect(":memory:")
    S.ensure_schema(cx)  # seeds bottle_types incl. cello-refill on a fresh db
    dims = {r["name"]: (r["dia_mm"], r["height_mm"]) for r in S.list_bottle_types(cx)}
    assert S.CELLO_BOTTLE_TYPE in dims
    # cello pack occupies less volume than a default bottle -> more per box
    dia_c, h_c = dims[S.CELLO_BOTTLE_TYPE]; dia_d, h_d = dims["default"]
    assert dia_c*dia_c*h_c < dia_d*dia_d*h_d
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_cello_packing_type.py -v`
Expected: FAIL — `AttributeError: module 'dashboard.shipping' has no attribute 'CELLO_BOTTLE_TYPE'` (and `packing_bottle_type`).

*(Note: confirm the real accessor names for schema-seed + listing bottle types while implementing — the map cited `_STANDARD_BOTTLES` seeding and an admin CRUD; use the existing seed/list functions, adjusting `ensure_schema`/`list_bottle_types` in the test to their true names.)*

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/shipping.py — near resolve_bottle_type
CELLO_BOTTLE_TYPE = "cello-refill"
CELLO_FORMATS = frozenset({"refill"})

def packing_bottle_type(product, fmt):
    """Packing key for a cart line. Cello/refill lines resolve to the tighter
    cello unit; everything else keeps its product bottle_type."""
    if (fmt or "").strip().lower() in CELLO_FORMATS:
        return CELLO_BOTTLE_TYPE
    return resolve_bottle_type((product or {}).get("slug"), product)
```

Add `CELLO_BOTTLE_TYPE` to `PROD_BOTTLE_NAMES`, and a seed row to `_STANDARD_BOTTLES` with tight dims (a 30-cap cello pouch approximated as a small prism), e.g. `("cello-refill", 35, 60)` (dia_mm, height_mm — refine via `/admin/shipping`).

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_cello_packing_type.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dashboard/shipping.py tests/test_cello_packing_type.py
git commit -m "feat(shipping): cello-refill bottle_type + packing_bottle_type helper"
```

---

### Task 2: Route cello lines through the tighter unit in `_price_cart`

**Files:**
- Modify: `app.py` — `_price_cart` per-line loop (~6519-6575), the plain branch (~6572) and the bundle branch (~6556)
- Test: `tests/test_price_cart_cello_shipping.py`

**Interfaces:**
- Consumes: `shipping.packing_bottle_type` (Task 1).
- Produces: `_price_cart(...)` return dict gains `"cello_pack_units": int` and `"bottle_units": int` alongside the existing `shipping_cents`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_price_cart_cello_shipping.py  (app-importing -> fake-env)
import importlib, sys
from pathlib import Path
import pytest
repo = Path(__file__).resolve().parent.parent
if str(repo) not in sys.path: sys.path.insert(0, str(repo))

def _app(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    try:
        import app as a; importlib.reload(a)
    except Exception as e:
        pytest.skip(f"app not importable: {e}")
    # a small deterministic catalog: one shippable 'default' product
    monkeypatch.setattr(a, "_get_product", lambda slug: {"slug": "mag", "name": "Mag",
        "price_cents": 6997, "bottle_type": "default"} if slug == "mag" else None)
    return a

def test_cello_lines_ship_cheaper_and_count_separately(tmp_path, monkeypatch):
    a = _app(monkeypatch, tmp_path)
    ship = {"country": "US", "zip": "01950", "state": "MA", "city": "X", "street": "1 A St"}
    bottles = a._price_cart([{"slug": "mag", "qty": 6}], ship=ship)
    cello   = a._price_cart([{"slug": "mag", "qty": 6, "format": "refill"}], ship=ship)
    assert cello["shipping_cents"] <= bottles["shipping_cents"]
    assert cello["cello_pack_units"] == 6 and cello["bottle_units"] == 0
    assert bottles["cello_pack_units"] == 0 and bottles["bottle_units"] == 6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env -u DOPPLER_TOKEN PINECONE_API_KEY=pcsk_fake OPENAI_API_KEY=sk-fake ANTHROPIC_API_KEY=sk-ant-fake SECRET_KEY=ci CONSOLE_SECRET=ci-fake-console-secret python3 -m pytest tests/test_price_cart_cello_shipping.py -v`
Expected: FAIL — `KeyError: 'cello_pack_units'`.

- [ ] **Step 3: Write minimal implementation**

In `_price_cart`, initialize `total_cello = 0` next to `total_bottles`. In the plain branch (currently `bt = resolve_bottle_type(slug, p); box_counts[bt] += qty; total_bottles += qty`), replace with:

```python
_fmt = (c.get("format") or "").strip().lower()
bt = _shipping.packing_bottle_type(p, _fmt)
box_counts[bt] += qty
if bt == _shipping.CELLO_BOTTLE_TYPE:
    total_cello += qty
else:
    total_bottles += qty
```

Apply the same `packing_bottle_type(component_product, _fmt)` mapping in the bundle branch's per-component loop (the bundle line's own `format` governs its components). Add to the return dict: `"bottle_units": total_bottles, "cello_pack_units": total_cello`.

- [ ] **Step 4: Run test to verify it passes**

Run: `env -u DOPPLER_TOKEN ... python3 -m pytest tests/test_price_cart_cello_shipping.py -v` (same env prefix)
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_price_cart_cello_shipping.py
git commit -m "feat(shipping): cello-format lines pack tighter + counted separately in _price_cart"
```

---

### Task 3: Persist `format` on stored line records

**Files:**
- Modify: `app.py` — `_price_inhouse_invoice` rec builder (~`app.py:39870`, next to the `note` carry)
- Test: `tests/test_line_format_roundtrip.py`

**Interfaces:**
- Produces: each stored line dict carries `"format"` when non-default, so `pack_breakdown` (Task 4) and re-pricing can read it.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_line_format_roundtrip.py  (app-importing -> fake-env)
import importlib, sys
from pathlib import Path
import pytest
repo = Path(__file__).resolve().parent.parent
if str(repo) not in sys.path: sys.path.insert(0, str(repo))

def test_format_rides_on_the_stored_line(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    try:
        import app as a; importlib.reload(a)
    except Exception as e:
        pytest.skip(f"app not importable: {e}")
    monkeypatch.setattr(a, "_get_product", lambda slug: {"slug": "mag", "name": "Mag",
        "price_cents": 6997, "bottle_type": "default"} if slug == "mag" else None)
    priced = a._price_inhouse_invoice([{"slug": "mag", "qty": 2, "format": "refill"}],
                                      email="", pickup=True, ship=None)
    rec = priced["items_rec"][0]
    assert rec["format"] == "refill"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env -u DOPPLER_TOKEN ... python3 -m pytest tests/test_line_format_roundtrip.py -v`
Expected: FAIL — `KeyError: 'format'`.

- [ ] **Step 3: Write minimal implementation**

In `_price_inhouse_invoice`, right after the `note` carry, add:

```python
_fmt = (ln.get("format") or "").strip().lower()
if _fmt and _fmt != "bottle":
    rec["format"] = _fmt
```

- [ ] **Step 4: Run test to verify it passes**

Run: `env -u DOPPLER_TOKEN ... python3 -m pytest tests/test_line_format_roundtrip.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_line_format_roundtrip.py
git commit -m "feat(orders): persist per-line format on stored invoice lines"
```

---

### Task 4: `pack_breakdown` helper + expose in price-preview

**Files:**
- Modify: `dashboard/orders.py` (next to `physical_units` ~:593)
- Modify: `app.py` — `api_orders_price_preview` return (~40396-40468)
- Test: `tests/test_pack_breakdown.py`

**Interfaces:**
- Consumes: line dicts with optional `"format"` (Task 3).
- Produces: `orders.pack_breakdown(items, catalog) -> {"bottle_units": int, "cello_pack_units": int}` — same shippable/bundle/membership rules as `physical_units`, split by cello format. `physical_units` stays the COMBINED total (bottle+cello) — unchanged. price-preview response gains `"pack_breakdown"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pack_breakdown.py  (pure)
import sys
from pathlib import Path
repo = Path(__file__).resolve().parent.parent
if str(repo) not in sys.path: sys.path.insert(0, str(repo))
from dashboard.orders import pack_breakdown, physical_units

CAT = {"a": {"slug": "a", "name": "A"}, "svc": {"slug": "svc", "service": True}}

def test_split_by_format_and_total_unchanged():
    items = [{"slug": "a", "qty": 2}, {"slug": "a", "qty": 3, "format": "refill"},
             {"slug": "svc", "qty": 1}]
    assert pack_breakdown(items, CAT) == {"bottle_units": 2, "cello_pack_units": 3}
    # physical_units stays the COMBINED shippable total (reorder demand), unchanged
    assert physical_units(items, CAT) == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_pack_breakdown.py -v`
Expected: FAIL — `ImportError: cannot import name 'pack_breakdown'`.

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/orders.py — mirror physical_units' shippable/bundle loop, splitting by format
def pack_breakdown(items, catalog):
    """Shippable units split by packaging: {bottle_units, cello_pack_units}. Cello =
    lines whose format is a cello format. Bundles/services follow physical_units' rules.
    The SUM equals physical_units — this only partitions it for fulfillment display."""
    from dashboard import shipping as _sh
    catalog = catalog or {}
    bottles = cello = 0
    for it in (items or []):
        try: qty = int(it.get("qty") or 0)
        except (TypeError, ValueError): qty = 0
        if qty <= 0: continue
        if it.get("kind") == "membership" or str(it.get("slug") or "").startswith("membership:"):
            continue
        p = catalog.get(it.get("slug") or "") or {}
        if not _sh.is_shippable(p): continue
        n = qty  # bundles expand in physical_units; keep parity by reusing its per-line count
        if (it.get("format") or "").strip().lower() in _sh.CELLO_FORMATS:
            cello += n
        else:
            bottles += n
    return {"bottle_units": bottles, "cello_pack_units": cello}
```

*(While implementing: if a cello line can be a bundle, mirror `physical_units`' bundle expansion here so the split still sums to `physical_units`; add a bundle case to the test if so.)*

Add an app wrapper next to `_order_physical_units` and include it in the price-preview response: `"pack_breakdown": _order_pack_breakdown({"items": lines_in})`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_pack_breakdown.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dashboard/orders.py app.py tests/test_pack_breakdown.py
git commit -m "feat(orders): pack_breakdown split (bottles vs cello) + in price-preview"
```

---

### Task 5: Order-entry per-line Format picker + count display

**Files:**
- Modify: `static/order-new.html` — table header (~:94), `renderLines()` row, `linesPayload()`, `LINES` model, the units-to-ship display fed by `PREVIEW`
- Test: manual (frontend; no JS test harness). Verify via node `--check` + live render.

**Interfaces:**
- Consumes: `GET /api/orders/price-preview` now returns `pack_breakdown` (Task 4).
- Produces: each line payload includes `format`; the units line reads "N bottles + M cello packs".

- [ ] **Step 1: Add a Format column + model field**

Add `<th>Format</th>` to the header. In `renderLines()`, add a cell per row:

```html
<td><select onchange="editLine(${i},'format',this.value)">
  <option value="bottle"${(l.format||'bottle')==='bottle'?' selected':''}>Bottle</option>
  <option value="refill"${l.format==='refill'?' selected':''}>Cello refill</option>
</select></td>
```

Extend the `LINES.push({...})` in `addLine()` and the edit-load map with `format:(l.format||'bottle')`. In `editLine`, handle `f==='format'` (set `LINES[i].format=v; SHIP_CENTS=null;`).

- [ ] **Step 2: Send `format` in the payload**

In `linesPayload()`, add `if ((l.format||'bottle')!=='bottle') b.format=l.format;` to each line object.

- [ ] **Step 3: Show the split count**

Replace the units-to-ship update (`updateUnitsToShip`) to read `PREVIEW.pack_breakdown`:

```js
function updateUnitsToShip(pb){
  const el=$("units-to-ship"); if(!el) return;
  const b=(pb&&pb.bottle_units)||0, c=(pb&&pb.cello_pack_units)||0;
  el.textContent = "Total units to ship: " + b + " bottle" + (b===1?'':'s')
    + (c ? " + " + c + " cello pack" + (c===1?'':'s') : "");
}
```

Update the `refreshPreview` success handler to call `updateUnitsToShip(j.pack_breakdown)` (and `updateUnitsToShip({bottle_units:0,cello_pack_units:0})` on the empty branch).

- [ ] **Step 4: Syntax-check + verify**

Run node `--check` on the extracted inline script (same method used previously). Expected: OK.

- [ ] **Step 5: Commit**

```bash
git add static/order-new.html
git commit -m "feat(order-entry): per-line Format picker + bottles/cello split count"
```

---

### Task 6: Show the bottle/cello split on the customer + console surfaces

**Files:**
- Modify: `app.py` — invoice payload (`_invoice_summary` ~:41092 add `pack_breakdown`), order-board annotate (~42262)
- Modify: `static/console-orders.html` (~:244, :478 unit rendering), `static/invoice.html` (totals area) to render the split when `cello_pack_units > 0`
- Test: `tests/test_invoice_summary_pack_breakdown.py`

**Interfaces:**
- Consumes: `orders.pack_breakdown` (Task 4).
- Produces: invoice + order payloads carry `pack_breakdown`; UIs render "N bottles + M cello packs" where a unit count shows today.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_invoice_summary_pack_breakdown.py  (app-importing -> fake-env)
import importlib, sys
from pathlib import Path
import pytest
repo = Path(__file__).resolve().parent.parent
if str(repo) not in sys.path: sys.path.insert(0, str(repo))

def test_invoice_summary_carries_pack_breakdown(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    try:
        import app as a; importlib.reload(a)
    except Exception as e:
        pytest.skip(f"app not importable: {e}")
    monkeypatch.setattr(a, "_get_product", lambda slug: {"slug": "a", "name": "A"} )
    order = {"items": [{"slug": "a", "qty": 2}, {"slug": "a", "qty": 1, "format": "refill"}],
             "status": "proposed", "total_cents": 0}
    s = a._invoice_summary(order)
    assert s["pack_breakdown"] == {"bottle_units": 2, "cello_pack_units": 1}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env -u DOPPLER_TOKEN ... python3 -m pytest tests/test_invoice_summary_pack_breakdown.py -v`
Expected: FAIL — `KeyError: 'pack_breakdown'`.

- [ ] **Step 3: Write minimal implementation**

In `_invoice_summary` add `"pack_breakdown": _order_pack_breakdown(order),`. In the order-board annotate loop (~42262) add `o["pack_breakdown"] = _order_pack_breakdown(o)`. In `static/console-orders.html` and `static/invoice.html`, where the unit count renders, append `" + M cello packs"` when `pack_breakdown.cello_pack_units > 0` (leave the bottle-only display unchanged when it's 0).

- [ ] **Step 4: Run test to verify it passes**

Run: `env -u DOPPLER_TOKEN ... python3 -m pytest tests/test_invoice_summary_pack_breakdown.py -v`
Expected: PASS. Then node `--check` both HTML files.

- [ ] **Step 5: Commit**

```bash
git add app.py static/console-orders.html static/invoice.html tests/test_invoice_summary_pack_breakdown.py
git commit -m "feat: surface bottles/cello split on invoice + order board"
```

---

### Task 7: Regression sweep + live rate check

**Files:** none (verification)

- [ ] **Step 1:** Run the invoice/order/shipping suites under the fake env:

`env -u DOPPLER_TOKEN PINECONE_API_KEY=pcsk_fake OPENAI_API_KEY=sk-fake ANTHROPIC_API_KEY=sk-ant-fake SECRET_KEY=ci CONSOLE_SECRET=ci-fake-console-secret python3 -m pytest tests/test_orders_physical_units.py tests/test_inhouse_order_entry.py tests/test_invoice_edit.py tests/test_price_cart_cello_shipping.py tests/test_pack_breakdown.py tests/test_cello_packing_type.py tests/test_line_format_roundtrip.py tests/test_invoice_summary_pack_breakdown.py -q`
Expected: all pass, 0 new failures.

- [ ] **Step 2:** After deploy, quote a real cello order via `POST /api/orders/shipping-preview` and confirm the cello rate is below the all-bottle rate for the same qty. Note in the PR that Rae still needs to enter the authoritative `cello-refill` dims/capacity via `/admin/shipping` for prod rates to be exact (code seed is an estimate).

## Self-Review Notes

- **Spec coverage:** R1→Task 1; R2→Tasks 2-3; R3→Task 5; R4→Tasks 4,6 (with `physical_units` explicitly unchanged); R5 (label) already present, confirmed in Task 3/5; automation non-goal untouched.
- **`physical_units` guard:** Task 4's test asserts `physical_units` still returns the combined total — the reorder-demand invariant from the spec.
- **Data caveat:** prod needs Rae's `cello-refill` dims/capacity (Task 7 Step 2). Code degrades safely to today's behavior until then.
- **Open implementation detail flagged inline:** confirm the real `bottle_types` seed/list function names in Task 1, and bundle-expansion parity in Task 4, against the live code when implementing.
