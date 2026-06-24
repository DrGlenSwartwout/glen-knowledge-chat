# Geometric Order Packer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the dormant fractional-fill box picker with a true geometric packer that derives capacity from measured bottle dimensions, applies a tunable padding allowance for glass, and auto-splits oversized orders across multiple flat-rate boxes — so order-time shipping cost reflects real packing.

**Architecture:** A new pure-Python `dashboard/packing.py` does all geometry (no DB). `dashboard/shipping.py` gains bottle dimensions + a padding-settings table, and rewrites `pick_box`/`quote` to call the packer (falling back to the existing fractional logic for dimensionless types so current behavior is preserved). The checkout path (`_price_cart` → `_shipping_for_cart` → `quote`) is unchanged because `quote` keeps returning a single summed `shipping_cents`.

**Tech Stack:** Python 3, sqlite3 (`chat_log.db`), pytest. Stdlib only — no new dependencies.

## Global Constraints

- **No new dependencies.** Stdlib only (matches the cron-container constraint already noted in `shipping.py`).
- **Pricing is read-only.** Never write `usps_rates`; never alter the rate watcher or `/admin/shipping` rate flow. Rates are inputs.
- **Checkout never hard-fails.** Any unknown/dimensionless bottle type must fall back to existing behavior, never raise out of `quote`.
- **Box interiors (cm):** S = 5×15×23, M = 13×22×27, L = 14×29×30. Stored/computed in **mm** (cm×10) for integer math.
- **Bottle types + dims (Ø×H cm):** `120cap` 8×10, `100ml` 5×16, `30roll` 4×10, `50ml` 4×14, `15ml` 3×10, `5ml` 3×8, `100cos` 7×7, `30cap` 5×9.
- **Padding defaults:** `wrap_mm = 6` (added to each bottle Ø and H), `box_margin_mm = 10` (subtracted from each box interior dimension). Both tunable later.
- **Packer is conservative:** upright shelves only (no intra-layer stacking), no hex-nesting. "If it says it fits, it fits."
- **DB access pattern:** use `shipping._connect(db_path)` / `_default_db_path()`; `sqlite3.Row` factory; functions take an optional `db_path` kwarg (matches existing module style).
- **Schema migrations are idempotent**, run from `init_shipping_schema(cx)`.

---

### Task 1: Pure geometric packer — single-box fit

**Files:**
- Create: `dashboard/packing.py`
- Test: `tests/test_packing.py`

**Interfaces:**
- Consumes: nothing (pure stdlib).
- Produces:
  - `BOXES_MM: dict[str, tuple[int,int,int]]` = `{"S": (50,150,230), "M": (130,220,270), "L": (140,290,300)}`
  - `BOX_ORDER: tuple` = `("S","M","L")` (ascending cost/volume)
  - `fit_subset(items, box_mm, *, wrap_mm=0, box_margin_mm=0) -> set[int]` — indices of `items` placed in one box, best of 3 orientations. `items` is a list of `(diameter_mm, height_mm)`.
  - `fits_all(items, box_mm, *, wrap_mm=0, box_margin_mm=0) -> bool`
  - `pack_count(items, box_mm, *, wrap_mm=0, box_margin_mm=0) -> int`

- [ ] **Step 1: Write the failing test** (the 8 single-type counts, no padding — these are the verified reference numbers)

```python
# tests/test_packing.py
import pytest
from dashboard.packing import BOXES_MM, fit_subset, fits_all, pack_count

# Ø×H in mm (cm×10)
BOTTLES_MM = {
    "120cap": (80, 100), "100ml": (50, 160), "30roll": (40, 100),
    "50ml": (40, 140), "15ml": (30, 100), "5ml": (30, 80),
    "100cos": (70, 70), "30cap": (50, 90),
}
# Verified bare-geometry counts: (S, M, L)
EXPECTED = {
    "120cap": (0, 6, 9), "100ml": (3, 10, 12), "30roll": (6, 36, 63),
    "50ml": (5, 18, 49), "15ml": (10, 72, 108), "5ml": (10, 84, 120),
    "100cos": (0, 9, 32), "30cap": (6, 24, 36),
}

@pytest.mark.parametrize("key", list(EXPECTED))
def test_single_type_counts_match_reference(key):
    d, h = BOTTLES_MM[key]
    got = tuple(
        pack_count([(d, h)] * 500, BOXES_MM[size]) for size in ("S", "M", "L")
    )
    assert got == EXPECTED[key]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-59a2725d && python3 -m pytest tests/test_packing.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'dashboard.packing'`

- [ ] **Step 3: Write the implementation**

```python
# dashboard/packing.py
"""Pure-geometry packer: how many cylindrical bottles fit in a USPS flat-rate
box. Each bottle is modeled as a square prism (footprint Ø×Ø, height H), packed
upright in horizontal shelves; all 3 box orientations are tried and the best is
kept. Conservative: no intra-layer stacking, no hex nesting. No DB, no I/O.

All dimensions in millimetres (integers).
"""
from __future__ import annotations
from typing import List, Set, Tuple

# Interior dims in mm (cm x 10): S 5x15x23, M 13x22x27, L 14x29x30
BOXES_MM = {"S": (50, 150, 230), "M": (130, 220, 270), "L": (140, 290, 300)}
BOX_ORDER = ("S", "M", "L")  # ascending volume / flat-rate cost


def _pack2d(squares: List[int], bw: int, bl: int) -> List[bool]:
    """Shelf-pack square footprints (side lengths) into a bw x bl base.
    Returns a placed/not-placed flag per square. Largest-first within a row."""
    placed = [False] * len(squares)
    y = 0
    while True:
        idxs = [i for i, p in enumerate(placed) if not p]
        if not idxs:
            break
        cand = [i for i in idxs if y + squares[i] <= bl and squares[i] <= bw]
        if not cand:
            break
        first = max(cand, key=lambda i: squares[i])
        row_h = squares[first]
        x = 0
        while True:
            opts = [i for i in idxs if not placed[i]
                    and squares[i] <= row_h and x + squares[i] <= bw]
            if not opts:
                break
            pick = max(opts, key=lambda i: squares[i])
            placed[pick] = True
            x += squares[pick]
        y += row_h
    return placed


def _pack_oriented(items: List[Tuple[int, int]], bw: int, bl: int, H: int) -> Set[int]:
    """Layer-pack items (list of (d,h)) into a base bw x bl, vertical room H.
    Returns the set of placed indices."""
    order = sorted(range(len(items)), key=lambda i: (-items[i][1], -items[i][0]))
    placed: Set[int] = set()
    used = 0
    while True:
        rem = [i for i in order if i not in placed]
        if not rem:
            break
        tallest = items[rem[0]][1]
        if used + tallest > H:
            break
        layer_h = tallest
        eligible = [i for i in rem if items[i][1] <= layer_h]
        flags = _pack2d([items[i][0] for i in eligible], bw, bl)
        layer = [eligible[k] for k, f in enumerate(flags) if f]
        if not layer:
            break
        placed.update(layer)
        used += layer_h
    return placed


def _effective(items, wrap_mm):
    return [(d + wrap_mm, h + wrap_mm) for (d, h) in items]


def fit_subset(items, box_mm, *, wrap_mm=0, box_margin_mm=0) -> Set[int]:
    """Indices of `items` that fit in one box, best of 3 orientations."""
    if not items:
        return set()
    eff = _effective(items, wrap_mm)
    a, b, c = (d - box_margin_mm for d in box_mm)
    best: Set[int] = set()
    for vax, (bw, bl) in ((a, (b, c)), (b, (a, c)), (c, (a, b))):
        if vax <= 0 or bw <= 0 or bl <= 0:
            continue
        placed = _pack_oriented(eff, bw, bl, vax)
        if len(placed) > len(best):
            best = placed
    return best


def fits_all(items, box_mm, *, wrap_mm=0, box_margin_mm=0) -> bool:
    return len(fit_subset(items, box_mm, wrap_mm=wrap_mm,
                          box_margin_mm=box_margin_mm)) == len(items)


def pack_count(items, box_mm, *, wrap_mm=0, box_margin_mm=0) -> int:
    return len(fit_subset(items, box_mm, wrap_mm=wrap_mm,
                          box_margin_mm=box_margin_mm))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_packing.py -q`
Expected: PASS (8 parametrized cases)

- [ ] **Step 5: Add mixed-load + padding tests**

```python
def test_mixed_load_fits_medium():
    items = [BOTTLES_MM["120cap"]] * 2 + [BOTTLES_MM["15ml"]] * 6 + [BOTTLES_MM["5ml"]] * 10
    assert fits_all(items, BOXES_MM["M"])

def test_padding_lowers_capacity():
    bare = pack_count([BOTTLES_MM["5ml"]] * 500, BOXES_MM["M"])
    padded = pack_count([BOTTLES_MM["5ml"]] * 500, BOXES_MM["M"],
                        wrap_mm=6, box_margin_mm=10)
    assert padded < bare

def test_too_wide_bottle_never_fits_small():
    # 120cap Ø80mm exceeds S's two usable cross dims at any orientation
    assert pack_count([BOTTLES_MM["120cap"]], BOXES_MM["S"]) == 0
```

- [ ] **Step 6: Run all packing tests**

Run: `python3 -m pytest tests/test_packing.py -q`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add dashboard/packing.py tests/test_packing.py
git commit -m "feat(packing): pure geometric single-box bottle packer"
```

---

### Task 2: Multi-box split

**Files:**
- Modify: `dashboard/packing.py`
- Test: `tests/test_packing.py`

**Interfaces:**
- Consumes: `fit_subset`, `fits_all`, `BOXES_MM`, `BOX_ORDER` (Task 1).
- Produces: `split_into_boxes(items, *, wrap_mm=0, box_margin_mm=0) -> list[str] | None` — the box size(s) holding the whole load. Returns one element if a single box fits; otherwise fills L boxes greedily and sizes-down the last box. Returns `None` only if some single bottle cannot fit even an L box.

- [ ] **Step 1: Write the failing test**

```python
from dashboard.packing import split_into_boxes

def test_split_single_box_when_fits():
    assert split_into_boxes([BOTTLES_MM["15ml"]] * 5) == ["S"]

def test_split_picks_smallest_single_box():
    # 20 x 15ml fits L (108 cap) -> but also M (72). Smallest single box = M.
    assert split_into_boxes([BOTTLES_MM["15ml"]] * 20) == ["M"]

def test_split_into_multiple_boxes_when_oversized():
    # 200 x 15ml: L holds 108, so needs 2 boxes; last sizes down.
    boxes = split_into_boxes([BOTTLES_MM["15ml"]] * 200)
    assert boxes is not None
    assert len(boxes) == 2
    assert boxes[0] == "L"

def test_split_returns_none_when_bottle_too_big():
    # A bottle wider than every box cross-section.
    assert split_into_boxes([(200, 200)]) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_packing.py -k split -q`
Expected: FAIL — `ImportError: cannot import name 'split_into_boxes'`

- [ ] **Step 3: Implement**

```python
def split_into_boxes(items, *, wrap_mm=0, box_margin_mm=0):
    """Box size(s) that hold the whole load. Single smallest box if it fits;
    else greedily fill L boxes and size-down the final partial box. None if a
    single bottle cannot fit even an L box."""
    if not items:
        return []
    remaining = list(range(len(items)))
    out = []
    while remaining:
        sub = [items[i] for i in remaining]
        single = next(
            (s for s in BOX_ORDER
             if fits_all(sub, BOXES_MM[s], wrap_mm=wrap_mm, box_margin_mm=box_margin_mm)),
            None,
        )
        if single:
            out.append(single)
            break
        placed_local = fit_subset(sub, BOXES_MM["L"], wrap_mm=wrap_mm,
                                  box_margin_mm=box_margin_mm)
        if not placed_local:
            return None  # a single bottle doesn't fit even L
        out.append("L")
        placed_global = {remaining[k] for k in placed_local}
        remaining = [i for i in remaining if i not in placed_global]
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_packing.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dashboard/packing.py tests/test_packing.py
git commit -m "feat(packing): greedy multi-box split for oversized loads"
```

---

### Task 3: Schema — bottle dimensions + padding settings + seed

**Files:**
- Modify: `dashboard/shipping.py` (extend `init_shipping_schema`; add helpers)
- Test: `tests/test_shipping.py`

**Interfaces:**
- Consumes: existing `_connect`, `init_shipping_schema`, `BOX_SIZES`.
- Produces:
  - `bottle_types` gains nullable `diameter_mm INTEGER`, `height_mm INTEGER`.
  - New table `packing_settings(key TEXT PRIMARY KEY, value INTEGER)` seeded `wrap_mm=6`, `box_margin_mm=10` on first init.
  - 8 standard bottle types seeded with dims on first init (only if `bottle_types` is empty).
  - `get_packing_settings(db_path=None) -> dict` → `{"wrap_mm": int, "box_margin_mm": int}`
  - `set_packing_setting(key, value, db_path=None) -> None` (key must be one of the two; value ≥ 0)
  - `get_bottle_dims(db_path=None) -> dict[str, tuple[int,int]]` → `{name: (diameter_mm, height_mm)}` for rows where both dims are non-null.

- [ ] **Step 1: Write the failing test**

```python
def test_schema_adds_dims_and_seeds_standard_bottles(tmp_path):
    import sqlite3
    from dashboard.shipping import init_shipping_schema, get_bottle_dims, get_packing_settings
    db = str(tmp_path / "chat_log.db")
    with sqlite3.connect(db) as cx:
        init_shipping_schema(cx)
        init_shipping_schema(cx)  # idempotent
        cols = {r[1] for r in cx.execute("PRAGMA table_info(bottle_types)")}
    assert {"diameter_mm", "height_mm"} <= cols
    dims = get_bottle_dims(db_path=db)
    assert dims["15ml"] == (30, 100)
    assert dims["120cap"] == (80, 100)
    assert len(dims) == 8
    assert get_packing_settings(db_path=db) == {"wrap_mm": 6, "box_margin_mm": 10}

def test_set_packing_setting_updates_value(tmp_path):
    import sqlite3
    from dashboard.shipping import init_shipping_schema, set_packing_setting, get_packing_settings
    db = str(tmp_path / "chat_log.db")
    with sqlite3.connect(db) as cx:
        init_shipping_schema(cx)
    set_packing_setting("wrap_mm", 9, db_path=db)
    assert get_packing_settings(db_path=db)["wrap_mm"] == 9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_shipping.py -k "dims or packing_setting" -q`
Expected: FAIL — `OperationalError: no such column: diameter_mm` / `ImportError`

- [ ] **Step 3: Implement schema additions**

In `dashboard/shipping.py`, add the standard-bottle constant near `_DEFAULT_RATES_2026_04_26`:

```python
# Standard bottle types with measured dims (Ø_mm, H_mm) = cm x 10.
_STANDARD_BOTTLES = [
    ("120cap", "250 ml wide-mouth (120 caps)", 80, 100),
    ("100ml", "100 ml dropper", 50, 160),
    ("30roll", "30 ml roll-on", 40, 100),
    ("50ml", "50 ml dropper", 40, 140),
    ("15ml", "15 ml dropper", 30, 100),
    ("5ml", "5 ml dropper", 30, 80),
    ("100cos", "100 ml cosmetic (30 g powder)", 70, 70),
    ("30cap", "100 ml wide-mouth (30 caps)", 50, 90),
]
_PACKING_DEFAULTS = {"wrap_mm": 6, "box_margin_mm": 10}
_PACKING_KEYS = ("wrap_mm", "box_margin_mm")
```

Inside `init_shipping_schema(cx)`, after the three `CREATE TABLE` blocks and before `cx.commit()`, add idempotent column adds + the new table + seeds:

```python
    # Add dimension columns to bottle_types if missing (idempotent migration)
    cols = {r[1] for r in cx.execute("PRAGMA table_info(bottle_types)")}
    if "diameter_mm" not in cols:
        cx.execute("ALTER TABLE bottle_types ADD COLUMN diameter_mm INTEGER")
    if "height_mm" not in cols:
        cx.execute("ALTER TABLE bottle_types ADD COLUMN height_mm INTEGER")

    cx.execute("""
        CREATE TABLE IF NOT EXISTS packing_settings (
            key   TEXT PRIMARY KEY,
            value INTEGER NOT NULL
        )
    """)
    for k, v in _PACKING_DEFAULTS.items():
        cx.execute(
            "INSERT OR IGNORE INTO packing_settings (key, value) VALUES (?, ?)",
            (k, v),
        )

    # Seed the standard bottle types with dims only on a fresh catalog
    has_bottles = cx.execute("SELECT 1 FROM bottle_types LIMIT 1").fetchone()
    if not has_bottles:
        for name, notes, d_mm, h_mm in _STANDARD_BOTTLES:
            cx.execute(
                "INSERT INTO bottle_types (name, notes, diameter_mm, height_mm) "
                "VALUES (?, ?, ?, ?)",
                (name, notes, d_mm, h_mm),
            )
```

Add the helper functions (place after `get_capacity_matrix`):

```python
def get_bottle_dims(db_path: Optional[str] = None) -> Dict[str, tuple]:
    """{name: (diameter_mm, height_mm)} for types that have both dims set."""
    with _connect(db_path) as cx:
        rows = cx.execute(
            "SELECT name, diameter_mm, height_mm FROM bottle_types "
            "WHERE diameter_mm IS NOT NULL AND height_mm IS NOT NULL"
        ).fetchall()
    return {r["name"]: (r["diameter_mm"], r["height_mm"]) for r in rows}


def get_packing_settings(db_path: Optional[str] = None) -> Dict[str, int]:
    with _connect(db_path) as cx:
        rows = cx.execute("SELECT key, value FROM packing_settings").fetchall()
    out = dict(_PACKING_DEFAULTS)
    out.update({r["key"]: r["value"] for r in rows})
    return {k: int(out[k]) for k in _PACKING_KEYS}


def set_packing_setting(key: str, value: int, db_path: Optional[str] = None) -> None:
    if key not in _PACKING_KEYS:
        raise ValueError(f"key must be one of {_PACKING_KEYS}, got {key!r}")
    if int(value) < 0:
        raise ValueError("padding value must be >= 0")
    with _connect(db_path) as cx:
        cx.execute(
            "INSERT INTO packing_settings (key, value) VALUES (?, ?) "
            "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
            (key, int(value)),
        )
        cx.commit()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_shipping.py -k "dims or packing_setting" -q`
Expected: PASS

- [ ] **Step 5: Verify existing shipping tests still pass** (the seeded_db fixture inserts its own bottle types, so the seed-on-empty guard must not interfere)

Run: `python3 -m pytest tests/test_shipping.py -q`
Expected: PASS (all existing + new)

- [ ] **Step 6: Commit**

```bash
git add dashboard/shipping.py tests/test_shipping.py
git commit -m "feat(shipping): bottle dimensions + tunable padding settings"
```

---

### Task 4: Geometric quote with override caps + fractional fallback

**Files:**
- Modify: `dashboard/shipping.py` (`pick_box`, `quote`; add `pick_boxes`)
- Test: `tests/test_shipping.py`

**Interfaces:**
- Consumes: `dashboard.packing.split_into_boxes`, `fit_subset`, `BOXES_MM`; `get_bottle_dims`, `get_packing_settings`, `_capacity_lookup`, `get_current_rates`, `UnknownBottleType`.
- Produces:
  - `pick_boxes(bottles_by_type, db_path=None) -> list[str] | None` — geometric multi-box selection when **every** requested type has dims; else falls back to the existing fractional `pick_box` wrapped as a single-element list (or `None`). Raises `UnknownBottleType` if a type is in neither the dims set nor the capacity matrix. Applies per-(type,box) override caps from `box_capacity`.
  - `pick_box(bottles_by_type, db_path=None) -> str | None` — unchanged signature; returns the single box when one suffices, else `None`. Now geometric when all types have dims.
  - `quote(...)` — returns single-box payload as before, **plus** `box_sizes: list[str]` and (for multi-box) summed `shipping_cents` with a `box_breakdown: [{"box_size","charged_cents"}]`.

**Override-cap rule:** if `box_capacity` has a cell `(type, box)=cap`, that box may hold at most `cap` of that type regardless of geometry. Enforced by checking, for any box the geometry assigns, that no type exceeds its cap; if violated, that box is treated as not-fitting (forcing a split / larger box).

- [ ] **Step 1: Write the failing tests**

```python
# --- geometric path ---
@pytest.fixture
def geo_db(tmp_path):
    """Schema with the 8 seeded standard bottle types (dims) + rates."""
    import sqlite3
    from dashboard.shipping import init_shipping_schema
    db = str(tmp_path / "chat_log.db")
    with sqlite3.connect(db) as cx:
        init_shipping_schema(cx)
    return db

def test_quote_geometric_single_box(geo_db):
    from dashboard.shipping import quote
    q = quote({"15ml": 5}, db_path=geo_db)
    assert q["box_sizes"] == ["S"]
    assert q["shipping_cents"] == 1300  # S charged rate

def test_quote_geometric_multibox_sums_rates(geo_db):
    from dashboard.shipping import quote
    q = quote({"15ml": 200}, db_path=geo_db)  # needs 2 boxes
    assert len(q["box_sizes"]) == 2
    assert q["box_sizes"][0] == "L"
    assert q["shipping_cents"] == sum(b["charged_cents"] for b in q["box_breakdown"])

def test_pick_box_geometric_with_padding(geo_db):
    from dashboard.shipping import pick_box
    # 5ml in S with default padding still fits at least 1 -> S
    assert pick_box({"5ml": 4}, db_path=geo_db) == "S"

def test_override_cap_forces_larger_box(geo_db):
    import sqlite3
    from dashboard.shipping import pick_box, set_box_capacity
    with sqlite3.connect(geo_db) as cx:
        bid = cx.execute("SELECT id FROM bottle_types WHERE name='15ml'").fetchone()[0]
    set_box_capacity(bid, "S", 2, db_path=geo_db)  # cap S at 2 of 15ml
    # 4 of 15ml geometrically fit S, but cap=2 forces escalation to M
    assert pick_box({"15ml": 4}, db_path=geo_db) == "M"
```

These tests share the file with the existing dimensionless fractional tests (`seeded_db`), which must remain green.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_shipping.py -k "geometric or override_cap" -q`
Expected: FAIL — `KeyError: 'box_sizes'` / assertion failures (still fractional).

- [ ] **Step 3: Implement** — add `import` and rewrite the box-fit section.

At the top of `shipping.py` (with the other imports):

```python
from dashboard import packing
```

Replace `pick_box` (lines ~153–189) and add `pick_boxes` + an override-cap helper:

```python
def _expand_items(bottles_by_type, dims):
    """Flatten {type: qty} into a list of (d_mm, h_mm) plus a parallel list of
    type names (same order) so caps can be checked per box."""
    items, names = [], []
    for name, qty in bottles_by_type.items():
        d = dims[name]
        for _ in range(int(qty)):
            items.append(d)
            names.append(name)
    return items, names


def _caps_ok(box_size, names, placed_idx, caps):
    """True if the placed subset honours every (type, box) override cap."""
    from collections import Counter
    counts = Counter(names[i] for i in placed_idx)
    for tname, n in counts.items():
        cap = caps.get(tname, {}).get(box_size)
        if cap is not None and n > cap:
            return False
    return True


def pick_boxes(bottles_by_type, db_path: Optional[str] = None):
    """Geometric box selection. Returns a list of box sizes, or None.

    - If every requested type has dimensions -> geometric split (multi-box),
      honouring any (type, box) override caps from box_capacity.
    - Otherwise -> fall back to the legacy fractional pick_box (single box),
      returned as a one-element list, or None.
    Raises UnknownBottleType if a type is in neither the dims set nor the
    capacity matrix.
    """
    if not bottles_by_type:
        return None
    dims = get_bottle_dims(db_path=db_path)
    with _connect(db_path) as cx:
        caps = _capacity_lookup(cx)

    for name in bottles_by_type:
        if name not in dims and name not in caps:
            raise UnknownBottleType(name)

    if all(name in dims for name in bottles_by_type):
        settings = get_packing_settings(db_path=db_path)
        wrap, margin = settings["wrap_mm"], settings["box_margin_mm"]
        items, names = _expand_items(bottles_by_type, dims)
        # Geometric split; then verify each chosen box honours override caps.
        boxes = packing.split_into_boxes(items, wrap_mm=wrap, box_margin_mm=margin)
        if boxes is None:
            return None
        if not _caps_violated(items, names, caps, wrap, margin):
            return boxes
        return _split_with_caps(items, names, caps, wrap, margin)

    # Legacy fractional fallback (dimensionless types present)
    legacy = _pick_box_fractional(bottles_by_type, caps)
    return [legacy] if legacy else None


def _split_with_caps(items, names, caps, wrap, margin):
    """Greedy split that also respects override caps per box. Fills boxes one at
    a time, choosing the largest placement that satisfies caps; sizes the final
    box down. Returns list of box sizes or None."""
    remaining = list(range(len(items)))
    out = []
    while remaining:
        sub_items = [items[i] for i in remaining]
        sub_names = [names[i] for i in remaining]
        # smallest single box that fits all AND honours caps
        chosen = None
        for s in packing.BOX_ORDER:
            placed = packing.fit_subset(sub_items, packing.BOXES_MM[s],
                                        wrap_mm=wrap, box_margin_mm=margin)
            if len(placed) == len(sub_items) and _caps_ok(s, sub_names, placed, caps):
                chosen = s
                break
        if chosen:
            out.append(chosen)
            break
        # else pack into an L, dropping any bottle that would break a cap
        placed = packing.fit_subset(sub_items, packing.BOXES_MM["L"],
                                    wrap_mm=wrap, box_margin_mm=margin)
        placed = _trim_to_caps("L", sub_names, placed, caps)
        if not placed:
            return None
        out.append("L")
        placed_global = {remaining[k] for k in placed}
        remaining = [i for i in remaining if i not in placed_global]
    return out


def _trim_to_caps(box_size, names, placed_idx, caps):
    """Drop indices from a placed set until every cap is honoured."""
    from collections import Counter
    placed = set(placed_idx)
    counts = Counter(names[i] for i in placed)
    for tname, n in list(counts.items()):
        cap = caps.get(tname, {}).get(box_size)
        if cap is not None and n > cap:
            drop = [i for i in placed if names[i] == tname][cap:]
            placed.difference_update(drop)
    return placed


def _caps_violated(items, names, caps, wrap, margin):
    """Quick check: would the cap-free split place more of any type in a single
    box than its cap allows? Conservative — if any single box could exceed a
    cap, return True to route through _split_with_caps."""
    if not caps:
        return False
    # If no cap applies to any present type, nothing to enforce.
    present = set(names)
    return any(present & set(caps) for _ in (0,)) and any(
        any(s in caps.get(t, {}) for s in packing.BOX_ORDER) for t in present
    )


def _pick_box_fractional(bottles_by_type, caps):
    """Legacy fractional-fill: smallest box where sum(qty/capacity) <= 1.0."""
    for name in bottles_by_type:
        if name not in caps:
            raise UnknownBottleType(name)
    for size in BOX_SIZES:
        total_fill = 0.0
        ok = True
        for name, qty in bottles_by_type.items():
            cap = caps[name].get(size)
            if cap is None or cap <= 0:
                ok = False
                break
            total_fill += qty / cap
        if ok and total_fill <= 1.0:
            return size
    return None


def pick_box(bottles_by_type, db_path: Optional[str] = None):
    """Smallest single box that fits, or None (multi-box -> None). Geometric
    when all types have dims; legacy fractional otherwise."""
    boxes = pick_boxes(bottles_by_type, db_path=db_path)
    if boxes and len(boxes) == 1:
        return boxes[0]
    return None
```

Replace `quote` (lines ~217–244):

```python
def quote(bottles_by_type, db_path: Optional[str] = None) -> dict:
    """pick_boxes + current rates -> a single UI/checkout payload.

    Single box: {box_size, box_sizes:[size], shipping_cents, box_breakdown}.
    Multi box:  box_sizes has >1 entry; shipping_cents is the summed charged
                rate; box_breakdown lists each box + its charged_cents.
    """
    try:
        boxes = pick_boxes(bottles_by_type, db_path=db_path)
    except UnknownBottleType as e:
        return {"box_size": None, "box_sizes": [], "shipping_cents": None,
                "error": f"Unknown bottle type: {e}"}
    if not boxes:
        return {
            "box_size": None, "box_sizes": [], "shipping_cents": None,
            "error": ("Order is empty" if not bottles_by_type
                      else "Order does not fit available flat-rate boxes — "
                           "split shipment or use custom shipping."),
        }
    rates = get_current_rates(db_path=db_path)
    breakdown = []
    total = 0
    for size in boxes:
        if size not in rates:
            return {"box_size": size, "box_sizes": boxes, "shipping_cents": None,
                    "error": f"No confirmed USPS rate for {size} — check /admin/shipping."}
        cents = rates[size]["charged_cents"]
        breakdown.append({"box_size": size, "charged_cents": cents})
        total += cents
    return {
        "box_size": boxes[0],               # back-compat single-box field
        "box_sizes": boxes,
        "shipping_cents": total,
        "box_breakdown": breakdown,
        "rate_effective_date": rates[boxes[0]]["effective_date"],
    }
```

- [ ] **Step 4: Run the new + existing shipping tests**

Run: `python3 -m pytest tests/test_shipping.py -q`
Expected: PASS — new geometric/override tests pass; legacy `seeded_db` fractional tests still pass (they hit `_pick_box_fractional` because their types lack dims).

- [ ] **Step 5: Run the full suite for regressions**

Run: `python3 -m pytest tests/test_shipping.py tests/test_packing.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add dashboard/shipping.py tests/test_shipping.py
git commit -m "feat(shipping): geometric quote with override caps + multi-box sum"
```

---

### Task 5: Verify checkout integration (multi-box cost flows through)

**Files:**
- Test: `tests/test_packing_integration.py` (create)
- Modify: `app.py` only if the integration test reveals a gap (see note).

**Interfaces:**
- Consumes: `dashboard.shipping.quote` (Task 4). `_price_cart`/`_shipping_for_cart` already pass `box_counts` keyed by `bottle_type` and read `q["shipping_cents"]` — Task 4 keeps `shipping_cents` as a summed int, so no app change is expected.

- [ ] **Step 1: Write the integration test**

```python
# tests/test_packing_integration.py
"""quote() returns a single summed shipping_cents that _shipping_for_cart can
consume unchanged, including the multi-box case."""
import sqlite3
from dashboard.shipping import init_shipping_schema, quote

def _db(tmp_path):
    db = str(tmp_path / "chat_log.db")
    with sqlite3.connect(db) as cx:
        init_shipping_schema(cx)
    return db

def test_quote_shipping_cents_is_int_single(tmp_path):
    q = quote({"15ml": 5}, db_path=_db(tmp_path))
    assert isinstance(q["shipping_cents"], int) and q["shipping_cents"] > 0

def test_quote_shipping_cents_is_int_multibox(tmp_path):
    q = quote({"15ml": 200}, db_path=_db(tmp_path))
    assert isinstance(q["shipping_cents"], int)
    assert q["shipping_cents"] == 3200 + q["box_breakdown"][1]["charged_cents"]
```

- [ ] **Step 2: Run test**

Run: `python3 -m pytest tests/test_packing_integration.py -q`
Expected: PASS. If it passes, **no app.py change is needed** — `_shipping_for_cart` already reads `shipping_cents`. Skip Step 3.

- [ ] **Step 3 (only if Step 2 fails): adjust `_shipping_for_cart`**

If a gap appears (e.g. `quote` shape mismatch), update `app.py:3380-3383` to read `q.get("shipping_cents")` exactly as now; do not add multi-box branching there — the sum lives in `quote`. Re-run Step 2.

- [ ] **Step 4: Commit**

```bash
git add tests/test_packing_integration.py
git commit -m "test(shipping): checkout consumes summed multi-box shipping_cents"
```

---

### Task 6: Auto-infer product → bottle-type mapping

**Files:**
- Create: `scripts/infer_bottle_types.py`
- Test: `tests/test_infer_bottle_types.py`

**Interfaces:**
- Consumes: `dashboard.products.load_products`; the 8 type keys from `_STANDARD_BOTTLES` (Task 3).
- Produces:
  - `infer_bottle_type(product: dict) -> tuple[str, float]` → `(type_key_or_"default", confidence_0_to_1)`. Pure function (name/size/category heuristics).
  - `build_mapping(products: dict) -> dict` → `{"assignments": {slug: type_key}, "review": [{"slug","name","guess","confidence","reason"}]}`. Only products currently lacking a `bottle_type` are assigned; low-confidence (<0.6) guesses go to `default` and into `review`.
  - CLI: `python3 scripts/infer_bottle_types.py [--write]` — prints the review report; `--write` patches `data/products.json` in place (sets `bottle_type` only where unset), preserving all other fields and key order.

**Heuristics (explicit):** match on `name`/`description` lowercased —
`"dropper"`+`"1 oz"|"30 ml"|"1oz"` → `30roll`? No: dropper sizes map by ml — `"15 ml"`→`15ml`, `"5 ml"`→`5ml`, `"50 ml"`+`"dropper"`→`50ml`, `"100 ml"`+`"dropper"`→`100ml`; `"roll-on"|"rollon"|"roll on"` → `30roll`; `"powder"|"cosmetic"|"30 g"|"30g"` → `100cos`; `"capsule"|"caps"|"60 ct"|"vcaps"|"wide mouth"`+`"100 ml"` → `30cap`; `"250 ml"|"wide-mouth"|"wide mouth"` (no caps signal) → `120cap`. Confidence: 0.9 if both a form word (dropper/roll-on/powder/caps) and a size match; 0.7 if only one; else 0.3 → `default`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_infer_bottle_types.py
from scripts.infer_bottle_types import infer_bottle_type, build_mapping

def test_dropper_15ml():
    t, c = infer_bottle_type({"name": "Foo 15 ml Dropper", "description": ""})
    assert t == "15ml" and c >= 0.9

def test_rollon():
    t, c = infer_bottle_type({"name": "Bar Roll-On 30 ml", "description": ""})
    assert t == "30roll" and c >= 0.7

def test_powder_cosmetic():
    t, _ = infer_bottle_type({"name": "Baz Powder 30 g", "description": ""})
    assert t == "100cos"

def test_low_confidence_defaults_and_flags_for_review():
    m = build_mapping({"x": {"name": "Mystery Tonic", "description": ""}})
    assert m["assignments"]["x"] == "default"
    assert any(r["slug"] == "x" for r in m["review"])

def test_existing_bottle_type_is_preserved():
    m = build_mapping({"y": {"name": "Foo 15 ml Dropper", "bottle_type": "5ml"}})
    assert "y" not in m["assignments"]  # already set -> not reassigned
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_infer_bottle_types.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.infer_bottle_types'`

- [ ] **Step 3: Implement**

```python
# scripts/infer_bottle_types.py
"""Infer each product's bottle_type from its name/description, for the geometric
shipping packer. Pure heuristics + a CLI that patches data/products.json
(only where bottle_type is unset). Re-runnable; never overwrites an existing
assignment. Produces a human-review list for low-confidence guesses."""
from __future__ import annotations
import argparse
import json
import os
import sys

TYPES = ("120cap", "100ml", "30roll", "50ml", "15ml", "5ml", "100cos", "30cap")
_REVIEW_THRESHOLD = 0.6


def _text(p):
    return f"{p.get('name','')} {p.get('description','')}".lower()


def infer_bottle_type(product: dict):
    t = _text(product)
    has = lambda *ws: any(w in t for w in ws)
    form = None
    if has("roll-on", "rollon", "roll on"):
        return ("30roll", 0.9 if has("30 ml", "30ml") else 0.7)
    if has("powder", "cosmetic", "30 g", "30g"):
        return ("100cos", 0.9 if has("powder", "cosmetic") else 0.7)
    if has("dropper"):
        form = True
        if has("100 ml", "100ml"):
            return ("100ml", 0.9)
        if has("50 ml", "50ml"):
            return ("50ml", 0.9)
        if has("15 ml", "15ml"):
            return ("15ml", 0.9)
        if has("5 ml", "5ml"):
            return ("5ml", 0.9)
        return ("default", 0.3)  # dropper but no recognizable size
    if has("capsule", "caps", "vcaps", "60 ct", "60ct", "30 capsules"):
        return ("30cap", 0.9 if has("100 ml", "100ml") else 0.7)
    if has("250 ml", "250ml", "wide-mouth", "wide mouth"):
        return ("120cap", 0.9 if has("250 ml", "250ml") else 0.7)
    return ("default", 0.3)


def build_mapping(products: dict) -> dict:
    assignments, review = {}, []
    for slug, p in products.items():
        if p.get("bottle_type"):
            continue  # never overwrite an existing assignment
        guess, conf = infer_bottle_type(p)
        final = guess if conf >= _REVIEW_THRESHOLD else "default"
        assignments[slug] = final
        if final == "default" or conf < _REVIEW_THRESHOLD:
            review.append({"slug": slug, "name": p.get("name", ""),
                           "guess": guess, "confidence": conf,
                           "reason": "low confidence" if conf < _REVIEW_THRESHOLD
                                     else "no size match"})
    return {"assignments": assignments, "review": review}


def _products_json_path():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = os.environ.get("DATA_DIR")
    if d and os.path.exists(os.path.join(d, "products.json")):
        return os.path.join(d, "products.json")
    return os.path.join(here, "data", "products.json")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="patch products.json (set bottle_type only where unset)")
    args = ap.parse_args(argv)
    path = _products_json_path()
    doc = json.load(open(path))
    products = doc.get("products", {})
    m = build_mapping(products)
    print(f"{len(m['assignments'])} products to assign; "
          f"{len(m['review'])} need review.")
    for r in m["review"]:
        print(f"  REVIEW {r['slug']}: {r['name']!r} -> {r['guess']} "
              f"(conf {r['confidence']})")
    if args.write:
        for slug, t in m["assignments"].items():
            products[slug]["bottle_type"] = t
        json.dump(doc, open(path, "w"), indent=2, ensure_ascii=False)
        print(f"Wrote {len(m['assignments'])} assignments to {path}")
    else:
        print("(dry run — pass --write to patch products.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_infer_bottle_types.py -q`
Expected: PASS

- [ ] **Step 5: Dry-run the inference over the real catalog (no write) and eyeball the review list**

Run: `python3 scripts/infer_bottle_types.py`
Expected: prints assignment + review counts; no file change. (Glen reviews the list; `--write` is run later under his eye — not part of this task's commit.)

- [ ] **Step 6: Commit**

```bash
git add scripts/infer_bottle_types.py tests/test_infer_bottle_types.py
git commit -m "feat(shipping): product -> bottle-type inference script + review report"
```

---

## Self-Review

**Spec coverage:**
- Geometric packer (pure) → Task 1. ✓
- Multi-box auto-split → Task 2 + surfaced in `quote` (Task 4). ✓
- Bottle dimensions on `bottle_types` + seed 8 types → Task 3. ✓
- Tunable `wrap_mm`/`box_margin_mm` (defaults 6/10) → Task 3. ✓
- Per-product mapping (auto-infer + review, only fills unset) → Task 6. ✓
- Geometric `quote`, flat-rate pricing untouched, multi-box summed cost → Task 4. ✓
- `box_capacity` retained as override hard cap → Task 4. ✓
- Fallback to qty rule for unknown/dimensionless types (checkout never fails) → Task 4 (`_pick_box_fractional` + `quote` error payload → `_shipping_for_cart` qty fallback) + verified Task 5. ✓
- Tests across single-type, mixed, padding, multi-box, override, fallback → Tasks 1–6. ✓

**Deferred (explicitly out of this plan, per spec "open details"):** the admin UI surface for editing the two padding knobs is left to `set_packing_setting` + a future `/admin/shipping` control; the `--write` run of the inference script is operator-run under Glen's review, not automated here.

**Placeholder scan:** no TBD/TODO; every code step has complete code. ✓

**Type consistency:** `fit_subset`/`fits_all`/`pack_count`/`split_into_boxes` signatures consistent across Tasks 1–4; `pick_boxes` returns `list[str]|None`; `quote` keys (`box_size`, `box_sizes`, `shipping_cents`, `box_breakdown`) consistent between Task 4 definition and Task 5 consumption; `get_bottle_dims`/`get_packing_settings`/`set_packing_setting` consistent Tasks 3↔4. ✓
