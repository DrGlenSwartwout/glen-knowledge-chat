"""Shipping — USPS Flat Rate auto-update + box-fit matrix for manual orders.

Three tables on the existing chat_log.db (persistent /data disk on Render):

  bottle_types   — Glen/Rae catalog of SKU shapes (e.g., "dropper 1oz")
  box_capacity   — per (bottle_type, box_size) → max qty that fits
  usps_rates     — historical USPS retail prices + Glen's charged price after
                   the rounding rule (≥50¢ up, ≤49¢ down). Each rate row
                   carries `confirmed_by` so we can stage proposed updates
                   from the watcher and require Glen's manual approval before
                   they go live (matches the "money changes are human-triggered"
                   rule).

Public surface used by app.py routes + the order-entry page:

  init_shipping_schema(cx)      — idempotent migration
  round_to_dollar(cents)        — Glen's rounding rule
  pick_box({bottle_name: qty})  — smallest S/M/L box that fits the order
  quote({bottle_name: qty})     — pick_box + current charged shipping cost
  get_current_rates()           — latest confirmed rate per box size
  list_bottle_types() / add_bottle_type() / delete_bottle_type()
  get_capacity_matrix() / set_box_capacity()
  propose_rate_update() / list_pending_rate_updates() / confirm_rate_update()
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from dashboard import packing


BOX_SIZES = ("S", "M", "L")
PENDING = "pending"


class UnknownBottleType(Exception):
    """Raised when an order references a bottle type not in the catalog."""


# ── Path resolution ───────────────────────────────────────────────────────────

def _default_db_path() -> str:
    """Same resolution rule as app.LOG_DB so tests + prod both work."""
    base = os.environ.get("DATA_DIR", str(Path(__file__).resolve().parent.parent))
    return str(Path(base) / "chat_log.db")


def _connect(db_path: Optional[str] = None) -> sqlite3.Connection:
    cx = sqlite3.connect(db_path or _default_db_path())
    cx.row_factory = sqlite3.Row
    cx.execute("PRAGMA foreign_keys = ON")
    return cx


# ── Schema ────────────────────────────────────────────────────────────────────

# Current confirmed rates as of the April 26, 2026 USPS price adjustment
# (runs through Jan 17, 2027). These get seeded on first init so the order
# tool works immediately after deploy. Glen can override via /admin/shipping.
_DEFAULT_RATES_2026_04_26 = [
    # (box_size, usps_retail_cents, source_url, effective_date)
    ("S", 1265, "https://www.usps.com/business/prices.htm", "2026-04-26"),
    ("M", 2295, "https://www.usps.com/business/prices.htm", "2026-04-26"),
    ("L", 3150, "https://www.usps.com/business/prices.htm", "2026-04-26"),
]

# Standard bottle types with measured dims (Ø_mm, H_mm) = cm x 10.
_STANDARD_BOTTLES = [
    # Shipping proxy: the hand cradle is a flat device, not a bottle, but the packer
    # models cylinders. These dims resolve one cradle to the USPS Medium Flat Rate box
    # (M, $22.95) — the default for a $297 device shipped with its protective packaging.
    # Rae can manually drop to Small at pack time when shipping the cradle bare.
    ("handcradle", "ZYTO Hand Cradle — ships USPS Medium Flat Rate", 80, 100),
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
_PACKING_DEFAULTS = {"wrap_mm": 6, "box_margin_mm": 10}
_PACKING_KEYS = ("wrap_mm", "box_margin_mm")


def init_shipping_schema(cx: sqlite3.Connection) -> None:
    """Create the three shipping tables. Idempotent.

    Seeds the April 26 2026 USPS Flat Rate prices on first init (only when
    the usps_rates table is empty) so the order tool works immediately on
    a fresh deploy.
    """
    cx.execute("""
        CREATE TABLE IF NOT EXISTS bottle_types (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT    NOT NULL UNIQUE,
            notes       TEXT,
            created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
        )
    """)
    cx.execute("""
        CREATE TABLE IF NOT EXISTS box_capacity (
            bottle_type_id  INTEGER NOT NULL,
            box_size        TEXT    NOT NULL CHECK (box_size IN ('S','M','L')),
            qty             INTEGER NOT NULL CHECK (qty > 0),
            PRIMARY KEY (bottle_type_id, box_size),
            FOREIGN KEY (bottle_type_id) REFERENCES bottle_types(id) ON DELETE CASCADE
        )
    """)
    cx.execute("""
        CREATE TABLE IF NOT EXISTS usps_rates (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            box_size            TEXT    NOT NULL CHECK (box_size IN ('S','M','L')),
            usps_retail_cents   INTEGER NOT NULL,
            charged_cents       INTEGER NOT NULL,
            effective_date      TEXT    NOT NULL,
            source_url          TEXT,
            confirmed_by        TEXT,
            confirmed_at        TEXT,
            created_at          TEXT    NOT NULL DEFAULT (datetime('now'))
        )
    """)
    cx.execute(
        "CREATE INDEX IF NOT EXISTS idx_usps_rates_size_date "
        "ON usps_rates(box_size, effective_date)"
    )

    # Add dimension columns to bottle_types if missing (idempotent migration)
    cols = {r[1] for r in cx.execute("PRAGMA table_info(bottle_types)")}
    if "diameter_mm" not in cols:
        cx.execute("ALTER TABLE bottle_types ADD COLUMN diameter_mm INTEGER")
    if "height_mm" not in cols:
        cx.execute("ALTER TABLE bottle_types ADD COLUMN height_mm INTEGER")

    # Rename legacy 100cos -> 30g if present and 30g not already there
    have = {r[0] for r in cx.execute("SELECT name FROM bottle_types")}
    if "100cos" in have and "30g" not in have:
        cx.execute("UPDATE bottle_types SET name='30g' WHERE name='100cos'")
        have.discard("100cos"); have.add("30g")
    # Ensure 30ml exists with dims (insert if missing) — only on existing (non-empty) catalogs;
    # fresh DBs get it via the seed block below.
    if have and "30ml" not in have:
        cx.execute("INSERT INTO bottle_types (name, notes, diameter_mm, height_mm) "
                   "VALUES ('30ml', '30 ml dropper (infoceutical)', 40, 110)")

    cx.execute("""
        CREATE TABLE IF NOT EXISTS product_bottle_types (
            slug         TEXT PRIMARY KEY,
            bottle_type  TEXT NOT NULL,
            updated_at   TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
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

    # First-run seed: only if the table is empty
    has_any = cx.execute("SELECT 1 FROM usps_rates LIMIT 1").fetchone()
    if not has_any:
        now = datetime.now(timezone.utc).isoformat()
        for size, retail, source, eff in _DEFAULT_RATES_2026_04_26:
            cx.execute("""
                INSERT INTO usps_rates
                    (box_size, usps_retail_cents, charged_cents, effective_date,
                     source_url, confirmed_by, confirmed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (size, retail, round_to_dollar(retail), eff, source, "seed", now))

    cx.commit()


# ── Rounding rule ─────────────────────────────────────────────────────────────

def round_to_dollar(cents: int) -> int:
    """Round to the nearest whole dollar. ≥50¢ rounds up, ≤49¢ rounds down."""
    dollars, remainder = divmod(int(cents), 100)
    if remainder >= 50:
        dollars += 1
    return dollars * 100


# ── Box-fit logic ─────────────────────────────────────────────────────────────

def _capacity_lookup(cx: sqlite3.Connection) -> Dict[str, Dict[str, int]]:
    """Returns {bottle_name: {S: qty, M: qty, L: qty}}."""
    out: Dict[str, Dict[str, int]] = {}
    rows = cx.execute("""
        SELECT bt.name AS name, bc.box_size AS size, bc.qty AS qty
        FROM bottle_types bt
        JOIN box_capacity bc ON bc.bottle_type_id = bt.id
    """).fetchall()
    for r in rows:
        out.setdefault(r["name"], {})[r["size"]] = r["qty"]
    return out


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


# ── Rate retrieval ────────────────────────────────────────────────────────────

def get_current_rates(db_path: Optional[str] = None) -> Dict[str, dict]:
    """Latest confirmed rate per box size.

    Returns {S: {usps_retail_cents, charged_cents, effective_date, ...}, M: ..., L: ...}
    """
    out: Dict[str, dict] = {}
    with _connect(db_path) as cx:
        for size in BOX_SIZES:
            row = cx.execute("""
                SELECT id, box_size, usps_retail_cents, charged_cents,
                       effective_date, source_url, confirmed_by, confirmed_at
                FROM usps_rates
                WHERE box_size = ? AND confirmed_by IS NOT NULL AND confirmed_by != ?
                ORDER BY effective_date DESC, id DESC
                LIMIT 1
            """, (size, PENDING)).fetchone()
            if row:
                out[size] = dict(row)
    return out


# ── Quote (used by the order-entry page) ──────────────────────────────────────

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
                      else "Order exceeds available flat-rate boxes — "
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


# ── CRUD: bottle_types ────────────────────────────────────────────────────────

def add_bottle_type(
    name: str,
    diameter_mm: Optional[int] = None,
    height_mm: Optional[int] = None,
    notes: Optional[str] = None,
    db_path: Optional[str] = None,
) -> int:
    with _connect(db_path) as cx:
        cur = cx.execute(
            "INSERT INTO bottle_types (name, notes, diameter_mm, height_mm) "
            "VALUES (?, ?, ?, ?)",
            (name, notes, diameter_mm, height_mm),
        )
        cx.commit()
        return int(cur.lastrowid)


def list_bottle_types(db_path: Optional[str] = None) -> List[dict]:
    with _connect(db_path) as cx:
        rows = cx.execute(
            "SELECT id, name, notes, diameter_mm, height_mm, created_at "
            "FROM bottle_types ORDER BY name"
        ).fetchall()
        return [dict(r) for r in rows]


def delete_bottle_type(bottle_type_id: int, db_path: Optional[str] = None) -> None:
    with _connect(db_path) as cx:
        cx.execute("DELETE FROM bottle_types WHERE id = ?", (bottle_type_id,))
        cx.commit()


def update_bottle_type(
    bottle_type_id: int,
    name: str,
    diameter_mm: Optional[int] = None,
    height_mm: Optional[int] = None,
    notes: Optional[str] = None,
    db_path: Optional[str] = None,
) -> None:
    with _connect(db_path) as cx:
        cx.execute(
            "UPDATE bottle_types SET name=?, notes=?, diameter_mm=?, height_mm=? "
            "WHERE id=?",
            (name, notes, diameter_mm, height_mm, bottle_type_id),
        )
        cx.commit()


# ── CRUD: box_capacity ────────────────────────────────────────────────────────

def set_box_capacity(
    bottle_type_id: int,
    box_size: str,
    qty: int,
    db_path: Optional[str] = None,
) -> None:
    """Upsert capacity for one (bottle_type, box_size) cell."""
    if box_size not in BOX_SIZES:
        raise ValueError(f"box_size must be one of {BOX_SIZES}, got {box_size!r}")
    with _connect(db_path) as cx:
        cx.execute("""
            INSERT INTO box_capacity (bottle_type_id, box_size, qty)
            VALUES (?, ?, ?)
            ON CONFLICT (bottle_type_id, box_size) DO UPDATE SET qty = excluded.qty
        """, (bottle_type_id, box_size, qty))
        cx.commit()


def get_capacity_matrix(db_path: Optional[str] = None) -> List[dict]:
    """Returns one row per bottle type with S/M/L columns (None if unset).

    Shape: [{"id": 1, "name": "dropper 1oz", "notes": ..., "S": 6, "M": 12, "L": 24}, ...]
    """
    with _connect(db_path) as cx:
        bottles = cx.execute(
            "SELECT id, name, notes FROM bottle_types ORDER BY name"
        ).fetchall()
        caps = cx.execute(
            "SELECT bottle_type_id, box_size, qty FROM box_capacity"
        ).fetchall()
    by_id: Dict[int, Dict[str, Optional[int]]] = {
        b["id"]: {"S": None, "M": None, "L": None} for b in bottles
    }
    for c in caps:
        by_id[c["bottle_type_id"]][c["box_size"]] = c["qty"]
    return [
        {"id": b["id"], "name": b["name"], "notes": b["notes"], **by_id[b["id"]]}
        for b in bottles
    ]


def get_bottle_dims(db_path: Optional[str] = None) -> Dict[str, tuple]:
    """{name: (diameter_mm, height_mm)} for types that have both dims set."""
    with _connect(db_path) as cx:
        rows = cx.execute(
            "SELECT name, diameter_mm, height_mm FROM bottle_types "
            "WHERE diameter_mm IS NOT NULL AND height_mm IS NOT NULL"
        ).fetchall()
    return {r["name"]: (r["diameter_mm"], r["height_mm"]) for r in rows}


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


def is_shippable(product) -> bool:
    """True when a catalog product is a physical good the packer should count.

    Services (`service`) and digital/info-only listings (`info_only`) have no
    bottle, so counting them would push the "default" placeholder type into
    quote() and charge for a phantom bottle. Anything else stays shippable —
    a physical product with a missing bottle mapping must still fail loudly
    rather than silently ship for free."""
    p = product or {}
    return not (p.get("service") or p.get("info_only"))


def resolve_bottle_type(slug, product, db_path=None):
    with _connect(db_path) as cx:
        row = cx.execute(
            "SELECT bottle_type FROM product_bottle_types WHERE slug=?", (slug,)
        ).fetchone()
    if row:
        return row["bottle_type"]
    return (product or {}).get("bottle_type") or "default"


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


# ── Rate update flow (manual approval) ────────────────────────────────────────

def propose_rate_update(
    box_size: str,
    usps_retail_cents: int,
    source_url: str,
    effective_date: str,
    db_path: Optional[str] = None,
) -> int:
    """Stage a new rate row with confirmed_by='pending'. Glen confirms via UI."""
    if box_size not in BOX_SIZES:
        raise ValueError(f"box_size must be one of {BOX_SIZES}, got {box_size!r}")
    charged = round_to_dollar(usps_retail_cents)
    with _connect(db_path) as cx:
        cur = cx.execute("""
            INSERT INTO usps_rates
                (box_size, usps_retail_cents, charged_cents, effective_date,
                 source_url, confirmed_by, confirmed_at)
            VALUES (?, ?, ?, ?, ?, ?, NULL)
        """, (box_size, usps_retail_cents, charged, effective_date, source_url, PENDING))
        cx.commit()
        return int(cur.lastrowid)


def list_pending_rate_updates(db_path: Optional[str] = None) -> List[dict]:
    with _connect(db_path) as cx:
        rows = cx.execute("""
            SELECT id, box_size, usps_retail_cents, charged_cents,
                   effective_date, source_url, created_at
            FROM usps_rates
            WHERE confirmed_by = ?
            ORDER BY created_at DESC
        """, (PENDING,)).fetchall()
        return [dict(r) for r in rows]


def confirm_rate_update(
    rate_id: int,
    confirmed_by: str,
    db_path: Optional[str] = None,
) -> None:
    """Mark a pending rate row as live."""
    if confirmed_by == PENDING:
        raise ValueError("confirmed_by cannot be 'pending'")
    now = datetime.now(timezone.utc).isoformat()
    with _connect(db_path) as cx:
        cx.execute(
            "UPDATE usps_rates SET confirmed_by = ?, confirmed_at = ? WHERE id = ?",
            (confirmed_by, now, rate_id),
        )
        cx.commit()


def delete_rate(rate_id: int, db_path: Optional[str] = None) -> str:
    """Un-set a rate row (a wrong entry) so the box size reverts to its previous
    confirmed rate. Refuses to delete the last remaining confirmed rate for a size
    (which would leave that size with no price). Returns the deleted row's box_size.
    Raises ValueError if the row is missing or is the only fallback for its size."""
    with _connect(db_path) as cx:
        row = cx.execute("SELECT box_size, confirmed_by FROM usps_rates WHERE id = ?",
                         (rate_id,)).fetchone()
        if not row:
            raise ValueError(f"no rate with id {rate_id}")
        box_size = row["box_size"]
        # A confirmed rate may only be un-set when another confirmed rate exists for
        # the same size to fall back to; a pending row can always be removed.
        if row["confirmed_by"] not in (None, PENDING):
            others = cx.execute(
                "SELECT COUNT(*) FROM usps_rates WHERE box_size = ? AND id != ? "
                "AND confirmed_by IS NOT NULL AND confirmed_by != ?",
                (box_size, rate_id, PENDING)).fetchone()[0]
            if others == 0:
                raise ValueError(
                    f"can't un-set the only confirmed rate for size {box_size} "
                    f"(there would be no fallback price)")
        cx.execute("DELETE FROM usps_rates WHERE id = ?", (rate_id,))
        cx.commit()
        return box_size


# ── USPS rate watcher ────────────────────────────────────────────────────────
# Scrapes a USPS retail-prices page and compares to current confirmed rates.
# Pure parsing is split from network fetch so it's unit-testable. Source URL
# can be overridden via USPS_PRICES_URL env var if the page structure shifts.

import re

USPS_PRICES_URL = os.environ.get(
    "USPS_PRICES_URL",
    "https://www.usps.com/business/prices.htm",
)

# A scraped rate that moves more than this fraction from the current confirmed rate
# is treated as a likely mis-scrape (USPS page layout shift) and flagged for manual
# review rather than auto-proposed. Flat-rate box prices don't jump 40% year to year.
MAX_RATE_JUMP_PCT = 0.40

# Order matters: "Large" before "Medium" before "Small" so the regex doesn't
# greedy-match "Small" inside "Small Flat Rate Box". We match per-size
# independently anyway, but keep this list as the canonical lookup.
_SIZE_PATTERNS = [
    ("S", re.compile(r"Small\s+Flat\s+Rate\s+Box[^$]{0,200}\$(\d+\.\d{2})", re.I | re.S)),
    ("M", re.compile(r"Medium\s+Flat\s+Rate\s+Box[^$]{0,200}\$(\d+\.\d{2})", re.I | re.S)),
    ("L", re.compile(r"Large\s+Flat\s+Rate\s+Box[^$]{0,200}\$(\d+\.\d{2})", re.I | re.S)),
]


def _parse_usps_html(html: str) -> Dict[str, int]:
    """Extract S/M/L Flat Rate retail prices in cents from the USPS page HTML.

    Raises ValueError if any size is missing — caller should treat that as
    "source page changed, need a manual update via /admin/shipping".
    """
    out: Dict[str, int] = {}
    for size, pat in _SIZE_PATTERNS:
        m = pat.search(html)
        if m:
            dollars, cents = m.group(1).split(".")
            out[size] = int(dollars) * 100 + int(cents)
    missing = [s for s in ("S", "M", "L") if s not in out]
    if missing:
        raise ValueError(
            f"USPS price scrape missing sizes: {missing}. "
            f"Source page format may have changed — update USPS_PRICES_URL "
            f"or use the manual-propose form at /admin/shipping."
        )
    return out


def fetch_usps_retail_prices(timeout: int = 30) -> Dict[str, int]:
    """Fetch + parse current USPS Flat Rate retail prices in cents.

    Stdlib-only so it works in the cron container (no extra deps).
    """
    import urllib.request
    req = urllib.request.Request(
        USPS_PRICES_URL,
        headers={"User-Agent": "glen-knowledge-chat/1.0 (rate-watcher)"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        html = resp.read().decode("utf-8", errors="replace")
    return _parse_usps_html(html)


def check_usps_rates(
    today: Optional[str] = None,
    db_path: Optional[str] = None,
) -> dict:
    """Watcher entry point — called from /cron/usps-rate-check.

    Compares scraped USPS retail prices against the latest confirmed rate per
    box size. For any size whose retail price differs (and which doesn't
    already have an identical pending proposal), stages a `propose_rate_update`.

    Returns a summary dict for the cron log:
        {checked_at, scraped: {S, M, L}, proposed: [...], unchanged: [...], errors: [...]}
    """
    today = today or datetime.now(timezone.utc).date().isoformat()
    summary = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "scraped": None,
        "proposed": [],
        "unchanged": [],
        "flagged": [],
        "errors": [],
    }
    try:
        scraped = fetch_usps_retail_prices()
        summary["scraped"] = scraped
    except Exception as e:
        summary["errors"].append(f"fetch failed: {e}")
        return summary

    current = get_current_rates(db_path=db_path)
    pending = list_pending_rate_updates(db_path=db_path)
    pending_keys = {(p["box_size"], p["usps_retail_cents"]) for p in pending}

    for size, retail_cents in scraped.items():
        cur = current.get(size)
        if cur and cur["usps_retail_cents"] == retail_cents:
            summary["unchanged"].append(size)
            continue
        if (size, retail_cents) in pending_keys:
            summary["unchanged"].append(f"{size} (already pending)")
            continue
        # Plausibility guard: the USPS page layout can shift so the scraper grabs an
        # adjacent (wrong) price. A confirmed rate that jumps more than MAX_RATE_JUMP_PCT
        # is almost certainly a mis-scrape, so FLAG it for manual review instead of
        # auto-proposing garbage that could then be confirmed (the 2026-07 incident:
        # $12.65 -> $33.75). A first-ever rate (no current) is not gated.
        if cur and cur.get("usps_retail_cents"):
            base = cur["usps_retail_cents"]
            if abs(retail_cents - base) > base * MAX_RATE_JUMP_PCT:
                summary["flagged"].append({
                    "box_size": size, "scraped_cents": retail_cents, "current_cents": base,
                    "reason": (f"scraped ${retail_cents/100:.2f} differs more than "
                               f"{int(MAX_RATE_JUMP_PCT*100)}% from the current ${base/100:.2f}; "
                               f"not auto-proposed (possible USPS page change) — review at /admin/shipping"),
                })
                continue
        try:
            rid = propose_rate_update(
                box_size=size,
                usps_retail_cents=retail_cents,
                source_url=USPS_PRICES_URL,
                effective_date=today,
                db_path=db_path,
            )
            summary["proposed"].append({"id": rid, "box_size": size, "retail_cents": retail_cents})
        except Exception as e:
            summary["errors"].append(f"{size} propose failed: {e}")
    return summary
