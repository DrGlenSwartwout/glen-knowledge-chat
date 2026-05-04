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


def pick_box(
    bottles_by_type: Dict[str, int],
    db_path: Optional[str] = None,
) -> Optional[str]:
    """Return the smallest S/M/L box that fits this order, or None.

    Algorithm — fractional-fill:
        For each candidate box size, compute total fill =
            sum(qty_in_order / capacity_in_that_box) over each bottle type.
        If total fill ≤ 1.0, the order fits in that box.
        Pick the smallest fitting box.

    Raises UnknownBottleType if a bottle is not in the catalog.
    Returns None if the order is empty or exceeds the Large box.
    """
    if not bottles_by_type:
        return None

    with _connect(db_path) as cx:
        capacities = _capacity_lookup(cx)

    for bottle_name in bottles_by_type:
        if bottle_name not in capacities:
            raise UnknownBottleType(bottle_name)

    for size in BOX_SIZES:
        total_fill = 0.0
        size_works = True
        for bottle_name, qty in bottles_by_type.items():
            cap = capacities[bottle_name].get(size)
            if cap is None or cap <= 0:
                size_works = False
                break
            total_fill += qty / cap
        if size_works and total_fill <= 1.0:
            return size
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

def quote(
    bottles_by_type: Dict[str, int],
    db_path: Optional[str] = None,
) -> dict:
    """Combine pick_box + current rate into a single payload for the UI."""
    box = pick_box(bottles_by_type, db_path=db_path)
    if box is None:
        return {
            "box_size": None,
            "shipping_cents": None,
            "error": (
                "Order is empty"
                if not bottles_by_type
                else "Order exceeds the Large flat-rate box — split shipment or use custom shipping."
            ),
        }
    rates = get_current_rates(db_path=db_path)
    if box not in rates:
        return {
            "box_size": box,
            "shipping_cents": None,
            "error": f"No confirmed USPS rate for {box} — check /admin/shipping.",
        }
    return {
        "box_size": box,
        "shipping_cents": rates[box]["charged_cents"],
        "rate_effective_date": rates[box]["effective_date"],
    }


# ── CRUD: bottle_types ────────────────────────────────────────────────────────

def add_bottle_type(
    name: str,
    notes: Optional[str] = None,
    db_path: Optional[str] = None,
) -> int:
    with _connect(db_path) as cx:
        cur = cx.execute(
            "INSERT INTO bottle_types (name, notes) VALUES (?, ?)",
            (name, notes),
        )
        cx.commit()
        return int(cur.lastrowid)


def list_bottle_types(db_path: Optional[str] = None) -> List[dict]:
    with _connect(db_path) as cx:
        rows = cx.execute(
            "SELECT id, name, notes, created_at FROM bottle_types ORDER BY name"
        ).fetchall()
        return [dict(r) for r in rows]


def delete_bottle_type(bottle_type_id: int, db_path: Optional[str] = None) -> None:
    with _connect(db_path) as cx:
        cx.execute("DELETE FROM bottle_types WHERE id = ?", (bottle_type_id,))
        cx.commit()


def update_bottle_type(
    bottle_type_id: int,
    name: str,
    notes: Optional[str] = None,
    db_path: Optional[str] = None,
) -> None:
    with _connect(db_path) as cx:
        cx.execute(
            "UPDATE bottle_types SET name = ?, notes = ? WHERE id = ?",
            (name, notes, bottle_type_id),
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


# ── USPS rate watcher ────────────────────────────────────────────────────────
# Scrapes a USPS retail-prices page and compares to current confirmed rates.
# Pure parsing is split from network fetch so it's unit-testable. Source URL
# can be overridden via USPS_PRICES_URL env var if the page structure shifts.

import re

USPS_PRICES_URL = os.environ.get(
    "USPS_PRICES_URL",
    "https://www.usps.com/business/prices.htm",
)

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
