"""Practitioner progressive wholesale pricing engine (Phase 1, pure functions).

The per-bottle price is a BLENDED unit price for the whole order (the order
reprices as it grows), piecewise-linear in total order bottles ``q`` and
parameterized by the practitioner's certification floor ``F``:

    F  = 4000 - clamp(modules,0,12) * 125        # cents, $40 .. $25
    B  = bottles per full large flat-rate box    # from the box-fit matrix
    knots on the blended unit price (linear between):
        q = 1   -> 5000  ($50)
        q = B   -> (5000 + F) // 2
        q = 2B  -> F
        q >= 2B -> F     (floor held)
    order total = blended_unit_price(q) * q

No routes, no QBO, no Supabase, no writes. The only I/O is reading the box-fit
matrix (dashboard.shipping) and the product catalog (data/products.json);
both are injectable for tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Locked constants ──────────────────────────────────────────────────────────
KNOT_Q1_CENTS = 5000        # $50.00 single-bottle blended price
FLOOR_BASE_CENTS = 4000     # $40.00 uncertified floor
MODULE_STEP_CENTS = 125     # $1.25 per completed module
N_MODULES = 12
DEFAULT_B = 20              # fallback bottles/large box when the matrix is empty
DEFAULT_BOTTLE_TYPE = "default"
DEFAULT_RETAIL_CENTS = 7000  # $70 fallback retail anchor (margin reporting only)

_CATALOG_CACHE: Optional[Dict[str, dict]] = None


# ── Certification floor ───────────────────────────────────────────────────────

def certification_floor_cents(modules_completed: int) -> int:
    """F = 4000 - clamp(modules, 0, 12) * 125  ->  [2500, 4000] cents."""
    m = max(0, min(N_MODULES, int(modules_completed)))
    return FLOOR_BASE_CENTS - m * MODULE_STEP_CENTS


# ── Blended unit price ────────────────────────────────────────────────────────

def _lerp(qa: int, pa: int, qb: int, pb: int, q: int) -> int:
    """Linear-interpolate a price (cents) between knots (qa,pa) and (qb,pb),
    rounded to the nearest cent."""
    return pa + round((pb - pa) * (q - qa) / (qb - qa))


def blended_unit_price_cents(q: int, modules_completed: int, B: int) -> int:
    """Blended per-bottle price (cents) for an order of ``q`` total bottles."""
    if B < 2:
        B = 2
    F = certification_floor_cents(modules_completed)
    if q <= 1:
        return KNOT_Q1_CENTS
    if q >= 2 * B:
        return F
    Pb = (KNOT_Q1_CENTS + F) // 2
    if q <= B:
        return _lerp(1, KNOT_Q1_CENTS, B, Pb, q)
    return _lerp(B, Pb, 2 * B, F, q)


# ── Product catalog access ────────────────────────────────────────────────────

def _load_catalog() -> Dict[str, dict]:
    """Load the products map from data/products.json (cached)."""
    global _CATALOG_CACHE
    if _CATALOG_CACHE is None:
        path = Path(__file__).resolve().parent.parent / "data" / "products.json"
        data = json.loads(path.read_text())
        _CATALOG_CACHE = data.get("products", {})
    return _CATALOG_CACHE


def _product_pricing(slug: str, catalog: Optional[Dict[str, dict]] = None) -> dict:
    """Resolve pricing fields for a slug, with fallbacks. ``catalog`` (the
    products map) is injectable for tests; ``None`` reads products.json.

    Retail is the product's ``price_cents`` (reused as the retail anchor for
    margin reporting); ``cogs_cents``/``fulfillment_cents`` are optional and
    return ``None`` when unset (margin cannot be verified)."""
    cat = catalog if catalog is not None else _load_catalog()
    entry = cat.get(slug) or {}
    retail = entry.get("price_cents")
    if retail is None:
        retail = DEFAULT_RETAIL_CENTS
    return {
        "name": entry.get("name", slug),
        "qbo_item_id": entry.get("qbo_item_id"),
        "retail_cents": retail,
        "bottle_type": entry.get("bottle_type", DEFAULT_BOTTLE_TYPE),
        "cogs_cents": entry.get("cogs_cents"),
        "fulfillment_cents": entry.get("fulfillment_cents"),
    }


# ── Box size resolution ───────────────────────────────────────────────────────

def _resolve_B(
    items: List[dict],
    db_path: Optional[str] = None,
    catalog: Optional[Dict[str, dict]] = None,
) -> Tuple[int, List[str]]:
    """Resolve B (bottles per large box) from the box-fit matrix as the most
    conservative (minimum) large-box capacity across the order's bottle types.
    Falls back to DEFAULT_B with a loud warning when no capacity is known."""
    from dashboard.shipping import get_capacity_matrix

    matrix = {row["name"]: row for row in get_capacity_matrix(db_path)}
    capacities: List[int] = []
    warnings: List[str] = []
    for it in items:
        bt = _product_pricing(it.get("slug"), catalog)["bottle_type"]
        row = matrix.get(bt)
        L = row.get("L") if row else None
        if L:
            capacities.append(int(L))
        else:
            warnings.append(
                f"no large-box capacity for bottle_type {bt!r} "
                f"(slug {it.get('slug')!r})"
            )
    if capacities:
        return min(capacities), warnings
    warnings.append(f"box_capacity empty/unset; using fallback B={DEFAULT_B}")
    return DEFAULT_B, warnings


# ── Order quote ───────────────────────────────────────────────────────────────

def order_quote(
    items: List[dict],
    practitioner: dict,
    *,
    db_path: Optional[str] = None,
    catalog: Optional[Dict[str, dict]] = None,
) -> dict:
    """Price a whole wholesale order.

    items: [{"slug", "qty"}]; practitioner: {"modules_completed": int}.
    One blended unit price is computed for the whole order (q = sum of qty) and
    applied to every line. Margins (when COGS+fulfillment are known) gate
    ``margin_ok``; unknown costs emit a warning but do not fail."""
    modules = int(practitioner.get("modules_completed", 0) or 0)
    total_q = sum(int(it.get("qty", 0)) for it in items)
    B, warnings = _resolve_B(items, db_path=db_path, catalog=catalog)
    unit = blended_unit_price_cents(total_q, modules, B) if total_q >= 1 else 0

    lines: List[dict] = []
    margin_ok = True
    margin_warnings: List[str] = list(warnings)
    for it in items:
        slug = it.get("slug")
        qty = int(it.get("qty", 0))
        p = _product_pricing(slug, catalog)
        lines.append({
            "slug": slug,
            "name": p["name"],
            "qty": qty,
            "bottle_type": p["bottle_type"],
            "unit_price_cents": unit,
            "line_total_cents": unit * qty,
        })
        cogs, fulf = p["cogs_cents"], p["fulfillment_cents"]
        if cogs is None or fulf is None:
            margin_warnings.append(
                f"COGS/fulfillment not set for {slug!r}; margin unverified"
            )
        elif unit < cogs + fulf:
            margin_ok = False
            margin_warnings.append(
                f"margin floor breached for {slug!r}: "
                f"blended {unit} < cost {cogs + fulf}"
            )

    return {
        "floor_cents": certification_floor_cents(modules),
        "modules_completed": modules,
        "total_bottles": total_q,
        "B_effective": B,
        "blended_unit_price_cents": unit,
        "lines": lines,
        "subtotal_cents": sum(ln["line_total_cents"] for ln in lines),
        "margin_ok": margin_ok,
        "margin_warnings": margin_warnings,
    }
