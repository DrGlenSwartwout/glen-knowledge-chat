"""BOM demand + reorder shopping list — pure compute (Phase 3c-3)."""
import json
import math
import os
import sqlite3
from pathlib import Path
from typing import Optional

from dashboard.inventory import on_hand


def _default_db_path() -> str:
    base = os.environ.get("DATA_DIR", str(Path(__file__).resolve().parent.parent))
    return str(Path(base) / "chat_log.db")


def _connect(db_path: Optional[str] = None) -> sqlite3.Connection:
    cx = sqlite3.connect(db_path or _default_db_path())
    cx.row_factory = sqlite3.Row
    cx.execute("PRAGMA foreign_keys=ON")
    return cx


def _num(v):
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _round_up_order(shortfall, moq, unit_size) -> float:
    q = max(float(shortfall), float(moq) if moq else 0.0)
    us = _num(unit_size)
    if us and us > 0:
        q = math.ceil(q / us) * us
    return round(q, 6)


def bom_demand(plan, db_path=None) -> dict:
    out = {}
    if not plan:
        return out
    with _connect(db_path) as cx:
        for line in plan:
            fid = line.get("formulation_id")
            qty = _num(line.get("qty")) or 0.0
            if not fid or qty == 0:
                continue
            rows = cx.execute(
                "SELECT ingredient_id, dose, dose_unit FROM formulation_items WHERE formulation_id=?",
                (fid,)).fetchall()
            for r in rows:
                iid = r["ingredient_id"]
                dose = _num(r["dose"])
                if not iid or dose is None:
                    continue
                d = out.setdefault(iid, {"demand": 0.0, "unit": r["dose_unit"], "units_seen": []})
                d["demand"] += dose * qty
                if r["dose_unit"] and r["dose_unit"] not in d["units_seen"]:
                    d["units_seen"].append(r["dose_unit"])
    return out


def on_order_by_ingredient(db_path=None) -> dict:
    out = {}
    with _connect(db_path) as cx:
        rows = cx.execute("""
            SELECT pi.ingredient_id AS iid, pi.qty AS qty,
                   COALESCE((SELECT SUM(qty_received) FROM po_receiving r WHERE r.po_item_id = pi.id),0) AS received
            FROM po_items pi JOIN purchase_orders po ON po.id = pi.po_id
            WHERE po.status != 'closed' AND pi.ingredient_id IS NOT NULL
        """).fetchall()
    for r in rows:
        remaining = (_num(r["qty"]) or 0.0) - (_num(r["received"]) or 0.0)
        if remaining > 0:
            d = out.setdefault(r["iid"], {"on_order": 0.0})
            d["on_order"] += remaining
    return out


def reorder_report(plan=None, include_below_par=True, db_path=None) -> dict:
    demand = bom_demand(plan or [], db_path)
    onord = on_order_by_ingredient(db_path)
    with _connect(db_path) as cx:
        # candidate set: in demand, in on-order, or (if include_below_par) any ingredient with numeric par
        cand = set(demand) | set(onord)
        ing_rows = {r["id"]: r for r in cx.execute("SELECT id, name, extras, par_level, par_level_unit FROM ingredients").fetchall()}
        if include_below_par:
            for iid, r in ing_rows.items():
                if _num(r["par_level"]) is not None:
                    cand.add(iid)

        groups = {}
        for iid in cand:
            ing = ing_rows.get(iid)
            if not ing:
                continue
            par = _num(ing["par_level"]) or 0.0
            par_unit = ing["par_level_unit"]
            dem = demand.get(iid, {}).get("demand", 0.0)
            dem_unit = demand.get(iid, {}).get("unit")
            oo = onord.get(iid, {}).get("on_order", 0.0)
            oh = on_hand(iid, db_path)
            shortfall = par + dem - oh - oo
            if shortfall <= 0:
                continue
            # Pick the preferred source: explicit `preferred` flag first, then genuinely
            # cheapest by PER-BASE-UNIT cost (price_per_unit / unit_size), not raw pack price —
            # else a small cheap pack wrongly beats a cheaper-per-gram large pack. Nulls sort last.
            src = cx.execute("""
                SELECT s.supplier_id, s.supplier_name, sup.company AS company,
                       s.price_per_unit, s.unit_size, s.unit_type, s.minimum_order, s.minimum_order_unit
                FROM ingredient_sources s LEFT JOIN suppliers sup ON sup.id = s.supplier_id
                WHERE s.ingredient_id=?
                ORDER BY s.preferred DESC,
                         (CASE WHEN s.price_per_unit IS NOT NULL AND s.unit_size > 0
                               THEN s.price_per_unit * 1.0 / s.unit_size END) IS NULL,
                         (CASE WHEN s.price_per_unit IS NOT NULL AND s.unit_size > 0
                               THEN s.price_per_unit * 1.0 / s.unit_size END)
                LIMIT 1
            """, (iid,)).fetchone()
            price = _num(src["price_per_unit"]) if src else None
            unit_size = _num(src["unit_size"]) if src else None
            moq = src["minimum_order"] if src else None
            sugg = _round_up_order(shortfall, moq, unit_size)
            # price_per_unit is the price for ONE purchase of size `unit_size` (e.g. $140 for a
            # 1000 g bag), NOT a per-base-unit price. Cost = (#packs) * pack price. suggested_qty
            # is already rounded to a unit_size multiple, so packs is a whole number.
            packs = (sugg / unit_size) if (unit_size and unit_size > 0) else None
            est_cost = round(packs * price, 2) if (price is not None and packs is not None) else None
            units = [u for u in [par_unit if par else None, dem_unit if dem else None,
                                 (src["minimum_order_unit"] if src else None)] if u]
            unit_warning = len(set(units)) > 1
            sup_id = src["supplier_id"] if src else None
            sup_name = (src["company"] or src["supplier_name"]) if src else None
            key = sup_id if sup_id is not None else "—"
            g = groups.setdefault(key, {"supplier_id": sup_id,
                                        "supplier": sup_name or "— no preferred source —",
                                        "lines": [], "subtotal": 0.0})
            g["lines"].append({
                "ingredient_id": iid, "ingredient": ing["name"],
                "on_hand": round(oh, 4), "on_order": round(oo, 4), "par": par,
                "demand": round(dem, 4), "shortfall": round(shortfall, 4),
                "suggested_qty": sugg, "unit": (src["unit_type"] if src else None) or par_unit or dem_unit,
                "price_per_unit": price, "unit_size": unit_size,
                "packs": round(packs, 4) if packs is not None else None,
                "est_cost": est_cost, "unit_warning": unit_warning,
            })
            if est_cost is not None:
                g["subtotal"] = round(g["subtotal"] + est_cost, 2)

    group_list = sorted(groups.values(), key=lambda g: (g["supplier_id"] is None, (g["supplier"] or "").lower()))
    for g in group_list:
        g["lines"].sort(key=lambda l: (l["ingredient"] or "").lower())
    total_cost = round(sum(g["subtotal"] for g in group_list), 2)
    total_lines = sum(len(g["lines"]) for g in group_list)
    return {"groups": group_list, "totals": {"lines": total_lines, "est_cost": total_cost},
            "plan_echo": plan or []}


def _json_get(extras, key):
    if not extras:
        return None
    try:
        return json.loads(extras).get(key)
    except (ValueError, TypeError):
        return None
