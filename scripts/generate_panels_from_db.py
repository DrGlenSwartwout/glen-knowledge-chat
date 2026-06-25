"""Generate customer-facing products.json ingredient panels FROM the DB formulation
recipes (single source of truth). One-way DB -> products.json. Completeness guard:
a recipe with a dosed-but-unnamed line is held for review, never overwrites a panel.
Dry-run default; --write. Supersedes scripts/refresh_ingredients_from_fmp.py."""
from __future__ import annotations
import argparse, json, os, sqlite3, sys


def _dose_str(dose, unit):
    if dose is None:
        return ""
    d = int(dose) if float(dose) == int(dose) else dose
    return f"{d} {unit}".strip() if unit else f"{d}"


def build_panel(items):
    out = []
    for it in items:
        name = (it.get("ingredient_canonical") or it.get("ingredient_name") or "").strip()
        has_dose = it.get("dose") is not None
        if not name:
            if has_dose:
                return None, "incomplete: dosed line with no ingredient name"
            continue  # nameless, doseless (packaging) — skip
        out.append({"name": name, "dose": _dose_str(it.get("dose"), it.get("dose_unit"))})
    if not out:
        return None, "no named ingredients"
    return out, None


def build_assignments(formulations_with_items, fmp_to_slug):
    panels, review = {}, []
    for f in formulations_with_items:
        slug = fmp_to_slug.get(str(f["fmp_id"]).strip())
        if not slug:
            review.append({"fmp_id": f["fmp_id"], "name": f["name"], "reason": "no matched product"})
            continue
        panel, reason = build_panel(f["items"])
        if panel is None:
            review.append({"slug": slug, "name": f["name"], "reason": reason})
        else:
            panels[slug] = panel
    return {"panels": panels, "review": review}


def _products_path():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = os.environ.get("DATA_DIR")
    if d and os.path.exists(os.path.join(d, "products.json")):
        return os.path.join(d, "products.json")
    return os.path.join(here, "data", "products.json")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--db", default=None)
    ap.add_argument("--source-date", default="2026-06-24")  # stamp; pass run date
    args = ap.parse_args(argv)
    from dashboard.ingredient_catalog import _default_db_path
    from dashboard.formulations import search_formulations, list_items_for_formulation
    path = _products_path()
    with open(path) as f:
        doc = json.load(f)
    products = doc.get("products", {})
    fmp_to_slug = {str(p["fmp_id"]).strip(): slug for slug, p in products.items() if p.get("fmp_id")}
    db = args.db or _default_db_path()
    forms = search_formulations("", limit=100000, db_path=db)
    fwi = [{"fmp_id": f["fmp_id"], "name": f["name"],
            "items": list_items_for_formulation(f["id"], db_path=db)} for f in forms]
    m = build_assignments(fwi, fmp_to_slug)
    print(f"{len(m['panels'])} panels to write; {len(m['review'])} in review")
    for r in m["review"][:40]:
        print(f"  REVIEW {r.get('slug') or r.get('fmp_id')}: {r['name']!r} ({r['reason']})")
    if args.write:
        for slug, panel in m["panels"].items():
            products[slug]["ingredients"] = panel
            products[slug]["ingredients_source"] = f"db-formulations-{args.source_date}"
        with open(path, "w") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
        print(f"wrote {len(m['panels'])} panels")
    else:
        print("(dry run — pass --write)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
