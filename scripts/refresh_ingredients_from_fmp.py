"""Refresh each storefront product's ingredients list from FileMaker recipe data.

FMP is source of truth; products.json descriptions/ingredients are stale.
Dry-run by default; --write patches data/products.json.
"""
from __future__ import annotations
import argparse, csv, difflib, json, os, sys

# Re-use name-matching from the bottle-types populator (synergy/syntropy alias,
# suffix words).
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from populate_bottle_types import _norm, _SUFFIX_WORDS  # noqa: E402

FMP_PRODUCTS_CSV = os.environ.get(
    "FMP_PRODUCTS_CSV", "/tmp/fmp-export/newapp/products.csv"
)
FMP_ITEMS_CSV = os.environ.get(
    "FMP_ITEMS_CSV", "/tmp/fmp-export/newapp/products_items.csv"
)

_PACKAGING_WORDS = ("plantcaps", "capsule", "pullulan", "vegicap", "gelcap", "bottle")

# Packaging units whose blank-name lines are NOT incomplete-recipe signals.
_PACKAGING_UNITS = {"ea.", "ea"}


def _is_incomplete_signal(row: dict) -> bool:
    """Return True if this products_items row is an incomplete-recipe signal.

    A signal means: the ingredient name (text after the first " - " in
    zc_raw_display) is EMPTY **and** the line carries a real dose, i.e.
    EITHER zc_mg parses to > 0, OR unit_measurement is present and is NOT
    a packaging unit ("ea." / "ea").

    The "1ea. - " packaging/capsule lines — name empty, unit "ea.", mg 0 —
    are NOT signals; they are normal and already ignored by _parse_ingredient_line.
    """
    raw = row.get("zc_raw_display", "")
    if " - " not in raw:
        return False
    name = raw.split(" - ", 1)[1].strip()
    if name:
        # Named line — not a signal regardless of dose.
        return False

    # Name is blank — check whether it carries a real dose.
    zc_mg = row.get("zc_mg", "").strip()
    try:
        mg_val = float(zc_mg)
    except (ValueError, TypeError):
        mg_val = 0.0

    if mg_val > 0:
        return True

    unit = (row.get("unit_measurement") or "").strip()
    if unit and unit not in _PACKAGING_UNITS:
        return True

    return False


def _parse_ingredient_line(row: dict) -> dict | None:
    """Return {"name", "dose"} from a products_items row, or None to skip."""
    raw = row.get("zc_raw_display", "")
    if " - " not in raw:
        return None
    name = raw.split(" - ", 1)[1].strip()
    if not name:
        return None
    low = name.lower()
    if any(w in low for w in _PACKAGING_WORDS):
        return None

    # Dose: prefer zc_mg when it parses to a positive number.
    dose = ""
    zc_mg = row.get("zc_mg", "").strip()
    try:
        mg_val = float(zc_mg)
    except (ValueError, TypeError):
        mg_val = 0.0
    if mg_val > 0:
        # Drop trailing ".0" but preserve ".7", "1.3", etc.
        if mg_val == int(mg_val):
            dose = f"{int(mg_val)} mg"
        else:
            # Use the original string to avoid float repr issues (.7 stays .7)
            dose = f"{zc_mg} mg"
    else:
        qty = (row.get("qty") or "").strip()
        unit = (row.get("unit_measurement") or "").strip()
        if qty and unit:
            dose = f"{qty} {unit}".strip()

    return {"name": name, "dose": dose}


def _build_fmp_index(rows) -> dict[str, dict]:
    """Build normalised-name → FMP row dict (full key wins on collision)."""
    by_name: dict[str, dict] = {}
    for r in rows:
        full_key = _norm(r.get("product_name", ""))
        by_name.setdefault(full_key, r)
        stripped_key = _SUFFIX_WORDS.sub("", full_key)
        if stripped_key != full_key:
            by_name.setdefault(stripped_key, r)
    return by_name


def _load_fmp_items(path: str) -> dict[str, list[dict]]:
    """Return {id_pk: [row, ...]} grouped by id_fk_product."""
    items: dict[str, list[dict]] = {}
    if not os.path.exists(path):
        return items
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pk = row.get("id_fk_product", "").strip()
            if pk:
                items.setdefault(pk, []).append(row)
    return items


def _resolve_updates(
    products: dict,
    fmp_index: dict[str, dict],
    fmp_items: dict[str, list[dict]],
) -> tuple[dict[str, list], list[dict]]:
    """Return (staged, review).

    staged: {slug: [ingredient dicts]} — only products that have ≥1 ingredient
    review: list of {slug, name, reason} — no match or fuzzy match
    """
    staged: dict[str, list] = {}
    review: list[dict] = []
    fmp_keys = list(fmp_index.keys())

    for slug, p in products.items():
        norm_name = _norm(p.get("name", ""))

        # Match resolution: exact → suffix-strip → fuzzy
        row = fmp_index.get(norm_name)
        match_method = "exact" if row is not None else None

        if row is None:
            stripped = _SUFFIX_WORDS.sub("", norm_name)
            if stripped != norm_name:
                row = fmp_index.get(stripped)
                if row is not None:
                    match_method = "suffix-strip"

        if row is None:
            matches = difflib.get_close_matches(norm_name, fmp_keys, n=1, cutoff=0.92)
            if matches:
                row = fmp_index[matches[0]]
                match_method = "fuzzy"

        if row is None:
            review.append({"slug": slug, "name": p.get("name", ""), "reason": "no FMP match"})
            continue

        fmp_pk = str(row.get("id_pk", "")).strip()
        item_rows = fmp_items.get(fmp_pk, [])

        # --- Recipe-completeness guard ---
        # If ANY item row is an incomplete-recipe signal (dosed line with
        # blank ingredient name, not a packaging "ea." line), route to review
        # and leave existing ingredients untouched.
        if any(_is_incomplete_signal(r) for r in item_rows):
            review.append({
                "slug": slug,
                "name": p.get("name", ""),
                "reason": "FMP recipe incomplete (dosed line with no ingredient name)",
                "match_method": match_method,
            })
            continue

        ingredients = [ing for r in item_rows if (ing := _parse_ingredient_line(r)) is not None]

        if not ingredients:
            review.append({
                "slug": slug,
                "name": p.get("name", ""),
                "reason": "no FMP recipe",
                "match_method": match_method,
            })
            continue

        if match_method == "fuzzy":
            review.append({
                "slug": slug,
                "name": p.get("name", ""),
                "reason": "fuzzy match — verify",
                "fmp_name": row.get("product_name", ""),
                "ingredients": ingredients,
            })
            continue

        staged[slug] = ingredients

    return staged, review


def _products_path() -> str:
    root = os.path.dirname(_HERE)
    d = os.environ.get("DATA_DIR")
    if d and os.path.exists(os.path.join(d, "products.json")):
        return os.path.join(d, "products.json")
    return os.path.join(root, "data", "products.json")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true", help="Patch products.json")
    args = ap.parse_args(argv)

    path = _products_path()
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    products = doc.get("products", {})

    if not os.path.exists(FMP_PRODUCTS_CSV):
        print(f"ERROR: FMP products CSV not found: {FMP_PRODUCTS_CSV}", file=sys.stderr)
        return 1

    with open(FMP_PRODUCTS_CSV, newline="", encoding="utf-8") as f:
        fmp_index = _build_fmp_index(csv.DictReader(f))

    fmp_items = _load_fmp_items(FMP_ITEMS_CSV)

    staged, review = _resolve_updates(products, fmp_index, fmp_items)

    print(f"\n=== DRY RUN: {len(staged)} products would be refreshed ===\n")
    for slug, ingr in sorted(staged.items()):
        old_n = len(products[slug].get("ingredients") or [])
        print(f"  {slug} ({products[slug].get('name','')}): {old_n} -> {len(ingr)} ingredients")

    print(f"\n=== REVIEW ({len(review)} items — verify before writing) ===\n")
    for r in review:
        fmp_hint = f" [FMP: {r.get('fmp_name','')}]" if r.get("fmp_name") else ""
        print(f"  {r['slug']}: {r['name']!r} — {r['reason']}{fmp_hint}")

    if args.write:
        for slug, ingr in staged.items():
            products[slug]["ingredients"] = ingr
            products[slug]["ingredients_source"] = "fmp-products-items-2026-06-23"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
        print(f"\nWrote {len(staged)} updates to {path}")
    else:
        print("\n(dry run — pass --write to patch products.json)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
