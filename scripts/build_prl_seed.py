"""Assemble data/prl_seed.json from the vault PRL-catalog assets.
Run locally (Glen's machine) whenever the vault maps change; commit the output.
"""
import json, os

VAULT = os.path.expanduser("~/AI-Training/02 Products/PRL-catalog")
CATALOG = os.path.join(VAULT, "prl_catalog_enriched.json")
CROSSWALK = os.path.join(VAULT, "prl_ff_map_full.json")
FA_MAP = os.path.join(VAULT, "e4l-map", "e4l_prl_focus_area_map.json")
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "prl_seed.json")


def normalize_product_name(name, catalog_names):
    """Normalize product names from FA map to match catalog names.
    Handles trademark symbols, word order, and suffix variations.

    Note: Handles data quality issues where FA map references don't match
    exact catalog product names (e.g., missing trademark symbols, name order variations).
    """
    if name in catalog_names:
        return name

    # Hardcoded mappings for known missing products (data quality issues in source).
    # "Probiotic Caps" -> 60-softgel variant: CONFIRMED via E4L GetScanPatternsWithPRL
    # ExternalID 2690, which matches prlabs item #2690 = "Premier Probiotic (60 Softgels)"
    # (item #2680 is the 30-softgel version). Confirmed 2026-07-13.
    missing_mappings = {
        "Probiotic Caps": "Premier Probiotic (60 Softgels)",
        # E4L display names that differ from the prlabs catalog name for the SAME
        # product (confirmed by the product's URL slug):
        "Asta Complete-FX": "Astaxanthin Complex, Premier",   # asta-complete-fx.html
        "PQQ Complex with CoQ-10": "PQQ Complex, Premier",    # pqq-complex-with-coq10.html
        "Max B-ND": "Max B-ND™ (Fermented) (2 oz)",           # ™ + size in catalog name
        "Premier Magnesium": "Magnesium Glycinate Caps, Premier",  # PRL's magnesium SKU
    }
    if name in missing_mappings:
        mapped = missing_mappings[name]
        if mapped in catalog_names:
            return mapped

    # Try adding trademark symbols
    for sym in ['™', '®']:
        if name + sym in catalog_names:
            return name + sym
    # Try with ", Premier" suffix
    if name + ", Premier" in catalog_names:
        return name + ", Premier"
    # Try with "Premier " prefix
    if "Premier " + name in catalog_names:
        return "Premier " + name
    # Try swapping word order: "Premier X" -> "X, Premier"
    if name.startswith("Premier "):
        swapped = name[8:] + ", Premier"
        if swapped in catalog_names:
            return swapped
    # Try with fermentation marker
    if name + " (fermented)" in catalog_names:
        return name + " (fermented)"
    if name + " (Fermented)" in catalog_names:
        return name + " (Fermented)"
    # Try with trademark + fermentation
    for sym in ['™', '®']:
        if name + sym + " (fermented)" in catalog_names:
            return name + sym + " (fermented)"
        if name + sym + " (Fermented)" in catalog_names:
            return name + sym + " (Fermented)"
    # Try to find a variant by checking if there's a match without parenthetical info
    # Iterate sorted catalog_names for deterministic fallback; collect all base-name matches.
    base = name.split('(')[0].strip()
    candidates = []
    for cat_name in sorted(catalog_names):
        cat_base = cat_name.split('(')[0].strip()
        if base.lower() == cat_base.lower():
            candidates.append(cat_name)

    if candidates:
        if len(candidates) > 1:
            # Multiple base-name collisions found; warn and pick sorted-first deterministically.
            print(f"WARNING: Ambiguous fallback match for '{name}' -> {len(candidates)} catalog candidates: {candidates}")
        return candidates[0]

    # Return original; will fail validation in test if truly missing
    return name


def build():
    catalog = {p["name"]: p for p in json.load(open(CATALOG))["products"]}
    xwalk = {r["prl"]: r for r in json.load(open(CROSSWALK))["rows"]}
    fa = json.load(open(FA_MAP))["focus_areas"]
    catalog_names = set(catalog.keys())

    # Print all hardcoded assumptions at build time for auditability.
    print("Hardcoded name mappings applied:")
    print("  'Probiotic Caps' -> 'Premier Probiotic (60 Softgels)' [CONFIRMED: E4L ExternalID 2690 = prlabs item #2690]")

    products = []
    for name, p in catalog.items():
        x = xwalk.get(name, {})
        products.append({
            "name": name,
            "external_id": None,  # backfilled from mirror captures over time
            "url": p.get("url"),
            "focus_tags": p.get("focus_areas") or [],
            "product_type": p.get("product_type"),
            "best_ff": x.get("best_ff"),
            "relation": x.get("relation"),
            "ff_alts": x.get("alt_ffs") or [],
        })

    focus_area_products, focus_area_items = [], []
    stats = {"exact": 0, "normalized": 0, "fallback": 0}
    dropped = []  # E4L products with no catalog match (e.g. Noni, Lecithin) — skipped, not fatal
    for fid, v in fa.items():
        fid = int(fid)
        for i, prod in enumerate(v.get("prl_products") or []):
            pname = prod["name"] if isinstance(prod, dict) else prod
            normalized = normalize_product_name(pname, catalog_names)
            # Track resolution method for auditability
            if normalized == pname:
                stats["exact"] += 1
            elif pname in catalog_names:
                stats["exact"] += 1
            else:
                # If normalized != pname, it went through some normalization strategy or fallback
                base = pname.split('(')[0].strip()
                fallback_candidates = [cn for cn in catalog_names
                                      if base.lower() == cn.split('(')[0].strip().lower()]
                if fallback_candidates:
                    stats["fallback"] += 1
                else:
                    stats["normalized"] += 1
            if normalized not in catalog_names:
                # Truly unmapped (not in the 143-catalog, e.g. Premier Noni,
                # Lecithin Granules). Skip so the seed stays valid; the focus area
                # renders its remaining products. Tracked below for follow-up.
                dropped.append((v.get("name"), pname))
                continue
            focus_area_products.append({
                "focus_area_id": fid, "focus_area_name": v.get("name"),
                "prl_product_name": normalized, "rank": i})
        for it in (v.get("items") or []):
            code = it.get("code") if isinstance(it, dict) else it
            if code:
                focus_area_items.append({"focus_area_id": fid, "item_code": code})

    seed = {"products": products,
            "focus_area_products": focus_area_products,
            "focus_area_items": focus_area_items}
    with open(OUT, "w") as f:
        json.dump(seed, f, indent=1, ensure_ascii=False)
    print(f"Resolution: {stats['exact']} exact, {stats['normalized']} normalized, {stats['fallback']} fallback")
    if dropped:
        print(f"DROPPED {len(dropped)} unmapped E4L products (not in 143-catalog; focus area renders the rest, add to catalog later):")
        for fname, pname in dropped:
            print(f"  - [{fname}] {pname}")
    print(f"products={len(products)} fa_products={len(focus_area_products)} "
          f"fa_items={len(focus_area_items)} -> {OUT}")


if __name__ == "__main__":
    build()
