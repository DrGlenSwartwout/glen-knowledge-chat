#!/usr/bin/env python3
"""Build data/fullscript_seed.json from Fullscript's PUBLIC open catalog.

Run BY HAND, never by the app. The app makes no runtime calls to Fullscript.

The endpoint backing fullscript.com/catalog needs no authentication, but it is
undocumented and only permits allowlisted operations (arbitrary GraphQL returns
HTTP 400). It expects `variables` as a JSON-ENCODED STRING, not an object.

Usage:
    python3 scripts/build_fullscript_seed.py > data/fullscript_seed.json

Then review the output by hand: `best_ff` mappings are guesses and must be
corrected by Glen before the seed is committed.

focus_area_items (the E4L scan-code -> focus-area map that drives the scan
matcher) is NOT fetched from Fullscript -- it is copied from data/prl_seed.json,
the sibling PRL channel's seed, which already carries the full E4L focus-area
taxonomy. See _load_focus_area_items().
"""
import json
import os
import sys
import time
import urllib.request

ENDPOINT = "https://fullscript.com/api/fs-graphql"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")

TYPEAHEAD = """
  query TypeaheadSearchV2_Shared_Query($query: String, $filters: SearchFilterObject) {
    viewer {
      typeaheadSearchV2(query: $query, filters: $filters, useSkuIdentifier: true) {
        entityType
        data { id name entityType brandName productSlug }
      }
    }
  }
"""

# focus_area_id -> (focus area name, [search terms], {product name: best Functional
# Formulations equivalent}). Mirrors the focus areas PRL already covers so the two
# channels reach parity. Extend as Glen confirms mappings.
FOCUS_AREAS = {
    9: ("Nervous System", ["magnesium taurate", "l-theanine"], {}),
}


def _tokens(term):
    """Meaningful (>=4 char) whitespace-split tokens of a search term, lowercased."""
    return [t for t in term.lower().split() if len(t) >= 4]


def _name_matches_term(name, term):
    """Keep a hit only if a meaningful token from the search term appears in the
    product name (case-insensitive substring). Fullscript's typeahead is fuzzy
    and surfaces clinically off-target hits (a homeopathic topical gel, a
    thyroid-support capsule) for nervous-system searches; this is a blunt but
    predictable backstop the owner can extend by hand."""
    toks = _tokens(term)
    if not toks:
        return False
    lname = name.lower()
    return any(t in lname for t in toks)


def _load_focus_area_items():
    """Copy focus_area_items (E4L scan item code -> focus area) from the sibling
    PRL channel's seed, which carries the same E4L focus-area taxonomy. Only
    copies entries for focus areas this generator's FOCUS_AREAS map covers, and
    only after confirming the id-to-name mapping agrees on both sides -- refuses
    (raises) rather than silently copying a mismatched mapping."""
    prl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "data", "prl_seed.json")
    with open(prl_path, "r", encoding="utf-8") as f:
        prl = json.load(f)
    prl_names = {}
    for fp in prl.get("focus_area_products", []):
        prl_names.setdefault(fp["focus_area_id"], set()).add(fp.get("focus_area_name"))
    items = []
    for fa_id, (fa_name, _terms, _ff_map) in sorted(FOCUS_AREAS.items()):
        prl_fa_names = prl_names.get(fa_id)
        if not prl_fa_names:
            raise SystemExit(
                f"focus_area_id {fa_id} ({fa_name!r}) not present in "
                f"{prl_path}; refusing to copy focus_area_items")
        if prl_fa_names != {fa_name}:
            raise SystemExit(
                f"focus_area_id {fa_id} name mismatch: this generator has "
                f"{fa_name!r}, prl_seed.json has {sorted(prl_fa_names)!r}; "
                "refusing to copy a mismatched focus_area_items mapping")
        for fi in prl.get("focus_area_items", []):
            if fi["focus_area_id"] == fa_id:
                items.append({"focus_area_id": fa_id, "item_code": fi["item_code"]})
    return items


def search(term):
    body = json.dumps({
        "query": TYPEAHEAD,
        "variables": json.dumps({
            "query": term,
            "filters": {"list": ["PRODUCTS", "BRANDS", "INGREDIENTS"]},
        }),
    }).encode()
    req = urllib.request.Request(
        ENDPOINT, data=body,
        headers={"Content-Type": "application/json", "User-Agent": UA,
                 "Origin": "https://fullscript.com",
                 "Referer": "https://fullscript.com/catalog"})
    with urllib.request.urlopen(req, timeout=30) as r:
        payload = json.load(r)
    if payload.get("errors"):
        raise SystemExit(f"catalog error for {term!r}: {payload['errors']}")
    groups = payload["data"]["viewer"]["typeaheadSearchV2"]
    for g in groups:
        if g.get("entityType") == "Product":
            return g.get("data") or []
    return []


def main():
    products, fa_products = {}, []
    for fa_id, (fa_name, terms, ff_map) in sorted(FOCUS_AREAS.items()):
        rank = 0
        for term in terms:
            for hit in search(term):
                name = hit.get("name")
                if not name or name in products:
                    continue
                if not _name_matches_term(name, term):
                    print(f"dropped {name!r} (brand={hit.get('brandName')!r}): "
                          f"no meaningful token from {term!r} found in name",
                          file=sys.stderr)
                    continue
                products[name] = {
                    "name": name,
                    "brand": hit.get("brandName"),
                    "external_id": hit.get("id"),
                    "product_slug": hit.get("productSlug"),
                    "url": None,
                    "focus_tags": [fa_name],
                    "product_type": "supplement",
                    "best_ff": ff_map.get(name),
                    "relation": "substitute" if ff_map.get(name) else None,
                    "ff_alts": [],
                    "source": "seed",
                    "active": 1,
                }
                fa_products.append({"focus_area_id": fa_id, "focus_area_name": fa_name,
                                    "fs_product_name": name, "rank": rank})
                rank += 1
            time.sleep(1.0)  # browsing volume, deliberately unhurried
    json.dump({"products": sorted(products.values(), key=lambda p: p["name"]),
               "focus_area_products": fa_products,
               "focus_area_items": _load_focus_area_items()},
              sys.stdout, indent=1, ensure_ascii=False)
    sys.stdout.write("\n")
    print(f"{len(products)} products across {len(FOCUS_AREAS)} focus areas",
          file=sys.stderr)


if __name__ == "__main__":
    main()
