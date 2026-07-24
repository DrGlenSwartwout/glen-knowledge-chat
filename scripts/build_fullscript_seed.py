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
"""
import json
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
               "focus_area_items": []},
              sys.stdout, indent=1, ensure_ascii=False)
    sys.stdout.write("\n")
    print(f"{len(products)} products across {len(FOCUS_AREAS)} focus areas",
          file=sys.stderr)


if __name__ == "__main__":
    main()
