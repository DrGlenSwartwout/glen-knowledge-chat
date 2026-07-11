"""Scrape remedymatch "Related Products:" lists into data/related-harvested.json.
Run ad hoc:  python3 scripts/harvest_related_products.py
Politeness: 1 request/sec. Not on any request path."""
import json
import os
import re
import sys
import time
import urllib.request

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

# Grab the "Related Products:" block, then every href inside it.
_BLOCK = re.compile(r"Related Products:.*?(?=</section>|</div>\s*</div>|$)", re.I | re.S)
_HREF = re.compile(r'href="(https?://[^"]*remedymatch\.com/(?:remedies/[^"]+|resources/[^"]+))"', re.I)


def parse_related(html):
    block = _BLOCK.search(html or "")
    if not block:
        return []
    seen, out = set(), []
    for m in _HREF.finditer(block.group(0)):
        u = m.group(1)
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


# Storefront-slug -> catalog-slug overrides for hand-added device/book SKUs whose
# catalog slug differs from the storefront slug. Extend as the unmapped report shows.
ALIASES = {
    "healing-glaucoma-book": "book-healing-glaucoma",
    "denas-microcurrent-system-for-eye-healing": "denas-scenar",
    "living-water-ionizer-9-plate": "water-ionizer-9plate",
    "kloud-mini-pemf-mat": "kloud-pemf-mini",
}


def _fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "healing-oasis-harvest/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", "replace")


def main():
    from dashboard.related_products import map_storefront_slug
    products = json.load(open(os.path.join(_REPO, "data", "products.json")))["products"]
    catalog_slugs = set(products)
    out, unmapped = {}, []
    targets = [(s, v["url"]) for s, v in products.items()
               if v.get("url") and "remedymatch.com" in v["url"]]
    for i, (slug, url) in enumerate(targets):
        try:
            urls = parse_related(_fetch(url))
        except Exception as e:  # noqa: BLE001
            print(f"[harvest] {slug}: fetch failed: {e}", file=sys.stderr)
            continue
        related = []
        for u in urls:
            mapped = map_storefront_slug(u, catalog_slugs, ALIASES)
            if mapped and mapped != slug:
                related.append(mapped)
            elif not mapped:
                unmapped.append(u)
        if related:
            out[slug] = related
        print(f"[harvest] {i+1}/{len(targets)} {slug}: {len(related)} related", file=sys.stderr)
        time.sleep(1)
    with open(os.path.join(_REPO, "data", "related-harvested.json"), "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(f"[harvest] wrote {len(out)} products; {len(set(unmapped))} unmapped urls", file=sys.stderr)
    for u in sorted(set(unmapped)):
        print(f"[unmapped] {u}", file=sys.stderr)


if __name__ == "__main__":
    main()
