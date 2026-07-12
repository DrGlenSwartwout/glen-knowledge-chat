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

# Real remedymatch product pages introduce the related-products grid with an
# <h2>Related Products:</h2> heading; the grid itself (images + titles, each
# product linked twice) follows the heading as sibling markup, not nested
# inside a shared wrapper with the heading. So: find the heading, then scan
# everything from there to the end of the page for product-id links.
# Product-id links have the form
#   .../remedies/<category>/<id>-<slug>   or   .../resources/<id>-<slug>
# which excludes bare category/breadcrumb links like /remedies/syntropy/
# that lack the trailing <id>-<slug>.
_HEADING = re.compile(r"related products:", re.I)
_HREF = re.compile(
    r'href="(https?://[^"]*remedymatch\.com/(?:remedies/[^"/]+|resources)/\d+-[a-z0-9-]+)"',
    re.I,
)


def parse_related(html):
    html = html or ""
    heading = _HEADING.search(html)
    if not heading:
        return []
    tail = html[heading.start():]
    seen, out = set(), []
    for m in _HREF.finditer(tail):
        u = m.group(1)
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


# Storefront-slug -> catalog-slug overrides for hand-added device/book SKUs whose
# catalog slug differs from the storefront slug. Extend as the unmapped report shows.
ALIASES = {
    "healing-glaucoma-book": "book-healing-glaucoma",
    "healing-glaucoma-ebook": "book-healing-glaucoma",
    "denas-microcurrent-system-for-eye-healing": "denas-scenar",
    "denas-pcm-6-microcurrent": "denas-scenar",
    "living-water-ionizer-9-plate": "water-ionizer-9plate",
    "living-water-ionizer-5-plate": "water-ionizer-5plate",
    "living-water-ionizer-15-plate": "water-ionizer-15plate",
    "kloud-mini-pemf-mat": "kloud-pemf-mini",
    "kloud-maxi-pemf-mat": "kloud-pemf-maxi",
    "harmony-soft-laser-172-hz": "harmony-laser",
    "neuromagnesium": "neuro-magnesium",
    "free-easy": "free-and-easy",
    "cataract-solutions-book": "book-cataract-solutions",
    "cataract-solutions-ebook": "book-cataract-solutions",
    "macular-degeneration-macular-regeneration-book": "book-macular-regeneration",
    "macular-degeneration-macular-regeneration-ebook": "book-macular-regeneration",
    "electromagnetic-pollution-solutions-book": "book-emf-pollution-solutions",
    "electromagnetic-pollution-solutions-ebook": "book-emf-pollution-solutions",
    "refreshing-vision-book": "book-refreshing-vision",
    "refreshing-vision-ebook": "book-refreshing-vision",
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
        related, seen = [], set()
        for u in urls:
            mapped = map_storefront_slug(u, catalog_slugs, ALIASES)
            if mapped and mapped != slug:
                # Dedup by mapped slug: distinct storefront URLs (e.g. a book + its
                # ebook edition) can alias to the same catalog product.
                if mapped not in seen:
                    seen.add(mapped)
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
