"""Match storefront products (products.json) to FMP products by name, writing a
stable fmp_id onto each matched product. Reuses the bottle-type matcher. Idempotent
(never overwrites an existing fmp_id). Dry-run default; --write."""
from __future__ import annotations
import argparse, csv, difflib, json, os, sys
csv.field_size_limit(sys.maxsize)

from scripts.populate_bottle_types import _norm, _build_fmp_index, _SUFFIX_WORDS

FMP_PRODUCTS = os.environ.get("FMP_PRODUCTS_CSV", "/tmp/fmp-export/newapp/products.csv")


def match_products(products, fmp_by_name):
    keys = list(fmp_by_name.keys())
    matched, review = {}, []
    for slug, p in products.items():
        if p.get("fmp_id"):
            continue
        nm = _norm(p.get("name"))
        row = fmp_by_name.get(nm)
        if row is None:
            stripped = _SUFFIX_WORDS.sub("", nm).strip()
            row = fmp_by_name.get(stripped)
        if row is None:
            close = difflib.get_close_matches(nm, keys, n=1, cutoff=0.92)
            if close:
                row = fmp_by_name[close[0]]
        fid = (row or {}).get("id_pk")
        if fid:
            matched[slug] = str(fid).strip()
        else:
            review.append({"slug": slug, "name": p.get("name", "")})
    return {"matched": matched, "review": review}


def _products_path():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = os.environ.get("DATA_DIR")
    if d and os.path.exists(os.path.join(d, "products.json")):
        return os.path.join(d, "products.json")
    return os.path.join(here, "data", "products.json")


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)
    path = _products_path()
    with open(path) as f:
        doc = json.load(f)
    products = doc.get("products", {})
    with open(FMP_PRODUCTS, newline="") as f:
        fmp_by_name = _build_fmp_index(csv.DictReader(f))
    m = match_products(products, fmp_by_name)
    print(f"{len(m['matched'])} matched; {len(m['review'])} need review (of {len(products)})")
    for r in m["review"][:40]:
        print(f"  REVIEW {r['slug']}: {r['name']!r}")
    if args.write:
        for slug, fid in m["matched"].items():
            products[slug]["fmp_id"] = fid
        with open(path, "w") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
        print(f"wrote fmp_id to {len(m['matched'])} products")
    else:
        print("(dry run — pass --write)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
