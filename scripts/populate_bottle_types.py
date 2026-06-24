"""Populate each storefront product's bottle_type from the FileMaker packaging
export + family rules. Re-runnable; never overwrites an existing assignment.
Dry-run by default; --write patches data/products.json (committed baseline)."""
from __future__ import annotations
import argparse, csv, json, os, re, sys

FMP_EXPORT = os.environ.get("FMP_PRODUCTS_CSV", "/tmp/fmp-export/newapp/products.csv")
_INFO_RE = re.compile(r'^(ei|ed|es|et|mb|mr)\s*\d', re.I)


def _norm(s):
    return re.sub(r'[^a-z0-9]+', ' ', (s or '').lower()).strip()


def family_rule(slug, product):
    name = product.get("name", "")
    src = product.get("source", "")
    if src == "infoceutical-catalog" or _INFO_RE.match(name.strip()):
        return "30ml"
    text = f"{name} {product.get('description','')}".lower()
    if "eye drop" in text or "eyedrop" in text:
        return "5ml"
    return None


def classify_from_fmp(row):
    disp = (row.get("zc_sold_display") or "").lower().replace(" ", "")
    meas = (row.get("sold_measurement") or "").lower().strip()
    ftype = (row.get("type") or "").strip()
    mml = re.match(r'^(\d+(?:\.\d+)?)ml$', disp)
    if mml or meas == "ml":
        ml = float(mml.group(1)) if mml else None
        return {5.0: "5ml", 15.0: "15ml", 50.0: "50ml", 100.0: "100ml"}.get(ml)  # 30/bulk -> None
    if any(x in disp for x in ("pullulan", "enteric", "vegicap", "gelcap", "capsule")) \
       or meas in ("pullulan", "enteric", "vegicaps", "gelcaps", "00 capsules"):
        mc = re.match(r'^(\d+)', disp)
        n = int(mc.group(1)) if mc else None
        if n is None:
            return None
        if n <= 40:
            return "30cap"
        if n <= 140:
            return "120cap"
        return None
    if disp.endswith("g") or meas == "g":
        return "120cap" if ftype == "Pure Powders" else "30g"
    return None


def build_assignments(products, fmp_by_name):
    assignments, review = {}, []
    for slug, p in products.items():
        if p.get("bottle_type"):
            continue
        key = family_rule(slug, p)
        if not key:
            row = fmp_by_name.get(_norm(p.get("name")))
            key = classify_from_fmp(row) if row else None
        if key:
            assignments[slug] = key
        else:
            review.append({"slug": slug, "name": p.get("name", ""),
                           "reason": "no family rule + no FMP packaging match"})
    return {"assignments": assignments, "review": review}


def _load_fmp(path):
    by_name = {}
    if not os.path.exists(path):
        return by_name
    for r in csv.DictReader(open(path)):
        by_name.setdefault(_norm(r.get("product_name")), r)
    return by_name


def _products_path():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = os.environ.get("DATA_DIR")
    if d and os.path.exists(os.path.join(d, "products.json")):
        return os.path.join(d, "products.json")
    return os.path.join(here, "data", "products.json")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)
    path = _products_path()
    with open(path) as f:
        doc = json.load(f)
    products = doc.get("products", {})
    fmp = _load_fmp(FMP_EXPORT)
    if not fmp:
        print(f"WARNING: no FMP export at {FMP_EXPORT} — only family rules will apply.")
    m = build_assignments(products, fmp)
    print(f"{len(m['assignments'])} products assigned; {len(m['review'])} need review.")
    for r in m["review"]:
        print(f"  REVIEW {r['slug']}: {r['name']!r} ({r['reason']})")
    if args.write:
        for slug, key in m["assignments"].items():
            products[slug]["bottle_type"] = key
        with open(path, "w") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
        print(f"Wrote {len(m['assignments'])} assignments to {path}")
    else:
        print("(dry run — pass --write to patch products.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
