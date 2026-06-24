"""Infer each product's bottle_type from its name/description, for the geometric
shipping packer. Pure heuristics + a CLI that patches data/products.json
(only where bottle_type is unset). Re-runnable; never overwrites an existing
assignment. Produces a human-review list for low-confidence guesses."""
from __future__ import annotations
import argparse
import json
import os
import sys

TYPES = ("120cap", "100ml", "30roll", "50ml", "15ml", "5ml", "100cos", "30cap")
_REVIEW_THRESHOLD = 0.6


def _text(p):
    return f"{p.get('name','')} {p.get('description','')}".lower()


def infer_bottle_type(product: dict):
    t = _text(product)
    has = lambda *ws: any(w in t for w in ws)
    form = None
    if has("roll-on", "rollon", "roll on"):
        return ("30roll", 0.9 if has("30 ml", "30ml") else 0.7)
    if has("powder", "cosmetic", "30 g", "30g"):
        return ("100cos", 0.9 if has("powder", "cosmetic") else 0.7)
    if has("dropper"):
        form = True
        if has("100 ml", "100ml"):
            return ("100ml", 0.9)
        if has("50 ml", "50ml"):
            return ("50ml", 0.9)
        if has("15 ml", "15ml"):
            return ("15ml", 0.9)
        if has("5 ml", "5ml"):
            return ("5ml", 0.9)
        return ("default", 0.3)  # dropper but no recognizable size
    if has("capsule", "caps", "vcaps", "60 ct", "60ct", "30 capsules"):
        return ("30cap", 0.9 if has("100 ml", "100ml") else 0.7)
    if has("250 ml", "250ml", "wide-mouth", "wide mouth"):
        return ("120cap", 0.9 if has("250 ml", "250ml") else 0.7)
    return ("default", 0.3)


def build_mapping(products: dict) -> dict:
    assignments, review = {}, []
    for slug, p in products.items():
        if p.get("bottle_type"):
            continue  # never overwrite an existing assignment
        guess, conf = infer_bottle_type(p)
        final = guess if conf >= _REVIEW_THRESHOLD else "default"
        assignments[slug] = final
        if final == "default" or conf < _REVIEW_THRESHOLD:
            review.append({"slug": slug, "name": p.get("name", ""),
                           "guess": guess, "confidence": conf,
                           "reason": "low confidence" if conf < _REVIEW_THRESHOLD
                                     else "no size match"})
    return {"assignments": assignments, "review": review}


def _products_json_path():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = os.environ.get("DATA_DIR")
    if d and os.path.exists(os.path.join(d, "products.json")):
        return os.path.join(d, "products.json")
    return os.path.join(here, "data", "products.json")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="patch products.json (set bottle_type only where unset)")
    args = ap.parse_args(argv)
    path = _products_json_path()
    doc = json.load(open(path))
    products = doc.get("products", {})
    m = build_mapping(products)
    print(f"{len(m['assignments'])} products to assign; "
          f"{len(m['review'])} need review.")
    for r in m["review"]:
        print(f"  REVIEW {r['slug']}: {r['name']!r} -> {r['guess']} "
              f"(conf {r['confidence']})")
    if args.write:
        for slug, t in m["assignments"].items():
            products[slug]["bottle_type"] = t
        json.dump(doc, open(path, "w"), indent=2, ensure_ascii=False)
        print(f"Wrote {len(m['assignments'])} assignments to {path}")
    else:
        print("(dry run — pass --write to patch products.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
