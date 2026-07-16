"""Materialize rule-computed one-time prices into data/products.json.

For every product with price_rule == "components_less_10pct", set
price_cents = round(0.9 * sum(component price_cents * qty)). Idempotent.

Usage:
  python scripts/compute_bundle_prices.py          # write
  python scripts/compute_bundle_prices.py --check   # exit 1 if any drift
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashboard.bundle_pricing import compute_bundle_price_cents

PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "products.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report drift, do not write")
    args = ap.parse_args()

    with open(PATH) as f:
        data = json.load(f)
    products = data["products"]

    drift, changes = [], []
    for slug, p in products.items():
        if p.get("price_rule") != "components_less_10pct":
            continue
        want = compute_bundle_price_cents(p, products)
        have = p.get("price_cents")
        if have != want:
            drift.append((slug, have, want))
            p["price_cents"] = want
            changes.append(slug)

    if args.check:
        for slug, have, want in drift:
            print(f"DRIFT {slug}: {have} -> {want}")
        sys.exit(1 if drift else 0)

    if changes:
        with open(PATH, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        for slug, have, want in drift:
            print(f"set {slug}: {have} -> {want} (${want/100:.2f})")
    else:
        print("no changes; all bundle prices already match the rule")


if __name__ == "__main__":
    main()
