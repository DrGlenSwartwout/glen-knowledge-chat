#!/usr/bin/env python3
"""FMP -> sellable catalog importer.

Adds ACTIVE, PRICED FMP remedy-type products that are missing from
data/products.json so they can be invoiced/added to orders. Dry-run prints a
preview + writes data/fmp-import-preview.json; --apply merges into products.json.

Decisions settled with Glen 2026-07-07
(docs/superpowers/specs/2026-07-07-fmp-catalog-import-design.md):
- Types: the remedy/consumable set below (Books, Services, equipment excluded).
- Charge price = FMP sold_price; retail_sug_price kept as a crossed-out regular anchor.
- qty_pricing (FF volume rate) = Functional Formulation ONLY; other types list-price.
- No bottle_type (Glen enters shipping data); imported items marked no_groovekart.
"""
import argparse
import json
import os
import re
import sqlite3
from collections import Counter

TYPE_WHITELIST = {
    "Essence", "Functional Formulation", "Tincture", "Pure Powders",
    "Homeopathic", "Gemmotherapy", "Spirit Mineral", "Simple Solution", "Infoceutical",
}
FF_VOLUME_TYPES = {"Functional Formulation"}
# Infoceuticals are all priced at a flat $39.97 (Glen), overriding FMP's per-row price.
INFOCEUTICAL_PRICE_CENTS = 3997
# FMP records the standard FF price as a round '70'; the real charge price is $69.97.
# app._invoice_line_view derives the $80 Value anchor by testing price == 6997 exactly,
# so an FF imported at $70.00 silently loses its anchor (Value collapses to Regular).
# Normalize only the round-$70 shorthand — CDS ($35), WholOmega 120 ($190) et al are
# genuinely different price points and pass through untouched.
FF_ROUND_PRICE_CENTS = 7000
FF_BASE_CENTS = 6997
_DESC_FIELDS = ("healing_qualities", "indications", "zc_dosage_display")


def _cents(v):
    """A FMP price string ('70', '70.00', '$40', '1,200') -> integer cents, or None.
    FMP is inconsistent: most types store a bare number but Infoceuticals store '$40'."""
    v = (v or "").strip().lstrip("$").replace(",", "").strip()
    if not v:
        return None
    try:
        return int(round(float(v) * 100))
    except (ValueError, TypeError):
        return None


def slugify(name):
    s = re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")
    return s or "item"


def _description(row):
    parts = [(row.get(k) or "").strip() for k in _DESC_FIELDS]
    return " ".join(p for p in parts if p)[:1200] or None


def clean_name(name):
    """FMP names carry a trailing '*' internal marker; drop it for a clean catalog
    name (and so 'D-Mannose Syntropy*' dedups against an existing 'D-Mannose Syntropy')."""
    return (name or "").strip().rstrip("*").strip()


def build_entry(row):
    """FMP row dict -> (slug, catalog_entry) or None when it can't be sold
    (no name or no numeric price)."""
    name = clean_name(row.get("product_name"))
    is_ff = row.get("type") in FF_VOLUME_TYPES
    if row.get("type") == "Infoceutical":
        price_cents = INFOCEUTICAL_PRICE_CENTS   # flat $39.97 for all infoceuticals
    else:
        price_cents = _cents(row.get("sold_price"))
        if is_ff and price_cents == FF_ROUND_PRICE_CENTS:
            price_cents = FF_BASE_CENTS          # FMP's '70' means $69.97 -> keeps the $80 Value
    if not name or not price_cents:   # skip no-name and $0 (comp/sample) — not sellable lines
        return None
    entry = {
        "name": name,
        "price_cents": price_cents,
        "pinecone_title": name,
        "qty_pricing": is_ff,
        "fmp_id": str(row.get("id_pk") or ""),
        "ingredients_source": "fmp_snap",
        "no_groovekart": True,
    }
    reg = _cents(row.get("retail_sug_price"))
    if reg and reg > price_cents:
        # The struck-through Value/SRP anchor the invoice prints above Regular
        # (app._invoice_line_view). Only meaningful ABOVE the charge price — FMP has a
        # few rows where retail_sug_price is below sold_price, which would be incoherent.
        entry["regular_cents"] = reg
    desc = _description(row)
    if desc:
        entry["description"] = desc
    return slugify(name), entry


def select_and_build(fmp_rows, existing):
    """existing = products.json 'products' dict (slug -> product).
    Returns (additions {slug: entry}, skipped [(name, reason)], collisions
    [(base_slug, final_slug, name)], by_type Counter)."""
    have_names = {(p.get("name") or "").strip().lower() for p in existing.values()}
    have_slugs = {s.lower() for s in existing.keys()}
    additions, skipped, collisions, by_type = {}, [], [], Counter()
    for row in fmp_rows:
        if (row.get("active") or "").strip().lower() != "yes":
            continue
        if (row.get("type") or "") not in TYPE_WHITELIST:
            continue
        raw_name = (row.get("product_name") or "")
        if re.search(r"\bstock\b", raw_name, re.I):
            skipped.append((raw_name.strip(), "bulk stock / production input, not a retail SKU"))
            continue
        if clean_name(raw_name).lower() in have_names:
            continue   # already sold under this name (asterisk-variant aware)
        built = build_entry(row)
        if not built:
            skipped.append(((row.get("product_name") or "").strip(), "no price"))
            continue
        slug, entry = built
        if slug in have_slugs:
            # slugifies onto an EXISTING catalog slug -> almost certainly the same
            # product under a name variant (e.g. "ACES Eye Drops" vs "ACES Eyedrops").
            # Skip and flag for review rather than mint a "-2" duplicate.
            skipped.append((entry["name"], f"slug '{slug}' already in catalog (possible name variant)"))
            continue
        base, n = slug, 2
        while slug in additions:            # de-collide only against OTHER new additions
            slug = f"{base}-{n}"
            n += 1
        if slug != base:
            collisions.append((base, slug, entry["name"]))
        additions[slug] = entry
        by_type[row.get("type")] += 1
    return additions, skipped, collisions, by_type


def load_fmp(db_path):
    cols = ("product_name,type,active,sold_price,retail_sug_price,id_pk,"
            "healing_qualities,indications,zc_dosage_display")
    cx = sqlite3.connect(db_path)
    cx.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in cx.execute(
            f"SELECT {cols} FROM fmp_snap_products "
            "WHERE TRIM(COALESCE(product_name,'')) <> ''")]
    finally:
        cx.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.path.expanduser("~/deploy-chat/chat_log.db"))
    ap.add_argument("--products", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "products.json"))
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    pj = json.load(open(a.products))
    existing = pj.get("products", {})
    fmp = load_fmp(a.db)
    additions, skipped, collisions, by_type = select_and_build(fmp, existing)

    print(f"FMP rows: {len(fmp)}  |  additions: {len(additions)}  |  "
          f"skipped(no price): {len(skipped)}  |  slug collisions: {len(collisions)}")
    print("by type:", dict(by_type))
    for slug, e in list(additions.items())[:12]:
        extra = (f" reg ${e['regular_cents']/100:.2f}" if "regular_cents" in e else "")
        extra += " [FF-vol]" if e["qty_pricing"] else ""
        print(f"  + {slug}: {e['name']}  ${e['price_cents']/100:.2f}{extra}")
    if collisions:
        print("first collisions:", collisions[:5])

    if a.apply:
        existing.update(additions)
        pj["products"] = existing
        with open(a.products, "w") as f:
            json.dump(pj, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"APPLIED: products.json now has {len(existing)} products (+{len(additions)}).")
    else:
        preview = os.path.join(os.path.dirname(a.products), "fmp-import-preview.json")
        with open(preview, "w") as f:
            json.dump({"additions": additions, "skipped": skipped[:50],
                       "skipped_count": len(skipped), "collisions": collisions,
                       "by_type": dict(by_type)}, f, indent=2, ensure_ascii=False)
        print(f"DRY RUN — wrote {preview}. Re-run with --apply to merge into products.json.")


if __name__ == "__main__":
    main()
