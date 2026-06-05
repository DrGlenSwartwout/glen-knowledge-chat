#!/usr/bin/env python3
"""Stage C apply: merge the reviewed clean enrichment into data/products.json.

ONLY ADDS fields per product (ingredients, ingredients_source, description,
gk_stale, gk_stale_reason). NEVER overwrites existing fields (name, price_cents,
qbo_item_id, pinecone_title) and never touches the 162 recipe-less products.

Excludes the slugs in HOLD (low-confidence matches + Glen-flagged wrong/unconfirmed
matches) so nothing uncertain lands in the canonical catalog."""
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN = os.path.join(ROOT, "data", "products-enrich-clean.json")
PRODUCTS = os.path.join(ROOT, "data", "products.json")
CORRECTIONS = os.path.join(ROOT, "data", "products-manual-corrections.json")

# Held out of the apply (flagged for Glen's review):
HOLD = {
    # low-confidence name matches
    "c15-syntropy-pentadecanoic-acid", "dry-eye-relief-program",
    "glucose-tolerance-program", "humic-acid", "macular-wellness-program",
    "serenity",
    # magnesium-taurate: a single-ingredient Pure Powder (Glen); the FMP
    # "Magnesium Glycinate" match is wrong. Held; its one ingredient is itself.
    "magnesium-taurate",
    # heart-health: "Rhythm Section/Restore" is a DIFFERENT product (Glen).
    "heart-health",
    # reverse-age (Reverse AGE) is confirmed a real product (Glen) -> applied.
}


def _clean_desc(text):
    if not text:
        return ""
    t = re.sub(r"\([^)]*remedies[^)]*\)", " ", text, flags=re.I)   # breadcrumb
    t = re.sub(r"(Retail|Your Price)\s*\$[\d.,]+", " ", t, flags=re.I)  # prices
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) > 600:
        t = t[:600].rsplit(" ", 1)[0] + "..."
    return t


def main():
    clean = json.load(open(CLEAN))
    doc = json.load(open(PRODUCTS))
    products = doc.get("products", {})

    applied, held, missing = [], [], []
    for slug, e in clean.items():
        if slug in HOLD:
            held.append(slug)
            continue
        if slug not in products:
            missing.append(slug)
            continue
        p = products[slug]
        ings = e.get("ingredients") or []
        if not ings:
            continue
        p["ingredients"] = ings                      # clean [{name, dose}]
        p["ingredients_source"] = e.get("ingredients_source")
        desc = _clean_desc(e.get("description"))
        if desc:
            p["description"] = desc
        if e.get("gk_stale"):
            p["gk_stale"] = True
            if e.get("stale_reason"):
                p["gk_stale_reason"] = e["stale_reason"]
        applied.append(slug)

    # Glen's manual corrections override the LLM output and the HOLD list.
    corrected = []
    if os.path.exists(CORRECTIONS):
        corr = json.load(open(CORRECTIONS))
        for slug, c in corr.items():
            if slug.startswith("_") or slug not in products:
                continue
            p = products[slug]
            if c.get("ingredients"):
                p["ingredients"] = c["ingredients"]
            p["ingredients_source"] = c.get("ingredients_source", "manual")
            if c.get("note"):
                p["enrichment_note"] = c["note"]
            # a Glen-verified formula is authoritative here -> not a stale-GK item
            p.pop("gk_stale", None)
            p.pop("gk_stale_reason", None)
            corrected.append(slug)

    doc["_enriched"] = "ingredients (FMP/Formulations/GK) + descriptions applied 2026-06-05; see products-stale-gk-clean.md"
    json.dump(doc, open(PRODUCTS, "w"), indent=2, ensure_ascii=False)

    print(f"applied: {len(applied)}  corrected: {len(corrected)} ({', '.join(sorted(corrected))})  "
          f"held: {len(held)}  missing-from-products.json: {len(missing)}")
    if held:
        print("HELD (flagged for review):", ", ".join(sorted(held)))
    if missing:
        print("MISSING (in clean candidate but not products.json):", ", ".join(sorted(missing)))


if __name__ == "__main__":
    main()
