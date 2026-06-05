#!/usr/bin/env python3
"""Resolve the gk_has_extra (item 1c) per Glen's rule: when a product's
authoritative source is the NEW FMP (the newer, fewer-ingredient formula), GK's
extra ingredients are superseded old ones -> ACCEPT the gap (clear gk_has_extra,
do NOT merge). Keep gk_has_extra only where the authoritative source is the older
Formulations DB (t33) or GK-only, which stay open for review.

Operates directly on products.json + regenerates the stale report. Re-runnable."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRODUCTS = os.path.join(ROOT, "data", "products.json")
REPORT = os.path.join(ROOT, "data", "products-stale-gk-clean.md")


def main():
    doc = json.load(open(PRODUCTS))
    products = doc.get("products", {})

    accepted, kept = [], []
    for slug, p in products.items():
        if not p.get("gk_has_extra"):
            continue
        if p.get("ingredients_source") == "fmp_new":
            # newer FMP formula is authoritative; GK extras are superseded -> accept
            p.pop("gk_has_extra", None)
            p["gk_extra_accepted"] = True
            accepted.append(slug)
        else:
            kept.append((slug, p.get("name"), p.get("gk_has_extra")))

    stale = [(s, p.get("gk_stale_reason", "")) for s, p in products.items() if p.get("gk_stale")]

    json.dump(doc, open(PRODUCTS, "w"), indent=2, ensure_ascii=False)

    L = ["# Products — Stale GrooveKart Report (re-parsed, trustworthy)", "",
         "Stale = GK page genuinely OMITS current ingredients. The 'GK richer than our record' "
         "cases were resolved per Glen's rule: when the authoritative source is the NEW FMP "
         "(newer, fewer-ingredient formula), GK's extras are superseded old ingredients -> gap "
         "ACCEPTED (gk_extra_accepted). Only older-Formulations-DB cases stay open for review.", "",
         "## Counts", "", "| Metric | Count |", "|---|---|",
         f"| GK pages to update (missing current ingredients) | {len(stale)} |",
         f"| gk_has_extra ACCEPTED (new-FMP newer formula wins) | {len(accepted)} |",
         f"| gk_has_extra KEPT for review (older Formulations DB) | {len(kept)} |", "",
         "## (A) GK pages to update — missing current ingredients", "",
         "| Slug | Add to the GK page |", "|---|---|"]
    for slug, reason in sorted(stale):
        L.append(f"| {slug} | {reason.replace('GK page missing: ', '').replace('|','/')[:160]} |")
    L += ["", "## (B) Still open — our older-Formulations-DB record may be partial vs GK (review)", "",
          "| Slug | On GK but not in our record |", "|---|---|"]
    for slug, name, extra in sorted(kept):
        L.append(f"| {slug} | {', '.join(extra[:12])} |")
    L += ["", f"## (C) gk_has_extra accepted (new-FMP, no action): {len(accepted)} products", ""]
    open(REPORT, "w").write("\n".join(L))

    print(f"accepted (cleared): {len(accepted)}  kept for review: {len(kept)}  pages-to-update: {len(stale)}")
    print("kept:", ", ".join(s for s, _, _ in sorted(kept)))


if __name__ == "__main__":
    main()
