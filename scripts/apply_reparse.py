#!/usr/bin/env python3
"""Apply the re-parse re-judge (data/redif-output) to products.json + regenerate
a TRUSTWORTHY stale-GK report. Distinguishes:
  - GK page genuinely MISSING current ingredients -> gk_stale (real page update).
  - GK page has MORE than our FMP/Formulations record -> our record is partial
    (gk_has_extra note), NOT a stale page.
The 4 products with no re-parsed green-plus panel keep their flag (un-reverified)."""
import json
import os
import glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRODUCTS = os.path.join(ROOT, "data", "products.json")
OUTDIR = os.path.join(ROOT, "data", "redif-output")
INDIR = os.path.join(ROOT, "data", "redif-input")
REPORT = os.path.join(ROOT, "data", "products-stale-gk-clean.md")


def main():
    doc = json.load(open(PRODUCTS))
    products = doc.get("products", {})
    judged = {}
    for f in glob.glob(os.path.join(OUTDIR, "*.json")):
        r = json.load(open(f))
        judged[r.get("slug")] = r

    rejudged_slugs = {os.path.basename(f)[:-5] for f in glob.glob(os.path.join(INDIR, "*.json"))}
    # the originally-flagged set that did NOT get a re-parsed panel (un-reverified)
    unverified = [s for s, p in products.items() if p.get("gk_stale") and s not in rejudged_slugs]

    # Glen-confirmed stale overrides (his explicit knowledge wins). neuromagnesium
    # was here, but Glen has since updated its GK page, so the re-judge's clear stands.
    GLEN_STALE = {}

    page_missing, fmp_partial, cleared = [], [], []
    for slug, r in judged.items():
        p = products.get(slug)
        if not p:
            continue
        missing = r.get("missing_on_gk") or []
        added = r.get("added_on_gk") or []
        if slug in GLEN_STALE:
            p["gk_stale"] = True
            p["gk_stale_reason"] = GLEN_STALE[slug]
            page_missing.append((slug, p.get("name"), missing or ["(Glen-confirmed)"], added))
            if added:
                p["gk_has_extra"] = added[:12]
                fmp_partial.append((slug, p.get("name"), added))
            continue
        if missing:
            p["gk_stale"] = True
            p["gk_stale_reason"] = "GK page missing: " + ", ".join(missing[:10])
            page_missing.append((slug, p.get("name"), missing, added))
        else:
            p.pop("gk_stale", None)
            p.pop("gk_stale_reason", None)
            cleared.append((slug, p.get("name")))
        if added:
            p["gk_has_extra"] = added[:12]
            fmp_partial.append((slug, p.get("name"), added))

    json.dump(doc, open(PRODUCTS, "w"), indent=2, ensure_ascii=False)

    L = ["# Products — Stale GrooveKart Report (re-parsed, trustworthy)", "",
         "Built from the full re-parsed GK ingredient panels (the original scrape dropped them) "
         "+ a synonym-aware re-judge. A page is STALE only when it genuinely OMITS current "
         "ingredients; pages where GK lists MORE than our record mean our FMP/Formulations record "
         "is partial (not a page problem).", "",
         "## Counts", "", "| Metric | Count |", "|---|---|",
         f"| GK pages missing current ingredients (real updates) | {len(page_missing)} |",
         f"| Pages cleared (GK matches the formula) | {len(cleared)} |",
         f"| Products where GK is richer than our record (FMP partial) | {len(fmp_partial)} |",
         f"| Originally-flagged but no re-parsed panel (un-reverified) | {len(unverified)} |", "",
         "## (A) GK pages to update — genuinely missing current ingredients", "",
         "| Slug | Add to the GK page (missing) |", "|---|---|"]
    for slug, name, missing, added in sorted(page_missing):
        L.append(f"| {slug} | {', '.join(missing[:10])} |")
    L += ["", "## (B) Our FMP/Formulations record is partial — GK lists more (data follow-up, not a page fix)", "",
          "| Slug | On GK but not in our record |", "|---|---|"]
    for slug, name, added in sorted(fmp_partial):
        L.append(f"| {slug} | {', '.join(added[:12])} |")
    L += ["", f"## (C) Un-reverified (no green-plus panel re-parsed): {', '.join(sorted(unverified)) or 'none'}",
          "", f"## (D) Cleared: {', '.join(s for s, _ in sorted(cleared)) or 'none'}", ""]
    open(REPORT, "w").write("\n".join(L))

    print(f"page-missing (stale): {len(page_missing)}  cleared: {len(cleared)}  "
          f"fmp-partial: {len(fmp_partial)}  unverified: {len(unverified)}")


if __name__ == "__main__":
    main()
