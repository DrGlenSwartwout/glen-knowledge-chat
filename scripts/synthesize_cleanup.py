#!/usr/bin/env python3
"""Stage C-prep synthesis: merge the parallel LLM cleanup outputs
(data/cleanup-output/<slug>.json) with the Stage-B candidate
(data/products-enrich-candidate.json) into:
  - data/products-enrich-clean.json   (commit-ready: clean ingredients + description per slug)
  - data/products-stale-gk-clean.md   (trustworthy stale-GK list: real formula diffs only)
Read-only w.r.t. data/products.json. Re-runnable."""
import json
import glob
import os
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAND = os.path.join(ROOT, "data", "products-enrich-candidate.json")
OUTDIR = os.path.join(ROOT, "data", "cleanup-output")
CLEAN = os.path.join(ROOT, "data", "products-enrich-clean.json")
REPORT = os.path.join(ROOT, "data", "products-stale-gk-clean.md")


def main():
    cand = json.load(open(CAND))
    outs = {}
    for f in glob.glob(os.path.join(OUTDIR, "*.json")):
        try:
            r = json.load(open(f))
            outs[r.get("slug") or os.path.basename(f)[:-5]] = r
        except Exception as e:
            print("bad output:", f, e)

    clean, stale = {}, []
    for slug, r in outs.items():
        base = cand.get(slug, {})
        clean[slug] = {
            "name": r.get("name"),
            "ingredients_source": r.get("ingredients_source"),
            "ingredients": r.get("clean_ingredients") or [],
            "description": base.get("description", ""),
            "gk_stale": bool(r.get("gk_stale")),
            "stale_reason": r.get("stale_reason", ""),
            "added_on_gk": r.get("added_on_gk") or [],
            "missing_on_gk": r.get("missing_on_gk") or [],
        }
        if r.get("gk_stale"):
            stale.append((slug, r))

    json.dump(clean, open(CLEAN, "w"), indent=2)

    src = Counter(v["ingredients_source"] for v in clean.values())
    lines = [
        "# Products — Clean Stale-GrooveKart Report (LLM-normalized)", "",
        f"Generated from the parallel cleanup of {len(outs)} products. "
        "Synonym-aware: only REAL formula differences are flagged.", "",
        "## Counts", "", "| Metric | Count |", "|---|---|",
        f"| Products cleaned | {len(outs)} |",
        f"| **Genuinely stale GK pages** | **{len(stale)}** |", "",
        "Ingredient source: " + ", ".join(f"{k}={n}" for k, n in src.items()), "",
        "## Genuinely stale GK pages (real formula differences)", "",
        "`+ on GK` = on the page but not in the current formula. "
        "`- missing on GK` = in the current formula but absent from the page.", "",
        "| Slug | Reason | + on GK | - missing on GK |", "|---|---|---|---|",
    ]
    for slug, r in sorted(stale):
        add = ", ".join((r.get("added_on_gk") or [])[:8]) or "—"
        mis = ", ".join((r.get("missing_on_gk") or [])[:8]) or "—"
        reason = (r.get("stale_reason") or "").replace("|", "/")[:90]
        lines.append(f"| {slug} | {reason} | {add[:160]} | {mis[:160]} |")
    open(REPORT, "w").write("\n".join(lines))
    print(f"clean candidate: {len(clean)} products; genuinely stale: {len(stale)}")


if __name__ == "__main__":
    main()
