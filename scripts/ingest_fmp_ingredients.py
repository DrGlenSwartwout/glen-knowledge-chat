#!/usr/bin/env python3
"""Ingest the new-FMP Ingredients tab into data/fmp-ingredient-content.json so the
catalog/enrichment can compute elemental mineral content + %RDA itself (rather than
hand-applying it). Source: the FMP newapp ingredients export.

Captures per ingredient: the standardization/elemental concentration, scientific
name, compound form, active flag, and the RDA-calculator fields."""
import csv
import json
import os
import re

csv.field_size_limit(10 ** 8)
SRC = os.path.expanduser("~/AI-Training/00 System/fmp-extracts/2026-05-23/ingredients.csv")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "fmp-ingredient-content.json")


def _norm(s):
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def _pct(row):
    # prefer explicit strength %, else parse "(NN% X)" from name_raw, else contentration
    if (row.get("strength_unit") or "").strip() == "%" and row.get("strength"):
        try:
            return float(row["strength"])
        except ValueError:
            pass
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", row.get("name_raw") or "")
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", row.get("contentration") or "")
    return float(m.group(1)) if m else None


def main():
    rows = list(csv.DictReader(open(SRC, encoding="utf-8", errors="ignore")))
    out = {}
    for r in rows:
        name = (r.get("name_common") or r.get("name_compound") or r.get("name_raw") or "").strip()
        if not name:
            continue
        key = _norm(name)
        if not key or key in out:
            continue
        out[key] = {
            "name": name,
            "compound": (r.get("name_compound") or "").strip() or None,
            "scientific": (r.get("name_scientific") or "").strip() or None,
            "label_form": (r.get("name_raw") or "").strip() or None,
            "percent": _pct(r),                       # elemental/standardization %
            "active": (r.get("active") or "").strip() or None,
            "rda_content": (r.get("zg_rda_calculator_content") or "").strip() or None,
            "rda_mg": (r.get("zg_rda_calculator_rda_mg") or "").strip() or None,
        }
    json.dump({"_source": "FMP newapp Ingredients tab (2026-05-23)", "ingredients": out},
              open(OUT, "w"), indent=2, ensure_ascii=False)
    withpct = sum(1 for v in out.values() if v["percent"] is not None)
    print(f"ingested {len(out)} FMP ingredients ({withpct} with an elemental/standardization %) -> {OUT}")


if __name__ == "__main__":
    main()
