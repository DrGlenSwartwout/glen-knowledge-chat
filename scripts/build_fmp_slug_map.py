#!/usr/bin/env python3
"""Build data/fmp_slug_map.json: FMP id_fk_product -> catalog slug review sheet.

Purchase-history ingestion (Task 3) needs `id_fk_product` resolved to a
catalog slug before FMP invoice history can be written into
`purchase_history`. This script auto-resolves what it safely can and
produces a review sheet for a human (Glen) to confirm the rest.

Resolution order, cheapest/most-trusted first:
  1. products.json `fmp_id`            -> resolved (already hand-verified
                                           in the catalog; see
                                           scripts/match_products_to_fmp.py)
  2. keyword match on the modal FMP
     description                       -> exclude (equipment/service/
                                           non-product line items: Zyto,
                                           tuning fork, toothbrush, ASH
                                           certification, Biofield Analysis,
                                           rebounder, scanner, shipping,
                                           courtesy, invoice notes, ...)
  3. name_to_slug() on the modal
     description, else an exact/
     substring hit against
     data/product-aliases.json's
     catalog_name                      -> suggestion, filed under `review`
                                           (NOT auto-trusted -- pricing
                                           correctness needs a human look)
  4. everything else                   -> review with suggestion: null

Usage:
    python3 scripts/build_fmp_slug_map.py [--db PATH] [--products PATH]
        [--aliases PATH] [--out PATH]

DB defaults to $DATA_DIR/chat_log.db (or ./chat_log.db); pass --db to point
at a real mirror, e.g. /Users/remedymatch/deploy-chat/chat_log.db.

PRIVACY: reads only id_fk_product / description columns from
fmp_invoice_items -- never touches invoices/clients, so no email/PII ever
lands in the output JSON.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from dashboard.product_sales import slug_map_from_products_json  # noqa: E402
from dashboard.practitioner_portal import name_to_slug  # noqa: E402

# Substring keywords (case-insensitive) on the modal FMP description that mark
# a line item as equipment/service/non-product -- never a repertoire slug.
_EXCLUDE_KEYWORDS = [
    "zyto", "tuning fork", "toothbrush", "certification", "biofield analysis",
    "rebounder", "courtesy", "scanner", "shipping -", "invoice #", "goggles",
    "helmet", "laser", "ionizer",
]

_PRESEED_EXCLUDE = {"952"}  # Courtesy (also caught by keyword, kept explicit
                            # per the task brief)


def _default_db_path() -> str:
    d = os.environ.get("DATA_DIR") or os.environ.get("LOG_DB")
    if d:
        p = Path(d)
        return str(p / "chat_log.db") if p.is_dir() else str(p)
    return str(_ROOT / "chat_log.db")


def _load_catalog(products_path: str) -> dict:
    with open(products_path) as f:
        return (json.load(f) or {}).get("products", {})


def _load_alias_slug_index(aliases_path: str, catalog: dict) -> dict:
    """clinical-name (lowercased) -> catalog slug, resolved via name_to_slug
    on the alias's catalog_name."""
    idx = {}
    try:
        with open(aliases_path) as f:
            aliases = (json.load(f) or {}).get("aliases", {})
    except Exception:
        return idx
    for clinical_name, meta in aliases.items():
        catalog_name = (meta or {}).get("catalog_name") or ""
        slug = name_to_slug(catalog_name, catalog)
        if slug:
            idx[clinical_name.strip().lower()] = slug
    return idx


def _alias_suggestion(description: str, alias_idx: dict):
    d = (description or "").strip().lower()
    if not d:
        return None
    if d in alias_idx:
        return alias_idx[d]
    for alias_name, slug in alias_idx.items():          # substring both ways,
        if len(alias_name) > 4 and (alias_name in d or d in alias_name):
            return slug                                  # same tolerance as
    return None                                          # name_to_slug()


def _is_excluded(description: str) -> bool:
    d = (description or "").strip().lower()
    return any(kw in d for kw in _EXCLUDE_KEYWORDS)


def _fetch_id_stats(db_path: str) -> dict:
    """distinct non-empty id_fk_product -> {"description": modal, "line_count": N}."""
    cx = sqlite3.connect(db_path)
    try:
        rows = cx.execute(
            "SELECT id_fk_product, description FROM fmp_invoice_items "
            "WHERE id_fk_product IS NOT NULL AND TRIM(id_fk_product) != ''"
        ).fetchall()
    finally:
        cx.close()
    desc_counts = defaultdict(Counter)
    line_counts = Counter()
    for pid, desc in rows:
        pid = str(pid).strip()
        line_counts[pid] += 1
        first_line = (desc or "").strip().split("\n")[0].strip()
        if first_line:
            desc_counts[pid][first_line] += 1
    out = {}
    for pid, cnt in line_counts.items():
        modal = desc_counts[pid].most_common(1)[0][0] if desc_counts[pid] else ""
        out[pid] = {"description": modal, "line_count": cnt}
    return out


def _sort_key(pid: str):
    return int(pid) if pid.isdigit() else pid


def build_map(db_path: str, products_path: str, aliases_path: str) -> dict:
    id_stats = _fetch_id_stats(db_path)
    # products.json declares fmp_id on products with zero FMP invoice history
    # too (e.g. catalog-only additions) -- scope `resolved` to ids that
    # actually appear in fmp_invoice_items so counts line up with the 396
    # distinct ids this map exists to classify.
    resolved_all = slug_map_from_products_json(products_path)
    resolved = {pid: slug for pid, slug in resolved_all.items() if pid in id_stats}
    catalog = _load_catalog(products_path)
    alias_idx = _load_alias_slug_index(aliases_path, catalog)

    review = {}
    exclude = set(_PRESEED_EXCLUDE)

    for pid, stats in id_stats.items():
        if pid in resolved:
            continue
        desc = stats["description"]
        if pid in exclude or _is_excluded(desc):
            exclude.add(pid)
            continue
        suggestion = name_to_slug(desc, catalog) or _alias_suggestion(desc, alias_idx)
        review[pid] = {
            "suggestion": suggestion,
            "description": desc,
            "line_count": stats["line_count"],
        }

    exclude_sorted = sorted(exclude, key=_sort_key)
    return {
        "resolved": dict(sorted(resolved.items(), key=lambda kv: _sort_key(kv[0]))),
        "review": dict(sorted(review.items(), key=lambda kv: _sort_key(kv[0]))),
        "exclude": [int(x) if x.isdigit() else x for x in exclude_sorted],
        "_generated_note": (
            "resolved = auto (products.json fmp_id, already hand-verified in the "
            "catalog -- see scripts/match_products_to_fmp.py); review = human must "
            "confirm/replace each 'suggestion' slug or move the id to exclude "
            "(suggestion may be null -- no auto match found, pick a slug or "
            "exclude); exclude = non-products (Courtesy, equipment, services) "
            "pre-seeded by keyword match on the FMP description -- spot-check "
            "before Task 3 consumes this map. Move confirmed review entries into "
            "resolved when done."
        ),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=_default_db_path(),
                     help="path to chat_log.db (has fmp_invoice_items)")
    ap.add_argument("--products", default=str(_ROOT / "data" / "products.json"))
    ap.add_argument("--aliases", default=str(_ROOT / "data" / "product-aliases.json"))
    ap.add_argument("--out", default=str(_ROOT / "data" / "fmp_slug_map.json"))
    args = ap.parse_args(argv)

    if not os.path.exists(args.db):
        print(f"ERROR: db not found: {args.db}", file=sys.stderr)
        return 1

    doc = build_map(args.db, args.products, args.aliases)

    with open(args.out, "w") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")

    n_resolved = len(doc["resolved"])
    n_review = len(doc["review"])
    n_exclude = len(doc["exclude"])
    n_suggested = sum(1 for v in doc["review"].values() if v["suggestion"])
    total = n_resolved + n_review + n_exclude
    all_products_fmp_ids = len(slug_map_from_products_json(args.products))
    print(f"resolved={n_resolved} review={n_review} "
          f"(suggested={n_suggested}, unsuggested={n_review - n_suggested}) "
          f"exclude={n_exclude} total={total}")
    print(f"(products.json declares {all_products_fmp_ids} fmp_id entries total; "
          f"{all_products_fmp_ids - n_resolved} have no FMP invoice history in "
          f"this db and are not in the map)")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
