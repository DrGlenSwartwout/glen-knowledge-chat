#!/usr/bin/env python3
"""Restore the rich code->remedy formulation map in e4l.db from the Pinecone
`e4l-protocols` namespace, whose e4l_item vectors carry each code's curated
`formulations` (comma-joined, priority order) in their metadata. The local
e4l_formulation_map is a stripped 22-row remnant; Pinecone was ingested from the full
map, so this rebuilds it (22 -> ~300 mappings across ~93 codes).

Idempotent (formulation_map.add_mapping skips existing pairs, appends new ones at the
bottom). Curation edits Glen makes afterward are preserved on re-run. Read-only on
Pinecone.

  doppler run -p remedy-match -c prd -- \
    python3 scripts/backfill_formulation_map_from_pinecone.py [--db PATH] [--dry]
"""
import argparse
import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dashboard import formulation_map as fm  # noqa: E402

INDEX = "remedy-match-llc"
NAMESPACE = "e4l-protocols"


def _item_mappings():
    """{code: [formulation names, priority order]} from every e4l_item vector."""
    skills = os.path.expanduser("~/AI-Training/02 Skills")
    if skills not in sys.path:
        sys.path.insert(0, skills)
    from lib import pinecone_client as pc  # noqa: E402
    idx = pc.index(INDEX)
    dim = idx.describe_index_stats()
    dim = (dim.get("dimension") if isinstance(dim, dict) else dim.dimension) or 1536
    res = idx.query(namespace=NAMESPACE, vector=[0.0] * dim, top_k=500, include_metadata=True)
    matches = res.get("matches") if isinstance(res, dict) else res.matches
    out = {}
    for m in matches or []:
        md = (m.get("metadata") if isinstance(m, dict) else m.metadata) or {}
        if md.get("chunk_type") != "e4l_item":
            continue
        code = (md.get("code") or "").strip()
        forms = [f.strip() for f in (md.get("formulations") or "").split(",") if f.strip()]
        if code and forms:
            out[code] = forms
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.path.expanduser("~/AI-Training/e4l.db"))
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    mappings = _item_mappings()
    cx = sqlite3.connect(a.db)
    fm.init_tables(cx)
    before = cx.execute("SELECT COUNT(*) FROM e4l_formulation_map").fetchone()[0]
    added = 0
    for code, forms in mappings.items():
        for name in forms:
            if a.dry:
                exists = cx.execute(
                    "SELECT 1 FROM e4l_formulation_map m JOIN formulations f ON f.id=m.formulation_id "
                    "WHERE m.item_code=? AND lower(f.name)=lower(?)", (code, name)).fetchone()
                if not exists:
                    added += 1
            else:
                n0 = len(fm.mappings_for(cx, code))
                fm.add_mapping(cx, code, name)
                if len(fm.mappings_for(cx, code)) > n0:
                    added += 1
    after = before + (added if not a.dry else 0)
    print(f"{'DRY: would add' if a.dry else 'added'} {added} mappings across {len(mappings)} codes "
          f"({before} -> {after if not a.dry else before} rows)")


if __name__ == "__main__":
    main()
