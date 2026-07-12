#!/usr/bin/env python3
"""Backfill biofield_auth_chain.codes (per-layer stress codes) from the prod
begin-reveal drafts, so PRE-EXISTING authored tests get per-layer candidate lists.

New imports already carry codes (dashboard.biofield_reveal_import.import_layers_to_test
now passes L['codes']); this one-shot catches tests authored before that. Idempotent
-- re-running just rewrites the same codes. Matches an authored test to its reveal by
email (latest reveal wins) and maps reveal layer n -> chain rows at layer n.

  doppler run -p remedy-match -c prd -- \
    python3 scripts/backfill_layer_codes.py [--db PATH] [--dry]
"""
import argparse
import json
import os
import sqlite3
import urllib.request

REVEALS_URL = "https://illtowell.com/api/console/biofield-reveals"


def fetch_reveals():
    key = os.environ.get("CONSOLE_SECRET") or os.environ.get("CRON_SECRET") or ""
    req = urllib.request.Request(REVEALS_URL, headers={"X-Console-Key": key})
    d = json.load(urllib.request.urlopen(req, timeout=60))
    return [r for b in ("approved", "drafts", "pending") for r in (d.get(b) or [])]


def latest_by_email(reveals):
    out = {}
    for r in reveals:
        em = (r.get("email") or "").strip().lower()
        if not em:
            continue
        if em not in out or (r.get("created_at") or "") > (out[em].get("created_at") or ""):
            out[em] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.environ.get(
        "BIOFIELD_DB",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chat_log.db")))
    ap.add_argument("--dry", action="store_true", help="report only, write nothing")
    a = ap.parse_args()

    by_email = latest_by_email(fetch_reveals())
    cx = sqlite3.connect(a.db)
    try:
        cx.execute("ALTER TABLE biofield_auth_chain ADD COLUMN codes TEXT")
    except Exception:
        pass
    updated = tests = 0
    for tid, email in cx.execute("SELECT id, email FROM biofield_auth_tests").fetchall():
        rev = by_email.get((email or "").strip().lower())
        if not rev:
            continue
        hit = 0
        for L in rev.get("layers") or []:
            n, codes = L.get("n"), L.get("patterns") or []
            if n is None or not codes:
                continue
            if not a.dry:
                cur = cx.execute(
                    "UPDATE biofield_auth_chain SET codes=? WHERE test_id=? AND layer=?",
                    (json.dumps(codes), tid, n))
                hit += cur.rowcount
            else:
                hit += 1
        if hit:
            tests += 1
            updated += hit
    if not a.dry:
        cx.commit()
    print(f"{'DRY: would update' if a.dry else 'updated'} {updated} chain rows across {tests} tests")


if __name__ == "__main__":
    main()
