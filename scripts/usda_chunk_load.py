"""One-off resumable loader for the USDA farm bulk load.

The full ingest (~14k sequential upserts) outruns a single command window, so
this loads ONE directory in a bounded [offset, offset+count) slice. The bulk
JSON is cached under /tmp so repeated chunks don't re-download ~19MB each time.

Usage:
  doppler run --project remedy-match --config prd -- \
    python3 scripts/usda_chunk_load.py <directory> <offset> <count>

Idempotent (upsert keys on source_url), so re-running a slice is safe.
"""
import json
import os
import sys
import time

from scrapers.farm_finder.usda import fetch_directory, parse_row
from scrapers.farm_finder.mapping import to_practitioner_row
from scrapers.practitioner_finder.db import run_upsert_many

CACHE_DIR = "/tmp/usda_cache"


def load_records(directory: str) -> list[dict]:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{directory}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    records = fetch_directory(directory)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(records, fh)
    return records


def main() -> int:
    directory, offset, count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    records = load_records(directory)
    total = len(records)
    chunk = records[offset:offset + count]

    t0 = time.time()
    batch = [to_practitioner_row(row)
             for row in (parse_row(rec, directory) for rec in chunk) if row]
    written = run_upsert_many(batch)
    elapsed = time.time() - t0

    end = offset + len(chunk)
    rate = written / elapsed if elapsed else 0
    print(f"{directory}: wrote {written} rows [{offset}:{end}] of {total} "
          f"in {elapsed:.1f}s ({rate:.1f}/s) "
          f"{'DONE' if end >= total else 'MORE'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
