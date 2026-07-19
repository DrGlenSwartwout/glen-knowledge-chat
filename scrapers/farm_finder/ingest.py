"""Farm ingest runner: crawl every registered source -> dedupe -> upsert.

Farms are stored as practitioners rows (tier='farm'); the mapping puts practice
markers in specialties[] and products/order_options in the new columns. The
existing practitioner_finder.db.run_upsert is column-generic, so it writes the
farm columns automatically once migrations/practitioners-farms.sql is applied.
Idempotent on source_url — safe to re-run (weekly cron via run_all.py).

Multi-source (Phase 3): all sources in scrapers.farm_finder.sources are crawled,
their rows concatenated, then dedupe_farms() collapses the same farm appearing in
more than one directory (keeping the earliest/most-trusted source's row) BEFORE
upsert, so one farm = one card.

Usage:
  python3 -m scrapers.farm_finder.ingest                        # DRY RUN — default
  python3 -m scrapers.farm_finder.ingest --limit 25             # dry-run a sample
  python3 -m scrapers.farm_finder.ingest --only foodforhumans   # one source
  python3 -m scrapers.farm_finder.ingest --apply                # WRITE to practitioners

--apply is gated deliberately: it performs a prod DB write and requires the
migration to be applied first. Dry run validates crawl + dedupe + mapping only.
"""
import argparse
import sys

from scrapers.farm_finder.dedupe import dedupe_farms
from scrapers.farm_finder.mapping import to_practitioner_row
from scrapers.farm_finder.sources import get_sources


def ingest(limit=None, sleep=0.5, apply=False, only=None, sources=None,
           log=print) -> dict:
    """Crawl all sources, dedupe, map, and (if apply) upsert farms.

    apply=False (default) builds and validates rows but writes nothing — no DB
    import happens, so it is safe anywhere. `sources` (a list of (name, scrape)
    tuples) overrides the registry, for tests; otherwise get_sources(only=only)
    selects the registered sources."""
    srcs = sources if sources is not None else get_sources(only=only)

    all_farms = []
    per_source: dict[str, int] = {}
    for name, scrape_fn in srcs:
        farms = scrape_fn(limit=limit, sleep=sleep)
        per_source[name] = len(farms)
        log(f"  {name}: scraped {len(farms)}")
        all_farms.extend(farms)

    deduped = dedupe_farms(all_farms, log=log)
    rows = [to_practitioner_row(f) for f in deduped]

    written = 0
    if apply:
        # Import lazily so a dry run never touches the DB layer.
        from scrapers.practitioner_finder.db import run_upsert
        for r in rows:
            run_upsert(r)
            written += 1

    summary = {
        "per_source": per_source,
        "scraped": len(all_farms),
        "deduped": len(deduped),
        "mapped": len(rows),
        "written": written,
        "applied": apply,
        "with_geo": sum(1 for r in rows if r.get("lat") is not None),
        "with_website": sum(1 for r in rows if r.get("website")),
    }
    log(f"farm ingest: {summary}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sleep", type=float, default=0.5)
    ap.add_argument("--only", default="",
                    help="Comma-separated source names to run "
                         "(default: all registered).")
    ap.add_argument("--apply", action="store_true",
                    help="WRITE to practitioners (prod DB). Default is dry run.")
    args = ap.parse_args()
    only = [s for s in args.only.split(",") if s.strip()] or None
    summary = ingest(limit=args.limit, sleep=args.sleep, apply=args.apply,
                     only=only)
    if not args.apply:
        print("DRY RUN — nothing written. Re-run with --apply to write "
              "(after applying migrations/practitioners-farms.sql).")
    sys.exit(0 if summary["mapped"] else 1)


if __name__ == "__main__":
    main()
