"""Practitioner Finder weekly orchestrator.

Runs all registered adapters sequentially. Per-adapter failure is logged and
emailed to Glen but does NOT block sibling adapters. After all adapters complete,
runs a global geocode mop-up sweep, then triggers GHL prospect sync (real-run
only when --apply-ghl is passed).

Invoke:
  doppler run --project remedy-match --config prd -- \
    python3 -m scrapers.practitioner_finder.run_all [--apply-ghl] [--only oepf,iaomt]

Scheduled weekly Sunday 11pm HST on the upper Studio writer Mac via launchd
(com.remedymatch.practitioner-finder.plist).
"""
import argparse
import sys
import traceback
from datetime import datetime, timezone
from typing import Callable, Optional

from scrapers.practitioner_finder.db import (
    list_ungeocoded,
    update_geocode,
)
from scrapers.practitioner_finder.geocode import geocode_row, MapboxError
from scrapers.practitioner_finder.models import NormalizedPractitionerRow
from db_supabase import supabase_cursor


# ---------------------------------------------------------------------------
# Adapter registry — populated as adapters land in waves A–D.
# Each entry: (adapter_name, callable that runs the adapter's scrape + upsert
# and returns (rows_scraped, rows_inserted, rows_updated)). The wrappers SKIP
# inline geocoding — the orchestrator does one global sweep at the end via
# _global_geocode_sweep() to avoid double-paying Mapbox quota.
#
# rows_inserted / rows_updated cannot be cheaply distinguished given the
# ON CONFLICT (source_url) DO UPDATE upsert. We report scraped count in
# both fields; refinement is a future enhancement.
# ---------------------------------------------------------------------------
from scrapers.practitioner_finder.db import run_upsert


def _run_oepf_scrape() -> tuple[int, int, int]:
    from scrapers.practitioner_finder.oepf import (
        fetch_listing_index, fetch_directory_listing_html,
        parse_directory_listing_html,
    )
    index = fetch_listing_index()
    rows_total = 0
    for entry in index:
        url = entry.get("link")
        if not url:
            continue
        try:
            html = fetch_directory_listing_html(url)
        except Exception as e:
            print(f"  WARN: fetch failed for {url}: {e}", file=sys.stderr)
            continue
        for row in parse_directory_listing_html(html, source_url=url):
            run_upsert(row.to_dict())
            rows_total += 1
    return rows_total, rows_total, 0


def _run_iaomt_scrape() -> tuple[int, int, int]:
    from scrapers.practitioner_finder.iaomt import (
        fetch_all_directory_records, parse_directory_json,
    )
    records = fetch_all_directory_records()
    rows = parse_directory_json(records)
    for row in rows:
        run_upsert(row.to_dict())
    return len(rows), len(rows), 0


def _run_iabdm_scrape() -> tuple[int, int, int]:
    from scrapers.practitioner_finder.iabdm import (
        fetch_all_directory_records, parse_directory_json,
    )
    records = fetch_all_directory_records()
    rows = parse_directory_json(records)
    for row in rows:
        run_upsert(row.to_dict())
    return len(rows), len(rows), 0


def _run_owns_scrape() -> tuple[int, int, int]:
    from scrapers.practitioner_finder.owns import (
        build_category_country_map, fetch_categories,
        fetch_all_directory_records, parse_directory_json,
    )
    cat_country = build_category_country_map(fetch_categories())
    records = fetch_all_directory_records()
    rows = parse_directory_json(records, category_country=cat_country)
    for row in rows:
        run_upsert(row.to_dict())
    return len(rows), len(rows), 0


def _run_cso_scrape() -> tuple[int, int, int]:
    from scrapers.practitioner_finder.cso import (
        fetch_all_directory_records, parse_directory_json,
    )
    records = fetch_all_directory_records()
    rows = parse_directory_json(records)
    for row in rows:
        run_upsert(row.to_dict())
    return len(rows), len(rows), 0


def _run_nora_scrape() -> tuple[int, int, int]:
    from scrapers.practitioner_finder.nora import (
        fetch_all_directory_records, parse_directory_json,
    )
    records = fetch_all_directory_records()
    rows = parse_directory_json(records)
    for row in rows:
        run_upsert(row.to_dict())
    return len(rows), len(rows), 0


def _run_ovdr_scrape() -> tuple[int, int, int]:
    from scrapers.practitioner_finder.ovdr import fetch_all_records
    rows = fetch_all_records()
    for row in rows:
        run_upsert(row.to_dict())
    return len(rows), len(rows), 0


def _run_aanp_scrape() -> tuple[int, int, int]:
    from scrapers.practitioner_finder.migrate_aanp import (
        US_STATES, CA_PROVINCES, fetch_rows_for_state,
    )
    seen: set[str] = set()
    all_rows = []
    for state in US_STATES + CA_PROVINCES:
        try:
            rows = fetch_rows_for_state(state)
        except Exception as e:
            print(f"  WARN: AANP state={state!r} fetch failed: {e}", file=sys.stderr)
            continue
        for r in rows:
            if r.source_url and r.source_url not in seen:
                seen.add(r.source_url)
                all_rows.append(r)
    for row in all_rows:
        run_upsert(row.to_dict())
    return len(all_rows), len(all_rows), 0


def _run_acam_scrape() -> tuple[int, int, int]:
    from scrapers.practitioner_finder.acam import (
        fetch_all_directory_records, parse_directory_json,
    )
    records = fetch_all_directory_records()
    rows = parse_directory_json(records)
    for row in rows:
        run_upsert(row.to_dict())
    return len(rows), len(rows), 0


def _run_abfm_scrape() -> tuple[int, int, int]:
    from scrapers.practitioner_finder.abfm import (
        fetch_all_directory_records, parse_directory_json,
    )
    records = fetch_all_directory_records()
    rows = parse_directory_json(records)
    for row in rows:
        run_upsert(row.to_dict())
    return len(rows), len(rows), 0


def _run_aama_scrape() -> tuple[int, int, int]:
    from scrapers.practitioner_finder.aama import (
        fetch_all_directory_records, parse_directory_json,
    )
    records = fetch_all_directory_records()
    rows = parse_directory_json(records)
    for row in rows:
        run_upsert(row.to_dict())
    return len(rows), len(rows), 0


def _run_nanp_scrape() -> tuple[int, int, int]:
    from scrapers.practitioner_finder.nanp import fetch_all_records
    rows = fetch_all_records()
    for row in rows:
        run_upsert(row.to_dict())
    return len(rows), len(rows), 0


def _run_nccaom_scrape() -> tuple[int, int, int]:
    # Lives in migrate_nccaom now — the live fetch requires a non-headless
    # Playwright session to pass Cloudflare Turnstile (see the migrate
    # module docstring). The pure parser stays in scrapers.practitioner_finder.nccaom.
    from scrapers.practitioner_finder.migrate_nccaom import fetch_all_records
    rows = fetch_all_records()
    for row in rows:
        run_upsert(row.to_dict())
    return len(rows), len(rows), 0


ADAPTERS: list[tuple[str, Callable[[], tuple[int, int, int]]]] = [
    ("oepf", _run_oepf_scrape),
    ("iaomt", _run_iaomt_scrape),
    ("iabdm", _run_iabdm_scrape),
    ("owns", _run_owns_scrape),
    ("cso", _run_cso_scrape),
    ("nora", _run_nora_scrape),
    ("ovdr", _run_ovdr_scrape),
    ("aanp", _run_aanp_scrape),
    ("acam", _run_acam_scrape),
    ("abfm", _run_abfm_scrape),
    ("aama", _run_aama_scrape),
    ("nanp", _run_nanp_scrape),
    ("nccaom", _run_nccaom_scrape),
]


# ---------------------------------------------------------------------------
# scraper_runs logging
# ---------------------------------------------------------------------------
def _log_run_start(adapter_name: str) -> str:
    with supabase_cursor() as cur:
        cur.execute(
            """INSERT INTO scraper_runs (adapter_name, status)
               VALUES (%s, 'running')
               RETURNING id""",
            (adapter_name,),
        )
        return str(cur.fetchone()["id"])


def _log_run_complete(
    run_id: str,
    status: str,
    rows_scraped: int = 0,
    rows_inserted: int = 0,
    rows_updated: int = 0,
    error_message: Optional[str] = None,
) -> None:
    with supabase_cursor() as cur:
        cur.execute(
            """UPDATE scraper_runs
                  SET finished_at = now(),
                      status = %s,
                      rows_scraped = %s,
                      rows_inserted = %s,
                      rows_updated = %s,
                      error_message = %s
                WHERE id = %s""",
            (status, rows_scraped, rows_inserted, rows_updated,
             error_message, run_id),
        )


# ---------------------------------------------------------------------------
# Failure notification — reuses app._send_full_report_email().
# Imported lazily so a failing import here doesn't kill the whole run.
# ---------------------------------------------------------------------------
def _notify_glen(subject: str, body: str) -> None:
    try:
        from app import _send_full_report_email
        _send_full_report_email(
            to_email="drglenswartwout@gmail.com",
            name="Glen",
            subject=subject,
            body=body,
        )
    except Exception as e:
        print(f"  WARN: could not send notification email: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Per-adapter wrapper — try/except, log, notify on failure.
# ---------------------------------------------------------------------------
def _run_adapter(adapter_name: str, fn: Callable[[], tuple[int, int, int]]) -> bool:
    """Returns True if adapter succeeded, False if it failed."""
    print(f"\n=== {adapter_name} ===")
    run_id = _log_run_start(adapter_name)
    try:
        scraped, inserted, updated = fn()
        _log_run_complete(
            run_id, status="success",
            rows_scraped=scraped, rows_inserted=inserted, rows_updated=updated,
        )
        print(f"  ✓ scraped={scraped} inserted={inserted} updated={updated}")
        return True
    except Exception as e:
        tb = traceback.format_exc()
        _log_run_complete(run_id, status="failure", error_message=tb)
        print(f"  ✗ FAILED: {e}", file=sys.stderr)
        _notify_glen(
            subject=f"[practitioner-finder] {adapter_name} failed",
            body=f"Adapter {adapter_name} failed during weekly run.\n\n"
                 f"Traceback:\n{tb}",
        )
        return False


# ---------------------------------------------------------------------------
# Global geocode mop-up sweep — runs once after all adapters complete.
# Belt-and-suspenders: each migrate_*.py already does inline geocoding, but
# this catches any stragglers and is cheap given Mapbox 100k/mo free tier.
# ---------------------------------------------------------------------------
def _global_geocode_sweep() -> tuple[int, int]:
    """Returns (attempted, succeeded)."""
    print("\n=== global geocode sweep ===")
    ungeocoded = list_ungeocoded()
    print(f"  {len(ungeocoded)} ungeocoded rows")
    succeeded = 0
    for r in ungeocoded:
        row_for_geocode = NormalizedPractitionerRow(
            tier=r.get("tier", "org_member"),
            name=r.get("name", "X"),
            specialties=[],
            address1=r.get("address1"),
            city=r.get("city"),
            state=r.get("state"),
            postal=r.get("postal"),
            country=r.get("country", "US"),
        )
        try:
            lat, lng, quality = geocode_row(row_for_geocode)
            update_geocode(r["id"], lat, lng, quality)
            if lat is not None:
                succeeded += 1
        except MapboxError as e:
            print(f"  WARN: geocode failed for {r['id']}: {e}", file=sys.stderr)
    print(f"  ✓ geocoded {succeeded}/{len(ungeocoded)}")
    return len(ungeocoded), succeeded


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply-ghl", action="store_true",
                    help="Run real GHL prospect sync after adapters complete "
                         "(default: dry-run only).")
    ap.add_argument("--only", default="",
                    help="Comma-separated list of adapter names to run "
                         "(default: all registered).")
    args = ap.parse_args()

    started_at = datetime.now(timezone.utc)
    print(f"practitioner-finder run starting at {started_at.isoformat()}")

    selected = [a for a in args.only.split(",") if a.strip()] if args.only else None
    if selected:
        adapters = [(n, f) for (n, f) in ADAPTERS if n in selected]
        if not adapters:
            print(f"ERROR: no adapters matched --only {selected}", file=sys.stderr)
            return 1
    else:
        adapters = list(ADAPTERS)

    if not adapters:
        print("(no adapters registered — scaffolding test run)")

    succeeded = 0
    failed = 0
    for name, fn in adapters:
        if _run_adapter(name, fn):
            succeeded += 1
        else:
            failed += 1

    # Geocode sweep regardless of adapter results — pick up anything stale.
    _global_geocode_sweep()

    # GHL sync — dry-run by default, real-run when --apply-ghl set.
    # Always run, even if some adapters failed, so the allowlisted orgs that
    # DID succeed still flow through.
    print()
    from scrapers.practitioner_finder import ghl_sync
    ghl_sync.run(dry_run=not args.apply_ghl)

    finished_at = datetime.now(timezone.utc)
    elapsed = (finished_at - started_at).total_seconds()
    print(f"\nfinished at {finished_at.isoformat()} (elapsed {elapsed:.1f}s)")
    print(f"adapters: {succeeded} succeeded, {failed} failed")

    # Exit non-zero only if every adapter failed (and there were adapters).
    return 1 if adapters and succeeded == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
