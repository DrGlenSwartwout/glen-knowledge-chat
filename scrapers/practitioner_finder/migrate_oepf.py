"""One-shot migration: scrape OEPF directory and load into practitioners table.

Run via:
  doppler run --project remedy-match --config prd -- \
    python3 -m scrapers.practitioner_finder.migrate_oepf

Two-stage scrape:
  1. WP REST API (paginated, fast) discovers all listing URLs.
  2. For each listing URL, fetch the detail HTML and parse 1-2 rows
     (primary + optional 2nd doctor).

Idempotent — re-running upserts by source_url. After load, runs geocoder
over any rows still lacking lat/lng."""
import sys

from scrapers.practitioner_finder.oepf import (
    fetch_listing_index,
    fetch_directory_listing_html,
    parse_directory_listing_html,
)
from scrapers.practitioner_finder.geocode import geocode_row, MapboxError
from scrapers.practitioner_finder.db import (
    run_upsert,
    list_ungeocoded,
    update_geocode,
)
from scrapers.practitioner_finder.models import NormalizedPractitionerRow


def main() -> int:
    print("Fetching OEPF listing index from WP REST API...")
    index = fetch_listing_index()
    print(f"  found {len(index)} listings")

    all_rows: list[NormalizedPractitionerRow] = []
    for i, entry in enumerate(index, start=1):
        url = entry.get("link")
        if not url:
            continue
        try:
            html = fetch_directory_listing_html(url)
        except Exception as e:
            print(f"  WARN: fetch failed for {url}: {e}")
            continue
        rows = parse_directory_listing_html(html, source_url=url)
        all_rows.extend(rows)
        if i % 25 == 0:
            print(f"  fetched {i}/{len(index)} listings ({len(all_rows)} rows so far)")
    print(f"  parsed {len(all_rows)} total rows across {len(index)} listings")

    print(f"\nUpserting {len(all_rows)} rows...")
    for row in all_rows:
        run_upsert(row.to_dict())
    print("  upsert complete")

    print("\nGeocoding ungeocoded rows...")
    ungeocoded = list_ungeocoded()
    print(f"  {len(ungeocoded)} rows need geocoding")
    geocoded_count = 0
    for r in ungeocoded:
        row_for_geocode = NormalizedPractitionerRow(
            tier="org_member",
            name="X",
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
                geocoded_count += 1
        except MapboxError as e:
            print(f"  WARN: geocode failed for {r['id']}: {e}")
    print(f"  successfully geocoded {geocoded_count}/{len(ungeocoded)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
