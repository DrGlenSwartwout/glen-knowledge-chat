"""One-shot migration: scrape AAMA directory and load into practitioners table.

Run via:
  doppler run --project remedy-match --config prd -- \\
    python3 -m scrapers.practitioner_finder.migrate_aama

Single-stage scrape:
  - Prime cookies via GET /patient-referral-directory.
  - Read ``directory_search_id`` from the public search-form endpoint.
  - POST search-directory/ for page 1 (+ the upstream data_url).
  - Walk pages 2..N via search-directory-paged/ at pageSize=10.
  - Each search-result dict is fully-populated (name + title-credentials
    + address + phone + specialty + website) — no per-listing fetch is
    required.

Idempotent — re-running upserts by source_url (the synthesized
``patient-referral-directory#/profile/<id>`` fragment URL, stable across
re-runs because the MemberClicks profile id is stable). After load,
runs the shared geocoder over any rows still lacking lat/lng."""
import sys

from scrapers.practitioner_finder.aama import (
    fetch_all_directory_records,
    parse_directory_json,
)
from scrapers.practitioner_finder.geocode import geocode_row, MapboxError
from scrapers.practitioner_finder.db import (
    run_upsert,
    list_ungeocoded,
    update_geocode,
)
from scrapers.practitioner_finder.models import NormalizedPractitionerRow


def main() -> int:
    print("Fetching AAMA directory via MemberClicks ui-directory-search/v2 API...")
    records = fetch_all_directory_records()
    print(f"  fetched {len(records)} raw records")

    rows = parse_directory_json(records)
    print(f"  parsed {len(rows)} rows")

    print(f"\nUpserting {len(rows)} rows...")
    for row in rows:
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
