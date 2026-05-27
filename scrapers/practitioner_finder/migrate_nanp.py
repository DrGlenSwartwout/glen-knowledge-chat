"""One-shot migration: scrape NANP (National Association of Nutrition
Professionals) directory and load into the practitioners table.

Run via:
  doppler run --project remedy-match --config prd -- \
    python3 -m scrapers.practitioner_finder.migrate_nanp

Two-stage scrape (mirrors OEPF):
  Stage 1: walk every results page at
    GET https://mynanp.nanp.org/search/newsearch.asp?cdlMemberTypeID=1705148&...&page=N
    Each page is up to 20 rows of (profile_id, name, city, state, country).
  Stage 2: fetch each
    GET https://mynanp.nanp.org/profile/?ID=<numeric>
    and parse the full practitioner row (practice, address, phone, email,
    website, BCHN credential flag).

Idempotent — re-running upserts by source_url (the bare
``/profile/?ID=<numeric>`` URL is stable across re-runs). After load,
runs the shared geocoder over any rows still lacking lat/lng.
"""
import sys

from scrapers.practitioner_finder.nanp import fetch_all_records
from scrapers.practitioner_finder.geocode import geocode_row, MapboxError
from scrapers.practitioner_finder.db import (
    run_upsert,
    list_ungeocoded,
    update_geocode,
)
from scrapers.practitioner_finder.models import NormalizedPractitionerRow


def main() -> int:
    print("Fetching NANP directory via mynanp.nanp.org (YourMembership Classic)...")
    rows = fetch_all_records()
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
