"""One-shot migration: scrape eyehealingcenter.com and load into practitioners table.

Run via:
  doppler run --project remedy-match --config prd -- \
    python3 -m scrapers.practitioner_finder.migrate_eyehealingcenter

Idempotent — re-running upserts by source_url. After load, runs geocoder over
any rows still lacking lat/lng."""
import sys

from scrapers.practitioner_finder.eyehealingcenter import (
    fetch_by_state_html,
    fetch_by_city_html,
    parse_by_state_html,
    parse_by_city_html,
)
from scrapers.practitioner_finder.geocode import geocode_row, MapboxError
from scrapers.practitioner_finder.db import (
    run_upsert,
    list_ungeocoded,
    update_geocode,
)
from scrapers.practitioner_finder.models import NormalizedPractitionerRow


def main() -> int:
    print("Fetching by-state page...")
    state_html = fetch_by_state_html()
    state_rows = parse_by_state_html(state_html)
    print(f"  parsed {len(state_rows)} rows")

    print("Fetching by-city page...")
    city_html = fetch_by_city_html()
    city_rows = parse_by_city_html(city_html)
    print(f"  parsed {len(city_rows)} rows")

    all_rows = state_rows + city_rows
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
            tier="eyehealing", name="X", specialties=[],
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
