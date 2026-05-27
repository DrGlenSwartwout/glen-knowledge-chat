"""One-shot migration: scrape OWNS directory and load into practitioners table.

Run via:
  doppler run --project remedy-match --config prd -- \
    python3 -m scrapers.practitioner_finder.migrate_owns

Single-stage scrape:
  - Fetch the portfolio_category taxonomy once to build a
    {term_id: ISO2-country} map (Canada=CA, United Kingdom=GB, all 50
    US states -> US).
  - Page through the WP ``portfolio`` REST endpoint
    (GET /wp-json/wp/v2/portfolio) at per_page=100.
  - Each record carries a fully-populated content.rendered HTML block
    with the labelled contact fields (Clinic Name, Address, City,
    State/Province/County, Zip Code/Postal Code, Phone, Email, Website)
    — no per-listing fetch is required.

Idempotent — re-running upserts by source_url. After load, runs the
shared geocoder over any rows still lacking lat/lng."""
import sys

from scrapers.practitioner_finder.owns import (
    build_category_country_map,
    fetch_all_directory_records,
    fetch_categories,
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
    print("Fetching OWNS portfolio_category taxonomy...")
    categories = fetch_categories()
    cat_country = build_category_country_map(categories)
    print(f"  resolved {len(cat_country)} of {len(categories)} terms to ISO2 countries")

    print("Fetching OWNS directory via WP REST portfolio endpoint...")
    records = fetch_all_directory_records()
    print(f"  fetched {len(records)} raw records")

    rows = parse_directory_json(records, category_country=cat_country)
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
