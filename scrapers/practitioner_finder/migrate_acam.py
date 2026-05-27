"""One-shot migration: scrape ACAM directory and load into practitioners table.

Run via:
  doppler run --project remedy-match --config prd -- \\
    python3 -m scrapers.practitioner_finder.migrate_acam

Single-stage scrape:
  - ACAM's primary site (acam.org / www.acam.org) is behind a Cloudflare
    managed challenge that returns 403 to plain HTTP clients. The
    YourMembership-backed Find-a-Practitioner search at
    ``/search/custom.asp?id=1758`` is therefore unreachable without
    solving the JS challenge.
  - The MembersState page at ``/page/MembersState`` embeds a ZeeMaps
    iframe that pulls the entire member roster from a public JSON
    endpoint (``https://www.zeemaps.com/emarkers?g=3473180``) outside
    Cloudflare. We fetch that endpoint directly.
  - The payload is ~121 markers, all in a single response. No per-marker
    detail fetch — the marker dict carries name, address, lat/lng, and
    country only. Practice name, phone, email, website are not
    exposed on the public map and live behind the Cloudflare wall on
    each member's profile page.

Idempotent — re-running upserts by source_url (the per-marker anchored
ACAM MembersState URL). After load, runs the shared geocoder over any
rows still lacking lat/lng.
"""
import sys

from scrapers.practitioner_finder.acam import (
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
    print("Fetching ACAM directory via ZeeMaps emarkers endpoint...")
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
