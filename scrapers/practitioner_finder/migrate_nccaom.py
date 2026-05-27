"""One-shot migration: scrape the NCCAOM Find-a-Practitioner directory
and load into the practitioners table.

Run via:
  doppler run --project remedy-match --config prd -- \\
    python3 -m scrapers.practitioner_finder.migrate_nccaom

Single-stage scrape:
  - Walk every (Country, [State]) tuple NCCAOM's CountryCode/StateCode
    dropdowns expose (50 US states + DC + 5 territories + 20 non-US
    countries) at /FAP/SearchResultWithoutMap.
  - Each results page (<=20 practitioners) carries every field we need
    (name, status, credentials via cert-codes, address, phone, website)
    in the card view — no per-record detail fetch needed.

NCCAOM is the largest adapter in the Practitioner Finder fleet (~15-20k
listed diplomates). The on-page disclaimer notes the directory is
"voluntary, not all certified Diplomates will be listed" — total
NCCAOM certified count is ~25,000+, but only the opted-in subset is
publicly searchable.

Idempotent — re-running upserts by source_url (the bare
``/FAP/PractitionerDetail?AgencyClientId=<b64>`` URL is stable across
re-runs since AgencyClientId is the opaque NCCAOM-assigned identifier).
After load, runs the shared geocoder over any rows still lacking
lat/lng.

Production note: ``directory.nccaom.org`` is Cloudflare-protected. If a
live static-UA scrape returns HTTP 403 challenge pages, swap
``fetch_search_page`` for a Playwright-backed variant (the parser is
fully decoupled from fetch)."""
import sys

from scrapers.practitioner_finder.nccaom import fetch_all_records
from scrapers.practitioner_finder.geocode import geocode_row, MapboxError
from scrapers.practitioner_finder.db import (
    run_upsert,
    list_ungeocoded,
    update_geocode,
)
from scrapers.practitioner_finder.models import NormalizedPractitionerRow


def main() -> int:
    print("Fetching NCCAOM directory via directory.nccaom.org/FAP/SearchResultWithoutMap...")
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
