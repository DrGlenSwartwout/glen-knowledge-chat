"""One-shot migration: load VETTED ZYTO / EVOX practitioners from a reviewed CSV.

Unlike the org adapters, ZYTO publishes no directory — these rows come from a
human-vetted CSV produced by the discovery run (practitioners who self-publish
ZYTO / EVOX pages on their own sites). The CSV is the audit ledger and the only
source of truth; nothing here scrapes the web.

Two independent load paths driven off the same CSV (a practitioner can be both):

  MAP   (patient-facing) — rows where ``list_on_map`` is truthy (defaults to the
        ``offers_evox`` value when the column is absent). Upserted into the
        ``practitioners`` table with specialties ['holistic_health',
        'biocommunication'], tier='org_member', source_org='ZYTO', then geocoded
        via the shared Mapbox sweep. Surfaces under Holistic Health → "EVOX /
        Biocommunication" on /practitioner-finder.

  GHL   (B2B outreach) — rows where ``device_tier`` is Elite or Pro and an email
        is present. Upserted into GHL with tags 'zyto-elite-prospect' +
        'biofield-analysis-target' and a zyto_device_tier custom field, so Glen
        can pitch the proprietary Biofield Analysis software onto hardware they
        already own. DRY-RUN by default — pass --apply-ghl to actually upsert.

CSV columns (header row required):
  name, practice_name, city, state, postal, phone, email, website,
  remote_telehealth, device_tier, offers_evox, source_url, notes
Optional: list_on_map  (vetting override; defaults to offers_evox)

Run:
  doppler run --project remedy-match --config prd -- \
    python3 -m scrapers.practitioner_finder.migrate_zyto \
      --csv "/path/to/zyto-discovery-2026-06-04.csv" [--apply-ghl] \
      [--skip-map] [--skip-ghl]

Idempotent: map upsert dedups on source_url; GHL upsert matches by email.
"""
import argparse
import csv
import sys
from typing import Optional

from db_supabase import supabase_cursor
from scrapers.practitioner_finder.models import NormalizedPractitionerRow
from scrapers.practitioner_finder.db import (
    run_upsert,
    list_ungeocoded,
    update_geocode,
)
from scrapers.practitioner_finder.geocode import geocode_row, MapboxError
from scrapers.practitioner_finder.ghl_sync import _split_name


MAP_SPECIALTIES = ["holistic_health", "biocommunication"]
GHL_ELIGIBLE_TIERS = {"elite", "pro"}

_TRUTHY = {"true", "t", "yes", "y", "1", "x"}


def _truthy(val: Optional[str]) -> bool:
    return (val or "").strip().lower() in _TRUTHY


def _clean(val: Optional[str]) -> Optional[str]:
    """Trim; collapse empties to None so to_dict() drops them."""
    if val is None:
        return None
    v = val.strip()
    return v or None


def read_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    print(f"  read {len(rows)} rows from {path}")
    return rows


# --------------------------------------------------------------------------- #
# MAP path
# --------------------------------------------------------------------------- #
def _wants_map(rec: dict) -> bool:
    # Explicit vetting override wins; otherwise fall back to the EVOX flag.
    if "list_on_map" in rec and (rec.get("list_on_map") or "").strip() != "":
        return _truthy(rec.get("list_on_map"))
    return _truthy(rec.get("offers_evox"))


def _to_row(rec: dict) -> Optional[NormalizedPractitionerRow]:
    source_url = _clean(rec.get("source_url")) or _clean(rec.get("website"))
    name = _clean(rec.get("name")) or _clean(rec.get("practice_name"))
    if not source_url or not name:
        return None  # need a dedup key + a display name to list on the map
    return NormalizedPractitionerRow(
        tier="org_member",
        name=name,
        specialties=list(MAP_SPECIALTIES),
        source_org="ZYTO",
        source_url=source_url,
        practice_name=_clean(rec.get("practice_name")),
        credentials=_clean(rec.get("credentials")),
        phone=_clean(rec.get("phone")),
        email=_clean(rec.get("email")),
        website=_clean(rec.get("website")),
        city=_clean(rec.get("city")),
        state=_clean(rec.get("state")),
        postal=_clean(rec.get("postal")),
        country="US",
    )


def _set_telehealth(source_urls: list[str]) -> None:
    """telehealth lives in the table but not in NormalizedPractitionerRow, so set
    it in one post-upsert UPDATE rather than touching the shared row model."""
    if not source_urls:
        return
    with supabase_cursor() as cur:
        cur.execute(
            "UPDATE practitioners SET telehealth = true, updated_at = now() "
            "WHERE source_org = 'ZYTO' AND source_url = ANY(%s)",
            (source_urls,),
        )


def load_map(records: list[dict]) -> dict:
    candidates = [r for r in records if _wants_map(r)]
    print(f"\n=== MAP load === ({len(candidates)} of {len(records)} flagged for map)")

    loaded, skipped, remote_urls = 0, 0, []
    for rec in candidates:
        row = _to_row(rec)
        if row is None:
            skipped += 1
            print(f"  SKIP (no source_url/name): {rec.get('practice_name') or rec.get('name')!r}")
            continue
        run_upsert(row.to_dict())
        loaded += 1
        if _truthy(rec.get("remote_telehealth")):
            remote_urls.append(row.source_url)

    _set_telehealth(remote_urls)
    print(f"  upserted {loaded}, skipped {skipped}, telehealth flagged {len(remote_urls)}")

    print("  geocoding ungeocoded rows...")
    ungeocoded = list_ungeocoded()
    geocoded = 0
    for r in ungeocoded:
        stub = NormalizedPractitionerRow(
            tier="org_member", name="X", specialties=[],
            address1=r.get("address1"), city=r.get("city"),
            state=r.get("state"), postal=r.get("postal"),
            country=r.get("country", "US"),
        )
        try:
            lat, lng, quality = geocode_row(stub)
            update_geocode(r["id"], lat, lng, quality)
            if lat is not None:
                geocoded += 1
        except MapboxError as e:
            print(f"  WARN geocode failed for {r['id']}: {e}", file=sys.stderr)
    print(f"  geocoded {geocoded}/{len(ungeocoded)} pending rows")
    return {"upserted": loaded, "skipped": skipped, "geocoded": geocoded}


# --------------------------------------------------------------------------- #
# GHL path
# --------------------------------------------------------------------------- #
def _ghl_eligible(rec: dict) -> bool:
    tier = (rec.get("device_tier") or "").strip().lower()
    return tier in GHL_ELIGIBLE_TIERS and bool(_clean(rec.get("email")))


def load_ghl(records: list[dict], dry_run: bool = True) -> dict:
    candidates = [r for r in records if _ghl_eligible(r)]
    print(f"\n=== GHL prospect load ({'DRY-RUN' if dry_run else 'REAL'}) === "
          f"({len(candidates)} Elite/Pro + email of {len(records)})")

    synced, errors = 0, 0
    upsert = None
    if not dry_run:
        from app import ghl_upsert_contact  # lazy: avoid Flask import in dry-run
        upsert = ghl_upsert_contact

    for rec in candidates:
        email = _clean(rec.get("email"))
        first, last = _split_name(_clean(rec.get("name")))
        tier = (rec.get("device_tier") or "").strip()
        custom_fields = {"zyto_device_tier": tier}
        src = _clean(rec.get("source_url")) or _clean(rec.get("website"))
        if src:
            custom_fields["practitioner_source_url"] = src
        tags = ["zyto-elite-prospect", "biofield-analysis-target"]

        if dry_run:
            print(f"  WOULD UPSERT {email:<32} [{tier}] tags={tags} "
                  f"name={first} {last}".rstrip())
            synced += 1
            continue
        try:
            _cid, _created, err = upsert(
                email=email, first_name=first, last_name=last,
                phone=_clean(rec.get("phone")) or "",
                source_tag="zyto-elite-prospect",
                extra_tags=["biofield-analysis-target"],
                custom_fields=custom_fields,
            )
            if err:
                print(f"  WARN upsert failed for {email}: {err}", file=sys.stderr)
                errors += 1
                continue
            synced += 1
        except Exception as e:
            print(f"  WARN exception for {email}: {e}", file=sys.stderr)
            errors += 1

    print(f"  {'would upsert' if dry_run else 'upserted'} {synced}, errors {errors}")
    return {"eligible": len(candidates), "synced": synced, "errors": errors}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="Path to the vetted ZYTO CSV.")
    ap.add_argument("--apply-ghl", action="store_true",
                    help="Actually upsert GHL contacts (default: dry-run print).")
    ap.add_argument("--skip-map", action="store_true", help="Skip the map load.")
    ap.add_argument("--skip-ghl", action="store_true", help="Skip the GHL load.")
    args = ap.parse_args()

    print(f"=== ZYTO / EVOX migration from {args.csv} ===")
    records = read_csv(args.csv)

    if not args.skip_map:
        load_map(records)
    if not args.skip_ghl:
        load_ghl(records, dry_run=not args.apply_ghl)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
