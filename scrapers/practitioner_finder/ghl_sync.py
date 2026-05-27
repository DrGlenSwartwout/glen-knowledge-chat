"""GHL prospect sync — push allowlisted org_member practitioners into GHL.

Tag-only posture (Phase 2 first-run): no campaigns or workflows wired. Sync
applies `practitioner-prospect` + `practitioner-prospect-<org>` tags and a
`practitioner_source_url` custom field. Glen and Rae review the cohort
manually before any outreach.

Allowlist: env var GHL_PROSPECTING_ORGS (comma-separated, lowercase). Default
allowlist: oepf,cso,nora,ovdr,owns,iaomt,iabdm. NCCAOM/AANP/ACAM/ABFM/NANP/AAMA
are excluded by default (low natural fit, large volume).

Invoke:
  doppler run --project remedy-match --config prd -- \
    python3 -m scrapers.practitioner_finder.ghl_sync [--dry-run]

Called automatically from run_all.py at the end of each weekly run.
"""
import argparse
import os
import sys
import json
from typing import Optional

from db_supabase import supabase_cursor


DEFAULT_ALLOWLIST = "oepf,cso,nora,ovdr,owns,iaomt,iabdm"


def _allowlist() -> list[str]:
    raw = os.environ.get("GHL_PROSPECTING_ORGS", DEFAULT_ALLOWLIST)
    return [o.strip().lower() for o in raw.split(",") if o.strip()]


def _fetch_sync_candidates(allowlist: list[str]) -> list[dict]:
    """Return practitioners eligible for GHL sync: org_member tier, in
    allowlist, with non-empty email."""
    if not allowlist:
        return []
    sql = """
        SELECT id, name, email, phone, source_org, source_url
        FROM practitioners
        WHERE tier = 'org_member'
          AND lower(source_org) = ANY(%s)
          AND email IS NOT NULL
          AND email <> ''
          AND removal_requested = false
    """
    with supabase_cursor() as cur:
        cur.execute(sql, (allowlist,))
        return [dict(r) for r in cur.fetchall()]


def _split_name(full_name: Optional[str]) -> tuple[str, str]:
    """Best-effort split into (first, last). Empty strings for missing parts."""
    if not full_name:
        return "", ""
    parts = full_name.strip().split(maxsplit=1)
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[1]


def _log_dryrun(row: dict, tags: list[str], custom_fields: dict) -> None:
    first, last = _split_name(row.get("name"))
    sql = """
        INSERT INTO ghl_prospect_dryrun_log
          (practitioner_id, intended_email, intended_first_name,
           intended_last_name, intended_phone, intended_tags,
           intended_custom_fields, source_org)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """
    with supabase_cursor() as cur:
        cur.execute(sql, (
            row["id"], row["email"], first, last, row.get("phone"),
            tags, json.dumps(custom_fields), row.get("source_org"),
        ))


def _real_upsert(row: dict, tags: list[str], custom_fields: dict) -> tuple[Optional[str], Optional[str]]:
    """Returns (contact_id, error)."""
    # Lazy import to avoid Flask import overhead in dry-run path.
    from app import ghl_upsert_contact
    first, last = _split_name(row.get("name"))
    contact_id, _created, err = ghl_upsert_contact(
        email=row["email"],
        first_name=first,
        last_name=last,
        phone=row.get("phone") or "",
        source_tag="practitioner-prospect",
        extra_tags=tags,
        custom_fields=custom_fields,
    )
    return contact_id, err


def run(dry_run: bool = True) -> dict:
    """Returns a summary dict for orchestrator logging."""
    allowlist = _allowlist()
    print(f"=== GHL prospect sync ({'DRY-RUN' if dry_run else 'REAL'}) ===")
    print(f"  allowlist: {allowlist}")

    candidates = _fetch_sync_candidates(allowlist)
    print(f"  {len(candidates)} eligible candidates")

    synced = 0
    errors = 0
    for row in candidates:
        org = (row.get("source_org") or "").lower()
        tags = [f"practitioner-prospect-{org}"]
        custom_fields = {}
        if row.get("source_url"):
            custom_fields["practitioner_source_url"] = row["source_url"]

        try:
            if dry_run:
                _log_dryrun(row, ["practitioner-prospect"] + tags, custom_fields)
            else:
                contact_id, err = _real_upsert(row, tags, custom_fields)
                if err:
                    print(f"  WARN: upsert failed for {row['email']}: {err}",
                          file=sys.stderr)
                    errors += 1
                    continue
            synced += 1
        except Exception as e:
            print(f"  WARN: exception for {row['email']}: {e}", file=sys.stderr)
            errors += 1

    print(f"  ✓ {'logged' if dry_run else 'synced'} {synced}, errors {errors}")
    if dry_run:
        print(f"  → review: SELECT * FROM ghl_prospect_dryrun_log "
              f"ORDER BY logged_at DESC LIMIT 20;")
    return {"eligible": len(candidates), "synced": synced, "errors": errors}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="Write intended upserts to ghl_prospect_dryrun_log "
                         "instead of hitting GHL (default: real upsert).")
    args = ap.parse_args()
    summary = run(dry_run=args.dry_run)
    return 0 if summary["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
