"""Regenerate the GHL prospect dry-run log and export it to a dated CSV.

Truncates the ghl_prospect_dryrun_log staging table, re-runs ghl_sync in
dry-run mode against the current practitioners table (respecting the
GHL_PROSPECTING_ORGS allowlist), then exports a review CSV in the established
format:

    First Name,Last Name,Email,Phone,Org,Tags,Source URL,Logged At

Invoke:
    doppler run --project remedy-match --config prd -- \
        python3 -m scrapers.practitioner_finder.export_dryrun_csv OUTPUT.csv

Called by the monthly launchd job (run_monthly.sh) so each scrape leaves a
fresh review CSV for Glen and Rae before any real GHL outreach is flipped on.
"""
import csv
import json
import sys

from db_supabase import supabase_cursor
from scrapers.practitioner_finder import ghl_sync


def regenerate(out_path: str) -> int:
    # 1. Clean the staging log so we don't double up on prior runs.
    with supabase_cursor() as cur:
        cur.execute("TRUNCATE ghl_prospect_dryrun_log")
    print("truncated ghl_prospect_dryrun_log")

    # 2. Re-run dry-run (respects GHL_PROSPECTING_ORGS allowlist).
    summary = ghl_sync.run(dry_run=True)
    print("summary:", summary)

    # 3. Export to CSV in the established format.
    with supabase_cursor() as cur:
        cur.execute("""
            SELECT intended_first_name, intended_last_name, intended_email,
                   intended_phone, source_org, intended_tags,
                   intended_custom_fields, logged_at
            FROM ghl_prospect_dryrun_log
            ORDER BY source_org, intended_last_name, intended_first_name
        """)
        rows = cur.fetchall()

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["First Name", "Last Name", "Email", "Phone", "Org",
                    "Tags", "Source URL", "Logged At"])
        for r in rows:
            cf = r["intended_custom_fields"]
            if isinstance(cf, str):
                cf = json.loads(cf) if cf else {}
            source_url = (cf or {}).get("practitioner_source_url", "")
            tags = r["intended_tags"] or []
            org = (r["source_org"] or "").upper()
            logged = r["logged_at"].strftime("%Y-%m-%d %H:%M") if r["logged_at"] else ""
            w.writerow([
                r["intended_first_name"] or "",
                r["intended_last_name"] or "",
                r["intended_email"] or "",
                r["intended_phone"] or "",
                org,
                ", ".join(tags),
                source_url,
                logged,
            ])

    print(f"wrote {len(rows)} rows -> {out_path}")
    return len(rows)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python3 -m scrapers.practitioner_finder.export_dryrun_csv OUTPUT.csv",
              file=sys.stderr)
        return 2
    regenerate(sys.argv[1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
