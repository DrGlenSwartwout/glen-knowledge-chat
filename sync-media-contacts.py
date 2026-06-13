#!/usr/bin/env python3
"""
Run from LOCAL Mac (not Render) to feed PR/media contacts from the vault CSV
into the People hub. The vault CSV is not on Render, so this runs locally and
POSTs to /api/people?merge_tags=1 (additive, idempotent) tagged type:pr-media.

Source: ~/AI-Training/04 Copy/pr-pitches/media-contacts.csv  (override with
        MEDIA_CONTACTS_CSV env var). Rows without an email are skipped (the
        vault CSV stays the full record; email is the hub's dedup/automation key).

Usage:
  doppler run --project remedy-match --config prd -- python3 sync-media-contacts.py [--dry-run]
"""
import csv
import json
import os
import subprocess
import sys

RENDER_URL = os.environ.get("RENDER_URL", "https://glen-knowledge-chat.onrender.com")
CONSOLE_SECRET = os.environ.get("CONSOLE_SECRET", os.environ.get("WEBHOOK_SECRET", ""))
DEFAULT_CSV = os.path.expanduser("~/AI-Training/04 Copy/pr-pitches/media-contacts.csv")
CSV_PATH = os.environ.get("MEDIA_CONTACTS_CSV", DEFAULT_CSV)
DRY_RUN = "--dry-run" in sys.argv


def row_to_person(row):
    """Map a media-contacts.csv row to a people upsert dict, or None if no email."""
    email = (row.get("email") or "").strip().lower()
    if not email:
        return None
    name = (row.get("contact_name") or "").strip()
    first, last = "", ""
    if name:
        parts = name.split(" ", 1)
        first = parts[0]
        last = parts[1] if len(parts) > 1 else ""
    outlet = (row.get("outlet") or "").strip()
    return {
        "email": email, "name": name, "first_name": first, "last_name": last,
        "phone": (row.get("phone") or "").strip(),
        "organizations": [outlet] if outlet else [],
        "source": "media-outreach",
        "tags": ["type:pr-media", "consent:cold-no-consent", "source:media-outreach"],
    }


def post_people(persons):
    url = f"{RENDER_URL}/api/people?merge_tags=1"
    cmd = ["curl", "-s", "-X", "POST", url,
           "-H", "Content-Type: application/json",
           "-H", f"X-Console-Key: {CONSOLE_SECRET}",
           "-d", json.dumps(persons)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    try:
        return json.loads(r.stdout), None
    except Exception:
        return None, r.stdout[:300]


def main():
    if not os.path.exists(CSV_PATH):
        sys.exit(f"ERROR: CSV not found at {CSV_PATH}")
    persons, skipped = [], 0
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            person = row_to_person(row)
            if person:
                persons.append(person)
            else:
                skipped += 1
    print(f"Read {CSV_PATH}: {len(persons)} with email, {skipped} skipped (no email).")
    if DRY_RUN:
        for p in persons:
            print(f"  would upsert: {p['email']}  {p['organizations']}  {p['tags']}")
        print("Dry run, nothing posted.")
        return
    if not persons:
        print("Nothing to post.")
        return
    if not CONSOLE_SECRET:
        sys.exit("ERROR: CONSOLE_SECRET not set (run via doppler).")
    result, err = post_people(persons)
    if err:
        sys.exit(f"POST failed: {err}")
    print(f"Posted: {result}")


if __name__ == "__main__":
    main()
