#!/usr/bin/env python3
"""
Run from LOCAL Mac (not Render) to sync pending leads to GHL.
Render's AWS IP is blocked by GHL's Cloudflare; Mac is not.

Usage:
  python3 sync-ghl-leads.py [--dry-run]
"""
import os, sys, json, urllib.request as ur, urllib.error

RENDER_URL    = "https://glen-knowledge-chat.onrender.com"
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "")
GHL_API_KEY   = os.environ.get("GHL_API_KEY", "")
GHL_BASE      = "https://services.leadconnectorhq.com"
DRY_RUN       = "--dry-run" in sys.argv

def ghl_headers():
    return {
        "Authorization": f"Bearer {GHL_API_KEY}",
        "Content-Type":  "application/json",
        "Version":       "2021-07-28",
    }

def ghl_upsert(email, first, last, phone, tags):
    # Try GET first
    url = f"{GHL_BASE}/contacts/?email={urllib.parse.quote(email)}&limit=1"
    req = ur.Request(url, headers=ghl_headers())
    try:
        with ur.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
            contacts = data.get("contacts", [])
            if contacts:
                return contacts[0]["id"], False
    except Exception:
        pass

    # Create
    payload = {"email": email, "firstName": first, "lastName": last}
    if phone:
        payload["phone"] = phone
    if tags:
        payload["tags"] = tags
    req = ur.Request(f"{GHL_BASE}/contacts/",
                     data=json.dumps(payload).encode(),
                     headers=ghl_headers(), method="POST")
    try:
        with ur.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
            contact_id = data.get("contact", {}).get("id") or data.get("id")
            return contact_id, True
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:200]
        print(f"  GHL POST error {e.code}: {body}")
        return None, False

def mark_synced(lead_id, contact_id):
    payload = json.dumps({"id": lead_id, "contact_id": contact_id}).encode()
    headers = {"Content-Type": "application/json", "X-Webhook-Secret": WEBHOOK_SECRET}
    req = ur.Request(f"{RENDER_URL}/leads/mark-ghl-synced",
                     data=payload, headers=headers, method="POST")
    with ur.urlopen(req, timeout=10) as r:
        return json.loads(r.read())

def main():
    import urllib.parse
    if not GHL_API_KEY:
        print("Set GHL_API_KEY env var (from Doppler)")
        sys.exit(1)
    if not WEBHOOK_SECRET:
        print("Set WEBHOOK_SECRET env var")
        sys.exit(1)

    # Get pending leads from Render
    req = ur.Request(f"{RENDER_URL}/leads/pending-ghl",
                     headers={"X-Webhook-Secret": WEBHOOK_SECRET})
    with ur.urlopen(req, timeout=10) as r:
        data = json.loads(r.read())

    leads = data.get("leads", [])
    print(f"Found {len(leads)} leads to sync to GHL")

    for lead in leads:
        email = lead.get("email", "")
        first = lead.get("first_name", "") or ""
        last  = lead.get("last_name", "") or ""
        phone = lead.get("phone", "") or ""
        raw   = json.loads(lead.get("raw_json", "{}")) if lead.get("raw_json") else {}
        source = lead.get("source", "")

        # Build tags from source and raw data
        tags = [f"source:{source}"] if source else []

        print(f"  [{lead['id']}] {email} ({first} {last}) from {source}")
        print(f"    Error was: {lead.get('ghl_error', '')}")

        if DRY_RUN:
            print(f"  [DRY RUN] Would create/find GHL contact for {email}")
            continue

        contact_id, created = ghl_upsert(email, first, last, phone, tags)
        if contact_id:
            action = "created" if created else "found existing"
            print(f"  GHL contact {action}: {contact_id}")
            result = mark_synced(lead["id"], contact_id)
            print(f"  Marked synced: {result}")
        else:
            print(f"  FAILED to create GHL contact for {email}")

    if DRY_RUN:
        print("\n[Dry run complete - no changes made]")
    else:
        print(f"\nDone. {len(leads)} leads processed.")

if __name__ == "__main__":
    main()
