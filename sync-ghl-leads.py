#!/usr/bin/env python3
"""
Run from LOCAL Mac (not Render) to sync pending leads to GHL.
Render's AWS IP is blocked by GHL's Cloudflare WAF.
Also uses curl subprocess to bypass JA3 TLS fingerprint blocking.

Usage:
  doppler run --project remedy-match --config prd -- python3 sync-ghl-leads.py [--dry-run]
"""
import os, sys, json, subprocess, urllib.parse

RENDER_URL     = "https://glen-knowledge-chat.onrender.com"
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "")
GHL_API_KEY    = os.environ.get("GHL_API_KEY", "")
GHL_BASE       = "https://rest.gohighlevel.com/v1"
DRY_RUN        = "--dry-run" in sys.argv


def curl_get(url, headers=None):
    cmd = ["curl", "-s", url]
    for k, v in (headers or {}).items():
        cmd += ["-H", f"{k}: {v}"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    return json.loads(r.stdout)


def curl_post(url, payload, headers=None):
    cmd = ["curl", "-s", "-X", "POST", url,
           "-H", "Content-Type: application/json",
           "-d", json.dumps(payload)]
    for k, v in (headers or {}).items():
        cmd += ["-H", f"{k}: {v}"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    try:
        return json.loads(r.stdout), None
    except Exception:
        return None, r.stdout[:200]


def curl_put(url, payload, headers=None):
    cmd = ["curl", "-s", "-X", "PUT", url,
           "-H", "Content-Type: application/json",
           "-d", json.dumps(payload)]
    for k, v in (headers or {}).items():
        cmd += ["-H", f"{k}: {v}"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    try:
        return json.loads(r.stdout), None
    except Exception:
        return None, r.stdout[:200]


def ghl_auth():
    return {"Authorization": f"Bearer {GHL_API_KEY}"}


def ghl_upsert(email, first, last, phone, tags):
    enc_email = urllib.parse.quote(email)
    data = curl_get(f"{GHL_BASE}/contacts/?email={enc_email}&limit=1", ghl_auth())
    contacts = data.get("contacts", [])
    if contacts:
        contact_id = contacts[0]["id"]
        if tags:
            existing = set(contacts[0].get("tags", []))
            existing.update(tags)
            curl_put(f"{GHL_BASE}/contacts/{contact_id}", {"tags": list(existing)}, ghl_auth())
        return contact_id, False

    payload = {"email": email, "firstName": first, "lastName": last}
    if phone:
        payload["phone"] = phone
    if tags:
        payload["tags"] = tags

    data, err = curl_post(f"{GHL_BASE}/contacts/", payload, ghl_auth())
    if err:
        print(f"  GHL POST error: {err}")
        return None, False
    contact_id = (data or {}).get("contact", {}).get("id") or (data or {}).get("id")
    return contact_id, True


def render_get(path):
    headers = {"X-Webhook-Secret": WEBHOOK_SECRET}
    return curl_get(f"{RENDER_URL}{path}", headers)


def render_post(path, payload):
    headers = {"X-Webhook-Secret": WEBHOOK_SECRET}
    return curl_post(f"{RENDER_URL}{path}", payload, headers)


def main():
    if not GHL_API_KEY:
        print("Set GHL_API_KEY (use: doppler run --project remedy-match --config prd -- python3 sync-ghl-leads.py)")
        sys.exit(1)
    if not WEBHOOK_SECRET:
        print("Set WEBHOOK_SECRET env var")
        sys.exit(1)

    data = render_get("/leads/pending-ghl")
    leads = data.get("leads", [])
    print(f"Found {len(leads)} leads to sync to GHL")

    synced = 0
    failed = 0
    for lead in leads:
        email  = lead.get("email", "")
        first  = lead.get("first_name") or ""
        last   = lead.get("last_name") or ""
        phone  = lead.get("phone") or ""
        source = lead.get("source", "")
        tags   = [f"source:{source}"] if source else []

        print(f"  [{lead['id']}] {email} ({first} {last}) from {source}")

        if DRY_RUN:
            print(f"    [DRY RUN] would create/find GHL contact")
            continue

        contact_id, created = ghl_upsert(email, first, last, phone, tags)
        if contact_id:
            action = "created" if created else "found"
            print(f"    GHL contact {action}: {contact_id}")
            result, _ = render_post("/leads/mark-ghl-synced", {"id": lead["id"], "contact_id": contact_id})
            print(f"    Marked synced: {result}")
            synced += 1
        else:
            print(f"    FAILED for {email}")
            failed += 1

    if DRY_RUN:
        print("\n[Dry run — no changes made]")
    else:
        print(f"\nDone: {synced} synced, {failed} failed")


if __name__ == "__main__":
    main()
