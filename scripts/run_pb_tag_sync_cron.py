#!/usr/bin/env python3
"""Daily Practice Better → People DB + GHL tag sync cron entry point.

Runs on Render's cron worker. Posts to the web service's /admin/sync-pb-tags
endpoint, which fetches PB tag definitions + all client records, upserts into
the SQLite `people` table, and mirrors tags to GHL contacts in the `pb:`
namespace. Additive-only — tags are added, never removed.

Same cross-container pattern as run_personal_email_cron.py — the persistent
disk + PB credentials live on the web service, not the cron container.

Required env vars on the cron service:
  WEB_URL       — base URL of the web service (no trailing slash)
                  default: https://glen-knowledge-chat.onrender.com
  CRON_SECRET   — shared secret matching the web service's CRON_SECRET
                  (or CONSOLE_SECRET fallback). Sent as X-Cron-Secret.
"""
import os
import sys
import json
import urllib.request
import urllib.error


WEB_URL     = os.environ.get("WEB_URL", "https://glen-knowledge-chat.onrender.com").rstrip("/")
CRON_SECRET = os.environ.get("CRON_SECRET") or os.environ.get("CONSOLE_SECRET", "")

if not CRON_SECRET:
    print("ERROR: CRON_SECRET (or CONSOLE_SECRET) not set on cron service", flush=True)
    sys.exit(1)


def main():
    url = f"{WEB_URL}/admin/sync-pb-tags"
    req = urllib.request.Request(
        url,
        method="POST",
        headers={
            "X-Cron-Secret": CRON_SECRET,
            "Content-Type": "application/json",
        },
        data=b"{}",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            print(f"HTTP {resp.status}: {body}", flush=True)
            try:
                data = json.loads(body)
                if data.get("ok"):
                    s = data.get("summary", {})
                    print(
                        f"PB tag sync: {s.get('records_fetched',0)} records, "
                        f"{s.get('people_upserted',0)} people upserted, "
                        f"{s.get('ghl_synced',0)} GHL synced, "
                        f"{s.get('total_tags_attached',0)} tags attached, "
                        f"{s.get('ghl_errors',0)} errors, "
                        f"{s.get('elapsed_sec',0)}s",
                        flush=True,
                    )
                    if s.get("ghl_errors", 0) > 0:
                        sys.exit(2)
                else:
                    print(f"Cron failed: {data.get('error', 'unknown')}", flush=True)
                    sys.exit(2)
            except json.JSONDecodeError:
                print("Response was not valid JSON", flush=True)
                sys.exit(3)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"HTTPError {e.code}: {body}", flush=True)
        sys.exit(4)
    except urllib.error.URLError as e:
        print(f"URLError: {e}", flush=True)
        sys.exit(5)


if __name__ == "__main__":
    main()
