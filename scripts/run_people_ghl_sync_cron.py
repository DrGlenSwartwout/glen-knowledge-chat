#!/usr/bin/env python3
"""Daily People hub → GHL mirror cron entry point.

Runs on Render's cron worker. POSTs the web service's
/admin/sync-people-to-ghl endpoint, which classifies existing people
(type:client/prospect + consent) and enqueues tag_add ops for the opted-in
ones onto the GHL write-queue (drained from the Mac). Cold contacts are never
enqueued. No sends — just tag mirroring so GHL automations can segment.

Same cross-container pattern as run_pb_tag_sync_cron.py — the persistent disk +
queue live on the web service, not the cron container.

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
    url = f"{WEB_URL}/admin/sync-people-to-ghl"
    req = urllib.request.Request(
        url,
        method="POST",
        headers={"X-Cron-Secret": CRON_SECRET, "Content-Type": "application/json"},
        data=b"{}",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            print(f"HTTP {resp.status}: {body}", flush=True)
            try:
                data = json.loads(body)
                if data.get("ok"):
                    c, m = data.get("classified", {}), data.get("mirrored", {})
                    print(
                        f"People→GHL: classified {c.get('updated',0)} "
                        f"(+{c.get('added_client',0)} client, +{c.get('added_prospect',0)} prospect, "
                        f"+{c.get('added_opted_in',0)} opted-in); "
                        f"mirrored {m.get('enqueued',0)} enqueued of {m.get('opted_in',0)} opted-in",
                        flush=True,
                    )
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
