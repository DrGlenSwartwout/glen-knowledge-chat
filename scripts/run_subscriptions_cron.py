#!/usr/bin/env python3
"""Daily subscription charge cron entry point.

Runs on Render's cron worker (or folded into an existing multi-step daily
script). Posts to the web service's /api/cron/charge-subscriptions endpoint,
which executes the charge scheduler inside the web container (where the
persistent disk and chat_log.db live).

Render cron containers do NOT share the web service's persistent disk.
Running the scheduler directly here would crash on sqlite3.connect("/data/...")
because /data is not mounted in the cron container.

Required env vars on the cron service:
  WEB_URL       — base URL of the web service (no trailing slash)
                  default: https://glen-knowledge-chat.onrender.com
  CRON_SECRET   — shared secret matching the web service's CRON_SECRET
                  (or CONSOLE_SECRET fallback). Sent as X-Cron-Secret.

Optional:
  DRY_RUN=1     — append ?dry_run=1 to compute + log without charging cards.
"""
import os
import sys
import json
import urllib.request
import urllib.error


WEB_URL = os.environ.get("WEB_URL", "https://glen-knowledge-chat.onrender.com").rstrip("/")
CRON_SECRET = os.environ.get("CRON_SECRET") or os.environ.get("CONSOLE_SECRET", "")

if not CRON_SECRET:
    print("ERROR: CRON_SECRET (or CONSOLE_SECRET) not set on cron service", flush=True)
    sys.exit(1)


def main():
    dry_run = os.environ.get("DRY_RUN", "").lower() in ("1", "true", "yes")
    qs = "?dry_run=1" if dry_run else ""
    url = f"{WEB_URL}/api/cron/charge-subscriptions{qs}"
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
        with urllib.request.urlopen(req, timeout=300) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            print(f"HTTP {resp.status}: {body}", flush=True)
            try:
                data = json.loads(body)
                if data.get("ok"):
                    print(
                        f"Subscription cron: charged={data.get('charged', 0)}"
                        f" skipped={data.get('skipped', 0)}"
                        f" failed={data.get('failed', 0)}"
                        f" notified={data.get('notified', 0)}"
                        + (" [DRY RUN]" if data.get("dry_run") else ""),
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
