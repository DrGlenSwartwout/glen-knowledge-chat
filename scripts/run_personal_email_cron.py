#!/usr/bin/env python3
"""Daily Personal email cron entry point.

Runs on Render's cron worker. Posts to the web service's
/cron/personal-send endpoint, which executes the orchestrator inside
the web container (where the persistent disk and chat_log.db live).

Render cron containers do NOT share the web service's persistent disk.
Calling the orchestrator directly here would crash on
sqlite3.connect("/data/chat_log.db") because /data is not mounted in
the cron container.

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


WEB_URL = os.environ.get("WEB_URL", "https://glen-knowledge-chat.onrender.com").rstrip("/")
CRON_SECRET = os.environ.get("CRON_SECRET") or os.environ.get("CONSOLE_SECRET", "")

if not CRON_SECRET:
    print("ERROR: CRON_SECRET (or CONSOLE_SECRET) not set on cron service", flush=True)
    sys.exit(1)


def main():
    url = f"{WEB_URL}/cron/personal-send"
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
                    print(f"Personal email cron: sent {data.get('sent', '?')} email(s)", flush=True)
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
