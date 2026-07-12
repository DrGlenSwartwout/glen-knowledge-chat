#!/usr/bin/env python3
"""Monthly Family Plan charge cron entry point.

Posts to the web service's /api/cron/family-plan/charge endpoint, which runs the
charge scheduler inside the web container (where the persistent disk + chat_log.db
live). Render cron containers do NOT share the web service's disk, so the scheduler
cannot run here directly.

Required env vars on the cron service:
  WEB_URL       — base URL of the web service (no trailing slash)
                  default: https://glen-knowledge-chat.onrender.com
  CONSOLE_SECRET — shared secret matching the web service's CONSOLE_SECRET
                   (sent as X-Console-Key)
"""
import os
import sys
import json
import urllib.request
import urllib.error

WEB_URL = os.environ.get("WEB_URL", "https://glen-knowledge-chat.onrender.com").rstrip("/")
KEY = os.environ.get("CONSOLE_SECRET", "")

if not KEY:
    print("ERROR: CONSOLE_SECRET not set on cron service", flush=True)
    sys.exit(1)


def main():
    url = f"{WEB_URL}/api/cron/family-plan/charge"
    req = urllib.request.Request(
        url, method="POST", data=b"{}",
        headers={"X-Console-Key": KEY, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            print(f"HTTP {resp.status}: {body}", flush=True)
            data = json.loads(body)
            print(f"Family plan cron: charged={data.get('charged', 0)} "
                  f"failed={data.get('failed', 0)} cancelled={data.get('cancelled', 0)}",
                  flush=True)
    except urllib.error.HTTPError as e:
        print(f"HTTPError {e.code}: {e.read().decode('utf-8', errors='replace')}", flush=True)
        sys.exit(4)
    except urllib.error.URLError as e:
        print(f"URLError: {e}", flush=True)
        sys.exit(5)


if __name__ == "__main__":
    main()
