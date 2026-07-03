#!/usr/bin/env python3
"""Daily Intelligence-briefings cron entry point.

Runs on Render's cron worker. Posts to /cron/regenerate-briefings on the web
service, which gathers live stats, asks Claude to compose all 5 briefings,
and writes them to /data/intelligence/. The cron container itself stays
stdlib-only — all heavy lifting (Anthropic call, persistent disk write)
happens inside the web container where the disk and ANTHROPIC_API_KEY live.

Required env vars on the cron service:
  WEB_URL       — base URL of the web service (no trailing slash)
                  default: https://glen-knowledge-chat.onrender.com
  CRON_SECRET   — shared secret matching the web service's CRON_SECRET
                  (or CONSOLE_SECRET fallback). Sent as X-Cron-Secret.
"""
import os
import sys
import json
import urllib.error

from _cron_http import post_with_retry


WEB_URL = os.environ.get("WEB_URL", "https://glen-knowledge-chat.onrender.com").rstrip("/")
CRON_SECRET = os.environ.get("CRON_SECRET") or os.environ.get("CONSOLE_SECRET", "")

if not CRON_SECRET:
    print("ERROR: CRON_SECRET (or CONSOLE_SECRET) not set on cron service", flush=True)
    sys.exit(1)


def main():
    url = f"{WEB_URL}/cron/regenerate-briefings"
    headers = {"X-Cron-Secret": CRON_SECRET, "Content-Type": "application/json"}
    # Transient 5xx / connection blips are retried inside post_with_retry.
    try:
        body = post_with_retry(url, headers, timeout=600,
                               label="briefings-cron").decode("utf-8", errors="replace")
        print(f"HTTP 200: {body}", flush=True)
        try:
            data = json.loads(body)
            if not data.get("ok"):
                sys.exit(1)
        except json.JSONDecodeError:
            sys.exit(1)
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read().decode('utf-8', errors='replace')}", flush=True)
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"URL error: {e}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
