#!/usr/bin/env python3
"""Weekly USPS Flat Rate rate-check cron entry point.

Runs on Render's cron worker. Posts to the web service's
/cron/usps-rate-check endpoint, which scrapes USPS retail prices and
stages any changes as pending updates for Glen to confirm at
/admin/shipping.

Same cross-container pattern as run_personal_email_cron.py — the
persistent disk lives on the web service, not the cron container.

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
    url = f"{WEB_URL}/cron/usps-rate-check"
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
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            print(f"HTTP {resp.status}: {body}", flush=True)
            try:
                data = json.loads(body)
                if data.get("ok"):
                    summary = data.get("summary", {})
                    proposed = summary.get("proposed", [])
                    errors = summary.get("errors", [])
                    print(
                        f"USPS rate check: {len(proposed)} proposed, "
                        f"{len(summary.get('unchanged', []))} unchanged, "
                        f"{len(errors)} error(s)",
                        flush=True,
                    )
                    if errors:
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
