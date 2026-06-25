#!/usr/bin/env python3
"""Cron entrypoint: reconcile QBO-side invoice payments onto the orders board.

Stdlib-only (no deps): this runs in the cron container and just curls the web
service, which does the work where chat_log.db + the live QBO token live.

Env:
  WEB_URL       — base URL of the web service (no trailing slash)
  CRON_SECRET   — shared secret matching the web service (or CONSOLE_SECRET
                  fallback). Sent as X-Cron-Secret.
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
    url = f"{WEB_URL}/api/console/reconcile-qbo"
    req = urllib.request.Request(
        url, data=b"", method="POST",
        headers={"X-Cron-Secret": CRON_SECRET, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            body = json.load(r)
        print(f"[qbo-reconcile-cron] reconciled {body.get('count')} order(s): "
              f"{body.get('reconciled')}", flush=True)
    except urllib.error.HTTPError as e:
        print(f"[qbo-reconcile-cron] HTTP {e.code}: {e.read()[:300]!r}", flush=True)
        sys.exit(1)
    except Exception as e:
        print(f"[qbo-reconcile-cron] failed: {e!r}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
