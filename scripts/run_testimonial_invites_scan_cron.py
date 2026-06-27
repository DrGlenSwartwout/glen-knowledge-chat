#!/usr/bin/env python3
"""Cron entrypoint: daily testimonial-invite scan.

Stdlib-only: runs in the cron container and just curls the web service, which holds the
Gmail token (/data/google-token.json) + chat_log.db and does the work (read recent client
comms/email -> classify positive results -> add review candidates). A small rolling window
keeps each run fast; should_skip de-dups, so daily runs accumulate the backlog over time.

Env:
  WEB_URL         — base URL of the web service (no trailing slash)
  CONSOLE_SECRET  — console key (or CRON_SECRET fallback); sent as X-Console-Key
"""
import os
import sys
import json
import urllib.request
import urllib.error

WEB_URL = os.environ.get("WEB_URL", "https://glen-knowledge-chat.onrender.com").rstrip("/")
KEY = os.environ.get("CONSOLE_SECRET") or os.environ.get("CRON_SECRET", "")

if not KEY:
    print("ERROR: CONSOLE_SECRET (or CRON_SECRET) not set on cron service", flush=True)
    sys.exit(1)


def main():
    # 3-day rolling window (daily run + overlap); 200-email cap keeps it within request time.
    url = f"{WEB_URL}/api/console/testimonial-invites/scan?days=3&gmail_limit=200"
    req = urllib.request.Request(
        url, data=b"{}", method="POST",
        headers={"X-Console-Key": KEY, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=240) as r:
            body = json.load(r)
        print(f"[testimonial-invites-cron] added {len(body.get('candidates', []))} candidate(s); "
              f"scanned {body.get('scanned')} (gmail clients {body.get('gmail_clients')})", flush=True)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print("[testimonial-invites-cron] endpoint 404 (TESTIMONIAL_INVITES_ENABLED off) — skip",
                  flush=True)
            return
        print(f"[testimonial-invites-cron] HTTP {e.code}: {e.read()[:300]!r}", flush=True)
        sys.exit(1)
    except Exception as e:  # noqa: BLE001
        print(f"[testimonial-invites-cron] failed: {e!r}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
