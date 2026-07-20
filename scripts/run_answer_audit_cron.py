#!/usr/bin/env python3
"""Weekly answer-audit cron entry. Runs on Render Mondays.

Stdlib-only: runs in the cron container and just curls the web service's
/api/cron/answer-audit endpoint. The audit must run IN the web service — it
needs the catalog and the Gmail token (to email Dr. Glen on findings), which
live on the web disk, not the cron container's. Same cross-container pattern as
run_reply_watcher_cron.py.

Env (on the cron service):
  WEB_URL       base URL of the web service (default the onrender host)
  CRON_SECRET   shared secret (or CONSOLE_SECRET fallback); sent as X-Cron-Secret
"""
import json
import os
import sys
import urllib.error

from _cron_http import post_with_retry

WEB_URL = os.environ.get("WEB_URL", "https://glen-knowledge-chat.onrender.com").rstrip("/")
CRON_SECRET = os.environ.get("CRON_SECRET") or os.environ.get("CONSOLE_SECRET", "")

if not CRON_SECRET:
    print("ERROR: CRON_SECRET (or CONSOLE_SECRET) not set on cron service", flush=True)
    sys.exit(1)


def main():
    url = f"{WEB_URL}/api/cron/answer-audit"
    headers = {"X-Cron-Secret": CRON_SECRET, "Content-Type": "application/json"}
    # The audit asks ~11 questions of the live bot; give it room.
    try:
        body = json.loads(post_with_retry(url, headers, timeout=300,
                                           label="answer-audit-cron"))
    except urllib.error.HTTPError as e:
        print(f"[answer-audit-cron] HTTP {e.code}: {e.read()[:300]!r}", flush=True)
        sys.exit(1)
    except Exception as e:  # noqa: BLE001
        print(f"[answer-audit-cron] failed: {e!r}", flush=True)
        sys.exit(1)
    print(f"[answer-audit-cron] {json.dumps(body)}", flush=True)
    # A flagged run is informational, not a failure — Glen was emailed. Exit 0
    # so Render doesn't mark the cron red for doing its job.


if __name__ == "__main__":
    main()
