#!/usr/bin/env python3
"""Reply-watcher cron entry. Runs on Render every 15 minutes.

Stdlib-only (no deps): runs in the cron container and just curls the web service's
/api/cron/reply-watch endpoint. The watcher needs the Gmail token on the persistent
disk (/data/google-token.json) + chat_log.db, which live on the web service — NOT in
the cron container — so the work must happen there. Same cross-container pattern as
run_personal_email_cron.py.

Env (on the cron service):
  WEB_URL       base URL of the web service (default the onrender host)
  CRON_SECRET   shared secret (or CONSOLE_SECRET fallback); sent as X-Cron-Secret
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
    url = f"{WEB_URL}/api/cron/reply-watch"
    req = urllib.request.Request(
        url, data=b"{}", method="POST",
        headers={"X-Cron-Secret": CRON_SECRET, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=240) as r:
            body = json.load(r)
        if not body.get("ok"):
            print(f"[reply-watcher-cron] failed: {body.get('error')}", flush=True)
            sys.exit(2)
        print(f"[reply-watcher-cron] processed={body.get('processed')} "
              f"skipped_nonuser={body.get('skipped_nonuser')} "
              f"errored={body.get('errored')}", flush=True)
    except urllib.error.HTTPError as e:
        print(f"[reply-watcher-cron] HTTP {e.code}: {e.read()[:300]!r}", flush=True)
        sys.exit(1)
    except Exception as e:  # noqa: BLE001
        print(f"[reply-watcher-cron] failed: {e!r}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
