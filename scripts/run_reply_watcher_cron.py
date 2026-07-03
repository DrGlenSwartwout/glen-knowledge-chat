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
import time
import urllib.request
import urllib.error

WEB_URL = os.environ.get("WEB_URL", "https://glen-knowledge-chat.onrender.com").rstrip("/")
CRON_SECRET = os.environ.get("CRON_SECRET") or os.environ.get("CONSOLE_SECRET", "")

# The web service occasionally returns a transient 502/503/504 (cold start, brief
# unavailability, mid-deploy) or refuses the connection while spinning up. Those blips
# clear on their own, so retry a couple times before failing the whole cron run —
# otherwise every blip fires a "cron Exited with status 1" alert for nothing.
RETRY_STATUS = {502, 503, 504}
BACKOFF_SECONDS = [5, 15]  # waits before retry #1 and retry #2 (len+1 = max attempts)

if not CRON_SECRET:
    print("ERROR: CRON_SECRET (or CONSOLE_SECRET) not set on cron service", flush=True)
    sys.exit(1)


def _post_once():
    url = f"{WEB_URL}/api/cron/reply-watch"
    req = urllib.request.Request(
        url, data=b"{}", method="POST",
        headers={"X-Cron-Secret": CRON_SECRET, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=240) as r:
        return json.load(r)


def main():
    max_attempts = len(BACKOFF_SECONDS) + 1
    for attempt in range(1, max_attempts + 1):
        try:
            body = _post_once()
        except urllib.error.HTTPError as e:
            detail = e.read()[:300]
            if e.code in RETRY_STATUS and attempt < max_attempts:
                wait = BACKOFF_SECONDS[attempt - 1]
                print(f"[reply-watcher-cron] HTTP {e.code} (transient); "
                      f"retry {attempt}/{max_attempts - 1} in {wait}s", flush=True)
                time.sleep(wait)
                continue
            print(f"[reply-watcher-cron] HTTP {e.code}: {detail!r}", flush=True)
            sys.exit(1)
        except urllib.error.URLError as e:
            # Connection-level failure (refused/reset while spinning up) — also transient.
            if attempt < max_attempts:
                wait = BACKOFF_SECONDS[attempt - 1]
                print(f"[reply-watcher-cron] connection error {e.reason!r} (transient); "
                      f"retry {attempt}/{max_attempts - 1} in {wait}s", flush=True)
                time.sleep(wait)
                continue
            print(f"[reply-watcher-cron] failed: {e!r}", flush=True)
            sys.exit(1)
        except Exception as e:  # noqa: BLE001
            print(f"[reply-watcher-cron] failed: {e!r}", flush=True)
            sys.exit(1)

        # Reached the web service and got a JSON body.
        if not body.get("ok"):
            print(f"[reply-watcher-cron] failed: {body.get('error')}", flush=True)
            sys.exit(2)
        processed = body.get("processed") or 0
        errored = body.get("errored") or 0
        suffix = f" (after {attempt - 1} retry)" if attempt > 1 else ""
        print(f"[reply-watcher-cron] processed={processed} "
              f"skipped_nonuser={body.get('skipped_nonuser')} "
              f"errored={errored}{suffix}", flush=True)
        # Systematic failure (e.g. token lost gmail.modify): nothing processed but the
        # batch errored. Surface as a failed job instead of a healthy-looking exit 0.
        if errored and not processed:
            sys.exit(3)
        return


if __name__ == "__main__":
    main()
