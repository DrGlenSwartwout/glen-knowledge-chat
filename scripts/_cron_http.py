#!/usr/bin/env python3
"""Shared HTTP helper for the stdlib-only Render cron entry scripts.

Every cron container just POSTs to the web service (which holds the persistent
disk + creds and does the actual work). The web service occasionally returns a
transient 502/503/504 (cold start, brief unavailability, mid-deploy) or refuses
the connection while spinning up; those clear on their own. Retry a couple of
times before letting the caller fail the whole cron run, so a blip doesn't fire
a "cron Exited with status 1" alert for nothing. A genuine sustained outage
(all attempts fail) still surfaces — the caller decides the exit code.

Stdlib-only, same as its callers. When a cron is run as `python scripts/foo.py`,
Python puts `scripts/` on sys.path[0], so `from _cron_http import post_with_retry`
resolves.
"""
import time
import urllib.request
import urllib.error

# 5xx statuses worth retrying (transient upstream/gateway blips, not the app's fault).
RETRY_STATUS = {502, 503, 504}
# Waits (seconds) before each retry; total attempts = len(backoff) + 1.
DEFAULT_BACKOFF = (5, 15)


def post_with_retry(url, headers, *, data=b"{}", timeout=300,
                    backoff=DEFAULT_BACKOFF, label="cron"):
    """POST `url` and return the response body as bytes.

    Retries on transient 5xx (RETRY_STATUS) and connection-level errors with the
    given backoff. On the final failed attempt the underlying urllib error is
    re-raised so the caller can log it and pick an exit code. HTTPError bodies
    are left unread, so the caller can still `e.read()` them.
    """
    max_attempts = len(backoff) + 1
    for attempt in range(1, max_attempts + 1):
        try:
            req = urllib.request.Request(
                url, data=data, method="POST", headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            # HTTPError is a subclass of URLError, so it must be caught first.
            if e.code in RETRY_STATUS and attempt < max_attempts:
                wait = backoff[attempt - 1]
                print(f"[{label}] HTTP {e.code} (transient); "
                      f"retry {attempt}/{max_attempts - 1} in {wait}s", flush=True)
                time.sleep(wait)
                continue
            raise
        except urllib.error.URLError as e:
            # Connection refused/reset while the web service spins up — also transient.
            if attempt < max_attempts:
                wait = backoff[attempt - 1]
                print(f"[{label}] connection error {e.reason!r} (transient); "
                      f"retry {attempt}/{max_attempts - 1} in {wait}s", flush=True)
                time.sleep(wait)
                continue
            raise
