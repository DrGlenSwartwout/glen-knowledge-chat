#!/usr/bin/env python3
"""Daily People-hub sync chain (Render cron worker).

Curls three web-service endpoints in sequence (the work runs in the web
container, where the persistent disk + PB/Supabase/GHL creds live):

  1. /admin/sync-pb-tags           Practice Better -> people + GHL pb: tags
  2. /admin/sync-practitioner-tags Supabase practitioners -> people (tagged)
  3. /admin/sync-people-to-ghl     classify existing people + mirror opted-in -> GHL

Consolidated into one cron (was three) to stay under Render's cron-service
limit; ordered so tags are fresh before the GHL mirror. Each step is additive +
idempotent, so a failure in one does not corrupt the others.

Env (on the cron service):
  WEB_URL       base URL of the web service (default the onrender host)
  CRON_SECRET   shared secret (or CONSOLE_SECRET fallback); sent as X-Cron-Secret
"""
import json
import os
import sys
import urllib.error
import urllib.request

WEB_URL = os.environ.get("WEB_URL", "https://glen-knowledge-chat.onrender.com").rstrip("/")
CRON_SECRET = os.environ.get("CRON_SECRET") or os.environ.get("CONSOLE_SECRET", "")

if not CRON_SECRET:
    print("ERROR: CRON_SECRET (or CONSOLE_SECRET) not set on cron service", flush=True)
    sys.exit(1)

STEPS = [
    ("/admin/sync-pb-tags", "PB tags"),
    ("/admin/sync-practitioner-tags", "practitioner tags"),
    ("/admin/sync-people-to-ghl", "classify + GHL mirror"),
]


def _post(path):
    req = urllib.request.Request(
        f"{WEB_URL}{path}", method="POST", data=b"{}",
        headers={"X-Cron-Secret": CRON_SECRET, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                return False, f"HTTP {resp.status}: non-JSON {body[:160]}"
            return bool(data.get("ok")), json.dumps(data.get("summary") or data)[:300]
    except urllib.error.HTTPError as e:
        return False, f"HTTPError {e.code}: {e.read().decode('utf-8', 'replace')[:160]}"
    except urllib.error.URLError as e:
        return False, f"URLError: {e}"


def main():
    failures = 0
    for path, label in STEPS:
        ok, detail = _post(path)
        print(f"[{'ok ' if ok else 'FAIL'}] {label}: {detail}", flush=True)
        if not ok:
            failures += 1  # keep going; steps are independent + idempotent
    if failures:
        sys.exit(2)


if __name__ == "__main__":
    main()
