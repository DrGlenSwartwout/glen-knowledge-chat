#!/usr/bin/env python3
"""Daily Personal email cron entry point.

Runs on Render's cron worker. Posts to the web service's
/cron/personal-send endpoint, which executes the orchestrator inside
the web container (where the persistent disk and chat_log.db live).

Render cron containers do NOT share the web service's persistent disk.
Calling the orchestrator directly here would crash on
sqlite3.connect("/data/chat_log.db") because /data is not mounted in
the cron container.

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
from datetime import datetime, timezone


WEB_URL = os.environ.get("WEB_URL", "https://glen-knowledge-chat.onrender.com").rstrip("/")
CRON_SECRET = os.environ.get("CRON_SECRET") or os.environ.get("CONSOLE_SECRET", "")
# Distinct from CRON_SECRET: used only for the piggybacked Pay It Forward invite,
# whose endpoint is gated by require_console_key (X-Console-Key), not X-Cron-Secret.
CONSOLE_SECRET = os.environ.get("CONSOLE_SECRET", "")

if not CRON_SECRET:
    print("ERROR: CRON_SECRET (or CONSOLE_SECRET) not set on cron service", flush=True)
    sys.exit(1)


def main():
    url = f"{WEB_URL}/cron/personal-send"
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
        with urllib.request.urlopen(req, timeout=300) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            print(f"HTTP {resp.status}: {body}", flush=True)
            try:
                data = json.loads(body)
                if data.get("ok"):
                    print(f"Personal email cron: sent {data.get('sent', '?')} email(s)", flush=True)
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


def invite_pif_gift_notes():
    """Also fire the Pay It Forward gift-note invites (recipients ~14-60 days post-redeem).
    Piggybacked on this (the one always-on Render cron) so the invite reliably runs daily.
    Independent + best-effort: never affects the personal-email send. 404 = feature dark
    (PAY_IT_FORWARD_ENABLED off) -> skip. The endpoint is idempotent (note_invited_at) +
    windowed, so daily runs never re-invite and never blast the historical backlog.
    Uses CONSOLE_SECRET (X-Console-Key) because the endpoint is require_console_key-gated."""
    if not CONSOLE_SECRET:
        print("[pif-gift-note-cron] CONSOLE_SECRET not set on cron service — skip", flush=True)
        return
    url = f"{WEB_URL}/api/cron/pif-gift-note-invites"
    req = urllib.request.Request(
        url, data=b"{}", method="POST",
        headers={"X-Console-Key": CONSOLE_SECRET, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=240) as r:
            body = json.load(r)
        print(f"[pif-gift-note-cron] invited {body.get('invited')}", flush=True)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print("[pif-gift-note-cron] endpoint 404 (PAY_IT_FORWARD_ENABLED off) — skip",
                  flush=True)
            return
        print(f"[pif-gift-note-cron] HTTP {e.code}: {e.read()[:300]!r}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[pif-gift-note-cron] failed: {e!r}", flush=True)


# --- Additional daily piggybacks ------------------------------------------------------
# These were each declared as their own cron in render.yaml but are NOT provisioned as
# dedicated Render cron services (and were silently not running anywhere). Folding them
# onto this one always-on daily cron — the same pattern as invite_pif_gift_notes — makes
# them fire daily without depending on a Mac being awake. Each call is independent and
# best-effort: a failure here never affects the personal-email send or the other jobs.

def _piggyback_post(label, path, header, secret, *, timeout=300):
    """Best-effort POST to a web-service cron/admin endpoint. Never raises.
    404 = the endpoint/feature is dark -> skip quietly. The web service holds the
    persistent disk + creds, so (as with every cron here) the work happens there."""
    if not secret:
        print(f"[{label}] secret not set on cron service — skip", flush=True)
        return
    req = urllib.request.Request(
        f"{WEB_URL}{path}", data=b"{}", method="POST",
        headers={header: secret, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode("utf-8", errors="replace")
        print(f"[{label}] ok: {body[:300]}", flush=True)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"[{label}] endpoint 404 (feature off) — skip", flush=True)
            return
        print(f"[{label}] HTTP {e.code}: {e.read()[:300]!r}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[{label}] failed: {e!r}", flush=True)


def run_daily_piggybacks():
    """Daily jobs folded onto this always-on cron. All best-effort.
      - testimonial-invite scan (require_console_key -> X-Console-Key)
      - People-hub + subscription sync chain, ordered so tags are fresh before the GHL
        mirror; last step charges due subscriptions (X-Cron-Secret; each step idempotent)
      - USPS Flat Rate rate-check, weekly: only on Mondays (UTC)
    """
    _piggyback_post("testimonial-invites-cron",
                    "/api/console/testimonial-invites/scan?days=3&gmail_limit=200",
                    "X-Console-Key", CONSOLE_SECRET)
    for path in ("/admin/sync-pb-tags", "/admin/sync-practitioner-tags",
                 "/admin/sync-people-to-ghl", "/api/cron/charge-subscriptions"):
        _piggyback_post(f"pb-sync-chain {path}", path, "X-Cron-Secret", CRON_SECRET, timeout=600)
    if datetime.now(timezone.utc).weekday() == 0:  # Monday
        _piggyback_post("usps-rate-check", "/cron/usps-rate-check", "X-Cron-Secret", CRON_SECRET)
    else:
        print("[usps-rate-check] not Monday (UTC) — skip", flush=True)


if __name__ == "__main__":
    # `finally` guarantees the piggybacked jobs fire even if the personal-email send sys.exit()s.
    try:
        main()
    finally:
        invite_pif_gift_notes()
        run_daily_piggybacks()
