"""Staged GHL rollout sender — drip portal links to the tiered roster, wave by wave.

Reads a TIER-SORTED roster CSV and a running send-log; each run enrolls the next
--wave-size unsent recipients via POST /admin/portal/rollout-enroll (which mints a
stable link, sets the GHL portal-URL custom field, tags portal-invite, and enrolls
the portal-invite workflow that emails it). Because the roster is tier-sorted, waves
drain Tier 1 (PB members) first, then E4L-no-scan, then E4L-with-scan, then others.
Idempotent across runs via the send-log. Env: PUBLIC_BASE_URL, CONSOLE_SECRET.

Usage:
  python3 scripts/portal_ghl_rollout.py --roster rollout-roster-tiered.csv --dry-run
  python3 scripts/portal_ghl_rollout.py --roster rollout-roster-tiered.csv --send --wave-size 150
"""
import argparse
import csv
import json
import os
import sys
import urllib.request

_SKIP_CONFIDENCE = {"unresolved", "low", ""}


def next_wave(rows, sent_emails, wave_size):
    """The next `wave_size` recipients, in roster (tier) order, skipping already-sent,
    blank/invalid, and low-confidence rows. Pure — no I/O."""
    sent = {(e or "").strip().lower() for e in sent_emails}
    out = []
    for r in rows:
        email = (r.get("email") or "").strip().lower()
        conf = (r.get("confidence") or "").strip().lower()
        if not email or "@" not in email or conf in _SKIP_CONFIDENCE:
            continue
        if email in sent:
            continue
        out.append({"email": email, "name": (r.get("name") or "").strip()})
        if len(out) >= wave_size:
            break
    return out


def _load_sent(path):
    """Emails that SUCCEEDED (status == ok). Errored rows are intentionally NOT
    counted as sent, so a transient GHL failure is retried on the next wave."""
    if not path or not os.path.exists(path):
        return set()
    with open(path) as f:
        return {(row.get("email") or "").strip().lower()
                for row in csv.DictReader(f)
                if (row.get("status") or "").strip().lower() == "ok"}


def _post(base, key, email, name, mode):
    req = urllib.request.Request(
        base.rstrip("/") + "/admin/portal/rollout-enroll",
        data=json.dumps({"email": email, "name": name, "mode": mode}).encode(),
        headers={"Content-Type": "application/json", "X-Console-Key": key},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--roster", required=True)
    ap.add_argument("--send-log", default="portal-send-log.csv")
    ap.add_argument("--wave-size", type=int, default=150)
    ap.add_argument("--mode", choices=["link", "claim"], default="link",
                    help="link = pre-mint (warm tiers); claim = mint-on-click (cold Tier 4)")
    ap.add_argument("--send", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    base = os.environ["PUBLIC_BASE_URL"]
    key = os.environ["CONSOLE_SECRET"]
    with open(a.roster) as f:
        rows = list(csv.DictReader(f))
    sent = _load_sent(a.send_log)
    wave = next_wave(rows, sent, a.wave_size)
    tier_of = {(r.get("email") or "").strip().lower(): r.get("tier", "") for r in rows}
    print(f"{len(wave)} to enroll this wave (mode={a.mode}; already sent {len(sent)} of {len(rows)}; send={a.send}, dry_run={a.dry_run})")
    new = not os.path.exists(a.send_log)
    with open(a.send_log, "a", newline="") as lf:
        w = csv.writer(lf)
        if new:
            w.writerow(["email", "tier", "enrolled", "status"])
        for p in wave:
            if a.dry_run:  # do NOT write the send-log on a dry run — nobody is marked sent
                print("DRY", p["email"])
                continue
            try:
                res = _post(base, key, p["email"], p["name"], a.mode)
                status = "ok" if res.get("ok") else ("err:" + str(res.get("error")))
                w.writerow([p["email"], tier_of.get(p["email"], ""), res.get("enrolled"), status])
                lf.flush()
                print("OK " if res.get("ok") else "ERR", p["email"], "enrolled=" + str(res.get("enrolled")), res.get("error") or "")
            except Exception as e:  # log + continue; never abort the whole wave
                w.writerow([p["email"], tier_of.get(p["email"], ""), False, f"exc:{e}"])
                lf.flush()
                print("EXC", p["email"], e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
