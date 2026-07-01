"""Provision + optionally email personal portals for a roster CSV.

Idempotent: /admin/portal/upsert keeps an existing portal's token, so re-runs
do not rotate links. Defaults to a no-op preview; --send is required to email.

Usage:
  python3 scripts/portal_provision_cohort.py roster.csv --dry-run
  python3 scripts/portal_provision_cohort.py roster.csv --send --limit 1   # smoke one
  python3 scripts/portal_provision_cohort.py roster.csv --send             # full cohort

Env: PUBLIC_BASE_URL, CONSOLE_SECRET.
"""
import argparse
import csv
import json
import os
import sys
import urllib.request

_SKIP_CONFIDENCE = {"unresolved", "low", ""}


def build_payloads(rows, send):
    """Pure: roster rows -> list of upsert bodies. Drops blank/invalid emails
    and low/unresolved-confidence matches so a bad guess never emails a stranger."""
    out = []
    for r in rows:
        email = (r.get("email") or "").strip().lower()
        conf = (r.get("confidence") or "").strip().lower()
        if not email or "@" not in email or conf in _SKIP_CONFIDENCE:
            continue
        out.append({"email": email, "name": (r.get("name") or "").strip(), "send": bool(send)})
    return out


def _post(base, key, body):
    req = urllib.request.Request(
        base.rstrip("/") + "/admin/portal/upsert",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "X-Console-Key": key},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("roster")
    ap.add_argument("--send", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--manifest", default="portal-manifest.csv")
    a = ap.parse_args(argv)
    base = os.environ["PUBLIC_BASE_URL"]
    key = os.environ["CONSOLE_SECRET"]
    with open(a.roster) as f:
        rows = list(csv.DictReader(f))
    payloads = build_payloads(rows, send=a.send)
    if a.limit:
        payloads = payloads[: a.limit]
    print(f"{len(payloads)} portals to provision (send={a.send}, dry_run={a.dry_run})")
    with open(a.manifest, "w", newline="") as mf:
        w = csv.writer(mf)
        w.writerow(["email", "name", "portal_url", "emailed", "status"])
        for p in payloads:
            if a.dry_run:
                print("DRY", p["email"], "send=" + str(p["send"]))
                w.writerow([p["email"], p["name"], "", "", "dry-run"])
                continue
            try:
                res = _post(base, key, p)
                url = base.rstrip("/") + (res.get("url") or "")
                w.writerow([p["email"], p["name"], url, res.get("emailed"), "ok"])
                print("OK ", p["email"], url, "emailed=" + str(res.get("emailed")))
            except Exception as e:  # log + continue; never abort the whole cohort
                w.writerow([p["email"], p["name"], "", "", f"error:{e}"])
                print("ERR", p["email"], e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
