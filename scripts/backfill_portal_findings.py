#!/usr/bin/env python3
"""Backfill content.findings on portals published before findings were baked in
at publish time (#661). Surgical + guarded: only emails that already have a portal
AND whose scan yields findings are patched, via the findings-only endpoint. No
email is ever sent; no portal is ever created. Dry-run by default; --apply executes.

Env: CONSOLE_SECRET (console key), PORTAL_PUBLISH_BASE_URL or --base (prod base),
E4L_DB (defaults to ~/AI-Training/e4l.db via dashboard.biofield_e4l).

Run:  python3 scripts/backfill_portal_findings.py            # dry-run
      python3 scripts/backfill_portal_findings.py --apply    # execute
"""
import argparse
import json
import os
import sqlite3
import sys
import urllib.parse
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))  # repo root, for `dashboard`
from dashboard.biofield_e4l import scan_context, findings_for_scan_date  # noqa: E402

INTAKE_DB = os.environ.get(
    "BIOFIELD_DB", os.path.join(os.path.dirname(_HERE), "chat_log.db"))


def _trim(raw):
    return [{"code": f.get("code", ""), "name": f.get("name", ""),
             "description": f.get("description", ""), "rank": f.get("rank")}
            for f in (raw or [])]


def plan_backfill(portal_emails, intake_emails, report_dates_of, findings_of):
    """Pure planner. Returns (patches, skips).
    patches: [{email, scan_date (None=portal record), findings}]. skips: [{email, reason}]."""
    patches, skips = [], []
    for email in intake_emails:
        e = (email or "").strip().lower()
        if e not in portal_emails:
            skips.append({"email": e, "reason": "no existing portal (would create/dup)"})
            continue
        dates = report_dates_of(e) or []
        entries = []
        if dates:
            for d in dates:
                f = findings_of(e, d) or []
                if f:
                    entries.append({"email": e, "scan_date": d, "findings": f})
        else:
            f = findings_of(e, None) or []
            if f:
                entries.append({"email": e, "scan_date": None, "findings": f})
        if not entries:
            skips.append({"email": e, "reason": "no findings computed"})
            continue
        patches.extend(entries)
    return patches, skips


def _get(url, key):
    req = urllib.request.Request(url, headers={"X-Console-Key": key})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def _post(url, key, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"X-Console-Key": key,
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="execute (default: dry-run)")
    ap.add_argument("--base", default=os.environ.get("PORTAL_PUBLISH_BASE_URL", ""))
    args = ap.parse_args(argv)
    key = os.environ.get("CONSOLE_SECRET", "")
    base = args.base.rstrip("/")
    if not key or not base:
        print("need CONSOLE_SECRET and --base/PORTAL_PUBLISH_BASE_URL", file=sys.stderr)
        return 2

    portals = _get(f"{base}/api/console/portal-links", key).get("portals", [])
    portal_emails = {(p.get("email") or "").strip().lower()
                     for p in portals if p.get("has_token")}

    cx = sqlite3.connect(f"file:{INTAKE_DB}?mode=ro", uri=True)
    intake_emails = sorted({(r[0] or "").strip().lower()
                            for r in cx.execute("SELECT email FROM biofield_auth_tests")
                            if (r[0] or "").strip()})
    cx.close()

    # token cache so report_dates_of can read /api/portal/<token> scan_dates
    _tok = {}
    def token_for(email):
        if email not in _tok:
            j = _get(f"{base}/api/console/portal-link?email={urllib.parse.quote(email)}", key)
            link = j.get("link") or ""
            _tok[email] = link.rsplit("/portal/", 1)[-1] if "/portal/" in link else ""
        return _tok[email]

    def report_dates_of(email):
        tok = token_for(email)
        if not tok:
            return []
        d = _get(f"{base}/api/portal/{tok}", key)
        return d.get("scan_dates") or []

    def findings_of(email, scan_date):
        # A dated report row must get THAT scan's findings -- scan_context() always
        # returns the latest scan, so use the date-specific lookup. The portal-record
        # patch (scan_date None) legitimately wants the latest scan.
        if scan_date:
            return _trim(findings_for_scan_date(email, scan_date))
        from datetime import date
        try:
            return _trim(scan_context(email, date.today().isoformat()).get("findings") or [])
        except Exception:
            return []

    patches, skips = plan_backfill(portal_emails, intake_emails, report_dates_of, findings_of)

    for s in skips:
        print(f"SKIP {s['email']}: {s['reason']}")
    for p in patches:
        tgt = f"report {p['scan_date']}" if p["scan_date"] else "portal record"
        print(f"{'APPLY' if args.apply else 'DRY '} {p['email']} -> {tgt} "
              f"({len(p['findings'])} findings)")
        if args.apply:
            body = {"email": p["email"], "findings": p["findings"]}
            if p["scan_date"]:
                body["scan_date"] = p["scan_date"]
            res = _post(f"{base}/api/console/portal/backfill-findings", key, body)
            print(f"      -> {res}")
    print(f"\n{len(patches)} patch(es), {len(skips)} skip(s). "
          f"{'APPLIED' if args.apply else 'DRY-RUN (use --apply)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
