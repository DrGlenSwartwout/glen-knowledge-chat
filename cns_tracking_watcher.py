#!/usr/bin/env python3
"""CNS tracking watcher — draft customer tracking emails from USPS Click-N-Ship.

Flow (mirrors the E4L Gmail watcher):
  1. Search Glen's mailbox for unprocessed "USPS - Click-N-Ship(R) Payment
     Confirmation" emails from noreply-ecns@usps.com.
  2. Parse each into shipments (tracking #, recipient, address).
  3. Match each recipient name -> GHL contact email (with a confidence score).
  4. Build Glen's "tracking number" email and CREATE IT AS A DRAFT (To: prefilled
     for high/medium-confidence matches, blank + flagged needs_review otherwise).
  5. Record to the shipments table so re-runs never double-draft.

DEFAULT IS DRY-RUN: prints what it would do and mutates nothing (no drafts, no
DB writes). Pass --live to actually create drafts.

Run with GHL creds for real matching:
    doppler run -- python3 cns_tracking_watcher.py            # dry-run
    doppler run -- python3 cns_tracking_watcher.py --live     # create drafts

Auth: needs a Gmail token with gmail.readonly + gmail.compose at
~/.config/google/token.json (mint via scripts/gmail_tracking_auth.py).
"""

from __future__ import annotations

import argparse
import base64
import os
import sqlite3
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

from dashboard.tracking import (
    parse_cns_confirmation,
    build_tracking_email,
    init_tracking_schema,
    record_shipment,
    shipment_exists,
)

GMAIL_QUERY = 'from:noreply-ecns@usps.com subject:"Payment Confirmation"'
TOKEN_PATH = Path(os.environ.get(
    "GMAIL_TOKEN_PATH", str(Path.home() / ".config" / "google" / "token.json")))
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
]


def _db_path():
    base = os.environ.get("DATA_DIR", str(Path(__file__).resolve().parent))
    return str(Path(base) / "chat_log.db")


# ── Gmail plumbing (thin; the testable logic is handle_confirmation) ─────────

def gmail_service(token_path=TOKEN_PATH):
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        Path(token_path).write_text(creds.to_json())
    return build("gmail", "v1", credentials=creds)


def _extract_html(payload):
    """Walk a Gmail message payload and return the first text/html body."""
    if payload.get("mimeType") == "text/html":
        data = payload.get("body", {}).get("data")
        if data:
            return base64.urlsafe_b64decode(data).decode("utf-8", "replace")
    for part in payload.get("parts", []) or []:
        html = _extract_html(part)
        if html:
            return html
    return ""


def build_raw(subject, html, text, to=None):
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    if to:
        msg["To"] = to
    msg.attach(MIMEText(text, "plain", "utf-8"))
    msg.attach(MIMEText(html, "html", "utf-8"))
    return base64.urlsafe_b64encode(msg.as_bytes()).decode()


def make_draft_fn(service):
    def draft_fn(to, subject, html, text):
        raw = build_raw(subject, html, text, to=to)
        res = service.users().drafts().create(
            userId="me", body={"message": {"raw": raw}}).execute()
        return res.get("id")
    return draft_fn


# ── Core decision logic (pure of Gmail/GHL; unit-tested) ─────────────────────

def handle_confirmation(html, msg_id, cx, find_contact, draft_fn, dry_run=True):
    """Process one confirmation email's HTML. Returns a list of per-shipment
    result dicts. In dry-run, no drafts are created and nothing is recorded."""
    parsed = parse_cns_confirmation(html)
    results = []
    for s in parsed["shipments"]:
        if shipment_exists(cx, s["tracking"]):
            results.append({"tracking": s["tracking"],
                            "recipient": s["recipient_name"],
                            "action": "skipped (already processed)"})
            continue

        match = find_contact(s["recipient_name"])
        conf = match["confidence"] if match else "none"
        to = match["email"] if (match and conf in ("high", "medium")) else None
        status = "drafted" if to else "needs_review"
        email = build_tracking_email(s["tracking"], s["recipient_name"], resolved_email=to)

        draft_id = None
        if dry_run:
            action = "would draft"
        else:
            draft_id = draft_fn(to=to, subject=email["subject"],
                                html=email["html"], text=email["text"])
            record_shipment(
                cx, tracking_number=s["tracking"], order_uuid=parsed["order_uuid"],
                recipient_name=s["recipient_name"], address_block=s["address_block"],
                resolved_email=to, match_confidence=conf,
                ghl_contact_id=(match or {}).get("contact_id"),
                draft_id=draft_id, status=status, source_msg_id=msg_id)
            action = "drafted"

        results.append({"tracking": s["tracking"], "recipient": s["recipient_name"],
                        "to": to or "(blank — needs review)", "confidence": conf,
                        "status": status, "action": action, "draft_id": draft_id})
    return results


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--live", action="store_true",
                    help="actually create Gmail drafts + record (default: dry-run)")
    ap.add_argument("--days", type=int, default=14,
                    help="how many days back to scan (default 14)")
    ap.add_argument("--max", type=int, default=25, help="max emails to scan")
    ap.add_argument("--db", default=None, help="override chat_log.db path")
    args = ap.parse_args()
    dry_run = not args.live

    try:
        from dashboard.ghl import find_contact_by_name
    except Exception:
        find_contact_by_name = lambda name: None  # noqa: E731

    svc = gmail_service()
    draft_fn = make_draft_fn(svc) if not dry_run else (lambda **k: None)

    q = f"{GMAIL_QUERY} newer_than:{args.days}d"
    listing = svc.users().messages().list(
        userId="me", q=q, maxResults=args.max).execute()
    msg_ids = [m["id"] for m in listing.get("messages", [])]

    db = args.db or _db_path()
    print(f"{'DRY-RUN' if dry_run else 'LIVE'} | {len(msg_ids)} confirmation "
          f"email(s) in last {args.days}d | db={db}\n")

    totals = {"would draft": 0, "drafted": 0, "skipped (already processed)": 0}
    with sqlite3.connect(db) as cx:
        init_tracking_schema(cx)
        for mid in msg_ids:
            msg = svc.users().messages().get(
                userId="me", id=mid, format="full").execute()
            html = _extract_html(msg.get("payload", {}))
            for r in handle_confirmation(html, mid, cx, find_contact_by_name,
                                         draft_fn, dry_run=dry_run):
                totals[r["action"]] = totals.get(r["action"], 0) + 1
                line = (f"  [{r.get('confidence','-'):>6}] {r['recipient']:<22} "
                        f"{r['tracking']}  -> {r.get('to','')}  ({r['action']})")
                print(line)

    print("\nSummary:", ", ".join(f"{k}: {v}" for k, v in totals.items() if v))
    if dry_run:
        print("Dry-run only — no drafts created, nothing recorded. "
              "Re-run with --live when the matches look right.")


if __name__ == "__main__":
    main()
