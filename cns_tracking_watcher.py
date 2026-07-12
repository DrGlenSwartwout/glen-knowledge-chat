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

# Confidence tiers we trust enough to email the customer without a human glance.
#   "high"      — exactly one GHL contact whose name matches the ship-to EXACTLY
#   "harvested" — a single email precision-parsed from the order confirmation
# "medium" (single but fuzzy GHL match) and "low"/none stay drafts for review, so
# --auto-send never mails a guessed recipient.
AUTO_SEND_CONFIDENCES = ("high", "harvested")

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


def make_send_fn(service):
    """Send the tracking email outright (gmail.compose scope covers messages.send).
    Only ever called for AUTO_SEND_CONFIDENCES recipients with a non-blank To:."""
    def send_fn(to, subject, html, text):
        raw = build_raw(subject, html, text, to=to)
        res = service.users().messages().send(
            userId="me", body={"raw": raw}).execute()
        return res.get("id")
    return send_fn


def make_gmail_search_fn(service):
    """Return search(query) -> [{'sender','body'}] over the connected mailbox."""
    def _plain(payload):
        if payload.get("mimeType", "").startswith("text/"):
            data = payload.get("body", {}).get("data")
            if data:
                return base64.urlsafe_b64decode(data).decode("utf-8", "replace")
        for p in payload.get("parts", []) or []:
            t = _plain(p)
            if t:
                return t
        return ""
    def search(query):
        q = query  # caller passes the ship-to name; harvest widens as needed
        listing = service.users().messages().list(
            userId="me", q=q, maxResults=10).execute()
        out = []
        for m in listing.get("messages", []):
            full = service.users().messages().get(
                userId="me", id=m["id"], format="full").execute()
            headers = {h["name"].lower(): h["value"]
                       for h in full.get("payload", {}).get("headers", [])}
            out.append({"sender": headers.get("from", ""),
                        "body": _plain(full.get("payload", {}))})
        return out
    return search


def make_harvest_fn(gmail_search):
    from dashboard.order_harvest import harvest_buyer
    return lambda name: harvest_buyer(gmail_search, name)


def make_persist_contact():
    """Upsert the harvested buyer into GHL; onboard ONLY genuine new storefront
    buyers (source 'neworder' AND newly created). Returns {contact_id, onboarded}."""
    from app import ghl_upsert_contact, ghl_add_to_pipeline, ghl_enroll_workflow
    def persist(identity, ship_to_name):
        tag = ("source:gk-purchase" if identity.get("source") == "neworder"
               else "source:phone-email-order")
        cid, created, err = ghl_upsert_contact(
            identity["email"], first_name=identity.get("first") or "",
            last_name=identity.get("last") or "", phone=identity.get("phone") or "",
            source_tag=tag, extra_tags=["tracking-harvest"])
        onboarded = False
        if not err and cid and identity.get("source") == "neworder" and created:
            _opp, e_pipe = ghl_add_to_pipeline(cid, ship_to_name, identity["email"])
            _wf, e_wf = ghl_enroll_workflow(cid)
            onboarded = not e_wf          # workflow enroll is what actually onboards
            if e_pipe or e_wf:
                print(f"[tracking-harvest] onboarding partial for {identity['email']}: "
                      f"pipeline_err={e_pipe} workflow_err={e_wf}", flush=True)
        return {"contact_id": cid, "onboarded": onboarded}
    return persist


# ── Core decision logic (pure of Gmail/GHL; unit-tested) ─────────────────────

def handle_confirmation(html, msg_id, cx, find_contact, draft_fn,
                        harvest_fn=None, persist_contact=None, send_fn=None,
                        auto_send=False, dry_run=True):
    """Process one confirmation email's HTML. Returns a list of per-shipment
    result dicts. In dry-run, no drafts/sends/GHL writes/DB writes happen.

    When auto_send is on (and a send_fn is supplied), a recipient resolved at an
    AUTO_SEND_CONFIDENCES tier is emailed outright (status 'sent') instead of
    drafted. Fuzzy ('medium') and unresolved ('low'/none) recipients always draft,
    so an uncertain match is never auto-mailed."""
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
        ghl_contact_id = (match or {}).get("contact_id")

        # No confident GHL match: try a precision-safe harvest from order emails.
        harvested = None
        if to is None and harvest_fn is not None:
            harvested = harvest_fn(s["recipient_name"])
            if harvested and harvested.get("email"):
                to = harvested["email"]
                conf = "harvested"

        # Auto-send only high-confidence recipients; everything else drafts.
        send_eligible = (auto_send and to is not None
                         and conf in AUTO_SEND_CONFIDENCES)
        status = ("sent" if send_eligible else
                  "drafted" if to else "needs_review")
        email = build_tracking_email(s["tracking"], s["recipient_name"], resolved_email=to)

        draft_id = None
        onboarded = False
        if dry_run:
            verb = "would send" if send_eligible else "would draft"
            action = f"{verb} (harvested)" if conf == "harvested" else verb
        else:
            if harvested and conf == "harvested" and persist_contact is not None:
                pc = persist_contact(harvested, s["recipient_name"]) or {}
                ghl_contact_id = pc.get("contact_id") or ghl_contact_id
                onboarded = bool(pc.get("onboarded"))
            if send_eligible and send_fn is not None:
                draft_id = send_fn(to=to, subject=email["subject"],
                                   html=email["html"], text=email["text"])
                action = "sent"
            else:
                status = "drafted" if to else "needs_review"  # never sent
                draft_id = draft_fn(to=to, subject=email["subject"],
                                    html=email["html"], text=email["text"])
                action = "drafted"
            record_shipment(
                cx, tracking_number=s["tracking"], order_uuid=parsed["order_uuid"],
                recipient_name=s["recipient_name"], address_block=s["address_block"],
                resolved_email=to, match_confidence=conf,
                ghl_contact_id=ghl_contact_id,
                draft_id=draft_id, status=status, source_msg_id=msg_id)

        results.append({"tracking": s["tracking"], "recipient": s["recipient_name"],
                        "to": to or "(blank — needs review)", "confidence": conf,
                        "status": status, "action": action, "draft_id": draft_id,
                        "onboarded": onboarded})
    return results


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--live", action="store_true",
                    help="actually create Gmail drafts + record (default: dry-run)")
    ap.add_argument("--auto-send", action="store_true",
                    help="email high-confidence + harvested recipients outright "
                         "instead of drafting; fuzzy/unresolved still draft")
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
    send_fn = make_send_fn(svc) if (not dry_run and args.auto_send) else None
    harvest_fn = make_harvest_fn(make_gmail_search_fn(svc))
    persist_contact = None if dry_run else make_persist_contact()

    q = f"{GMAIL_QUERY} newer_than:{args.days}d"
    listing = svc.users().messages().list(
        userId="me", q=q, maxResults=args.max).execute()
    msg_ids = [m["id"] for m in listing.get("messages", [])]

    db = args.db or _db_path()
    mode = "DRY-RUN" if dry_run else ("LIVE+AUTO-SEND" if args.auto_send else "LIVE")
    print(f"{mode} | {len(msg_ids)} confirmation "
          f"email(s) in last {args.days}d | db={db}\n")

    totals = {}
    with sqlite3.connect(db) as cx:
        init_tracking_schema(cx)
        for mid in msg_ids:
            msg = svc.users().messages().get(
                userId="me", id=mid, format="full").execute()
            html = _extract_html(msg.get("payload", {}))
            for r in handle_confirmation(html, mid, cx, find_contact_by_name,
                                         draft_fn, harvest_fn=harvest_fn,
                                         persist_contact=persist_contact,
                                         send_fn=send_fn, auto_send=args.auto_send,
                                         dry_run=dry_run):
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
