"""Tracking — parse USPS Click-N-Ship payment-confirmation emails into shipments.

Glen ships via USPS Click-N-Ship. Every label triggers a "Payment Confirmation"
email from noreply-ecns@usps.com that lists, per label:

    Priority Mail®
    <the IMpb barcode digits, linked to USPS tracking>
    Scheduled delivery date: MM/DD/YYYY
    Shipped To:
      <recipient name>
      <street>
      <city ST zip-4 US>

USPS is *supposed* to email the recipient directly but frequently doesn't, so we
parse these confirmations ourselves, match each recipient to a GHL contact, and
draft Glen's "tracking number" email for review.

This module is the pure parsing core — no network, no DB — so it's unit-testable
and safe to run in the cron container (stdlib only, mirrors dashboard.shipping).

Public surface:
    normalize_tracking(impb)        — IMpb barcode digits -> 22-digit USPS tracking
    tracking_url(tracking)          — the customer-facing TrackConfirm link
    parse_cns_confirmation(html)    — confirmation HTML -> {order_uuid, shipments:[...]}

Each shipment dict:
    {tracking, recipient_name, street, city, state, zip, service,
     delivery_date, address_block}
"""

from __future__ import annotations

import re
import sqlite3
from html import escape, unescape
from typing import Dict, List, Optional


TRACK_URL = "https://tools.usps.com/go/TrackConfirmAction?tLabels={}"

# Glen's standard sign-off, mirroring the "tracking number" emails he sends today
# (drglenswartwout@gmail.com). Edit here if his signature changes. The tracking
# link is injected above this block by build_tracking_email().
SIGNATURE_HTML = (
    '<div><br></div>'
    '<div style="font-family:\'arial black\',sans-serif;font-size:large;color:#000">'
    'Dr. Glen Swartwout<br>'
    '(808) 217-9647<br>'
    'Healing Oasis<br>'
    '351 Wailuku Drive<br>'
    "Hilo, Kingdom of Hawai'i&nbsp; [96720]<br>"
    'Learn More with Our <b>Accelerated Self Healing&#8482;</b> Community at '
    '<a href="http://truly.vip/ASH">Truly.VIP/ASH</a><br><br>'
    'Video Channel: <a href="http://youtube.com/user/DoctorGlen">youtube.com/user/DoctorGlen</a><br>'
    'Author Page: <a href="http://amazon.com/default/e/B00AXTFZ26">amazon.com/default/e/B00AXTFZ26</a><br>'
    'LinkedIn: <a href="http://linkedin.com/in/drglen">linkedin.com/in/drglen</a><br>'
    'Fan Page: <a href="http://facebook.com/DrSwartwout">facebook.com/DrSwartwout</a><br>'
    'Remedies: <a href="https://remedymatch.com/">https://remedymatch.com/</a><br>'
    '&nbsp;&nbsp;&nbsp;&nbsp; Consultation: apply at bottom of page'
    '</div>'
)

SIGNATURE_TEXT = (
    "\n\n--\nDr. Glen Swartwout\n(808) 217-9647\nHealing Oasis\n351 Wailuku Drive\n"
    "Hilo, Kingdom of Hawai'i  [96720]\n"
    "Learn More with Our Accelerated Self Healing™ Community at Truly.VIP/ASH\n\n"
    "Video Channel: youtube.com/user/DoctorGlen\n"
    "Author Page: amazon.com/default/e/B00AXTFZ26\n"
    "LinkedIn: linkedin.com/in/drglen\nFan Page: facebook.com/DrSwartwout\n"
    "Remedies: https://remedymatch.com/\n     Consultation: apply at bottom of page\n"
)

EMAIL_SUBJECT = "tracking number"

# USPS retail/Priority tracking numbers (the part Glen pastes) are 22 digits and
# begin with a service banner — 9405/9400/9407/9270/9361/9205 etc. The full IMpb
# printed on the label/email prepends a "420" + destination ZIP routing block, so
# the human-facing tracking number is the trailing 22 digits.
_TRACK_22 = re.compile(r"(9[0-9]{21})")


def normalize_tracking(impb: str) -> Optional[str]:
    """Reduce an IMpb barcode string to the 22-digit USPS tracking number.

    Handles the three forms seen in the wild:
        '9405530109355381515251'                     -> itself (already 22)
        '4208522452499405530109355381515251'         -> '9405530109355381515251'
        '4205 8522 4524 9 9405 5301 0935 5381 5152 51'-> stripped, trailing 22

    Returns None if no plausible tracking number is present.
    """
    digits = re.sub(r"\D", "", impb or "")
    if not digits:
        return None
    # Prefer an explicit 9xxxxxxxxxxxxxxxxxxxxx run anchored to the end.
    m = _TRACK_22.search(digits)
    if m and digits.endswith(m.group(1)):
        return m.group(1)
    if len(digits) >= 22:
        return digits[-22:]
    return None


def tracking_url(tracking: str) -> str:
    return TRACK_URL.format(tracking)


# One shipment = one "item-contents-column" cell. Inside it, the anchor text is
# the (clean) IMpb digit string and the Shipped-To <p> lines carry the recipient.
_BLOCK_RE = re.compile(
    r'item-contents-column.*?</td>\s*<td class="item-total-column"',
    re.I | re.S,
)
_IMPB_RE = re.compile(r"<a[^>]*>\s*([\d ]{20,48})\s*</a>", re.I)
_DELIVERY_RE = re.compile(r"Scheduled delivery date:\s*([0-9/]+)", re.I)
_SERVICE_RE = re.compile(r"<p[^>]*>\s*([^<]*?Mail[^<]*?)</p>", re.I)
_SHIPPED_TO_RE = re.compile(r"Shipped To:\s*</p>(.*?)$", re.I | re.S)
_P_LINE_RE = re.compile(r'<p[^>]*class="[^"]*pt-5[^"]*"[^>]*>(.*?)</p>', re.I | re.S)
# city ST zip(-4) [US]
_CSZ_RE = re.compile(r"^(.*?)[, ]+([A-Z]{2})\s+(\d{5}(?:-\d{4})?)(?:\s+US)?\s*$")

_ORDER_UUID_RE = re.compile(
    r"(?:orderUUID=?|/history/orders/)([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12})",
    re.I,
)


def _clean(text: str) -> str:
    """Strip tags + collapse whitespace inside a captured fragment."""
    return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", "", text))).strip()


def _parse_block(block: str) -> Optional[dict]:
    impb_m = _IMPB_RE.search(block)
    if not impb_m:
        return None
    tracking = normalize_tracking(impb_m.group(1))
    if not tracking:
        return None

    lines = [_clean(p) for p in _P_LINE_RE.findall(block)]
    lines = [ln for ln in lines if ln]
    recipient_name = lines[0] if lines else ""
    addr_lines = lines[1:]

    city = state = zipcode = ""
    if addr_lines:
        csz = _CSZ_RE.match(addr_lines[-1])
        if csz:
            city, state, zipcode = csz.group(1).strip(), csz.group(2), csz.group(3)
    street = " ".join(addr_lines[:-1]) if len(addr_lines) > 1 else (
        addr_lines[0] if addr_lines and not _CSZ_RE.match(addr_lines[0]) else ""
    )

    svc_m = _SERVICE_RE.search(block)
    del_m = _DELIVERY_RE.search(block)
    return {
        "tracking": tracking,
        "recipient_name": recipient_name,
        "street": street,
        "city": city,
        "state": state,
        "zip": zipcode,
        "service": _clean(svc_m.group(1)) if svc_m else "",
        "delivery_date": del_m.group(1) if del_m else "",
        "address_block": " / ".join(lines[1:]),
    }


def parse_cns_confirmation(html: str) -> Dict[str, object]:
    """Parse a Click-N-Ship Payment Confirmation email body into shipments.

    Returns {order_uuid: str|None, shipments: [shipment_dict, ...]}.
    Order of shipments matches their order in the email.
    """
    if not html:
        return {"order_uuid": None, "shipments": []}

    uuid_m = _ORDER_UUID_RE.search(html)
    order_uuid = uuid_m.group(1).lower() if uuid_m else None

    shipments: List[dict] = []
    for block in _BLOCK_RE.findall(html):
        parsed = _parse_block(block)
        if parsed:
            shipments.append(parsed)
    return {"order_uuid": order_uuid, "shipments": shipments}


# ── Draft email (Glen's "tracking number" email, replicated) ─────────────────

def build_tracking_email(tracking: str, recipient_name: Optional[str] = None) -> dict:
    """Build the draft Glen reviews + sends. Mirrors his manual email exactly:
    a greeting (when we know the name), the tracking number as a live USPS link,
    then his standard sign-off.

    Returns {subject, html, text}. The watcher fills To: separately.
    """
    url = tracking_url(tracking)
    greeting_html = (
        f"<p>Hi {escape(recipient_name.split()[0])},</p>" if recipient_name else ""
    )
    greeting_text = (
        f"Hi {recipient_name.split()[0]},\n\n" if recipient_name else ""
    )
    html = (
        f"<div dir=\"ltr\">{greeting_html}"
        f"<p>Your order is on its way. Here is your USPS tracking number:</p>"
        f'<h3><a href="{url}">{tracking}</a></h3>'
        f"{SIGNATURE_HTML}</div>"
    )
    text = (
        f"{greeting_text}Your order is on its way. "
        f"Here is your USPS tracking number:\n\n{tracking}\n{url}"
        f"{SIGNATURE_TEXT}"
    )
    return {"subject": EMAIL_SUBJECT, "html": html, "text": text}


# ── Persistence: shipments table (idempotency + audit) ───────────────────────
#
# One row per tracking number. status:
#   'drafted'      — Gmail draft created, awaiting Glen/Rae review + send
#   'sent'         — Glen/Rae sent it (set later if we wire send-detection)
#   'needs_review' — parsed, but no confident GHL email match (To: left blank)
#
# tracking_number is UNIQUE so re-running the watcher over the same confirmation
# email is a no-op (we never double-draft).

def init_tracking_schema(cx: sqlite3.Connection) -> None:
    """Create the shipments table. Idempotent."""
    cx.execute("""
        CREATE TABLE IF NOT EXISTS shipments (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            tracking_number TEXT    NOT NULL UNIQUE,
            order_uuid      TEXT,
            recipient_name  TEXT,
            address_block   TEXT,
            resolved_email  TEXT,
            match_confidence TEXT,
            ghl_contact_id  TEXT,
            draft_id        TEXT,
            status          TEXT    NOT NULL DEFAULT 'needs_review',
            source_msg_id   TEXT,
            created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
            updated_at      TEXT
        )
    """)
    cx.execute(
        "CREATE INDEX IF NOT EXISTS idx_shipments_status ON shipments(status)"
    )
    cx.commit()


def shipment_exists(cx: sqlite3.Connection, tracking_number: str) -> bool:
    row = cx.execute(
        "SELECT 1 FROM shipments WHERE tracking_number = ? LIMIT 1",
        (tracking_number,),
    ).fetchone()
    return row is not None


def record_shipment(cx: sqlite3.Connection, **fields) -> Optional[int]:
    """Insert one shipment. No-op (returns None) if the tracking number already
    exists — this is what makes the watcher safe to re-run."""
    tn = fields.get("tracking_number")
    if not tn or shipment_exists(cx, tn):
        return None
    cols = [
        "tracking_number", "order_uuid", "recipient_name", "address_block",
        "resolved_email", "match_confidence", "ghl_contact_id", "draft_id",
        "status", "source_msg_id",
    ]
    vals = [fields.get(c) for c in cols]
    placeholders = ", ".join("?" for _ in cols)
    cur = cx.execute(
        f"INSERT INTO shipments ({', '.join(cols)}) VALUES ({placeholders})",
        vals,
    )
    cx.commit()
    return int(cur.lastrowid)
