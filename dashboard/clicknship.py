"""USPS Click-N-Ship confirmation email parser + Gmail IMAP harvester.

Stage 1 (read-only, no money) of the USPS shipping automation project. See
docs/superpowers/specs/2026-07-18-usps-shipping-automation-design.md.

USPS Click-N-Ship emails a "Payment Confirmation" to Rae's Gmail
(suerae1111@gmail.com) from noreply-ecns@usps.com every time a shipping
label is bought. Each confirmation is a single HTML email that can contain
MULTIPLE packages (one per item row) — each with its own tracking number,
service, optional per-item scheduled delivery date, and recipient block.

This module is intentionally standalone: stdlib only (imaplib, email, re,
html, datetime). No app/DB imports — order-matching, membership logic, and
persistence are later stages, not this one.

Credentials come from env only:
    GMAIL_IMAP_USER            e.g. suerae1111@gmail.com
    GMAIL_IMAP_APP_PASSWORD    Gmail app password (Google displays it with
                                spaces for readability; strip them before use)
"""

from __future__ import annotations

import argparse
import email
import html as html_mod
import imaplib
import os
import re
from email.header import decode_header

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

# One package/item block: the item-contents-column td (service, tracking,
# optional scheduled delivery, recipient) followed — with arbitrary markup
# in between (the inner table/tr/td closing tags) — by the sibling
# item-total-column td that carries the price. Both non-greedy spans are
# needed because item-contents-column itself contains a nested, unnamed
# <td> that closes (with its own </td>) before item-contents-column's own
# closing tag.
_ITEM_RE = re.compile(
    r'<td class="item-contents-column">(.*?)</td>'
    r'.*?'
    r'<td class="item-total-column">\s*<p class="price-col-p">\$([\d,]+\.\d{2})</p>',
    re.DOTALL,
)

_BOLD_P_RE = re.compile(r'<p class="bold">\s*([^<]*?)\s*</p>')
_TRACK_LINK_RE = re.compile(
    r'<a\s+href=["\']?(https://tools\.usps\.com/go/TrackConfirmAction[^"\'>\s]*)["\']?\s*>\s*([^<]+?)\s*</a>',
    re.IGNORECASE,
)
_SCHEDULED_RE = re.compile(r'Scheduled delivery date:\s*(\d{2}/\d{2}/\d{4})')
_SHIPPED_TO_RE = re.compile(r'Shipped To:\s*</p>(.*)', re.DOTALL)
_PT5_RE = re.compile(r'<p class="pt-5">\s*([^<]*?)\s*</p>')

_ORDER_UUID_RE = re.compile(r'orderUUID=([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})')
_PLACED_ON_RE = re.compile(
    r'Placed on:.*?<p>\s*([A-Za-z]{3,9}\.?\s+\d{1,2},\s*\d{4})\s*</p>',
    re.DOTALL,
)

# USPS tracking numbers commonly begin with one of these prefixes once the
# "420" + 5-digit destination-ZIP routing prefix (used on some Click-N-Ship
# labels/IMbs) is stripped off.
_USPS_TRACKING_PREFIXES = ("9400", "9405", "9407", "9408", "92", "93", "94")


def _decode_header_value(raw) -> str:
    if not raw:
        return ""
    out = []
    for part, enc in decode_header(raw):
        if isinstance(part, bytes):
            out.append(part.decode(enc or "utf-8", "replace"))
        else:
            out.append(part)
    return "".join(out)


def _extract_html_body(msg: "email.message.Message") -> str:
    """Return the text/html part of a parsed email.Message, or ''."""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/html":
                try:
                    payload = part.get_payload(decode=True)
                    if payload is None:
                        continue
                    return payload.decode(part.get_content_charset() or "utf-8", "replace")
                except Exception:
                    continue
        return ""
    if msg.get_content_type() == "text/html":
        try:
            payload = msg.get_payload(decode=True)
            if payload is None:
                return ""
            return payload.decode(msg.get_content_charset() or "utf-8", "replace")
        except Exception:
            return ""
    return ""


def _mmddyyyy_to_iso(s: str) -> str | None:
    try:
        mm, dd, yyyy = s.split("/")
        return f"{int(yyyy):04d}-{int(mm):02d}-{int(dd):02d}"
    except Exception:
        return None


def _placed_on_to_iso(s: str) -> str | None:
    """'Jan 12, 2026' -> '2026-01-12'."""
    try:
        m = re.match(r'([A-Za-z]{3,9})\.?\s+(\d{1,2}),\s*(\d{4})', s.strip())
        if not m:
            return None
        mon_raw, day, year = m.groups()
        mon = _MONTHS.get(mon_raw[:3].lower())
        if not mon:
            return None
        return f"{int(year):04d}-{mon:02d}-{int(day):02d}"
    except Exception:
        return None


def _normalize_tracking(raw: str) -> str | None:
    """Strip a leading '420' IMb routing prefix when present.

    Click-N-Ship's visible tracking text is the full Intelligent Mail
    barcode routing form: '420' + delivery-point ZIP+4 (9 digits) + the
    actual USPS tracking number, e.g.
    '4204306297079405530109355270819057' = '420' + '430629707' (ZIP+4)
    + '9405530109355270819057' (tracking). Some labels carry only the
    plain 5-digit ZIP in that slot ('420' + 5 digits), so both offsets
    are tried. The real USPS tracking number reliably starts with one of
    _USPS_TRACKING_PREFIXES once the routing prefix is stripped; if
    neither offset produces a recognized prefix, the digits are returned
    unstripped rather than guessed at.
    """
    if not raw:
        return None
    digits = re.sub(r"\D", "", raw)
    if not digits:
        return None
    if digits.startswith("420"):
        for offset in (12, 8):  # 420+ZIP4 (9 digits) first, then 420+ZIP5
            if len(digits) > offset:
                candidate = digits[offset:]
                if candidate.startswith(_USPS_TRACKING_PREFIXES):
                    return candidate
    return digits


def _parse_recipient(block: str) -> dict:
    out = {
        "recipient_name": None,
        "recipient_company": None,
        "recipient_street": None,
        "recipient_city": None,
        "recipient_state": None,
        "recipient_zip": None,
        "recipient_country": None,
    }
    m = _SHIPPED_TO_RE.search(block)
    if not m:
        return out
    lines = [html_mod.unescape(x).strip() for x in _PT5_RE.findall(m.group(1))]
    lines = [ln for ln in lines if ln]
    if not lines:
        return out

    out["recipient_name"] = lines[0]
    if len(lines) == 1:
        return out

    last = lines[-1]
    parts = last.split()
    if len(parts) >= 4:
        country = parts[-1]
        zip_code = parts[-2]
        state = parts[-3]
        city = " ".join(parts[:-3])
        if re.match(r'^\d{5}(-\d{4})?$', zip_code) and re.match(r'^[A-Za-z]{2}$', state):
            out["recipient_city"] = city
            out["recipient_state"] = state.upper()
            out["recipient_zip"] = zip_code
            out["recipient_country"] = country.upper()

    middle = lines[1:-1]
    if not middle:
        return out
    if len(middle) == 1:
        # Single mid-line: treat as street unless it clearly has no digits
        # AND no common street-suffix token (heuristic for a company name).
        line = middle[0]
        looks_like_street = bool(re.search(r'\d', line)) or bool(
            re.search(r'\b(ST|RD|AVE|DR|LN|BLVD|WAY|CT|PL|HWY|PKWY|CIR|TER|SQ|LOOP)\b', line, re.IGNORECASE)
        )
        if looks_like_street:
            out["recipient_street"] = line
        else:
            out["recipient_company"] = line
    else:
        out["recipient_company"] = middle[0]
        out["recipient_street"] = ", ".join(middle[1:])
    return out


def _parse_item(block: str, price_str: str) -> dict:
    shipment = {
        "tracking": None,
        "tracking_raw": None,
        "service": None,
        "scheduled_delivery": None,
        "recipient_name": None,
        "recipient_company": None,
        "recipient_street": None,
        "recipient_city": None,
        "recipient_state": None,
        "recipient_zip": None,
        "recipient_country": None,
        "cost_cents": None,
    }

    try:
        for bold_text in _BOLD_P_RE.findall(block):
            text = html_mod.unescape(bold_text).strip()
            if not text:
                continue
            if text.startswith("Scheduled delivery") or text.startswith("Shipped To"):
                continue
            shipment["service"] = text
            break
    except Exception:
        pass

    try:
        link_m = _TRACK_LINK_RE.search(block)
        if link_m:
            visible = html_mod.unescape(link_m.group(2)).strip()
            shipment["tracking_raw"] = re.sub(r"\s+", "", visible) or None
            shipment["tracking"] = _normalize_tracking(shipment["tracking_raw"])
    except Exception:
        pass

    try:
        sched_m = _SCHEDULED_RE.search(block)
        if sched_m:
            shipment["scheduled_delivery"] = _mmddyyyy_to_iso(sched_m.group(1))
    except Exception:
        pass

    try:
        shipment.update(_parse_recipient(block))
    except Exception:
        pass

    try:
        cost = price_str.replace(",", "")
        shipment["cost_cents"] = round(float(cost) * 100)
    except Exception:
        shipment["cost_cents"] = None

    return shipment


def parse_confirmation(raw) -> list:
    """Parse a Click-N-Ship Payment Confirmation email into a list of
    per-package shipment dicts.

    `raw` may be raw RFC822 email bytes, an email.message.Message, or an
    HTML string (a bare body, for tests/tools that already extracted it).

    Never raises on malformed input; returns [] if `raw` doesn't look like
    a Click-N-Ship confirmation.
    """
    try:
        html_body = ""
        subject = ""

        if isinstance(raw, (bytes, bytearray)):
            msg = email.message_from_bytes(bytes(raw))
            subject = _decode_header_value(msg.get("Subject"))
            html_body = _extract_html_body(msg)
        elif isinstance(raw, email.message.Message):
            subject = _decode_header_value(raw.get("Subject"))
            html_body = _extract_html_body(raw)
        elif isinstance(raw, str):
            if "<" in raw and ">" in raw:
                html_body = raw
            else:
                # Might be a raw RFC822 string.
                msg = email.message_from_string(raw)
                subject = _decode_header_value(msg.get("Subject"))
                html_body = _extract_html_body(msg)
        else:
            return []

        if not html_body:
            return []

        looks_like_cns = (
            "click-n-ship" in subject.lower()
            or "click-n-ship" in html_body.lower()
            or "TrackConfirmAction" in html_body
        )
        if not looks_like_cns:
            return []
        if "item-contents-column" not in html_body:
            return []

        order_uuid = None
        m = _ORDER_UUID_RE.search(html_body)
        if m:
            order_uuid = m.group(1)

        placed_on = None
        m2 = _PLACED_ON_RE.search(html_body)
        if m2:
            placed_on = _placed_on_to_iso(m2.group(1))

        shipments = []
        for block, price_str in _ITEM_RE.findall(html_body):
            try:
                shipment = _parse_item(block, price_str)
            except Exception:
                continue
            shipment["order_uuid"] = order_uuid
            shipment["placed_on"] = placed_on
            shipments.append(shipment)

        return shipments
    except Exception:
        return []


# ---------------------------------------------------------------------------
# IMAP harvest
# ---------------------------------------------------------------------------

USPS_SENDER = "noreply-ecns@usps.com"


def fetch_confirmations(*, limit=None, uids_after=None, imap=None) -> list:
    """Connect to Gmail via IMAP, pull Click-N-Ship confirmations from
    noreply-ecns@usps.com, parse each, and return a flat list of shipment
    dicts (one per package) with `uid` and `message_id` attached for later
    idempotency.

    `uids_after`: an IMAP UID (int or str); only messages with a strictly
    greater UID are processed. Lets a cron process only new mail.
    `limit`: cap on the number of *messages* fetched (most recent first).
    `imap`: inject an already-connected imaplib.IMAP4_SSL for testing;
    otherwise a connection is opened here using GMAIL_IMAP_USER /
    GMAIL_IMAP_APP_PASSWORD and closed before returning.
    """
    user = os.environ.get("GMAIL_IMAP_USER", "")
    password = os.environ.get("GMAIL_IMAP_APP_PASSWORD", "").replace(" ", "")

    own_connection = imap is None
    if own_connection:
        if not user or not password:
            raise RuntimeError(
                "GMAIL_IMAP_USER and GMAIL_IMAP_APP_PASSWORD must be set in env"
            )
        imap = imaplib.IMAP4_SSL("imap.gmail.com")
        imap.login(user, password)

    results = []
    try:
        imap.select("INBOX")
        typ, data = imap.search(None, "FROM", USPS_SENDER)
        uids = data[0].split() if data and data[0] else []

        if uids_after is not None:
            threshold = int(uids_after)
            uids = [u for u in uids if int(u) > threshold]

        # Most-recent-first when capped by limit.
        uids = sorted(uids, key=lambda u: int(u))
        if limit:
            uids = uids[-limit:]

        for uid in uids:
            try:
                typ, msg_data = imap.fetch(uid, "(RFC822)")
                if typ != "OK" or not msg_data or not msg_data[0]:
                    continue
                raw = msg_data[0][1]
                msg = email.message_from_bytes(raw)
                message_id = msg.get("Message-ID")
                shipments = parse_confirmation(raw)
                for shipment in shipments:
                    shipment["uid"] = uid.decode() if isinstance(uid, bytes) else str(uid)
                    shipment["message_id"] = message_id
                    results.append(shipment)
            except Exception:
                # Never let one malformed message abort the whole harvest.
                continue
    finally:
        if own_connection:
            try:
                imap.close()
            except Exception:
                pass
            try:
                imap.logout()
            except Exception:
                pass

    return results


# ---------------------------------------------------------------------------
# CLI / smoke test
# ---------------------------------------------------------------------------

def _run_smoke():
    total_messages = 0
    parsed_ok = 0
    with_tracking = 0
    with_scheduled = 0
    samples = []

    user = os.environ.get("GMAIL_IMAP_USER", "")
    password = os.environ.get("GMAIL_IMAP_APP_PASSWORD", "").replace(" ", "")
    if not user or not password:
        print("GMAIL_IMAP_USER / GMAIL_IMAP_APP_PASSWORD not set in env.")
        return

    imap = imaplib.IMAP4_SSL("imap.gmail.com")
    imap.login(user, password)
    try:
        imap.select("INBOX")
        typ, data = imap.search(None, "FROM", USPS_SENDER)
        uids = data[0].split() if data and data[0] else []
        total_messages = len(uids)

        for uid in uids:
            try:
                typ, msg_data = imap.fetch(uid, "(RFC822)")
                if typ != "OK" or not msg_data or not msg_data[0]:
                    continue
                raw = msg_data[0][1]
                shipments = parse_confirmation(raw)
                if shipments:
                    parsed_ok += 1
                for s in shipments:
                    if s.get("tracking"):
                        with_tracking += 1
                    if s.get("scheduled_delivery"):
                        with_scheduled += 1
                    if len(samples) < 3 and s.get("tracking"):
                        samples.append(s)
            except Exception:
                continue
    finally:
        try:
            imap.close()
        except Exception:
            pass
        imap.logout()

    print(f"Total messages from {USPS_SENDER}: {total_messages}")
    print(f"Messages parsed to >=1 package: {parsed_ok}")
    print(f"Packages with a tracking number: {with_tracking}")
    print(f"Packages with a scheduled delivery date: {with_scheduled}")
    print("\nSample shipments:")
    for s in samples:
        print(
            f"  tracking={s.get('tracking')} "
            f"city={s.get('recipient_city')} state={s.get('recipient_state')} "
            f"scheduled_delivery={s.get('scheduled_delivery')}"
        )


def main():
    parser = argparse.ArgumentParser(description="Click-N-Ship confirmation parser / harvester")
    parser.add_argument("--smoke", action="store_true", help="Run a live smoke test against the inbox")
    args = parser.parse_args()
    if args.smoke:
        _run_smoke()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
