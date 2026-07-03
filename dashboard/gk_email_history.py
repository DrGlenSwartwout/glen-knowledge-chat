"""One-time backfill: derive slug-keyed `purchase_history` rows from
GrooveKart order-confirmation emails in Gmail (source `support@remedymatch.com`,
subject "New order : #NNNN - CODE"). GrooveKart's admin/API are unavailable,
but every order triggers a confirmation email carrying the customer email,
order date, and a product link per line item whose URL embeds the catalog
slug — this is the only surviving record of that order history.

Structural model: dashboard.fmp_history.rebuild_from_fmp (same shape: parse
source rows -> filter against a known slug set -> purchase_history.replace_source).

Pure/testable: parse_order_email() takes a body string; rebuild_from_gk_emails()
takes an injected fetch_fn so it never calls Gmail directly. Gmail access
itself (fetch_gk_order_emails) reuses dashboard.inbox's existing OAuth
service/body-decoding helpers rather than rebuilding auth plumbing.
"""
import re

from dashboard import purchase_history as _ph

_EMAIL_RE = re.compile(r"\(([^()\s]+@[^()\s]+)\)")
_DATE_RE = re.compile(r"Placed on (\d{2})-(\d{2})-(\d{4})")
_ORDER_REF_RE = re.compile(r"ORDER:\s*([A-Z0-9]+)")
_SLUG_RE = re.compile(r"remedymatch\.com/remedies/[^/\"']+/\d+-([a-z0-9-]+)")

# Gmail search: every GrooveKart order-confirmation email.
_GK_ORDER_QUERY = 'from:support@remedymatch.com subject:"New order"'


def parse_order_email(body, subject=""):
    """Parse a GrooveKart order-confirmation email body into
    {email, purchased_at, order_ref, slugs}. Robust to missing pieces —
    never raises; absent fields come back as "" / None / []."""
    body = body or ""

    email_m = _EMAIL_RE.search(body)
    email = email_m.group(1).strip().lower() if email_m else ""

    date_m = _DATE_RE.search(body)
    purchased_at = None
    if date_m:
        mm, dd, yyyy = date_m.groups()
        purchased_at = f"{yyyy}-{mm}-{dd}"

    order_m = _ORDER_REF_RE.search(body)
    order_ref = order_m.group(1) if order_m else None

    slugs = _SLUG_RE.findall(body)

    return {"email": email, "purchased_at": purchased_at, "order_ref": order_ref, "slugs": slugs}


def rebuild_from_gk_emails(cx, *, fetch_fn, catalog_slugs):
    """Rebuild the 'groovekart' slice of `purchase_history` from GrooveKart
    order-confirmation emails. `fetch_fn()` returns a list of
    {"body":..., "subject":...} dicts (injected, so this stays unit-testable
    without touching Gmail). `catalog_slugs` is a set of known product slugs;
    any parsed slug not in it is skipped and counted. Returns counts."""
    catalog_slugs = catalog_slugs or set()
    emails = fetch_fn() or []

    rows = []
    skipped_unmapped = 0
    skipped_noemail = 0

    for msg in emails:
        parsed = parse_order_email(msg.get("body") or "", msg.get("subject") or "")
        if not parsed["email"]:
            skipped_noemail += 1
            continue
        for slug in parsed["slugs"]:
            if slug not in catalog_slugs:
                skipped_unmapped += 1
                continue
            rows.append((parsed["email"], slug, parsed["purchased_at"], parsed["order_ref"]))

    n = _ph.replace_source(cx, "groovekart", rows)
    return {"orders": len(emails), "rows": n,
             "skipped_unmapped": skipped_unmapped, "skipped_noemail": skipped_noemail}


def fetch_gk_order_emails():
    """Fetch all GrooveKart order-confirmation emails from Gmail via the
    app's existing OAuth access (dashboard.inbox). Paginates the full result
    set (Gmail messages().list, 100/page) and returns
    [{"body": <plaintext>, "subject": ...}, ...]."""
    from dashboard import inbox as _inbox

    svc = _inbox._get_gmail_service()
    out = []
    page_token = None
    while True:
        res = svc.users().messages().list(
            userId="me", q=_GK_ORDER_QUERY, maxResults=100, pageToken=page_token
        ).execute()
        for m in (res.get("messages") or []):
            full = svc.users().messages().get(userId="me", id=m["id"], format="full").execute()
            headers = full.get("payload", {}).get("headers", [])
            subject = _inbox._header(headers, "Subject")
            body = _inbox._extract_body(full.get("payload", {}))
            plain = body.get("plain") or _inbox._strip_html_to_text(body.get("html") or "")
            out.append({"body": plain, "subject": subject})
        page_token = res.get("nextPageToken")
        if not page_token:
            break
    return out
