"""Linkable-record registry for the dashboard briefings.

Pure module (no network, no Flask). The briefing runner builds a registry from
the live snapshot, stamps a short `ref` token onto each person record, and the
LLM is told to cite those records by ref. The registry (ref -> url) is persisted
beside the briefing and resolved client-side. No console key is ever stored in
the url; it is appended at click time in the dashboard.
"""

import os
import re

from urllib.parse import quote


_ANGLE_RE = re.compile(r"<([^>]+)>")

# Inbox "oldest senders" include the connected mailbox itself and automated
# system addresses (bounces, no-reply). None of those are linkable clients.
# The self list is env-overridable; default covers the connected Gmail account
# and its aggregated identities.
_SELF_EMAILS = {
    e.strip().lower()
    for e in os.environ.get(
        "BRIEFING_LINK_EXCLUDE_EMAILS",
        "drglenswartwout@gmail.com,this.elf@gmail.com,suerae1111@gmail.com",
    ).split(",")
    if e.strip()
}
_AUTOMATED_LOCALPARTS = {
    "mailer-daemon", "postmaster", "no-reply", "noreply", "donotreply",
    "do-not-reply", "notifications", "notification", "bounce", "bounces",
    "noreply-dmarc-support",
}


def _is_person_email(email):
    """True only for an address that could be a real, linkable client: a valid
    address that is neither the connected mailbox itself nor an automated /
    no-reply system sender."""
    email = (email or "").strip().lower()
    if "@" not in email:
        return False
    if email in _SELF_EMAILS:
        return False
    local = email.split("@", 1)[0]
    if local in _AUTOMATED_LOCALPARTS:
        return False
    if local.startswith(("no-reply", "noreply", "do-not-reply", "donotreply")):
        return False
    return True


def _parse_sender(raw):
    """From a possibly 'Name <addr@x.com>' From-header, return (email, display).
    No angle brackets -> the whole string is treated as both email and display.
    Mirrors dashboard.inbox._normalize_sender_email's extraction, kept local so
    this module stays pure."""
    raw = (raw or "").strip()
    m = _ANGLE_RE.search(raw)
    if m:
        email = m.group(1).strip()
        name = raw[:m.start()].strip().strip('"').strip()
        return email, (name or email)
    return raw, raw


def person_url(email):
    """Canonical console destination for a person, keyed by email. The console
    key is intentionally absent; it is appended client-side at click time."""
    return "/console/crm?email=" + quote(email or "", safe="")


def _iter_person_records(snapshot):
    """Yield (record_dict, display, email) for each person-bearing record.
    Phase 1 sources: inbox oldest senders (email in `from`) and Practice Better
    invoice clients (email in `email`). `_error` blocks and missing keys are
    skipped. Add later-phase sources here."""
    inbox = snapshot.get("inbox") or {}
    for rec in (inbox.get("oldest") or []):
        if isinstance(rec, dict):
            email, display = _parse_sender(rec.get("from"))
            yield rec, display, email
    pb = ((snapshot.get("money") or {}).get("practice_better") or {})
    for rec in (pb.get("invoices") or []):
        if isinstance(rec, dict):
            yield rec, rec.get("name"), rec.get("email")


def build_linkables(snapshot):
    """Stamp `ref` onto each person record that has an email and return the
    registry {ref: {type, display, url}}. Dedup by url. Mutates `snapshot`."""
    registry = {}
    url_to_ref = {}
    n = 0
    for rec, display, email in _iter_person_records(snapshot):
        email = (email or "").strip().lower()
        if not _is_person_email(email):
            continue
        url = person_url(email)
        ref = url_to_ref.get(url)
        if ref is None:
            n += 1
            ref = "r%d" % n
            url_to_ref[url] = ref
            registry[ref] = {"type": "person",
                             "display": (display or email).strip() or email,
                             "url": url}
        rec["ref"] = ref
    return registry
