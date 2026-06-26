"""Linkable-record registry for the dashboard briefings.

Pure module (no network, no Flask). The briefing runner builds a registry from
the live snapshot, stamps a short `ref` token onto each person record, and the
LLM is told to cite those records by ref. The registry (ref -> url) is persisted
beside the briefing and resolved client-side. No console key is ever stored in
the url; it is appended at click time in the dashboard.
"""

from urllib.parse import quote


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
            yield rec, rec.get("from"), rec.get("from")
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
        email = (email or "").strip()
        if "@" not in email:
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
