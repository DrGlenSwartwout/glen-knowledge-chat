"""Slice 3: provision-only backfill of client portals for existing reveal clients.
No emails. Idempotent. Dry-run by default."""
from dashboard import client_portal as _cp


def backfill_portals(cx, commit=False, limit=None):
    """Provision a bare portal (ensure_token — no System B report) for every
    biofield_reveals email that lacks one. Dry-run unless commit=True. `limit` caps
    NEW portals this run; the rest are counted in `remaining`. Never emails."""
    emails = [r[0] for r in cx.execute(
        "SELECT DISTINCT lower(email) FROM biofield_reveals "
        "WHERE email IS NOT NULL AND email <> '' ORDER BY lower(email)").fetchall()]
    already = provisioned = remaining = 0
    for email in emails:
        if cx.execute("SELECT 1 FROM client_portals WHERE email=?", (email,)).fetchone():
            already += 1
            continue
        if commit and (limit is None or provisioned < limit):
            _cp.ensure_token(cx, email, "")
            provisioned += 1
        else:
            remaining += 1
    return {"reveal_emails": len(emails), "already": already,
            "provisioned": provisioned, "remaining": remaining, "committed": bool(commit)}
