"""Mint a client portal for a buyer at order time (idempotent, email-keyed).
Returns the raw token only when a NEW portal is minted, so the caller emails once."""
from dashboard import client_portal as _cp


def ensure_portal_for_buyer(cx, email, name):
    """Idempotent portal mint for a buyer at order time.

    Creates a portal for `email` if none exists (content `{"biofield_status": "none"}`),
    returns the raw token **only if newly minted** (else None). Empty email → None.

    Returns: str | None — raw token only on first create.
    """
    em = (email or "").strip().lower()
    if not em:
        return None
    token, _id = _cp.upsert_portal(cx, em, (name or "").strip(), {"biofield_status": "none"})
    return token   # non-None only on first create
