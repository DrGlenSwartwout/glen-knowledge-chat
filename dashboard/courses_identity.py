"""Member identity glue for MentorshipU LMS.

Thin adapter mapping a portal token to a member level. Reuses the existing
portal identity seam (`dashboard.portal_identity`) rather than reinventing
token resolution. Level 2 (paid) arrives in a later stage; this module only
ever returns 0 or 1. Kept pure of `app` (never imports it) so it unit-tests
fast and in isolation, mirroring `dashboard/portal_identity.py`.
"""

from __future__ import annotations

from dashboard import portal_identity


def member_level_for(cx, token: str | None) -> int:
    """0 = anonymous, 1 = registered member. Paid (2) arrives in Stage 3."""
    if not token:
        return 0
    try:
        # identity_from_token resolves against the `people` table but does not
        # create it (production already has the full table via
        # app._init_people_table; callers on a bare/isolated cx must ensure it
        # first — see dashboard/portal_identity.py's own tests).
        portal_identity._ensure_people_table(cx)
        ident = portal_identity.identity_from_token(cx, token)
    except Exception:
        return 0
    if ident is None:
        return 0
    return 1
