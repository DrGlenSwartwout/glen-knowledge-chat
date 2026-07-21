"""Member identity glue for MentorshipU LMS.

Resolves a course-scoped access token (`dashboard.course_tokens`) to a member
level. Distinct from the client portal token — a course link grants course
access only, never portal/PII access. Level 2 (paid) arrives in a later
stage; this module only ever returns 0 or 1. Kept pure of `app` (never
imports it) so it unit-tests fast and in isolation.
"""

from __future__ import annotations

from dashboard import course_tokens


def member_level_for(cx, token: str | None) -> int:
    """0 = anonymous, 1 = registered member. Paid (2) arrives in Stage 3.

    Backed by a course-scoped token, distinct from the client portal token,
    so a course link grants course access only — never portal/PII access.
    """
    if not token:
        return 0
    try:
        email = course_tokens.resolve_course_token(cx, token)
    except Exception:
        return 0
    return 1 if email else 0
