"""Member identity glue for MentorshipU LMS.

Resolves a course-scoped access token (`dashboard.course_tokens`) to a member
level. Distinct from the client portal token — a course link grants course
access only, never portal/PII access. Kept pure of `app` (never imports it)
so it unit-tests fast and in isolation.
"""

from __future__ import annotations

from dashboard import course_tokens


def member_level_for(cx, token: str | None) -> int:
    """0 = anonymous, 1 = registered member, 2 = paid.

    Backed by a course-scoped token (distinct from the client portal token), so a
    course link grants course access only. Level 2 comes from an active paid
    entitlement on the token's email. Fail-safe: any error yields the lower level.
    """
    if not token:
        return 0
    try:
        email = course_tokens.resolve_course_token(cx, token)
    except Exception:
        return 0
    if not email:
        return 0
    try:
        from dashboard import course_entitlements
        return max(1, course_entitlements.paid_level_for(cx, email))
    except Exception:
        return 1
