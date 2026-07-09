"""One way to read a timestamp column.

`auth_tokens.expires_at` (and friends) were written by several different
minters over time, and they do not agree on a format. All three of these are
live in the production table right now:

    2026-07-09T17:49:00.002684+00:00     offset-aware
    2026-08-08T17:33:59.998605Z          Z-suffixed, naive
    2027-07-09T17:33:59.999677           bare naive

Each validator used to parse the way its own minter happened to write, so the
pairs were self-consistent by luck. Any code that read a column generically hit

    TypeError: can't compare offset-naive and offset-aware datetimes

and several validators swallowed that in a bare `except`, which quietly reports
a perfectly good token as EXPIRED. `parse_utc` accepts all three shapes and
always returns an aware UTC datetime, so a comparison can never raise.

Naive values are treated as UTC, which is correct: every minter that writes
them uses `datetime.utcnow()`.
"""
from datetime import datetime, timezone


def now_utc() -> datetime:
    """Timezone-aware current time. The only clock this codebase should read."""
    return datetime.now(timezone.utc)


def parse_utc(value) -> datetime:
    """Parse a stored ISO timestamp into an aware UTC datetime.

    Handles offset-aware, 'Z'-suffixed, bare-naive, and the malformed
    '...+00:00Z' that results from appending 'Z' to an already-aware
    isoformat(). Raises ValueError on anything it cannot parse -- callers
    that want a lenient default should catch it explicitly rather than
    relying on a bare `except` to mean "expired".
    """
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = (value or "").strip()
    if not text:
        raise ValueError("empty timestamp")
    if text.endswith("Z"):
        # covers both '...Z' (naive) and '...+00:00Z' (aware, Z wrongly appended)
        text = text[:-1]
    dt = datetime.fromisoformat(text)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def is_expired(expires_at, *, now=None) -> bool:
    """True if `expires_at` is in the past. Unparseable timestamps count as
    expired -- a token whose expiry we cannot read must not be honoured."""
    try:
        return parse_utc(expires_at) < parse_utc(now) if now is not None \
            else parse_utc(expires_at) < now_utc()
    except (ValueError, TypeError):
        return True
