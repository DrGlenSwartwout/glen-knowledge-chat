"""Validation for the owner-only QBO read endpoint (/api/qbo/query).

Kept OUT of app.py so it is unit-testable with no secrets and no network — the
app module reaches Pinecone/QBO/Stripe at import time, which would push these
tests into the "skips silently" bucket.

QBO's /query API has no write verbs, so SELECT-only is guaranteed by the
protocol. This guard is the second lock: it makes the intent explicit at our
boundary and stops the endpoint from ever being repurposed into a write path
(or a stacked-statement vector) by a later edit.
"""
import re

SELECT_RE = re.compile(r"^\s*SELECT\s+", re.I)
MAXRESULTS_RE = re.compile(r"\bMAXRESULTS\b", re.I)
DEFAULT_CAP = 200


def sanitize(q, cap=DEFAULT_CAP):
    """Return (safe_query, error). Exactly one is non-None.

    Rules: non-empty, a single leading SELECT, no stacked statements, and a
    MAXRESULTS cap appended when the caller omitted one."""
    q = (q or "").strip()
    if not q:
        return None, "missing q"
    if not SELECT_RE.match(q):
        return None, "only SELECT queries are allowed"
    # A trailing ';' is harmless; one embedded means stacked statements.
    if ";" in q.rstrip().rstrip(";"):
        return None, "multiple statements not allowed"
    q = q.rstrip().rstrip(";").strip()
    if not MAXRESULTS_RE.search(q):
        q = f"{q} MAXRESULTS {int(cap)}"
    return q, None
