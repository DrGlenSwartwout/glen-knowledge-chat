"""Public, unauthenticated surfaces: the sample portal and the practitioner storefront.

Design rule (see spec 2026-07-19-public-surfaces-storefront-and-demo.md §2):
public payloads are built from an explicit field WHITELIST, never by filtering a
private payload down. Filters fail open — the day someone adds a field to the
portal payload, a filter-based public view silently publishes it. A whitelist
fails closed.

`build_demo_view()` deliberately takes no `cx`. That is the enforcement
mechanism, not a convention: there is no code path from this function to real
client data, so no flag can be misconfigured into leaking one.

Sample content is clinically neutral on purpose. A compelling fictional outcome
functions as an implied health claim.
"""

import copy
import sqlite3
from datetime import datetime, timezone

from dashboard import share_header as _sh

DEMO_FIXTURE = {
    "sample": True,
    "greeting": "This is a sample portal. The person and data below are illustrative.",
    "phase": "Rejuvenate",
    "practitioner": {
        "name": "Dr. Glen Swartwout",
        "practice": "Remedy Match",
    },
    "layers": [
        {"n": 1, "title": "Terrain", "meaning": "Where the body is starting from."},
        {"n": 2, "title": "Drainage", "meaning": "How well the body clears what it releases."},
        {"n": 3, "title": "Support", "meaning": "What the body is asking for next."},
    ],
    "findings": [
        {"name": "Sample finding A", "note": "Your own findings appear here."},
        {"name": "Sample finding B", "note": "Your own findings appear here."},
        {"name": "Sample finding C", "note": "Your own findings appear here."},
    ],
    "orders": [
        {"date": "2026-05-02", "label": "Sample order", "total": "$84.00"},
        {"date": "2026-06-14", "label": "Sample order", "total": "$126.00"},
    ],
    "pricing": [
        {"label": "Single formulation", "price": "$42.00"},
        {"label": "Member price", "price": "$33.60"},
    ],
    "body_map": {"available": True},
}


def build_demo_view():
    """Return the synthetic sample-portal payload.

    Takes no connection by design. See module docstring.
    """
    return copy.deepcopy(DEMO_FIXTURE)


# Every key a practitioner storefront may publish. Adding a key here is a
# deliberate decision to make that data public — treat edits to this set as a
# privacy change, not a refactor.
PRACTITIONER_PUBLIC_FIELDS = frozenset({
    "slug",
    "practitioner_name",
    "practice_name",
    "bio",
    "photo_url",
    "logo_url",
    "services",
    "location",          # city/state only, never a street address
    "accepting_clients",
    "featured_products",  # retail prices only
    "catalog_url",
    "profit_disclosure",
})

PROFIT_DISCLOSURE = (
    "Your practitioner earns a portion of what you spend here. "
    "Your price is the same either way."
)


def _public_only(view, allowed):
    """Final fail-closed guard: drop any key not explicitly allowed."""
    return {k: v for k, v in view.items() if k in allowed}


def build_practitioner_storefront(cx, slug):
    """Build the public storefront payload for `slug`, or None if unknown.

    Whitelisted: the returned dict is filtered against PRACTITIONER_PUBLIC_FIELDS
    as the last step, so a new column on affiliate_signups can never silently
    become public.
    """
    row = cx.execute(
        "SELECT name, organization, slug FROM affiliate_signups"
        " WHERE slug=? AND status='approved'", (slug,)).fetchone()
    if not row:
        return None

    view = {
        "slug": row["slug"],
        "practitioner_name": row["name"] or "",
        "practice_name": row["organization"] or "",
        "bio": "",
        "photo_url": "",
        "logo_url": "",
        "services": [],
        "location": "",
        "accepting_clients": True,
        "featured_products": [],
        "catalog_url": "/begin/explore",
        "profit_disclosure": PROFIT_DISCLOSURE,
    }
    return _public_only(view, PRACTITIONER_PUBLIC_FIELDS)


SHARE_HEADER_PUBLIC_FIELDS = frozenset({"display_name", "body"})


def build_share_header(cx, slug):
    """Return the APPROVED self-authored header for `slug`, or None.

    Only two fields ever reach the public page. Nothing from get_portal_view
    touches this path.

    Fails closed: a missing affiliate_signups or share_headers table (e.g. a
    fresh deployment, or a LOG_DB that predates one of these tables) means
    "no header", not a 500. This function is a read path on a public surface
    and must not perform a schema write, so it does not create either table
    itself.
    """
    try:
        row = cx.execute(
            "SELECT email FROM affiliate_signups WHERE slug=? AND status='approved'",
            (slug,)).fetchone()
    except sqlite3.Error:
        return None
    if not row:
        return None
    try:
        hdr = _sh.get_approved(cx, row["email"])
    except sqlite3.Error:
        return None
    if not hdr:
        return None
    return _public_only(hdr, SHARE_HEADER_PUBLIC_FIELDS)


def _ensure_views_table(cx):
    """Issue the public_surface_views DDL. Does NOT commit — callers that
    only need the schema (record_view, which commits once at the end
    alongside its INSERT) and callers that need it committed immediately
    (init_public_surface_views_table) share this so the DDL is never
    duplicated. Deliberately no module-level "already created" flag: tests
    construct a fresh database per test, and a cached flag would skip
    creation on a new DB."""
    cx.execute("""
        CREATE TABLE IF NOT EXISTS public_surface_views (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            slug      TEXT NOT NULL,
            surface   TEXT NOT NULL,
            viewed_at TEXT NOT NULL
        )
    """)
    cx.execute("CREATE INDEX IF NOT EXISTS ix_psv_slug ON public_surface_views(slug)")


def init_public_surface_views_table(cx):
    """Anonymous page-view counts per slug. Deliberately NOT referral_events —
    that table is lead-shaped (email, lead_id, utm_*) and these views have no
    lead attached."""
    _ensure_views_table(cx)
    cx.commit()


def record_view(cx, slug, surface):
    """Record one public-surface visit. Not deduped — per-slug view counts are
    the instrumentation this feature is being measured by.

    Issues the DDL (IF NOT EXISTS, so a no-op once the table exists) and the
    INSERT under a single commit, rather than committing the schema and the
    row separately — this runs under the app's global write lock on the
    hottest public routes, so it should touch the WAL/journal once per call,
    not twice."""
    _ensure_views_table(cx)
    cx.execute(
        "INSERT INTO public_surface_views (slug, surface, viewed_at) VALUES (?,?,?)",
        (slug, surface, datetime.now(timezone.utc).isoformat(timespec="seconds")))
    cx.commit()
