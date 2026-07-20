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
