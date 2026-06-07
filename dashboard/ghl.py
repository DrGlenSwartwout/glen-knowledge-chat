"""GHL pipeline — RemedyMatch sub-account contacts grouped by stage.

Uses GHL v1 API (rest.gohighlevel.com) which auths via the legacy location-level
API key (JWT). The v2 API at services.leadconnectorhq.com requires a Private
Integration Token and is not used here.
"""

import os
import requests
from datetime import datetime, timezone
from .cache import cached, last_success

GHL_API_KEY = os.environ.get("GHL_API_KEY", "")  # Legacy location-level JWT
BASE = "https://rest.gohighlevel.com/v1"


def _headers():
    return {"Authorization": f"Bearer {GHL_API_KEY}",
            "Accept": "application/json"}


def _norm_name(s):
    """Lowercase, collapse whitespace, drop punctuation for name comparison."""
    import re
    return re.sub(r"[^a-z0-9 ]", "", (s or "").lower()).strip()


def find_contact_by_name(name, timeout=15):
    """Resolve a 'Shipped To' name to a GHL contact email + confidence.

    Uses the v1 search endpoint (`/contacts/?query=`), which — unlike the
    `?email=` param — actually filters server-side. Returns:

        {"email", "contact_id", "name", "confidence"}  or  None

    confidence:
        "high"   — exactly one candidate AND its name matches exactly (has email)
        "medium" — exactly one candidate with an email (name not exact)
        "low"    — multiple candidates; best-effort first one with an email
    The watcher prefills To: for high/medium and leaves it blank (needs_review)
    for low/none, so a fuzzy match never silently mails the wrong person.
    """
    name = (name or "").strip()
    if not name:
        return None
    try:
        r = requests.get(f"{BASE}/contacts/", headers=_headers(),
                         params={"query": name, "limit": 20}, timeout=timeout)
        r.raise_for_status()
        contacts = r.json().get("contacts", []) or []
    except Exception:
        return None

    target = _norm_name(name)
    with_email = [c for c in contacts if c.get("email")]
    if not with_email:
        return None

    def full_name(c):
        return _norm_name(
            c.get("contactName")
            or " ".join(filter(None, [c.get("firstName"), c.get("lastName")]))
        )

    exact = [c for c in with_email if full_name(c) == target]
    if len(exact) == 1:
        c = exact[0]
        return {"email": c["email"], "contact_id": c.get("id"),
                "name": c.get("contactName"), "confidence": "high"}
    if len(with_email) == 1:
        c = with_email[0]
        return {"email": c["email"], "contact_id": c.get("id"),
                "name": c.get("contactName"), "confidence": "medium"}
    # Ambiguous — return best exact-or-first, flagged low for human review.
    c = exact[0] if exact else with_email[0]
    return {"email": c["email"], "contact_id": c.get("id"),
            "name": c.get("contactName"), "confidence": "low"}


@cached("ghl.pipelines")
def pipelines():
    r = requests.get(f"{BASE}/pipelines/", headers=_headers(), timeout=15)
    r.raise_for_status()
    pls = r.json().get("pipelines", [])
    return {"pipelines": [{"id": p["id"], "name": p["name"],
                           "stages": [{"id": s["id"], "name": s["name"]}
                                      for s in p.get("stages", [])]}
                          for p in pls],
            "last_success": last_success("ghl.pipelines")}


# Pipelines retired from the business — hidden from the dashboard card (and the
# total) even though they may still exist with contacts in GHL. Matched as a
# case-insensitive substring of the pipeline name.
#   "mctb" → both "MCTB Sales Pipeline" and "MCTB Onboarding Pipeline", a defunct
#   2024 funnel (no recent activity; only 2 real client contacts).
_HIDDEN_PIPELINES = ("email paramedic", "mctb")


def _is_hidden(name):
    n = (name or "").strip().lower()
    return any(h in n for h in _HIDDEN_PIPELINES)


@cached("ghl.opportunities")
def opportunities_by_stage():
    """Returns total opportunity count for each pipeline, sorted highest first.

    Glen has 10+ pipelines. Showing one pipeline's stage breakdown buries the
    overall picture. This returns a ranked list so the dashboard widget shows
    every active pipeline at a glance. Retired pipelines (_HIDDEN_PIPELINES) are
    excluded.
    """
    pls = [p for p in pipelines()["pipelines"] if not _is_hidden(p["name"])]
    if not pls:
        return {"empty": True, "message": "No pipelines configured in GHL"}

    pipeline_counts = []
    for pipe in pls:
        try:
            r = requests.get(f"{BASE}/pipelines/{pipe['id']}/opportunities",
                             headers=_headers(),
                             params={"limit": 100},
                             timeout=15)
            r.raise_for_status()
            opps = r.json().get("opportunities", [])
            pipeline_counts.append({"name": pipe["name"], "count": len(opps)})
        except Exception:
            # If one pipeline fails, skip it rather than fail the whole widget
            pipeline_counts.append({"name": pipe["name"], "count": None})

    pipeline_counts.sort(key=lambda x: (x["count"] is None, -(x["count"] or 0)))
    total = sum(p["count"] for p in pipeline_counts if p["count"] is not None)

    return {"pipelines": pipeline_counts,
            "total": total,
            "pipeline_count": len(pipeline_counts),
            "last_success": last_success("ghl.opportunities"),
            "as_of": datetime.now(timezone.utc).isoformat()}
