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


@cached("ghl.opportunities")
def opportunities_by_stage():
    """Returns total opportunity count for each pipeline, sorted highest first.

    Glen has 10+ pipelines. Showing one pipeline's stage breakdown buries the
    overall picture. This returns a ranked list so the dashboard widget shows
    every active pipeline at a glance.
    """
    pls = pipelines()["pipelines"]
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
