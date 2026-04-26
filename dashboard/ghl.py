"""GHL pipeline — RemedyMatch sub-account contacts grouped by stage."""

import os
import requests
from datetime import datetime, timezone
from .cache import cached, last_success

GHL_API_KEY     = os.environ.get("GHL_API_KEY", "")          # Location-level API key
GHL_LOCATION_ID = os.environ.get("GHL_LOCATION_ID", "")
BASE = "https://services.leadconnectorhq.com"


def _headers():
    return {"Authorization": f"Bearer {GHL_API_KEY}",
            "Version": "2021-07-28",
            "Accept": "application/json"}


@cached("ghl.pipelines")
def pipelines():
    r = requests.get(f"{BASE}/opportunities/pipelines",
                     headers=_headers(),
                     params={"locationId": GHL_LOCATION_ID},
                     timeout=15)
    r.raise_for_status()
    pls = r.json().get("pipelines", [])
    return {"pipelines": [{"id": p["id"], "name": p["name"],
                            "stages": [s["name"] for s in p.get("stages", [])]}
                          for p in pls],
            "last_success": last_success("ghl.pipelines")}


@cached("ghl.opportunities")
def opportunities_by_stage():
    """Returns count per stage for the first pipeline."""
    pls = pipelines()["pipelines"]
    if not pls:
        return {"empty": True}
    pipeline_id = pls[0]["id"]
    r = requests.get(f"{BASE}/opportunities/search",
                     headers=_headers(),
                     params={"location_id": GHL_LOCATION_ID,
                             "pipeline_id": pipeline_id,
                             "limit": 100},
                     timeout=15)
    r.raise_for_status()
    opps = r.json().get("opportunities", [])
    by_stage = {}
    for o in opps:
        stage = o.get("pipelineStageId", "unknown")
        by_stage[stage] = by_stage.get(stage, 0) + 1
    return {"pipeline": pls[0]["name"],
            "by_stage": by_stage,
            "total": len(opps),
            "last_success": last_success("ghl.opportunities"),
            "as_of": datetime.now(timezone.utc).isoformat()}
