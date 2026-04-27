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
    """Returns count per stage for the first pipeline (with stage names, not IDs)."""
    pls = pipelines()["pipelines"]
    if not pls:
        return {"empty": True, "message": "No pipelines configured in GHL"}
    pipeline = pls[0]
    pipeline_id = pipeline["id"]
    stage_id_to_name = {s["id"]: s["name"] for s in pipeline.get("stages", [])}

    r = requests.get(f"{BASE}/pipelines/{pipeline_id}/opportunities",
                     headers=_headers(),
                     params={"limit": 100},
                     timeout=15)
    r.raise_for_status()
    opps = r.json().get("opportunities", [])
    by_stage = {}
    for o in opps:
        stage_id = o.get("pipelineStageId", "unknown")
        stage_name = stage_id_to_name.get(stage_id, stage_id)
        by_stage[stage_name] = by_stage.get(stage_name, 0) + 1
    return {"pipeline": pipeline["name"],
            "by_stage": by_stage,
            "total": len(opps),
            "last_success": last_success("ghl.opportunities"),
            "as_of": datetime.now(timezone.utc).isoformat()}
