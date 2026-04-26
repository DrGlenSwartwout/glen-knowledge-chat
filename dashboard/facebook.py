"""Facebook Ads — Boulder Test campaign 182914969 (verification-pending stub)."""

import os
import requests
from datetime import datetime, timezone
from .cache import cached, last_success

META_ACCESS_TOKEN = os.environ.get("META_ACCESS_TOKEN", "")
AD_ACCOUNT_ID     = os.environ.get("META_AD_ACCOUNT_ID", "act_324449786288723")
CAMPAIGN_ID       = os.environ.get("META_CAMPAIGN_ID", "182914969")


@cached("facebook.boulder", ttl=600)
def boulder_test_stats():
    if not META_ACCESS_TOKEN:
        return {"status": "verification_pending",
                "message": "Meta Business verification not yet complete. Manual CSV upload required for now.",
                "campaign_id": CAMPAIGN_ID,
                "as_of": datetime.now(timezone.utc).isoformat()}
    r = requests.get(
        f"https://graph.facebook.com/v18.0/{CAMPAIGN_ID}/insights",
        params={"access_token": META_ACCESS_TOKEN,
                "fields": "spend,impressions,clicks,ctr,cpc,reach"},
        timeout=15)
    r.raise_for_status()
    data = r.json().get("data", [])
    return {"status": "live",
            "campaign_id": CAMPAIGN_ID,
            "metrics": data[0] if data else {},
            "last_success": last_success("facebook.boulder"),
            "as_of": datetime.now(timezone.utc).isoformat()}
