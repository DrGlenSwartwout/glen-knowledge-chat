"""HeyGen recent video renders."""

import os
import requests
from datetime import datetime, timezone
from .cache import cached, last_success

HEYGEN_API_KEY = os.environ.get("HEYGEN_API_KEY", "")


@cached("heygen.recent", ttl=600)
def recent_videos(limit=5):
    if not HEYGEN_API_KEY:
        return {"empty": True, "message": "HEYGEN_API_KEY not set"}
    r = requests.get("https://api.heygen.com/v1/video.list",
                     headers={"X-Api-Key": HEYGEN_API_KEY},
                     params={"limit": limit}, timeout=15)
    r.raise_for_status()
    data = r.json().get("data", {})
    videos = data.get("videos", [])[:limit]
    return {"videos": [{"id": v.get("video_id"),
                        "title": v.get("video_title"),
                        "status": v.get("status"),
                        "created_at": v.get("created_at"),
                        "url": v.get("video_url")}
                       for v in videos],
            "last_success": last_success("heygen.recent"),
            "as_of": datetime.now(timezone.utc).isoformat()}
