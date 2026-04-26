"""Brain state — read latest brain-state.json snapshot uploaded from Mac cron."""

import json
import os
from pathlib import Path
from datetime import datetime, timezone

# Server-side snapshot path (uploaded via /api/brain/upload)
SNAPSHOT = Path(os.environ.get("BRAIN_SNAPSHOT_PATH",
                                "/tmp/brain-state.json"))


def read_brain():
    """Return top 3 of each category from the latest snapshot."""
    if not SNAPSHOT.exists():
        return {"empty": True, "message": "No brain-state snapshot uploaded yet."}
    raw = json.loads(SNAPSHOT.read_text())
    meta = raw.get("metadata", {})
    return {
        "empty": False,
        "last_run": meta.get("last_run"),
        "total_observations": meta.get("total_observations", 0),
        "total_days": meta.get("total_days_accumulated", 0),
        "commitments": raw.get("commitments", [])[:3],
        "follow_ups":  raw.get("follow_ups", [])[:3],
        "predictions": raw.get("predictions", [])[:3],
        "priorities":  raw.get("priorities", {}).get("declared_priorities", [])[:3],
        "as_of": datetime.now(timezone.utc).isoformat(),
    }


def write_brain(payload_bytes):
    """Persist uploaded brain-state.json. Validates JSON before writing."""
    text = payload_bytes.decode("utf-8")
    json.loads(text)  # raises if invalid
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT.write_text(text)
    return {"saved": True, "path": str(SNAPSHOT), "bytes": len(text)}
