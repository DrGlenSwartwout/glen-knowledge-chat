"""
Incentive engine — content selector + send orchestrator + feedback
processor for the Phase 0 beta. Imported into app.py at module load.
"""

import json
from datetime import datetime, timedelta, timezone
from typing import Optional


ANTI_STALE_DAYS = 30
HIGH_AFFINITY_DAYS = 14


def _parse_state_field(state: dict, key: str, default):
    raw = state.get(key)
    if not raw:
        return default
    try:
        return json.loads(raw)
    except Exception:
        return default


def select_topic_for_user(
    user_state: dict,
    candidate_topics: list,
    audience: str = "client",
) -> Optional[str]:
    """Pick the next topic for this user from the candidate pool.

    Rules (MVP):
      1. Eliminate topics sent within the last 30 days (anti-stale).
      2. Among remaining, prefer the topic with highest click count
         in the user's topic_engagement_history.
      3. If no engagement history, pick the first candidate
         (deterministic fallback).
      4. Return None if no candidate survives anti-stale filter.
    """
    now = datetime.now(timezone.utc)
    send_history = _parse_state_field(user_state, "topic_send_history", [])
    engagement = _parse_state_field(user_state, "topic_engagement_history", [])

    recent = set()
    for entry in send_history:
        try:
            sent_at = datetime.fromisoformat(entry["last_sent_at"])
            if (now - sent_at) <= timedelta(days=ANTI_STALE_DAYS):
                recent.add(entry["topic"])
        except Exception:
            continue

    fresh = [t for t in candidate_topics if t not in recent]
    if not fresh:
        return None

    clicks = {e["topic"]: e.get("click_count", 0) for e in engagement}

    fresh.sort(key=lambda t: (-clicks.get(t, 0), t))
    return fresh[0]
