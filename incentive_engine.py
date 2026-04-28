"""
Incentive engine — content selector + send orchestrator + feedback
processor for the Phase 0 beta. Imported into app.py at module load.
"""

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
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


# ── Personal email generator (Task 7) ────────────────────────────────

_TPL_DIR = Path(__file__).parent / "templates"
_jinja_env = None


def _get_jinja_env():
    global _jinja_env
    if _jinja_env is None:
        from jinja2 import Environment, FileSystemLoader
        _jinja_env = Environment(
            loader=FileSystemLoader(str(_TPL_DIR)),
            autoescape=False,  # plain text — no HTML escaping
            keep_trailing_newline=True,
        )
    return _jinja_env


def _llm_complete(prompt: str, max_tokens: int = 500) -> str:
    """Call Claude Haiku for the teaching body. Replaceable in tests."""
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()


def generate_personal_email(
    user: dict,
    topic: str,
    topic_source_text: str,
    product: dict,
    is_beta: bool = False,
    audience: str = "client",
    monthly_window: bool = False,
) -> dict:
    """Build a Glen-voice plain-text Personal email for one user.

    Returns:
      {"subject": str, "body": str}
    """
    audience_framing = (
        "Write at a clinical-practitioner level — assume the reader can "
        "interpret mechanism and dosage. Lean into mechanism-of-action depth."
        if audience == "practitioner"
        else
        "Write in warm, accessible language. No clinical jargon. "
        "Speak to the reader's own healing intelligence."
    )

    prompt = (
        "You are writing a short plain-text email FROM Dr. Glen Swartwout "
        "to a single subscriber, in his personal voice. ~150-300 words.\n\n"
        "Today's topic: " + topic + "\n\n"
        "Source passage to ground the teaching (synthesize, don't quote):\n"
        + topic_source_text[:2000] + "\n\n"
        "Style: " + audience_framing + "\n\n"
        "Open with ONE teaching nugget — a distinction, a question, or a "
        "surprising connection. Then naturally transition to recommending "
        "the product " + product["name"] + ". Keep it warm and direct. No "
        "subject line, no greeting, no signature — just the body. ~150-300 "
        "words."
    )
    teaching_body = _llm_complete(prompt, max_tokens=500)

    subj_prompt = (
        f"Write a short email subject line (max 50 chars) for a personal "
        f"email about {topic}. Should sound like a question or a "
        f"distinction, NOT a sale. Output only the subject line."
    )
    subject = _llm_complete(subj_prompt, max_tokens=30).strip().strip('"')[:60]

    env = _get_jinja_env()
    template = env.get_template("personal_email.txt.j2")
    body = template.render(
        user=user,
        teaching_body=teaching_body,
        product=product,
        is_beta=is_beta,
        monthly_window=monthly_window,
        unsubscribe_url=(
            f"https://glen-knowledge-chat.onrender.com/"
            f"unsubscribe?email={user['email']}&channel=personal"
        ),
    )

    return {"subject": subject, "body": body}
