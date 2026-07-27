"""Framework-aware AI analysis of MentorshipU homework. Advisory only: returns a
rating + suggestions for improvement; NEVER raises (an outage must not block a
learner's completion). Mirrors dashboard/inbox_ai.py's lazy-client pattern."""
from __future__ import annotations

import json
import os

_MODEL = "claude-haiku-4-5-20251001"

# module slug -> Glen's framework lens for the AI system prompt. Body wired now;
# later modules (Symptoms->tissue layers+penetration, Terrain->5 Phases, Response->
# 5 levels of regulation, Prognosis->5 stages, Epigenetics->E4L+Remedy Match, ...)
# fill in on the drip cadence. Unmapped modules use the generic lens.
_FRAMEWORK = {
    "02-body": "Analyze through the Minding Body lens (the material, informational, and "
               "spirit body as one system). Note tissue-layer / whole-body patterns where relevant.",
}
_GENERIC = ("You are Dr. Glen Swartwout's teaching assistant for the Accelerated Self Healing "
            "certification. Evaluate the learner's homework supportively and specifically.")


def _client():
    """Lazy import so tests can mock this module without importing anthropic."""
    import anthropic
    return anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))


def analyze(module: str, assignment: str, submission: str) -> dict:
    """Return {'rating': str, 'feedback': str}. Advisory + best-effort — any error
    (no key, network, bad JSON) yields empty strings, never an exception."""
    if not (submission or "").strip():
        return {"rating": "", "feedback": ""}
    try:
        lens = _FRAMEWORK.get(module, "")
        system = (_GENERIC + (" " + lens if lens else "") +
                  " Reply as STRICT JSON: {\"rating\": <one short phrase>, "
                  "\"feedback\": <2-4 sentences of concrete suggestions for improvement "
                  "and further work>}. No prose outside the JSON.")
        user = f"ASSIGNMENT:\n{assignment}\n\nLEARNER SUBMISSION:\n{submission}"
        resp = _client().messages.create(
            model=_MODEL, max_tokens=600, system=system,
            messages=[{"role": "user", "content": user}])
        text = "".join(getattr(b, "text", "") for b in resp.content).strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text[text.find("{"):]
        d = json.loads(text)
        return {"rating": str(d.get("rating", "")), "feedback": str(d.get("feedback", ""))}
    except Exception:
        return {"rating": "", "feedback": ""}
