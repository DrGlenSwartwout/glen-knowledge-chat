"""Quiz engine for the /begin/quiz lead-magnet funnel.

Pure functions over the quiz config + an answers dict, mirroring begin_funnel.py.
Quiz content lives in data/quizzes.json (editable without a migration). The only
mutating I/O is the quiz_responses table (init/store/get), all via a
caller-supplied sqlite3 connection.
"""

import json
import os
import sqlite3
from datetime import datetime, timezone

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "data", "quizzes.json")

# High-signal (magnesium-depletion / foundational-gap) answer values per question.
_HIGH_SIGNALS = {
    "q2": {"frequent"},
    "q3": {"often"},
    "q4": {"frequent_fog"},
    "q5": {"6plus"},
    "q6": {"avoid"},
    "q7": {"rarely"},
    "q8": {"none"},
}


def _now():
    return datetime.now(timezone.utc).isoformat()


def load_config() -> dict:
    try:
        with open(_CONFIG_PATH) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"version": 0, "quizzes": {}}


def get_quiz(quiz_id: str, config: dict | None = None) -> dict | None:
    cfg = config if config is not None else load_config()
    return (cfg.get("quizzes") or {}).get(quiz_id)


def segment_of(answers: dict) -> str:
    return (answers or {}).get("q1") or "general"


def depletion_score(answers: dict) -> int:
    answers = answers or {}
    return sum(1 for q, highs in _HIGH_SIGNALS.items() if answers.get(q) in highs)


def _band_key(answers: dict) -> str:
    a = answers or {}
    q1 = a.get("q1")
    q8 = a.get("q8")
    if q1 in ("watch_wait", "family") or q8 in ("eye_formula", "both"):
        return "barrier"
    if a.get("q2") == "frequent" or a.get("q3") == "often":
        return "calm"
    if a.get("q4") in ("frequent_fog", "occasional_fog"):
        return "clarity"
    if a.get("q5") == "6plus" or a.get("q6") in ("avoid", "some_difficulty"):
        return "hardworking"
    return "foundational"


def result_for(quiz: dict, answers: dict) -> dict:
    band = _band_key(answers)
    spec = (quiz.get("bands") or {}).get(band) or (quiz.get("bands") or {}).get("foundational") or {}
    return {
        "band": band,
        "headline": spec.get("headline", ""),
        "reasoning": spec.get("reasoning", ""),
        "bullets": list(spec.get("bullets", [])),
        "segment": segment_of(answers),
        "depletion": depletion_score(answers),
    }
