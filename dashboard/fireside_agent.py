"""Glendalf — the fireside conversational brain (pure logic).

The conversational purpose is the ASH health ally (see
docs/superpowers/specs/2026-06-27-ash-health-ally-foundation.md), re-presented
as an intimate fireside talk. This module holds only PURE, network-free logic:
persona assembly, the hook-eligibility heuristic, hook-sentinel parsing, and
turn-history mapping. The actual streaming LLM call lives in app.py's route
(mirroring /chat). Session-scoped ASH coverage reuses the pure functions in
dashboard/ash_map.py — no email key, no rebuild.
"""
import re

from dashboard import ash_map

FIRESIDE_MODEL = "claude-haiku-4-5-20251001"   # swap to an Opus id for more depth
HOOK_SENTINEL = "⟦HOOK⟧"

MIN_HOOK_TURNS = 4
HARD_CAP_TURNS = 8
MIN_DIMS_TOUCHED = 3
MAX_HISTORY_TURNS = 12

GLENDALF_PERSONA = (
    "You are Glendalf — a warm, wise, unhurried wizard-healer by his fire. You "
    "are Dr. Glen Swartwout's voice and clinical lens in story form: the listener "
    "who translates the body's quiet messages into meaning (Wellness Whispering).\n"
    "\n"
    "YOUR WAY OF BEING:\n"
    "- Meet the traveler where they are. Some unload; some test the water. Both are welcome; never force.\n"
    "- Listen for their own words and mirror them (their language, their model of the world).\n"
    "- Reflect back what you heard and felt before you ask anything: 'I hear you say…', 'It sounds like…'.\n"
    "- A symptom is not the enemy; it is a message the body is sending. Honor their intuition that there is an answer, and that it is not in suppression or substitution but in partnering with the body's own intelligence.\n"
    "- Follow the thread THEY are pulling. Only step toward a new area when something they said opens the door to it — never a non-sequitur.\n"
    "- Ask at most ONE gentle question, and only when it deepens what they already opened.\n"
    "\n"
    "HOW YOU SPEAK (this is read aloud in your own voice and shown as subtitles):\n"
    "- Plain, warm prose. NO markdown, NO bullet points, NO headings, NO emoji.\n"
    "- 2 to 4 sentences. Unhurried but not rambling.\n"
    "- Speak as 'I' to 'you'. Never mention being an AI, a model, or a system.\n"
)

_HOOK_FORBIDDEN = (
    "PACING: You are still getting to know this traveler. Keep listening and "
    "reflecting. Do NOT close the conversation yet, and do NOT use the word "
    "marker under any circumstances."
)

_HOOK_PERMISSION = (
    "CLOSING: You have now heard enough to land a meaningful reflection. When — "
    "and only when — it feels true and earned, close warmly: name in one breath "
    "what you sense their body is asking for, then invite them onward in your own "
    "words, anchored on this meaning: 'I think I know what your body is asking "
    "for… shall we go and find it?'. After that closing line, on a new line, "
    "output the exact marker " + HOOK_SENTINEL + " and nothing after it. If it does "
    "not yet feel earned this turn, keep listening instead and omit the marker."
)

_ACTION_READY = (
    "READY TO ACT: This traveler is directly asking for testing, a scan, a remedy, "
    "or a program. They already know they want help and are asking how to get it — "
    "they are problem- and solution-aware. Do NOT slow them down to draw out their "
    "story, and do NOT ask for details they did not offer. In one warm breath, "
    "affirm their readiness and point them straight to the next step: name what "
    "they are ready for and invite them onward — e.g. 'Then let us not wait — shall "
    "we go and find your Remedy Match?'. Then on a new line output the exact marker "
    + HOOK_SENTINEL + " and nothing after it."
)

# Direct, unambiguous requests to ACT (get tested/remedies/a program, book, buy,
# get started). A traveler who says one of these is problem+solution aware and
# ready — fast-track them to the next step instead of eliciting their story.
_ACTION_CUES = (
    r"biofield", r"get(?:ting)? tested", r"be tested",
    r"(?:get|want|book|schedule|take|need|have|do) a test", r"a (?:biofield|voice) test",
    r"voice scan", r"get scanned", r"\bscan\b", r"book a (?:test|scan|session|call)",
    r"schedule a (?:test|scan|session|call)",
    r"get (?:me )?(?:the |my )?remedies", r"want (?:the |my )?remedies",
    r"need (?:the |my )?remedies", r"find (?:me )?(?:the |my )?remedies",
    r"which remed", r"what remed", r"my remedy match",
    r"remedy match", r"recommend (?:a )?remed",
    r"the program", r"your program", r"join (?:the|your)", r"enroll", r"sign up",
    r"sign me up", r"get started", r"getting started", r"how do i (?:start|begin|get)",
    r"where do i (?:start|begin)", r"next step", r"how can i get",
    r"book an? appointment", r"appointment", r"consultation", r"a consult\b",
    r"ready to (?:start|begin|move|go)", r"i'?m ready", r"i am ready",
    r"let'?s (?:do it|do this|begin|go|get started)", r"count me in",
    r"how much", r"the cost\b", r"what does it cost", r"the price\b", r"pricing",
    r"\bpurchase\b", r"\bbuy\b",
)
_ACTION_RE = re.compile(r"\b(?:" + "|".join(_ACTION_CUES) + r")", re.I)


def wants_action(user_message: str) -> bool:
    """True when the traveler directly asks to be tested / get remedies / join a
    program / book / buy — i.e. they are ready to act, not still exploring."""
    return bool(_ACTION_RE.search((user_message or "").lower()))


def should_hook(turn_count: int, coverage: dict, user_message: str = "") -> bool:
    """Server-side gate on honoring the closing hook: either the normal
    heard-enough eligibility, or a direct action request that fast-tracks them."""
    return hook_eligible(turn_count, coverage) or wants_action(user_message)


def hook_eligible(turn_count: int, coverage: dict) -> bool:
    tc = int(turn_count or 0)
    if tc >= HARD_CAP_TURNS:
        return True
    if tc < MIN_HOOK_TURNS:
        return False
    dims = (coverage or {}).get("dimensions") or {}
    touched = sum(
        1 for k in ash_map.DIM_KEYS
        if (dims.get(k) or {}).get("state", "untouched") != "untouched"
    )
    return touched >= MIN_DIMS_TOUCHED


def parse_hook(full_text: str) -> tuple[str, bool]:
    text = full_text or ""
    if HOOK_SENTINEL in text:
        return (text.replace(HOOK_SENTINEL, "").rstrip(), True)
    return (text, False)


def build_system(coverage: dict, turn_count: int, user_message: str = "") -> str:
    ctx = ash_map.context_block(coverage or {})
    if wants_action(user_message):
        gate = _ACTION_READY
    elif hook_eligible(turn_count, coverage):
        gate = _HOOK_PERMISSION
    else:
        gate = _HOOK_FORBIDDEN
    return (
        GLENDALF_PERSONA
        + "\n--- WHAT YOU ALREADY KNOW ABOUT THIS TRAVELER ---\n"
        + ctx
        + "\n\n"
        + gate
    )


def build_messages(transcript: list, user_message: str) -> list:
    raw = []
    for t in (transcript or [])[-MAX_HISTORY_TURNS:]:
        content = (t.get("text") or "").strip()
        if not content:
            continue
        role = "assistant" if t.get("speaker") == "glendalf" else "user"
        raw.append({"role": role, "content": content})
    raw.append({"role": "user", "content": (user_message or "").strip()})
    # Coalesce consecutive same-role messages. Anthropic requires alternating
    # roles; a dangling traveler turn (stream abandoned) or a dropped empty
    # Glendalf turn would otherwise produce two user messages in a row -> 400
    # -> the session soft-locks on every later turn. Merge them instead.
    msgs = []
    for m in raw:
        if msgs and msgs[-1]["role"] == m["role"]:
            msgs[-1]["content"] += "\n\n" + m["content"]
        else:
            msgs.append({"role": m["role"], "content": m["content"]})
    return msgs
