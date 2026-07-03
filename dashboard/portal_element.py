"""Refresh a member's TCM element state from their recent portal-chat messages.

Text mode: the portal chat is typed, so we run the engine over a rolling window
of the member's recent client messages (not one line — a single message carries
almost no elemental signal). Upgrades to full voice/lexical analysis for free
once portal voice-in ships; the engine call is identical.
"""
from dashboard.tcm_analysis import _haiku_analyze
from dashboard import portal_chat, member_element_state

_WINDOW = 6      # most-recent client messages to analyze together
_MIN_CHARS = 60  # below this the transcript is too thin to score meaningfully


def refresh(cx, email, window=_WINDOW):
    email = (email or "").strip().lower()
    if not email:
        return None
    msgs = portal_chat.list_messages(cx, email, limit=50) or []
    client_texts = [m["content"] for m in msgs
                    if m.get("role") == "client" and (m.get("content") or "").strip()]
    transcript = "\n".join(client_texts[-window:]).strip()
    if len(transcript) < _MIN_CHARS:
        return None
    haiku = _haiku_analyze(transcript, {"word_count": len(transcript.split())})
    elements = (haiku or {}).get("elements") or {}
    if not elements:
        return None
    return member_element_state.upsert(cx, email, elements, source="portal_chat")
