"""Pure helpers for the /begin voice doorway (the native first-scan).
Dependency-free so tests run without importing app.py."""
import re


def _slug(s) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-")


def voice_signal_tags(signals: dict) -> list:
    """GHL tags from a doorway voice-scan analysis summary.
    signals: {dominant_element, dominant_treasure, polyvagal_state(dict|str), top_themes:[str]}."""
    signals = signals or {}
    tags = []
    el = _slug(signals.get("dominant_element"))
    if el:
        tags.append(f"element:{el}")
    tr = _slug(signals.get("dominant_treasure"))
    if tr:
        tags.append(f"treasure:{tr}")
    pv = signals.get("polyvagal_state")
    if isinstance(pv, dict) and pv:
        top = max(pv, key=lambda k: pv.get(k) or 0)
        if (pv.get(top) or 0) > 0:
            tags.append(f"state:{_slug(top)}")
    elif isinstance(pv, str) and pv.strip():
        tags.append(f"state:{_slug(pv)}")
    for t in (signals.get("top_themes") or [])[:5]:
        s = _slug(t)
        if s:
            tags.append(f"theme:{s}")
    return tags
