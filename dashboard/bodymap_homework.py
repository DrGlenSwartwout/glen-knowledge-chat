"""Body-Map homework helper: parse/validate/summarize a client's Body-Map
homework payload (JSON from the interactive Body-Map tool used in the
ash-certification course's 02-body module).

Pure module: stdlib + bodymap_store only. No Flask, no DB, no network.
"""
import json

import bodymap_store

# The ten whole-body systems allowed for this homework tool (organs first).
# Reflex-microsystem / meridian systems (iridology, sclerology, ear, foot,
# hand, meridian, eav, neurotome, face, dental, organclock) are deliberately
# excluded even though they exist in bodymap_store.SYSTEMS.
WHOLE_BODY_SYSTEMS = [
    "organs",
    "skeleton",
    "muscle",
    "nervous",
    "endocrine",
    "respiratory",
    "digestive",
    "cardiovascular",
    "urogenital",
    "lymph",
]

_NOTE_MAX = 500
_OVERALL_NOTE_MAX = 2000


def _clean_note(value, cap):
    if not isinstance(value, str):
        value = "" if value is None else str(value)
    value = value.strip()
    return value[:cap]


def parse_marks(payload):
    """Parse+validate a Body-Map homework JSON payload.

    Returns a normalized dict:
        {"system": str, "marks": [{"zone","anatomy","note"}, ...], "note": str}
    or None if the payload is malformed / unusable in any way. Never raises.
    """
    try:
        data = json.loads(payload)
    except Exception:
        return None

    if not isinstance(data, dict):
        return None

    system = data.get("system")
    if not isinstance(system, str) or not system.strip():
        return None
    system = system.strip()
    if system not in WHOLE_BODY_SYSTEMS:
        return None

    try:
        valid_ids = set(bodymap_store.zone_ids(system))
        zones = bodymap_store.load_map(system).get("zones", [])
        anatomy_by_id = {z.get("id"): z.get("anatomy", "") for z in zones}

        raw_marks = data.get("marks") or []
        if not isinstance(raw_marks, list):
            raw_marks = []

        marks = []
        for item in raw_marks:
            if not isinstance(item, dict):
                continue
            zone = item.get("zone")
            if not isinstance(zone, str) or zone not in valid_ids:
                continue
            anatomy = anatomy_by_id.get(zone, "") or ""
            note = _clean_note(item.get("note", ""), _NOTE_MAX)
            marks.append({"zone": zone, "anatomy": anatomy, "note": note})

        overall_note = _clean_note(data.get("note", ""), _OVERALL_NOTE_MAX)

        return {"system": system, "marks": marks, "note": overall_note}
    except Exception:
        return None


def summarize_marks(payload):
    """Human-readable plain-text summary of a homework payload, or None."""
    parsed = parse_marks(payload)
    if parsed is None:
        return None

    lines = [f"Body system: {parsed['system'].title()}."]

    marks = parsed.get("marks") or []
    if marks:
        parts = []
        for m in marks:
            anatomy = m.get("anatomy") or m.get("zone") or ""
            note = m.get("note") or ""
            if note:
                parts.append(f"{anatomy} (note: {note})")
            else:
                parts.append(anatomy)
        lines.append("Areas of concern: " + "; ".join(parts))
    else:
        lines.append("Areas of concern: none marked specifically.")

    overall_note = parsed.get("note") or ""
    if overall_note:
        lines.append(f"Overall reflection: {overall_note}")

    return "\n".join(lines)


def has_content(parsed):
    """True if the parsed payload has marks or a non-blank overall note."""
    if not isinstance(parsed, dict):
        return False
    marks = parsed.get("marks")
    if marks:
        return True
    note = parsed.get("note")
    if isinstance(note, str) and note.strip():
        return True
    return False
