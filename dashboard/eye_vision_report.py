"""Portal-facing Eye & Vision report derived from canonical E4L mappings.

Phase 1 is descriptive only: structure/function relationships and reported
history context. It intentionally contains no diagnosis, treatment, product,
remedy, protocol, or dosing suggestions.
"""
from __future__ import annotations

import hashlib
import json

from dashboard import biofield_e4l, intake, portal_extended_history


SECTION_KEY = "eye_vision_report"
OCULAR_STRUCTURES = {
    "eye", "eyes", "retina", "retinal", "optic nerve",
    "vision", "visual processing", "visual pathway",
}
EYE_HISTORY_TERMS = (
    "eye", "eyes", "vision", "visual", "ocular", "glaucoma", "cataract",
    "macular", "retina", "retinal", "optic nerve", "dry eye", "myopia",
    "hyperopia", "astigmatism", "floaters", "uveitis", "keratoconus",
    "amblyopia", "strabismus", "diplopia",
)
LIMITATIONS = [
    "This report describes E4L Infoceutical pattern-to-structure relationships.",
    "It does not diagnose an eye condition or establish damage, severity, prognosis, or causation.",
    "It does not replace an eye examination or other appropriate clinical assessment.",
    "Remedies, products, protocols, dosing, and support suggestions are outside this phase.",
]


def _norm(value):
    return " ".join(str(value or "").strip().lower().replace("-", " ").split())


def _is_ocular(value):
    return _norm(value) in OCULAR_STRUCTURES


def _flatten_strings(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        out = []
        for item in value.values():
            out.extend(_flatten_strings(item))
        return out
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            out.extend(_flatten_strings(item))
        return out
    return []


def reported_eye_history(cx, email):
    """Return matching client-reported/canonical eye-history labels."""
    labels = []
    try:
        response = intake.get_response(cx, email) or {}
        labels.extend(_flatten_strings(response.get("answers") or {}))
    except Exception:
        pass
    try:
        extended = portal_extended_history.get(cx, email) or {}
        labels.extend(_flatten_strings(extended.get("answers") or {}))
    except Exception:
        pass
    try:
        from dashboard import canonical_tags
        labels.extend(str(v) for v in
                      (canonical_tags.get_person(cx, email).get("conditions") or []))
    except Exception:
        pass
    try:
        row = cx.execute(
            "SELECT conditions, tags FROM people WHERE lower(email)=lower(?)",
            ((email or "").strip(),)).fetchone()
        if row:
            for cell in (row[0], row[1]):
                try:
                    labels.extend(_flatten_strings(json.loads(cell or "[]")))
                except Exception:
                    pass
    except Exception:
        pass
    matches = []
    for label in labels:
        text = _norm(label)
        if text and any(term in text for term in EYE_HISTORY_TERMS):
            clean = str(label).strip()
            if clean and clean not in matches:
                matches.append(clean)
    return matches


def _mappings_for(codes, db_path=None):
    mappings = {}
    cx = biofield_e4l._connect_ro(biofield_e4l._db_path(db_path))
    if cx is None or not codes:
        return mappings
    try:
        placeholders = ",".join("?" for _ in codes)
        rows = cx.execute(
            f"SELECT code, structure, stype, is_primary, source_phrase "
            f"FROM e4l_pattern_structures WHERE code IN ({placeholders}) "
            f"ORDER BY is_primary DESC, structure ASC", tuple(codes)).fetchall()
        for row in rows:
            mappings.setdefault(row["code"], []).append(dict(row))
    except Exception:
        return {}
    finally:
        cx.close()
    return mappings


def _finding(pattern, mappings, history):
    ocular = [m for m in mappings if _is_ocular(m.get("structure"))]
    if not ocular:
        return None
    structures = list(dict.fromkeys(
        str(m.get("structure") or "").strip() for m in ocular
        if str(m.get("structure") or "").strip()))
    direct = any(bool(m.get("is_primary")) for m in ocular)
    code = str(pattern.get("code") or "").strip()
    name = str(pattern.get("name") or code or "E4L pattern").strip()
    seed = code + "|" + "|".join(sorted(_norm(v) for v in structures))
    if direct:
        body = (f"The scan prioritized the {name} Infoceutical pattern, which "
                f"the E4L catalog maps directly to {', '.join(structures)}.")
        classification = "direct_ocular"
    else:
        body = (f"The broader {name} Infoceutical pattern includes an E4L "
                f"catalog relationship with {', '.join(structures)}.")
        classification = "associated_ocular"
    history_context = None
    if history:
        history_context = (
            f"Your reported history includes {', '.join(history)}. It is shown "
            "alongside this relationship for context; the scan does not diagnose "
            "the reported condition or establish a causal relationship.")
    return {
        "finding_id": hashlib.sha256(seed.encode()).hexdigest()[:16],
        "code": code or None,
        "name": name,
        "priority_rank": pattern.get("rank"),
        "classification": classification,
        "ocular_structures": structures,
        "client_text": body,
        "history_context": history_context,
    }


def build_portal_block(cx, email, today, saved_collapsed=None, db_path=None):
    """Build the safe client view and resolve its initial/persisted open state."""
    history = reported_eye_history(cx, email)
    scan = biofield_e4l.scan_context(email, today, db_path=db_path, limit=100)
    if not scan.get("found"):
        return None
    patterns = scan.get("findings") or []
    mappings = _mappings_for(
        [p.get("code") for p in patterns if p.get("code")], db_path=db_path)
    findings = []
    for pattern in patterns:
        finding = _finding(pattern, mappings.get(pattern.get("code"), []), history)
        if finding:
            findings.append(finding)
    findings.sort(key=lambda row: (
        0 if row["classification"] == "direct_ocular" else 1,
        row["priority_rank"] if isinstance(row["priority_rank"], (int, float)) else 10**9,
        row["code"] or "",
    ))
    default_open = bool(history)
    is_open = (not saved_collapsed) if saved_collapsed is not None else default_open
    return {
        "section_key": SECTION_KEY,
        "title": "Your Eye & Vision Pattern Report",
        "intro": ("This focused view gathers the eye- and vision-related "
                  "structure/function relationships present in your E4L scan."),
        "status": "ready" if findings else "no_eye_patterns",
        "scan_date": scan.get("scan_date"),
        "findings": findings,
        "empty_state": ("This scan did not contain a canonical eye- or "
                        "vision-mapped Infoceutical pattern.") if not findings else None,
        "limitations": LIMITATIONS,
        "open": is_open,
        "default_open": default_open,
        "preference_saved": saved_collapsed is not None,
        "history_eye_issue": bool(history),
    }
