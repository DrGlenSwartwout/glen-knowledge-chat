"""Phase 4: detect a positive result/success in a client's recent communications.

Mirrors the dashboard/biofield_interpret.interpret_stresses pattern: an injected
`complete(system, user) -> str` returning strict JSON, so the classifier is fully testable
offline. Privacy: only ever fed AI summaries / the client's own words (never email raw_text).
"""
import json

_DEFAULT = {"positive": False, "confidence": 0.0, "quote": "", "kind": "general"}
_KINDS = ("remedy", "service", "general")


def build_positive_prompt(comms_text):
    system = (
        "You read a client's recent communications with a natural wellness practice — chat questions, "
        "AI summaries of their email replies, intake notes, and journal reflections. Decide whether "
        "the client CLEARLY expresses a positive result or success worth a testimonial: feeling "
        "better, improved energy / sleep / focus / function, relief, or a shift they are happy about. "
        "Do NOT count neutral questions, requests, complaints, or still-unresolved problems. "
        "Return STRICT JSON ONLY, no prose: {\"positive\": bool, \"confidence\": number 0..1, "
        "\"quote\": \"the client's own short positive phrase, verbatim\", "
        "\"kind\": \"remedy\" | \"service\" | \"general\"}."
    )
    user = f"Client communications:\n{comms_text or '(none)'}\n\nReturn only the JSON object."
    return {"system": system, "user": user}


def _parse_json(text):
    try:
        t = (text or "").strip()
        a, b = t.find("{"), t.rfind("}")
        if a < 0 or b < 0:
            return {}
        return json.loads(t[a:b + 1])
    except Exception:
        return {}


def classify_positive_result(comms_text, complete):
    """-> {positive: bool, confidence: float 0..1, quote: str, kind: remedy|service|general}.
    Safe default (positive=False) on empty input or unparseable model output."""
    if not (comms_text or "").strip():
        return dict(_DEFAULT)
    p = build_positive_prompt(comms_text)
    data = _parse_json(complete(p["system"], p["user"]))
    if not data:
        return dict(_DEFAULT)
    try:
        conf = float(data.get("confidence") or 0)
    except (TypeError, ValueError):
        conf = 0.0
    kind = data.get("kind") if data.get("kind") in _KINDS else "general"
    return {
        "positive": bool(data.get("positive")),
        "confidence": max(0.0, min(1.0, conf)),
        "quote": str(data.get("quote") or "").strip()[:300],
        "kind": kind,
    }
