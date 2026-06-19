import json

_MODEL = "claude-haiku-4-5-20251001"

_COMPLIANCE = (
    "Reject (compliance_ok=false) any review that claims to diagnose, treat, cure, or prevent a "
    "disease, names a disease as cured/healed, contains personal contact info (PII), is spam, or is "
    "abusive. Otherwise compliance_ok=true. Use structure/function framing for what is acceptable."
)


def build_review_prompt(product, body):
    name = (product or {}).get("name", "")
    system = (
        "You score short customer product reviews for Dr. Glen Swartwout's supplements. "
        "Return ONLY a JSON object with keys: compliance_ok (bool), reasons (short string), "
        "quality_points (integer 0, 1, or 2), recommend_publish (bool). "
        "quality_points rewards specificity, authenticity, and usefulness to other shoppers, "
        "NOT length or keyword stuffing. " + _COMPLIANCE)
    user = (f"Product: {name}\n\nReview:\n{body or '(no written review)'}\n\n"
            "Return only the JSON object, no prose.")
    return system, user


def _safe_default(reasons):
    return {"compliance_ok": False, "reasons": reasons, "quality_points": 0, "recommend_publish": False}


def score_review(client, product, body, *, strip=lambda s: s):
    system, user = build_review_prompt(product, body)
    try:
        msg = client.messages.create(model=_MODEL, max_tokens=300, system=system,
                                      messages=[{"role": "user", "content": user}])
        text = "".join(getattr(b, "text", "") for b in msg.content if getattr(b, "type", "") == "text")
        text = text.strip()
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < 0:
            return _safe_default("unparseable scoring response")
        data = json.loads(text[start:end + 1])
        return {
            "compliance_ok": bool(data.get("compliance_ok")),
            "reasons": strip(str(data.get("reasons", "")))[:500],
            "quality_points": max(0, min(2, int(data.get("quality_points", 0)))),
            "recommend_publish": bool(data.get("recommend_publish")),
        }
    except Exception as e:  # noqa: BLE001
        return _safe_default(f"scoring error: {e}")
