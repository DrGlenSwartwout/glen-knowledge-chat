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


_VIDEO_RISK = (
    "Set publish_risk=true with a short risk_reasons when the spoken review claims to diagnose, "
    "treat, cure, or prevent a disease, names a disease as cured, or contains personal contact info "
    "(PII). This flags a PUBLISHING risk for a human reviewer; it does NOT lower video_points."
)


def build_video_prompt(product, transcript):
    name = (product or {}).get("name", "")
    system = (
        "You score the transcript of a short spoken customer video review of Dr. Glen Swartwout's "
        "supplements. Return ONLY a JSON object with keys: video_points (integer 0..5), "
        "publish_risk (bool), risk_reasons (short string), recommend_publish (bool). "
        "video_points rewards a clear, specific, authentic spoken experience; low-effort, vague, "
        "spammy, or abusive transcripts score 0. " + _VIDEO_RISK)
    user = (f"Product: {name}\n\nVideo transcript:\n{transcript or '(empty)'}\n\n"
            "Return only the JSON object, no prose.")
    return system, user


def _safe_video_default(reasons):
    return {"video_points": 0, "publish_risk": False, "risk_reasons": reasons, "recommend_publish": False}


def score_video(client, product, transcript, *, strip=lambda s: s):
    system, user = build_video_prompt(product, transcript)
    try:
        msg = client.messages.create(model=_MODEL, max_tokens=300, system=system,
                                      messages=[{"role": "user", "content": user}])
        text = "".join(getattr(b, "text", "") for b in msg.content if getattr(b, "type", "") == "text").strip()
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < 0:
            return _safe_video_default("unparseable scoring response")
        data = json.loads(text[start:end + 1])
        return {
            "video_points": max(0, min(5, int(data.get("video_points", 0)))),
            "publish_risk": bool(data.get("publish_risk")),
            "risk_reasons": strip(str(data.get("risk_reasons", "")))[:500],
            "recommend_publish": bool(data.get("recommend_publish")),
        }
    except Exception as e:  # noqa: BLE001
        return _safe_video_default(f"scoring error: {e}")


def build_gift_prompt(review_text, product, order_history, catalog):
    name = (product or {}).get("name", "")
    items = "\n".join(f"- {g['sku']}: {g.get('label','')} ({g.get('description','')})" for g in catalog)
    hist = ", ".join(order_history or []) or "(none)"
    system = ("You pick a thank-you gift for a customer who left an excellent video review of "
              "Dr. Glen Swartwout's supplements. Choose ONE gift from the catalog that best fits this "
              "person. Return ONLY a JSON object: {\"sku\": <one catalog sku>, \"reason\": <short why>}.")
    user = (f"Reviewed product: {name}\nTheir recent orders: {hist}\n\nReview:\n{review_text or '(none)'}\n\n"
            f"Gift catalog:\n{items}\n\nReturn only the JSON object.")
    return system, user


def suggest_gift(client, review_text, product, order_history, catalog, *, strip=lambda s: s):
    if not catalog:
        return None
    valid = {g["sku"] for g in catalog}
    system, user = build_gift_prompt(review_text, product, order_history, catalog)
    try:
        msg = client.messages.create(model=_MODEL, max_tokens=200, system=system,
                                      messages=[{"role": "user", "content": user}])
        text = "".join(getattr(b, "text", "") for b in msg.content if getattr(b, "type", "") == "text").strip()
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < 0:
            return None
        data = json.loads(text[start:end + 1])
        sku = data.get("sku")
        if sku not in valid:
            return None
        return {"sku": sku, "reason": strip(str(data.get("reason", "")))[:300]}
    except Exception:  # noqa: BLE001
        return None
