"""AI drafting + compliance gate for public topic pages.

Mirrors dashboard/ingredient_copy.py for prompts. Adds a wellness-framed brief,
catalog link validation, and a fail-closed compliance scan.
"""
import json
import re

NARRATIVE_SECTIONS = ("overview", "contributing_factors", "what_people_explore")

_MODEL = "claude-haiku-4-5-20251001"

COMPLIANCE = (
    "Use educational, structure/function language only (supports, promotes, helps maintain). "
    "Do NOT claim to diagnose, treat, cure, reverse, or prevent any disease, and name no "
    "disease as something this addresses. Prefer framing like 'people exploring X often look "
    "into Y'. Describe observations, not medical outcomes or probabilities. This is "
    "educational and not a substitute for medical advice. Do not use em dashes; use commas."
)

SECTION_BRIEFS = {
    "overview": (
        "Write ONE warm, plain-language paragraph (2-4 sentences) describing what this "
        "{kind} is in everyday terms and why people pay attention to it. Educational tone."
    ),
    "contributing_factors": (
        "Write a short lay-language paragraph (2-4 sentences) on the lifestyle, nutritional, "
        "and environmental factors people commonly associate with this {kind}. "
        "Frame as common associations, not causation or diagnosis."
    ),
    "what_people_explore": (
        "Write a short paragraph (2-4 sentences) on the wellness directions people commonly "
        "explore around this {kind} (nutrition, daily habits, targeted support). "
        "Use 'people often explore' framing. Make no promises and name no disease."
    ),
}

# Disease/condition anchor: category words plus common named conditions. A claim verb only
# trips the local gate when one of these sits just after it (verb-then-disease structure).
_DISEASE = (
    r"(?:disease|illness|condition|disorder|syndrome|infection|ailment|"
    r"cancer|tumou?r|diabetes|arthritis|asthma|eczema|psoriasis|hypertension|"
    r"depression|anxiety|alzheimer'?s?|dementia|parkinson'?s?|autism|adhd|"
    r"ibs|crohn'?s?|colitis|lupus|fibromyalgia|migraine|insomnia|allerg(?:y|ies)|"
    r"influenza|covid|copd|osteoporosis|neuropathy|gout)"
)

# Claim verbs (verb forms only -- NOT the bare noun "treatment", which over-blocked
# "water treatment"). These flag only when a disease/condition word follows within ~30 chars.
_CLAIM_VERBS = r"(?:cure[sd]?|treat(?:s|ed|ing)?|reverse[sd]?|prevent(?:s|ed|ing)?|heal(?:s|ed|ing)?)"

# Hard denylist: any match in draft copy fails the gate locally, no model needed.
_BANNED = [
    (rf"\b{_CLAIM_VERBS}\b[\w\s,'-]{{0,30}}\b{_DISEASE}\b",
     "claims to treat/cure/reverse/prevent a disease"),
    (r"\bdiagnos(e|es|is|ing)\b", "claims to diagnose"),
    (r"\bguarantee[sd]?\b", "outcome guarantee"),
]


def build_section_prompt(section, topic):
    """Return (system, user) for one section and a topic dict {name, kind}."""
    name = topic.get("name", "")
    kind = topic.get("kind", "topic") or "topic"
    brief = SECTION_BRIEFS[section].format(kind=kind)
    system = (
        "You are writing public, SEO-friendly educational copy about a health topic for "
        "Dr. Glen Swartwout's wellness site. Voice: warm, clear, specific, no fluff, no "
        "clichés. " + COMPLIANCE
    )
    user = (
        f"Topic: {name} (kind: {kind})\n\n"
        f"Task: {brief}\n\n"
        "Return only the copy itself, with no headings, labels, or preamble."
    )
    return system, user


def _text_of(msg):
    return "".join(getattr(b, "text", "") for b in msg.content
                   if getattr(b, "type", "") == "text").strip()


def propose_curation(topic, client):
    """Propose SEO title + meta + raw related slugs. Never raises."""
    safe = {"title": topic.get("name", ""), "meta_description": "",
            "links": {"ingredients": [], "products": [], "topics": []}}
    try:
        name = topic.get("name", "")
        kind = topic.get("kind", "topic") or "topic"
        system = (
            "You return ONLY valid JSON with keys:\n"
            "  title: a concise SEO page title (max 60 chars)\n"
            "  meta_description: one plain sentence (max 155 chars), no disease claims\n"
            "  links: {\"ingredients\": [kebab-case slug...], \"products\": [slug...], "
            "\"topics\": [slug...]} of clearly related items.\n"
            "Slugs are lowercase, hyphens only. Only propose links you are confident relate. "
            "No disease claims. No em dashes. Return ONLY the JSON object."
        )
        user = f"Topic: {name} (kind: {kind}). Propose the curation JSON now."
        msg = client.messages.create(model=_MODEL, max_tokens=600, system=system,
                                     messages=[{"role": "user", "content": user}])
        raw = _text_of(msg)
        if raw.startswith("```"):
            raw = "\n".join(l for l in raw.splitlines() if not l.strip().startswith("```")).strip()
        data = json.loads(raw)
        links = data.get("links") or {}
        return {
            "title": (data.get("title") or name).strip()[:60],
            "meta_description": (data.get("meta_description") or "").strip()[:155],
            "links": {
                "ingredients": [str(s).strip() for s in (links.get("ingredients") or []) if str(s).strip()],
                "products": [str(s).strip() for s in (links.get("products") or []) if str(s).strip()],
                "topics": [str(s).strip() for s in (links.get("topics") or []) if str(s).strip()],
            },
        }
    except Exception as exc:  # noqa: BLE001 - never raises
        print(f"[topic-copy] propose_curation failed: {exc}", flush=True)
        return safe


def validate_links(links_raw, *, ingredient_slugs, product_slugs, topic_slugs):
    """Drop any proposed slug not present in the real catalog. Pure."""
    def _keep(slugs, catalog):
        out, seen = [], set()
        for s in (slugs or []):
            s = str(s).strip()
            if s and s in catalog and s not in seen:
                out.append({"slug": s, "name": catalog[s]})
                seen.add(s)
        return out
    links_raw = links_raw or {}
    return {
        "ingredients": _keep(links_raw.get("ingredients"), ingredient_slugs or {}),
        "products": _keep(links_raw.get("products"), product_slugs or {}),
        "topics": _keep(links_raw.get("topics"), topic_slugs or {}),
    }


def local_claim_flags(content):
    """Pure regex denylist over all section text. Returns [{phrase, reason}]."""
    text = " ".join(str(v) for v in (content or {}).values()).lower()
    flags = []
    for pattern, reason in _BANNED:
        m = re.search(pattern, text)
        if m:
            flags.append({"phrase": m.group(0), "reason": reason})
    return flags


def compliance_scan(content, client):
    """Fail-closed compliance gate. Local denylist first, then a model judgment."""
    import datetime
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    local = local_claim_flags(content)
    if local:
        return {"passed": False, "flags": local, "scanned_at": now, "model": "local"}
    try:
        text = "\n\n".join(f"[{k}] {v}" for k, v in (content or {}).items())
        system = (
            "You are an FDA/FTC compliance reviewer for supplement wellness copy. Return ONLY "
            "JSON: {\"passed\": bool, \"flags\": [{\"phrase\": str, \"reason\": str}]}. "
            "passed=false if the copy claims to diagnose, treat, cure, reverse, or prevent any "
            "disease, names a disease as something it addresses, or promises a medical outcome. "
            "Structure/function and 'people explore' framing is allowed. Return ONLY the JSON."
        )
        msg = client.messages.create(model=_MODEL, max_tokens=500, system=system,
                                     messages=[{"role": "user", "content": text}])
        raw = _text_of(msg)
        if raw.startswith("```"):
            raw = "\n".join(l for l in raw.splitlines() if not l.strip().startswith("```")).strip()
        data = json.loads(raw)
        passed = bool(data.get("passed"))
        flags = data.get("flags") or []
        flags = [{"phrase": str(f.get("phrase", "")), "reason": str(f.get("reason", ""))}
                 for f in flags if isinstance(f, dict)]
        return {"passed": passed and not flags, "flags": flags, "scanned_at": now, "model": _MODEL}
    except Exception as exc:  # noqa: BLE001 - fail closed
        print(f"[topic-copy] compliance_scan failed: {exc}", flush=True)
        return {"passed": False, "flags": [{"phrase": "", "reason": "scan error (fail-closed)"}],
                "scanned_at": now, "model": "error"}


def extract_topic_candidate(query, answer, client):
    """Name the single health topic a conversation is about, for the create-a-page offer.

    Returns {"name","kind","slug"} or None. One haiku call; never raises.
    """
    try:
        from dashboard import ingredients as _ingredients
        system = (
            "You decide whether a chat is centrally about ONE health topic a person would search "
            "for (a symptom, a named condition, or a physiological function). Return ONLY JSON. "
            "If yes: {\"name\": \"Title Case Topic\", \"kind\": \"symptom|condition|function\"}. "
            "If it is small talk, multiple unrelated topics, or not health, return {}. "
            "No commentary, no markdown."
        )
        user = f"User: {query}\n\nAssistant answer: {answer[:600]}\n\nReturn the JSON now."
        msg = client.messages.create(model=_MODEL, max_tokens=120, system=system,
                                     messages=[{"role": "user", "content": user}])
        raw = _text_of(msg)
        if raw.startswith("```"):
            raw = "\n".join(l for l in raw.splitlines() if not l.strip().startswith("```")).strip()
        data = json.loads(raw)
        name = (data.get("name") or "").strip()
        kind = (data.get("kind") or "").strip().lower()
        if not name or kind not in ("symptom", "condition", "function"):
            return None
        return {"name": name, "kind": kind, "slug": _ingredients.slugify(name)}
    except Exception as exc:  # noqa: BLE001 - never raises
        print(f"[topic-copy] extract_topic_candidate failed: {exc}", flush=True)
        return None
