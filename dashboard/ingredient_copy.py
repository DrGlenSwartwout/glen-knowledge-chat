"""AI copy generation for ingredient pages.

Mirrors dashboard/sales_copy.py: NARRATIVE_SECTIONS, build_section_prompt,
and propose_curation (the curation equivalent of product page-gen).
"""
import json

from dashboard import ingredients as _ingredients

NARRATIVE_SECTIONS = ("what_it_is", "research")

COMPLIANCE = (
    "Use structure/function language only (supports, promotes, helps maintain). "
    "Do NOT claim to diagnose, treat, cure, or prevent any disease. Make no medical "
    "claims and cite no invented studies. This is educational and not a substitute for "
    "medical advice. Do not use em dashes; use commas."
)

_MODEL = "claude-haiku-4-5-20251001"

SECTION_BRIEFS = {
    "what_it_is": (
        "Write ONE warm, concrete paragraph (about 2-4 sentences): what this ingredient is, "
        "its primary structure/function role in the body, and why it matters. "
        "Ground it in the ingredient's known chemistry and physiological roles."
    ),
    "research": (
        "Write a heavy lay-language summary (2-4 paragraphs) synthesizing the listed research "
        "studies. Explain the mechanisms the studies highlight, what they found, and the "
        "overall weight of evidence. Cite mechanisms, not brand names. "
        "Do not invent studies or exaggerate findings. "
        "The raw study citations are shown separately, so focus on synthesis and meaning."
    ),
}


def _fmp_lines(ingredient):
    fmp = ingredient.get("fmp") or {}
    lines = []
    for key in ("scientific", "label_form", "percent", "active", "rda_content", "rda_mg"):
        v = fmp.get(key)
        if v is not None and str(v).strip():
            lines.append(f"- {key}: {v}")
    return "\n".join(lines)


def _study_lines(ingredient):
    studies = ingredient.get("studies") or []
    out = []
    for s in studies[:12]:
        title = s.get("study_title") or s.get("title") or ""
        pub = s.get("publication") or s.get("source") or ""
        year = s.get("year") or ""
        text = s.get("text") or ""
        line_parts = [p for p in [title, pub, str(year) if year else ""] if p]
        header = " | ".join(line_parts)
        if text:
            out.append(f"[{header}] {text[:300]}")
        elif header:
            out.append(f"[{header}]")
    return "\n".join(out)


def build_section_prompt(section, ingredient):
    """Return (system, user) tuple for the given section and ingredient dict."""
    name = ingredient.get("name", "")
    fmp_block = _fmp_lines(ingredient)
    study_block = _study_lines(ingredient)
    brief = SECTION_BRIEFS[section]

    system = (
        "You are writing ingredient-page copy for Dr. Glen Swartwout's nutritional supplement line. "
        "Voice: warm, clinically grounded, and specific: no fluff, no AI-pleasantry filler, "
        "no clichés. " + COMPLIANCE
    )
    user = (
        f"Ingredient: {name}\n\n"
        f"Structured data:\n{fmp_block or '(not available)'}\n\n"
        f"Research studies:\n{study_block or '(not available)'}\n\n"
        f"Task: {brief}\n\n"
        "Return only the copy itself, with no headings, labels, or preamble."
    )
    return system, user


def propose_curation(ingredient, client):
    """Propose research_score, traditional_score, related_forms, and traditional_use.

    Makes ONE synchronous haiku JSON call. Returns a safe-default dict on ANY failure
    (never raises). Clamps scores to 1-10. The prompt instructs the model to omit any
    classical formula it is not confident is real.
    """
    safe_default = {
        "research_score": None,
        "traditional_score": None,
        "related_forms": [],
        "traditional_use": [],
    }
    try:
        name = ingredient.get("name", "")
        study_block = _study_lines(ingredient)
        fmp_block = _fmp_lines(ingredient)

        system = (
            "You are a clinical nutritionist and herbalist. Return ONLY valid JSON with these keys:\n"
            "  research_score: integer 1-10 (1=minimal evidence, 10=robust RCTs; base on the studies provided)\n"
            "  traditional_score: integer 1-10 (1=no traditional use, 10=central to multiple systems)\n"
            "  related_forms: array of {\"name\": str, \"slug\": str, \"verdict\": \"superior\"|\"inferior\"|\"comparable\", \"note\": str}\n"
            "  traditional_use: array of {\"system\": str, \"formula\": str, \"uses\": str, \"forms\": str}\n"
            "CRITICAL for related_forms: only list other molecular forms of the same nutrient you are "
            "confident exist. Set slug to the kebab-case name (lowercase, hyphens only, max 40 chars). "
            "CRITICAL for traditional_use: OMIT any classical formula you are not confident is a real, "
            "historically documented formula. Do not invent formulas. Only include entries you are certain about. "
            "No em dashes; use commas. No disease claims. Return ONLY the JSON object, no commentary."
        )
        user = (
            f"Ingredient: {name}\n\n"
            f"Structured data:\n{fmp_block or '(not available)'}\n\n"
            f"Research studies:\n{study_block or '(not available)'}\n\n"
            "Propose the curation JSON now."
        )

        msg = client.messages.create(
            model=_MODEL,
            max_tokens=1200,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        raw = "".join(
            getattr(b, "text", "")
            for b in msg.content
            if getattr(b, "type", "") == "text"
        ).strip()

        # strip markdown fences if present
        if raw.startswith("```"):
            lines = raw.splitlines()
            raw = "\n".join(
                l for l in lines if not l.strip().startswith("```")
            ).strip()

        data = json.loads(raw)

        def _clamp(v):
            if v is None:
                return None
            try:
                return max(1, min(10, int(v)))
            except (TypeError, ValueError):
                return None

        # Normalize related_forms slugs via the slugify function
        related = []
        for item in (data.get("related_forms") or []):
            if not isinstance(item, dict):
                continue
            nm = (item.get("name") or "").strip()
            if not nm:
                continue
            slug = _ingredients.slugify(nm)
            related.append({
                "name": nm,
                "slug": slug,
                "verdict": (item.get("verdict") or "comparable").strip(),
                "note": (item.get("note") or "").strip(),
            })

        trad = []
        for item in (data.get("traditional_use") or []):
            if not isinstance(item, dict):
                continue
            trad.append({
                "system": (item.get("system") or "").strip(),
                "formula": (item.get("formula") or "").strip(),
                "uses": (item.get("uses") or "").strip(),
                "forms": (item.get("forms") or "").strip(),
            })

        return {
            "research_score": _clamp(data.get("research_score")),
            "traditional_score": _clamp(data.get("traditional_score")),
            "related_forms": related,
            "traditional_use": trad,
        }

    except Exception as exc:  # noqa: BLE001 - propose_curation must never raise
        print(f"[ingredient-copy] propose_curation failed: {exc}", flush=True)
        return safe_default
