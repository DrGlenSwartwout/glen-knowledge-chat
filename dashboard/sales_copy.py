NARRATIVE_SECTIONS = ("intro", "description", "research")

COMPLIANCE = (
    "Use structure/function language only (supports, promotes, helps maintain). "
    "Do NOT claim to diagnose, treat, cure, or prevent any disease. Make no medical "
    "claims and cite no invented studies. This is educational and not a substitute for "
    "medical advice. Do not use em dashes; use commas."
)

SECTION_BRIEFS = {
    "intro": ("Write ONE warm, concrete paragraph (about 2-4 sentences): what this formula "
              "does for the person and why it matters, grounded in its ingredients."),
    "description": ("Write a fuller plain-language overview in 2-3 short paragraphs: what the "
                    "formula is, what it's built from, and who it's for."),
    "research": ("Explain how it works in lay language, 1-2 short paragraphs, grounded in the "
                 "mechanisms of the listed ingredients."),
}

def _ingredient_lines(product):
    out = []
    for ing in (product.get("ingredients") or []):
        if isinstance(ing, dict):
            out.append((f"- {ing.get('name','')} {ing.get('dose','')}").rstrip())
        elif isinstance(ing, str) and ing.strip():
            out.append(f"- {ing.strip()}")
    return "\n".join(out)

def build_section_prompt(section, product):
    name = product.get("name", "")
    ings = _ingredient_lines(product)
    brief = SECTION_BRIEFS[section]
    system = ("You are writing sales-page copy for Dr. Glen Swartwout's nutritional formulas. "
              "Voice: warm, clinically grounded, and specific: no fluff, no AI-pleasantry filler, "
              "no clichés. " + COMPLIANCE)
    user = (f"Product: {name}\n\nIngredient stack:\n{ings or '(not specified)'}\n\n"
            f"Task: {brief}\n\nReturn only the copy itself, with no headings, labels, or preamble.")
    return system, user
