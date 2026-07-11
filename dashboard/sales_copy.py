NARRATIVE_SECTIONS = ("intro", "description", "research")

COMPLIANCE = (
    "Use structure/function language only (supports, promotes, helps maintain). "
    "Do NOT claim to diagnose, treat, cure, or prevent any disease. Make no medical "
    "claims and cite no invented studies. This is educational and not a substitute for "
    "medical advice. Do not use em dashes; use commas."
)

SECTION_BRIEFS = {
    "intro": ("Write ONE warm, concrete paragraph (about 2-4 sentences): what this product "
              "does for the person and why it matters, grounded in its ingredients or, when no "
              "ingredients are listed, in what the product is and how it works."),
    "description": ("Write a fuller plain-language overview in 2-3 short paragraphs: what the "
                    "product is, what it is built from or how it works, and who it is for."),
    "research": ("Explain how it works in lay language, 1-2 short paragraphs, grounded in the "
                 "mechanisms of the listed ingredients or, when none are listed, the product's "
                 "described mechanism."),
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
    desc = (product.get("description") or "").strip()
    brief = SECTION_BRIEFS[section]
    # Products are formulas AND devices/tools/books. Devices have no ingredient list, so the
    # prompt must ground them in the authored description and forbid inventing a formula or
    # asking for a "missing" ingredient stack (that produced hallucinated nutritional-formula
    # copy and literal LLM refusals on device pages).
    system = ("You are writing sales-page copy for Dr. Glen Swartwout's products, which include "
              "nutritional formulas as well as devices, tools, and books. Voice: warm, clinically "
              "grounded, and specific: no fluff, no AI-pleasantry filler, no cliches. Write only "
              "about THIS product, using the details given below. Some products (devices, tools, "
              "books) have no ingredient list: for those, ground the copy in the product "
              "description and how it works. Never ask for missing information, and never invent "
              "an ingredient list, a formula, or nutrients the product does not have. " + COMPLIANCE)
    parts = [f"Product: {name}"]
    if desc:
        parts.append(f"Product description:\n{desc}")
    parts.append("Ingredient stack:\n" + (ings or "(none: this product has no ingredient list)"))
    parts.append(f"Task: {brief}\n\nReturn only the copy itself, with no headings, labels, or preamble.")
    user = "\n\n".join(parts)
    return system, user
