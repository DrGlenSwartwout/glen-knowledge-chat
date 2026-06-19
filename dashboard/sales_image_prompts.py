IMAGE_KINDS = ("botanical", "mechanism")

def _ingredient_names(product):
    out = []
    for ing in (product.get("ingredients") or []):
        if isinstance(ing, dict) and ing.get("name"): out.append(ing["name"])
        elif isinstance(ing, str) and ing.strip(): out.append(ing.strip())
    return out

# Two style directives per kind so the variants are genuinely distinct (Phase-4 A/B).
_BOTANICAL_VARIANTS = [
    "warm natural daylight, eye-level composition",
    "soft golden-hour light, slightly elevated three-quarter angle",
]
_MECHANISM_VARIANTS = [
    "clean studio render, deep teal background",
    "luminous dark background with volumetric light, dramatic angle",
]

def build_image_prompts(product):
    name = product.get("name", "")
    ings = _ingredient_names(product)
    ing_phrase = ", ".join(ings[:6]) if ings else "fresh botanicals"
    botanical = [
        (f"Photo-quality botanical lifestyle scene for the supplement '{name}': the formula's fresh and "
         f"powdered botanical ingredients ({ing_phrase}) arranged on a natural wooden kitchen counter, an "
         f"attractive mature woman preparing them, a lush herb garden visible behind her; {style}.")
        for style in _BOTANICAL_VARIANTS
    ]
    mechanism = [
        (f"Photo-quality conceptual mechanism render for the supplement '{name}': a living human cell "
         f"surrounded by a radiant protective energy field, nourished by the formula's key compounds "
         f"({ing_phrase}), conveying cellular resilience and protection; {style}.")
        for style in _MECHANISM_VARIANTS
    ]
    return {"botanical": botanical, "mechanism": mechanism}
