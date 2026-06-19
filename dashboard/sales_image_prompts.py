IMAGE_KINDS = ("botanical", "mechanism")

# Hard exclusions appended to every prompt. Flux renders garbled "labels" when
# ingredient or product names are in the prompt, so we keep names OUT entirely
# and explicitly forbid any text or product packaging in the image.
_NO_TEXT = ("No text, no words, no letters, no numbers, no labels, no captions, no logos, "
            "and no product packaging, bottles, jars, tubes, or containers anywhere in the image.")

# Two style directives per kind so the variants are genuinely distinct (Phase-4 A/B).
_BOTANICAL_VARIANTS = [
    "warm natural daylight, eye-level composition",
    "soft golden-hour light, slightly elevated three-quarter angle",
]
_MECHANISM_VARIANTS = [
    "clean studio render, deep teal background",
    "luminous dark background with volumetric light, dramatic angle",
]


def build_image_prompts(product=None):
    """Two image prompts per kind (botanical lifestyle + mechanism), with NO text,
    labels, or product packaging. `product` is accepted for interface stability but
    intentionally unused — injecting names is what makes Flux render text."""
    botanical = [
        ("Photo-quality botanical wellness lifestyle scene: an abundance of fresh herbs, "
         "green leaves, flowers, roots, and colorful whole botanical ingredients arranged on a "
         "natural wooden kitchen counter, an attractive mature woman gently preparing fresh "
         f"herbs, a lush green herb garden visible behind her. {_NO_TEXT} {style}.")
        for style in _BOTANICAL_VARIANTS
    ]
    mechanism = [
        ("Photo-quality conceptual render: a single glowing living human cell surrounded by a "
         "radiant protective energy field, luminous particles flowing inward toward it, "
         f"conveying cellular resilience, vitality, and protection. {_NO_TEXT} {style}.")
        for style in _MECHANISM_VARIANTS
    ]
    return {"botanical": botanical, "mechanism": mechanism}
