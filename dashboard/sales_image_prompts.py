IMAGE_KINDS = ("botanical", "mechanism")

# Hard exclusions appended to every prompt. Flux renders garbled "labels" when
# ingredient or product names are in the prompt, so we keep names OUT entirely
# and explicitly forbid any text or product packaging in the image.
_NO_TEXT = ("No text, no words, no letters, no numbers, no labels, no captions, no logos, "
            "and no product packaging, bottles, jars, tubes, or containers anywhere in the image.")
NO_TEXT = _NO_TEXT   # public alias

_BOTANICAL_BODY = ("Photo-quality botanical wellness lifestyle scene: an abundance of fresh herbs, "
                   "green leaves, flowers, roots, and colorful whole botanical ingredients arranged on a "
                   "natural wooden kitchen counter, an attractive mature woman gently preparing fresh "
                   "herbs, a lush green herb garden visible behind her.")
_MECHANISM_BODY = ("Photo-quality conceptual render: a single glowing living human cell surrounded by a "
                   "radiant protective energy field, luminous particles flowing inward toward it, "
                   "conveying cellular resilience, vitality, and protection.")
_BODY = {"botanical": _BOTANICAL_BODY, "mechanism": _MECHANISM_BODY}

_STYLES = {
    "botanical": ["warm natural daylight, eye-level composition",
                  "soft golden-hour light, slightly elevated three-quarter angle",
                  "bright airy morning light, overhead flat-lay composition",
                  "cozy warm interior light, close intimate framing"],
    "mechanism": ["clean studio render, deep teal background",
                  "luminous dark background with volumetric light, dramatic angle",
                  "iridescent blue-violet palette, centered symmetrical composition",
                  "warm amber glow on a black background, shallow depth of field"],
}


def build_one_prompt(kind, variant_index):
    """Return a single image prompt for `kind` using the style at
    `_STYLES[kind][(variant_index-1) % len]`."""
    styles = _STYLES[kind]
    style = styles[(int(variant_index) - 1) % len(styles)]
    return f"{_BODY[kind]} {_NO_TEXT} {style}."


def build_image_prompts(product=None):
    """Two image prompts per kind (botanical lifestyle + mechanism), with NO text,
    labels, or product packaging. `product` is accepted for interface stability but
    intentionally unused — injecting names is what makes Flux render text."""
    return {k: [f"{_BODY[k]} {_NO_TEXT} {_STYLES[k][i]}." for i in range(2)] for k in IMAGE_KINDS}
