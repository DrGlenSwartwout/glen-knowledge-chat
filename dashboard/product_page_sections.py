"""Which /begin/product/<slug> sections to show for a given product.

The ingredient list, the formula-vs-formula comparison table, and — rendered by
the frontend *inside* the comparison section — the Miron violet-glass rotator +
"Learn the science" story are all formulation-only. They are meaningless (and
misleading) on a device/tool/book SKU that has no ingredient list and does not
ship in Miron glass, so they are dropped for those products. The Miron
educational video ("How Miron violet glass is made") is likewise not appended for
them, so the Watch section is dropped too unless the product has a video of its own.
"""

# Sections that only make sense when the product has an ingredient list.
FORMULATION_ONLY = ("ingredients", "comparison")


def filter_sections(sections, *, has_ingredients, has_own_video):
    """Return `sections` minus the formulation-only ones when the product has no
    ingredient list. `has_own_video` keeps the Watch section for a device that
    carries its own product video (only the Miron educational clip is withheld)."""
    if has_ingredients:
        return list(sections)
    drop = set(FORMULATION_ONLY)
    if not has_own_video:
        drop.add("video")
    return [s for s in sections if s.get("id") not in drop]
