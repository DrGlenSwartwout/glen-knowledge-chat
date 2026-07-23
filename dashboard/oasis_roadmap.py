"""My Healing Oasis — recommended-additions roadmap.

Flat, all-phase recommendations for a client's Healing Oasis:
  1. HERO_TOOLS     — the near-universal tools Glen recommends to almost
     everyone (fixed lead order: harmony-laser -> water-ionizer-15plate ->
     kloud-pemf-maxi). These sort ABOVE the secondary tail because they are
     foundational, not situational.
  2. SECONDARY_TOOLS — an unprioritized tail of every other catalog tool
     Glen recommends, grouped by modality (see CATEGORY_ORDER) for display.
     Healing tools work in all phases, so there is no per-terrain-phase
     bucketing (an earlier per-phase TERRAIN_TOOLS dict was removed for
     this reason -- see build_roadmap's docstring). `nes-mihealth`
     intentionally appears TWICE (PEMF and Microcurrent -- Glen: it does
     both) -- build_roadmap preserves this duplicate rather than deduping
     secondary entries against each other.

Pure module, no DB — build_roadmap() takes owned_slugs (see
dashboard/owned_tools.owned_slugs()) and returns the client's personal gap
list. Glen is expected to extend HERO_TOOLS / SECONDARY_TOOLS as his
product line and recommendations grow.
"""

# The three near-universal tools, in FIXED lead order. Order is significant:
# it is the order clients see them in when none are owned yet. Slugs are
# the real catalog SKUs (see data/products.json), not simplified aliases.
# Glen extends this list as more near-universal tools are confirmed.
HERO_TOOLS = [
    {
        "slug": "harmony-laser",
        "name": "Harmony Laser",
        "why": "Low-level laser therapy Glen recommends to nearly every client for "
               "cellular energy and pain/inflammation, in every terrain phase.",
        "tier": "hero",
    },
    {
        "slug": "water-ionizer-15plate",
        "name": "Water Ionizer (15-Plate)",
        "why": "15 plates for maximum molecular hydrogen; a daily-use foundation for "
               "every phase (9-plate and 5-plate options available).",
        "tier": "hero",
    },
    {
        "slug": "kloud-pemf-maxi",
        "name": "Kloud PEMF Mat (Maxi)",
        "why": "Pulsed electromagnetic field therapy with a setting for each phase, "
               "supporting circulation, recovery, and cellular voltage for nearly "
               "every client (Mini option available).",
        "tier": "hero",
    },
]

# Display order for secondary tool categories. Glen extends this as new
# modalities are confirmed.
CATEGORY_ORDER = [
    "Light", "Water", "Air", "PEMF", "Microcurrent", "EMF", "Sound",
    "Bioenergetic", "Detox",
]

# Secondary tools: an unprioritized list of every other tool Glen
# recommends, grouped by modality (`category`, one of CATEGORY_ORDER) for
# display. No per-item `why` -- these are an unprioritized list, not a
# ranked one. `nes-mihealth` intentionally appears TWICE (PEMF and
# Microcurrent) -- see build_roadmap, which preserves this duplicate.
# Glen: extend this list as more tools are confirmed.
SECONDARY_TOOLS = [
    # Light
    {"slug": "acupuncture-point-cold-laser", "name": "Acupuncture Point Cold Laser", "category": "Light", "tier": "secondary", "why": ""},
    {"slug": "infrared-therapy-flashlight", "name": "Infrared Therapy Flashlight", "category": "Light", "tier": "secondary", "why": ""},
    {"slug": "red-mitochondrial-therapy-630nm", "name": "Red Mitochondrial Therapy 630 nm", "category": "Light", "tier": "secondary", "why": ""},
    {"slug": "nir-brain-frequency-helmet", "name": "NIR Brain Frequency Helmet", "category": "Light", "tier": "secondary", "why": ""},
    {"slug": "nir-nasal-clip", "name": "NIR Near-Infrared Nasal Clip", "category": "Light", "tier": "secondary", "why": ""},
    {"slug": "hair-growth-helmet", "name": "Hair Growth Helmet", "category": "Light", "tier": "secondary", "why": ""},
    {"slug": "biocompatible-nightlight", "name": "Biocompatible Nightlight", "category": "Light", "tier": "secondary", "why": ""},
    {"slug": "therapeutic-nightlight", "name": "Therapeutic Nightlight", "category": "Light", "tier": "secondary", "why": ""},
    {"slug": "blue-blocking-photochromic-sunglasses", "name": "Blue Blocking Photochromic Sunglasses", "category": "Light", "tier": "secondary", "why": ""},
    # Water
    {"slug": "miracule-water-system", "name": "Miracule Water System with Molecular Hydrogen", "category": "Water", "tier": "secondary", "why": ""},
    {"slug": "molecular-hydrogen-bottle", "name": "Molecular Hydrogen (portable bottle)", "category": "Water", "tier": "secondary", "why": ""},
    {"slug": "shower-filter", "name": "Shower Filter", "category": "Water", "tier": "secondary", "why": ""},
    # Air
    {"slug": "air-surface-pro-plus", "name": "Air & Surface PRO+", "category": "Air", "tier": "secondary", "why": ""},
    {"slug": "freshair-mobile-purifier", "name": "FreshAir Mobile Portable Air Purifier", "category": "Air", "tier": "secondary", "why": ""},
    {"slug": "car-ionizer", "name": "Car Ionizer", "category": "Air", "tier": "secondary", "why": ""},
    {"slug": "wearable-ionizer", "name": "Ionizer - Wearable", "category": "Air", "tier": "secondary", "why": ""},
    # PEMF  (nes-mihealth listed here AND in Microcurrent -- intentional, see module docstring)
    {"slug": "nes-mihealth", "name": "NES miHealth", "category": "PEMF", "tier": "secondary", "why": ""},
    # Microcurrent
    {"slug": "denas-scenar", "name": "DENAS PCM Pro (SCENAR Microcurrent)", "category": "Microcurrent", "tier": "secondary", "why": ""},
    {"slug": "denas-eyeglasses-electrode", "name": "DENAS Eyeglasses Electrode", "category": "Microcurrent", "tier": "secondary", "why": ""},
    {"slug": "denas-microcurrent-eye-system", "name": "DENAS Microcurrent System for Eye Healing", "category": "Microcurrent", "tier": "secondary", "why": ""},
    {"slug": "microgen-microcurrent-generator", "name": "Microgen Wearable Microcurrent Generator", "category": "Microcurrent", "tier": "secondary", "why": ""},
    {"slug": "vagus-nerve-stimulation-kit", "name": "Vagus Nerve Stimulation Kit for miHealth", "category": "Microcurrent", "tier": "secondary", "why": ""},
    {"slug": "nes-mihealth", "name": "NES miHealth", "category": "Microcurrent", "tier": "secondary", "why": ""},
    # EMF
    {"slug": "emf-free-headset", "name": "EMF Free Headset", "category": "EMF", "tier": "secondary", "why": ""},
    {"slug": "neutralizer-3-pack", "name": "The Neutralizer 3 Pack", "category": "EMF", "tier": "secondary", "why": ""},
    {"slug": "whole-house-neutralizer", "name": "Whole House Neutralizer", "category": "EMF", "tier": "secondary", "why": ""},
    {"slug": "car-neutralizer-usb", "name": "Car Neutralizer USB", "category": "EMF", "tier": "secondary", "why": ""},
    {"slug": "shungite-stick-plate", "name": "Shungite Stick Plate", "category": "EMF", "tier": "secondary", "why": ""},
    {"slug": "aulterra-energy-pendant-silver-gold", "name": "Aulterra Energy Pendant Silver & Gold", "category": "EMF", "tier": "secondary", "why": ""},
    {"slug": "mithreal-silver-baseball-cap", "name": "Baseball Cap with Mithreal Silver", "category": "EMF", "tier": "secondary", "why": ""},
    {"slug": "mithreal-silver-blue-zipper-hoodie", "name": "Blue Zipper Hoodie with Mithreal Silver", "category": "EMF", "tier": "secondary", "why": ""},
    {"slug": "mithreal-silver-ivy-hat", "name": "Ivy Hat with Mithreal Silver", "category": "EMF", "tier": "secondary", "why": ""},
    # Sound
    {"slug": "breath-tuning-fork-1283hz", "name": "Breath Tuning Fork 1283 Hz", "category": "Sound", "tier": "secondary", "why": ""},
    {"slug": "mind-tuning-fork-5000hz", "name": "Mind Tuning Fork 5000 Hz", "category": "Sound", "tier": "secondary", "why": ""},
    {"slug": "spirit-tuning-fork-172hz", "name": "Spirit Tuning Fork 172 Hz", "category": "Sound", "tier": "secondary", "why": ""},
    {"slug": "frosted-quartz-tuning-fork-172hz", "name": "Frosted Quartz Tuning Fork 172 Hz", "category": "Sound", "tier": "secondary", "why": ""},
    {"slug": "tibetan-singing-bowl-172hz", "name": "Tibetan Glass Singing Bowl 172 Hz", "category": "Sound", "tier": "secondary", "why": ""},
    # Bioenergetic
    {"slug": "nes-scanner", "name": "NES Scanner", "category": "Bioenergetic", "tier": "secondary", "why": ""},
    {"slug": "bioenergetic-wellness-scanner", "name": "Bioenergetic Wellness Scanner", "category": "Bioenergetic", "tier": "secondary", "why": ""},
    {"slug": "hand-cradle", "name": "ZYTO Hand Cradle", "category": "Bioenergetic", "tier": "secondary", "why": ""},
    {"slug": "dowsing-rods", "name": "Dowsing Rods", "category": "Bioenergetic", "tier": "secondary", "why": ""},
    # Detox
    {"slug": "wicking-toothbrush", "name": "Wicking Toothbrush", "category": "Detox", "tier": "secondary", "why": ""},
]


def build_roadmap(owned_slugs, terrain_phase=None):
    """Personal recommended-additions roadmap: hero tools (fixed order, not
    owned), THEN secondary tools (list/category order, not owned). Heroes
    are deduped by slug against each other and against owned_slugs; a
    secondary slug already used by a HERO is also skipped (none currently
    overlap). Secondary entries are deliberately NOT deduped against each
    other -- `nes-mihealth` intentionally appears twice (PEMF and
    Microcurrent, see SECONDARY_TOOLS) and both copies are preserved here.
    `owned_slugs` still filters by slug, so owning `nes-mihealth` removes
    BOTH of its entries. Every item keeps {slug, name, why, tier}; secondary
    items also carry `category` (one of CATEGORY_ORDER).

    `terrain_phase` is accepted for backward-compatible call signatures
    (callers/tests/plumbing) but is now IGNORED: Glen's direction is that
    healing tools work in all phases, so there is no per-phase tool
    selection any more -- the result is identical for every phase or None.
    Pure, no DB."""
    owned_slugs = owned_slugs or set()
    hero_seen = set()
    roadmap = []

    for tool in HERO_TOOLS:
        slug = tool["slug"]
        if slug in owned_slugs or slug in hero_seen:
            continue
        hero_seen.add(slug)
        item = dict(tool)
        item.setdefault("why", "")
        roadmap.append(item)

    for tool in SECONDARY_TOOLS:
        slug = tool["slug"]
        if slug in owned_slugs or slug in hero_seen:
            continue
        item = dict(tool)
        item.setdefault("why", "")
        roadmap.append(item)

    return roadmap
