"""My Healing Oasis — recommended-additions roadmap.

Flat, all-phase recommendations for a client's Healing Oasis:
  1. HERO_TOOLS     — the near-universal tools Glen recommends to almost
     everyone (fixed lead order: harmony-laser -> water-ionizer-15plate ->
     kloud-pemf-maxi). These sort ABOVE the secondary tail because they are
     foundational, not situational.
  2. SECONDARY_TOOLS — an unprioritized flat tail of every other catalog
     tool Glen recommends. Healing tools work in all phases, so there is no
     per-terrain-phase bucketing (an earlier per-phase TERRAIN_TOOLS dict
     was removed for this reason -- see build_roadmap's docstring).

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

# Secondary tools: a flat, unprioritized list of every other tool Glen
# recommends. No per-item `why` -- these are an unprioritized list, not a
# ranked one. Glen: extend this list as more tools are confirmed.
SECONDARY_TOOLS = [
    {"slug": "acupuncture-point-cold-laser", "name": "Acupuncture Point Cold Laser", "tier": "secondary"},
    {"slug": "air-surface-pro-plus", "name": "Air & Surface PRO+", "tier": "secondary"},
    {"slug": "aulterra-energy-pendant-silver-gold", "name": "Aulterra Energy Pendant Silver & Gold", "tier": "secondary"},
    {"slug": "mithreal-silver-baseball-cap", "name": "Baseball Cap with Mithreal Silver", "tier": "secondary"},
    {"slug": "biocompatible-nightlight", "name": "Biocompatible Nightlight", "tier": "secondary"},
    {"slug": "bioenergetic-wellness-scanner", "name": "Bioenergetic Wellness Scanner", "tier": "secondary"},
    {"slug": "blue-blocking-photochromic-sunglasses", "name": "Blue Blocking Photochromic Sunglasses", "tier": "secondary"},
    {"slug": "mithreal-silver-blue-zipper-hoodie", "name": "Blue Zipper Hoodie with Mithreal Silver", "tier": "secondary"},
    {"slug": "breath-tuning-fork-1283hz", "name": "Breath Tuning Fork 1283 Hz", "tier": "secondary"},
    {"slug": "car-ionizer", "name": "Car Ionizer", "tier": "secondary"},
    {"slug": "car-neutralizer-usb", "name": "Car Neutralizer USB", "tier": "secondary"},
    {"slug": "denas-eyeglasses-electrode", "name": "DENAS Eyeglasses Electrode", "tier": "secondary"},
    {"slug": "denas-microcurrent-eye-system", "name": "DENAS Microcurrent System for Eye Healing", "tier": "secondary"},
    {"slug": "denas-scenar", "name": "DENAS PCM Pro (SCENAR Microcurrent)", "tier": "secondary"},
    {"slug": "dowsing-rods", "name": "Dowsing Rods", "tier": "secondary"},
    {"slug": "emf-free-headset", "name": "EMF Free Headset", "tier": "secondary"},
    {"slug": "freshair-mobile-purifier", "name": "FreshAir Mobile Portable Air Purifier", "tier": "secondary"},
    {"slug": "frosted-quartz-tuning-fork-172hz", "name": "Frosted Quartz Tuning Fork 172 Hz", "tier": "secondary"},
    {"slug": "hair-growth-helmet", "name": "Hair Growth Helmet", "tier": "secondary"},
    {"slug": "hypoxia-free-face-shield", "name": "Hypoxia-Free Face Shield", "tier": "secondary"},
    {"slug": "infrared-therapy-flashlight", "name": "Infrared Therapy Flashlight", "tier": "secondary"},
    {"slug": "wearable-ionizer", "name": "Ionizer - Wearable", "tier": "secondary"},
    {"slug": "mithreal-silver-ivy-hat", "name": "Ivy Hat with Mithreal Silver", "tier": "secondary"},
    {"slug": "microgen-microcurrent-generator", "name": "Microgen Wearable Microcurrent Generator", "tier": "secondary"},
    {"slug": "mind-tuning-fork-5000hz", "name": "Mind Tuning Fork 5000 Hz", "tier": "secondary"},
    {"slug": "miracule-water-system", "name": "Miracule Water System with Molecular Hydrogen", "tier": "secondary"},
    {"slug": "molecular-hydrogen-bottle", "name": "Molecular Hydrogen (portable bottle)", "tier": "secondary"},
    {"slug": "nes-mihealth", "name": "NES miHealth", "tier": "secondary"},
    {"slug": "nes-scanner", "name": "NES Scanner", "tier": "secondary"},
    {"slug": "nir-brain-frequency-helmet", "name": "NIR Brain Frequency Helmet", "tier": "secondary"},
    {"slug": "nir-nasal-clip", "name": "NIR Near-Infrared Nasal Clip", "tier": "secondary"},
    {"slug": "red-mitochondrial-therapy-630nm", "name": "Red Mitochondrial Therapy 630 nm", "tier": "secondary"},
    {"slug": "shower-filter", "name": "Shower Filter", "tier": "secondary"},
    {"slug": "shungite-stick-plate", "name": "Shungite Stick Plate", "tier": "secondary"},
    {"slug": "smokey-quartz-healing-tool", "name": "Smokey Quartz Healing Tool", "tier": "secondary"},
    {"slug": "spirit-tuning-fork-172hz", "name": "Spirit Tuning Fork 172 Hz", "tier": "secondary"},
    {"slug": "neutralizer-3-pack", "name": "The Neutralizer 3 Pack", "tier": "secondary"},
    {"slug": "therapeutic-nightlight", "name": "Therapeutic Nightlight", "tier": "secondary"},
    {"slug": "tibetan-singing-bowl-172hz", "name": "Tibetan Glass Singing Bowl 172 Hz", "tier": "secondary"},
    {"slug": "vagus-nerve-stimulation-kit", "name": "Vagus Nerve Stimulation Kit for miHealth", "tier": "secondary"},
    {"slug": "whole-house-neutralizer", "name": "Whole House Neutralizer", "tier": "secondary"},
    {"slug": "wicking-toothbrush", "name": "Wicking Toothbrush", "tier": "secondary"},
    {"slug": "hand-cradle", "name": "ZYTO Hand Cradle", "tier": "secondary"},
]


def build_roadmap(owned_slugs, terrain_phase=None):
    """Personal recommended-additions roadmap: hero tools (fixed order, not
    owned), THEN secondary tools (list order, not owned). Deduped by slug
    across tiers -- a slug already added by an earlier tier is not re-added
    later. Every item keeps {slug, name, why, tier} (secondary items carry
    why="", since SECONDARY_TOOLS is an unprioritized list with no per-item
    rationale).

    `terrain_phase` is accepted for backward-compatible call signatures
    (callers/tests/plumbing) but is now IGNORED: Glen's direction is that
    healing tools work in all phases, so there is no per-phase tool
    selection any more -- the result is identical for every phase or None.
    Pure, no DB."""
    owned_slugs = owned_slugs or set()
    seen = set()
    roadmap = []

    for tool in HERO_TOOLS:
        slug = tool["slug"]
        if slug in owned_slugs or slug in seen:
            continue
        seen.add(slug)
        item = dict(tool)
        item.setdefault("why", "")
        roadmap.append(item)

    for tool in SECONDARY_TOOLS:
        slug = tool["slug"]
        if slug in owned_slugs or slug in seen:
            continue
        seen.add(slug)
        item = dict(tool)
        item.setdefault("why", "")
        roadmap.append(item)

    return roadmap
