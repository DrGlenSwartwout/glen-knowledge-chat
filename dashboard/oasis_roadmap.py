"""My Healing Oasis — recommended-additions roadmap.

Analysis-ordered, hero-tool-led recommendations for a client's Healing Oasis:
  1. HERO_TOOLS  — the near-universal tools Glen recommends to almost everyone
     (fixed lead order: harmony -> water-ionizer -> kloud). These sort ABOVE
     terrain-specific and general items because they are foundational, not
     situational.
  2. TERRAIN_TOOLS — gap tools keyed by the client's current 5 R terrain phase
     (energize/rejuvenate/regenerate/cleanse/balance, matching
     dashboard/terrain_phase.py's Energize->Balance ordering).
  3. A general low-cost-high-leverage tail for clients with no terrain phase
     on file, or once hero + terrain recommendations are exhausted/owned.

Pure module, no DB — build_roadmap() takes owned_slugs (see
dashboard/owned_tools.owned_slugs()) and returns the client's personal gap
list. Glen is expected to extend HERO_TOOLS / TERRAIN_TOOLS / GENERAL_TOOLS
as his product line and recommendations grow.
"""

# The three near-universal tools, in FIXED lead order. Order is significant:
# it is the order clients see them in when none are owned yet.
HERO_TOOLS = [
    {
        "slug": "harmony",
        "name": "Harmony Laser",
        "why": "Low-level laser therapy Glen recommends to nearly every client for "
               "cellular energy and pain/inflammation support, regardless of terrain phase.",
        "tier": "hero",
    },
    {
        "slug": "water-ionizer",
        "name": "Water Ionizer",
        "why": "Alkaline, antioxidant-rich, micro-clustered water is a daily-use "
               "foundation Glen recommends across every terrain phase.",
        "tier": "hero",
    },
    {
        "slug": "kloud",
        "name": "Kloud PEMF Mat",
        "why": "Pulsed electromagnetic field therapy supports circulation, recovery, "
               "and cellular voltage for nearly every client's protocol.",
        "tier": "hero",
    },
]

# Gap tools per 5 R terrain phase. Ordered highest-leverage-first within each phase.
# Keys match dashboard/terrain_phase.py's Energize -> Balance spectrum.
# Glen: extend each phase's list as more terrain-specific tools are confirmed.
TERRAIN_TOOLS = {
    "energize": [
        {
            "slug": "red-light-panel",
            "name": "Red Light Therapy Panel",
            "why": "Mitochondrial photobiomodulation to rebuild energy reserves in a "
                   "depleted, catabolic terrain.",
            "tier": "terrain",
        },
        {
            "slug": "grounding-mat",
            "name": "Grounding Mat",
            "why": "Restores the body's electrical baseline, supporting recharge in "
                   "an energize-phase terrain.",
            "tier": "terrain",
        },
    ],
    "rejuvenate": [
        {
            "slug": "infrared-sauna",
            "name": "Infrared Sauna",
            "why": "Gentle deep-tissue heat supports circulation and tissue repair "
                   "as the terrain moves from depletion toward rejuvenation.",
            "tier": "terrain",
        },
        {
            "slug": "vibration-plate",
            "name": "Whole-Body Vibration Plate",
            "why": "Low-impact lymphatic and circulatory stimulation to support "
                   "rejuvenation without overtaxing a still-fragile terrain.",
            "tier": "terrain",
        },
    ],
    "regenerate": [
        {
            "slug": "denas-pcm-pro",
            "name": "DENAS PCM Pro",
            "why": "Neuro-adaptive electrostimulation to support tissue regeneration "
                   "and nervous-system recalibration.",
            "tier": "terrain",
        },
        {
            "slug": "mihealth",
            "name": "NES miHealth",
            "why": "Bioenergetic feedback device that supports the body's own "
                   "regenerative signaling.",
            "tier": "terrain",
        },
    ],
    "cleanse": [
        {
            "slug": "dry-brush",
            "name": "Dry Skin Brush",
            "why": "Simple daily lymphatic stimulation to support drainage during a "
                   "cleanse-phase terrain.",
            "tier": "terrain",
        },
        {
            "slug": "far-infrared-sauna",
            "name": "Far-Infrared Sauna",
            "why": "Deep sweating supports elimination of stored toxins in an "
                   "excess/cleanse terrain.",
            "tier": "terrain",
        },
    ],
    "balance": [
        {
            "slug": "hrv-biofeedback",
            "name": "HRV Biofeedback Device",
            "why": "Heart-rate-variability training helps a near-balanced terrain "
                   "hold its gains long term.",
            "tier": "terrain",
        },
        {
            "slug": "dowsing-rods",
            "name": "Dowsing Rods",
            "why": "A simple, low-cost self-check tool clients in a balanced terrain "
                   "can use to keep tracking their own field.",
            "tier": "terrain",
        },
    ],
}

# General low-cost-high-leverage tail: useful regardless of terrain phase or when
# no terrain phase is on file. Glen: extend as more broadly-useful tools are confirmed.
GENERAL_TOOLS = [
    {
        "slug": "own-box",
        "name": "OWN Box",
        "why": "A low-cost daily EMF-mitigation tool with broad applicability "
               "across every client's protocol.",
        "tier": "general",
    },
    {
        "slug": "nir-intranasal-clip",
        "name": "NIR Intranasal Light Clip",
        "why": "Inexpensive near-infrared light therapy with wide-ranging, "
               "low-effort daily benefit.",
        "tier": "general",
    },
]

_KNOWN_TERRAIN_PHASES = ("energize", "rejuvenate", "regenerate", "cleanse", "balance")


def build_roadmap(owned_slugs, terrain_phase=None):
    """Personal recommended-additions roadmap: hero tools (fixed order, not owned),
    then the terrain phase's gap tools (not owned), then a general tail (not owned).
    Deduped by slug across tiers -- a slug already added by an earlier tier is not
    re-added later. Unknown/None terrain_phase -> hero + general only. Pure, no DB."""
    owned_slugs = owned_slugs or set()
    seen = set()
    roadmap = []

    for tool in HERO_TOOLS:
        slug = tool["slug"]
        if slug in owned_slugs or slug in seen:
            continue
        seen.add(slug)
        roadmap.append(dict(tool))

    if terrain_phase in _KNOWN_TERRAIN_PHASES:
        for tool in TERRAIN_TOOLS.get(terrain_phase, []):
            slug = tool["slug"]
            if slug in owned_slugs or slug in seen:
                continue
            seen.add(slug)
            roadmap.append(dict(tool))

    for tool in GENERAL_TOOLS:
        slug = tool["slug"]
        if slug in owned_slugs or slug in seen:
            continue
        seen.add(slug)
        roadmap.append(dict(tool))

    return roadmap
