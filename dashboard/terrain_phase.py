"""Consumer-facing names for the 5 terrain phases (the BSI 'phase P' reading Glen
speaks into a biofield transcript). Phase number (1-5) -> the Terrain R-name Glen
reports on the client report + invoice. See the five-phases-of-terrain skill: the
number ordering matches the depleted->excess terrain spectrum
(Energize / Rejuvenate / Regenerate / Cleanse / Balance)."""

PHASE_NAMES = {
    1: "Terrain Revive",
    2: "Terrain Repair",
    3: "Terrain Renew",
    4: "Terrain Refresh",
    5: "Terrain Relief",
}

PHASE_CLINICAL_NAMES = {
    1: "Energize",
    2: "Rejuvenate",
    3: "Regenerate",
    4: "Cleanse",
    5: "Balance",
}

PHASE_DESCRIPTIONS = {
    1: "This phase focuses on restoring foundational cellular energy and the body's capacity to begin healing.",
    2: "This phase focuses on rebuilding compromised systems, including enzyme and microbial function.",
    3: "This phase focuses on tissue restoration and restoring microbial coherence.",
    4: "This phase focuses on clearing accumulated burdens after the body has built the capacity to move them.",
    5: "This phase focuses on restoring endocrine, life-stress, and regulatory balance.",
}


def phase_num(v):
    """Coerce a spoken/stored phase to an int in 1..5, else None. Never raises.
    Only a bare integer counts -- 'phase 3' returns None (the caller passes P)."""
    try:
        n = int(str(v).strip())
    except (TypeError, ValueError):
        return None
    return n if n in PHASE_NAMES else None


def phase_name(v):
    """'Terrain Refresh' for phase 4, '' when the phase is missing/out of range."""
    return PHASE_NAMES.get(phase_num(v), "")


def phase_display(v):
    """'Terrain Refresh (Phase 4)', or '' when there is no valid phase."""
    n = phase_num(v)
    return f"{PHASE_NAMES[n]} (Phase {n})" if n else ""


def phase_narrative_description(v, location=""):
    """Plain-language terrain context for the generated patient narrative."""
    n = phase_num(v)
    if not n:
        return ""
    loc = str(location or "").strip()
    where = f" The reading located this phase at {loc}." if loc else ""
    return f"Phase {n}, {PHASE_CLINICAL_NAMES[n]}: {PHASE_DESCRIPTIONS[n]}{where}"
