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
