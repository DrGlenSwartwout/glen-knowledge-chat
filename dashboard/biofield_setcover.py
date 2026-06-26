"""Greedy set-cover for the Biofield Intake balancing loop (B4): the fewest
remedies that cover the most active stress codes. Pure; deterministic."""


def minimal_remedies(active_codes, coverage):
    """active_codes: iterable of codes to cover. coverage: {remedy: set(codes)}.
    Greedy: repeatedly pick the remedy covering the most still-uncovered codes,
    tie broken by remedy name ascending. Returns picks + the uncovered remainder."""
    remaining = set(active_codes or [])
    if not remaining:
        return {"picks": [], "uncovered": []}
    # Restrict each remedy to the active codes; drop remedies that cover nothing.
    cov = {}
    for remedy, codes in (coverage or {}).items():
        c = set(codes) & remaining
        if c:
            cov[remedy] = c
    picks = []
    while remaining:
        best, best_n = None, 0
        for remedy in sorted(cov):            # alphabetical -> deterministic tie-break
            n = len(cov[remedy] & remaining)
            if n > best_n:
                best, best_n = remedy, n
        if not best:                          # nothing left covers a remaining code
            break
        covered = sorted(cov[best] & remaining)
        picks.append({"remedy": best, "covers": covered})
        remaining -= set(covered)
    return {"picks": picks, "uncovered": sorted(remaining)}
