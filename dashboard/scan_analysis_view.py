"""Tier-gating for the member-facing longitudinal scan-analysis page (SP2).

Pure module — no Flask/DB. Decides how much of a member's stored longitudinal
analysis artifact to expose, given their tier and the SCAN_ANALYSIS_FREE flag.

Gating posture (Glen, 2026-06-22): gate NOW so the system isn't overwhelmed
while it's being trained; flip SCAN_ANALYSIS_FREE on later to open the full
analysis to everyone. Until then only paid/active members see the full
over-time analysis; free (ToS-agreed) members see a teaser + upsell; everyone
else sees a locked upsell.
"""

# tiers, richest first
PAID, FREE, NONE = "paid", "free", "none"


def gated_payload(analysis: dict, *, tier: str, free_enabled: bool) -> dict:
    """Return the display payload for the given tier.

    analysis: the stored artifact (build_artifact output) or {} / None.
    tier: 'paid' | 'free' | 'none'.
    free_enabled: SCAN_ANALYSIS_FREE — when True the full analysis is public to
      any ToS member (gate then lives only on the chat). When False (default),
      only 'paid' sees the full analysis.

    access marker: 'full' | 'teaser' | 'locked'. Withheld sections are emptied
    server-side (never serialized) so the client cannot reveal them.
    """
    a = analysis or {}
    base = {
        "scan_count": a.get("scan_count", 0),
        "date_range": a.get("date_range", [None, None]),
        "generated_at": a.get("generated_at"),
    }
    full = (tier == PAID) or (free_enabled and tier in (PAID, FREE))

    if full:
        return {**base, "access": "full", "upsell": False,
                "narrative": a.get("narrative", ""),
                "top_patterns": a.get("top_patterns", []),
                "clusters": a.get("clusters", []),
                "functional_relation": a.get("functional_relation", [])}

    if tier == FREE:
        # a taste: the top few recurring patterns, but no clusters / functional
        # view / narrative — those are the paid depth.
        return {**base, "access": "teaser", "upsell": True,
                "narrative": "",
                "top_patterns": (a.get("top_patterns") or [])[:3],
                "clusters": [], "functional_relation": []}

    # NONE — no membership, no ToS: nothing but the invitation to unlock.
    return {**base, "access": "locked", "upsell": True,
            "narrative": "", "top_patterns": [],
            "clusters": [], "functional_relation": []}


def resolve_tier(*, is_paid: bool, has_tos: bool) -> str:
    """Map membership/ToS facts to a tier string."""
    if is_paid:
        return PAID
    if has_tos:
        return FREE
    return NONE


# ── Tiered chat grounding (SP2 slice 2) ──────────────────────────────────────

def format_facts(analysis: dict) -> str:
    """Compact text block of the member's analysis for grounding the chat LLM.
    Only the computed facts — no invented numbers downstream."""
    a = analysis or {}
    lines = []
    sc = a.get("scan_count")
    dr = a.get("date_range") or [None, None]
    if sc:
        span = f" ({dr[0]} to {dr[1]})" if dr and dr[0] else ""
        lines.append(f"Scans analyzed: {sc}{span}.")
    tp = a.get("top_patterns") or []
    if tp:
        lines.append("Most consistent patterns (code, name, % of scans):")
        for p in tp[:12]:
            pct = p.get("pct")
            pcts = f"{round(pct * 100)}%" if pct is not None else ""
            lines.append(f"  - {p.get('code', '')} {p.get('name', '')} {pcts}".rstrip())
    cl = a.get("clusters") or []
    if cl:
        lines.append("Clinical clusters (structure: the member's codes touching it):")
        for c in cl[:10]:
            lines.append(f"  - {c.get('structure', '')}: {', '.join(c.get('codes') or [])}")
    fr = [g for g in (a.get("functional_relation") or []) if g.get("is_functional")]
    if fr:
        lines.append("Functional stress patterns (most-loaded first): "
                     + ", ".join(g.get("structure", "") for g in fr[:10]))
    if a.get("narrative"):
        lines.append("Prior summary: " + a["narrative"])
    return "\n".join(lines)


def chat_context(analysis: dict, *, access: str) -> dict:
    """What the chat may use, by page access level.
    full -> grounded in the member's own analysis facts.
    teaser/locked -> general education only + upsell (no personal facts)."""
    if access == "full":
        return {"grounded": True, "facts": format_facts(analysis), "upsell": False}
    return {"grounded": False, "facts": "", "upsell": True}
