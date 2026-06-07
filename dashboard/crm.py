"""Business-OS Sales & CRM. Lights up the CRM Home cell from the local work
queue: pending household-dedup candidates, queued merges, and unreplied new
leads. All counts are fast local SQLite reads. The household/merge ACTIONS are
already on the registry (Phase 1c); GHL-write actions are deferred (the WAF
blocks GHL writes from the server)."""
from dashboard.signals import signal as _signal, request_cached, RED, AMBER, GREEN, GRAY


def crm_summary(cx):
    """Single source of the CRM work-queue counts: pending household-dedup
    candidates, queued merges, and unreplied new leads. Defined once here so the
    signal (and any future consumer) share one definition. Request-cached during
    a home-signals aggregation."""
    def _read():
        cand = cx.execute(
            "SELECT COUNT(*) FROM household_candidates WHERE status='pending'").fetchone()[0]
        merges = cx.execute(
            "SELECT COUNT(*) FROM pending_merges WHERE status='pending'").fetchone()[0]
        leads = cx.execute(
            "SELECT COUNT(*) FROM inbound_leads "
            "WHERE (status IS NULL OR status='pending') "
            "  AND (last_outbound_at IS NULL OR last_outbound_at='') "
            "  AND email IS NOT NULL AND email!=''").fetchone()[0]
        return {"candidates": cand, "merges": merges, "leads": leads,
                "total": cand + merges + leads}
    return request_cached("crm:summary", _read)


def crm_signal(cx, actor=None):
    try:
        s = crm_summary(cx)
        cand, merges, leads = s["candidates"], s["merges"], s["leads"]
    except Exception:
        return {"level": GRAY, "summary": "Not yet wired", "top_actions": [], "count": 0}

    total = cand + merges + leads
    if total == 0:
        return {"level": GREEN, "summary": "CRM clear", "top_actions": [], "count": 0}

    bits = []
    if leads:
        bits.append(f"{leads} new lead{'s' if leads != 1 else ''}")
    if cand:
        bits.append(f"{cand} household candidate{'s' if cand != 1 else ''}")
    if merges:
        bits.append(f"{merges} merge{'s' if merges != 1 else ''} to apply")
    # Unreplied leads and queued merges are time-sensitive -> red; dedup-only -> amber.
    level = RED if (leads or merges) else AMBER
    return {"level": level, "summary": ", ".join(bits),
            "top_actions": [{"label": "Open people", "href": "/console"}],
            "count": total}


# Register the signal on import.
crm_signal = _signal("crm")(crm_signal)
