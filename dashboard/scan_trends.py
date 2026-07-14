"""Longitudinal trend analysis across a client's E4L scan history. Pure reader
over e4l.db; for each stress item, how it behaves across the client's scans over
time — persistent, emerging, resolving, or intermittent — using appearance and
priority-rank. Severity is derived from that rank: the report's own legend states
the dot colour ('red/purple being the highest priority') encodes relative
priority, which is exactly the recommendation order we store as priority_rank.
The literal dot colour is not in the parsed text (pdftotext discards it), so we
reconstruct severity by normalizing each item's rank within its own scan — rank 1
= 1.0 ('purple'/top), the last ranked item = 0.0 — and track whether that
normalized severity is climbing (worsening) or falling (easing) over the window.
Never raises into callers."""
import math

# A per-half difference in normalized severity below this is treated as no real
# movement (noise), so an item only reads worsening/easing on a clear shift.
SEVERITY_TREND_MIN_DELTA = 0.15


def _severity(rank, n_ranked):
    """Normalize a within-scan priority rank to 0..1, where 1.0 = rank 1 = the
    top ('purple') priority and 0.0 = the last ranked item. Normalizing per scan
    makes ranks comparable across scans of different lengths. None if unranked."""
    if rank is None:
        return None
    if n_ranked is None or n_ranked <= 1:
        return 1.0
    v = (n_ranked - rank) / float(n_ranked - 1)
    return max(0.0, min(1.0, v))


def _severity_band(sev):
    """Coarse label for a normalized severity: top / high / moderate / low.
    'top' is the red/purple end of the report's own priority colouring."""
    if sev is None:
        return None
    if sev >= 0.75:
        return "top"
    if sev >= 0.5:
        return "high"
    if sev >= 0.25:
        return "moderate"
    return "low"


def _severity_trend(recent_sev, older_sev):
    """Compare mean severity in the newer vs older half of the appearances.
    worsening = severity climbing toward the top; easing = falling away; steady =
    little change. 'na' when either half has no ranked appearance to compare."""
    if recent_sev is None or older_sev is None:
        return "na"
    delta = recent_sev - older_sev
    if delta >= SEVERITY_TREND_MIN_DELTA:
        return "worsening"
    if delta <= -SEVERITY_TREND_MIN_DELTA:
        return "easing"
    return "steady"


def client_scans(cx, client_id, last_n=None, with_findings_only=True):
    """Newest-first [(scan_id, scan_date)] for a client, optionally the last_n.
    By default only scans that carry ranked findings are counted — many stored
    scans have no priority-ranked results and would otherwise skew the trend
    window (an empty scan can't hold a pattern)."""
    try:
        if with_findings_only:
            rows = cx.execute(
                "SELECT s.scan_id, s.scan_date FROM e4l_scans s "
                "WHERE s.client_id=? AND EXISTS (SELECT 1 FROM e4l_scan_results r "
                "  WHERE r.scan_id=s.scan_id AND r.priority_rank IS NOT NULL) "
                "ORDER BY s.scan_date DESC, s.scan_id DESC", (client_id,)).fetchall()
        else:
            rows = cx.execute(
                "SELECT scan_id, scan_date FROM e4l_scans WHERE client_id=? "
                "ORDER BY scan_date DESC, scan_id DESC", (client_id,)).fetchall()
    except Exception:
        return []
    out = [(r["scan_id"], r["scan_date"]) for r in rows]
    return out[:last_n] if last_n else out


def _classify(n_appear, n_scans, recent_count, older_count):
    """recent_count/older_count = appearances in the newer vs older half of the
    client's scan window. Persistent = chronic (≥50% of scans); emerging = only in
    the recent half; resolving = only in the older half; else intermittent."""
    if n_scans >= 2 and n_appear >= math.ceil(0.5 * n_scans):
        return "persistent"
    if recent_count and not older_count:
        return "emerging"
    if older_count and not recent_count:
        return "resolving"
    return "intermittent"


def client_trends(cx, client_id, last_n=None):
    """{n_scans, first_date, last_date, items:[{code, name, category, appearances,
    frequency_pct, best_rank, latest_rank, first_date, last_date, trend, severity,
    severity_band, severity_trend, severity_delta}]}, sorted persistent → emerging
    → resolving → intermittent, then by frequency/best rank. severity is the
    normalized rank (0..1) at the item's most recent appearance; severity_trend is
    worsening/easing/steady/na across the window (see module docstring)."""
    scans = client_scans(cx, client_id, last_n)
    n = len(scans)
    if n == 0:
        return {"n_scans": 0, "first_date": None, "last_date": None, "items": []}
    scan_ids = [s[0] for s in scans]
    latest_id, earliest_id = scan_ids[0], scan_ids[-1]
    # ordinal position of each scan, newest=0 .. oldest=n-1
    pos = {sid: i for i, (sid, _) in enumerate(scans)}
    date_of = {sid: d for sid, d in scans}
    qmarks = ",".join("?" * len(scan_ids))
    try:
        rows = cx.execute(
            "SELECT r.scan_id AS sid, r.item_code AS code, r.priority_rank AS rank, "
            "  i.name AS name, i.full_name AS full_name, i.category AS category "
            "FROM e4l_scan_results r LEFT JOIN e4l_items i ON i.code = r.item_code "
            "WHERE r.scan_id IN (%s)" % qmarks, scan_ids).fetchall()
    except Exception:
        return {"n_scans": n, "first_date": scans[-1][1], "last_date": scans[0][1], "items": []}
    # ranked-item count per scan = the denominator for normalizing rank -> severity
    scan_ranked = {}
    for r in rows:
        if r["rank"] is not None:
            scan_ranked[r["sid"]] = scan_ranked.get(r["sid"], 0) + 1
    agg = {}
    for r in rows:
        code = (r["code"] or "").strip()
        if not code:
            continue
        a = agg.setdefault(code, {"code": code,
                                  "name": (r["full_name"] or r["name"] or code).strip(),
                                  "category": (r["category"] or "").strip(),
                                  "appts": []})  # (pos, sid, rank)
        a["appts"].append((pos.get(r["sid"], n), r["sid"], r["rank"]))
    items = []
    order = {"persistent": 0, "emerging": 1, "resolving": 2, "intermittent": 3}
    for code, a in agg.items():
        appts = a["appts"]
        n_appear = len(appts)
        ranks = [x[2] for x in appts if x[2] is not None]
        best_rank = min(ranks) if ranks else None
        latest_rank = next((rk for p, sid, rk in appts if sid == latest_id), None)
        # newer half = position < n/2 (newest scans); older half = the rest
        recent_count = sum(1 for p, _, _ in appts if p < n / 2.0)
        older_count = n_appear - recent_count
        appt_dates = sorted(date_of[sid] for _, sid, _ in appts)
        trend = _classify(n_appear, n, recent_count, older_count)
        # severity from normalized rank: current level = the most recent (lowest
        # pos) appearance; trend = mean severity of the newer vs older half.
        sev_by_pos = [(p, _severity(rk, scan_ranked.get(sid)))
                      for p, sid, rk in appts]
        latest_sev = min(sev_by_pos, key=lambda t: t[0])[1] if sev_by_pos else None
        recent_sevs = [s for p, s in sev_by_pos if s is not None and p < n / 2.0]
        older_sevs = [s for p, s in sev_by_pos if s is not None and p >= n / 2.0]
        recent_mean = sum(recent_sevs) / len(recent_sevs) if recent_sevs else None
        older_mean = sum(older_sevs) / len(older_sevs) if older_sevs else None
        sev_trend = _severity_trend(recent_mean, older_mean)
        sev_delta = (round(recent_mean - older_mean, 2)
                     if recent_mean is not None and older_mean is not None else None)
        items.append({
            "code": code, "name": a["name"], "category": a["category"],
            "appearances": n_appear, "frequency_pct": round(100.0 * n_appear / n),
            "best_rank": best_rank, "latest_rank": latest_rank,
            "first_date": appt_dates[0], "last_date": appt_dates[-1], "trend": trend,
            "severity": round(latest_sev, 2) if latest_sev is not None else None,
            "severity_band": _severity_band(latest_sev),
            "severity_trend": sev_trend, "severity_delta": sev_delta,
        })
    items.sort(key=lambda it: (order.get(it["trend"], 9), -it["frequency_pct"],
                               it["best_rank"] if it["best_rank"] is not None else 9999))
    return {"n_scans": n, "first_date": scans[-1][1], "last_date": scans[0][1], "items": items}
