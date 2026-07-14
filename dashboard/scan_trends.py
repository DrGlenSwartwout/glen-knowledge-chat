"""Longitudinal trend analysis across a client's E4L scan history. Pure reader
over e4l.db; for each stress item, how it behaves across the client's scans over
time — persistent, emerging, resolving, or intermittent — using appearance and
priority-rank (severity/'purple level' is not in the parsed data, so trends rest
on recommendation rank + presence). Never raises into callers."""
import math


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
    frequency_pct, best_rank, latest_rank, first_date, last_date, trend}]}, sorted
    persistent → emerging → resolving → intermittent, then by frequency/best rank."""
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
        items.append({
            "code": code, "name": a["name"], "category": a["category"],
            "appearances": n_appear, "frequency_pct": round(100.0 * n_appear / n),
            "best_rank": best_rank, "latest_rank": latest_rank,
            "first_date": appt_dates[0], "last_date": appt_dates[-1], "trend": trend,
        })
    items.sort(key=lambda it: (order.get(it["trend"], 9), -it["frequency_pct"],
                               it["best_rank"] if it["best_rank"] is not None else 9999))
    return {"n_scans": n, "first_date": scans[-1][1], "last_date": scans[0][1], "items": items}
