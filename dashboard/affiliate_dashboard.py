"""Affiliate/ambassador dashboard data — shared by the standalone /affiliate/portal-data
route and the personal portal's Ambassador section. Pure, LOG_DB-based, none-raising."""


def _mask_lead_name(first, last):
    fn = (first or "").strip()
    ln = (last or "").strip()
    if ln:
        return f"{fn} {ln[0]}.".strip()
    return fn


def build_dashboard(cx, slug, *, quiz_url, public_base_url):
    """Full affiliate dashboard dict for a slug. {} if the slug isn't an enrolled
    affiliate. Mirrors the legacy /affiliate/portal-data payload exactly."""
    row = cx.execute(
        "SELECT name, organization, short_url, created_at FROM affiliate_signups WHERE slug=?",
        (slug,)).fetchone()
    if not row:
        return {}
    name, org, short_url, created_at = row[0], row[1] or "", row[2] or "", row[3] or ""
    base = (public_base_url or "").rstrip("/")
    long_url = f"{quiz_url}?utm_source={slug}&utm_medium=affiliate&utm_campaign=scoreapp-quiz"
    tracking_url = short_url if short_url else long_url
    recruit_url = f"{base}/affiliate?ref={slug}"
    try:
        stats = cx.execute(
            "SELECT COUNT(*), MAX(received_at) FROM referral_events WHERE utm_source=?",
            (slug,)).fetchone()
        recent = cx.execute(
            "SELECT received_at, first_name, last_name, quiz_score FROM referral_events "
            "WHERE utm_source=? ORDER BY received_at DESC LIMIT 10", (slug,)).fetchall()
        recruited_count = cx.execute(
            "SELECT COUNT(*) FROM affiliate_signups WHERE referred_by=? AND status='approved'",
            (slug,)).fetchone()[0]
        conversions_count = cx.execute(
            "SELECT COUNT(*) FROM affiliate_conversions WHERE affiliate_slug=?",
            (slug,)).fetchone()[0]
        offers = cx.execute(
            "SELECT name, description, url_template, COALESCE(instructions,'') "
            "FROM affiliate_offers WHERE active=1 ORDER BY sort_order ASC").fetchall()
        social = cx.execute(
            "SELECT url, points, views, likes, shares, ts FROM affiliate_social_links "
            "WHERE slug=? ORDER BY id DESC", (slug,)).fetchall()
    except Exception:
        stats, recent, recruited_count, conversions_count, offers, social = None, [], 0, 0, [], []
    return {
        "name": name, "organization": org, "slug": slug,
        "tracking_url": tracking_url, "recruit_url": recruit_url,
        "total_leads": stats[0] if stats else 0,
        "last_lead": stats[1] if stats else None,
        "recruited_count": recruited_count,
        "conversions_count": conversions_count,
        "recent": [{"received_at": r[0], "name": _mask_lead_name(r[1], r[2]), "score": r[3]}
                   for r in recent],
        "offers": [{"name": o[0], "description": o[1],
                    "url": o[2].replace("{slug}", slug), "instructions": o[3]} for o in offers],
        "social_links": [{"url": s[0], "points": s[1], "views": s[2], "likes": s[3],
                          "shares": s[4], "ts": s[5]} for s in social],
        "member_since": created_at,
    }
