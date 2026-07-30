"""Read-only 3-phase onboarding status for the portal tile. Pure computation
from existing per-email stores; NO writes, NO new tables. Token-agnostic hrefs
(anchors) are prefixed with /portal/<token> by the route (Task 2).

NOTE on recommendation_events.product_sources: the brief's sketch assumed a
{product_key: {source_key: {...}}} shape. The real implementation returns a
LIST of per-product dicts: [{"product_key":..., "hidden":..., "sources":
[{"source": sk, "count":..., "first_touch":..., "last_touch":...}, ...]}, ...].
_has_source below is written against the real (list) shape.
"""
from dashboard import (client_scans, intake, client_photos,
                        portal_biofield_reports, recommendation_events,
                        membership_products, portal_health_history,
                        portal_extended_history, condition_triage)


def _has_scan(cx, email):
    try:
        return bool(client_scans.scans_for(cx, email))
    except Exception:
        return False


def _has_e4l_account(cx, email):
    """A synced scan proves this portal user already has an E4L account."""
    if _has_scan(cx, email):
        return True
    try:
        return cx.execute(
            "SELECT 1 FROM scan_freshness WHERE lower(email)=lower(?) LIMIT 1",
            (email,),
        ).fetchone() is not None
    except Exception:
        return False


def _has_source(cx, email, source_key):
    # recommendation_events.product_sources(cx, email) -> list of
    # {"product_key":..., "sources": [{"source": sk, ...}, ...]}
    try:
        products = recommendation_events.product_sources(cx, email) or []
        return any(
            any(s.get("source") == source_key for s in (p.get("sources") or []))
            for p in products
        )
    except Exception:
        return False


def _has_condition_history(cx, email):
    """Any submitted checklist response, including free-text Other."""
    try:
        return bool(cx.execute(
            "SELECT 1 FROM condition_triage WHERE lower(email)=lower(?) LIMIT 1",
            (email,),
        ).fetchone())
    except Exception:
        return False


def _safe(fn, cx, email):
    try:
        return bool(fn(cx, email))
    except Exception:
        return False


def build_status(cx, email):
    email = (email or "").strip().lower()
    conditions_done = (
        _has_source(cx, email, "condition") or _has_condition_history(cx, email)
    )
    products_done = _safe(portal_health_history.has, cx, email)
    extended_done = _safe(portal_extended_history.has, cx, email)
    condition_triage.init_table(cx)
    condition_rows = cx.execute(
        "SELECT condition FROM condition_triage WHERE lower(email)=lower(?) "
        "ORDER BY updated_at", (email,)
    ).fetchall()
    conditions = []
    for row in condition_rows:
        condition = row[0]
        answers = condition_triage.get_triage(cx, email, condition) or {}
        answers.pop("resolved_programs", None)
        conditions.append({"condition": condition, "answers": answers})
    product_history = portal_health_history.get(cx, email) or {}
    extended_record = portal_extended_history.get(cx, email) or {}

    def step(k, label, done, href, **extra):
        d = {"key": k, "label": label, "done": done, "href": href}
        d.update(extra)
        return d

    has_scan = _has_scan(cx, email)
    voice_href = (
        "https://portal.e4l.com"
        if _has_e4l_account(cx, email)
        else "https://truly.vip/E4L"
    )
    be_read = [
        step("voice", "Voice analysis", has_scan, voice_href),
        step("intake", "Intake", _safe(intake.is_submitted, cx, email), "#intake"),
        step("photo", "Photo", _safe(client_photos.has, cx, email), "#photo"),
        step("biofield", "Biofield Analysis",
             _safe(lambda c, e: portal_biofield_reports.latest_report(c, e) is not None, cx, email),
             "#biofield"),
    ]
    match = [
        step("history", "Starter remedies from your history",
             conditions_done and products_done and extended_done,
             "#recs"),
        step("scan_match", "Personalized match from your scan",
             _has_source(cx, email, "scan") or _has_source(cx, email, "biofield"), "#recs"),
    ]
    heal = [
        step("light", "Light", None, "https://clinicalpraxis.com"),
        step("pemf", "PEMF", None, "", soon=True),
        step("h2water", "Molecular hydrogen microwater", None, "", soon=True),
    ]
    return {
        "phases": [
            {
                "key": "be_read",
                "title": "Discover What Your Body Is Saying",
                "steps": be_read,
            },
            {"key": "match", "title": "Match remedies", "steps": match},
            {"key": "heal", "title": "Accelerate healing", "steps": heal},
        ],
        "member": _safe(membership_products.owns_group, cx, email),
        "history_conditions_done": conditions_done,
        "history_products_done": products_done,
        "history_extended_done": extended_done,
        "history_prefill": {
            "conditions": conditions,
            "products": product_history,
            "extended": extended_record.get("answers") or {},
        },
    }
