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
                        membership_products)


def _has_scan(cx, email):
    try:
        return bool(client_scans.scans_for(cx, email))
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


def _safe(fn, cx, email):
    try:
        return bool(fn(cx, email))
    except Exception:
        return False


def build_status(cx, email):
    email = (email or "").strip().lower()

    def step(k, label, done, href, **extra):
        d = {"key": k, "label": label, "done": done, "href": href}
        d.update(extra)
        return d

    be_read = [
        step("voice", "Voice analysis", _has_scan(cx, email), "https://truly.vip/E4L"),
        step("intake", "Intake", _safe(intake.is_submitted, cx, email), "https://truly.vip/Join"),
        step("photo", "Photo", _safe(client_photos.has, cx, email), "#photo"),
        step("biofield", "Biofield Analysis",
             _safe(lambda c, e: portal_biofield_reports.latest_report(c, e) is not None, cx, email),
             "#biofield"),
    ]
    match = [
        step("history", "Starter remedies from your history",
             _has_source(cx, email, "condition"), "#recs"),
        step("scan_match", "Personalized match from your scan",
             _has_source(cx, email, "biofield"), "#recs"),
    ]
    heal = [
        step("light", "Light", None, "https://clinicalpraxis.com"),
        step("pemf", "PEMF", None, "", soon=True),
        step("h2water", "Molecular hydrogen microwater", None, "", soon=True),
    ]
    return {
        "phases": [
            {"key": "be_read", "title": "Be read", "steps": be_read},
            {"key": "match", "title": "Match remedies", "steps": match},
            {"key": "heal", "title": "Accelerate healing", "steps": heal},
        ],
        "member": _safe(membership_products.owns_group, cx, email),
    }
