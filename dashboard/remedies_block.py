"""The "My Remedies" client-portal tile: a ranked list of products the client is
already engaging with (biofield/intake/scan/purchased/... — same data the
`/api/portal/<token>/recommendations` endpoint serves) plus their externally-
maintained supplement stack (dashboard/supplement_reviews.py), each row
optionally pointed at our clinically-equivalent formulation
(dashboard/remedy_upgrades.py).

`build_block(cx, email, enabled)` is the single entrypoint, threaded through
`portal_view.get_portal_view` exactly like `_supplement_reviews_block`.
Dark by default (PORTAL_REMEDIES_ENABLED); returns {"enabled": False} when off.
Both sub-builds are failure-isolated — a broken recommendations read or a
broken external-stack read degrades to an empty list rather than breaking the
rest of the portal payload."""
import re


def _product_key(name, brand):
    """Stable dedupe key for an external product: case- and whitespace-
    insensitive name|brand. Mirrors the normalization in
    dashboard.supplement_reviews._key (kept independent/local rather than
    importing a private helper, same convention as dashboard.remedy_upgrades)."""
    raw = "%s|%s" % ((name or "").strip().lower(), (brand or "").strip().lower())
    return re.sub(r"\s+", " ", raw)


def _build_ranked(cx, email, top_n=5):
    """Read-through of the SAME recommendations data the
    `/api/portal/<token>/recommendations` endpoint serves (product_sources ->
    build_sections), flattened across sections and deduped by product_key,
    truncated to the top_n. Never invents a different ranking."""
    from dashboard import recommendation_events as _re
    from dashboard import recommendation_prefs as _rp
    from dashboard import portal_recommendations as _pr
    from dashboard import products as _products

    _re.init_recommendation_events(cx)
    _rp.init_recommendation_prefs(cx)
    ps = _re.product_sources(cx, email)
    notes = _rp.get_notes(cx, email)
    state = _rp.get_section_state(cx, email)
    catalog = _products.load_products()

    def resolve(slug):
        p = catalog.get(slug) or {}
        return {"name": p.get("name"), "url": p.get("url")}

    sections = _pr.build_sections(ps, notes, state, resolve, top_n=top_n)
    seen = set()
    ranked = []
    for sec in sections:
        for prod in (sec.get("products") or []):
            pk = prod.get("product_key")
            if not pk or pk in seen:
                continue
            seen.add(pk)
            ranked.append({
                "product_key": pk,
                "name": prod.get("name") or pk,
                "url": prod.get("url") or "",
                "source": sec.get("source"),
                "reason": prod.get("operator_note") or prod.get("client_note") or "",
            })
            if len(ranked) >= top_n:
                return ranked
    return ranked


def _build_external(cx, email):
    """The client's externally-maintained stack (supplement_reviews rows),
    each enriched with an optional our-equivalent upgrade pointer. `review`
    text is included ONLY when status == 'confirmed' (mirror
    portal_view._supplement_reviews_block). Per-client access respected: a
    revoked client sees no external list."""
    from dashboard import supplement_reviews as _sr
    from dashboard import remedy_upgrades as _ru

    _sr.init_table(cx)
    if not _sr.access_enabled(cx, email):
        return []
    rows = _sr.list_for_email(cx, email)
    out = []
    for r in rows:
        item = {
            "product_key": _product_key(r.get("product_name"), r.get("product_brand")),
            "product_name": r.get("product_name"),
            "product_brand": r.get("product_brand"),
            "reason": r.get("reason"),
            "importance": r.get("importance"),
            "status": r.get("status"),
        }
        if r.get("status") == "confirmed":
            item["review"] = r.get("review_text") or ""
        try:
            item["upgrade"] = _ru.suggest_upgrade(r.get("product_name"), r.get("product_brand") or "")
        except Exception:
            item["upgrade"] = None
        out.append(item)
    return out


def build_block(cx, email, enabled):
    """Assemble the 'remedies' portal block. Dark by default: returns
    {"enabled": False} when `enabled` is False. Otherwise always returns
    {"enabled": True, "ranked": [...], "external": [...]} — `ranked` and
    `external` are built independently, each degrading to [] on any internal
    error so a failure in one never breaks the other or the rest of the
    portal payload."""
    if not enabled:
        return {"enabled": False}
    em = (email or "").strip().lower()
    try:
        ranked = _build_ranked(cx, em)
    except Exception:
        ranked = []
    try:
        external = _build_external(cx, em)
    except Exception:
        external = []
    return {"enabled": True, "ranked": ranked, "external": external}
