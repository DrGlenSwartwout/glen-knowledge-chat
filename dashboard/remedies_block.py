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
def _product_key(name, brand):
    """Stable dedupe key for an external product: case- and whitespace-
    insensitive name|brand. Delegates to dashboard.supplement_reviews.product_key,
    the single source of truth for this normalization (kept as its own function
    here so existing call sites in this module are unaffected)."""
    from dashboard import supplement_reviews as _sr
    return _sr.product_key(name, brand)


def _recommendation_sections(cx, email, top_n=5):
    """The single expensive read-through of the SAME recommendations data the
    `/api/portal/<token>/recommendations` endpoint serves (product_sources ->
    notes/section_state -> catalog -> build_sections). Both the ranked list
    and the from-history (condition) list are derived from this ONE result --
    see `_build_ranked_from_sections` / `_build_from_history_from_sections` --
    so `build_block` no longer does this read-through twice per page load."""
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

    return _pr.build_sections(ps, notes, state, resolve, top_n=top_n)


def _build_ranked_from_sections(sections, top_n=5):
    """Flatten `sections` (already built by `_recommendation_sections`) across
    non-condition sources, deduped by product_key, truncated to the top_n.
    Never invents a different ranking."""
    seen = set()
    ranked = []
    for sec in sections:
        if sec.get("source") == "condition":
            # Condition-seeded (triage) remedies get their own "Suggested remedies
            # from your history" section (see _build_from_history_from_sections)
            # -- keep them out of the generic ranked list so they aren't shown
            # twice.
            continue
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
                "reason": prod.get("client_note") or "",
            })
            if len(ranked) >= top_n:
                return ranked
    return ranked


def _build_from_history_from_sections(sections):
    """Triage-seeded (condition) remedies for their own 'Suggested remedies from
    your history' portal section -- distinct from the general 'Top recommended
    for you' list built by `_build_ranked_from_sections` (which skips the
    condition section). Operates on the SAME `sections` (already built by
    `_recommendation_sections`) as `_build_ranked_from_sections`; returns ONLY
    the products from the section whose source == "condition", deduped by
    product_key. [] if there is no such section (e.g. no triage has ever
    seeded this client)."""
    seen = set()
    out = []
    for sec in sections:
        if sec.get("source") != "condition":
            continue
        for prod in (sec.get("products") or []):
            pk = prod.get("product_key")
            if not pk or pk in seen:
                continue
            seen.add(pk)
            out.append({
                "product_key": pk,
                "name": prod.get("name") or pk,
                "url": prod.get("url") or "",
                "reason": prod.get("client_note") or "",
            })
    return out


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
    {"enabled": True, "ranked": [...], "external": [...], "from_history": [...]}
    — the shared recommendations read-through (`_recommendation_sections`) runs
    ONCE and feeds both `ranked` and `from_history`; `external` is independent.
    Each of the three is still isolated by its own try/except and degrades to
    [] on any internal error, so a failure building one (or the shared read
    itself) never breaks the others or the rest of the portal payload."""
    if not enabled:
        return {"enabled": False}
    em = (email or "").strip().lower()
    try:
        sections = _recommendation_sections(cx, em)
    except Exception:
        sections = []
    try:
        ranked = _build_ranked_from_sections(sections)
    except Exception:
        ranked = []
    try:
        external = _build_external(cx, em)
    except Exception:
        external = []
    try:
        from_history = _build_from_history_from_sections(sections)
    except Exception:
        from_history = []
    return {"enabled": True, "ranked": ranked, "external": external, "from_history": from_history}
